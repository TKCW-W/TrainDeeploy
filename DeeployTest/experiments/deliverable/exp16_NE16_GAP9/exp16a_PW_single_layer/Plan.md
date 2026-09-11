# exp16a — SpeechNet block-1 conv on NE16 via all-pointwise decomposition

**Branch:** `feat/GAP9_w_NE16` · **Container:** `deeploy_gap9` (device), `agitated_hugle` (export)
**Parent:** `exp16_NE16_GAP9` (STEP 1 + STEP 2a done — see `../Findings.md` and
`TrainDeeploy/WorkLog/GAP9_w_NE16_Worklog.md`)

---

## 1. Goal

Run **SpeechNet block 1's real convolution** — `1×16`, from the **training** graph — on NE16, and
prove it computes exactly what the cluster computes.

This is STEP 2b of the plan (`ETH/docs/TRAIN_GAP9_NE16/03-qzo-ne16-plan.md`). exp16a differs from
STEP 2a in the two ways that matter:

| | STEP 2a (done) | **exp16a** |
|---|---|---|
| source graph | inference fixture (`qinfer`) | **training fixture** (`speechnet_qzo12_train`) |
| weight | `gs.Constant` | **runtime tensor** (graph input) |
| kernel | reduced `1×16 → 1×1` | **the real `1×16`** |
| coverage of block 1 | 6.2 % of its MACs | **100 %** |

## 2. What block 1 actually is (verified, `speechnet_qzo12_train/network.onnx`)

```
blocks.1.conv.weight_int8  [16,8,1,16] int8   GRAPH INPUT      ← weights-as-inputs
        │
        ▼  RQSPerturbRademacher(pmul[16] CONST, div=32768, seed=42, idx=2)
blocks.1.conv.weight_int8_pert [16,8,1,16] int8
        │
Quant ──┴──► Conv  kernel_shape=[1,16] pads=[0,8,0,8] group=1
             in [1,8,14,87] int8  ->  out [1,16,14,88] int32
                                   │
                                   ▼  RequantShift  mul[16] CONST,
                                      add = blocks.1.conv.bias_rqsadd_pert (also perturbed),
                                      div=65536, signed=1  ->  int8
```

Note the Conv already emits **int32** and `RequantShift` is a separate node — the decomposition
does not have to fight a fused `RequantizedConv`.

## 3. The two blockers, and how exp16a addresses each

### Blocker 1 — the weight is not a `gs.Constant`

Two distinct sub-problems; exp16a solves the first and **explicitly defers** the second.

| # | sub-problem | exp16a |
|---|---|---|
| 1a | `NE16Engine.canExecute` requires `isinstance(node.inputs[1], gs.Constant)`, and `NE16AdjustWeightMemoryLayoutPass` rewrites `weightTensor.values` at compile time — neither works on a runtime tensor | **SOLVED**: accept a variable weight; the weight arrives as a **graph input already in NE16 bit-serial layout**, encoded by the host with the same `_weightEncode` used at compile time |
| 1b | in the full QZO loop the perturbed weight is produced **on device** by `ApplyPerturbQuantRademacher_CHW`, so nobody has encoded it | **DEFERRED to exp16b** — either the linearity decomposition (`conv(w+δz,x) = conv(w,x) + δ·conv(z,x)`) or a device-side encode kernel |

**The key enabler for 1a: `weight_offset = -128`, fixed.**
NE16 stores weights unsigned and compensates with `Wmin·Σx`. PR #183 computes
`weight_offset = values.min()` at compile time — impossible for a changing weight. But any int8
weight satisfies `w ∈ [-128,127] ⟹ w + 128 ∈ [0,255]`, so **-128 is valid for every possible
weight and never needs recomputation.** The host encodes `w + 128`; the node carries
`weight_offset = -128` as a static attribute.

Cost of the fixed offset: none arithmetically. `Wmin` is applied by one extra shift cycle
(`SHIFT_CYCLES = 2`) regardless of its value, so `-128` is exactly as cheap as `-123`.

### Blocker 2 — `1×16` is not an NE16 filter mode

Decompose into **16 pointwise convolutions accumulating in int32**:

```
conv_{1×16}(W, X)[co,h,w] = Σ_{j=0..15} conv_{1×1}(W[:,:,0,j], X)[co,h,w+j-8]
```

with the original `pads=[0,8,0,8]` becoming **per-tap input slice offsets** — never NE16 padding,
because padding is invalid in 1×1 mode **and its guard in `fsm.cpp:53` is commented out**, so it
would fail silently.

Why pointwise and not masked-3×3 (evaluated and rejected — see §7).

## 4. Design

### 4.1 Graph shape

Accumulation is expressed **in the ONNX graph** (explicit `Add` chain), not via NE16 `streamin`.

```
input int8 [1,8,14,87]
   │
   ├─ Pad  (W: 8 left, 8 right)  ->  [1,8,14,103]        (cluster, once)
   │
   ├─ Slice j=0  [.., 0:87]  ─► Conv1x1(Wenc_0) ─┐        (NE16)
   ├─ Slice j=1  [.., 1:88]  ─► Conv1x1(Wenc_1) ─┤ Add    (cluster, int32)
   │        ⋮                          ⋮         ⋮
   └─ Slice j=15 [..,15:102] ─► Conv1x1(Wenc_15)┘
                                                 │
                                    RequantShift (mul, add, div=65536) -> int8
```

**Why an `Add` chain rather than `streamin`:** it isolates *arithmetic correctness of the
decomposition* from *a new NE16 accumulation mode*. If the Add-chain version matches the
reference, the decomposition is right; only then is it worth replacing 15 `Add`s with
`streamin` (which additionally needs the tile-residency work). Bringing both up at once is the
failure mode PR #183's author documented — four consecutive wrong diagnoses.

`streamin` is therefore **exp16b**, as an optimisation of a known-correct baseline.

### 4.2 Three graphs, one dataset

| graph | purpose | engine |
|---|---|---|
| `reference/` | the original `1×16` Conv + RequantShift, untouched | cluster (`GAP9`) |
| `decomposed/` | the 16-tap pointwise version | NE16 (`GAP9_w_NE16`) |
| `decomposed/` again | same graph, `canExecute` forced `False` | cluster (`GAP9_w_NE16`, A/B control) |

Real data throughout: the int8 activation from evaluation window 0 of the exp12 fixture, the real
`blocks.1.conv.weight_int8`, the real perturbation `pmul`, the real `mul`/`add`/`div`.

The **perturbed** weight is computed on the host with a numpy port of
`ApplyPerturbQuantRademacher_CHW` (`TargetLibraries/PULPOpen/src/RandomNoiseQuant.c:15-84`),
validated against `onnx4deeploy`'s own `RQSPerturbRademacher` implementation, then encoded.
That is what makes the weight a *runtime* tensor with *real training values* while keeping the
device graph free of the perturbation op for now.

### 4.3 Deeploy changes (all Python; **no NE16 ISA change**)

Everything below alters only which nodes Deeploy offers to NE16 and what it writes into the
existing `ne16_task_t` fields. No new register, no undocumented bit, no GVSoC-only behaviour —
so the result must run on real GAP9 silicon exactly as on `gap9.evk`.

| # | file | change |
|---|---|---|
| C1 | `Deeploy/Targets/NE16/Engine.py` | `isPWConv`: accept a **non-constant** weight when the node carries `ne16_weight_preencoded=1`; keep requiring `gs.Constant` otherwise (so PR #183's own tests are unaffected) |
| C2 | `Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py` | `_ne16_adjust_weight_memory_layout_fun`: skip nodes already marked pre-encoded (do not touch `.values`, do not overwrite `weight_offset`) |
| C3 | `Deeploy/Targets/NE16/Parsers.py` | already accepts rank-3 weights and requires `weight_offset` — verify no change needed |

If C3 turns out to need a change, that is a finding to record, not a silent patch.

## 5. Acceptance criteria

1. **Host, before any device run:** decomposed graph output == reference graph output,
   **bit-exact**, through `onnx4deeploy`'s `run_onnx_graph`. Catches slicing/padding errors for
   the price of a numpy run.
2. **Device, NE16:** `grep -c "ne16_nnx_dispatch(" Network.c` == 16 (one per tap) × tiles, and
   `grep -c pulp_nn_conv Network.c` == 0.
3. **Device, bit-exact:** `Errors: 0 out of 19488` against the host reference.
4. **A/B control:** the same decomposed graph with `canExecute` forced `False` produces
   identical output — proving the NE16 path, not the graph rewrite, is what was tested.
5. **Cycles recorded** for: reference-on-cluster, decomposed-on-cluster, decomposed-on-NE16.
   The middle one is the honest denominator — it separates "NE16 is fast" from "the
   decomposition is expensive".

## 6. What exp16a deliberately does NOT do

* **No `streamin`** — 15 explicit `Add`s instead (exp16b).
* **No on-device perturbation** — the host supplies the already-perturbed, already-encoded weight
  (blocker 1b, exp16b).
* **No other SpeechNet layer** — blocks 0/2/3/4 are STEP 3. Block 0 additionally needs the
  signed-activation fix; blocks 1–4 do not (measured: activations ≥ 0).
* **No accuracy claim** — this is one layer, bit-exactness only. Accuracy is a full-round result.

## 7. Alternative evaluated and rejected: masked 3×3

Embedding each `1×3` slice in a zero-padded 3×3 kernel would need 6 dispatches instead of 16.
Rejected on the cycle model (`ne16_matrixvec.cpp:316,342`):

NE16's 9 row-slots per column hold **either** the 8 bitplanes (1×1 mode, `mv_qw_lim = 1`,
`use_row_as_scale`) **or** the 9 spatial taps (3×3 mode, `mv_qw_lim = qw = 8`). Utilisation is
therefore `real taps / 9`:

| | rows used | MAC/cycle | cycles for 3 taps × 32 cout |
|---|---|---|---|
| 3×3, all 9 taps real | 9/9 | 162 (peak) | — |
| **1×1 pointwise** | 8/9 | **144** | **96** |
| 3×3 with a `1×3` slice | 3/9 | 54 | 256 |

A **1-D** kernel touches at most one row of any 3×3 window, so at most 3 of 9 taps can ever be
real. Break-even against pointwise needs >8. SpeechNet can never reach it — this is geometry,
not tuning.

## 8. Risks

| risk | mitigation |
|---|---|
| 16 int32 buffers of `[1,16,14,87]` (78 KB each) blow the L1 budget (`--l1 110000`) | an `Add` chain keeps only 2 live; rely on the tiler, start single-buffer, expect `"Allocation failed for allocator 2"` as the first failure |
| `Slice` on the cluster costs more than the conv it feeds | measured explicitly by acceptance criterion 5 |
| the pre-encoded weight and Deeploy's compile-time encoder disagree | the fixture builder uses **the same `_weightEncode` function**, imported from Deeploy, not a reimplementation |
| tiling the weight: `serializeTilingSolution` assumes an encoded-tail layout | if it breaks, record it — it is the same class of bug as PR #183's 2 KB weight-tail overrun |

## 9. Order of work

1. Build the three fixtures + host reference; verify acceptance criterion 1. **No device.**
2. Deeploy changes C1–C3; re-verify the graph colours (`middleware_post_lowering.onnx`).
3. Device run on `GAP9_w_NE16`; criteria 2–4.
4. Device run on `GAP9` for the cluster denominators; criterion 5.
5. `Findings.md`, worklog, plan status.
