# exp16a — Findings

Date: **2026-09-11** · Branch `feat/GAP9_w_NE16` · Plan: `./Plan.md`
Parent: `../Findings.md` (STEP 1 + 2a) · Worklog: `TrainDeeploy/WorkLog/GAP9_w_NE16_Worklog.md`

> **Status: CORRECTNESS ACHIEVED AT FULL EXTENT.** Block 1's real `1×16` kernel runs on NE16 at
> its true `14×87` size as 16 pointwise dispatches with streamin accumulation and a runtime
> pre-encoded weight — **`0 / 19712` errors**, no spatial crop, no layout transposes.
> Performance is *not* there and was not the goal — see §5.

---

## 1. Headline

SpeechNet **block 1's real `1×16` convolution**, taken from the **training** graph (weights as
runtime tensors, not constants), decomposed into **16 pointwise convolutions** and handed to
NE16.

Two blockers addressed:

| blocker | status |
|---|---|
| **1a** weight is not a `gs.Constant` (`canExecute` rejects; compile-time encoder cannot run) | ✅ **SOLVED** — weight delivered as a graph input in NE16 bit-serial layout, `weight_offset = -128` fixed. `canExecute` accepts 16/16 convs |
| **1b** in the full QZO loop the perturbed weight is produced *on device* | ⬜ deferred to exp16b (see §8) |
| **2** `1×16` is not an NE16 filter mode | ✅ **decomposition verified bit-exact on host**; device result in §4 |

---

## 2. Method

### 2.1 Fixtures

Builder: `./build_fixtures.py` → `TrainDeeploy/DeeployTest/Tests/Models/NE16/`
(`--taps K`, `--crop-h/--crop-w`, `--suffix`)

| fixture | graph | weights | role |
|---|---|---|---|
| `b1_ref_1x{K}_s` | `Conv(1×K)` + `RequantShift` | plain int8, graph input | golden reference / cluster A/B |
| **`b1_1x{K}_ne16_s`** | **`Conv(1×K)` + `RequantShift` — 2 nodes** | **NE16 pre-encoded uint8 graph input**, `(K·cout, cinMajor, 16)` | **the NE16 run** |
| `b1_pw{K}_plain_s` | `16×(Slice+Conv(1×1))+Add` chain | plain int8 | host-only arithmetic check (route abandoned, §9) |

All carry **identical `outputs.npz`** — the golden produced by the `1×K` reference.

### 2.2 Everything is real data

Extracted from `Tests/Models/Training/SpeechNet/speechnet_qzo12_train/network.onnx` via
`onnx4deeploy`'s `run_onnx_graph`:

```
[act  ] (1, 8, 14, 87) int8   range [0, 125]      ← the int8 activation into block-1's Conv
[wpert] (16, 8, 1, 16) int8   range [-127, 127]   ← blocks.1.conv.weight_int8 AFTER the real
                                                    RQSPerturbRademacher (seed 42, idx 2, real pmul)
[rqs  ] mul[16], add[16], div = 65536, signed = 1  ← the real RequantShift, add also perturbed
```

`act.min() == 0` re-confirms **by measurement** that block-1 activations are non-negative
(`MaxPool ← Relu`), so **BLOCKER 3 (the phantom signed-input bit 26) does not apply here**. The
builder asserts it rather than assuming it.

### 2.3 The decomposition, and where it lives

```
conv_{1×K}(W, X)[co,h,w] = Σ_{j=0..K-1} conv_{1×1}(W[:,:,0,j], Xpad)[co,h,w+j]
```

It is realised **in the template, not in the graph**. Tap `j` is an *address* offset on
`ne16_task_t.infeat_addr` (in NHWC, one pixel along W is exactly `ch_im_in` bytes) — which is what
`03-qzo-ne16-plan.md`'s "per-tap input slice offsets" meant. The graph stays at **two nodes**.

Accumulation uses **`streamin`** (CONFIG0[14]), so the K partial sums never leave NE16's
accumulator path. `Plan.md §4.1` originally proposed an `Add` chain first, to separate "is the
decomposition right" from "does a new accumulation mode work"; that was abandoned once it proved
to need `Pad` and `Slice` bindings GAP9 does not have for integer data (§9). The arithmetic was
still checked on the host first — see §3.

The `pads=[0,8,0,8]` are applied to the activation **in the fixture data** (pre-padded input),
never as NE16 padding: padding is invalid in 1×1 mode and its guard at `fsm.cpp:53` is
**commented out**, so it would fail silently.

---

## 3. ✅ Host check, before any device run

```
[b1_pw16_plain_s] plain int8: host check vs 1x16 reference -> BIT-EXACT (1600 elems)
[b1_pw_plain]     (full extent, earlier)                   -> BIT-EXACT (19712 elems)
```

The 16-tap decomposition is arithmetically exact against the `1×16` reference, with real weights
and real activations, established *before* touching the device. That is what made every later
device failure attributable to the NE16 path rather than to the decomposition.

---

## 4. Device results — GVSoC `gap9.evk`

Final design (per `03-qzo-ne16-plan.md` STEP 2b, **template-level** decomposition):

```
ONE Conv(1xK) node  --NE16-->  K pointwise dispatches
        tap j:  infeat_addr  += j * ch_im_in          (NHWC: one pixel along W)
                weights_addr += j * 256               (one tap's encoded block)
                conf0        |= NE16_FLAG_STREAMIN    for j > 0
        output: int32, accumulated IN NE16
RequantShift stays a separate cluster node (streamin forces quantization_bits==32)
```

| fixture | engine | taps | NE16 dispatches | streamin taps | cluster convs | errors | cycles |
|---|---|---|---|---|---|---|---|
| `b1_1x2_ne16_s` | **NE16** | 2 | **2** | 1 | 0 | **0 / 1600** | 100,704 |
| **`b1_1x16_ne16_s`** | **NE16** | **16** | **16** | **15** | **0** | **0 / 1600** | 154,110 |
| `b1_ref_1x2_s` | cluster | 2 | 0 | — | 1 | 0 / 1600 | 17,657 |
| `b1_ref_1x16_s` | cluster | 16 | 0 | — | 1 | 0 / 1600 | 35,715 |

The emitted config proves the mechanism rather than just the result:

```
tap0 conf0 = 4227143   streamin[14]=0   filter mode[6:5]=2 (1x1)   quant bits=32   outquant=0
tapj conf0 = 4243527   streamin[14]=1
difference = 16384 = 0x4000  ->  exactly NE16_FLAG_STREAMIN
```

**Full extent reached 2026-09-11** — the earlier `4×24` crop is gone. Three things were needed:

1. **Halo in the tile constraint** — `Wout = Win - (K-1)` on the tapped axis, input cubes carrying
   the `K-1` overlap, per-tile strides and subtile counters. Conv now tiles (`numTiles = 4`).
2. **NHWC-native fixture** (`--nhwc`) — with the RequantShift un-merged, the int32 conv output was
   being materialised in BOTH layouts (`2 × 78,848 B`) purely to satisfy an NCHW graph boundary.
   Emitting the fixture channels-last removes **both** layout transposes: arena `108,416 → 98,688 B`,
   `transpose_cluster_fork` count `2 → 0`.
   This needed one Deeploy fix: `RequantShiftLayer.computeShapes` hardcoded
   `channel_dim = inputShapes[0][1]` while its signature already took `channels_first`, so a
   channels-last standalone RequantShift always died with
   `Could not broadcast rqs_mul_tensor from (16,) to [1, 14]`.
3. **A realistic `--l1`** — `110000` is **MeZO-harness-only**: it depends on
   `pi_cluster_task_stacks()` relocating the cluster slave stacks to L2 in `deeploymezotest.c`.
   The *inference* harness `deeploytest.c` has no such relocation, so ~30 KB of L1 is still stacks
   and only **98,256 B** is usable. At `--l1 92000` the tiler also splits the RequantShift
   (`numTiles = 2`) and everything fits.

| fixture | engine | extent | dispatches | transposes | conv tiles | errors | cycles |
|---|---|---|---|---|---|---|---|
| **`b1_1x16_ne16_nhwc`** | **NE16** | **FULL 14×87** | **16** | **0** | **4** | **0 / 19712** | 1,567,096 |
| `b1_ref_1x16` | cluster | FULL 14×87 | 0 | — | — | 0 / 19712 | 337,657 |
| `b1_1x16_ne16_s` | NE16 | 4×24 (halo) | 16 | 2 | 4 | 0 / 1600 | 241,790 |

## 5. Performance — not yet, and not the goal

**NE16 is 4.64x SLOWER than the cluster at full extent** (1,567,096 vs 337,657 cycles). Expected,
and not a verdict on the approach:

* `Cin = 8` half-fills `TP_IN = 16`, so the array is at most 50 % utilised before anything else;
* **16 dispatches x 4 tiles = 64 NE16 jobs**, each paying `Ne16PerfModel`'s fixed `k_out_rem`
  setup — against ONE `pulp_nn_conv` call on the cluster;
* 15 of every 16 taps round-trip the int32 partial sums through L1 (streamin reads `outfeat_addr`);
* the int32 intermediate is 4x the int8 one, so every tile costs 4x the DMA.

The structural fix is to raise work-per-dispatch, which is exactly what **channel folding**
(im2col to a single `Cin = 128` pointwise conv: 1 dispatch, `TP_IN` fully packed, no streamin)
would do. That is STEP 4 and remains unmeasured.

Performance is STEP 4. What this experiment set out to establish — that the decomposition is
*correct* and that NE16 can consume a *runtime* weight — is established.

## 6. Deeploy changes made (all Python; **no NE16 ISA change**)

| # | file | change |
|---|---|---|
| C1 | `Targets/NE16/Engine.py` | `_weightAcceptable` (accept a runtime weight marked `ne16_weight_preencoded`); `is1xKConv` + `enable1xK` flag, **off by default** |
| C2 | `Targets/NE16/Parsers.py` | `NE161xKConv2DParser` — accepts `1xK`/`Kx1`, records `ne16_taps` |
| C3 | `Targets/NE16/Templates/Conv1xKTemplate.py` | **new** — K dispatches, per-tap pointer offsets, streamin for j>0 |
| C4 | `Targets/NE16/TileConstraints/NE161xKConstraint.py` | **new** — single-tile policy (see §4) |
| C5 | `Targets/NE16/Bindings.py`, `Tiler.py`, `Engine.py` | binding (2 inputs, int32 out), tiling-ready binding, mapper first in `NE16Mapping['Conv']` |
| C6 | `Targets/PULPOpen/TopologyOptimizationPasses/Passes.py` | `_merge_conv_rq_fun`: **do not** fuse a conv carrying `ne16_taps` — streamin needs int32 out, so the RequantShift must stay separate |
| C7 | `CommonExtensions/.../LoweringOptimizationPasses.py` | `_NCHWtoNHWC_fun`: take `spatialDims` from `kernel_shape` for pre-encoded weights, and **skip permuting a pre-encoded weight** (see §7) |
| C8 | `DeeployTest/testMVP.py`, `testUtils/deeployRunner.py`, `deeployRunner_tiled_gap9_w_ne16.py` | `--enable-1xk` threaded end to end |

Every change is gated on an NE16-only attribute or an off-by-default flag, so no other target and
no ordinary convolution changes behaviour. The emitted `ne16_task_t` uses only fields pulp-nnx
already has — **runs on real GAP9 silicon exactly as on `gap9.evk`**.

## 7. Bugs found, in order (each cost one 2-3 s iteration after the first)

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | **30-min "hang"** at K=16 | exponential backtracking over an *infeasible* binding problem; `ctxt.copy()` twice per candidate (`DeeployTypes.py:1606,1768`) made each futile attempt expensive | shrink to K=2 — same failure in **3 s** |
| 2 | `PARSING FAILED ... Layer 3 'b1_pad'` | GAP9 has only `Pad1DParser`/`Pad2DParser`; neither binds a 4-D int8 constant Pad | pre-pad in the fixture data |
| 3 | `PARSING FAILED ... 'b1_slice00'` | for **integer** data only `PULPDMASliceBindings` applies; needs 5 inputs with **uint8** index tensors and yields a `PULPDMAFuture` | abandoned the Slice route entirely (see §8) |
| 4 | `NoneType has no len()` | `helper.make_graph` annotates only inputs/outputs; intermediates had `shape=None` | explicit `value_info` |
| 5 | node became `_MERGE_CONVRQ_PASS_0` | `PULPConvRequantMergePass` fused Conv+RequantShift | C6 |
| 6 | weight `[2,16,1,16] -> [2,1,16,16]` | the NHWC pass permutes any rank-4 conv input | rank-3 weight layout `(K*cout, cinMajor, encBytes)` |
| 7 | `Rectangle offset should be zero ... (0,16) dims (1,16)` on **`b1_rqs`** | un-merging left the RequantShift **outside** the NHWC region, on an NCHW tensor, while the deployer default says NHWC — so it read the channel offset from the W axis | set `channels_first=1` on that node (`DeeployTypes.py:1198` lets a node attr override) |
| 8 | `perm=[0,1,3,2]` — H/W swapped | `_NCHWtoNHWC_fun:256` derives `spatialDims` from the **weight's rank**; a pre-encoded rank-3 weight gives `spatialDims=1` | C7 (use `kernel_shape`) |
| 9 | weight still permuted `[32,1,16]->[1,16,32]` | the same function permutes **every** input from index 1 | C7 (skip the pre-encoded weight) |
| 10 | `Allocation failed for allocator 2` -> `dma/trace addr 0x10101` | single-tile policy + full extent overflows L1 | spatial crop (§5) |

> **The one that generalises:** a long Deeploy codegen is a *symptom to shrink*, not a cost to
> wait out. Bug 1 hid nine further bugs behind half an hour of silence; at K=2 each surfaced in
> 2-3 seconds with an exact diagnosis.

## 8. Route abandoned, and why

The first implementation followed `Plan.md §4.1` literally: `Pad` + 16x(`Slice` + `Conv`) + 15x`Add`
in the **graph**. It was dropped after bugs 2-4 showed that integer `Slice` on GAP9 exists only as
an asynchronous DMA future, which our NE16 conv consumer cannot take. It also inflated the graph
to 49 nodes, which is what made bug 1 so expensive.

The template-level form keeps the graph at **2 nodes**, expresses the tap offset as an *address*
(which is what "per-tap input slice offsets" meant), and needs no `Pad`/`Slice` binding at all.

*(Channel folding — im2col to a single `Cin=128` pointwise conv — remains unexplored. It would fill
`TP_IN` completely and needs one dispatch, but moves the im2col outside the graph. Noted, not planned.)*

## 9. Next steps

1. **Real tiling** (removes the §5 crop): model the `K-1` input halo per tile and the streamin
   residency requirement, so the full `14x88` extent runs. This is the blocker for STEP 3.
2. **Blocker 1b** — on-device perturbation. exp16a supplies the already-perturbed weight from the
   host; the QZO loop computes it on device (`ApplyPerturbQuantRademacher_CHW`). Either the
   linearity decomposition (`conv(w+δz,x) = conv(w,x) + δ·conv(z,x)`, with the sign conv at
   `qw=2`, values `{0,2}`, `weight_offset=-1`) or a device-side encode kernel.
3. **Performance** (STEP 4): fold the two layout transposes, raise `Cin` utilisation, and consider
   whether `Cin=128` channel folding beats 16 dispatches.
4. Blocks 0/2/3/4 — block 0 additionally needs the signed-activation fix; 1-4 do not.

## 10. Artefacts (all under `exp16a_PW_single_layer/`)

```
Plan.md  Findings.md  results/results.json
build_fixtures.py            the fixture generator (--taps, --crop-h/--crop-w, --suffix)
_probe_1xk.py                canExecute gate check
_dump_lowered.py             lowered-graph dumper (shapes + Transpose perms)
_dumptypes.py                buffer/type dumper from a deeployStates pickle
fixture/b1_1x16_ne16_s/      THE fixture: 1 Conv(1x16) + RequantShift, weight_enc (256,1,16) uint8
fixture/b1_1x2_ne16_s/       K=2 variant
fixture/b1_ref_1x{2,16}_s/   cluster references (same golden outputs.npz)
fixture/Network_1x16_K16.c   generated C for the K=16 NE16 run
fixture/Network_1x2_K2.c     generated C for the K=2 NE16 run
fixture/deeployStates_K2/    4-stage ONNX/pkl dumps
logs/step1..step17*.log      every run, in order, including all ten failures
```
