# exp16c — on-device weight encoding (blocker 1b) and STEP 3

Date opened: **2026-09-11** · Branch `feat/GAP9_w_NE16`
Predecessors: `../exp16a_PW_single_layer` (all-pointwise, `0/19712`),
`../exp16b_Dense_single_layer` (dense 3×3 chunks, `0/19712`, 1.15× faster)

---

## 1. Why this experiment exists

exp16a and exp16b both prove NE16 computes SpeechNet's block-1 `1×16` conv **bit-exactly**. Both
cheat in the same place: the **host** hands the device a weight that is already perturbed and
already bit-serial encoded, as a `gs.Constant`.

In the real QZO loop neither is true. The conv's weight is the output of `RQSPerturbRademacher`
(`Deeploy/Targets/PULPOpen/Templates/RQSPerturbRademacherTemplate.py` →
`ApplyPerturbQuantRademacher_CHW`), recomputed **on device, twice per ZO step** (L⁺ and L⁻). It is
a runtime `int8` tensor, not a constant.

That is **blocker 1b** (= BLOCKER 2 in `docs/TRAIN_GAP9_NE16/03-qzo-ne16-plan.md §3.2`). Closing it
is what turns "NE16 computes this layer" into "NE16 can be used by QZO training".

**STEP 3** then follows: the other four SpeechNet convs.

## 2. The two candidate fixes — and why the plan's preferred one is now the second choice

### Option A — device-side encode kernel *(chosen)*

Insert one node between the perturbation and the conv:

```
RQSPerturbRademacher ──int8 w──▶ NE16WeightEncode ──uint8 w_enc──▶ NE16 Conv (16 PW dispatches)
```

The kernel reproduces `_weightEncode`
(`Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py:24`) in C. That function does two
things, and **exp16a already removed the hard half**:

| host step | on device |
|---|---|
| `weight_offset = values.min()`; `values -= weight_offset` | **not needed** — exp16a hardwires `weight_offset = −128`, so the offset step is exactly `w_u = w ^ 0x80` |
| bit-plane transpose + `packbits` | a **data-independent permutation** — pure bit shuffling, no arithmetic, no data dependence |

So what remains is a fixed bit transpose. For one `(co, cinMajor)` pair: read 16 `uint8`, emit 16
`uint8` where output byte `b*2 + k` collects bit `b` of source bytes `8k..8k+7`. Two 8×8 bit
transposes. Parallelise over `cout`.

Size for block 1: `cout=16`, `cin=8→16` padded, `K=16` taps ⇒ encoded output
`(K·cout, cinMajor, bits·cinMinorBytes) = (256, 1, 16) = 4096 B`. Against a 1.57 M-cycle conv this
should be noise — **which is the hypothesis to measure, not assume** (exp16b is a standing reminder
that cost models here have been wrong).

### Option B — linearity decomposition *(deprioritised, was the plan's recommendation)*

`03-qzo-ne16-plan.md §3.2` recommends exploiting `w_pert = w + δ_co·z`:

```
conv(w_pert, x)[co] = conv(w, x)[co] + δ_co · conv(z, x)[co]
```

with `conv(w,x)` encoded offline once and `conv(z,x)` run at `qw=1` (`z ∈ {−1,+1}`, one bit-serial
pass instead of eight, "encoding" = packing sign bits).

> **exp16b's measurement undercuts this.** The argument for B is that it avoids eight bit-serial
> passes — i.e. it is a **MAC-cycle** argument. exp16b showed this layer is **DMA- and setup-bound,
> not MAC-bound**: dropping from 16 to 6 dispatches made it *faster* despite 8× the bitplane
> passes. Option B goes the wrong way on the metric that actually dominates — it **doubles the
> dispatch count** (16 → 32 for block 1), and additionally needs:
>
> - a new per-channel `int32` scale-and-accumulate kernel for `δ_co · conv(z,x)`;
> - both `δ⁺` and `δ⁻` carried, since `>>S` makes the perturbation asymmetric (§3.2's own caveat);
> - `z` still re-packed every step anyway — so it does not even eliminate a device-side encode,
>   only shrinks it 8×.
>
> Option A keeps the already-bit-exact exp16a datapath completely untouched and adds one cheap,
> self-contained kernel. It is both lower-risk and aligned with the measured bottleneck.

B stays on the table as a fallback **if and only if** Phase 1 measures the encode cost as
non-negligible. Recording the reversal explicitly so §3.2 of the plan doc is not silently ignored.

## 3. Scope and phases

| phase | goal | gate to next |
|---|---|---|
| **1** | `NE16WeightEncode` kernel + node; weight arrives as a **graph input** (still unperturbed), encoded on device. Bit-exact vs exp16a's host-encoded golden, at full 14×87 extent. | `0/19712` |
| **2** | Wire `RQSPerturbRademacher → NE16WeightEncode → Conv` in a QZO-shaped fixture; run L⁺ and L⁻. Bit-exact vs a host reference that perturbs then convolves. | `0/19712` both signs |
| **3** | STEP 3a: blocks **2, 3, 4** on NE16 (`1×8`, `7×1`, `7×1`) — no new blocker, only the `K×1` axis and smaller extents. | each bit-exact |
| **4** | STEP 3b: block **0** (`1×4`, `1→8`) — additionally needs the **signed-activation** fix (BLOCKER 3): `+128` on the input and `−128·Σw` folded into the bias. | bit-exact |

Phases 3 and 4 are STEP 3. Phase 4 is the only one carrying an unsolved blocker of its own; if it
resists, phases 1–3 still constitute a complete, reportable result (4 of 5 convs = 91.5 % of MACs).

**Non-goals:** STEP 4 optimisation (channel folding), the `update` graph, backward ops,
multi-step training accuracy. NE16 is still 4× slower than `pulp_nn_conv` — that is STEP 4's
problem, not this experiment's.

## 4. Hard constraints carried forward

1. **No NE16 ISA change.** The encode kernel is ordinary C compiled into `deeploygap9`
   (`TargetLibraries/GAP9/src/` is `GLOB_RECURSE`'d, so a new `.c` is picked up automatically).
   `TargetLibraries/third_party/pulp-nnx/` stays pristine. This must run on real GAP9 silicon.
2. **`weight_offset = −128`, fixed.** Not `values.min()`. A data-dependent offset cannot work when
   the weight changes every step, and it is what makes the device offset step a single XOR.
3. **int32 conv output**, RequantShift stays a separate cluster node — forced by
   `assert(!(streamin && quantization_bits != 32))` (`fsm.cpp:50`).
4. **Commit and document before each phase**, updating `WorkLog/GAP9_w_NE16_Worklog.md`.

## 5. Success criteria

- Phase 1–2: `Errors: 0 out of 19712` with the weight encoded **on device**, and the encode cost
  measured as a fraction of the layer's cycles.
- Phases 3–4: each remaining SpeechNet conv bit-exact on NE16.
- A recorded answer to "is Option A's encode cost negligible?" — with numbers, not a model.
