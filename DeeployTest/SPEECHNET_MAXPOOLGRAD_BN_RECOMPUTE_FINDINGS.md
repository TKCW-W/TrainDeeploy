# MaxPoolGrad: recompute X from the BN output — experiment & findings

**Status: experiment complete, memory goal NOT achieved (clean negative result, root-caused).**
Someone else is now working on "recompute from BN", so this records what was tried, what
works, the measured numbers, and *why* the memory win does not materialise — so we can circle
back without redoing it.

Branches (all the code below lives here, uncommitted → now committed on these branches):
- Onnx4Deeploy: `feat/maxpoolgrad-relu-checkpoint`
- TrainDeeploy:  `feat/maxpoolgrad-bn-recompute-experiment`

---

## 1. The idea

On-device MaxPool training currently uses **recompute-from-X**: the forward MaxPool emits only
`Y` (no indices), and the backward `MaxPoolGrad(dY, X)` re-scans the forward input `X` to find
the argmax and scatter `dY`. `X` here is the **ReLU output** (block layout: `Conv → BN → ReLU →
MaxPool`). The hypothesis was that holding `X` from forward to backward wastes memory, and that
since the **BN output is already kept resident for ReLUGrad**, we could free `X` and instead
get it at backward via `X = ReLU(BN output)`.

Two ways to realise this were implemented and measured, plus the baseline:

| # | approach | how |
|---|---|---|
| 0 | **baseline** (shipped) | `MaxPoolGrad(dY, X)`, `X` = forward ReLU output, held to backward |
| 1 | **checkpoint** | insert a new backward `ReLU'(BN out) → X'` node; `MaxPoolGrad(dY, X')` |
| 2 | **fused** | `MaxPoolGrad(dY, BN out)` + `apply_relu=1`; kernel applies `max(0,·)` inline (no `X'` tensor) |

All three are numerically identical: `argmax(max(0, BN)) == argmax(ReLU(BN)) == argmax(X)`, and
the output `dX` is the gradient w.r.t. `X` in every case.

---

## 2. What was implemented (where to look)

### Onnx4Deeploy — `onnx4deeploy/core/base_exporter.py`
Graph rewrites applied to the final `network.onnx` in `export_training()` (dispatched by config):
- `_rewire_maxpoolgrad_recompute` — baseline (existing): `MaxPoolGrad.input[1] = X` (ReLU out).
- `_checkpoint_maxpoolgrad_recompute` — flag `maxpool_checkpoint=True`: inserts a fresh
  `Relu(BN out) → X'` per pooled block, points `MaxPoolGrad.input[1]` at `X'`.
- `_fuse_maxpoolgrad_relu` — flag `maxpool_fuse_relu=True`: points `MaxPoolGrad.input[1]` at the
  **BN output** and appends attribute `apply_relu=1`. No new node.

### TrainDeeploy (compiler + kernel) — the `apply_relu` / fused path (additive, backward-compatible)
- `TargetLibraries/PULPOpen/src/MaxPool.c` — new kernel `PULP_MaxPoolReLUGrad2d_fp32_fp32_HWC`
  (copy of `PULP_MaxPoolGrad2d` with `if (val < 0) val = 0;` inline ReLU before the argmax compare).
- `TargetLibraries/PULPOpen/inc/kernel/MaxPool.h` — prototype.
- `Deeploy/Targets/PULPOpen/Templates/FloatMaxPoolTemplate.py` — `referenceGradTemplate` now emits
  `PULP_MaxPool${"ReLU" if apply_relu else ""}Grad2d_...` (picks the fused kernel when set).
- `Deeploy/Targets/Generic/Parsers.py` — `MaxPoolGradParser` reads `apply_relu` (default `0` →
  classic path, existing graphs unaffected).
- No binding / tiler / typechecker / lowering changes needed: the fused op is the **same**
  `MaxPoolGrad` op with the same 2 inputs / 1 output and the same shapes (BN output has the same
  shape as the ReLU output `X`), so the existing `NCHWtoNHWCMaxPoolGradPass` and tiling apply.

---

## 3. Correctness

- **checkpoint**: builds, runs on GVSoC, and **PASSES the tiled-l2 training test bit-exact** vs the
  ORT reference (same `1e-3` tolerance as the baseline `cmpbase`). ✅
- **fused**: graph verified (Relu count stays 5 — no new node; `MaxPoolGrad.input[1]` = BN output;
  `apply_relu=1`; forward ReLU outputs consumed only by the forward MaxPool). Build + GVSoC sim was
  **not run** (the memory result below made it moot) — but it is numerically identical by
  construction, and the new kernel is a trivial variant of the verified one.

---

## 4. Memory results — the goal was NOT achieved

Config: SpeechNet 4-step (random data), `--l1 128000 --l2 2000000 --defaultMemLevel L2
--memAllocStrategy MiniMalloc --searchStrategy random-max`.

| graph | L2 (peak) | L1 |
|---|---|---|
| baseline (held X) | **1,793,760 B (89.7%)** | 130,004 B (101.6%) |
| checkpoint (new ReLU' node) | **1,946,208 B (97.3%)** | 130,138 B |
| **fused** (read BN + apply_relu) | **1,793,760 B (89.7%)** | 130,004 B |

- **checkpoint is WORSE** (+152 KB): the scheduler **hoists** `ReLU'` into the forward pass (it only
  depends on the BN output, ready early) and **holds `X'` across `[12..102]` to backward** —
  recreating the exact hold it was meant to remove, plus a brief forward-`X`/`X'` overlap.
- **fused EQUALS baseline** (to the byte): frees the forward `X` but yields **no peak reduction**.

### Why (definitive root cause)
The **peak L2 is in the FORWARD pass** (~tiling step 12), where **five 314 KB tensors coexist**
(block-0 numbers; `X = (1,8,14,701)` fp32 = 314,048 B):

| tensor | lifetime (step) | held for |
|---|---|---|
| Conv output | [8..102] | BNGrad (reads Conv out + saved mean/inv_std) |
| BN output | [8..102] | ReLUGrad (+ MaxPoolGrad in the fused variant) |
| **forward X** (ReLU out) | **[10..12]** | **the forward MaxPool reads it here** |
| MaxPoolGrad's transposed input | [10..100] | NHWC transpose, **hoisted into forward** & held |
| forward MaxPool's transposed input | [12..14] | forward pool NHWC input |

The premise — "holding `X` for the *backward* MaxPoolGrad costs peak memory" — is **false for this
architecture**:
1. `X` is **unavoidably live at the forward peak** because the *forward* MaxPool consumes it there.
   Recompute/checkpoint/fuse only change whether `X` lives *after* step ~14 (backward), which is
   **not** the peak.
2. The NHWC **transpose** that Deeploy inserts for `MaxPoolGrad.input[1]` gets **hoisted into the
   forward and held** `[10..100]` — recreating a 314 KB tensor at the peak regardless of whether the
   input is `X` (baseline) or BN (fused).
3. The real memory drivers are the **Conv-out and BN-out forward activations** (held for ConvGrad /
   BNGrad), not the MaxPool input.

### Why a pure exporter fix can't pin the recompute late
Deeploy schedules with `graph.cleanup().toposort()` (graphsurgeon) at every stage, default
`scheduler = list(graph.nodes)`, and has **no "as-late-as-possible" / rematerialisation
mechanism**. So toposort always hoists a recompute (or a transpose) next to its earliest-ready
dependency. Pinning it late would need a real backward data-dependency (math-hacky) or a scheduler
change.

---

## 5. Reproduce

Generate (in the `agitated_hugle` / Onnx4Deeploy container; model must be MaxPool — see the
`feat/maxpoolgrad-relu-checkpoint` cleanup):
```python
from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter
ex = SpeechNetExporter(save_path='<DST>/speechnet_train_<variant>')
ex._config_overrides = {'dataset':'random','n_batches':4,'n_accum':1,
                        'maxpool_fuse_relu':True}   # or 'maxpool_checkpoint':True, or neither (baseline)
ex.export_training()
```
Measure peak memory (codegen only, fast; in `traindeeploy` / TrainDeeploy `DeeployTest`):
```
python testMVPTraining.py -d /tmp/dump -t Tests/Models/Training/SpeechNet/speechnet_train_<variant> \
  -p Siracusa --cores 8 --l1 128000 --l2 2000000 --defaultMemLevel L2 \
  --memAllocStrategy MiniMalloc --searchStrategy random-max --plotMemAlloc -vv | grep -A8 'Memory Usage Report'
# Per-tensor lifetimes: parse <dump>/deeployStates/memory_alloc.html (Plotly traces: name=tensor, x=step, y=address)
```
Correctness (build+GVSoC+loss): temporarily add the fixture to `L2_SINGLEBUFFER_TRAINING_MODELS`
in `test_siracusa_tiled_config.py`, then
`pytest test_platforms.py -m 'siracusa_tiled and training and l2 and singlebuffer' -k <variant>`.

(The `speechnet_train_cmp{base,ckpt,fuse}` scratch fixtures were **not committed** — regenerate
with the commands above. `--searchStrategy random-max` is randomised, so re-run a couple of times
for stable numbers.)

---

## 6. Conclusion & what would actually help

The recompute-from-BN idea **cannot reduce peak L2 for SpeechNet** — the MaxPool input is not on the
critical path; the peak is forward-bound by the Conv/BN forward activations (held for their
gradients) plus hoisted backward transposes.

Levers that *would* move peak L2 (out of scope here, for whoever continues):
- **Stop hoisting the backward transposes** (pin them late) — needs Deeploy scheduler support.
- **Avoid the redundant MaxPoolGrad-input transpose** by sharing the already-resident NHWC BN output
  with ReLUGrad (a lowering / transpose-CSE optimisation) — the fused variant sets this up (same BN
  tensor feeds both) but the lowering still emits a separate transpose.
- **Reduce the big forward activations** themselves: L3-offload or recompute the **Conv/BN** forward
  activations (the actual 314 KB drivers), or smaller L1 tiles.

The `apply_relu` fused op is a correct, additive feature and is kept for reference even though it
does not save memory; it (or the checkpoint) becomes useful only once the transpose/scheduling
issue above is addressed.
