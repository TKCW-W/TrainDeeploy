# BP (First-Order) On-Device Incremental Fine-Tuning — Reproduction Flow + MaxPool Optimization

> End-to-end reproduction of **backprop (first-order) on-device fine-tuning** of SpeechNet on Siracusa/GAP9,
> from graph export (**Onnx4Deeploy**) through code generation + tiling + build + cycle-accurate GVSoC
> simulation (**TrainDeeploy/Deeploy**), for the **incremental b1→b5 protocol** (train on batch *r*, evaluate
> on batch *r+1*, carry the trained weights forward). Uses the **MaxPool argmax-mask + transpose-dedup**
> memory optimization (§B) that brings the graph under GAP9's 1.5 MB L2.
>
> Companion of `ZO_TECHNICAL_REPORT.md` (the zeroth-order flow). Reference: the original plan lives at
> `DeeployTest/experiments/exp1/ondevice_sim_S01_fold3/PLAN.md`; the memory optimization write-ups are
> `DeeployTest/experiments/exp3/FINDING.md` and `DeeployTest/experiments/exp4/FINDING.md`.

---

## A. Reproduction flow

### A.0 Setup — containers, paths, recipe

**Containers** (repos are bind-mounted; same files inside each):
- **`agitated_hugle`** — Onnx4Deeploy at `/app/Onnx4Deeploy` (has `onnxscript`/onnxruntime-training needed for
  export). It also sees the TrainDeeploy tree at `/app/ETH/TrainDeeploy`, so exports write fixtures there directly.
- **`traindeeploy`** — TrainDeeploy at `/app/ETH/TrainDeeploy` (LLVM/RISC-V toolchain + GVSoC).

Enter a container with e.g. `docker exec -it agitated_hugle bash` / `docker exec -it traindeeploy bash`; every
command block below states which container it runs in.

**Fixed paths (exact — no placeholders):**
| what | path |
|---|---|
| Onnx4Deeploy (export) | `/app/Onnx4Deeploy` (agitated_hugle) |
| TrainDeeploy DeeployTest | `/app/ETH/TrainDeeploy/DeeployTest` (traindeeploy) |
| SilentWear data | `/app/SilentWear/SilentWear_data/data_raw_and_filt` |
| Pretrained round-1 checkpoint (S01 / vocalized / fold-3) | `/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt` |
| Train fixture (round *r*) | `/app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b{r}_fold3` |
| Infer fixture (round *r*, eval batch *r+1*) | `/app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b{r}_fold3` |
| Carry checkpoint (device weights after round *r*) | `/tmp/carry_fullfrozen_b{r}_fold3.pt` |
| Experiment dir (scripts, logs, results) | `/app/ETH/TrainDeeploy/DeeployTest/experiments/exp1/ondevice_sim_S01_fold3/` |

**Recipe (S2 — full-model, frozen-BN):**
| knob | value |
|---|---|
| scope | full model (conv + BN γ/β + fc), **no BN folding** |
| BN | **frozen pretrained stats**, train ≡ inference (`--bn-frozen-stats`; device `BN_FROZEN_STATS=ON`) |
| optimizer | SGD, no momentum, no weight-decay |
| lr | 3e-4, static |
| n_accum | 4, **SUM** grads (effective batch 1) |
| epochs | 40; FT data 54 windows (30 %, 6/class, seed 42, `--stratified`) |
| device steps | 540 = 40 epochs × 54 windows ÷ n_accum 4 |
| protocol | incremental b1→b5: round *r* trains on batch *r*, evaluates batch *r+1*; weights carried forward |
| MaxPool | **argmax-mask + transpose-dedup** (`--maxpool-argmax-mask`; §B) → fits GAP9 L2 (1.5 MB) |

**Per-round parameter table** (substitute into the round-1 commands below):
| round *r* | train batch | pretrained-weights | carry out | eval batch (*r+1*) |
|---|---|---|---|---|
| 1 | 1 | `…/leave_one_session_out_fold_3.pt` (official) | `/tmp/carry_fullfrozen_b1_fold3.pt` | 2 |
| 2 | 2 | `/tmp/carry_fullfrozen_b1_fold3.pt` | `/tmp/carry_fullfrozen_b2_fold3.pt` | 3 |
| 3 | 3 | `/tmp/carry_fullfrozen_b2_fold3.pt` | `/tmp/carry_fullfrozen_b3_fold3.pt` | 4 |
| 4 | 4 | `/tmp/carry_fullfrozen_b3_fold3.pt` | `/tmp/carry_fullfrozen_b4_fold3.pt` | 5 |

The blocks below are written **concretely for round 1**; for rounds 2–4 swap `b1`→`b{r}`, the batch number,
and the `--pretrained-weights` per the table. `b1 zero-shot` = evaluate batch 1 with the official checkpoint
via the eval path (A.5) — the chain's start point (no FT).

---

### A.1 Export the TRAIN fixture — Onnx4Deeploy (run in `agitated_hugle`)
Emits the training graph (perturb-free backprop graph with `BatchNormInternal` + frozen BN + SGD artifacts),
the **MaxPoolArgmax/MaxPoolGradMask** rewrite (`--maxpool-argmax-mask`), and the npz (init weights + 54 FT
windows + per-step frozen reference losses [Option A] + updated params).
```bash
cd /app/Onnx4Deeploy
python3 Onnx4Deeploy.py -model SpeechNet -mode train \
  -o /app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3 \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
  --subject S01 --session 3 --batch 1 --condition vocalized \
  --data-size 54 --n-epochs 40 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --stratified \
  --maxpool-argmax-mask
```
Produces in that dir: `network.onnx` / `network_train.onnx` (training graph), `inputs.npz` (init weights +
54 windows), `outputs.npz` (per-step frozen reference losses + updated params).

> ⚠️ **The `-o` directory name MUST contain `_train`.** The BP fixture is **two dirs**: the `_train` dir (above)
> and a sibling **`_optimizer` dir** holding the standalone SGD graph (`Counter({'SGD': 22})`) as its
> `network.onnx`. The export auto-creates that sibling via `derive_optimizer_dir` (`onnx4deeploy/core/optimizer_onnx.py`),
> which **replaces `_train`→`_optimizer` and returns `None` (skips generation) if the name has no `_train`**
> (`base_exporter.py:1201`). The device runner (A.2) resolves its optimizer graph the same way
> (`resolve_optimizer_dir`: `name.replace("_train","_optimizer")`, or an explicit `--optimizer-dir`).
> So `…/speechnet_train_fullfrozen_b1_fold3` → auto `…/speechnet_optimizer_fullfrozen_b1_fold3`. If you name the
> dir without `_train` (e.g. `…/bp_argmax`), **no optimizer dir is generated**, the runner falls back to the
> *same* dir, and it tries to tile the **full training graph** as the optimizer → *"Backtracking exhausted at
> SoftmaxCrossEntropyLoss"*. Always name it `<model>_train[...]`.

### A.2 TRAIN on device — tiled Siracusa runner, dump device weights (run in `traindeeploy`)
Kill any orphan GVSoC first, then run 540 update steps with `n_accum=4`, frozen-BN, weight-dump. The
`--l2 1500000` matches **GAP9's 1.5 MB L2** (the argmax-mask + dedup make it fit; §B).
```bash
pgrep -f gvsoc_launcher | xargs -r kill -9          # kill orphan GVSoC before each run (§gotcha)
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3 \
  --n-steps 540 --n-accum 4 --cores 8 \
  --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D DUMP_WEIGHTS=ON BN_FROZEN_STATS=ON \
  2>&1 | tee experiments/exp1/ondevice_sim_S01_fold3/logs/round1_gvsoc_train.log
```
- **Save the full log** (`logs/round1_gvsoc_train.log`) — it holds the `[WDUMP s=<step> wi=<i> n=<n>] <hex…>`
  weight dumps *and* the per-step losses used for the MaxPool-drift analysis.
- Loss **breaches after the drift onset (~step 133 for round 1)** are expected — the inherent device-vs-ORT
  MaxPool argmax tie-flip (§B), **not** a failure; the `[WDUMP]` weights are still valid.
- Memory-only check (no long sim): add `--skipsim --plotMemAlloc` → `memory_alloc.html` under
  `TEST_SIRACUSA/Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3/deeployStates/`.

### A.3 Extract device weights → carry checkpoint (run in `traindeeploy`)
Parses the **last** `[WDUMP …]` block from the train log, value-matches each of the 32 dumped tensors to its
ORT reference name in `outputs.npz`, reshapes, and writes them over the official checkpoint's trainable
tensors (BN running_mean/var stay frozen).
```bash
cd /app/ETH/TrainDeeploy/DeeployTest/experiments/exp1/ondevice_sim_S01_fold3
python3 extract_device_weights.py \
  --gvsoc-log logs/round1_gvsoc_train.log \
  --base-ckpt /app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
  --out-carry /tmp/carry_fullfrozen_b1_fold3.pt
```
(Script: `extract_device_weights.py` in this dir — CLI: `--gvsoc-log`, `--base-ckpt`, `--out-carry`. For
rounds ≥2 pass `--base-ckpt /tmp/carry_fullfrozen_b{r-1}_fold3.pt`.)

### A.4 Export the INFER fixture with the carry weights — Onnx4Deeploy (run in `agitated_hugle`)
Bakes the device-trained weights into an **inference** graph (frozen BN → inference form) and emits the eval
batch (*r+1*) windows + their ORT-reference logits.
```bash
cd /app/Onnx4Deeploy
python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
  -o /app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b1_fold3 \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /tmp/carry_fullfrozen_b1_fold3.pt \
  --subject S01 --session 3 --batch 2 --condition vocalized
```
Produces `network.onnx` (updated weights baked as initializers), `inputs.npz` (180 eval windows of batch 2),
`outputs.npz` (ORT reference logits/predictions).

### A.5 EVALUATE on device — untiled inference accuracy harness (run in `traindeeploy`)
Runs the Siracusa inference harness over the 180 windows of batch *r+1* → balanced accuracy (the on-device FT
accuracy for round *r*).
```bash
cd /app/ETH/TrainDeeploy/DeeployTest
python3 experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
  --infer-dir Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b1_fold3
```
`b1 zero-shot`: run A.4 with `--pretrained-weights …/leave_one_session_out_fold_3.pt --batch 1 -o …_infer_fullfrozen_b0_fold3`
then A.5 on that dir → accuracy of the **un-fine-tuned** model on batch 1 (chain start point).

### A.6 PyTorch reference on the SAME data (matched comparison, run in `agitated_hugle`)
Removes the draw confound: loads the official checkpoint, and for each round extracts the exact 54 FT windows
from the round's `speechnet_train_fullfrozen_b{r}_fold3/inputs.npz`, runs the S2 recipe in PyTorch
(`model.eval()` frozen BN, full model, SGD lr 3e-4, n_accum 4 SUM, 40 ep), carries forward, and evaluates
batch *r+1* on the same eval windows — so on-device vs PyTorch is apples-to-apples.
```bash
cd /app/ETH/TrainDeeploy/DeeployTest/experiments/exp1/ondevice_sim_S01_fold3
python3 run_pytorch_fullfrozen_ondevicedata_chain.py
```
Reference S2 numbers (host, own draw): 4-subject b2–b5 = 77.05; S01 b2–b5 = 85.32.

### A.7 Deliverables
- `experiments/exp1/ondevice_sim_S01_fold3/results/ondevice_vs_pytorch_S01_fold3.csv` — per batch: on-device
  acc, PyTorch (matched) acc.
- `experiments/exp1/ondevice_sim_S01_fold3/logs/round{1..4}_gvsoc_train.log` — kept for drift analysis.
- `experiments/exp1/ondevice_sim_S01_fold3/FINDING.md` — dated results + drift + comparison.

**Runtime note.** Full-model training is ~2160 fwd+bwd passes/round on GVSoC (heavier backward than
head-only) → each round is a **multi-hour** sim; 4 rounds + evals span a long wall-clock. Background the
runs, save the logs. Static allocation ⇒ the memory footprint is independent of `--n-steps` (a short
`--n-steps 4` run is a valid memory-fit check).

---

## B. The MaxPool argmax-mask memory optimization

> Goal: shrink the on-device backprop **L2 peak** so SpeechNet's full-model FT graph fits **GAP9's 1.5 MB L2**.
> Result: **L2 peak 1,793,800 B → 1,511,308 B (−15.7 %) at a loose 2 MB budget, repacking to 1,482,636 B
> under a strict 1.5 MB budget** → fits GAP9 with ~17 KB (decimal) / ~90 KB (binary) headroom, **bit-exact**.
> (exp3 = the optimization; exp4 = the GAP9 fit.)

### B.1 The idea (conceptual)
MaxPool's backward needs to know **which input element was the window max** so it can scatter `dY` to it.
Deeploy's stock `PULP_MaxPoolGrad2d` **recomputes the argmax from the forward input X at backward time** —
which forces the large **block-0 activation X** (~314 KB) to be **stashed live from the forward pass all the
way to the backward pass**. On an L1/L2-bound MCU that long-lived stash is a big chunk of the peak.

**Idea:** have the forward pass emit a small **within-window argmax offset** buffer (the mask); the backward
reads the mask and scatters — so the big activation X can be **freed right after the forward pass**.

**Correctness invariant (why it survives tiling):** store the **within-window offset** `p·Q+q` (the winner's
position *inside* its pooling window), which is **tile-position-independent**. The backward reconstructs the
absolute input position **locally** from the output position + offset (`p=off/Q, q=off%Q`, input =
`window_origin + (p,q)`), so it does not depend on where the tile sits in the full tensor. The tie-break
(strict `>`, first-max) is identical in the forward MaxPool and MaxPoolArgmax, so the stored winner is exactly
the pooled value's source.

**Design choice — a separate single-output op.** The naive "2-output MaxPool (pooled + mask)" is blocked by
the tiler (`TileConstraint.py:134` asserts one output per node). So we add a **separate single-output
`MaxPoolArgmax` op** (and a matching `MaxPoolGradMask`), keeping every node single-output → **no tiler-core
changes**.

### B.2 The implementation (technical — where & how)
1. **Device kernels** — `TargetLibraries/PULPOpen/src/MaxPool.c` (+ `inc/kernel/MaxPool.h`):
   - `PULP_MaxPoolArgmax2d_fp32_fp32_HWC` — forward argmax: the same window scan + tie-break as
     `PULP_MaxPool2d`, writes the within-window offset `p·Q+q` per output element into the mask.
   - `PULP_MaxPoolGradMask2d_fp32_fp32_HWC` — backward: reads the offset, reconstructs the input position
     locally, scatters `dY` — **no recompute, no forward-activation read**. (Offset stored as **fp32**, not
     uint8 — see Issue A.)
2. **Codegen templates** — `Deeploy/Targets/PULPOpen/Templates/FloatMaxPoolTemplate.py`: `argmaxTemplate`
   (calls `PULP_MaxPoolArgmax2d`, output = mask, same `(H,W)=(y,x)` dim mapping as the forward template) and
   `referenceGradMaskTemplate` (calls `PULP_MaxPoolGradMask2d`; 2nd input = the pooled-shape mask).
3. **Op registration (PULPOpen)** — `TypeCheckers.py` (`PULPMaxPoolArgmaxChecker`), `Bindings.py`
   (`PULPMaxPoolArgmaxBindings`, `PULPMaxPoolGradMaskBindings`; the recompute-grad binding is kept),
   `Tiler.py` (`PULPMaxPoolArgmaxTilingReadyBindings` reusing the single-output MaxPool channel-tiling
   constraint; `PULPMaxPoolGradMaskTilingReadyBindings`), `Platform.py`
   (`MaxPoolArgmaxMapper = NodeMapper(MaxPool2DParser(), …)`; `'MaxPoolArgmax'`/`'MaxPoolGradMask'` mappings).
   Parsers reuse `MaxPool2DParser` / `MaxPoolGradParser`.
4. **Graph rewrite** — Onnx4Deeploy `onnx4deeploy/core/base_exporter.py`, behind the CLI flag
   **`--maxpool-argmax-mask`**: instead of `_rewire_maxpoolgrad_recompute` (which rewires `MaxPoolGrad(dY, X)`
   and stashes X), the exporter **inserts a `MaxPoolArgmax` node** (input = the MaxPool's forward input; output
   = the offset mask) and wires `MaxPoolGradMask(dY, mask)`. `MaxPool` stays single-output (pooled only).
   Default (no flag) = unchanged recompute path.
5. **Layout passes (the tiler unblock)** —
   `Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py`:
   add `"MaxPoolArgmax"`/`"MaxPoolGradMask"` to `_NCHWtoNHWC_fun`'s `spatialDims` op-list; extend the
   MaxPoolGrad 2nd-input transpose special-case to `MaxPoolGradMask` (its 2nd input = the mask); add
   `NCHWtoNHWCMaxPoolArgmaxPass` + `NCHWtoNHWCMaxPoolGradMaskPass` and register them in the composite
   `PULPNCHWtoNHWCPass`.
6. **Transpose-dedup (the actual memory win)** — `LoweringOptimizationPasses.py` +
   `Deeploy/Targets/PULPOpen/Deployer.py`: add **`MergeSiblingTransposesPass`** (`@contextagnostic`) that
   merges `Transpose` nodes with identical `(input tensor, perm)` into one shared (fanout) transpose, rewiring
   all consumers; wire it as the **last** pass in `PULPDeployer`'s lowering pipeline (after the final
   `TransposeSplitPass`, so the merge is not re-split).

### B.3 Challenges encountered → root cause → fix
- **Issue A — uint8-mask "type friction" (worked around).** The first design used a **uint8** mask (4× smaller)
  and dtype-dispatched the existing `MaxPoolGrad`. It hit type friction and was switched to an **fp32 offset
  mask + a distinct `MaxPoolGradMask` op**. Two changes were bundled so the exact blocker isn't isolated —
  most likely a uint8 tensor flowing as an *activation* through the all-fp32 transpose/tiling pipeline and/or
  the dtype-dispatch binding ambiguity. **Deferred:** retry uint8 now (distinct op + layout fix + dedup all in
  place) to shrink the long-lived mask ~4× (not peak-changing).
- **Issue B — tiler `minimizeRectangle` assertion (SOLVED).** Build failed at `TilingCodegen.py:537`
  (`offset should be zero when dims == reference`). **Root cause:** the new ops had **no NCHW→NHWC lowering
  pass**, so their tensors stayed NCHW while the PULP channel-tiler assumes **HWC (C=last)** → it tiled the
  wrong (W) axis. **Fix:** the layout passes in §B.2-5. (Not the fundamental tiler limitation — the mask has
  the pooled-output shape and tiles cleanly once in NHWC.)
- **Issue C — the correct build first looked like a +1.8 % memory REGRESSION (SOLVED).** After Issue B the
  build was correct (0/16) but L2 = 1,825,356 B, *higher* than baseline. Two false leads were ruled out —
  (1) `random-max` search noise (wrong: 4 re-runs gave identical 1,825,356, i.e. deterministic), and (2) a
  stale pre-layout-fix html that misleadingly showed the expected reduction. **Root cause (buffer diff):** the
  correct build had **2 extra 314 KB block-0 buffers** = per-op **input transposes**. `MaxPool`,
  `MaxPoolArgmax`, `MaxPoolGradMask` each transpose the *same* 314 KB block-0 activation to NHWC, and the
  PULP pipeline's `TransposeSplitPass` (runs twice) **deliberately gives each consumer its own input
  transpose** → the transpose is duplicated per op (+1.8 % ≈ one 314 KB buffer). **Fix:** the
  `MergeSiblingTransposesPass` (§B.2-6) collapses the three siblings to one shared transpose → **−15.7 %**.

### B.4 Result
| build | L2 peak | vs baseline | fits GAP9 1.5 MB? | loss |
|---|--:|--:|:--:|---|
| recompute `MaxPoolGrad` (baseline, **pre-dedup**) | 1,793,800 B | — | ❌ | bit-exact |
| recompute `MaxPoolGrad` (**with** transpose-dedup) | 1,735,180 B | −3.3 % | ❌ | bit-exact 0 err |
| argmax-mask, **no** transpose-dedup | 1,825,356 B | +1.8 % | ❌ | bit-exact 0/16 |
| **argmax-mask + transpose-dedup** (loose 2 MB budget) | **1,511,308 B** | **−15.7 %** | — | **bit-exact 0/16** |
| **argmax-mask + dedup, strict `--l2 1500000`** | **1,482,636 B** | | ✅ (~17 KB spare @ decimal, ~90 KB @ 1.5 MiB) | **bit-exact 0/16** |

- **Dedup note.** The `1,793,800 B` recompute baseline was measured *before* `MergeSiblingTransposesPass`
  (transpose-dedup) was wired **unconditionally** into the PULP lowering pipeline (`Deployer.py:64`). That pass
  is general — it also dedups the recompute path — so recompute-*with*-dedup is **1,735,180 B** (A/B-confirmed:
  disabling the pass returns it to exactly 1,793,800 B). The fair, both-with-dedup comparison is therefore
  **recompute 1,735,180 B → argmax-mask 1,511,308 B (−12.9 %)** (see
  `DeeployTest/experiments/deliverable/SUMMARY.md`). It is a memory reduction, correctness-preserving.
- Deterministic (identical across re-runs). L1 fits at `--l1 128000` (< GAP9's 131,072 B).
- **Where the peak is now:** after the dedup the L2 ceiling (1,511,308 B) is set in the **forward** pass by the
  `Conv_input_*_transposed` buffers of blocks 2–3 — **not** by any MaxPool/argmax buffer. The MaxPool-side win
  is captured; further L2 reduction means deduping/fusing those forward conv-input transposes (a separate
  target), or a real uint8 mask.

### B.5 Important caveat — the argmax-mask does NOT remove the device-vs-ORT MaxPool drift
The exp1 loss "breaches" come from **fp reduction-order differences in X** (the conv/BN output; device-tiled
vs ORT-untiled, ~1e-6) that **flip a MaxPool tie**. Both the recompute `MaxPoolGrad` and the new
`MaxPoolArgmax` compute `argmax(X)` from the **same** stashed X, so the tie-flip susceptibility is
**unchanged** — the mask only guarantees device forward/backward agree (which they already did in the
recompute path, both scanning the same X). The 0/16 memtests are too short to see it (`--n-steps 4` = 16
micro-batches; round-1 drift onset ≈ step 133). **So the optimization is a memory + math-correctness win; the
device-vs-ORT MaxPool argmax drift is inherent and persists at full round length.**

### B.6 Commit / artifact pointers
- TrainDeeploy (`feat/BNFRozen_OptionB`): kernels `24c4a82` · templates `ca1952d` · registration `e844ada` ·
  fp32 mask `e322b08` · layout fix `22be691` · **transpose-dedup `c6aa5bc`** · exp3 write-up `a3bb820` ·
  drift correction `1d7cefe` · exp3/exp4 capstone `8d4682e` · GAP9 fit `809d364`.
- Onnx4Deeploy (`feat/BNFRozen_OptionB`): graph rewrite `56b3709` · fp32 mask `e6af342` · ceil_mode `3f8ca50`.
- Artifacts: `experiments/exp3/{FINDING.md,PLAN.md,logs/,results/memory_alloc_dedup.html}`,
  `experiments/exp4/{FINDING.md,logs/gap9_l2_1p5M.log,results/memory_alloc_gap9_1p5M.html}`,
  `experiments/maxpool_numerical_drift/SPEECHNET_MAXPOOL_DRIFT_ANALYSIS.md`.
