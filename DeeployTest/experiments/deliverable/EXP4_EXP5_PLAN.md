# Plan — exp4 (BP) & exp5 (ZO): complete round-1 on-device fine-tuning + carry checkpoint + batch-2 accuracy

## Context
We have single-step deliverables for BP (argmax off/on) and ZO. The next step is the **complete first
round** of incremental fine-tuning on device (Siracusa/GVSoC), for both first-order (BP) and zeroth-order
(ZO): train on batch 1, **dump the device-trained weights into a carry checkpoint**, then evaluate accuracy
on the **whole batch-2** data. BP already has a fully-documented, proven flow (`BP_FLOW.md` §A, `exp1/`).
ZO has the validated recipe and export/run flow, but **cannot dump weights today** — its harness only
compares losses — so exp5 needs a small, BP-mirrored weight-dump addition first. Branch: `feat/BP+ZO`
(both repos). Outputs land under `TrainDeeploy/DeeployTest/experiments/deliverable/`.

Fixed paths (both containers see the shared tree; export in `agitated_hugle`, sim in `traindeeploy`):
- Official round-1 ckpt: `/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt`
- Data: `/app/SilentWear/SilentWear_data/data_raw_and_filt` (subject S01, session 3, condition vocalized)
- Deliverable dirs (new): `experiments/deliverable/exp4_BP_round1/{logs,results}` and `exp5_ZO_round1/{logs,results}`
- **This plan is saved to `experiments/deliverable/EXP4_EXP5_PLAN.md`** (copied there as the first execution step).

### Same FT data for BP and ZO (must differ only in graph)
Confirmed in code: BP (`speechnet_exporter.py:384`) and ZO (`speechnet_exporter.py:630`) both draw the FT
windows via the **identical call** `data_source.load_batches(effective_data_size, input_shape, num_classes,
seed=42)`. The SilentWear stratified draw (`silent_wear_datasource.py:216`) is deterministic —
`np.random.RandomState(42)`, `n_per_class = n // n_classes`, iterating `sorted(class_to_idx)`. So with
**identical** `--data-size 54 --stratified --batch 1 --subject S01 --session 3 --condition vocalized`, both
fixtures get the **exact same 54 windows** (BP's 40 epochs vs ZO's 200 epochs only change how many times those
same 54 windows are cycled, via `mb % effective_data_size`). The batch-2 eval uses `-mode infer --batch 2`
which loads the **whole** batch 2 (180 windows, no random draw) → identical for both automatically.
**Guarantee step (do this after both train-fixture exports, before the long sims):** byte-compare the drawn
windows in the two `inputs.npz` files — the per-mini-batch `mbN_arr_0000` (input) and `mbN_arr_0001` (label)
entries must be bit-identical between `speechnet_train_fullfrozen_b1_fold3` and `speechnet_zo_train_b1_fold3`.
If they ever differ, force the match by reusing one fixture's window arrays in the other. Abort the runs on mismatch.

### `--num-data-inputs 2` (what it is)
The ZO runner flag = the count of graph inputs that **change every mini-batch** = the *data* inputs:
`input` (the EMG window `[1,1,14,700]`) + `label` (`int64 [1]`) = **2**. The other 22 graph inputs are the
persistent weight buffers (shared/aliased across steps, not re-fed per mini-batch). It is the ZO analogue of
BP's auto-detected `DATA_INPUTS=2`; the runner uses it to know how many fresh tensors to pull from `inputs.npz`
per step vs. which buffers to carry over. Value is 2 for SpeechNet (input + label).

---

## exp4 — BP round 1 (no new code; run the documented flow)

**FT settings (recipe S2):** full model (conv + BN γ/β + fc), **frozen BN** (`--bn-frozen-stats` /
`BN_FROZEN_STATS=ON`), SGD lr **3e-4**, no momentum/wd, **n_accum 4 SUM**, **40 epochs**, **54** stratified
windows (6/class, seed 42), **540** device steps, MaxPool **argmax-mask**, L1 128000 / L2 1500000 (GAP9).

**Paths:** train fixture `Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3` (name contains
`_train` → auto `speechnet_optimizer_fullfrozen_b1_fold3` sibling); carry `/tmp/carry_fullfrozen_b1_fold3.pt`;
infer fixture `Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b1_fold3` (batch 2, 180 windows).

**Commands (exact):**
1. **Export train** (`agitated_hugle`, `/app/Onnx4Deeploy`):
```bash
CKPT=/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
python3 Onnx4Deeploy.py -model SpeechNet -mode train \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3 \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights $CKPT --subject S01 --session 3 --batch 1 --condition vocalized \
  --data-size 54 --n-epochs 40 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --stratified --maxpool-argmax-mask
```
2. **Train on device + dump weights** (`traindeeploy`, `/app/ETH/TrainDeeploy/DeeployTest`; kill orphan gvsoc by explicit PID first, `rm -rf TEST_SIRACUSA`; ~4–5 h, background it):
```bash
python3 deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3 \
  --n-steps 540 --n-accum 4 --cores 8 \
  --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D DUMP_WEIGHTS=ON BN_FROZEN_STATS=ON \
  2>&1 | tee experiments/deliverable/exp4_BP_round1/logs/round1_gvsoc_train.log
```
3. **Extract → carry** (reuse `experiments/exp1/ondevice_sim_S01_fold3/extract_device_weights.py`):
```bash
python3 experiments/exp1/ondevice_sim_S01_fold3/extract_device_weights.py \
  --gvsoc-log experiments/deliverable/exp4_BP_round1/logs/round1_gvsoc_train.log \
  --base-ckpt $CKPT --out-carry /tmp/carry_fullfrozen_b1_fold3.pt
```
4. **Export infer (batch 2, carry baked)** (`agitated_hugle`):
```bash
python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b1_fold3 \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /tmp/carry_fullfrozen_b1_fold3.pt \
  --subject S01 --session 3 --batch 2 --condition vocalized
```
5. **Evaluate whole batch 2** (`traindeeploy`; balanced acc = mean per-class recall over 180 windows):
```bash
python3 experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
  --infer-dir Tests/Models/Training/SpeechNet/speechnet_infer_fullfrozen_b1_fold3 \
  2>&1 | tee experiments/deliverable/exp4_BP_round1/results/eval_b2.log
```
**Expected:** b2 balanced accuracy ≈ 87.2% (per `exp1/FINDING.md`). MaxPool argmax drift onset ≈ step 133 → per-step loss breaches after are expected, not failures; the WDUMP weights remain valid.

---

## exp5 — ZO round 1 (needs the weight-dump addition first, then the run)

### Part A — Add ZO weight-dump (mirror BP; ~50 lines, QW-tagged)
Reuse BP's proven `[WDUMP]` mechanism; the ZO updated weights live in the **same** persistent buffers as BP
(`DeeployNetwork_inputs[TRAINING_NUM_DATA_INPUTS + wi]`, wi 0–21), aliased from `zo_update`. **All dump code
must be FPU-free** (raw `uint32` hex, no float ops on the FPU-less Fabric Controller).
1. `DeeployTest/Platforms/Siracusa/src/deeploymezotest.c` — add `#ifdef DUMP_WEIGHTS dump_zo_weights(step)`
   (mirror `deeploytraintest.c:212–239`: per-tensor `[WDUMP s=.. wi=.. n=..] <hex>`), called once after the
   final update step, before the loss comparison. The CMake passthrough already exists
   (`Platforms/Siracusa/CMakeLists.txt` `MEZO_TRAINING`→`DUMP_WEIGHTS`).
2. MeZo runner path — ensure `-D DUMP_WEIGHTS=ON` reaches cmake for the MeZo build (verify
   `deeployMezoRunner_tiled_siracusa.py`/`testUtils/deeployMezoRunner.py` forward `-D`; add a `--dump-weights`
   passthrough if not, matching the BP runner).
3. `experiments/deliverable/exp5_ZO_round1/extract_zo_weights.py` — copy/adapt BP's
   `extract_device_weights.py` (identical 22-tensor `NAME_MAP` and `[WDUMP]` regex; overwrite base ckpt →
   carry). **Validate first** with a 1-step ZO run (dump vs the known bit-exact single-step weights) before the long run.

### Part B — Run (full recipe; ~21,600 GVSoC forwards, ~1 day — background it)
**FT settings (exp18 PyTorch recipe, validated):** all 22 params (10 conv + 10 BN γ/β + 2 fc), **frozen BN**,
**200 epochs**, **54** stratified windows (6/class, seed 42), **n_accum 4**, lr **3e-6**, ε **0.01** (baked in
ONNX), q **1**, seed **42**. Device update steps ≈ 200×⌈54/4⌉ (≈2700; runner auto-detects from fixture meta).
**Paths:** export dir `onnx/model/speechnet_zo_b1_fold3`; packed train/update dirs
`Tests/Models/Training/SpeechNet/speechnet_zo_train_b1_fold3` + `..._update_b1_fold3`; carry
`/tmp/carry_zo_b1_fold3.pt`; infer `Tests/Models/Training/SpeechNet/speechnet_zo_infer_b1_fold3` (batch 2).

**Commands:**
1. **Export ZO fixture** (`agitated_hugle`; ε=0.01 baked by exporter):
```bash
ln -sfn /app/SilentWear/SilentWear_data /app/SilentWear_data
EXPORT_BASE="--noise-type rademacher --bn-frozen-stats --dataset silentwear --pretrained-weights $CKPT --subject S01 --session 3 --condition vocalized --batch 1 --stratified"
python3 Onnx4Deeploy.py -model SpeechNet -mode zo-train $EXPORT_BASE \
  --n-epochs 200 --data-size 54 --n-accum 4 --lr 3e-6 -o ./onnx/model/speechnet_zo_b1_fold3
```
2. **Pack** (`traindeeploy`; train dir contains `_train`):
```bash
python3 experiments/zo_smoke/pack_2step_fixture.py \
  /app/ETH/Onnx4Deeploy/onnx/model/speechnet_zo_b1_fold3 \
  Tests/Models/Training/SpeechNet speechnet_zo_train_b1_fold3 speechnet_zo_update_b1_fold3
```
3. **Train on device + dump** (`traindeeploy`; ~1 day, background; BENCH cycle counter is uint32 → wraps, cosmetic):
```bash
python3 deeployMezoRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_zo_train_b1_fold3 \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_zo_update_b1_fold3 \
  --n-steps <auto/2700> --n-accum 4 --num-data-inputs 2 \
  --eps 0.01 --lr 3e-6 --q 1 --seed 42 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 --dump-weights \
  2>&1 | tee experiments/deliverable/exp5_ZO_round1/logs/round1_gvsoc_zo.log
```
4. **Extract → carry**:
```bash
python3 experiments/deliverable/exp5_ZO_round1/extract_zo_weights.py \
  --gvsoc-log experiments/deliverable/exp5_ZO_round1/logs/round1_gvsoc_zo.log \
  --base-ckpt $CKPT --out-carry /tmp/carry_zo_b1_fold3.pt
```
5. **Export infer (batch 2, ZO carry) + evaluate** (same eval harness as BP):
```bash
# agitated_hugle
python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_zo_infer_b1_fold3 \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /tmp/carry_zo_b1_fold3.pt --subject S01 --session 3 --batch 2 --condition vocalized
# traindeeploy
python3 experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
  --infer-dir Tests/Models/Training/SpeechNet/speechnet_zo_infer_b1_fold3 \
  2>&1 | tee experiments/deliverable/exp5_ZO_round1/results/eval_b2.log
```
**Expected:** b2 balanced accuracy ≈ 87.4% (exp18 ZO sim = 87.36; BP sim = 86.11; zero-shot = 80.56).

---

## Runtime & caveats
- BP round: ~4–5 h GVSoC (540 steps × 4 = 2160 fwd+bwd). ZO round: ~1 day (≈21,600 forwards). Run both in
  the background (detached `setsid`), poll logs; kill orphan gvsoc by **explicit PID** before each run.
- `BENCH train_cycles` is uint32 → wraps on the long ZO run (cosmetic; doesn't affect correctness or weights).
- ZO lr must be **3e-6** (lr≥1e-3 diverges to NaN); ε on the runner (`--eps 0.01`) must match the baked ε.
- Fixture naming: BP `-o` dir must contain `_train` (auto `_optimizer`); ZO passes `--optimizer-dir` explicitly.

## Verification
- **Same FT data (before either long run):** export both train fixtures, then byte-compare the `mbN_arr_0000`
  (input) + `mbN_arr_0001` (label) arrays across `speechnet_train_fullfrozen_b1_fold3/inputs.npz` and
  `speechnet_zo_train_b1_fold3/inputs.npz` — must be bit-identical (same 54 windows). Abort on mismatch.
- **exp5 dump correctness (before the long run):** run ZO for 1 step with `--dump-weights`, extract, and
  compare the 22 dumped tensors against the fixture's `outputs.npz` updated-param reference (should match to
  the single-step bit-exact tolerance). Only launch the 200-epoch run once this passes.
- **Both experiments:** the eval harness prints per-window `Errors: k/9` (bit-exactness vs ORT) and the final
  balanced accuracy; a successful round shows the carry checkpoint written and accuracy in the expected band.
- Save per-experiment: `logs/round1_gvsoc_*.log`, `results/eval_b2.log`, the carry `.pt`, and a short
  `FINDING.md` (device vs PyTorch-reference accuracy). Commit the deliverable to `feat/BP+ZO` when done.
