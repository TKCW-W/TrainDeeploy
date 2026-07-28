# On-device simulation — full-training + frozen-BN, S01 vocalized fold 3 (session 3)

**Started:** 2026-07-28
**Branch:** TrainDeeploy `feat/speechnet/on-device-FT` · Onnx4Deeploy `feat/speechnet/inference/maxpool_ondevice`
**Goal:** run the complete on-device (Siracusa/GVSoC) incremental fine-tuning chain for the deployment recipe
**full-model FT + BatchNorm frozen at pretrained running stats** (S2), collect the **on-device accuracy**
per batch (b1→b5), and compare to the **PyTorch** simulation on the **same data**. Save all train-runner
logs for later numerical-drift analysis (MaxPool argmax drift — see `../FINDING.md` caveat).

This uses the Option A frozen-BN reference fix (`../FINDING.md`) so the train runner's loss check is
against a *frozen* reference. NOTE: over a full 540-update run the **MaxPool argmax drift** will breach the
tight loss tolerance (expected, not a training error) — acceptance is by **accuracy**, not per-step loss.

---

## Recipe (S2 — from `SilentWear/.../frozen_stat_full_training`, 4-subj repro `exp15_edgeft_official`)
| knob | value |
|---|---|
| scope | full model (conv + BN γ/β + fc), **no BN folding** |
| BN | **frozen pretrained stats**, train ≡ inference (`--bn-frozen-stats`; device `BN_FROZEN_STATS=ON`) |
| optimizer | SGD, no momentum/wd |
| lr | 3e-4 static |
| n_accum | 4, SUM ; effective batch 1 |
| epochs | 40 ; FT data 54 windows (30%, 6/class, seed 42) |
| protocol | incremental b1→b5: round r trains on batch r, evaluates batch r+1; weights carried forward |

## Relevant file paths
- **Data source:** `/app/SilentWear/SilentWear_data/data_raw_and_filt` (dataset `silentwear`), via
  `Onnx4Deeploy/onnx4deeploy/data/silent_wear_datasource.py` (subject S01, session 3, condition vocalized).
- **Pretrained weights (round-1 base):**
  `/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt`
  (dict with `model_state_dict`).
- **Fixtures dir:** `/app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b{r}_fold3`
  (train) and `.../speechnet_infer_fullfrozen_b{r}_fold3` (inference eval).
- **Carry checkpoints:** `/tmp/carry_fullfrozen_b{r}_fold3.pt` (device weights after round r).
- **This experiment:** `DeeployTest/experiments/exp1/ondevice_sim_S01_fold3/` — `PLAN.md`, `FINDING.md`,
  `logs/round{r}_gvsoc_train.log`, `results/*.csv`, extract/prep/compare scripts.

## Flow (per round r = 1..4; mirrors `ondevice_simulation_lastblock`/`_headonly`)

### (a) Export the TRAIN fixture — Onnx4Deeploy.py (graph + npz)
```
python3 Onnx4Deeploy.py -model SpeechNet -mode train \
  -o .../SpeechNet/speechnet_train_fullfrozen_b{r}_fold3 \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights <round-1: official ckpt | round>1: /tmp/carry_fullfrozen_b{r-1}_fold3.pt> \
  --subject S01 --session 3 --batch {r} --condition vocalized \
  --data-size 54 --n-epochs 40 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --stratified
```
Produces `network.onnx` / `network_train.onnx` (training graph) / `inputs.npz` (init weights + 54 windows)
/ `outputs.npz` (per-step **frozen** reference losses [Option A] + updated params).

### (b) TRAIN on device — tiled Siracusa runner (dump weights)
```
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
python deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b{r}_fold3 \
  --n-steps 540 --n-accum 4 --cores 8 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D DUMP_WEIGHTS=ON BN_FROZEN_STATS=ON
```
**Save the full log** → `logs/round{r}_gvsoc_train.log` (needed for the MaxPool-drift analysis: breach
rate, onset step, max |diff|). Expect loss breaches after the drift onset (~step 36 updates) — this is the
inherent MaxPool argmax drift, NOT a failure; the `[WDUMP]` weights are still produced and valid.

### (c) Extract device weights → carry checkpoint
Parse `[WDUMP s=<step> wi=<i> n=<n>] <hex...>` (last step) from the train log; each of the 32 dumped
tensors is value-matched to its ORT reference name (`outputs.npz`), reshaped, and written over the
official checkpoint's trainable tensors (BN running_mean/var stay frozen). → `/tmp/carry_fullfrozen_b{r}_fold3.pt`.
Script: `extract_device_weights.py` (this dir).

### (d) Prepare the INFERENCE npz — Onnx4Deeploy.py (updated weights + eval batch)
The inference runner needs an inference graph whose initializers are the **updated device weights**, plus
the eval batch (r+1) windows and their ORT-reference logits. We regenerate the inference fixture with the
carry checkpoint as the pretrained weights:
```
python3 Onnx4Deeploy.py -model SpeechNet -mode infer \
  -o .../SpeechNet/speechnet_infer_fullfrozen_b{r}_fold3 \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /tmp/carry_fullfrozen_b{r}_fold3.pt \
  --subject S01 --session 3 --batch {r+1} --condition vocalized
```
This bakes the device-trained weights into `network.onnx` (frozen BN → inference form) and emits
`inputs.npz` (the 180 eval windows of batch r+1) + `outputs.npz` (ORT reference logits/predictions). So the
"new npz" for evaluation = the inference fixture regenerated from the carry checkpoint; the updated weights
enter as ONNX initializers (baked in), not as runtime inputs.

### (e) EVALUATE on device — inference runner → balanced accuracy
Run the (untiled) Siracusa inference accuracy harness over the 180 windows of batch r+1:
```
python experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
  --infer-dir Tests/Models/.../speechnet_infer_fullfrozen_b{r}_fold3
# -> balanced accuracy on batch r+1 (the on-device FT accuracy for round r)
```
Cross-check: host PyTorch inference with the carry checkpoint on the SAME 180 windows is bit-exact to ORT
(established in the head-only/K=1 studies), so it doubles as a fast equivalence check of the device eval.

### b1 zero-shot
Eval batch 1 with the **official** checkpoint (no FT) via the same inference path → the chain's start point.

## PyTorch reference on the SAME data (matched comparison)
`run_pytorch_fullfrozen_ondevicedata_chain.py` (this dir): load the official ckpt, and for each round
extract the exact 54 FT windows from `speechnet_train_fullfrozen_b{r}_fold3/inputs.npz`, run the S2 recipe
in PyTorch (`model.eval()` frozen BN, full model, SGD lr3e-4, n_accum4 SUM, 40 ep), carry forward, and
evaluate batch r+1 on the same eval windows. This removes the draw confound so on-device vs PyTorch is
apples-to-apples. Reference S2 numbers: 4-subj b2–5 = 77.05; S01 b2–5 = 85.32 (host, own draw).

## Deliverables
- `results/ondevice_vs_pytorch_S01_fold3.csv` — per batch: on-device acc, PyTorch (matched) acc.
- `logs/round{1..4}_gvsoc_train.log` — kept for drift analysis.
- `FINDING.md` — dated results + drift observations + comparison.

## Runtime note
Full-model training is ~2160 fwd+bwd passes/round on GVSoC (heavier backward than head-only/K=1) → each
round is a multi-hour sim; 4 rounds + evals span a long wall-clock. Runs are backgrounded; logs saved.
