# On-device (Siracusa/GVSoC) simulation of head-only incremental fine-tuning — reproducible plan

**Date:** 2026-07-23
**Goal:** Reproduce, on the Siracusa PULP accelerator under GVSoC simulation, the **head-only + BN-fold**
incremental fine-tuning we picked for deployment (see `../Pre_Deployment_Analysis.md`), for **S01
vocalized, fold 3**, across the full incremental chain **b1→b5**, and compare the on-device per-batch
balanced accuracy to the PyTorch simulation.

This document is written so someone WITHOUT Claude Code can reproduce it from our codebase: every step
names the file used, its purpose, and the exact command.

---

## 0. Environment

| container | repo (in container) | role |
|---|---|---|
| `agitated_hugle` | `/app/Onnx4Deeploy` (+ sees `/app/SilentWear`, `/app/TrainDeeploy`) | ONNX export + ORT reference + accuracy eval |
| `traindeeploy` | `/app/ETH/TrainDeeploy` | Deeploy codegen, C build, GVSoC simulation |

Reference paths (in containers):
```
PRETRAINED=/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt      # SilentWear EMG (session_3 batches 1..5)
SN=/app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet     # training fixtures
INFER=/app/ETH/TrainDeeploy/DeeployTest/Tests/Models                     # inference fixtures
```
NOTE the pretrained checkpoint dir is `inter_session_ft/` (matches the PyTorch experiments in
`../exp7_headonly_4subj/run_headonly.py`). An older on-device run used `inter_session/` — use
`inter_session_ft/` here so the on-device run matches the PyTorch baseline.

## Recipe (identical to the PyTorch head-only, `../exp7_headonly_4subj`)
head-only (train only `fc`, ~297 params), **BN folded into Conv** (automatic when
`--training-strategy last_layer`), SGD **lr 0.01, n_accum 4** (sum), **40 epochs**, **30 % data**
(54 windows, 6/class), incremental (fc carried forward across batches). Inference uses the folded
(pretrained-stats) conv — no BN kernels.

---

## Key fidelity fact (why this is a *verification*, not a new measurement)
On-device head-only training is **bit-exact to the host ORT/PyTorch reference**: the existing run
`experiments/headonly_ondevice_finetune/speechnet_head_ep40_ondevice.log` shows every
`[loss k] computed=… ref=… diff=0.000000`, and `extract_device_fc.py` reports
`max|device − ORT| fc_weight,fc_bias < 1e-4` (`VALID device == ORT within fp32`). On-device inference
is likewise bit-exact (`speechnet_infer_original` → 70.56 %, matching ORT). Therefore the on-device
per-batch accuracy **equals** the host accuracy **for the same data draw**; the simulation's purpose is
to confirm this holds across the full incremental chain.

**Comparison methodology (important):** compare each on-device number to the ORT reference computed from
the *same* Onnx4Deeploy export (identical 54-window draw) — that is the bit-exact check. The PyTorch
numbers in `pytorch_fold3_baseline.py` use an independent draw (`windowing.stratified_draw(seed=42)`),
so they agree only to ~1 pp (e.g. device b2_ft 89.44 vs PyTorch 90.56) — that gap is the **stratified
draw difference**, not device error (zero-shot b1/b2 match exactly: 80.56 / 81.67).

---

## Files used (per purpose)

| step | file | purpose |
|---|---|---|
| export | `/app/Onnx4Deeploy/Onnx4Deeploy.py` (`-mode train`) | build SpeechNet training graph (fwd+bwd+SGD) + ORT reference + `inputs/outputs.npz` |
| export (model) | `/app/Onnx4Deeploy/onnx4deeploy/models/speechnet_exporter.py` | SpeechNet model, `last_layer` strategy, BN-fold, trainable-param selection, ORT reference training |
| build+sim | `/app/ETH/TrainDeeploy/DeeployTest/deeployTrainingRunner_tiled_siracusa.py` | codegen → compile → GVSoC run of the training graph; prints `[loss k]` and (with weight dump) `[WDUMP]` |
| codegen | `.../DeeployTest/testUtils/codeGenerateTraining.py` | encodes the 54 windows into `.weightmem_sram` buffers; sets `N_TRAIN_STEPS`, `N_ACCUM_STEPS`, baked `lr` |
| device loop | `.../DeeployTest/Platforms/Siracusa/src/deeploytraintest.c` | on-device epoch loop (`N_TRAIN_STEPS`), gradient accumulation, SGD step, `[WDUMP]` |
| extract | `.../experiments/headonly_ondevice_ft_fixedwindow/extract_device_fc.py` | parse `[WDUMP]` hex → `device_fc_weight.npy`/`device_fc_bias.npy`; validate vs ORT |
| eval | `.../experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py` | 180 per-sample GVSoC inferences on an eval batch → balanced accuracy JSON |

---

## Procedure — incremental chain (b1→b5), head-only, fold 3

State carried between rounds: the fine-tuned **fc weight+bias** (conv/BN frozen & folded throughout).

### Round r (r = 1..4): fine-tune on batch r, evaluate on batch r+1

**(a) Export the training fixture for batch r**, initialised from the previous round's device fc
(round 1 initialises from the pretrained checkpoint):
```bash
docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py \
  -model SpeechNet -mode train \
  -o $SN/speechnet_train_b${r}_fold3 \
  --data-path $DATA --pretrained-weights $PRETRAINED \
  --subject S01 --condition vocalized \
  --data-size 54 --n-accum 4 --n-epochs 40 --lr 0.01 \
  --training-strategy last_layer --stratified"
# (batch selection + fc-init-from-previous-round handled via the exporter; see incremental driver below)
```
Outputs in `speechnet_train_b${r}_fold3/`: `network.onnx` (train graph), `network_infer.onnx` (eval
graph), `inputs.npz` (54 windows + init weights), `outputs.npz` (ORT reference: trained fc + per-step
losses); sibling `speechnet_train_b${r}_fold3_optimizer/network.onnx` (SGD op).

**(b) Build + simulate on GVSoC** (bit-exactness check against the ORT reference):
```bash
docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
  python deeployTrainingRunner_tiled_siracusa.py \
    -t Tests/Models/Training/SpeechNet/speechnet_train_b${r}_fold3 \
    --n-accum 4 --cores 8 --l1 128000 --l2 2000000 \
    --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max -v" \
  2>&1 | tee round${r}_train.log
# PASS = all [loss k] diff < 1e-3 (train graph bit-exact to ORT).
```

**(c) Extract the on-device fc weights** (from `[WDUMP]`) and validate vs ORT:
```bash
# adapt extract_device_fc.py's LOG/TR paths to round${r}_train.log and speechnet_train_b${r}_fold3
docker exec traindeeploy python3 .../extract_device_fc.py    # -> device_fc_weight.npy, device_fc_bias.npy
```

**(d) Evaluate on batch r+1**: inject device fc into the batch-(r+1) inference fixture and run the
per-sample GVSoC accuracy harness:
```bash
docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && \
  python experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
    --infer-dir Tests/Models/speechnet_infer_b$((r+1))_ft_fold3 --cores 8"
# -> Balanced accuracy on batch r+1 (the on-device FT accuracy for this round)
```
Also record the batch-(r+1) **zero-shot** accuracy (device fc = previous round's, i.e. no FT on batch
r+1 yet — that IS the carried state) for the incremental table.

**(e) Carry** `device_fc_weight/bias.npy` into round r+1's export as the fc initialiser.

### Incremental driver
`run_incremental_ondevice.sh` (in this folder) orchestrates rounds 1..4 end-to-end: export (with fc
carry) → GVSoC train → extract → GVSoC eval on the next batch → carry. Logs land in `results/`.

---

## Expected result & comparison
Because training and inference are bit-exact, the on-device per-batch balanced accuracy should match the
ORT reference from the same export, and land within ~1 pp of the PyTorch fold-3 head-only baseline
(`pytorch_fold3_headonly.csv`):

| batch | PyTorch fold-3 head-only (no-FT / FT) | on-device (to be filled) |
|---|---|---|
| 1 | 80.56 / 80.56 (zero-shot) | b1_zs = 80.56 ✓ (existing) |
| 2 | 81.67 / 90.56 | b2_zs 81.67 ✓, b2_ft 89.44 (existing; draw-diff) |
| 3 | 76.67 / 80.56 | — |
| 4 | 87.78 / 88.89 | — |
| 5 | 76.11 / 81.67 | — |

Success criterion: on-device b(r+1)_ft equals its export's ORT reference (bit-exact), and tracks the
PyTorch baseline within the stratified-draw tolerance (~1–2 pp).

## Cost note
GVSoC is cycle-accurate and slow: each 40-epoch head-only training round ≈ hours; each 180-sample
accuracy eval ≈ 1–2 h. The full b1→b5 chain (4 train rounds + 4 FT evals + zero-shot evals) is a
multi-hour campaign, run sequentially via `run_incremental_ondevice.sh`.
