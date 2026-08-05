# exp3 — ZO (MeZO) single step

Single-step zeroth-order (MeZO) fine-tuning of SpeechNet. Forward-only (two perturbed forward passes, no
backward) — so there is **no MaxPool backward at all**, and the large activation stash the BP path fights is
absent. Two-graph flow: `zo_train` (perturbed forward → loss) + `zo_update` (in-place θ update).

## Result
- **PASSED — bit-exact, 0 out of 2 errors.**
  - `[loss+ 0] computed=0.016488  ref=0.016488  diff=0.000000`
  - `[loss- 0] computed=0.069274  ref=0.069274  diff=0.000000`
  - `BENCH train_cycles=35643810 opt_cycles=144403`
- **Training-net L2 peak ≈ 789,688 B** — **−55% vs exp1 recompute** (forward-only; no gradient buffers / no X stash).
- Log: `logs/sim.log` · Memory plots: `results/memory_alloc_deeployStates.html` (zo_train) + `..._optimizer.html` (zo_update).

## 1. Export the ZO fixture — Onnx4Deeploy (`agitated_hugle`)
```bash
ln -sfn /app/SilentWear/SilentWear_data /app/SilentWear_data      # once
cd /app/Onnx4Deeploy
CKPT=/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
EXPORT_BASE="--noise-type rademacher --bn-frozen-stats --dataset silentwear --pretrained-weights $CKPT --subject S01 --session 3 --condition vocalized"
python3 Onnx4Deeploy.py -model SpeechNet -mode zo-train $EXPORT_BASE \
  --n-steps 1 --n-accum 1 --lr 0.001 -o ./onnx/model/speechnet_zo_deliv
```

## 2. Pack into the two-dir layout — TrainDeeploy (`traindeeploy`)
Train dir name contains `_train`; update dir is passed explicitly as `--optimizer-dir`.
```bash
cd /app/ETH/TrainDeeploy/DeeployTest
python3 experiments/zo_smoke/pack_2step_fixture.py \
  /app/ETH/Onnx4Deeploy/onnx/model/speechnet_zo_deliv \
  Tests/Models/Training/SpeechNet speechnet_zo_train_deliv speechnet_zo_update_deliv
```

## 3. Single-step MeZO sim + memory plot — TrainDeeploy (`traindeeploy`)
```bash
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 deeployMezoRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_zo_train_deliv \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_zo_update_deliv \
  --n-steps 1 --n-accum 1 --num-data-inputs 2 \
  --eps 0.01 --lr 0.001 --q 1 --seed 42 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 --plotMemAlloc
```
