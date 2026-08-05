# exp2 — BP, MaxPool argmax-mask **ON**

Same single-step BP flow as exp1, but MaxPool backward uses our **argmax-mask** optimization
(`--maxpool-argmax-mask`): the forward emits a small within-window offset mask (`MaxPoolArgmax`) and the
backward reads it (`MaxPoolGradMask`), so the large block-0 activation X is freed right after the forward
pass. Works **directly on top of the shipped single-output MaxPool** — it does not depend on the recompute
rewrite (it is the *alternative* branch of the same exporter step).

## Result
- **PASSED — 0 errors** (`N_TRAIN_STEPS=1 N_ACCUM_STEPS=4 DATA_INPUTS=2`).
- **Training-net L2 peak ≈ 1,511,308 B** — **−13% vs exp1 recompute** (X no longer held to backward).
- Graph: `MaxPool ×3` (single-output) + `MaxPoolArgmax ×3` + `MaxPoolGradMask ×3` (no `MaxPoolGrad`).
- Log: `logs/sim.log` · Memory plot: `results/memory_alloc_deeployStates.html` (+ `_optimizer`).

## 1. Export the fixture — Onnx4Deeploy (`agitated_hugle`)
```bash
cd /app/Onnx4Deeploy
CKPT=/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
python3 Onnx4Deeploy.py -model SpeechNet -mode train \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/deliverable_bp_train_argmaxmask \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights $CKPT --subject S01 --session 3 --batch 1 --condition vocalized \
  --data-size 18 --n-epochs 1 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --stratified \
  --maxpool-argmax-mask
```

## 2. Single-step sim + memory plot — TrainDeeploy (`traindeeploy`)
```bash
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/deliverable_bp_train_argmaxmask \
  --n-steps 1 --n-accum 4 --cores 8 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D BN_FROZEN_STATS=ON --plotMemAlloc
```
