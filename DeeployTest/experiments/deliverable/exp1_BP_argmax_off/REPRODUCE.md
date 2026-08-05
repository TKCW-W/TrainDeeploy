# exp1 — BP, MaxPool argmax-mask **OFF** (recompute-from-X)

Single-step on-device BP fine-tuning of SpeechNet (frozen-BN recipe), MaxPool backward via the **shipped
recompute-from-X kernel** (`PULP_MaxPoolGrad2d`). The exporter's default rewrite feeds `MaxPoolGrad(dY, X)`
(no `--maxpool-argmax-mask` flag). Fresh fixture from Onnx4Deeploy, whole flow.

## Result
- **PASSED — 0 errors** (`N_TRAIN_STEPS=1 N_ACCUM_STEPS=4 DATA_INPUTS=2`).
- **Training-net L2 peak ≈ 1,735,180 B** (keeps the block-0 forward activation X live to backward).
- Graph: `MaxPool ×3` (single-output) + `MaxPoolGrad ×3`.
- Log: `logs/sim.log` · Memory plot: `results/memory_alloc_deeployStates.html` (+ `_optimizer`).

## 1. Export the fixture — Onnx4Deeploy (container `agitated_hugle`)
The `-o` dir name **contains `_train`** so the sibling `_optimizer` dir (SGD graph) is auto-created.
```bash
cd /app/Onnx4Deeploy
CKPT=/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
python3 Onnx4Deeploy.py -model SpeechNet -mode train \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/deliverable_bp_train_recompute \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights $CKPT --subject S01 --session 3 --batch 1 --condition vocalized \
  --data-size 18 --n-epochs 1 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --stratified
# (no --maxpool-argmax-mask  ->  recompute path)
```

## 2. Single-step sim + memory plot — TrainDeeploy (container `traindeeploy`)
```bash
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/deliverable_bp_train_recompute \
  --n-steps 1 --n-accum 4 --cores 8 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D BN_FROZEN_STATS=ON --plotMemAlloc
```
`--optimizer-dir` is auto-derived (`_train`→`_optimizer`). Memory html lands under
`TEST_SIRACUSA/Tests/Models/Training/SpeechNet/deliverable_bp_train_recompute/deeployStates/memory_alloc.html`.
