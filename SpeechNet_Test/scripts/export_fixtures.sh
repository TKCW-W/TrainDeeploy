#!/usr/bin/env bash
# Regenerate every SpeechNet fixture in TrainDeeploy/SpeechNet_Test from Onnx4Deeploy.
# Run from the HOST; needs the `agitated_hugle` container (ETH mounted at /app).
#
#   bash export_fixtures.sh <target> [<target> ...]
#   targets: bp_round1 bp_single zo_round1 zo_single qzo_single bp_infer zo_infer kernels all
#   (qzo_round1 / qzo_infer are kept below but NOT part of `all`: dropped 2026-10-12 because the QZO
#    round-1 host reference takes ~3 h and the round-1 QZO recipe will change with ECO.)
#
# Cell: subject S01, vocalized, session 3 (fold 3). Fine-tuning data = 54 stratified windows of
# batch 1 (30 %, seed 42, drawn by the exporter). Evaluation = all 180 windows of batch 2.
# Weights: artifacts_reference (paper-matching). Hyper-parameters are those of the correct-weight
# round-1 experiments deliverable/exp17 (BP), exp18 (ZO), exp19 (QZO).
set -euo pipefail

H=/Users/qiwenwu/ETH/TrainDeeploy/SpeechNet_Test          # host view
C=/app/TrainDeeploy/SpeechNet_Test                         # agitated_hugle view
DLV=/app/TrainDeeploy/DeeployTest/experiments/deliverable
CKPT=/app/TrainDeeploy/SpeechNet_Test/pretrained/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
DATA=/app/SilentWear/SilentWear_data/data_raw_and_filt
CELL="--dataset silentwear --data-path $DATA --subject S01 --session 3 --condition vocalized"
THR=$C/calibration/pooled_9999_fold3_ref.json              # = exp19 fixture/pooled_9999_fold3_ref.json
PACK=/app/TrainDeeploy/DeeployTest/experiments/zo_smoke/pack_2step_fixture.py
mkdir -p "$H/logs" "$H/exports"

o4d() {  # $1 = log name, rest = Onnx4Deeploy.py arguments
  local log=$1; shift
  docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && PYTHONDONTWRITEBYTECODE=1 python3 -u Onnx4Deeploy.py -model SpeechNet $*" \
    > "$H/logs/$log.log" 2>&1
}
o4d_qzo() {
  local log=$1; shift
  docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && QZO_POOLED_THRESHOLDS=$THR PYTHONDONTWRITEBYTECODE=1 python3 -u Onnx4Deeploy.py -model SpeechNet $*" \
    > "$H/logs/$log.log" 2>&1
}
pack() {  # $1 = export dir name, $2 = train dir, $3 = update dir
  docker exec agitated_hugle bash -lc "python3 $PACK $C/exports/$1 $C $2 $3" >> "$H/logs/pack.log" 2>&1
}

# ---- 1. BP round 1 (recipe S2: full model, frozen BN stats, argmax mask, SUM accumulation) ------
# writes speechnet_train_BP_round1/ and, by naming convention, speechnet_optimizer_BP_round1/
bp_round1() { o4d export_bp_round1 -mode train -o $C/speechnet_train_BP_round1 $CELL --batch 1 \
  --pretrained-weights $CKPT --data-size 54 --stratified --n-epochs 40 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --maxpool-argmax-mask; }

# ---- 2. BP single step (n_accum 1, one optimizer step) ------------------------------------------
bp_single() { o4d export_bp_single -mode train -o $C/speechnet_train_BP_single_step $CELL --batch 1 \
  --pretrained-weights $CKPT --data-size 54 --stratified --n-steps 1 --n-accum 1 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --maxpool-argmax-mask; }

# ---- 3. ZO round 1 (MeZO, 2700 steps x n_accum 4 = 21600 forwards, full reference) --------------
zo_round1() { o4d export_zo_round1 -mode zo-train --noise-type rademacher --bn-frozen-stats $CELL --batch 1 \
  --pretrained-weights $CKPT --data-size 54 --stratified --n-epochs 200 --n-accum 4 --lr 3e-6 \
  -o $C/exports/zo_round1
  pack zo_round1 speechnet_train_ZO_round1 speechnet_update_ZO_round1; }

# ---- 4. ZO single step ---------------------------------------------------------------------------
zo_single() { o4d export_zo_single -mode zo-train --noise-type rademacher --bn-frozen-stats $CELL --batch 1 \
  --pretrained-weights $CKPT --data-size 54 --stratified --n-steps 1 --n-accum 1 --lr 3e-6 \
  -o $C/exports/zo_single_step
  pack zo_single_step speechnet_train_ZO_single_step speechnet_update_ZO_single_step; }

# ---- 5. QZO round 1 -- NOT GENERATED (see header). Existing equivalent: DeeployTest/Tests/Models/
#         Training/SpeechNet/speechnet_qzo19_{train,update} (exp19). Kept for when it is needed again. ---
qzo_round1() { o4d_qzo export_qzo_round1 -mode q-zo-train --noise-type rqs_rademacher $CELL --batch 1 \
  --pretrained-weights $CKPT --data-size 54 --stratified --n-epochs 200 --n-accum 4 --lr 1e-5 \
  -o $C/exports/qzo_round1
  pack qzo_round1 speechnet_train_QZO_round1 speechnet_update_QZO_round1; }

# ---- 6. QZO single step --------------------------------------------------------------------------
qzo_single() { o4d_qzo export_qzo_single -mode q-zo-train --noise-type rqs_rademacher $CELL --batch 1 \
  --pretrained-weights $CKPT --data-size 54 --stratified --n-steps 1 --n-accum 1 --lr 1e-5 \
  -o $C/exports/qzo_single_step
  pack qzo_single_step speechnet_train_QZO_single_step speechnet_update_QZO_single_step; }

# ---- evaluation fixtures: batch 2, all 180 windows, with the ROUND-1 DEVICE-TRAINED weights -------
# The carry checkpoints are the device weight dumps of exp17 / exp18 / exp19. After a new device
# round, extract its [WDUMP] into a new carry checkpoint and re-run the same command with it.
bp_infer() { o4d export_bp_infer -mode infer -o $C/speechnet_infer_BP_round1 $CELL --batch 2 \
  --pretrained-weights $DLV/exp17_BP_round1_correct_weights/results/carry_fullfrozen_b1_fold3_ref.pt; }
zo_infer() { o4d export_zo_infer -mode infer -o $C/speechnet_infer_ZO_round1 $CELL --batch 2 \
  --pretrained-weights $DLV/exp18_ZO_round1_correct_weights/results/carry_zo_b1_fold3_ref.pt; }
# QZO: the int8 inference graph is rebuilt from the QZO export plus the dumped device weights
# (exp19 results/dumped_weights.npz, extracted by exp9_QZO_round1/extract_qzo_weights.py).
qzo_infer() { docker exec agitated_hugle bash -lc "cd $DLV/exp19_QZO_round1_correct_weights && python3 -u \
  ../exp12_QZO_clean_round_1/build_qzo_infer_fixture11.py --fixture-dir $C/exports/qzo_round1 \
  --dump-npz $DLV/exp19_QZO_round1_correct_weights/results/dumped_weights.npz \
  --out-dir $C/speechnet_infer_QZO_round1" > "$H/logs/build_qzo_infer.log" 2>&1; }

# ---- block-0 conv kernel fixtures: hand-built (no exporter path yet), copied from DeeployTest/Tests -
kernels() {
  local K=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/Tests/Kernels/Integer
  rm -rf "$H"/speechnet_{unfused,fused}_b0_{cluster,NE16}
  cp -R $K/QZO_Conv          "$H/speechnet_unfused_b0_cluster"   # build_qzo_conv_fixture.py    (exp3)
  cp -R $K/QZO_DualConv      "$H/speechnet_fused_b0_cluster"     # build_dual_conv_fixture.py   (exp5)
  cp -R $K/QZO_Conv_NE16     "$H/speechnet_unfused_b0_NE16"      # build_ne16_im2col_fixture.py (exp6 task 1)
  cp -R $K/QZO_DualConv_NE16 "$H/speechnet_fused_b0_NE16"        # build_ne16_dual_fixture.py   (exp6 task 2)
}

for t in "$@"; do
  case $t in
    all) for x in kernels bp_round1 bp_single zo_round1 zo_single qzo_single bp_infer zo_infer; do $x; done ;;
    *) $t ;;
  esac
done
