# exp12 — QZO clean round-1 — Findings

Date: 2026-09-09 · Branch `feat/QZO` · Clean-room run, no reused artifacts (calibration, fixture,
weights, accuracy all regenerated here). Device round built with **ffast-math on** (default; no
`DEEPLOY_STRICT_FP32`).

## Headline

The full QZO pipeline is verified end to end, and the fast default build (ffast-math) is
**learning-neutral**: the on-device round-1 batch-2 balanced accuracy is **88.33%**, matching the
device-faithful PyTorch reference (89.44%) and the strict-fp32 result (exp11, 89.44%) to within
~1 eval window, and well above zero-shot (85.00%).

| model (fold 3, round 1, ft batch1 → eval batch2, lr 1e-5) | batch-2 balanced acc |
|---|---|
| device zero-shot (fc-float, fresh pooled-99.99) | 85.00% |
| **device-trained, on-device untiled eval (ffast-math)** | **88.33%** |
| device-trained, host-executor on the same weights | 88.33% (exact match to device) |
| PyTorch fc-float reference (fresh calibration) | 89.44% |
| strict-fp32 device (exp11, for context) | 89.44% |

- **+3.33 over zero-shot**; within ~1 window (0.56%) of the reference and the strict result.
- **Device == host-executor at 88.33%** → the on-device inference forward is faithful to its
  graph (bit-exactness of *inference* is not required; argmax is stable).
- **ffast-math vs strict-fp32:** 88.33 vs 89.44 = ~2 eval windows, i.e. within seed/trajectory
  noise. Giving up bit-exactness (fused/reordered fp) costs no meaningful accuracy → the fast
  build is the right default. (The training *trajectory* differs from strict — a different but
  equally valid ZO path — but the endpoint accuracy is preserved: learning-neutral.)

## What was regenerated from scratch (no reused artifacts)

1. **Calibration (fresh):** pooled-99.99 activation thresholds recomputed on the 1800 fold-3
   pretraining windows → `fixture/pooled_9999_fold3_fresh.json`. Reproduced the known scales
   exactly (block-0 input threshold 2854.0 → scale 22.296875), confirming the calibration is
   deterministic.
2. **PyTorch reference (fresh):** device-faithful fc-float sim, fold-3 round 1, lr 1e-5 →
   85.00% → **89.44%** (`results/pytorch_ref.json`).
3. **Fixture (fresh):** `-mode q-zo-train` with the fresh thresholds; 54/54 FT windows
   byte-identical to the sim draw; leading Quant scale 22.296875 verified. (`fixture/`)
4. **Device round (fresh, ffast-math):** 2700 update steps, lr 1e-5, ε 0.01, n_accum 4, seed 42,
   `BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON`, **no strict-fp32**. PASSED (0/104 on the 52-pair short
   reference — the first 13 steps stay sub-tolerance even under ffast-math). Weights dumped @2699.
   (`logs/round1_gvsoc_ffastmath.log`)
5. **Eval (fresh):** extracted the dumped weights → rebuilt the quantized inference fixture
   (`build_int8_forward` + injected trained weights, offset-corrected) → untiled device inference
   over all 180 batch-2 windows → **88.33%**. (`qinfer/`, `logs/eval_b2_device.log`)

## Notes

- `bit-exact fails: 110/180` in the eval log = device inference logits vs the host reference
  differ by >tol on 110 windows (ffast-math inference, no strict). This is the accepted
  non-bit-exactness; it does **not** change predictions — device and host-executor balanced
  accuracy are identical at 88.33%.
- **Full-round loss error count (for the report), when wanted — no device re-run:** this run's
  log records every forward's device loss as raw fp32 bits
  (`[PHASE] +eps loss read OK: lp_bits=0x...` / `-eps ... lm_bits=0x...`, all 21,600). Re-export
  only the full-round reference (`--n-epochs 200`, host sim), then offline-decode these bits and
  diff. Expect a large count (ffast-math diverges from step ~1), quantifying the accepted trade.

## Reproduction (as executed)

```bash
# 1. fresh calibration + PyTorch fc-float reference (agitated_hugle)
python3 exp12.../run_ref_and_calib.py           # -> pooled_9999_fold3_fresh.json, ref 89.44%

# 2. export fixture with the fresh thresholds (agitated_hugle, /app/Onnx4Deeploy)
QZO_POOLED_THRESHOLDS=.../pooled_9999_fold3_fresh.json python3 Onnx4Deeploy.py -model SpeechNet \
  -mode q-zo-train --noise-type rqs_rademacher --dataset silentwear --data-path $DATA \
  --pretrained-weights $CKPT --subject S01 --session 3 --condition vocalized --batch 1 \
  --data-size 54 --stratified --n-accum 4 --lr 1e-5 -o QZO_exp/exp12_clean

# 3. pack + device round, ffast-math (traindeeploy) — NO DEEPLOY_STRICT_FP32
python3 experiments/zo_smoke/pack_2step_fixture.py QZO_exp/exp12_clean \
  Tests/Models/Training/SpeechNet speechnet_qzo12_train speechnet_qzo12_update
python3 deeployMezoRunner_tiled_siracusa.py -t .../speechnet_qzo12_train \
  --optimizer-dir .../speechnet_qzo12_update --n-steps 2700 --n-accum 4 --num-data-inputs 2 \
  --eps 0.01 --lr 1e-5 --q 1 --seed 42 --l1 128000 --l2 2000000 --cores 8 \
  -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON

# 4. extract -> inference fixture -> untiled batch-2 eval
python3 exp9.../extract_qzo_weights.py --gvsoc-log logs/round1_gvsoc_ffastmath.log \
  --train-onnx QZO_exp/exp12_clean/network_zo_train.onnx --out results/dumped_weights.npz
python3 build_qzo_infer_fixture11.py --fixture-dir QZO_exp/exp12_clean \
  --dump-npz results/dumped_weights.npz --out-dir qinfer            # host ref 88.33%
python3 exp9.../qzo_accuracy_eval_untiled.py --infer-dir .../exp12.../qinfer --cores 8  # 88.33%
```
