# SpeechNet on-device head-only fine-tuning — re-run with fixed (onset) windowing

**Status:** ✅ COMPLETE
**Started / finished:** 2026-07-16

## Why this re-run

The SilentWear windowing in `Onnx4Deeploy/onnx4deeploy/data/silent_wear_datasource.py`
was extracting each window from the **center** of a label run; the paper
(`SilentWear/WINDOWING_SPEC.md`) anchors **one window per utterance at the run onset**
and cuts a fixed 700-sample window (dropping only windows that overrun the recording end).
Fixed in Onnx4Deeploy commit `77e269d`. On S01 sess3 batch2 the fix moves the reproduced
zero-shot from 68.33% (center) to **81.67%** (onset), exactly matching the paper's
`balanced_acc_no_ft`.

This re-run redoes the on-device head-only + BN-fold fine-tuning end-to-end with:
- the **corrected onset windowing** (now the loader default), and
- the **`inter_session_ft`** pretrained base model (the one the paper's `ft_summary.csv`
  scores), *not* the `inter_session` model used in the original
  `headonly_ondevice_finetune/` report.

Base checkpoint:
`SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt`

## Configuration (unchanged from the shipped recipe)

Head-only (last-layer) training, BatchNorm folded into the frozen Conv feature extractor.
`S01 / vocalized / session 3`. Fine-tune on **30 % of batch 1** (54 stratified windows,
6/class, seed 42), evaluate on **whole batch 2** (180 windows, 20/class).

| knob | value |
|---|---|
| training-strategy | `last_layer` (auto BN-fold) |
| data-size | 54 (30 % of batch 1) |
| lr | 0.01 |
| n_accum | 4 |
| n_epochs | 40 |
| optimizer | SGD (sum accumulator, eff-batch 1) |

## Host reference numbers (fixed windowing + inter_session_ft) — targets for the device

| quantity | host / ORT balanced acc |
|---|---|
| batch 1 zero-shot | **80.56 %** |
| batch 2 zero-shot | **81.67 %** (= paper `balanced_acc_no_ft`) |
| batch 2 after head-only FT | **89.44 %** (ORT reference = faithful device predictor; **+7.78 pp**) |

The ORT reference is the trustworthy predictor of the on-device result for the folded
head-only path (the original report verified device≡ORT bit-exact, 0 loss errors). The
on-device runs below confirm it end-to-end.

## Deliverables (user request)

1. On-device **zero-shot accuracy of batch 1** — does it match the PyTorch host (80.56 %)?
2. On-device **head-only + BN-fold fine-tune** (config above) on batch 1.
3. **Dump** the optimized `fc`, **regenerate the infer graph**, test **batch-2 accuracy**
   on-device; compare to batch-2 zero-shot (81.67 %) and the paper.

## Fixtures / file naming (suffix `_isft_fw` = inter_session_ft + fixed window)

| fixture | path under `DeeployTest/Tests/Models/` |
|---|---|
| training graph (head, ep40) | `Training/SpeechNet/speechnet_train_head_ep40_isft_fw` |
| batch-1 eval windows/infer | `speechnet_infer_b1_isft_fw` |
| batch-2 eval windows/infer | `speechnet_infer_b2_isft_fw` |
| batch-1 zero-shot infer fixture | `speechnet_infer_b1_zs_isft_fw` |
| batch-2 zero-shot infer fixture | `speechnet_infer_b2_zs_isft_fw` |
| batch-2 fine-tuned infer fixture | `speechnet_infer_b2_ft_isft_fw` |

## Progress log

- 2026-07-16: windowing fix committed+pushed (Onnx4Deeploy `77e269d`).
- 2026-07-16: host references computed (b1 zs 80.56 %, b2 zs 81.67 %).
- 2026-07-16: fixtures generated — training graph `speechnet_train_head_ep40_isft_fw`
  (n_batches 2160 / n_steps 540 / n_accum 4 / lr 0.01), eval windows
  `speechnet_infer_b{1,2}_isft_fw` (320 windows each; exporter host balanced acc 80.56 % / 81.67 %).
- 2026-07-16: ORT-predicted FT target = **89.44 %** on batch 2 (+7.78 pp vs zero-shot).
- 2026-07-16: folded zero-shot infer fixtures assembled
  `speechnet_infer_b{1,2}_zs_isft_fw` (ORT balanced acc 80.56 % / 81.67 %, matches host).
- 2026-07-16: on-device training launched (GVSoC, 8 cores, DUMP_WEIGHTS=ON); compiled OK,
  running 2160 forward+backward passes.
- 2026-07-16: **on-device training COMPLETE** — `update 540/540`, `Errors: 0 out of 2160`
  (on-device losses **bit-exact to the ORT reference** over all 2160 passes → folded
  head-only path is drift-free), `train_cycles=3.79e9`, test PASSED.
- 2026-07-16: device `fc` extracted from `[WDUMP s=539]` (wi=0 → fc_weight 9×32,
  wi=1 → fc_bias 9) → `device_fc_{weight,bias}.npy`. Validated vs ORT reference:
  **max|device − ORT| = 4.5e-7 (weight) / 1.8e-7 (bias)** = fp32 rounding → device head is
  bit-exact to ORT. (`extract_device_fc.py`)
- 2026-07-16: fine-tuned batch-2 fixture `speechnet_infer_b2_ft_isft_fw` assembled with the
  device `fc` (folded frozen conv + device-trained head) — ORT balanced acc 89.44 %.
- 2026-07-16: on-device inference evals launched (b1_zs → b2_zs → b2_ft; 180 per-sample
  GVSoC runs each via `speechnet_accuracy_eval_untiled.py`). `run_ondevice_evals.sh`.
- 2026-07-16: **all on-device evals COMPLETE** — b1 zero-shot 80.56 %, b2 zero-shot 81.67 %,
  b2 fine-tuned **89.44 %** (all `sim_errors=0`). Experiment done.

## Artifacts

- `SPEECHNET_ONDEVICE_FT_FIXEDWINDOW.md` — this report.
- `gen_train_head.log`, `run_train_isft_fw.log` — fixture-gen / on-device training logs.
- `extract_device_fc.py` (+ `device_fc_weight.npy`, `device_fc_bias.npy`) — WDUMP → device `fc`.
- `assemble_infer_fixtures.py` — builds the folded zero-shot / fine-tuned infer fixtures.
- `ort_ft_eval.py` — ORT reference for zero-shot / fine-tuned batch-2 accuracy.
- `gain_decomposition.py` — base-model × windowing gain decomposition + draw variance.
- `run_ondevice_evals.sh`, `eval_b{1_zs,2_zs,2_ft}.log`, `results_b*.json` — on-device eval driver + logs.
- Fixtures (under `Tests/Models/`): `Training/SpeechNet/speechnet_train_head_ep40_isft_fw`,
  `speechnet_infer_b{1,2}_isft_fw`, `speechnet_infer_b{1_zs,2_zs,2_ft}_isft_fw`.

### Reproduction commands (for the record)

```bash
# fixtures (agitated_hugle / Onnx4Deeploy)
python3 Onnx4Deeploy.py -model SpeechNet -mode train \
  -o Tests/Models/Training/SpeechNet/speechnet_train_head_ep40_isft_fw \
  --dataset silentwear --data-path .../data_raw_and_filt \
  --pretrained-weights .../inter_session_ft/.../leave_one_session_out_fold_3.pt \
  --subject S01 --session 3 --batch 1 --condition vocalized \
  --stratified --data-size 54 --n-epochs 40 --n-accum 4 --lr 0.01 --training-strategy last_layer
python3 Onnx4Deeploy.py -model SpeechNet -mode infer -o .../speechnet_infer_b{1,2}_isft_fw \
  --subject S01 --session 3 --batch {1,2} --condition vocalized  (+ same data/weights)

# on-device training + weight dump (traindeeploy / DeeployTest)
rm -rf TEST_SIRACUSA && python deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_head_ep40_isft_fw \
  --n-steps 540 --n-accum 4 --cores 8 -D DUMP_WEIGHTS=ON

# on-device inference accuracy (per fixture)
python experiments/headonly_ondevice_finetune/speechnet_accuracy_eval_untiled.py \
  --infer-dir Tests/Models/speechnet_infer_b{1_zs,2_zs,2_ft}_isft_fw --cores 8
```

## Why the FT gain differs from the old report (+4.44 pp → +7.78 pp)

The original `headonly_ondevice_finetune/` report measured **+4.44 pp** (78.33 → 82.78) on
the `inter_session` base with **center** windowing. This re-run predicts **+7.78 pp**
(81.67 → 89.44, ORT) on the `inter_session_ft` base with **onset** windowing. Two variables
changed, not one. Host decomposition (`gain_decomposition.py`; head-only + BN-fold sim,
eval-mode BN = folded-equivalent; zero-shots exact, FT endpoints ±~1–2 pp vs the exporter
fixture because the 54-window draw/order isn't bit-identical):

| base checkpoint | windowing | b2 zero-shot | b2 after head-FT | gain |
|---|---|---|---|---|
| `inter_session` | center *(old report)* | 78.33% | 81.67% | +3.33 pp |
| `inter_session` | onset | 91.67% | 88.33% | −3.33 pp |
| `inter_session_ft` | center | 68.33% | 83.89% | +15.56 pp |
| `inter_session_ft` | onset *(this re-run)* | 81.67% | 90.56% | +8.89 pp |

**10-draw variance, new config (onset + `inter_session_ft`): gain = mean +5.44 pp, std 1.75,
range +1.11 … +7.22.** The seed-42 fixture draw (ORT +7.78) sits at the top of the band.

Takeaways:
- The **base-checkpoint swap** is the dominant lever — `inter_session_ft` fine-tunes up to
  ~89–90 % on batch 2, a higher-accuracy regime than the old `inter_session` base.
- Onset windowing **raises absolute accuracy** (83.89 → 90.56 for `inter_session_ft`) but
  **compresses the headline gap** (it lifts the zero-shot floor 68.33 → 81.67 faster than the
  ceiling), so the gain is +8.89 not +15.56. The FT gain is a difference of two independent
  evaluations — no conservation law keeps it fixed when the windows are re-cut.
- The gain is a **single noisy draw** (±1.75 pp); the *expected* gain for the shipped config
  is ≈ **+5.4 pp**, with seed-42 landing optimistically high.

## Results (on-device GVSoC)

_(to fill)_

| infer graph (on-device GVSoC) | balanced acc | overall | host/ORT | match |
|---|---|---|---|---|
| batch-1 zero-shot | **80.56 %** | 145/180 | 80.56 % | ✅ exact |
| batch-2 zero-shot | **81.67 %** | 147/180 | 81.67 % | ✅ exact |
| batch-2 **fine-tuned** | **89.44 %** | 161/180 | 89.44 % | ✅ exact |

All 540 per-sample GVSoC inferences ran with `sim_errors=0` (device forward bit-exact to ORT).

### Deliverables — all confirmed on-device

1. **✅ On-device batch-1 zero-shot = 80.56 %**, bit-exact to the PyTorch host (80.56 %).
2. **✅ On-device head-only + BN-fold fine-tune** (30 % batch 1, lr 0.01, n_accum 4, 40 epochs)
   completed with `Errors: 0 out of 2160` — training losses bit-exact to ORT, device-trained
   `fc` bit-exact to the ORT reference (max\|Δ\| 4.5e-7).
3. **✅ Device `fc` dumped → infer graph regenerated → on-device batch-2 = 89.44 %**
   (**+7.78 pp** over the 81.67 % batch-2 zero-shot), overall 161/180.

### Comparison to the paper (SilentWear `ft_summary.csv`, S01 vocalized fold_3, batch 2)

| quantity | paper | ours (on-device) |
|---|---|---|
| base model, no fine-tune (`balanced_acc_no_ft`) | 81.67 % | **81.67 %** (exact match) |
| after fine-tuning on prior batch, eval batch 2 (`zero_shot_balanced_acc`) | 87.78 % | **89.44 %** |

- Our on-device **zero-shot reproduces the paper's `balanced_acc_no_ft` exactly (81.67 %)** —
  the windowing fix closed the earlier 68.3 % → 81.67 % gap.
- Our **on-device head-only + BN-fold fine-tune (89.44 %)** lands **+1.66 pp above the paper's
  full fine-tune (87.78 %)** on this batch. Caveats: the paper fine-tunes the **whole model**
  (Adam, batch 32) off-device, whereas ours is the **hardware-constrained head-only SGD**
  (eff-batch-1, BN folded) run on Siracusa; and both are single 54-window draws (the shipped
  seed-42 draw is optimistic — expected head-only gain ≈ +5.4 pp, see the decomposition above).
  Net: the on-device head-only recipe is competitive with the paper's unconstrained full FT
  on this transition.
