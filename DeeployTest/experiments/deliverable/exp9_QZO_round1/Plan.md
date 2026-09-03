# exp9 — On-device QZO round-1 fine-tuning at lr 1e-5 (Plan)

Date: 2026-09-03 · Branch `feat/QZO` (both repos) · Export in `agitated_hugle`, GVSoC in `traindeeploy`

## Goal

Run the lr-1e-5 direct-int8 quantized-ZO **round 1** on-device (Siracusa/GVSoC) and compare its
batch-2 accuracy against the **device-faithful PyTorch reference** (fc-float): **88.89%**.

## Setting (matches the PyTorch @1e-5 sim exactly)

- S01 / vocalized / **fold 3** (session 3 held out).
- **Fine-tune on batch 1** (54 stratified windows = 30% of 180, 6/class, **seed 42**), **eval on
  whole batch 2** (180 windows).
- **lr 1e-5**, **n_accum 4**, **n_epochs 200** → 2700 update steps, **ε 0.01**, direct int8
  (no master weights), z-seed 42.
- **Activation scales = pooled@99.99** (calibrated on 1800 pretraining windows, sessions 1+2) —
  BAKED into the fixture (not the default 9-window calibration). Weight scales = abs-max.
- fc head = **float** on device (`build_int8_forward`), hence the fc-float reference.

## Reference (PyTorch, `pytorch_ref/`)

`run_fc_float_ref.py` (uses the canonical stored pooled@99.99 thresholds):
- fc-int8 (sanity, reproduces run_incremental): 85.56 → **90.00%**
- **fc-float (device-faithful target): 85.00 → 88.89%**  ← compare device to this.

## Steps

1. **Bake pooled@99.99 into the export.** Inject the exp_calibration fold-3 pooled@99.99
   activation thresholds into the Brevitas model's act quantizers before the integer pipeline,
   replacing the default 9-window `calibration_mode` calibration in `base_exporter.py:706`.
2. **Generate the fixture** with `Onnx4Deeploy.py -mode <qzo-train>` (exp8 quantized flow), FT
   data = batch 1, 54 stratified seed-42 windows, ε 0.01 baked. Save under `fixture/`.
3. **Verify same 54 windows**: byte-compare the fixture's per-mini-batch input/label arrays
   against the PyTorch sim's `trX1`/`trY1`. Abort on mismatch.
4. **Single-step bit-exact check** (like exp7): run 1 ZO step on GVSoC, compare the dumped
   weights / loss against a host reference using the fixture's exact graph. Only proceed if exact.
5. **Full round-1 run**: 2700 steps on GVSoC (~long; background). Kill orphan gvsoc by explicit
   PID first. Save `logs/round1_gvsoc.log` (for the bit-exactness study).
6. **Extract device weights → carry ckpt → export infer (batch 2) → evaluate** balanced accuracy;
   compare to the 88.89% fc-float reference.

## Commands (reproduction — exact flags filled in after step 1)

```bash
# 1-2. export (agitated_hugle, /app/Onnx4Deeploy):
#   python3 Onnx4Deeploy.py -model SpeechNet -mode <qzo-train> \
#     --pretrained-weights <fold3 ckpt> --subject S01 --session 3 --batch 1 --condition vocalized \
#     --data-size 54 --stratified --n-epochs 200 --n-accum 4 --lr 1e-5 --noise-type rademacher \
#     --bn-frozen-stats  -o ./onnx/model/speechnet_qzo_b1_fold3_1e5   [+ bake-pooled flag]
# 3. pack (traindeeploy): experiments/zo_smoke/pack_2step_fixture.py ...
# 5. run (traindeeploy): deployMezoRunner_tiled_siracusa.py -t <train> --optimizer-dir <update> \
#     --n-steps 2700 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
#     --l1 128000 --l2 2000000 --cores 8 --dump-weights | tee logs/round1_gvsoc.log
```

## Files

```
exp9_QZO_round1/
  Plan.md, Findings.md
  pytorch_ref/  run_fc_float_ref.py, results.json, run.log   (the 88.89% fc-float reference)
  fixture/      exported ONNX + packed train/update dirs (pooled@99.99 baked, 54 seed-42 windows)
  logs/         round1_gvsoc.log (bit-exactness study), single_step_check.log
  results/      device carry ckpt, batch-2 eval log
```

## Notes / risks

- 2700 GVSoC steps is a long run (~many hours). Validate single-step bit-exact BEFORE the long run.
- fc is float on device → compare to 88.89% (fc-float), not 90.00% (fc-int8).
- Containers: export/reference in `agitated_hugle`, GVSoC in `traindeeploy`; kill orphan gvsoc by
  explicit PID before each run (killall doesn't reach gvsoc_launcher).
