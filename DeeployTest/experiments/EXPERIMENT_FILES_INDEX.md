# Experiments — file index

All SpeechNet on-device fine-tuning experiment files, organized by experiment. Each subfolder holds
that experiment's scripts, docs, logs, data, and figures. (Deeploy infrastructure — `testMVP*.py`,
`generate*.py`, `benchmark_training.py`, the training runners, and the fixtures under
`Tests/Models/…` — is left in place.)

## `maxpool_numerical_drift/` — the argmax tie-flip / precision-drift investigation
Device-vs-ORT loss drift traced to MaxPool argmax tie-flips (fp non-associativity).
- `SPEECHNET_MAXPOOL_DRIFT_ANALYSIS.md`, `..._DRIFT_EXPLANATION.md`, `..._PROGRESS.md`, `..._COMMITLOG.md` — analysis/writeups.
- `SPEECHNET_FINETUNE_PRECISION_FINDINGS.md` — precision findings.
- `speechnet_argmax_ort_ref.py` (+ `.npz`) — host ORT replica computing the MaxPoolGrad argmax checksum.
- `speechnet_maxpool_90*`, `speechnet_maxpool_2sample*`, `*_argmax.log`, `*_loss.png`, `speechnet_drift_argmax_*.png` — device logs / losses / figures.

## `headonly_ondevice_finetune/` — the shipped head-only + BN-fold fine-tuning
The deployable, drift-free FT that gains +4.44 pp (train→extract→infer on-device).
- `SPEECHNET_ONDEVICE_FINETUNE_REPORT.md` — **the authoritative report**; plus `..._PLAN.md`, `..._RESULTS.md`, `SPEECHNET_PROGRESS.md`.
- `speechnet_accuracy_eval*.py` (+ results json/log) — on-device batch-2 accuracy harness.
- `speechnet_ft_*` (curve/faithful/folded/progressive/ortsweep/sim_search), `speechnet_loss_experiment.py`, `speechnet_debug_weights_step1.py` — FT exploration scripts.
- `speechnet_head_ep*`, `speechnet_b1ft_*`, `speechnet_b2_*`, `speechnet_100step_run.log`, `*_loss.png` — device logs / figures.

## `fulltraining_naccum8_regression/` — full-model FT regresses on-device
The full-model n_accum-8 experiment showing the −17.78 pp regression + its decomposition.
- `FULL_TRAINING_ONDEVICE_RESULTS.md` (authoritative), `FULL_TRAINING_EXPERIMENT_PROGRESS.md`.
- `speechnet_ft_full_naccum8_sweep.py`/`.json`, `speechnet_ft_fullmodel_ortsweep.py`/`.log` — host/ORT sweeps.
- `speechnet_full_device_reconstruct.py`, `speechnet_full_device_verify.py` — device weight extraction + drift table.
- `speechnet_full_gpu_ablation.py` — host ablation decomposing the device-vs-paper gap.
- `full8_e20_device.log`, `full8_e40_device.log`, `full8_sweep.log` — device WDUMP / sweep logs.

## `headonly_ondevice_ft_fixedwindow/` — head-only FT re-run with fixed (onset) windowing
Re-run of the shipped head-only + BN-fold on-device FT after fixing the SilentWear windowing
(center → onset; Onnx4Deeploy `77e269d`) and switching the base model to `inter_session_ft`.
Reproduces the paper's zero-shot exactly and confirms the FT gain fully on-device.
- `SPEECHNET_ONDEVICE_FT_FIXEDWINDOW.md` — **the authoritative report** (results, paper comparison, gain decomposition).
- `extract_device_fc.py` (+ `device_fc_{weight,bias}.npy`), `assemble_infer_fixtures.py`, `ort_ft_eval.py`,
  `gain_decomposition.py`, `run_ondevice_evals.sh` — extraction / fixture assembly / ORT ref / decomposition / eval driver.
- `run_train_isft_fw.log`, `eval_b{1_zs,2_zs,2_ft}_summary.log`, `results_b*.json` — on-device training + inference logs.
- Result: on-device b1 zero-shot **80.56 %** (= host), b2 zero-shot **81.67 %** (= paper `balanced_acc_no_ft`),
  b2 head-only fine-tuned **89.44 %** (+7.78 pp), all `sim_errors=0`.

## `frozen_bn_kernel_finetune/` — frozen-stat BN (the kernel modification)
The frozen-stat BN recipe + the validated `BatchNorm.c` kernel modification for paper-faithful
full-model FT at batch-1.
- `FROZEN_BN_FINETUNE_NOTE.md` — the full investigation + the final kernel-mod deployment config.
- `speechnet_frozenbn_gridsearch.py`/`.json`, `speechnet_frozenbn_variance.py`, `speechnet_frozenbn_fair.py`,
  `speechnet_frozenbn_device_final.py` — host studies (grid / variance / fair / device-faithful summing).
- `speechnet_fullfrozen_validate.py` — validates the `BN_FROZEN_STATS` kernel (device vs host, bit-exact).
- `frozenbn_*.log`, `frozenbn_grid_results.txt`, `fullfrozen_validate.log` — study outputs / device validation log.
- **`../fullfrozen_device.log`** (still in `DeeployTest/` — the LIVE 2160-forward run; moves here when it finishes).

## `reference/` — background references
- `SILENTWEAR_FINETUNE_SETUP.md` — the SilentWear paper's FT protocol, verified from their repo.
- `BN_AND_NACCUM_FINDINGS.md` — the two BN kernels; why `n_accum` ≠ batch for BN; the SUMMING convention.

---

### Modified source (not in `experiments/`; all QW-tagged)
- `TargetLibraries/PULPOpen/src/BatchNorm.c` — frozen-stat BN training kernel (fwd + bwd).
- `Platforms/Siracusa/src/deeploytraintest.c` — harness: `BN_FROZEN_STATS` enable, weight dump, argmax instrumentation.
- `Platforms/Siracusa/CMakeLists.txt` — `-D BN_FROZEN_STATS` / `-D DUMP_WEIGHTS` / `-D DUMP_ARGMAX` wiring.
- *(Onnx4Deeploy)* `onnx4deeploy/models/speechnet_exporter.py` — `_fold_bn_into_conv` (head-only BN-fold).

> Note: reproduction commands inside the moved docs may reference old top-level script paths; scripts
> now live under `experiments/<experiment>/`. They use absolute container paths internally, so they
> still run; only the doc-relative paths shifted.
