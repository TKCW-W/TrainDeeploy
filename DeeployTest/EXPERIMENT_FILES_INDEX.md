# Experiment files index

Every file created during the SpeechNet on-device fine-tuning experiments, with a one-line
description. Grouped by type. (Modified *source* files are listed at the end for reference.)

## Docs / reports

| file | what it is |
|---|---|
| `SPEECHNET_ONDEVICE_FINETUNE_REPORT.md` | Authoritative report on the whole on-device FT effort (head-only + BN-fold, drift, methodology, reproduction). |
| `SILENTWEAR_FINETUNE_SETUP.md` | Reference: the SilentWear paper's fine-tuning protocol (Adam, batch-32, 50 ep, ~70% data), verified from their repo. |
| `BN_AND_NACCUM_FINDINGS.md` | Why training vs inference use two different BN kernels; why `n_accum` ≠ batch for BatchNorm; the memory/convention notes. |
| `FULL_TRAINING_ONDEVICE_RESULTS.md` | Results of the full-training on-device experiments: the −17.78pp regression decomposition, drift analysis, host ablation. |
| `FULL_TRAINING_EXPERIMENT_PROGRESS.md` | Progress tracker for the full-training (n_accum 8) experiment. |
| `FROZEN_BN_FINETUNE_NOTE.md` | The frozen-stat BN investigation: grid/variance/fair studies + the final kernel-modification deployment config. |
| `EXPERIMENT_FILES_INDEX.md` | This file. |

## Scripts — host PyTorch / ORT experiments

| file | what it is |
|---|---|
| `speechnet_ft_full_naccum8_sweep.py` | Host/ORT sweep of full-training `n_accum 8` configs (data × lr × epochs), eval batch-2. |
| `speechnet_full_gpu_ablation.py` | Host PyTorch ablation decomposing the device-vs-paper gap (batch-1→batch-N, running-stat update, SGD→Adam). |
| `speechnet_frozenbn_gridsearch.py` | Host grid search of frozen-stat BN full-FT over lr × n_accum × data-size {30/40/50/70%}. |
| `speechnet_frozenbn_variance.py` | Variance study: frozen-BN full-FT vs head-only across 8 stratified data draws × 2 seeds. |
| `speechnet_frozenbn_fair.py` | Fair head-to-head — each recipe (full-FT, head-only) at its own best lr, same draws. |
| `speechnet_frozenbn_device_final.py` | Device-faithful (SUMMING convention) variance study to pick the final config. |

## Scripts — device weight extraction / verification

| file | what it is |
|---|---|
| `speechnet_full_device_reconstruct.py` | Parse device `[WDUMP]`, inject the weights into the infer graph, eval batch-2 (full-model). |
| `speechnet_full_device_verify.py` | Rigorous device-weight verification + per-weight drift table (device vs ORT reference). |
| `speechnet_fullfrozen_validate.py` | Validate the `BN_FROZEN_STATS` kernel: device 20-step weights vs a host frozen-BN reference (bit-exact check). |

## Logs & data

| file | what it is |
|---|---|
| `full8_e20_device.log` | Device WDUMP log — full-training n_accum8, ep20 (1080 forwards). |
| `full8_e40_device.log` | Device WDUMP log — full-training n_accum8, ep40 (2160 forwards). |
| `full8_sweep.log` | Raw host ORT sweep log (untracked; results in the JSON). |
| `speechnet_ft_full_naccum8_sweep.json` | Results of the full-training n_accum8 ORT sweep. |
| `frozenbn_grid_results.txt` | Frozen-BN grid-search leaderboard (filtered from the raw log). |
| `speechnet_frozenbn_gridsearch.json` | Frozen-BN grid full results (all 60 configs). |
| `frozenbn_variance.log` | Variance-study output (full-FT vs head-only, mean±std). |
| `frozenbn_fair.log` | Fair head-to-head output (each recipe at its best lr). |
| `frozenbn_device_final.log` | Device-faithful summing study output (untracked). |
| `fullfrozen_validate.log` | 20-step `BN_FROZEN_STATS` kernel-validation device log (untracked). |
| `fullfrozen_device.log` | Full 2160-step frozen-BN device run log — the paper-faithful weight-gathering run (untracked, in progress). |

## Modified source (not created — for reference; all QW-tagged)

| file | change |
|---|---|
| `TargetLibraries/PULPOpen/src/BatchNorm.c` | Frozen-stat BN training kernel: `PULP_BatchNormInternal_fp32` (fwd) + `PULP_BatchNormGrad_fp32` (bwd), gated by `g_bn_frozen_stats`. |
| `Platforms/Siracusa/src/deeploytraintest.c` | Harness: enable frozen-BN under `#ifdef BN_FROZEN_STATS`; on-device weight dump + MaxPool argmax instrumentation. |
| `Platforms/Siracusa/CMakeLists.txt` | `-D BN_FROZEN_STATS` / `-D DUMP_WEIGHTS` / `-D DUMP_ARGMAX` compile-flag wiring. |
| *(Onnx4Deeploy)* `onnx4deeploy/models/speechnet_exporter.py` | `_fold_bn_into_conv` (BN-fold for head-only, frozen running stats). |

## Fixtures generated (under `Tests/Models/Training/SpeechNet/`, untracked/regenerable)

| dir | what it is |
|---|---|
| `speechnet_train_fullfrozen` (+ optimizer) | Full-training fixture (unfolded, keeps BatchNormInternal) — used with the frozen-BN kernel. THE deployment fixture. |
| `speechnet_train_full8_e20` / `_e40` (+ optimizers) | Full-training n_accum8 fixtures (ep20/ep40) for the earlier on-device full-model runs. |
| `speechnet_ftpool_ds72` / `_ds90` / `_ds126` / `_ds180` | Fine-tune data pools at 40/50/70/100% for the grid/variance studies (windows only). |
