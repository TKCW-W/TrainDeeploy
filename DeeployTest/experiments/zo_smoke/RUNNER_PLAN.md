<!-- 2026-08-04  ZO runner implementation plan (condensed from harness study). -- QW -->
# ZO runner — implementation plan (deeploymezotest.c + codegen + cmake)

## Mapping (confirmed)
- `zo_train` ↔ **TrainingNetwork** (`DeeployNetwork_*`): perturb-all → fwd → SoftmaxCE → `outputs[0]` = scalar loss. No grad bufs, no lazy_reset_grad (layout collapses to `[data,label] + [22 weights]`).
- `zo_update` ↔ **OptimizerNetwork** (`DeeployOptNetwork_*`): 22 in-place PerturbRademacher, **no outputs**. `build_shared_buffer_maps` name-matches its 22 inputs → the training weight buffers; `_patch_shared_buffers` redirects their mallocs + drops file loads → **in-place update on the shared weights** (exactly what ZO needs). `shared_output_map` empty → C main must NOT touch `DeeployOptNetwork_outputs[]`.
- Update magnitude is runtime: `perturb_eps_override = -lr*g_proj` (NOT baked lr).

## The C loop (deeploymezotest.c — copy deeploytraintest.c, strip grad/optimizer-copy machinery)
```
for update_step in [0,N_TRAIN_STEPS):
  acc = 0                                   # scalar (L+ - L-)
  seed_base = update_step*ZO_Q + 0          # q_i=0 for q=1
  for accum_step in [0,N_ACCUM_STEPS):
    mb = update_step*N_ACCUM_STEPS+accum_step
    load mb into DeeployNetwork_inputs[0..NUM_DATA-1]
    perturbation_sign=1; perturb_seed_base=seed_base; perturb_eps_use_override=0
    RunTrainingNetwork();  memcpy outputs[0]->Lp;  stored_loss_plus[mb]=Lp
    perturbation_sign=0
    RunTrainingNetwork();  memcpy outputs[0]->Lm;  stored_loss_minus[mb]=Lm
    acc += (Lp - Lm)                        # FP -> on CLUSTER
  g_proj = acc/(2*ZO_EPS*N_ACCUM_STEPS)     # FP -> on CLUSTER
  perturb_eps_use_override=1; perturb_eps_override = -ZO_LR*g_proj
  perturbation_sign=1; perturb_seed_base=seed_base
  RunOptimizerNetwork();                    # zo_update, in-place on shared weights
  perturb_eps_use_override=0
compare stored_loss_plus vs testLossPlusRef, stored_loss_minus vs testLossMinusRef
```
**FPU caveat:** FC has no FPU. Read loss scalar FPU-free (memcpy like BP @deeploytraintest.c:419-427), but do `(Lp-Lm)`, `g_proj`, `-lr*g_proj` in an on-cluster helper (mirror `CompareLossesOnCluster` @252-270) that writes `perturb_eps_override`. Globals are set on FC BEFORE `pi_cluster_send_task_to_cl` (read inside kernel on cluster).
**Keep:** init-weights copy (@348-354), per-mb data load (@393-397), `BN_FROZEN_STATS` (@363-368; SpeechNet needs it). **Storage:** `stored_loss_plus/minus[N_TRAIN_STEPS*N_ACCUM_STEPS]`.

## Files
- NEW `DeeployTest/Platforms/Siracusa/src/deeploymezotest.c` (`#include "kernel/ZORuntime.h"`; ZO_EPS/ZO_LR/ZO_Q/ZO_SEED macros).
- EDIT `Platforms/Siracusa/CMakeLists.txt`: `option(MEZO_TRAINING)`; select deeploymezotest.c; link `training_network`(=zo_train)+`optimizer_network`(=zo_update); `-DN_TRAIN_STEPS -DN_ACCUM_STEPS -DTRAINING_NUM_DATA_INPUTS -DZO_EPS -DZO_LR -DZO_Q -DZO_SEED` + BN_FROZEN_STATS/DUMP_WEIGHTS.
- EDIT `DeeployTest/CMakeLists.txt`: `if(TRAINING OR MEZO_TRAINING)` for the two OBJECT libs.
- EDIT `testUtils/core/config.py`: `mezo`, `zo_eps/zo_lr/zo_q/zo_seed`.
- EDIT `testUtils/core/execution.py`: `generate_network` → `if config.mezo: run_zo_codegen(); return`; `configure_cmake` → append MEZO -D's.
- EDIT `testUtils/trainingUtils.py`: `run_zo_codegen` (Stage1 testMVPTraining on zo_train w/ ZO outputs header; Stage2 testMVPOptimizer on zo_update as-is); `resolve_zo_update_dir` (_zo_train→_zo_update).
- EDIT `testUtils/codeGenerateTraining.py`: `generateZOTestOutputsHeader` → `testLossPlusRef[]`/`testLossMinusRef[]`/`ZO_TOLERANCE_ABS`; load `loss_plus`/`loss_minus` explicitly (not first-`loss` key).
- NEW `deeployMezoRunner_tiled_siracusa.py` (or reuse deeployTrainingRunner.main w/ mezo=True + --eps/--lr/--q/--seed).
- Test dirs: `Tests/Models/Training/SpeechNet/{speechnet_zo_train,speechnet_zo_update}/` (DONE; from `Onnx4Deeploy/onnx/model/speechnet_zo_n1_smoke/`).

## Risks (verify)
1. zo_update in-place correctness rests on name-matched buffer sharing → inspect emitted `OptimizerNetwork.c` for 22 `DeeployOptNetwork_input_N = (float32_t*)DeeployNetwork_input_M;` + dropped file loads. Any weight-name mismatch → garbage buffer, silent no-op.
2. zo_update has **0 outputs** → C main must not index outputs[]; if tiler balks at zero-output graph, fall back to marking the 22 weights as outputs (still out==in).
3. zo_train perturbs base→perturbed (NOT in-place) → clean base weights must stay live so the −ε pass perturbs the clean base (SB-tiler extends input lifetime; verify).
4. Single-tile RNG bit-exactness: reference assumes tile_seed_offset=0; verify none of the 22 tensors tiles (fc weight largest). If tiled, mismatch.
5. Same `seed_base` across +ε/−ε/update so z is identical.

## Bring-up order
n_accum=1 fixture (`speechnet_zo_train`, 9 steps) → compare loss_plus/minus (step 0 first) → DUMP_WEIGHTS vs final weights → n_accum=4 → multi-step → exp18 (ε=0.01, lr=3e-6, q=1, 200 ep).
