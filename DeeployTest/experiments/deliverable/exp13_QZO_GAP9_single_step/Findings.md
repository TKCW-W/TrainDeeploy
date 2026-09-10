# exp13 — Port on-device QZO simulation to GAP9 — Findings

Branch `feat/QZO`. Goal: run the exp10 single-step QZO flow (validated on Siracusa) on **GAP9**,
via GVSoC, as the last step before the NE16 accelerator.

## Status (2026-09-10): ✅ PASSES on GVSoC/GAP9

The single-step QZO SpeechNet smoke test **PASSES on GVSoC/GAP9** — the full forward + `zo_update`
optimizer step run, and all 8 device losses match the host reference within tolerance. This is the
GAP9 analog of the exp10 Siracusa single-step, and completes the QZO→GAP9 port.

Evidence (`logs/gap9_PASS_l1_110000_stacksL2.log`), device stdout:
```
=== GAP9 MeZO (ZO) Training Harness ===
update 1/1 accum 1..4/4: +eps/-eps forwards DONE, losses read
  (lp/lm bits: 0x3f9bbcb3/0x3e94fc49, 0x3e413bf6/0x3f90fdd0, 0x3dde8fd2/0x3c89c9ed, 0x3fce24e4/0x3f0b9629)
[PHASE] update (zo_update) START
[PHASE] update (zo_update) DONE (372562 cyc)
Errors: 0 out of 8
BENCH train_cycles_lo=66819806  opt_cycles_lo=372562
✓ Test speechnet_qzo12_train PASSED - No errors found
```
Config: `--l1 110000 --l2 1500000 -D BN_FROZEN_STATS=ON`, n_accum 4, ε 0.01, lr 1e-5, seed 42.

The path there had two runtime memory steps beyond the build fixes: (1) `--l1 90000` (not 128000)
got all forwards running but the `zo_update` step crashed on a tight-fit DMA; (2) **relocating the
cluster slave stacks from L1 to an L2 buffer** freed ~30 KB of L1 (usable L1 ~93 KB → ~123 KB),
letting `--l1` rise to 110000 so the optimizer arena has room — after which `zo_update` completes
and the test passes 0/8. (Earlier forwards-only evidence: `logs/gap9_forwards_run_l1_90000.log`.)

## How we got here

1. **Codegen (SDK-free):** the graph maps, tiles, and generates GAP9 C with the QZO kernels
   emitted — validated first without the SDK (`fixture/emitted_kernels.txt`). See `Plan.md`.
2. **SDK access:** the GAP9 SDK is private; the supervisor granted ghcr package-read on the
   prebuilt image `ghcr.io/runwangdl/deeploy:gap9`. Pulled it (amd64 → runs under qemu on the
   arm64 Mac), container `deeploy_gap9`, TrainDeeploy mounted at `/app/Deeploy`,
   `pip install -e .` into the SDK venv.
3. **~12 backward-compatible GAP9 build/portability fixes** (commit `ed2aeba`), each found by
   running to the real failure — none affect Siracusa:

| layer | fix |
|---|---|
| runner PATH | `deeployMezoRunner.py`: keep the SDK venv python ahead of `/usr/bin` when a venv is active (GAP9's `kconfigtool` needs `kconfiglib`); Siracusa (no venv) unchanged |
| perf counters | `perf_utils.h`: guard the `CSR_PCER_*` alias block (Siracusa only); fill GAP9's `PI_PERF_JMP_STALL`/`TAKEN_BRANCH`; `#include <assert.h>` |
| libc | `RandomNoiseQuant.c`: `#include <math.h>` for `lrintf` |
| ZO globals | `DeeployGAP9Math.h`: include `kernel/RandomNoise.h` + `kernel/ZORuntime.h` |
| DMA API | `DeeployMchan.h`: include `mchan_v7.h` (GAP9 **is** Mchan v7 → `MCHAN_TRANSFER_LEN_SIZE 17`, identical encoding; supplies `mchan_channel_*`), `MCHAN_CHANNEL_ID_MAX`, no-op `assert` fallback |
| warnings | `DeeployTest/CMakeLists.txt`: `-Wno-error` on `optimizer_network` too |
| harness | `deeploymezotest.c` (GAP9): wrap network Init/Run as `void(void*)` cluster entries; drop `cluster_task.stack_size` (GAP9 uses only `slave_stack_size`); drop `fflush`/`stdout` (GAP9 pmsis printf, no newlib stdio) |

4. **L1 tiling budget:** GAP9's usable cluster L1 is only **~93 KB**, not 128 KB — the 8 cluster
   slave stacks (`8 × ~3800 B ≈ 30 KB`) + FreeRTOS runtime live *in* L1 (Siracusa's lighter
   pulp-sdk runtime leaves ~128 KB free, which is why `--l1 128000` is fine there and fatal here).
   `--l1 128000` → the tiler builds a 128 KB arena that can't allocate at runtime → benign-looking
   "Allocation failed for allocator 2" (the cluster L1 allocator) turns into a bad pointer + DMA
   crash. `--l1 45000` → tiler reports "geometrical constraints infeasible" (SpeechNet block-0 is
   ~88 KB, `DEEPLOY_PATTERN_MEM…(1..88408)`). **`--l1 90000`** generates and runs the whole
   forward path — block-0 (88 KB) just fits under ~93 KB usable.

## Key runtime fix: slave stacks → L2 (credit: supervisor's GAP9 session)

The supervisor's agent (`ETH/gap9_deeploy_session.jsonl.gz`) had already mapped the GAP9 memory
constraint: the ~30 KB of cluster slave stacks are *required* (2048/3072 crash) and live in L1,
so usable L1 for the tiling arena is only ~93 KB. GAP9's SDK only `pi_cl_l1_malloc`s the stacks
when `task->stacks == NULL` (`cluster.c:440-444`), so handing it an L2 buffer via
`pi_cluster_task_stacks()` skips that L1 allocation. We added a static L2 stack buffer and wired
all task sites (`deeploymezotest.c`), freeing ~30 KB → usable L1 ~123 KB → `--l1 110000` fits both
the training (block-0 ~88 KB) and optimizer arenas, which cleared the `zo_update` DMA fault.

## Notes / follow-ups

- **MaxPool kernel name — VERIFIED benign (resolved).** The `-Wno-error`-demoted warning
  (`implicit declaration of 'PULP_MaxPool2d_fp32_fp32_HWC'; did you mean 'MaxPool2d_fp32_fp32_NCHW'`)
  is NOT a wrong/missing kernel: `MaxPool.c:16` defines `PULP_MaxPool2d_fp32_fp32_HWC` (globbed into
  `deeploygap9`), `nm` on the ELF shows it as a defined symbol (`T`), and the generated code calls
  exactly that name. It's only an implicit declaration (the generated `.c` lacks the `kernel/MaxPool.h`
  prototype in scope); clang's "did you mean …NCHW" is a nearest-match suggestion for a *different*
  (channels-first) kernel — a red herring, since `_HWC`/NHWC is the PULP/GAP9 datapath layout. The
  call is provably safe under an implicit decl because every arg is a pointer or `uint32_t` (no
  by-value float/double → no promotion mismatch), and the 0/8 loss match independently confirms
  correct MaxPool numerics. Cosmetic-only follow-up: emit `#include "kernel/MaxPool.h"` in codegen
  to silence the warning.
- Runs under qemu (amd64 image on the arm64 Mac), so cycle counts (train ~66.8 M, opt ~0.37 M) are
  functionally correct but not perf-calibrated; a native-arch or board run is the bar for latency.
- **Next:** NE16 accelerator on GAP9 (the final HW platform).

## Reproduction (in the deeploy-gap9 container)

```bash
docker exec deeploy_gap9 bash -lc '
  source /app/install/gap9-sdk/.gap9-venv/bin/activate
  source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
  export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
  cd /app/Deeploy/DeeployTest
  python3 deeployMezoRunner_tiled_GAP9.py \
    -t Tests/Models/Training/SpeechNet/speechnet_qzo12_train \
    --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo12_update \
    --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
    --l1 110000 --l2 1500000 --defaultMemLevel L2 -D BN_FROZEN_STATS=ON'
```
`--l1 110000` (not 128000) works on GAP9 **with the slave-stacks-in-L2 fix**; without it usable L1
is ~93 KB (see §L1) and only ~90000 fits (forwards only). Logs:
`logs/gap9_PASS_l1_110000_stacksL2.log` (the pass, Errors 0/8),
`logs/gap9_forwards_run_l1_90000.log` (forwards-only, pre-stacks-fix).
