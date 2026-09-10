# exp13 — Port on-device QZO simulation to GAP9 — Findings

Branch `feat/QZO`. Goal: run the exp10 single-step QZO flow (validated on Siracusa) on **GAP9**,
via GVSoC, as the last step before the NE16 accelerator.

## Status (2026-09-10)

The QZO SpeechNet single-step **builds, links, and runs on GVSoC/GAP9, and the entire forward
path executes** — all four accum mini-batches' ±ε losses are computed and read off the device.
The run currently stops in the **optimizer (`zo_update`) step** with a bad-address cluster DMA.
So: **not yet a clean single-step pass**, but the whole compile/link/run port and the forward
datapath are done.

Evidence (`logs/gap9_forwards_run_l1_90000.log`), device stdout:
```
=== GAP9 MeZO (ZO) Training Harness ===
update 1/1  accum 1/4 ... [PHASE] +eps forward DONE  lp_bits=0x3f9bbcb3 ... lm_bits=0x3e94fc49
                 accum 2/4 ... lp_bits=0x3e413bf6 ... lm_bits=0x3f90fdd0
                 accum 3/4 ... lp_bits=0x3dde8fd2 ... lm_bits=0x3c89c9ed
                 accum 4/4 ... lp_bits=0x3fce24e4 ... lm_bits=0x3f0b9629
[PHASE] update (zo_update) START  →  /chip/cluster/dma  Got error during transfer (addr: 0x42215)
```

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

## Remaining work (mapped from the supervisor's GAP9 session)

The supervisor's agent ported several (different) training models to GAP9 and hit the same class
of issues; their session (`ETH/gap9_deeploy_session.jsonl.gz`) gives the path:

- **Optimizer-step crash** (our current blocker, `zo_update` DMA to `0x42215`): they bisected the
  same "crash entering the optimizer step" to the **generated tiled OptimizerNetwork DMA codegen**
  (ruling out stack overflow for it), and got 2/9 e2e models fully passing — so the optimizer DMA
  path *can* work on GAP9. Ours uses the ZO perturb `zo_update` (theirs SGD), so it's related but
  needs its own root-cause of the perturb-op tiled DMA descriptor.
- **L1 head-room (stacks→L2):** they proved the ~30 KB of slave stacks are *required* (2048/3072
  crash), and freed L1 by **relocating the slave stacks into an L2 static buffer** (~30 KB back to
  L1 → usable ~123 KB). Applying this lets `--l1` rise well above 90 KB, removing the tight-fit
  risk (and possibly the optimizer DMA fault if it is tight-fit-induced). **This is the next step.**
- **Latent:** the generated code calls `PULP_MaxPool2d_fp32_fp32_HWC` but GAP9's lib has
  `MaxPool2d_fp32_fp32_NCHW` (currently `-Wno-error`) — verify MaxPool numerics once it runs clean.

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
    --l1 90000 --l2 1500000 --defaultMemLevel L2 -D BN_FROZEN_STATS=ON'
```
`--l1 90000` (not 128000) is required on GAP9; see §L1. Logs: `logs/gap9_forwards_run_l1_90000.log`
(forwards pass, optimizer crashes), `logs/gap9_smoke_run.log` (latest).
