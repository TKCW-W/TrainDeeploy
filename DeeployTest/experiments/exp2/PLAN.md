# PLAN — exp2: on-device fine-tuning latency & memory analysis

**Started:** 2026-07-29
**Branch:** TrainDeeploy `feat/BNFRozen_OptionB` (shared with exp1 Option B work)

## Goal
Measure and explain the **latency (cycles)** and **memory (L1/L2)** of one on-device SpeechNet
fine-tuning round, and study how they could be improved.

## Subtasks
### 2a — Tutorial: obtain / understand / analyse latency & memory
- Run ONE training round with `--profileTiling` and `--plotMemloc` on the tiled Siracusa runner.
- Convert the `--profileTiling` output to a Perfetto trace via `profiling_to_perfetto.py`; open in the
  Perfetto UI; explain how to read per-layer / per-tile cycle costs.
- Use the memory-allocation output (`--plotMemloc` / MiniMalloc report) to read peak L1/L2 usage and
  buffer lifetimes.
- Write a step-by-step tutorial with exact commands, output paths, and how to interpret each artifact.

### 2b — Improvement study (analysis only, no implementation)
- Where the cycles go (forward vs backward, which layers dominate — likely the early large-spatial
  conv/BN + their grads), and levers: tiling/parallelism (cores), memory level placement, kernel fusion,
  im2col/DMA overhead, training-only-a-subset (head-only / K=1) trade-offs.
- Where the memory goes (activation stashes for backward, grad-accumulation buffers, im2col scratch) and
  levers: recomputation, in-place, tile sizing, L1 budget.

## Deliverables
- `TUTORIAL.md` — the 2a walkthrough.
- `profiling_run/` — collected `--profileTiling` output + Perfetto trace + `--plotMemloc` artifacts.
- `FINDING.md` — measured cycles/latency + memory numbers (2a results) and the improvement study (2b).

## Tooling facts (confirmed 2026-07-29)
- **Latency:** `--profileTiling` injects `getCycles()` around every tiled op and prints UART lines:
  `===== Profiling <node> =====` then `[name][SB|DB][N ops][Tile T] (Pre-Kernel|Kernel|Post-Kernel|Total): NNNN cycles`
  — **per-tile**, three phases (Pre-Kernel = ingress DMA L2→L1, Kernel = compute, Post-Kernel = egress DMA
  L1→L2). Capture the runner's stdout to a log. Source: `TilingPrototypes.py:68-103`,
  `SingleBufferingTilingCodeGeneration.py:323-382`. A reference parser is
  `DeeployTest/benchmark_training.py:parse_profiling` (→ per-op-type + compute/DMA cycle breakdown).
- **Perfetto:** `profiling_to_perfetto.py` **exists at `/app/profiling_to_perfetto.py`** (container root,
  outside the mounted repo — that's why a repo search missed it). Usage:
  `python deeployTrainingRunner_tiled_siracusa.py ... --profileTiling 2>&1 | tee log.txt | python /app/profiling_to_perfetto.py - -o trace.json`
  then open `trace.json` in https://ui.perfetto.dev. Layout: one Process per **Step K**; threads
  Timeline(L3), Timeline(L2), Kernel, DMA L3→L2 / L2→L1 / L1→L2 / L2→L3. `classify_node` tags each op
  forward / backward / Optimizer (sgd_/gradientaccumulator/accumulategrad). 473 LOC, Chrome Trace JSON.
- **Memory:** the flag is **`--plotMemAlloc`** (NOT `--plotMemloc` — verify exact spelling in
  `testUtils/deeployRunner.py:152-174` before the run). Emits **`memory_alloc.html`** (interactive Plotly)
  under the deeployState dir (≈`TEST_SIRACUSA/.../deeployStates/`). Shows per-memory-level (L1 128KB / L2
  2MB / L3) buffer lifetime windows + address packing + peak-utilization %. Impl:
  `Deeploy/TilingExtension/TilerExtension.py:plotMemoryAlloc` (153-330); alloc strategies TetrisRandom /
  TetrisCo-Opt / MiniMalloc.
- **Profiling command (one round):** the existing train invocation + `--profileTiling --plotMemAlloc`,
  piping stdout through the perfetto converter. Collect: `trace.json`, `memory_alloc.html`, the raw UART log.
