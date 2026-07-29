# exp2 · 2a — Tutorial: profiling latency (cycles) & memory of on-device SpeechNet fine-tuning

This walks through how to **obtain**, **understand**, and **analyse** the per-step latency and the
L1/L2 memory footprint of one on-device SpeechNet fine-tuning round on the Siracusa (GVSoC) target,
using the exact artifacts produced under `experiments/exp2/profiling_run/`.

Everything runs inside the container **`deeploy_arm_mounted`** (GVSoC + Deeploy installed), branch
`feat/BNFRozen_OptionB`, with **sole use of the `TEST_SIRACUSA` build dir**.

Recipe under test = the deployment recipe: **full-model fine-tune, frozen BN stats, batch 1, n-accum 4**,
fixture `Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3`.

---

## 0. The two profiling knobs (verified in `testUtils/deeployRunner.py`)

| Flag | What it does | Output |
|------|--------------|--------|
| `--profileTiling` | Wraps every tiled op with `getCycles()` and prints per-tile UART lines | cycle lines on stdout (capture to a log) |
| `--profileNodes SUBSTR[,SUBSTR...]` | With `--profileTiling`, only instrument nodes whose name contains one of these substrings | fewer arrays → less stack pressure |
| `--plotMemAlloc` | Emits interactive Plotly memory maps at **generation** time | `memory_alloc.html` in each `deeployStates*` dir |

The memory flag is **`--plotMemAlloc`** (not `--plotMemloc`). It is produced during code-gen/tiling, so it
is available even if the GVSoC run itself never starts.

### UART cycle-line format
```
===== Profiling <node> =====
[<node>][SB|DB][N ops][Tile T] Pre-Kernel :  NNNN cycles   <- ingress DMA  (L2 -> L1)
[<node>][SB|DB][N ops][Tile T] Kernel     :  NNNN cycles   <- compute on the 8 cores
[<node>][SB|DB][N ops][Tile T] Post-Kernel:  NNNN cycles   <- egress DMA   (L1 -> L2)
[<node>][SB|DB][N ops][Tile T] Total      :  NNNN cycles (X% Kernel + Y% Overhead, ker + dma)
```
Deeploy emits **two blocks per node**: a *cluster* block (Pre/Post = L2↔L1 DMA, Kernel = compute) and an
*L3* block (Pre/Post = L3↔L2 DMA, Kernel = the whole cluster dispatch — do **not** re-count it, it would
double-count the compute). Our SpeechNet FT fixture is L2-resident (`--defaultMemLevel L2`), so the L3
blocks carry ~0 DMA here.

---

## 1. IMPORTANT gotcha: full-graph `--profileTiling` overflows the core stack on this fixture

Running `--profileTiling` over the **whole** 78-node training graph builds fine but GVSoC dies at runtime
right after `Initializing TrainingNetwork...`:
```
/chip/cluster/pe0/lsu ... Invalid access (pc: 0x1c00b968, offset: 0x57575757, size: 0x1, is_write: 0)
```
`0x57='W'` is the uninitialized-stack fill pattern. Each instrumented node adds ~6 `getCycles()`
measurement arrays on the core stack; over 78 nodes this overruns the L1 stack. (See the failed run
`profiling_run/train_profile.log`.)

**Workaround (used here):** profile the graph in a few passes with `--profileNodes`, each covering a
disjoint set of op families, then merge the logs. Per-op cycles are **deterministic** across the
n-accum micro-batches, so merging is exact. The five passes we ran:

```bash
cd /app/TrainDeeploy/DeeployTest
BASE="python deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3 \
  --n-steps 1 --n-accum 4 --cores 8 --l1 128000 --l2 2000000 \
  --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D BN_FROZEN_STATS=ON --profileTiling"

rm -rf TEST_SIRACUSA && $BASE --profileNodes Conv                                          --plotMemAlloc 2>&1 | tee experiments/exp2/profiling_run/probe_conv.log
rm -rf TEST_SIRACUSA && $BASE --profileNodes BatchNorm,Relu,MaxPool,GlobalAveragePool,AveragePool 2>&1 | tee experiments/exp2/profiling_run/probe_bn_relu_pool.log
rm -rf TEST_SIRACUSA && $BASE --profileNodes Gemm,Softmax,ReduceSum,Reshape               2>&1 | tee experiments/exp2/profiling_run/probe_gemm_loss.log
rm -rf TEST_SIRACUSA && $BASE --profileNodes Accumulat                                    2>&1 | tee experiments/exp2/profiling_run/probe_opt.log
rm -rf TEST_SIRACUSA && $BASE --profileNodes sgd                                          2>&1 | tee experiments/exp2/profiling_run/probe_sgd.log
```
Notes:
- `--n-steps 1` is enough — the op cost is identical every step.
- `--profileNodes` matching is **case-sensitive**: the optimizer nodes are named `sgd_...` (lowercase),
  so use `sgd`, not `SGD`.
- `Conv` also captures `ConvGrad*` (backward). `Accumulat` captures the 22 `GradientAccumulator*`
  (gradient-accumulation) nodes that live in the *training* network.
- Keep `DUMP_WEIGHTS` **off** (not passed) — profiling doesn't need it and it bloats runtime.

Cross-check: each pass also prints a `BENCH train_cycles=... opt_cycles=...` line — the authoritative
aggregate the merged per-op numbers must reconcile to (they match to <0.2 %).

---

## 2. Convert the cycle log(s) → a Chrome/Perfetto trace

Note on `profiling_to_perfetto.py`: it **does** exist — but in the **export** container (`agitated_hugle`)
at `/app/profiling_to_perfetto.py`, **not** in the GVSoC container (`deeploy_arm_mounted`) where the train
runner runs. Its CLI is `python /app/profiling_to_perfetto.py <log|-> -o trace.json` (same UART format). To
avoid the cross-container hop — and to add per-step cycle analysis + the multi-pass merge (see §1) — we ship
a self-contained equivalent: `experiments/exp2/profiling_run/analyze_and_perfetto.py`. It

1. parses all `--profileTiling` logs (regex on the UART lines),
2. de-dupes sections by node name, classifies each as **forward / backward / optimizer**
   (`sgd`/`accumulator`→optimizer, `*grad*`/`*backward*`→backward, else forward),
3. multiplies fwd/bwd/accum ops by **n-accum (=4)** and the SGD update by 1 to get **per training step**,
4. prints the cycle breakdown + top ops + per-family table, and
5. writes a **Chrome Trace JSON** (`trace.json`).

```bash
cd experiments/exp2/profiling_run
python analyze_and_perfetto.py \
  probe_conv.log probe_bn_relu_pool.log probe_gemm_loss.log probe_opt.log probe_sgd.log \
  trace.json | tee cycle_summary.txt
```

Open **`trace.json`** in <https://ui.perfetto.dev> (drag-and-drop). Layout:
- **Process** "SpeechNet FT step".
- **Tracks** `forward`, `backward`, `optimizer` (compute laid out sequentially in cycle-time) and a `DMA`
  track (Pre+Post per op). Each slice's `args` carry `kernel`, `dma`, `invoc`, and the full `node` name.
- Reading it: wide `forward`/`backward` slices = the dominant Conv / ConvGrad / MaxPoolGrad kernels; the
  `DMA` track staying thin under them = **compute-bound** (which this network is). Use the marquee
  (drag-select) to sum durations of a region, and the search box to jump to a node by name.

---

## 3. Read the memory maps

`--plotMemAlloc` writes one `memory_alloc.html` per network:
- Training: `TEST_SIRACUSA/Tests/.../speechnet_train_fullfrozen_b1_fold3/deeployStates/memory_alloc.html`
- Optimizer: `.../deeployStates_optimizer/memory_alloc.html`

We copied them to `profiling_run/memory_alloc_training.html` and `.../memory_alloc_optimizer.html`.

Open in a browser. Three stacked panels — **L1 (128 KB)**, **L2 (2 MB)**, **L3 (64 MB)**:
- **x axis = tiling/schedule step (a proxy for time)**, **y axis = byte address**. Each buffer is a
  rectangle: its **height = buffer size**, its **horizontal width = lifetime** (how many schedule steps it
  must stay resident). The dashed red line is the memory-level **budget**.
- **Peak utilisation** = the highest rectangle top vs the budget line. You can also read the exact peak
  from the generated header: `TrainingNetwork.h` → `DeeployNetwork_MEMORYARENA_L1_len` /
  `..._L2_len` (bytes actually reserved).
- **Long, tall rectangles** are the expensive ones: a large activation that must be **stashed across the
  whole graph** for the backward pass (lifetime spanning the full x-range) is exactly the recompute-vs-stash
  trade-off candidate.
- To pull the numbers programmatically, each panel is a `var fig = {...}` JSON blob in the HTML; parse the
  `data` array (rectangles as `x`/`y` coords). `profiling_run/analyze_and_perfetto.py`'s sibling snippet
  and the FINDING.md table were produced this way.

### How to read the interpreted result
- **L1 is the binding constraint** here: reserved 127,808 / 128,000 B ≈ **99.9 %**. The tallest single L1
  tile is the block-0 conv-grad activation (~118 KB) — one tile nearly fills L1, which forces small tiles
  elsewhere and adds DMA rounds.
- **L2 peak ≈ 89.7 %** (1,793,800 / 2,000,000 B). The tallest L2 buffers are the 314 KB block-0
  large-spatial activations; several have **lifetime ≈ 88–93** (nearly the whole schedule) — those are the
  forward activations kept alive for backward.

---

## 4. Artifact index (what each file is)

| File | What it is |
|------|-----------|
| `profiling_run/train_profile.log` | the **failed** full-graph `--profileTiling` run — evidence of the stack-overflow gotcha |
| `profiling_run/probe_*.log` | the 5 per-family `--profileTiling` passes that together cover the whole graph |
| `profiling_run/analyze_and_perfetto.py` | parser + per-step cycle analyser + Chrome-trace writer |
| `profiling_run/cycle_summary.txt` | captured stdout of the analyser (the numbers in FINDING.md) |
| `profiling_run/trace.json` | Chrome Trace JSON → open in ui.perfetto.dev |
| `profiling_run/memory_alloc_training.html` | Plotly L1/L2/L3 map of the training (fwd+bwd) network |
| `profiling_run/memory_alloc_optimizer.html` | Plotly L1/L2/L3 map of the SGD optimizer network |

See `FINDING.md` for the measured numbers and the 2b improvement study.
