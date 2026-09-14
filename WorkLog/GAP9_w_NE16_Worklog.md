# GAP9_w_NE16 Worklog — NE16 accelerator for QZO SpeechNet on GAP9

Progress tracking for the **NE16-on-GAP9** effort. Read `WorkLog/README.md` first for the
QZO/ZO pipeline as a whole; this file only covers NE16.

**Branch: `feat/GAP9_w_NE16`** in both `TrainDeeploy` and `Onnx4Deeploy` (cut from `feat/QZO`
at `TrainDeeploy@28ff5fa` / `Onnx4Deeploy@2be5137`, both of which are pushed to `TKCW-W/*`).

**Scope: QZO only.** Backpropagation is out of scope — see the plan's Appendix A for why.

---

# ⏸ PAUSED 2026-09-14 — READ THIS FIRST TO RESUME

**State: the goal is met.** SpeechNet QZO runs end-to-end on GAP9 with NE16 and is
**bit-identical** to the cluster-only baseline (`max |NE16 − cluster| = 0.00e+00` across all eight
per-pass losses). Working tree clean, 10 commits, head `ac6fe16`.

**Pushed** to `origin` (`TKCW-W/TrainDeeploy`) 2026-09-14 — `feat/GAP9_w_NE16` tracks
`origin/feat/GAP9_w_NE16`. `Onnx4Deeploy` is untouched by this work, still at `2be5137`.

### What runs today

`deeployMezoRunner_tiled_GAP9_w_NE16.py` on `speechnet_qzo12_train` with `--enable-1xk`:
**14 NE16 dispatches, 2 on-device weight encodes, 3 cluster convs.** Blocks **3 and 4** execute on
NE16 with their weights perturbed *and* bit-serial encoded on device; blocks 0/1/2 stay on the
cluster. Exact command in `exp16c_SDK_port/Findings.md §2`.

### The two open items, in priority order

| # | item | why it matters | where to start |
|---|---|---|---|
| **1** | **Padded convolutions** | Blocks 0/1/2 carry pads, and `ne16_1xkAdmissible` refuses them — so only **5.8 % of conv MACs** reach NE16. Lifting this brings block 1 (68 % of MACs on its own) onto NE16. | `Prepare1xKPass.ne16_1xkAdmissible`; the fix shape is the SDK's `NE16_ComputeBorders` — pointer arithmetic + border subtiles, **not** NE16 padding (its 1×1 mode cannot pad and the gvsoc guard is commented out, so it fails *silently*). |
| **2** | **Performance** | NE16 is **4.0–7.5× slower per layer** than `pulp_nn_conv`; the full step is ~1.9 % slower overall. | Block 1 retires **~1.6 MAC/cycle ≈ 700 cycles per output subtile** for work that should take tens. **Profile that first** — it dominates, and channel folding alone is estimated to close only about half the gap (see the caveat below). |

**Block 0 additionally needs the signed-activation path wired into the pass.** The arithmetic is
already built and validated (`NE16SignedInputBias`, exp16c_PW phase 4, `0/78512`); what is missing
is a graph-level op applying `x ^ 0x80`. Note that **im2col would dissolve both blockers at once** —
the cores write pad values *and* the `+128` offset into the column buffer, and NE16 then runs an
unpadded, unsigned 1×1 conv.

### Estimate caveat — do not trust the folding numbers without measuring

Channel folding only reduces MAC work where `Cin` leaves a `TP_IN` group half-empty: **blocks 2, 3
and 4 already fill it exactly**, so for them folding buys only the removal of streamin traffic and
per-dispatch setup. Estimated overall: NE16 conv time 7.75 M → ~3–4 M cycles against the cluster's
1.23 M — i.e. **probably still slower**. My cost models have been wrong twice here in both
directions (predicted dense-3×3 3× slower, measured 1.15× *faster*; expected pipelining to matter,
measured 1.65 %). Measure one folded block-1 dispatch before building on any of this.

### Three traps that will cost you a day if forgotten

1. **The harness verdict is not a correctness check.** `Errors: 0 out of 8` checks the *update*
   graph, driven by the fixture's reference losses — **not** the device-measured ones. It reported
   `PASSED` for the entire period when losses were 20–55 % wrong. Compare device losses against
   `speechnet_qzo12_update/outputs.npz` directly.
2. **Single-layer fixtures must use `--variable-bias`.** Baking the perturbed bias in as a
   *constant* puts the fixture in the branch where merged and un-merged agree by construction —
   which is exactly why every isolated test passed while the network was wrong. The faithful
   reproducer (`b3_plain_vb`, ~2 min) is what found all three bugs.
3. **`-D BN_FROZEN_STATS=ON` and `--l1 110000` are load-bearing.** Without them `.weightmem_sram`
   overflows `L2_shared` by ~706 KB — **with or without NE16**, so it is not an NE16 cost.

### Three bugs fixed here — all pre-existing, all unreachable before NE16

With every convolution fused into a `RequantizedConv`, no standalone RequantShift with a runtime
bias ever existed; NE16 is the first thing to leave one un-merged, and it must, because `streamin`
forces int32 output.

| # | file | defect |
|---|---|---|
| 1 | `Targets/{PULPOpen,Generic}/Templates/RequantShiftTemplate.py` | `rounding = 1` hardcoded; the merge bakes `+div/2` only for a **constant** add → off by `div/2` with a runtime bias |
| 2 | `Targets/PULPOpen/TileConstraints/RQSPerturbTileConstraint.py` (serialize) | `channelDim = dims[0]` on a broadcast `(1,cout)` bias → kernel collapses to `M[0]` for every channel |
| 3 | same file (geometry) | input/output dims paired by raw index → 4-byte tile against a 128-byte DMA → 124-byte L1 overlap |

Plus two earlier ones: `RequantShiftLayer.computeShapes` ignoring `channels_first`, and
`UniformRequantShiftParser.parseNode` *raising* instead of declining on a runtime mul/add.

### Where the detail is

`exp16c_SDK_port/Findings.md` — results, a copy-pasteable reproduction section, and every changed
file by path. Sibling experiments: `exp16a` (all-pointwise), `exp16b` (dense 3×3),
`exp16c_PW_single_layer` (blocker 1b + STEP 3). SDK study:
`ETH/WorkLog/GAP9_SDK_NE16_Corner_Cases.md` and `ETH/WorkLog/GAP9_Container_Inventory.md`.

---

### Where the detailed material lives (do not rely on this file alone)

| what | exact path |
|---|---|
| The plan (four steps, blockers, tasks) | `ETH/docs/TRAIN_GAP9_NE16/03-qzo-ne16-plan.md` |
| What upstream PR #183 adds, file by file | `ETH/docs/TRAIN_GAP9_NE16/01-pr183-anatomy.md` |
| NE16 hardware facts, CONFIG0 map, PR #183's bugs | `ETH/docs/TRAIN_GAP9_NE16/02-ne16-hardware-and-gotchas.md` |
| PR #183 → our tree file index | `ETH/docs/TRAIN_GAP9_NE16/04-file-index.md` |
| Experiment artefacts, logs, scripts | `TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/` |
| Memory note (auto-loaded) | `~/.claude/projects/-Users-qiwenwu-ETH/memory/ne16_gap9_training_study.md` |

### The four steps

| | step | state |
|---|---|---|
| 1 | **Make it compile** — `GAP9_w_NE16` exists, QZO builds+runs on it with NE16 claiming nothing | ✅ **DONE** 2026-09-10 |
| 2a | **Make it work (plumbing)** — a real SpeechNet conv dispatches to NE16, bit-exact | ✅ **DONE** 2026-09-10 |
| 2b | **Make it work (real shape)** — block 1 at its true `1×16` via the 1×k decomposition | ✅ **DONE** 2026-09-11 (cropped extent; see exp16a) |
| 3 | **Make it correct** — full SpeechNet, inference then training | ✅ **DONE** 2026-09-14 — all 5 convs bit-exact standalone; full QZO step bit-identical to the cluster with blocks 3/4 on NE16 (0/1/2 blocked by padding) |
| 4 | **Optimise** | ⬜ **OPEN** — NE16 is 4.0–7.5× slower per layer; see the PAUSED section at the top |

---

## 2026-09-10 — Session 1: study, vendoring, STEP 1 and STEP 2a

### 09-08 → 09-10 (prior sessions) — study of upstream PR #183

Read `pulp-platform/Deeploy` PR #183 *"[NE16] Add GAP9_w_NE16 platform"* (head `45ca06c`, base
`bf64cfae`, 42 commits, **still open**) and wrote the four documents listed above. PR is fetchable
without `gh`:

```bash
git ls-remote https://github.com/pulp-platform/Deeploy 'refs/pull/183/*'
git fetch --depth=200 <remote> refs/pull/183/head:refs/heads/pr183head
```

### 2026-09-10 — verification of every load-bearing claim against the real SDK

Everything the study had inferred from PR #183's prose was re-checked against the GAP9 SDK's own
NE16 sources inside container `deeploy_gap9`, at
`/app/install/gap9-sdk/gvsoc/gvsoc_gap/gap/ne16/src/`:

| claim | verdict | evidence |
|---|---|---|
| Only 3 filter modes | ✅ | `ne16_regfile.cpp:222` — `[6:5]`: `00`=3×3, `01`=3×3-DW, `10`=1×1, `11`=reserved |
| Requant truncates | ✅ | `ne16_normquant.cpp:169` — `accum32[i] >> shift`, no rounding term |
| Bit 26 is a phantom | ✅ | CONFIG0 decode ends at `[25]`; nothing reads bit 26 |
| `[14]` streamin exists | ✅ | `ne16_regfile.cpp:214` |
| `qw` can be 1 | ✅ **better than assumed** | `ne16_regfile.cpp:230` — `qw = (value & 0x7) + 1`, so 1…8. The 1-bit sign-conv in the linearity decomposition is hardware-supported |
| `weight_offset` "not implemented" | ❌ **wrong — it IS implemented** | The `// FIXME not implemented` applies only to CONFIG0 `[15]`, the symmetric-vs-layerwise *selector*. The offset itself lives as `Wmin` and is applied by `__weightoffs()` (`ne16_matrixvec.cpp:182-220`) as a fake all-ones pass computing `Wmin·Σx`. `SHIFT_CYCLES = 2` is a constant, so it runs **regardless of `qw`** |

**Two new hard constraints found, both design-deciding** (`fsm.cpp:49-54`):

```c
// streamin mode is not compatible with quantization_bits != 32 at the moment. sorry!
assert(!(_this->streamin && _this->quantization_bits!=32));

// padding is not compatible with FS=1. sorry!
// assert(...)      <-- NOTE: COMMENTED OUT -> silent wrong results, not a trap
```

Consequences, both recorded in the plan §3.1/§3.1.1:
1. A decomposed (`k>1`) conv chain must be **int32** throughout — no dispatch in it may
   requantise, so the cluster requantises. A **`k==1`** conv keeps `RequantizedConv` fused and
   NE16 requantises natively (verified empirically in STEP 2a below).
2. Padding must be expressed as **per-tap input slice offsets**, never as NE16 padding.

### 2026-09-10 — Phase-0 measurement: does NE16 claim anything today?

Script: `DeeployTest/experiments/deliverable/exp16_NE16_GAP9/_ne16_canexecute_probe.py`

```
speechnet_qzo12_train                  0/5 accepted
  decline k=[1,4]/[1,16]/[1,8]/[7,1]/[7,1]
          weight=Variable<-RQSPerturbRademacher   <- not gs.Constant; kernel_shape not [1,1]/[3,3]
speechnet_qzo12_update                 0/0 (no convs)
exp12 qinfer/network.onnx (inference)  0/5 — weight=Constant; kernel_shape is the ONLY reason
```

The inference fixture therefore isolates the kernel-shape blocker from the perturbed-weight one.
That is why STEP 2a is built from it rather than from a toy model.

### 2026-09-10 — STEP 1: make it compile ✅

Files changed (all on `feat/GAP9_w_NE16`):

| path | change |
|---|---|
| `Deeploy/Targets/NE16/**` (19 files) | **new** — vendored verbatim from PR #183 `45ca06c`; imports clean with zero source changes |
| `Deeploy/Targets/NE16/Platform.py` | one deviation: uses the plain `GAP9ClusterEngine()` default include list instead of PR #183's trimmed one. Our GAP9 target never compiled the SDK's `CNN_BasicKernels_NE16.c`, so the `NE16_REG_*` redefinition the PR works around cannot occur. Upstream version kept commented above it |
| `DeeployTest/testUtils/platformMapping.py` | register `GAP9_w_NE16`; NE16 branches placed **before** the GAP9 ones — `NE16Platform` subclasses `GAP9Platform`, so a later `isinstance` check would silently swallow it |
| `CMakeLists.txt` | `GAP9_w_NE16` accepted everywhere `GAP9` was (platform list, SDK setup, `common.cmake` skip, target block) |
| `DeeployTest/CMakeLists.txt` | same |
| `DeeployTest/testUtils/core/execution.py` | `image` target (MRAM `.bin` for GVSoC) also for `GAP9_w_NE16` |
| `TargetLibraries/GAP9/CMakeLists.txt` | link pulp-nnx with `USE_NE16` **only** for this platform. Note we do *not* need PR #183's `CNN_BasicKernels_NE16.c` exclusion |
| `DeeployTest/testMVPTraining.py`, `DeeployTest/testMVPOptimizer.py` | wrap in `EngineColoringDeployerWrapper` between `mapDeployer()` and `setupMemoryPlatform()` |
| `DeeployTest/testMVP.py` | same gate widened to `GAP9_w_NE16` (this is the inference codegen path) |
| `DeeployTest/deeployMezoRunner_tiled_GAP9_w_NE16.py` | **new** — MeZO/QZO runner |
| `DeeployTest/deeployRunner_tiled_gap9_w_ne16.py` | **new** — inference runner |

**Exit criterion — the inert baseline.** The *existing, unmodified* `speechnet_qzo12_*` QZO single
step, built and run as `GAP9_w_NE16`:

```
Errors: 0 out of 8   PASSED
BENCH train_cycles_lo=66813897  opt_cycles_lo=372315
ne16_nnx_dispatch in TrainingNetwork.c : 0
pulp_nn_conv_i8_i8_i8                  : 5     (all convs still on the cluster)
```

vs exp13's plain-GAP9 baseline `66819806 / 372562` — a **0.009 %** difference (the `ne16_nnx_init`
call plus codegen noise). The NE16 engine is present, initialised, linked against pulp-nnx, and
claims nothing.

> This checkpoint matters more than it looks: a graph where NE16 accepts *nothing* builds and
> passes **identically** to one where it works. Establishing inertness first is what makes every
> later delta attributable.

Log: `DeeployTest/experiments/deliverable/exp16_NE16_GAP9/logs/step1_inert_baseline.log`

Exact command:
```bash
docker exec deeploy_gap9 bash -lc '
  source /app/install/gap9-sdk/.gap9-venv/bin/activate
  source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
  export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
  rm -rf /app/Deeploy/DeeployTest/TEST_GAP9_W_NE16
  cd /app/Deeploy/DeeployTest
  python3 deeployMezoRunner_tiled_GAP9_w_NE16.py \
    -t Tests/Models/Training/SpeechNet/speechnet_qzo12_train \
    --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo12_update \
    --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
    --l1 110000 --l2 1500000 --defaultMemLevel L2 -D BN_FROZEN_STATS=ON'
```
(`--l1 110000`, not 128000 — GAP9 usable L1 is ~93 KB before the stacks are moved to L2. `-D
BN_FROZEN_STATS=ON` is mandatory. Build dir is `TEST_GAP9_W_NE16`.)

### 2026-09-10 — STEP 2a: make it work (plumbing) ✅

**Fixture** — a single-layer, real-data **block-1** model that NE16 accepts as-is.
Builder: `DeeployTest/experiments/deliverable/exp16_NE16_GAP9/build_block1_pw_fixture.py`
Output:  `DeeployTest/Tests/Models/NE16/speechnet_b1_pw_rq/{network.onnx,inputs.npz,outputs.npz}`

Taken from the exp12 **inference** fixture
(`experiments/deliverable/exp12_QZO_clean_round_1/qinfer/network.onnx`), which has the real int8
weights as `gs.Constant`. Only the kernel is reduced `1×16 → 1×1` (tap 0). Everything else is
real: `Cin=8`, `Cout=16`, `14×87`, the real per-channel `mul`/`add`, `div=65536`, and a real
activation extracted from evaluation window 0 through the same host executor the QZO pipeline
uses (`onnx4deeploy.utils.onnx_node_implementations.run_onnx_graph`).

Block 1 is the right layer: **2,523,136 of SpeechNet's 3,711,680 conv MACs (68 %)**.

Two blockers are absent by construction, and one was **confirmed by measurement**:
the extracted activation has `min=0, max=73`, i.e. block-1 activations really are non-negative
(fed by `MaxPool ← Relu`), so the phantom signed-input bit 26 cannot bite here. The builder
asserts this rather than assuming it.

**Result — GVSoC, `gap9.evk`:**

| build | conv executed by | cycles | errors |
|---|---|---|---|
| `GAP9` (8-core cluster, `pulp_nn_conv_i8_i8_i8`) | cluster | **148,436** | 0 / 19,488 |
| `GAP9_w_NE16` | **NE16** | **44,607** | 0 / 19,488 |

**3.33×, bit-exact.** Generated `Network.c` shows `1 × ne16_nnx_dispatch(` and **0 ×
`pulp_nn_conv`** — a real dispatch, not a silent fallback.

> **Do not read 3.33× as "block 1 is 3.33× faster".** This fixture is the `1×1` stand-in:
> 155,904 MAC vs block 1's real 2,523,136 — **6.2 % of the layer**, one tap of a 16-tap filter,
> with padding dropped and output width 87 instead of 88. NE16 reaches only 3.50 MAC/cycle here
> (~2 % of its 162 peak) because `Cin=8` fills half of `TP_IN=16`, `Cout=16` fills half of
> `TP_OUT=32`, and 156 k MAC is too little to amortise job setup. The real `1×16` changes both
> terms in opposite directions — 16× the MACs (better amortisation) but 16 dispatches each
> paying setup again, plus an int32 intermediate and a cluster requant pass. **Which wins is an
> open measurement, not an extrapolation.**

The emitted `conf0 = 43032663 = 0x0290a057`, decoded against `ne16_regfile.cpp`:

```
[2:0] qw-1=7 -> qw=8      [6:5] filter mode = 2 = 1x1     [14] streamin = 0
[4]   outquant = 1        [20:16] quant r-shift = 16 = log2(65536) = log2(div)
[15]  weight offset = 1   [22:21] quant bits = 8          [23] norect = 1 (signed out)
[25]  norm bias = 1       [26] phantom signed bit = 0
```

> **This empirically settles the `k==1` half of plan §3.1.1:** `outquant=1` and `quant bits=8`
> mean **NE16 requantised the output itself** — `RequantizedConv` stayed fused, one dispatch,
> int8 out. The int32-output/cluster-requant split is required *only* for the decomposed `k>1`
> chain, where streamin forces 32-bit.

Logs:
`.../exp16_NE16_GAP9/logs/step2a_block1_pw.log` (NE16) and
`.../exp16_NE16_GAP9/logs/step2a_block1_pw_cluster_AB.log` (cluster A/B).

Exact command:
```bash
docker exec deeploy_gap9 bash -lc '
  source /app/install/gap9-sdk/.gap9-venv/bin/activate
  source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
  export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
  rm -rf /app/Deeploy/DeeployTest/TEST_GAP9_W_NE16
  cd /app/Deeploy/DeeployTest
  python3 deeployRunner_tiled_gap9_w_ne16.py -t Tests/Models/NE16/speechnet_b1_pw_rq \
    --l1 110000 --l2 1500000 --defaultMemLevel L2 --cores 8'
```
Rebuild the fixture (in **agitated_hugle**, the only container with Onnx4Deeploy importable):
```bash
docker exec agitated_hugle bash -lc 'cd /app && PYTHONPATH=/app/Onnx4Deeploy python3 \
  TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/build_block1_pw_fixture.py'
```

### Problems hit and fixed this session

1. **`AttributeError: 'int' object has no attribute 'values'`** in
   `Deeploy/Targets/PULPOpen/TopologyOptimizationPasses/Passes.py:171` during lowering.
   Cause: my fixture builder rebuilt the `RequantShift` attributes as plain ints.
   `PULPConvRequantMergePass` does `int(np.log2(rqs.attrs['div'].values))`, i.e. `div` must be a
   **TENSOR-valued** attribute — which is exactly how the QZO exporter emits it
   (`Onnx4Deeploy/onnx4deeploy/transform/qzo_weight_integerize.py`, `make_attribute("div",
   numpy_helper.from_array(...))`). Fix: copy the source node's raw `AttributeProto`s verbatim.
2. **`EngineColoringDeployerWrapper` inserted at the wrong site** in `testMVPOptimizer.py` by a
   regex that matched inside the argparse block. Syntactically valid, semantically wrong;
   `py_compile` did not catch it. Fixed by anchoring on the `# 5. Set up memory hierarchy.`
   comment. Lesson: verify *placement*, not just that it compiles.
3. `set -e` in the docker exec wrapper aborted the run right after sourcing the GAP9 config
   (exit 128) with no diagnostic. Dropped it.

### Known gaps / deliberately not done

* `--enable-3x3` is **not** plumbed through `deeployRunner.py` / `generateNetwork.py`
  (PR #183 does this). Not needed for 1×1, which `canExecute` accepts unconditionally. Add it
  when 3×3 or the decomposition needs it.
* `Deeploy/Targets/NE16/OptimizationPasses/MemoryLevelAnnotationPasses.py` and
  `Templates/AllocateTemplate.py` are dead code in PR #183 (never imported); the
  `WeightMemory_SRAM` / `use_wmem` path is unreachable. Left as-is.
* PR #183's `TargetLibraries/GAP9/{inc,src}/ne16_utils.{h,c}` (the `ne16_int8_to_uint8` v4s
  kernel) is **not** vendored yet — it is needed only for the signed-activation fix, i.e. block 0.
* The stride-2 border-extent bug in PR #183's `getCounters` is unfixed upstream. SpeechNet is all
  stride 1, so do **not** pass `--enableStrides`.

### Next: STEP 2b — block 1 at its real `1×16`

Needs BLOCKER 1, in four pieces (plan §5, 2b):
1. topology pass `Conv(1×k) → k × Conv(1×1)`, NE16-coloured only, padding as per-tap input slice
   offsets;
2. `streamin` (bit 14) for dispatches 2..k — `NE16_FLAG_STREAMIN` already exists in
   `TargetLibraries/third_party/pulp-nnx/ne16/hal/ne16_task_defs.h:79`; the template writes
   `conf0` directly, so no HAL change;
3. int32 output on the chain, `RequantShift` kept unmerged for those convs;
4. tile constraint keeping the output tile resident across the `k` dispatches.

Ordering trap to respect: engine coloring calls `canExecute` *before* the rewrite, so NE16 must
claim the original `1×k` node (an `isDecomposable1xK` predicate) and let `NE16OptimizationPass`
do the rewrite — and that rewrite must run **before** `NE16AdjustWeightMemoryLayoutPass`, or the
bit-serial encoder packs the `1×16` weight instead of the `k` per-tap `1×1` slices.


---

## 2026-09-11 — Session 2: STEP 2b — block-1 `1×16` on NE16, bit-exact

**Experiment:** `DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16a_PW_single_layer/`
(`Plan.md`, `Findings.md`, `results/results.json`, `fixture/`, `logs/step1..17*.log`)

### Result

SpeechNet **block 1's real `1×16` convolution**, from the **training** graph (weight is a runtime
tensor, not a constant), running on NE16 as **16 pointwise dispatches accumulating via
`streamin`** — **bit-exact**.

| fixture | engine | taps | NE16 dispatches | streamin | errors | cycles |
|---|---|---|---|---|---|---|
| `b1_1x2_ne16_s` | NE16 | 2 | 2 | 1 | **0 / 1600** | 100,704 |
| **`b1_1x16_ne16_s`** | **NE16** | **16** | **16** | **15** | **0 / 1600** | 154,110 |
| `b1_ref_1x2_s` | cluster | 2 | 0 | — | 0 / 1600 | 17,657 |
| `b1_ref_1x16_s` | cluster | 16 | 0 | — | 0 / 1600 | 35,715 |

`conf0` tap0 `4227143` vs tap_j `4243527` — difference **exactly `0x4000` = `NE16_FLAG_STREAMIN`**;
filter mode `[6:5]=2` (1×1), quant bits 32, outquant 0. 0 cluster convs.

**Performance is NOT there yet — NE16 is 4.31× slower than the cluster here** (154,110 vs 35,715).
Expected on a `4×24` cropped extent where per-dispatch setup dominates and `Cin=8` half-fills
`TP_IN=16`. Performance is STEP 4; correctness was the goal.

### Two design decisions that differ from the written plan

1. **Template-level, not graph-level.** The plan said "topology pass `Conv(1×k)` → `k × Conv(1×1)`".
   That route needs `Pad` and `Slice` bindings GAP9 lacks for integer data, and inflates the graph
   to 49 nodes. Instead the graph keeps **one** `Conv(1×K)` node and the K dispatches are emitted
   by the template, with tap `j` as an `infeat_addr` offset — which is what "per-tap input slice
   offsets" meant.
2. **`streamin` immediately, not an `Add` chain.** `Plan.md §4.1` wanted `Add` first for debug
   separation; it was abandoned once it proved to introduce *more* new blockers than it avoided.
   The arithmetic was still validated on the host first (bit-exact, 19,712 elems).

### The enabler for variable weights: `weight_offset = -128`, fixed

PR #183 computes `weight_offset = values.min()` at compile time — impossible for a weight that
changes every step. Every int8 weight satisfies `w ∈ [-128,127] ⟹ w+128 ∈ [0,255]`, so `-128` is
valid for **any** weight and never needs recomputing. The host encodes `w+128` with Deeploy's own
`_weightEncode` and ships it as a graph input. Arithmetically free (`Wmin` costs one shift cycle
regardless of value).

### Files changed — all Python, **no NE16 ISA change** (board-safe)

| file | change |
|---|---|
| `Deeploy/Targets/NE16/Engine.py` | `_weightAcceptable` (runtime weight via `ne16_weight_preencoded`); `is1xKConv` + `enable1xK` flag (**off by default**); `NE161xKConv2DMapper` first in `NE16Mapping['Conv']` |
| `Deeploy/Targets/NE16/Parsers.py` | `NE161xKConv2DParser` |
| `Deeploy/Targets/NE16/Templates/Conv1xKTemplate.py` | **new** — K dispatches, per-tap pointer offsets, `streamin` for j>0 |
| `Deeploy/Targets/NE16/TileConstraints/NE161xKConstraint.py` | **new** — single-tile policy (streamin residency) |
| `Deeploy/Targets/NE16/Bindings.py`, `Tiler.py` | 2-input / int32-output binding + tiling-ready binding |
| `Deeploy/Targets/PULPOpen/TopologyOptimizationPasses/Passes.py` | `_merge_conv_rq_fun`: do **not** fuse a conv carrying `ne16_taps` |
| `Deeploy/CommonExtensions/.../LoweringOptimizationPasses.py` | `_NCHWtoNHWC_fun`: `spatialDims` from `kernel_shape`, and **skip permuting** a pre-encoded weight |
| `DeeployTest/testMVP.py`, `testUtils/deeployRunner.py`, `deeployRunner_tiled_gap9_w_ne16.py` | `--enable-1xk` threaded end to end |

Every change is gated on an NE16-only attribute or an off-by-default flag.

### Reproduce

```bash
# fixture (agitated_hugle)
docker exec agitated_hugle bash -lc 'cd /app && PYTHONPATH=/app/Onnx4Deeploy python3 \
  TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16a_PW_single_layer/build_fixtures.py \
  --taps 16 --crop-h 4 --crop-w 24 --suffix _s'

# device (deeploy_gap9)
docker exec deeploy_gap9 bash -lc '
  source /app/install/gap9-sdk/.gap9-venv/bin/activate
  source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
  export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9 ; rm -rf /app/Deeploy/DeeployTest/TEST_GAP9_W_NE16
  cd /app/Deeploy/DeeployTest
  python3 deeployRunner_tiled_gap9_w_ne16.py -t Tests/Models/NE16/b1_1x16_ne16_s --enable-1xk \
    --l1 110000 --l2 1500000 --defaultMemLevel L2 --cores 8'
```

### Known limitations (both recorded in `exp16a/Findings.md §4, §9`)

* **Spatial crop `4×24`.** The single-tile constraint sidesteps the `1×K` **halo** (an output tile
  of `Wt` columns needs `Wt + K-1` input columns). At full `14×88` the int32 output is 78,848 B and, with the two layout
  transposes, overflows GAP9's ~110 KB L1 (`Allocation failed for allocator 2`). Lifting this needs
  only the halo relation in the tile constraint — the same one `NE16DenseConv2DTileConstraint`
  already implements for 3×3. **Correction to an earlier note:** streamin does NOT require the
  output tile to persist across tiling iterations; all K dispatches run inside ONE `TILING_I`
  iteration, so residency is automatic (verified in the generated `Network.c`).
* **Blocker 1b** still open: the host supplies the already-perturbed weight; the QZO loop computes
  it on device. Next = the linearity decomposition or a device-side encode kernel.

### Lesson worth keeping

A long Deeploy codegen is a **symptom to shrink, not a cost to wait out**. The first `K=16`
attempt "hung" for 30+ minutes; it was exponential backtracking over an *infeasible* binding
problem. At `K=2` the same failure surfaced in **3 seconds** — and nine further bugs after it,
each with an exact diagnosis. Ten bugs total, listed in `exp16a/Findings.md §7`.


---

## 2026-09-11 — Session 2 (cont.): full extent reached, crop removed

Block 1's real `1×16` conv now runs on NE16 at its **true `14×87` extent** — `0 / 19712` errors,
no spatial crop, **zero layout transposes**.

| fixture | engine | extent | dispatches | transposes | conv tiles | errors | cycles |
|---|---|---|---|---|---|---|---|
| **`b1_1x16_ne16_nhwc`** | **NE16** | **FULL 14×87** | **16** | **0** | **4** | **0 / 19712** | 1,567,096 |
| `b1_ref_1x16` | cluster | FULL 14×87 | 0 | — | — | 0 / 19712 | 337,657 |

Three fixes, in order:

1. **Halo tile constraint** — `Wout = Win − (K−1)`, input cubes with the `K−1` overlap, per-tile
   strides + subtile counters. (Also corrected a wrong claim that streamin forbids tiling: all K
   dispatches run inside ONE `TILING_I` iteration, so residency is automatic.)
2. **NHWC-native fixture** (`--nhwc`) — un-merging the RequantShift had left the int32 conv output
   materialised in **both** layouts (`2 × 78,848 B`) just to meet an NCHW graph boundary. Going
   channels-last removes both transposes: arena `108,416 → 98,688 B`. Required one Deeploy fix:
   `RequantShiftLayer.computeShapes` hardcoded `channel_dim = inputShapes[0][1]` while already
   receiving `channels_first`, so a channels-last standalone RequantShift always failed with
   `Could not broadcast rqs_mul_tensor from (16,) to [1, 14]`.
3. **`--l1 92000`, not 110000.** `110000` is **MeZO-harness-only** — it relies on
   `pi_cluster_task_stacks()` moving the cluster slave stacks to L2 in `deeploymezotest.c`. The
   **inference** harness `deeploytest.c` has no such relocation, so ~30 KB of L1 is still stacks
   and only **98,256 B** is usable. Worth remembering: the WorkLog's `--l1 110000` applies to the
   ZO/training runners, *not* to `deeployRunner_tiled_gap9*`.

**Performance: NE16 is 4.64× slower than the cluster** (1,567,096 vs 337,657). Cause is
work-per-dispatch: `Cin = 8` half-fills `TP_IN = 16`, and 16 taps × 4 tiles = **64 NE16 jobs**
each paying fixed setup, against one `pulp_nn_conv` call. Channel folding (single `Cin = 128`
dispatch) is the structural answer — STEP 4, unmeasured.


---

## 2026-09-11 — Session 3: exp16b, 3×3-dense chunks — bit-exact, and the cycle model was wrong

**Experiment:** `DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16b_Dense_single_layer/`
(`Plan.md`, `Findings.md`, `results/results.json`, `fixture/`, `logs/step1..step9*.log`)

Ran block 1's `1×16` as **6 dense 3×3 dispatches** (each a 3×3 kernel with only the middle row
populated, `infeat_addr += 3c·ch_im_in`, native H padding 1/1, streamin) instead of exp16a's 16
pointwise ones, on identical data.

| decomposition | dispatches | errors | cycles |
|---|---|---|---|
| **dense 3×3 chunks** | **6** | **0 / 19712** ✓ | **1,364,975** |
| all-pointwise (exp16a) | 16 | **0 / 19712** ✓ | 1,567,096 |
| cluster `pulp_nn_conv` | — | 0 / 19712 ✓ | 337,657 |

Both NE16 decompositions of the block-1 layer are now **bit-exact at full 14×87 extent**.

### The finding: dispatch count dominates, not MAC utilisation

**Dense 3×3 is 1.15× FASTER than pointwise — the cycle model predicted ~3× SLOWER.**

`docs/TRAIN_GAP9_NE16/03-qzo-ne16-plan.md §7` rejected masked-3×3 on the argument that 3×3 mode
spends its 9 row-slots on spatial taps and therefore needs 8 bitplane passes (`mv_qw_lim = qw`),
where 1×1 mode spends them on bitplanes and finishes in one — `6 × 8 = 48` cycle-units against
`16 × 1 = 16`.

That argument assumed **the MAC array is the bottleneck. At this problem size it is not.** Six
dispatches instead of 16 removes 10 job setups and 10 of the 15 int32 `streamin` round-trips
through L1. With `Cin = 8` (half of `TP_IN`) and an int32 intermediate 4× the int8 one, the layer
is **DMA/setup-bound**.

**Consequence for STEP 4:** the lever is *fewer, larger dispatches*, not tap-packing. That points
at **channel folding** (im2col to a single `Cin = 128` pointwise conv: one dispatch, `TP_IN` fully
packed, no streamin) rather than at 3×3 tricks. Price of the halfway version: **13,824 B of
weights vs 4,096 B** (3.4×), since 6 of every 9 tap slots are zeros.

### Correctness: achieved — the bug was geometry, not the datapath

Earlier in this session the dense variant failed (`14328/19712` full extent; `431/1536` on a
single-chunk K=3 isolation fixture, every diff **±1 LSB** on outputs ranging only `[-2,3]`).

**Root cause** — `Deeploy/Targets/NE16/TileConstraints/NE161xKConstraint.py`,
`serializeTilingSolution`: it unconditionally emitted the **pointwise** subtile counters with zero
padding for *both* variants:

```python
replacements["input_addr_offset"].append(0)
counters = NE162DPWConvTemplate.getCounters(inCSz, hSz, wSz, cSz, 0, 0, operatorRepresentation)
```

NE16 retires output in 3×3 subtiles and needs the **border** subtile's *input* extent declared in
`bHi`/`bWi`. For a 1×1 job that equals the output border; for a 3×3 job it is
`bHi = height_out_border + 2 − padding_bottom` (`NE162DDenseConvTemplate.getCounters`), the `+2`
being the receptive field. With the pointwise formula the border subtile read a window two pixels
short in each direction, dropping edge taps — small int32 deficits that requantise to ±1 LSB.

**Fix:** branch on `isDense3x3 = 'ne16_chunks' in operatorRepresentation`, using
`NE162DDenseConvTemplate.getCounters` with the real `padding_y_bottom` / `padding_x_right` and a
non-zero `input_addr_offset` from `getInputAddrOffset(...)` for the dense path.

`b1_1x3_dense3x3_k3` → **0/1536** (93,534 cycles); `b1_1x16_dense3x3_d33` → **0/19712**
(1,364,975 cycles, 6 dispatches, 0 transposes, 0 cluster convs).

**Wrong hypotheses, recorded so they are not retried:** zero-weight rows failing to cancel (they
cancel exactly — `w_u = 128`, `Wmin = −128`); and `filter_mask = (1<<24)|(1<<8)` to skip them,
which made it *worse* (431 → 557) and was reverted (`ne16_filter_mask` stays 0). The tell that it
was geometry, not arithmetic, was that the errors were **position-dependent** — clustered at
subtile borders — which no datapath explanation accounts for.

The pinning of the non-tapped axis when `ne16_chunks` is set is **still required** and kept: the
constraint models neither a per-tile vertical halo nor per-tile padding. SpeechNet's non-tapped
extent is 14 rows, so not tiling it is cheap.

### exp16a regression

Re-verified after all exp16b changes: **still `0 / 19712`, 1,567,096 cycles**. Unaffected.

### Files

`Targets/NE16/Templates/Conv3x3ChunkTemplate.py` (new), `NE163x3ChunkConv2DParser`, its binding /
tiling-ready binding / mapper, `ne16_halo` generalised in the tile constraint (`taps-1` pointwise,
`3*(chunks-1)+2` for chunks), the non-tapped-axis pin, the variant-specific counters/padding fix
above, and `--dense3x3` in `exp16a/build_fixtures.py`. All Python; **no NE16 ISA change** — the
templates use only fields `ne16_task_t` already has, so this runs on real GAP9 silicon.

### Next: exp16c — blocker 1b and STEP 3

exp16b's gate is satisfied. `exp16c_PW_single_layer` takes on **blocker 1b** (on-device
perturbation — the host still supplies a pre-perturbed, pre-encoded weight) and **STEP 3** (the
remaining four SpeechNet convs; block 0 additionally needs the signed-activation fix).

---

## 2026-09-11 — Session 4: exp16c phase 1 — blocker 1b's hard half closed

**Experiment:** `DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16c_PW_single_layer/`
(`Plan.md`, `Findings.md`, `results/results.json`, `fixture/`, `logs/`)

exp16a and exp16b both let the **host** bit-serial encode the conv weight, which only works while
it is a `gs.Constant`. In QZO the weight is an `RQSPerturbRademacher` output, recomputed on device
twice per ZO step. exp16c phase 1 moves the encoding onto the device.

| fixture | weight encoded by | dispatches | errors | cycles |
|---|---|---|---|---|
| `b1_1x16_devenc_dev` | **device** | 16 | **0 / 19712** ✓ | 1,593,285 |
| `b1_1x16_ne16_nhwc` (exp16a) | host | 16 | 0 / 19712 ✓ | 1,567,096 |

### Encode cost: 26,189 cycles = 1.64 % — and it settles a design question

That 1.64 % covers the L2→L1 DMA of the 2,048 B raw int8 weight **and** the bit-serial encode of
4,096 B across 8 cores.

`docs/TRAIN_GAP9_NE16/03-qzo-ne16-plan.md §3.2` recommended the **linearity decomposition**
(`conv(w+δz,x) = conv(w,x) + δ·conv(z,x)`, sign conv at `qw=1`) whose entire purpose is to avoid an
on-device 8-bit re-encode. It would pay for that 1.64 % by **doubling the dispatch count**
(16 → 32) — the metric exp16b showed dominates this layer — plus a new per-channel int32
scale-and-accumulate kernel, two perturbation constants (`δ⁺ ≠ δ⁻`), and per-step sign packing
anyway. **Withdrawn on the measurement**, not merely deprioritised.

### Why the encoder is only a bit transpose

`_weightEncode` (`Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py:24`) does two things.
Fixing `weight_offset = −128` (exp16a's choice) removes the first: `values.min()` would change
every step and would have to be recomputed, cross-core reduced, and pushed into the conv's
`weight_offset_factor`. At −128 the offset step is exactly `w_u = w ^ 0x80`. What remains is a
**data-independent permutation** — output byte `b*2 + k` of each `(row, cinMajor)` pair collects
bit `b` of input lanes `8k..8k+7`, LSB-first. Rows (`taps*cout`) are independent → chunked across
the 8 cluster cores.

### Method note — the reason this went green on the FIRST device run

The kernel was compiled **natively on the host** and compared against Deeploy's own `_weightEncode`
via `ctypes`, for all five SpeechNet conv shapes plus an odd `cin=20` case, single-core and
8-core-chunked: **MATCH everywhere**. The bug class that cost two sessions in exp16a/exp16b was
priced out in a 3-second test. Two results fall out that de-risk later phases:

* `cinMajor > 1` and non-multiple-of-16 `cin` already work → block 4 (`cin=32`) needs no new code;
* `K×1` needs no new code → in NCHW the tap is the last index for both `1×K` (H=1,W=K) and `K×1`
  (H=K,W=1), so `src[(co*cin+ci)*taps + j]` covers both. Blocks 3 and 4 reuse the encoder.

### Files — all plain C / Python, **no NE16 ISA change**

`TargetLibraries/GAP9/src/NE16WeightEncode.c` (new kernel), its prototype in
`TargetLibraries/GAP9/inc/DeeployGAP9Math.h` (deliberately NOT in a pulp-nnx header — that tree
stays pristine), `Deeploy/Targets/NE16/WeightEncode.py` (new: parser / checker / template /
binding / layer), `Deeploy/Targets/NE16/TileConstraints/NE16WeightEncodeConstraint.py` (new:
pins both tensors full — the op is deliberately untiled, the largest encoded weight is 7 KB),
`Deeploy/Targets/NE16/Tiler.py`, `Deeploy/Targets/GAP9/Platform.py` (the op runs on the CLUSTER
cores, so it is registered in `GAP9Mapping`, not claimed by NE16), and `--device-encode` in
`exp16a/build_fixtures.py`.

### Next

Phase 2: replace the graph-input weight with a real `RQSPerturbRademacher` producer and run L⁺/L⁻.
Then STEP 3 — blocks 2/3/4, then block 0 with the signed-activation fix.

### 2026-09-11 — Session 4 (cont.): exp16c phase 2 — BLOCKER 1b CLOSED

The weight now arrives **unperturbed**; the device does perturbation *and* encoding.

| fixture | weight path | errors | cycles |
|---|---|---|---|
| `b1_1x16_pertenc_lp` | RQSPerturbRademacher (L⁺) → NE16WeightEncode → NE16 Conv | **0 / 19712** ✓ | 1,608,939 |
| `b1_1x16_pertenc_ln` | same, L⁻ | **0 / 19712** ✓ | 1,608,945 |

Generated C: `perturb=1`, `encode=1`, `dispatches=16`, `transposes=0`, `cluster convs=0`.

**Cost of the complete on-device QZO weight path:**

| stage | cycles | Δ |
|---|---|---|
| host-perturbed, host-encoded (exp16a) | 1,567,096 | — |
| + device encode | 1,593,285 | +26,189 |
| + device perturb | 1,608,939 | +15,654 |
| **total** | | **+41,843 = 2.7 %** |

**Testing L⁻ without driving the ZO runtime:** `RandomNoiseQuant.c:31` — negating the Rademacher
sign is exactly equivalent to negating the per-channel multiplier `M`. So `--neg-pmul` gives the
L⁻ golden while the device keeps the neutral `perturbation_sign = +1` default; both ZO passes are
testable under the plain inference runner. It is a real second test: **all 2048** weight elements
differ between L⁺ and L⁻ and the output range moves `[-85,60] → [-82,57]`.

**Why the RNG streams agree:** the device seed is
`(seed + perturb_seed_base) + NUM_CORES*node_id + core_id` with `node_id = attrs['idx']`. The
fixture copies the source node's `AttributeProto`s verbatim (`idx=2`, `seed=42`) and keeps the
weight's shape, so the per-core chunking — and hence the stream — matches the host reference.

**Status:** blocker 1b is closed for the pointwise route. Remaining: STEP 3 (blocks 2/3/4, then
block 0 with the signed-activation fix) and STEP 4 (NE16 is still ~4× slower than `pulp_nn_conv`).

### 2026-09-11 — Session 4 (cont.): exp16c phase 3 — STEP 3a, blocks 1–4 all bit-exact

Every non-block-0 SpeechNet conv now runs the complete on-device path
`RQSPerturbRademacher → NE16WeightEncode → NE16 Conv → RequantShift`.

| block | kernel | taps | Cin→Cout | out | disp. | errors | NE16 | cluster | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1×16 | W | 8→16 | 14×88 | 16 | **0 / 19712** ✓ | 1,608,939 | 337,999 | 4.76× |
| 2 | 1×8 | W | 16→16 | 14×23 | 8 | **0 / 5152** ✓ | 420,743 | 98,723 | 4.26× |
| 3 | 7×1 | **H** | 16→32 | 8×5 | 7 | **0 / 1280** ✓ | 183,651 | 26,835 | 6.84× |
| 4 | 7×1 | **H** | 32→32 | 2×5 | 7 | **0 / 320** ✓ | 145,990 | 30,329 | 4.81× |
| 0 | 1×4 | W | 1→8 | 14×701 | — | **blocked (BLOCKER 3)** | — | — | — |

All four passed on their **first** device run.

**The one code change it needed — the `K×1` tap stride.** `Conv1xKTemplate` stepped `infeat_addr`
by `ch_im_in * bytes`, i.e. one **pixel** along W. Blocks 3/4 are `7×1`, where one tap is one
**row**. The row stride is `dim_im_in_x_stride` (names are transposed: `ioStridesFromDimensions`
returns `(height_stride, width_stride)` and `NE161xKConstraint` assigns them to `x_stride,
y_stride` in that order) and it is a **per-tile** value, so the template references the substituted
variable rather than a constant from `alignToContext`. Verified in the generated C: block 3 emits
`+ 1 * *..._dim_im_in_x_stride_ref`, block 1 the compile-time `+ 1 * 8`.

Everything else came free: `_tapAxis` already returned 1 for `kh != 1`, the parser already admitted
`K×1`, and the device encoder already handled both axes and `cinMajor = 2` — all three validated in
the native ctypes test before any device run, which is why phase 3 cost four runs and no debugging.

**Performance: NE16 is 4.3×–6.8× SLOWER than `pulp_nn_conv` on every block.** Correctness of the
NE16 path is established; the gap is STEP 4, and exp16b already names the lever (fewer, larger
dispatches → channel folding). The NE16 figures include the on-device perturb+encode the cluster
reference does not do — 2.7 % on block 1, so that is not the explanation.

**New file:** `exp16c_PW_single_layer/build_fixtures.py`, generalised over blocks and both tap
axes. `exp16a/build_fixtures.py` deliberately left as-is (hardwired to block 1 / `1×K`, still
carrying exp16a's abandoned Slice/Add exploration) so exp16a and exp16b stay reproducible.

**Block 0 is genuinely blocked:** its activation range is `[-66, 127]` — actually signed, so
BLOCKER 3 is real, not a conservative assertion. Phase 4 next.

### 2026-09-11 — Session 4 (cont.): exp16c phase 4 — BLOCKER 3, **STEP 3 COMPLETE**

| block | kernel | Cin→Cout | out | disp. | errors | NE16 | cluster | ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 1×4 | 1→8 | 14×701 | 4 | **0 / 78512** ✓ | 5,517,302 | 732,178 | 7.54× |

**All five SpeechNet convs now run bit-exactly on NE16**, each with the complete on-device QZO
weight path. STEP 3 ("make it correct") is done.

**The fix.** NE16 reads activations as unsigned and CONFIG0 bit 26 (PR #183's `input_signed`) is
undecoded by the hardware, so there is no register to flip. Block 0's activation is genuinely
signed (`[-66,127]`), so: feed `x_u = x + 128` as **uint8** (`NE16PWConv2DBindings` already admits
a `uint8_t` `data_in` — no binding work), then remove the induced per-output-channel term, since
`Σw·x = Σw·x_u − 128·Σw`.

**Where it is applied is the subtle part.** RequantShift computes `(acc·mul + add) >> log2(div)`,
so `add` lands *after* the multiply: correcting the accumulator by `−128·Σw` is equivalent to
correcting `add` by `−128·Σw·mul`, not by `−128·Σw`.

**Why on device:** `Σw` is not a host constant — the weight is a per-step `RQSPerturbRademacher`
output. New kernel `NE16SignedInputBias_i32` (in `TargetLibraries/GAP9/src/NE16WeightEncode.c`)
recomputes it from the same perturbed weight the encoder consumes, parallelised over `cout`; new op
`NE16SignedInputBias` in `Deeploy/Targets/NE16/WeightEncode.py` with its own untiled constraint.
Range checked against the real tensors *first*: `|128·Σw·mul| ≤ 128·508·454 ≈ 3.0e7`, well inside
int32.

**Pre-existing Deeploy bug surfaced and fixed.**
`UniformRequantShiftParser.parseNode` (`Deeploy/Targets/Generic/Parsers.py:1478`) called
`node.inputs[1:3].values` unconditionally. `.values` exists only on a `gs.Constant`, so a
RequantShift with a runtime `add` raised `AttributeError: 'Variable' object has no attribute
'values'` during parsing and **aborted the whole deployment** rather than declining the node. A
`parseNode` must decline, never raise. Guarded with `isinstance(..., gs.Constant)`. **This would
have bitten the real QZO graph too**, where `bias_rqsadd_pert` is a runtime tensor.
Regression: block 1 re-run, `0/19712`, 1,608,939 cycles — unchanged.

**Why block 0 is the slowest (7.5×):** `Cin = 1` uses one lane of `TP_IN = 16`, so 15/16 of the
array is idle on each of its 4 dispatches, over the network's largest spatial extent (14×701).

### Status after session 4

| step | state |
|---|---|
| STEP 1 compile | ✅ |
| STEP 2 work (block 1) | ✅ bit-exact, two decompositions |
| blocker 1b (on-device perturb + encode) | ✅ closed, 2.7 % overhead |
| BLOCKER 3 (signed activations) | ✅ closed |
| **STEP 3 correct (all 5 convs)** | ✅ **all bit-exact** |
| STEP 4 optimise | ❌ open — NE16 is 4.3×–7.5× SLOWER than `pulp_nn_conv` on every block |

STEP 4 is now the only thing between this and a useful accelerator. exp16b named the lever
(dispatch count, not MAC utilisation) and block 0 illustrates the other half (`Cin` packing:
`Cin=1` wastes 15/16 of `TP_IN`).

---

## 2026-09-14 — Session 5: exploring the GAP9 SDK — the vendor had already solved this

**Trigger:** supervisor's point that GreenWaves must already handle corner cases like our
`1×K` / `K×1`. Correct on every count.

**Summaries written:** `ETH/WorkLog/GAP9_Container_Inventory.md` (what the `deeploy:gap9` image
contains) and `ETH/WorkLog/GAP9_SDK_NE16_Corner_Cases.md` (the NE16 findings, with file/line
references). This entry is the short version.

### What the container actually is

`ghcr.io/runwangdl/deeploy:gap9`, 13.7 GB, Ubuntu 22.04 x86_64. It is not a GAP9 runner — it is the
**complete GreenWaves GAP9 SDK** (3.8 GB, commit `8c42b653…`) plus toolchains for SoftHier, Snitch,
MemPool, Chimera and PULPOpen. The GAP9 SDK carries `tools/autotiler_v3` (their code generator,
including `CNN_Generators_NE16.c` 243 KB and `CNN_BasicKernels_NE16.c` 267 KB), `tools/nntool`
(their NN compiler), pmsis RTOS, and their own GVSoC with the GAP9 chip models. Two separate GVSoC
installs exist; our runs use the SDK's, not `/app/install/gvsoc`.

### Finding 1 — our exp16a decomposition **is** the vendor's own 1-D kernel

`KerConv1D_StrideS_NE16` (`CNN_BasicKernels_NE16.c:1643`) is line-for-line what
`Conv1xKTemplate.py` does: loop `subfilter_i = 0 … Fx-1`, `pIn += Tile_InFeat*subfilter_i*Dx`,
`pFilt += <one encoded tap block>*subfilter_i`, `SET_STREAMIN` for `subfilter_i > 0`, padding via
pointer arithmetic with `SetNE16_ConfigPad({0,0,0,0}, 0)`.

Independent convergence — the algorithm is validated, but it was never an open problem.

### Finding 2 — their PREFERRED route is the channel folding we filed as STEP 4

`Ker_MM_Conv1DSmallv2_NE16` (`:4681`): the 8 cores build an im2col column buffer, NE16 then runs an
ordinary **1×1** convolution with `ColBuffSize = InFeat*Fx` input channels. Two column buffers
alternate so the cores prepare job *n+1* while NE16 computes job *n*.

For SpeechNet this is close to ideal — `TP_IN = 16`:

| block | kernel | Cin | Cin·K | TP_IN groups | our disp. | folded |
|---|---|---|---|---|---|---|
| 0 | 1×4 | 1 | 4 | 1 (¼ full) | 4 | 1 |
| 1 | 1×16 | 8 | **128** | **8 exact** | 16 | 1 |
| 2 | 1×8 | 16 | **128** | **8 exact** | 8 | 1 |
| 3 | 7×1 | 16 | **112** | **7 exact** | 7 | 1 |
| 4 | 7×1 | 32 | **224** | **14 exact** | 7 | 1 |

Both levers exp16b/exp16c identified — dispatch count and `Cin` packing — at once.

Note the generator's native `KOP_CONV1D` path is gated on `Height == 1`
(`CNN_Generators_NE16.c:1169`), i.e. genuine 1-D signals. Our maps are 2-D, so for **our** shape
their answer is specifically the im2col route; the kernel matcher selects on filter dims only and
`Ker_MM_Conv1DSmallv2_NE16` is registered as `Fx = any, Fy = 1`.

Also worth knowing: if the filter spans the whole input they collapse the conv to
`CNN_LinearAct_NE16` with `InFeat*Fcx*Fcy` inputs — channel folding taken to its limit, and the
first thing `CNN_ConvolutionNE16` checks.

### Finding 3 — two concrete defects in OUR dispatch loop

This is the immediately actionable part, and it plausibly accounts for much of the 4.3–7.5× gap.

1. **We serialise where they pipeline.** NE16 has a 2-deep job queue
   (`NE16_TASK_QUEUE_SIZE (2)`). They acquire a slot, program it, `COMMIT_AND_TRIGGER`, and
   immediately acquire the next — *"already commit and trigger NE16 computation, while programming
   the next one"* — blocking only when both slots are full. Our template calls
   `ne16_nnx_resolve_wait` after **every** dispatch, and under GVSoC that reduces to
   `ne16_task_queue_empty(dev)` — a full drain. Block 1's 16 taps therefore run with **zero**
   overlap and the queue depth is never used.
2. **We rewrite the whole register file every dispatch.** `ne16_nnx_dispatch` writes all
   `sizeof(ne16_task_data_t)/4` = **24 words** each time. They write **10** in steady state
   (3 pointers + 3 remainder + 2 dim + 2 config) and guard the invariant block —
   strides, bias/scale pointers, padding, filter mask, weight offset — with `if (SubTileCount < 2)`,
   because the two job contexts only need them once. exp16b proved this layer is **setup-bound**, so
   this sits directly on the critical path.

### Finding 4 — a trick not applicable to us, recorded anyway

For a genuine 1-D signal they remap the W axis onto NE16's 3×3 output grid (`HW_SIZE = 3`,
`SetNE16_Dim(Nb_KI, Nb_KO, 1, 1)`, `PreferedHWTile = OneDInput ? 9 : 3`) to retire **9** output
pixels per dispatch instead of 3. Our maps are 2-D so we already fill the subtile.

### Finding 5 — `ne16v2` exists in the SDK but is NOT GAP9

`gvsoc/gvsoc_gap/gap/ne16v2/` decodes a far richer CONFIG0 than v1: **bit [28] `mode_signed`
(0=uint8, 1=int8)**, **bit [15] `use_rounding`**, bias/scale broadcast [27]/[26], `ki_scatter`
[11:9], matadd modes [31:30]. A signed-input bit and hardware rounding would remove **both** of our
arithmetic workarounds (BLOCKER 3's `x+128` + `add − 128·Σw·mul`, and the `+div/2` baked into the
bias).

**Verified it is not ours:** `gap/gap9/cluster.py:27` does `from gap.ne16.ne16 import Ne16` and
instantiates it at line 136; `gap/gap9/gap9.py` builds the chip from that cluster and the
`gap9_v2` efuse/ROM — and `gap9_v2` is exactly the `TARGET_NAME` our `gap9_evk_audio.sh` sets. A
tree-wide grep finds `ne16v2` referenced only inside `ne16v2.py` itself. So it is a
next-generation HWPE model, and our workarounds remain necessary. (PR #183's `input_signed` at
bit 26 matches neither model — in v2 that bit is `scale_broadcast`.)

### Revised plan for STEP 4

1. **Fix the dispatch loop first** (Finding 3) — one template, no new algorithm, aimed squarely at
   the cost exp16b measured as dominant.
2. **Then channel folding** (Finding 2), using `Ker_MM_Conv1DSmallv2_NE16` as the specification.
3. **Keep our decomposition**: it matches the vendor's own 1-D kernel and, unlike theirs, already
   handles runtime-perturbed weights — the QZO-specific part no inference SDK needs.
4. **Ignore `ne16v2` features** — not in GAP9.

### Honest framing

The decomposition was a rediscovery, not a contribution. What remains ours is the QZO machinery
around it — on-device perturbation and bit-serial encoding of a weight that changes every training
step. And the performance gap is unsurprising: a textbook dispatch loop against a library with
years of shape-specific tuning.

---

## 2026-09-14 — Session 6: exp16c_SDK_port phase 1 — pipelined dispatch

**Experiment:** `DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16c_SDK_port/`

Ported SDK **technique A** (`KerConv1D_StrideS_NE16` — *"already commit and trigger NE16
computation, while programming the next one"*): the per-tap loop now blocks only while NE16's
2-deep job queue is full, and waits for completion **once, after the whole chain**, instead of
after every dispatch.

**Safety, verified before changing anything:** `streamin` creates a RAW dependency from tap *j*'s
output to tap *j+1*'s accumulator preload. GVSoC's `fsm_end_handler` (`gap/ne16/src/fsm.cpp:88`)
decrements `job_pending`, flips `cxt_use_ptr`, and only **then** enqueues the next job;
`fsm_start_handler` sets `job_running = 1`. Queued jobs execute **one at a time, in order** — the
queue is a program-ahead buffer, not concurrency. The dependency is preserved.

| block | before | after | saved | cluster | ratio |
|---|---|---|---|---|---|
| 0 | 5,517,302 | 5,480,672 | 0.66 % | 732,178 | 7.49× |
| 1 | 1,608,939 | 1,548,078 | **3.78 %** | 337,999 | 4.58× |
| 2 | 420,743 | 398,739 | **5.23 %** | 98,723 | 4.04× |
| 3 | 183,651 | 176,503 | 3.89 % | 26,835 | 6.58× |
| 4 | 145,990 | 142,915 | 2.11 % | 30,329 | 4.71× |
| **total** | **7,876,625** | **7,746,907** | **1.65 %** | 1,226,064 | **6.32×** |

All five still **bit-exact** (`0/78512`, `0/19712`, `0/5152`, `0/1280`, `0/320`).

### This corrects my own exp16b interpretation

exp16b concluded the layer is "DMA- and **setup**-bound", and I read "setup" as job programming.
Pipelining removes almost all of that programming cost and buys only **1.65 %** — so job setup was
*not* the dominant term. exp16b's observation stands (fewer dispatches **is** faster) but the
mechanism is the **int32 `streamin` round-trips through L1**, not register writes.

Two consequences:

* **SDK technique B** (10 register writes instead of 24) is now clearly not worth its cost — it
  would mean abandoning pulp-nnx's task API for a fraction of an already-1.65 % term. Dropped.
* **SDK technique C** (im2col channel folding) matters *more* than I thought, because it removes
  the streamin traffic outright: one dispatch, no accumulator round-trips, `TP_IN` fully packed.

### Files changed

* `Deeploy/Targets/NE16/Templates/Conv1xKTemplate.py` — task declared once outside the loop and
  reassigned per tap; `ne16_nnx_resolve_wait` moved out of the loop.
* `Deeploy/Targets/NE16/Templates/Conv3x3ChunkTemplate.py` — same, per chunk.

### Next

Phase 2: an automatic `1×K` → NE16 graph pass, so the **real** QZO graph compiles without a fixture
builder hand-authoring the NE16 attributes. That, not performance, is the blocker for running full
SpeechNet QZO on NE16.

### 2026-09-14 — Session 6 (cont.): exp16c_SDK_port phase 2 — automatic `1×K` → NE16 rewrite

Until now the NE16 form of a `1×K` conv — per-tap stacked bit-serial weight, `ne16_taps` /
`weight_offset` / `ne16_weight_preencoded`, and the `NE16WeightEncode` node — was **authored by
hand in the fixture builder**. The real QZO graph has none of it, so nothing could compile without
fixture surgery. `NE16Prepare1xKPass` now does the rewrite.

**Verified on a PLAIN graph** — ordinary NCHW `Conv`, runtime weight from `RQSPerturbRademacher`,
no NE16 attributes at all:

| fixture | errors | cycles | dispatches | device encodes | cluster convs |
|---|---|---|---|---|---|
| `b3_plain` (7×1) | **0 / 1280** ✓ | 180,585 | 7 | 1 | 0 |
| `b4_plain` (7×1) | **0 / 320** ✓ | 146,409 | 7 | 1 | 0 |

(Slightly above the hand-authored NHWC fixtures — 176,503 / 142,915 — because the plain graph is
NCHW and pays for the layout Transposes the NHWC-native fixtures avoided.)

#### Two ordering constraints that decide where the pass runs

1. **After engine coloring, before `PULPNCHWtoNHWCPass`.** That pass permutes every rank-4 conv
   input and takes the spatial rank from the *weight's* rank — both wrong for a bit-serial weight.
   `_NCHWtoNHWC_fun` skips a weight marked `ne16_weight_preencoded`, but only if the mark is there
   by then. `NE16Deployer` therefore **inserts at index 1**, right after the `EngineColoringPass`
   that `EngineColoringDeployer` places at index 0 — appending (as the other NE16 passes do) would
   put it after all coloring *and* after NHWC.
2. **The node must still be a `Conv`.** The existing `ne16_taps` guard in `PULPConvRequantMergePass`
   fires correctly, because index 1 is before every lowering pass.

#### Three things the pass has to do that the fixtures did by hand

* **Relax the engine gate.** `is1xKConv` required a `gs.Constant` weight via `_weightAcceptable`.
  Coloring runs *before* the pass, so a runtime weight was rejected and the rewrite never saw the
  node. The 1×K path re-encodes either way, so the requirement is dropped there (still behind
  `enable1xK`, so PR #183 is untouched).
* **Tag the RequantShift `channels_first = 1`.** It is deliberately not merged, so it sits
  *outside* the NHWC region and consumes the NCHW tensor the layout pass transposes back. Without
  the tag, `RequantShiftLayer.computeShapes` reads the channel count from the last axis:
  `Could not broadcast rqs_mul_tensor from (32,) to [1, 5]`.
* **Decide signedness structurally.** Every `Quant` in the QZO graph declares `signed = 1`, so the
  type says nothing. Walking the producer chain does: blocks 1–4 reach a `Relu` through
  `Quant ← MaxPool`, block 0 reaches the graph input. A conv that can be negative is **handed back
  to the cluster**, never silently miscomputed. `ne16_unsigned_input=1` is an explicit override for
  graphs where the chain cannot show it — needed by the single-layer fixtures, whose activation is
  a bare graph input, and where an int8 `Relu` cannot be spliced in to express it because GAP9's
  only `Relu` binding is fp32 (`Targets/GAP9/Bindings.py:381`).

#### A silent-wrongness hazard closed

The pass now **refuses padded convolutions**. NE16's 1×1 mode cannot pad, and its guard in
`gvsoc fsm.cpp:53` is commented out — a padded 1×1 job returns wrong results rather than trapping.
Every experiment so far pre-padded the activation *in the fixture data*; a real graph carries
padding on the node. Without this guard the pass would have silently miscomputed blocks 0/1/2.

**This is now the main coverage limit:** SpeechNet blocks 0/1/2 are padded
(`[0,2,0,2]`, `[0,8,0,8]`, `[0,4,0,4]`), so only the two `7×1` blocks reach NE16 automatically.
The GAP9 SDK solves this with pointer arithmetic plus border-subtile computation
(`NE16_ComputeBorders`), not with NE16 padding — that is the shape of the fix, and it belongs in
`NE161xKConstraint.serializeTilingSolution`.

#### Files changed

* `Deeploy/Targets/NE16/TopologyOptimizationPasses/Prepare1xKPass.py` — **new**, the pass
* `Deeploy/Targets/NE16/Deployer.py` — insert at index 1
* `Deeploy/Targets/NE16/Engine.py` — 1×K no longer requires a constant weight
* `.../exp16c_PW_single_layer/build_fixtures.py` — `--plain`

### 2026-09-14 — Session 6 (cont.): exp16c_SDK_port phase 3 — SpeechNet QZO runs on GAP9 with NE16

```
PASSED          Errors: 0 out of 8      (the harness's optimizer-output check)
NE16 dispatches 14        device weight encodes 2        cluster convs 3
train cycles    68,077,617  (cluster-only baseline: 66,817,357)
```

Blocks **3 and 4** (`7×1`, unpadded) execute on NE16 with their weights perturbed **and** bit-serial
encoded on device; blocks 0/1/2 stay on the cluster because they are padded. Reproducible — two
independent runs gave identical loss bits.

Two harness gaps had to be closed first, both silent: `deeployMezoRunner` never threaded
`--enable-1xk` into `gen_args` (its own header warns about exactly this), and `testMVPTraining.py`
did not accept the flag or set `enable1xK` on the engine. Until both were fixed the NE16 engine
claimed nothing and the whole graph fell back to the cluster.

Also worth recording: `-D BN_FROZEN_STATS=ON` and `--l1 110000` are **load-bearing**. Without them
`.weightmem_sram` overflows `L2_shared` by ~706 KB — **with or without NE16** (checked against the
cluster-only baseline), so it is a pre-existing property of this graph, not an NE16 cost.

#### OPEN, and important: full-network numerical equivalence is NOT established

The per-pass losses differ from the cluster-only baseline
(`lp = 0x3ff172fd` vs `0x3f9bbcb3`, and so on). Both runs pass the harness check and both are
deterministic, but that check covers only 8 elements and the divergence is far too large to be
last-ulp. **The port runs; it is not yet shown to be correct.**

Three hypotheses tested and eliminated:

* **Layout** — the lowered graph shows conv → NHWC int32 → `Transpose` → NCHW → RQS tagged
  `channels_first=1`. Correct.
* **Perturbation RNG** — generated C has `tile_seed_offset = 0` for every perturb node and
  `node_id` from the stable `idx` attribute, so both runs use identical random directions.
* **Requant rounding on the original RequantShift** — `b3_ref` (merged on cluster, `+div/2` baked)
  and `b3_plain` (un-merged, NE16) both score 0 errors against the *same* golden.

**Leading remaining hypothesis:** in-network, blocks 3/4 are requantised by a
`QSTANDALONE_QCDQ_..._Quant`-derived RequantShift, **not** the original `RequantShift` the
single-layer fixtures exercised. Preventing the merge changes which node requantises, and
`_merge_conv_rq_fun` bakes `+div/2` into a constant add when it merges. That path was never covered
by a single-layer test.

**Next session's first task:** layer-probe the full network — dump block 3's int8 output under both
configurations and diff.

#### Second open item: padded convolutions

`ne16_1xkAdmissible` refuses non-zero pads, so blocks 0/1/2 stay on the cluster and only **5.8 %**
of conv MACs reach NE16. This is a correctness guard, not conservatism (NE16's 1×1 mode cannot pad
and the gvsoc guard is commented out). The SDK's fix is pointer arithmetic plus border-subtile
computation (`NE16_ComputeBorders`), and it belongs in `NE161xKConstraint.serializeTilingSolution`.
Lifting it would bring block 1 — 68 % of conv MACs on its own — onto NE16.

Full write-up with reproduction commands and a file-by-file change list:
`DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16c_SDK_port/Findings.md`.

#### Session 6 addendum — two more hypotheses eliminated for the loss divergence

* **Requant rounding is self-consistent, not the cause.** Generated C: `b3_ref` (merged, cluster)
  carries `rqs_add = {65468, 65476, …}` = `32700 + 32768` — the `+div/2` the merge bakes in — while
  `b3_plain` (NE16, un-merged) carries the raw `{32700, 32708, …}`. **Both score `0/1280` against
  the same golden.** So the fused kernel truncates and needs the baked rounding, while
  `RequantShift_s32_s8_NCHW` rounds internally and must not get it. The two conventions cancel.
* **Same kernel in-network.** The full NE16 network contains `RequantShift_s32_s8_NCHW` exactly
  twice — one per NE16 conv — the same kernel that is bit-exact in isolation.
* **The pass criterion is weak.** `Errors: 0 out of 8` checks 8 elements of the update graph, whose
  fixture carries `loss_plus (52,)`, `loss_minus (52,)`, `grad (13,)` and ~20 output tensors. It
  cannot certify forward-pass equivalence, which is why it passes in both configurations.

Still unexplained, still the top priority: a device layer-probe of block 3's int8 output under both
configurations.

### 2026-09-14 — Session 6 (cont.): the full-network NE16 result is **WRONG** — correcting the earlier entry

The previous entry said the full network "runs and passes the harness check, equivalence not
established". **That was too generous.** The update fixture ships host reference losses, which
settles it:

| pass | NE16 | cluster | host reference |
|---|---|---|---|
| L+ #0 | **1.8863** | 1.2167 | **1.2167** |
| L− #0 | **0.4563** | 0.2910 | **0.2910** |
| L+ #3 | **2.1837** | 1.6105 | **1.5946** |

The cluster tracks the reference to ~1e-5 (the known fp32 last-ulp behaviour); NE16 is **20–55 %
off**. It is wrong.

**Why the harness says `PASSED`.** `Errors: 0 out of 8` checks the **update** graph, which is driven
by the reference `loss_plus` / `loss_minus` arrays in the fixture — not by the losses the device
just measured. It is structurally blind to the forward pass. Future NE16 work on the full network
must compare device losses against `speechnet_qzo12_update/outputs.npz` directly.

**Bisection** (via `QW_NE16_ONLY`, a diagnostic env var in `Prepare1xKPass.py`, off by default):
block 3 alone → 1.8192 (wrong); block 4 alone → 1.2584 (wrong); cluster only → 1.2167 (correct).
**Both convs are independently wrong in-network.** Block 4 is the sharper clue: with only block 4 on
NE16 the preceding blocks are all cluster, so it receives a **known-correct input** and still
produces a wrong result — while being bit-exact in isolation.

**Six hypotheses eliminated:** layout of the un-merged RequantShift (verified NCHW +
`channels_first=1`); perturbation RNG (`tile_seed_offset = 0`, stable `node_id`); requant rounding
(the fused path truncates with `+div/2` baked, the standalone rounds internally — verified
`rounding = 1` in the emitted call — and both score `0/1280` on the same golden); the requant call
arguments (checked against `RequantShift.h:105`: `log2D=16`, `HW=40` for block 3 — my initial read
of `16` as a channel count was wrong); **spatial tiling** (the single-layer fixture forced to the
*same* 4-way tiling at `--l1 20000/12000/8000` still scores `0/1280`); and output-channel tiling
(`nKo`/`bKo` are scalars, `Cout` is not split).

**A structural observation that matters.** The cluster has **no int8 standalone `Conv` binding** —
only `PULPFPConv2DParser` / `PULPFPDWConv2DParser`; forcing the same un-merge on the cluster fails
to bind. So the un-merged `Conv(int8) + RequantShift` path **exists only on NE16** and has never
been validated against an independent implementation: the single-layer goldens come from
`run_onnx_graph` on that same un-merged graph. That is a weaker check than it appeared, and it is
why every isolated test can pass while the network is wrong.

**Next step:** dump the encoded weight and block 3's int32 conv output from the *full* network and
diff against host-computed values. Leading hypothesis: buffer lifetime or aliasing of `weight_enc`
or the int32 intermediate under full-network memory pressure — something the isolated fixture never
stresses.

A second diagnostic (`QW_NOMERGE_KS`, forcing the same un-merge on the cluster) was written and then
**removed**: it cannot work, because of the missing int8 cluster Conv binding above.

### 2026-09-14 — Session 6 (cont.): the `+div/2` rounding bug — FOUND and FIXED

The wrong full-network loss has a single, concrete cause, and it is a **pre-existing Deeploy bug
that only NE16 could reach**.

**File:** `Deeploy/Targets/PULPOpen/Templates/RequantShiftTemplate.py` (and the Generic twin) —
the standalone RequantShift passed `rounding = 1` to the kernel **unconditionally**.

`PULPConvRequantMergePass._merge_conv_rq_fun` bakes `+div/2` into the requant's `add` **only when
that add is a `gs.Constant`**. For a *runtime* add — a perturbed bias, i.e. an
`RQSPerturbRademacher` output, which is exactly the QZO case — it bakes nothing and the fused
kernel truncates, matching the host reference. So:

| | merged (cluster) | standalone (NE16) |
|---|---|---|
| **constant** add | `+div/2` baked, kernel truncates | kernel adds `div/2` itself → **agree** ✓ |
| **variable** add | nothing baked, kernel truncates | kernel still adds `div/2` → **off by `div/2`** ✗ |

With `div = 65536` that is a systematic **+0.5 LSB on every output element** of blocks 3/4.

**Why every isolated fixture missed it:** the single-layer builders evaluate the perturbed bias on
the host and bake it in as a **constant initializer** — the agreeing row. The real graph takes the
disagreeing row. This is also why `b3_ref` (merged, `add = 32700 + 32768`) and `b3_plain`
(un-merged, `add = 32700`) matched each other exactly: both end up computing
`(acc·mul + 65468) >> 16`.

**Why it had never been hit:** with every convolution fused into a `RequantizedConv`, no standalone
RequantShift with a runtime add ever existed. NE16 is the first thing to leave one un-merged — and
it must, because `streamin` forces int32 output.

**Fix:** derive the flag instead of hardcoding it —
`rqs_rounding = int(hasattr(addBuffer, "values") and addBuffer.values is not None)`.
Generated C now ends both NE16 requant calls with `-128, 127, 0)`.

#### Result

| pass | NE16 before | **NE16 after** | cluster | host ref |
|---|---|---|---|---|
| L+ #0 | 1.886322 | **1.216181** | 1.216696 | 1.216687 |
| L− #0 | 0.456273 | **0.289454** | 0.290987 | 0.290998 |
| L+ #3 | 2.183723 | **1.610300** | 1.610501 | 1.594573 |

```
max |NE16_after − host_ref| = 1.57e-2
max |cluster    − host_ref| = 1.59e-2     <- pre-existing device-vs-host gap
```

**NE16 is now as close to the host reference as the cluster is** (marginally closer). The residual
~1.6e-2 on L+ #3 appears in the cluster baseline too, so it is the known int8/fp32 divergence, not
an NE16 artefact.

#### Residual, stated honestly

Against the cluster directly, six of eight passes agree to ~1e-3 or better, but **L− #1 differs by
1.29e-2** while the cluster tracks the host to 7e-6 there. Identical int8 datapaths would give
identical losses, so **a second, ~40× smaller discrepancy remains**. The signature has changed from
a uniform offset to sparse disagreement, which points at a handful of 1-LSB elements rather than a
systematic bias. That is the next thing to chase.

**Regression after the shared-template change** (the fix touches a template used by every target,
so this matters): `b3_ref` `0/1280`, `b3_plain` `0/1280`, `b4_plain` `0/320` — all unchanged, and
all three still emit `rounding = 1` because their fixtures bake the bias in as a constant. Only the
runtime-add case changes behaviour, which is the one that was wrong.

### 2026-09-14 — Session 6 (cont.): a faithful single-layer reproducer, and a second latent bug

#### The fixtures were lying by construction — now they aren't

Every single-layer fixture so far evaluated the perturbed requant bias on the host and baked it in
as a **constant**. The real graph's bias is `blocks.N.conv.bias_rqsadd_pert`, a runtime
`RQSPerturbRademacher` output. That is precisely the branch where merged and un-merged agree by
construction, which is *why* the `+div/2` bug survived every isolated test.

`exp16c_PW_single_layer/build_fixtures.py` gained **`--variable-bias`**: it feeds the *unperturbed*
bias as a graph input and reproduces the graph's own bias perturbation, exactly as the network does.

**It immediately reproduced the residual at single-layer scale** — `b3_plain_vb` (NE16)
`40 / 1280` against `b3_ref_vb` (cluster) `0 / 1280`. A ~2-minute run instead of a ~10-minute
network run. That reproducer is the main deliverable of this stretch.

#### Second latent bug: `RQSPerturbTileConstraint` channel dimension

`Deeploy/Targets/PULPOpen/TileConstraints/RQSPerturbTileConstraint.py` took
`channelDim = cube.dims[0]`. A perturbed requant bias is rank-1 `(cout,)` **until something
downstream needs it broadcast** — which is exactly what the un-merged RequantShift does — at which
point Deeploy rewrites the *shared* buffer's shape to `(1, cout)`. Then `dims[0] == 1` and

```
channel_width = size // 1 = cout        (32, instead of 1)
```

The kernel indexes `M[(start_offset + i) / channel_width]` (`RandomNoiseQuant.c:182`), so **every
bias element took `M[0]`** — one multiplier for all 32 output channels. Confirmed in the generated
C: the NE16 build emitted `% 32` where the cluster build emitted `% 1`.

Like the rounding bug, this was **unreachable before NE16**: with every conv fused, the bias is
consumed by the fused node and never broadcast. Fixed by skipping leading unit axes, which is right
for both shapes — `(1, cout)` → `cout`, and a weight `(cout, cin, H, W)` → `cout` (unchanged).

Regression: `b3_plain` `0/1280`, `b4_plain` `0/320` — unchanged.

#### But it does NOT explain the residual, and one result is unexplained

Full network after the fix: `PASSED`, losses essentially unchanged (only L+ #1 moved,
0.190928 → 0.192629), **max |NE16 − cluster| still 1.29e-2**.

And on the new fixture the fix made things *worse*: `b3_plain_vb` went from **40 → 1240 / 1280**
errors. I cannot currently explain that, and I am not going to pretend otherwise. Two things are
worth noting before the next session picks this up:

* The fixture's comparison is **not symmetric**: `b3_ref_vb` takes the *host*-perturbed weight as a
  graph input, while `b3_plain_vb` perturbs on device. That asymmetry was harmless while the bias
  was constant (`b3_plain` scored `0/1280`), so it is not obviously the cause — but it is a
  confound the reproducer should remove.
* The only structural difference between `b3_plain` (0 errors) and `b3_plain_vb` (1240 errors) is
  the **presence of the bias-perturb node**. That points at ordering or buffer handling of the
  runtime `add` in the un-merged path — e.g. the requant's `add` being DMA'd before the perturbation
  that writes it — rather than at arithmetic. 1240/1280 ≈ 97 % is the signature of a systematic
  shift, not sparse disagreement.

The channel-width fix is kept because `M[0]` for all output channels is indefensible regardless, its
full-network effect is negligible, and it makes the NE16 build agree with the cluster build on the
perturbation. But it is **not** the residual's cause.

**Next:** make the reproducer symmetric (perturb the weight on device in the reference too, or feed
both the pre-perturbed weight), then chase the ordering/aliasing hypothesis for the runtime `add`.

### 2026-09-14 — Session 6 (cont.): the residual localised to an L1 buffer overlap

#### A symmetric control removes every excuse

Running the **same** `b3_plain_vb` fixture with and without `--enable-1xk` — same graph, same golden,
both perturbations on device, the only variable being which engine runs the conv:

```
cluster (no --enable-1xk):     Errors: 0 out of 1280
NE16    (--enable-1xk):     Errors: 1240 out of 1280
```

The fixture and golden are consistent and both device perturbations are correct. **NE16 is at
fault**, and this is now a ~2-minute reproducer.

#### The error signature names the mechanism

```
Expected:    5  Actual:   73  Diff:  -68 at Index   40      <- channel 1
Expected:  -14  Actual:   39  Diff:  -53 at Index 1279      <- channel 31
```

The difference is a **constant per output channel**, and **channel 0 is exactly right** (the 40
correct outputs are precisely channel 0's `8×5 = 40` elements). That is not arithmetic drift — it is
a wrong per-channel *multiplier* in the bias perturbation.

#### The mechanism, in the generated C

```c
b3_bpert_data_in_ref  = ARENA_L1 + 128;
b3_bpert_mul_ref      = ARENA_L1 + 132;   // only 4 bytes later
...
mchan_transfer_1d(1441920, ..._data_in_ref, ...);   // cmd & 0xFFFF = 128 bytes
mchan_transfer_1d(1441920, ..._mul_ref,     ...);   // cmd & 0xFFFF = 128 bytes
```

Two buffers **4 bytes apart**, each receiving a **128-byte** DMA — a 124-byte overlap. The bias
transfer clobbers the multiplier array, so the perturbation uses corrupted per-channel multipliers.
Symptom and mechanism match exactly.

The root is a disagreement between the two halves of `RQSPerturbTileConstraint`:
`addGeometricalConstraint` **sizes** the tile buffers, `serializeTilingSolution` **sizes the DMAs**,
and they pick the channel axis independently. Once the requant bias is broadcast to `(1, cout)` by
the un-merged RequantShift, they disagree.

#### Status of the two channel-axis changes — kept, with a caveat

The committed `serializeTilingSolution` fix (skip leading unit axes) is **semantically right**:
without it `channel_width = cout` and the kernel's `M[(start+i)/channel_width]` collapses to `M[0]`
for every output channel. But on its own it makes the reproducer go **40 → 1240** errors, because it
starts requesting the full 128-byte multiplier transfer into a buffer the tiler still sized at 4
bytes. **The number went up because the change exposes the allocation bug instead of masking it** —
both states are wrong, and 40 was the *quieter* wrong.

A matching change to `addGeometricalConstraint` was written and **reverted**: it did not move the
offsets, so the 4-byte buffer is `data_in`, not `mul`, and the sizing comes from somewhere I have
not yet traced. Leaving an unproven change in the tree would be worse than leaving the bug
documented.

Regression after all of this: `b3_plain` `0/1280`, `b4_plain` `0/320` — unchanged. Full network
unchanged (max |NE16 − cluster| still 1.29e-2).

**Next:** trace why `b3_bpert`'s `data_in` tile is allocated 4 bytes when its DMA is 128, fix the
sizing, then re-check the reproducer — it should go to 0, and with it the full-network residual.

### 2026-09-14 — Session 6 (cont.): **BIT-IDENTICAL** — the third bug, and the goal met

`RQSPerturbTileConstraint.addGeometricalConstraint` paired input and output dims **by raw index**.
The op is elementwise, but the ranks diverge once the *output* is broadcast to `(1, cout)` while the
*input* stays `(cout,)`:

```
data_in.dim0 (cout)  ==  data_out.dim0 (1)      <- forces the input tile to ONE element
```

So the tiler sized `data_in` at **4 bytes** while `serializeTilingSolution` scheduled a **128-byte**
DMA — the overlap seen directly in the generated C (`data_in` at arena+128, `mul` at arena+132).
Fixed by aligning axes from the first non-unit axis, numpy-broadcast style. Buffers now land at
`+0 / +128 / +256`.

#### Result — the loop's goal is met, and met correctly

```
b3_plain_vb (the faithful reproducer):   Errors: 0 out of 1280
full SpeechNet QZO on GAP9 + NE16:       max |NE16 − cluster| = 0.00e+00
                                         max |NE16 − host_ref| = 1.59e-2
                                         max |cluster − host_ref| = 1.59e-2
```

**Every per-pass loss is bit-identical to the cluster-only baseline**, and NE16's distance to the
host reference is now exactly the cluster's — the known pre-existing device-vs-host int8/fp32 gap,
not an NE16 artefact. 14 NE16 dispatches, 2 on-device weight encodes, 3 cluster convs.

Regression: `b3_plain` `0/1280`, `b4_plain` `0/320`, `b3_ref_vb` `0/1280` — all clean.

#### Three bugs, one structural cause

| # | file | defect |
|---|---|---|
| 1 | `Templates/RequantShiftTemplate.py` (PULPOpen + Generic) | `rounding = 1` hardcoded; the merge bakes `+div/2` only for a **constant** add → off by `div/2` with a runtime bias |
| 2 | `TileConstraints/RQSPerturbTileConstraint.py` (serialize) | `channelDim = dims[0]` → `channel_width = cout` on a broadcast bias → kernel collapses to `M[0]` for every channel |
| 3 | `TileConstraints/RQSPerturbTileConstraint.py` (geometry) | input/output dims paired by raw index → 4-byte tile against a 128-byte DMA → 124-byte buffer overlap |

**All three were pre-existing and unreachable before NE16.** With every convolution fused into a
`RequantizedConv`, no standalone RequantShift with a runtime bias ever existed — the bias was never
broadcast and the rounding branch was never taken. NE16 is the first thing to leave a conv
un-merged, and it must, because `streamin` forces int32 output.

#### What made them findable

* **A faithful fixture.** `--variable-bias` keeps the requant bias a runtime tensor, as the real
  graph has it. Earlier fixtures baked it in as a constant — exactly the branch where merged and
  un-merged agree by construction, which is why they all passed while the network was wrong. It
  reproduced the whole failure in a **2-minute** run.
* **A symmetric control.** The same fixture with and without `--enable-1xk` — same graph, same
  golden, only the engine differs — removed every "unfair fixture" excuse.
* **Reading the error signature.** A constant diff *per output channel* with channel 0 correct named
  the mechanism (corrupted per-channel multipliers) before any code was read.

One intermediate state worth remembering: fixing bug 2 alone took the reproducer from **40 → 1240**
errors. That was not a regression — it made the code request the full multiplier transfer into a
buffer still sized at 4 bytes, **exposing** bug 3 instead of masking it.

#### Also worth recording

The harness's `Errors: 0 out of 8` reported `PASSED` throughout the period when the losses were
20–55 % wrong: it checks the **update** graph, which is driven by the fixture's reference losses,
not the device-measured ones. Future NE16 work must compare device losses against
`speechnet_qzo12_update/outputs.npz` directly.

#### Remaining (not blockers for the goal)

* **Padded convolutions** — blocks 0/1/2 stay on the cluster, so only 5.8 % of conv MACs reach NE16.
  The SDK's fix is pointer arithmetic plus border-subtile computation (`NE16_ComputeBorders`).
* **Performance** — NE16 is 4.3×–7.5× slower than `pulp_nn_conv` per layer; exp16b/exp16c point at
  im2col channel folding as the lever.
