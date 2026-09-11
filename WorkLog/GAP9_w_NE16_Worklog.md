# GAP9_w_NE16 Worklog — NE16 accelerator for QZO SpeechNet on GAP9

Progress tracking for the **NE16-on-GAP9** effort. Read `WorkLog/README.md` first for the
QZO/ZO pipeline as a whole; this file only covers NE16.

**Branch: `feat/GAP9_w_NE16`** in both `TrainDeeploy` and `Onnx4Deeploy` (cut from `feat/QZO`
at `TrainDeeploy@28ff5fa` / `Onnx4Deeploy@2be5137`, both of which are pushed to `TKCW-W/*`).

**Scope: QZO only.** Backpropagation is out of scope — see the plan's Appendix A for why.

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
| 3 | **Make it correct** — full SpeechNet, inference then training | ⬜ |
| 4 | **Optimise** | ⬜ |

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
