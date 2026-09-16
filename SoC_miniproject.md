# SoC Mini-Project — End-to-End On-Device Fine-Tuning of SpeechNet (BP flow)

> A code-grounded walkthrough of the **first-order (backprop) on-device fine-tuning** pipeline for **SpeechNet**,
> from ONNX graph export (**Onnx4Deeploy**) through Deeploy's **frontend → middleware → backend** to a
> cycle-accurate **GVSoC on-device training simulation** on Siracusa/GAP9 (**TrainDeeploy**).
>
> This document is a learning/reference companion to `BP_FLOW.md` (the reproduction recipe). Where `BP_FLOW.md`
> tells you *which commands to run*, this document explains *what the code actually does at each stage* — with
> `file:line` anchors so any "where / how / why" question can be answered from the source.
>
> **Scope:** the BP (first-order) flow only. The zeroth-order (MeZO/ZO) path is deliberately out of scope.
> **Repos:** Onnx4Deeploy at `/app/Onnx4Deeploy`; TrainDeeploy at `/app/ETH/TrainDeeploy` (all `file:line`
> citations below are in the **TrainDeeploy** tree unless the path starts with `Onnx4Deeploy/`).
> **Running example throughout:** SpeechNet **block-0** `Conv2d(in=1, out=8, kernel=(1,4), pad=(0,2), stride=1)`
> on input `(1,1,14,700)`, and its backward gradient ops.

---

## Table of contents

0. [Overview & mental model](#0-overview--mental-model)
1. [Part I — Onnx4Deeploy: fixture generation](#part-i--onnx4deeploy-fixture-generation)
2. [Part II — TrainDeeploy orchestration](#part-ii--traindeeploy-orchestration-runner--build--sim)
3. [Part III — Deeploy frontend](#part-iii--deeploy-frontend)
4. [Part IV — Deeploy middleware (tiling, memory, schedule, DMA, layout)](#part-iv--deeploy-middleware)
5. [Part V — Deeploy backend (kernels & im2col)](#part-v--deeploy-backend)
6. [Part VI — On-device simulation & results](#part-vi--on-device-simulation--results)
7. [Part VII — MaxPool argmax-mask memory optimization](#part-vii--maxpool-argmax-mask-memory-optimization-b)
8. [Appendix — file/line index & glossary](#appendix--fileline-index--glossary)

---

## 0. Overview & mental model

On-device fine-tuning here is a **two-repo pipeline**:

```
┌─────────────────────── Onnx4Deeploy (host, PyTorch + ORT) ───────────────────────┐
│  SpeechNet.pt ─► ONNX export ─► ORT autodiff (fwd+loss+bwd) ─► Deeploy-ify graph  │
│                 ─► SGD optimizer graph ─► fixtures (inputs.npz / outputs.npz)     │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                         │  a fixture directory:
                                         │  network.onnx (+ _train/_infer variants),
                                         │  inputs.npz, outputs.npz, ../<name>_optimizer/network.onnx
                                         ▼
┌─────────────────────── TrainDeeploy (Deeploy compiler + GVSoC) ──────────────────┐
│  runner ─► codegen(train graph)  ─┐                                               │
│           codegen(optimizer graph)─┴─► CMake ─► LLVM/RISC-V build ─► GVSoC sim    │
│                                                                                    │
│  Deeploy compiler internally:  FRONTEND ─► MIDDLEWARE ─► BACKEND                   │
│      (parse ONNX)        (tile + memory + DMA)     (emit C kernels)                │
│                                                                                    │
│  on device:  for step:  { for accum: fwd+bwd+grad-accumulate }  SGD update        │
│  outputs:  per-step loss (vs reference), [WDUMP] device weights                    │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
              extract device weights ─► carry checkpoint ─► export INFER fixture ─►
              evaluate on device (batch r+1) ─► balanced accuracy  (b1→b5 chain)
```

**The incremental b1→b5 protocol** (`BP_FLOW.md` §A.0): round *r* trains on batch *r* (54 windows), the trained
device weights are extracted and carried forward, then batch *r+1* is evaluated for accuracy. This document
focuses on **round 1's TRAIN + the compiler internals**, which is where the frontend/middleware/backend live.

**The "frontend / middleware / backend" framing** (Deeploy's own pipeline, `Deeploy/DeeployTypes.py`):
- **Frontend** (`NetworkDeployer.frontEnd`, `DeeployTypes.py:3407`): import the ONNX graph, lower it (layout
  transforms), and parse each node into a typed internal representation.
- **Middleware** (`NetworkDeployer.midEnd`, `DeeployTypes.py:3466`): bind each node to a concrete kernel, then
  **tile** every kernel to fit L1/L2 and schedule the **DMA** transfers.
- **Backend** (`NetworkDeployer.backEnd`, `DeeployTypes.py:3483`): run code-transformation passes and emit the C
  source (kernels + tiling loops + buffer allocation).

These three are driven by `prepare()` (`DeeployTypes.py:3503`) and `generateFunction()` (`DeeployTypes.py:3554`).

---

## Part I — Onnx4Deeploy: fixture generation

Goal of this stage: turn a trained-checkpoint PyTorch SpeechNet into a **training ONNX graph** (forward + loss +
backward + gradient accumulation), a **standalone SGD optimizer graph**, and the **npz fixtures** that carry the
initial weights, the fine-tuning data windows, and the ORT reference losses/updated-weights.

Command (`BP_FLOW.md` §A.1):
```bash
python3 Onnx4Deeploy.py -model SpeechNet -mode train -o …/speechnet_train_fullfrozen_b1_fold3 \
  --dataset silentwear --data-path … --pretrained-weights …/leave_one_session_out_fold_3.pt \
  --subject S01 --session 3 --batch 1 --condition vocalized \
  --data-size 54 --n-epochs 40 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-frozen-stats --stratified --maxpool-argmax-mask
```

### I.1 Entry & dispatch

`Onnx4Deeploy/Onnx4Deeploy.py` parses the CLI (argparse block ~613–915) and dispatches to `generate_model()`
(365–578). The `-model SpeechNet` name is resolved through a registry (231–237):

```python
"SpeechNet": {
    "class": SpeechNetExporter,
    "input_shape": "(B, 1, 14, 700)",
    "classes": 9,
    "config": {"num_channels": 14, "time_steps": 700, "num_classes": 9},
},
```

`generate_model()` instantiates `SpeechNetExporter(save_path=output_path)`, folds every CLI flag into
`exporter._config_overrides` (`--bn-frozen-stats`, `--maxpool-argmax-mask`, `--pretrained-weights`,
`--stratified`, …), then calls `exporter.export_training()` for `-mode train`
(`Onnx4Deeploy/onnx4deeploy/core/base_exporter.py:715`).

### I.2 The SpeechNet model

`Onnx4Deeploy/onnx4deeploy/models/pytorch_models/speechnet/speechnet.py` (`SpeechNetDeploy`) defines the paper's
5-block architecture (lines 64–113):

| block | Conv2d | pool |
|---|---|---|
| 0 | `Conv2d(1→8,  k=(1,4),  pad=(0,2))` → BN → ReLU | `MaxPool(1,8)` |
| 1 | `Conv2d(8→16, k=(1,16), pad=(0,8))` → BN → ReLU | `MaxPool(1,4)` |
| 2 | `Conv2d(16→16,k=(1,8),  pad=(0,4))` → BN → ReLU | `MaxPool(1,4)` |
| 3 | `Conv2d(16→32,k=(7,1),  pad=(0,0))` → BN → ReLU | `Identity` |
| 4 | `Conv2d(32→32,k=(7,1),  pad=(0,0))` → BN → ReLU | `Identity` |
| head | `AdaptiveAvgPool2d(1,1)` → `reshape(1,32)` → `Linear(32→9)` | |

Two deployment-driven choices matter downstream:
- **`(1,1)` pools are emitted as `nn.Identity`** (speechnet.py:83-92) so the ONNX graph carries **no degenerate
  MaxPool/MaxPoolGrad** — a degenerate MaxPool (input shape == output shape) would trip Deeploy's
  `MaxPoolGradCTileConstraint` (comment at speechnet.py:85-88).
- The head uses a **static `reshape(1, self._fc_in)`** (speechnet.py:123) so there are no dynamic `Shape` ops in
  the exported graph.

The npz tensor names confirm the architecture end-to-end: `blocks_{0..4}_0_{weight,bias}` (conv) +
`blocks_{0..4}_1_{weight,bias,running_mean,running_var}` (BN) + `fc_*`.

### I.3 Data & fine-tuning windows

`SpeechNetExporter.get_data_source()` builds a `SilentWearDataSource`
(`Onnx4Deeploy/onnx4deeploy/data/silent_wear_datasource.py`) which reads one `.h5`
(`sess_{session}_batch_{batch}.h5`), extracts 14 filtered EMG channels, segments at label changes, and takes one
**onset-anchored** `(1,1,14,700)` window per utterance. `load_batches(..., seed=42)` selects the fine-tuning pool:
with `--stratified` it samples class-balanced (`--data-size 54` → 54÷9 = 6 windows/class).

### I.4 Building the training graph (`export_training`, base_exporter.py:715)

1. **Create model + snapshot BN stats.** `create_model()` loads the checkpoint (`--pretrained-weights`, key
   `model_state_dict`). The BN `running_mean`/`running_var` are snapshotted *before* `model.train()` (tracing a
   forward would otherwise corrupt them).
2. **`--bn-frozen-stats`.** Every BatchNorm is switched to `eval()` while the rest of the model is in `train()`,
   so the exported graph **normalizes with the frozen pretrained running stats** (train ≡ inference for BN). This
   is what the device mirrors via `BN_FROZEN_STATS=ON`. The snapshotted stats are restored into the ONNX
   initializers after export.
3. **ORT autodiff.** `onnxruntime.training.artifacts.generate_artifacts(..., optimizer=SGD,
   loss=CrossEntropyLoss, requires_grad=…, frozen_params=…)` produces the **training graph**: forward + loss +
   backward gradients + per-parameter `InPlaceAccumulator` (gradient-accumulation) nodes. BN buffers
   (`running_mean/var/num_batches_tracked`) are placed in `frozen_params` (not trained). This is renamed
   `network_train.onnx`.
4. **Deeploy-ify.** `run_training_optimization(...)` rewrites ORT/`com.microsoft` ops into Deeploy-friendly
   standard ops (Concat/Split canonicalization, ReduceSum→Reshape, FusedMatMul→Gemm, softmax-grad renaming,
   backward-marker annotation, etc.) → `network_train_optim.onnx`, copied to the final `network.onnx`.
5. **MaxPool rewire** (`_rewire_maxpoolgrad_recompute`, base_exporter.py:912). ORT emits `MaxPool` with a 2nd
   *Indices* output and a `MaxPoolGrad(dY, Indices)`. Deeploy has no Indices tensor, so either:
   - *default:* rewire `MaxPoolGrad` to recompute the argmax from the forward input `X` (`MaxPoolGrad(dY, X)`); or
   - *`--maxpool-argmax-mask`:* insert a `MaxPoolArgmax` node emitting a small within-window offset **mask**, and
     rewire to `MaxPoolGradMask(dY, mask)` (the §B optimization — see [Part VII](#part-vii--maxpool-argmax-mask-memory-optimization-b)).
6. **SGD optimizer graph** (`create_optimizer`, base_exporter.py:1180 → `optimizer_onnx.py`). One `SGD` node per
   trainable parameter, inputs interleaved `[param, param_grad.accumulation.buffer]`, output `param_updated`, attr
   `lr`. This is written to a **sibling directory** derived by `derive_optimizer_dir` (optimizer_onnx.py:129):
   `…_train…` → `…_optimizer…`, and it returns `None` (skipping generation) if the name has no `_train`. This is
   why **the `-o` directory name must contain `_train`** (`BP_FLOW.md` A.1 warning).
7. **Reference data** (`create_training_test_data`, base_exporter.py:1103). With `--bn-frozen-stats`, a frozen-BN
   PyTorch model (`model.eval()`, SGD lr 3e-4, `n_accum=4` SUM) runs the full 540-step loop to produce the
   **per-step reference losses** and the **ORT-updated weights**, written to `outputs.npz`.

### I.5 The fixture on disk

For `speechnet_train_fullfrozen_b1_fold3/` (round 1):

| file | contents |
|---|---|
| `network.onnx` | final **training graph** (fwd+loss+bwd+accum), consumed by the device runner |
| `network_train.onnx` / `network_train_optim.onnx` / `network_infer.onnx` | intermediate ORT / optimized / inference forms |
| `inputs.npz` | **134 keys**: `meta_data_size=54`, `meta_n_batches=2160`, `meta_n_accum=4`; **25 base `arr_*`** (window + label + conv/BN weights); per-mini-batch data `mb{i}_arr_0000/0001` |
| `outputs.npz` | **33 keys**: ORT-updated trainable params (`blocks_*`, `fc_*`) + a per-step `loss` array |
| `../speechnet_optimizer_fullfrozen_b1_fold3/network.onnx` | the standalone SGD graph (`Counter({'SGD': 22})`) |

`meta_n_batches = 2160 = 540 steps × 4 accum`; `meta_data_size = 54` unique windows the device cycles via modulo.

> **INFER fixture** (`-mode infer`, §A.4): after device weights are carried back, `export_inference` bakes the
> updated weights as initializers into an inference-form graph (frozen BN → inference), and writes the eval-batch
> windows + ORT reference logits. Same machinery, no backward/optimizer.

---

## Part II — TrainDeeploy orchestration (runner → build → sim)

Command (`BP_FLOW.md` §A.2):
```bash
python3 deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3 \
  --n-steps 540 --n-accum 4 --cores 8 \
  --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D DUMP_WEIGHTS=ON BN_FROZEN_STATS=ON
```

### II.1 The runner is thin

`DeeployTest/deeployTrainingRunner_tiled_siracusa.py` just calls `main(tiling_enabled=True)` from
`testUtils/deeployTrainingRunner.py`. `main` (deeployTrainingRunner.py:26) parses the args, forwards the tiling
knobs into `gen_args` (`--l1/--l2/--defaultMemLevel/--memAllocStrategy/--searchStrategy`, lines 101-123), and
builds a `DeeployTestConfig` with `training=True, tiling=True, n_train_steps, n_accum_steps` (125-144). Then
`run_complete_test(config)` (deeployTrainingRunner.py:149).

### II.2 The four-step spine

`testUtils/core/execution.py:run_complete_test` (245) is literally:

```python
generate_network(config, skip=skipgen)   # Step 1: emit C
configure_cmake(config)                   # Step 2: CMake
build_binary(config)                      # Step 3: LLVM/RISC-V build
result = run_simulation(config, skip=skipsim)  # Step 4: GVSoC
```

- **Step 1 — codegen.** For a training config, `generate_network` delegates to `run_training_codegen`
  (execution.py:35-37 → `testUtils/trainingUtils.py:431`).
- **Step 2 — CMake.** `configure_cmake` (execution.py:75) passes toolchain/platform/testname and, via
  `add_training_cmake_flags` (trainingUtils.py:259), emits `-DTRAINING=ON -DN_TRAIN_STEPS=540 -DN_ACCUM_STEPS=4`.
  The `-D DUMP_WEIGHTS=ON BN_FROZEN_STATS=ON` from the CLI become CMake cache entries.
- **Step 3 — build.** `build_binary` (execution.py:137) runs `cmake --build … --target <testname>`.
- **Step 4 — sim.** `run_simulation` (execution.py:172) runs `cmake --build … --target gvsoc_<testname>`, streams
  stdout, and `parse_test_output` scrapes `Errors: X out of Y`.

### II.3 Two-graph codegen (train + optimizer)

`run_training_codegen` (trainingUtils.py:431) runs **two** Deeploy code-generations into the same `gen_dir`:

1. **Training network** — `testMVPTraining.py -t <train_dir>` (the fwd+bwd+accum graph). It writes
   `training_meta.json` with the resolved `n_train_steps/n_accum_steps/num_data_inputs`, which the runner reads
   back into `config` (trainingUtils.py:494-502).
2. **Optimizer network** — `testMVPOptimizer.py -t <opt_dir> --training-dir=<train_dir>`. `<opt_dir>` is resolved
   by `resolve_optimizer_dir` (trainingUtils.py:243): `_train` → `_optimizer` (or an explicit `--optimizer-dir`).
   `--training-dir` lets the optimizer codegen **share the training network's weight & grad-accumulation buffers**
   (so the SGD graph writes back in place instead of allocating its own copies).

> **Gotcha (matches `BP_FLOW.md` A.1):** if the fixture dir has no `_train`, both the exporter (`derive_optimizer_dir`
> returns `None`) and the runner (`resolve_optimizer_dir` maps to the same dir) fall back to the training graph as
> the "optimizer", and tiling the full training graph as an optimizer fails at `SoftmaxCrossEntropyLoss`.

---

## Part III — Deeploy frontend

Everything below happens inside one script, `DeeployTest/testMVPTraining.py`
(`generateTiledTrainingNetwork`, line 34), which constructs a Deeploy deployer and calls `deployer.prepare()`
(line 161) — that single call drives frontEnd → midEnd → backEnd.

### III.1 Import & type inference (frontend input)

```python
onnx_graph = onnx.load_model(f'{args.dir}/network.onnx')      # testMVPTraining.py:38
graph = gs.import_onnx(onnx_graph)                            # onnx_graphsurgeon graph
```
UNDEFINED-typed unused optional outputs (e.g. leftover MaxPool Indices) are stripped (41-50). `inputs.npz` is
loaded and every graph input is typed: gradient-accumulation buffers (`_grad.accumulation.buffer`,
constant `_GRAD_ACC`) and float data become `PointerClass(float32_t)`; bool → `uint8_t`; others get their
type/offset from `inferTypeAndOffset` (77-105). This `inputTypes`/`inputOffsets` map seeds the frontend's typing.

### III.2 Platform & deployer selection

```python
platform, signProp = mapPlatform(args.platform)              # → PULPPlatform (Siracusa)  :57
for cluster in clusters: cluster.n_cores = args.cores        # 8 cores                     :60-62
deployer = mapDeployer(platform, graph, inputTypes,
                       name="DeeployTrainingNetwork",
                       inputOffsets=inputOffsets,
                       scheduler=_mockScheduler)              # :110-116
```
`mapDeployer` returns a **`PULPDeployer`** (`Deeploy/Targets/PULPOpen/Deployer.py`). Two important defaults live
there: `default_channels_first = False` (Deployer.py:40 → the target is **channels-last / NHWC**), and the
lowering pipeline is registered including `PULPNCHWtoNHWCPass` and `MergeSiblingTransposesPass`
(Deployer.py:55/64). `_mockScheduler` (trainingUtils.py:171) wraps each node in a singleton list — the per-node
"schedule" the tiler expects.

### III.3 `frontEnd`: lower + parse

`NetworkDeployer.frontEnd` (`Deeploy/DeeployTypes.py:3407`) runs:
- `self.lower(self.graph)` (DeeployTypes.py:3255) — applies the registered lowering/topology passes, most
  importantly the **NCHW→NHWC** layout transform for the forward convolution path (see [IV.a](#iva-layout-convention-chw-vs-hwc)).
- `self.parse(default_channels_first)` (DeeployTypes.py:2624) — walks the graph and, for each node, finds a
  matching **parser** via the platform's op-mapping table.

The op-mapping table is `Deeploy/Targets/PULPOpen/Platform.py`. Each ONNX op name maps to a `NodeMapper(parser,
tilingReadyBindings)`; the training ops relevant to SpeechNet:

| ONNX op | mapper / parser | Platform.py |
|---|---|---|
| `SoftmaxCrossEntropyLoss` / `…Grad` | `SoftmaxCrossEntropyLoss(Grad)Parser` | 135 / 136 |
| `SGD` | `SGDParser` | 138 |
| `InPlaceAccumulatorV2` | `InPlaceAccumulatorV2Parser` | 139 |
| `BatchNormInternal` / `…Grad` | `BatchNormInternalParser` / `BatchNormalizationGradParser` | 113 / 114 |
| `ConvGradX` / `ConvGradW` / `ConvGradB` | `PULPConvGradX2DParser` / `PULPConvGradW2DParser` / `Conv2DGradBParser` | 145 / 148 / 151 |
| `MaxPoolArgmax` / `MaxPoolGradMask` | `MaxPool2DParser` / `MaxPoolGradParser` | 105 / 110 |

`ConvGradX`/`ConvGradW` each register **several** candidate mappers (pointwise / depthwise / regular), e.g.
`'ConvGradW': ConvGradWLayer([PWConvGradW2DMapper, DwConvGradWMapper, ConvGradWMapper])` (Platform.py:156) — the
concrete one is chosen at binding time.

---

## Part IV — Deeploy middleware

This is the heart of on-device training: **binding → tiling → memory allocation → DMA scheduling**. It is driven
by `NetworkDeployer.midEnd` (`DeeployTypes.py:3466`), wrapped by two decorators set up in `testMVPTraining.py`:

```python
deployer = MemoryDeployerWrapper(deployer, annotation_passes)                 # :146  (memory levels)
deployer = TilerDeployerWrapper(deployer, TrainingSBTiler, …)                 # :152  (the tiler)
deployer.tiler.memoryAllocStrategy = args.memAllocStrategy   # "MiniMalloc"    :154
deployer.tiler.searchStrategy      = args.searchStrategy     # "random-max"    :155
_ = deployer.prepare(verbosityCfg)                                            # :161  runs the whole pipeline
```

The memory hierarchy is declared just above (testMVPTraining.py:118-123):
```python
L3 = MemoryLevel("L3", ["L2"], size=64_000_000)
L2 = MemoryLevel("L2", ["L3","L1"], size=args.l2)   # 1_500_000  (GAP9 1.5 MB L2)
L1 = MemoryLevel("L1", ["L2"], size=args.l1)        #   128_000  (< GAP9 131_072)
```

### IV.a Layout convention (CHW vs HWC)

**The forward path is NHWC (HWC); the backward gradient path is NCHW (CHW).** This split is the single most
important layout fact in the flow, and it is visible three ways:

1. **The lowering pass only converts Conv.** `_NCHWtoNHWC_fun`
   (`Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py:26`)
   bails out for anything that is not a convolution:
   ```python
   if node.op not in ["Conv", "RequantizedConv"]:
       return graph   # (…later the real transform runs only for Conv/RequantizedConv, :255)
   ```
   So the **forward `Conv` is transposed to NHWC**, but the **`ConvGradX`/`ConvGradW`/`ConvGradB` nodes are left in
   NCHW**.
2. **The tile constraint reads NHWC for the forward Conv.** `ConvTileConstraint.addPolicyConstraint`
   (`Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py:348-353`) comments `# NHWC layout` and reads
   H=dimIdx1, W=dimIdx2, **C=dimIdx3 (last)**.
3. **The kernel names encode it.** Forward kernels end in `_HWC`
   (`TargetLibraries/PULPOpen/src/Convolution_fp32.c`), the gradient kernels end in `_CHW`
   (`TargetLibraries/PULPOpen/src/ConvGrad.c`), and the grad templates are literally titled *"NCHW trainlib naive"*
   (`Deeploy/Targets/PULPOpen/Templates/FloatConvGradTemplate.py:173/228`). `ConvGradConstraint.py:1309` confirms
   `# X, dY are NCHW`.

Why: the forward Conv kernel is a PULP im2col+GEMM that wants **C contiguous (last)**; the backward kernels are
pulp-trainlib-derived and operate in trainlib's native **CHW** blob layout (`conv_args.HWC = 0`, ConvGrad.c:74).

### IV.b The tiler as a constraint solver

"Tiling a kernel" = choosing tile sizes for each tensor dimension so a tile fits in L1, subject to the kernel's
geometric relations. Deeploy models this as a **constraint-programming problem** solved by **Google OR-Tools**:

```python
# Deeploy/TilingExtension/TilerModel.py
self._model: Solver = Solver('CPSimple')                    # :40   (ortools constraint_solver)
def addObjective(self, objective, objectiveType):           # :97
    if objectiveType == 'maximize': …                       # :98
```

The objective is to **maximize the memory a tile pattern uses** — i.e. *fill L1 as full as possible* (bigger tiles
⇒ fewer tiles ⇒ less DMA/loop overhead). `TilerExtension.py` builds a `DEEPLOY_PATTERN_MEM` variable per pattern
and maximizes it. The **search strategy** picks how OR-Tools assigns values (TilerModel.py:365):
```python
if searchStrategy == 'random-max':
    # non-permutation vars: CHOOSE_FIRST_UNBOUND + ASSIGN_MAX_VALUE   (:374 grab the biggest tile)
    # permutation vars:     ASSIGN_RANDOM_VALUE                       (randomize packing order)
```
So `--searchStrategy random-max` = "take the largest feasible tile, randomize the packing order" — a good default
for the greedy fill-L1 objective. `max`/`min` are the deterministic alternatives.

Two structural asserts constrain *what can even be tiled*:
- `Deeploy/TilingExtension/TileConstraint.py:134` — a tileable node must have exactly one output. (This is why the
  argmax-mask optimization keeps every node single-output; see Part VII.)
- `Deeploy/TilingExtension/TilingCodegen.py:537` — `offset should be zero when dims == reference` (guards tile
  collapse; the origin of the "wrong-axis" bug fixed in §B).

### IV.c How each kernel gets tiled — the block-0 example

**Forward `Conv2d(in=1,out=8,k=(1,4),pad=(0,2))` on `(1,1,14,700)` (NHWC ⇒ N=1,H=14,W=700,C=1 in / C=8 out):**

`ConvTileConstraint.addPolicyConstraint` (ConvTileConstraint.py:369-384) declares:
```python
# Keep whole input channels (required for im2col algorithm)
tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])            # :371  C_in NOT tiled
tilerModel.addConstraint(effectiveInputHeight >= parseDict['dim_kernel_x'])   # :374  H tile ≥ kernel
tilerModel.addConstraint(effectiveInputWidth  >= parseDict['dim_kernel_y'])   # :375  W tile ≥ kernel
tilerModel.addConstraint((effectiveInputHeight % strides[0]) == 0)            # :378  stride-aligned
tilerModel.addConstraint((effectiveInputWidth  % strides[1]) == 0)            # :379
tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])        # :382  weights NOT tiled
```
So the tiler **keeps input channels whole, tiles the spatial H/W, and (separately) tiles the output channels**.
The `effectiveInput*` expressions (ConvTileConstraint.py:364-367) add the **halo** — a spatial tile must overlap
its neighbours by `kernel−1` and re-include edge padding — and `computeInputCube` (409-459) turns an output tile
back into the exact input tile (with per-tile padding, so the im2col kernel's internal 0-padding is correct).

**Backward gradients** tile *differently*, because their reduction structure differs:
- `ConvGradW` (dW): the constraint file comments *"Tiles along C (output channels). N, H, W are kept full
  (reduction dims)"* (`ConvGradConstraint.py:1579`). dW = Σ over spatial of `dY·X`, so spatial can't be split
  freely; the free axis is the (output) channel.
- `ConvGradX` (dX): default *"CinSlice first — perf path for small-spatial / big-channel"*
  (`ConvGradConstraint.py:1280`); it tiles the input-channel slice (matching the scatter kernel that parallelizes
  over Cin).
- Generic elementwise/binary gradient ops (ReluGrad, adds, …) inherit `BOPTileConstraint`, which ties **all**
  input/output dims equal — uniform tiling.

**Goal restated:** pick the largest tile that (a) fits L1, (b) respects the halo/stride/reduction constraints, so
that the whole layer is covered in as few tiles as possible.

### IV.d Tile-dimension strategy & the CHW/HWC tradeoff

When a dimension is **not** tiled, its full extent travels in **every** tile. That is the cost model behind the
layout choice:

- In NHWC, **C is the innermost (contiguous) axis**. Keeping C whole and tiling H/W means each tile is a set of
  **contiguous C-length lines** — a clean 2D DMA (`pi_cl_ram_copy_2d`, one strided transfer of contiguous lines).
- The tiler even guards against splitting the contiguous axis too finely:
  `Deeploy/Targets/PULPOpen/TileConstraints/DMASafeTileConstraints.py` forces the innermost tiled extent to stay
  `≥ _MIN_INNERMOST_TILE (=2)`, because a 1-element line makes `pi_cl_ram_copy_2d`'s "line length" collapse and the
  DMA wait never completes (documented hang).

So the empirical rule *"put the big / contiguous dimension last"* is **correct and code-grounded**: a contiguous
last axis (C in HWC) gives long DMA lines and avoids re-copying a large non-tiled dimension in fragmented strides.
If SpeechNet's forward Conv stayed NCHW, tiling channels would force each channel-tile to gather all H×W with huge
strides — exactly the "tiled the wrong (W) axis" failure the §B write-up describes for a mis-lowered op.

(Nuance: this reasoning is about the **forward** HWC path. The backward CHW kernels are pulp-trainlib blobs; their
tile constraints keep the reduction dims whole and tile the channel, which is the CHW-appropriate analogue.)

### IV.e DMA, double buffering & the schedule

The tiled loop and its DMA transfers are emitted by code-transformation passes in
`Deeploy/TilingExtension/CodeTransformationPasses/`:

- **Single buffering** (`SingleBufferingTilingCodeGeneration.py`): for each tile — DMA-in → compute → DMA-out,
  serially. `TrainingSBTiler` (the tiler class wired in testMVPTraining.py:152, "SB" = single-buffer) uses this
  strategy, extended so that a **forward activation stays live until the backward** consumes it.
- **Double buffering** (`DoubleBufferingTilingCodeGeneration.py`): a ping-pong `switch(tileIdx % bufferCount)`
  that **DMAs the NEXT tile while computing the CURRENT one**, overlapping transfer with compute. It costs 2× L1
  per double-buffered tensor (`multiBufferCoefficient = 2` in the memory-scheduler cost vector).
- **Async DMA** (`AsyncDma.py`): transfers are `Future`s with `init/alloc/wait`; waiting can be per-tensor,
  per-direction, or a single barrier.

The **schedule** itself is the `_mockScheduler` singleton-list (one node per pattern, trainingUtils.py:171): each
kernel is tiled and scheduled independently, in graph order.

### IV.f Memory allocation (L1/L2/L3, MiniMalloc)

`MemoryDeployerWrapper` annotates each tensor's home level: `AnnotateIOMemoryLevel(defaultIoMemLevel)` and
`AnnotateDefaultMemoryLevel(memoryHierarchy)` (testMVPTraining.py:131-133). With `--defaultMemLevel L2`, IO and
weights live in L2 (GAP9's 1.5 MB), tiles stream into L1.

Placement of live buffers within a level is a **lifetime bin-packing** problem. `--memAllocStrategy MiniMalloc`
decouples it from tiling: Deeploy writes a CSV of `(id, lower, upper, size)` lifetimes and calls an external
`minimalloc` solver, then reads back each buffer's base offset (the L1/L2 arena is one big scratch buffer + offset
arithmetic). `TetrisRandom` / `TetrisCo-Opt` are the built-in heuristic alternatives. The final L1 arena size is
the max address any tile occupies (×`multiBufferCoefficient` where double-buffered). Because allocation is
**static**, the memory footprint is independent of `--n-steps` (a short `--n-steps 4` run is a valid fit check —
`BP_FLOW.md` runtime note).

---

## Part V — Deeploy backend

`NetworkDeployer.backEnd` (`DeeployTypes.py:3483`) runs `codeTransform` then emits C. For training, the emission
is orchestrated by `generateTrainingTestNetwork` (`DeeployTest/testUtils/codeGenerateTraining.py`, called from
testMVPTraining.py:228), which produces the generated project:

- `testinputs.h` — `#define N_TRAIN_STEPS/N_ACCUM_STEPS/TRAINING_NUM_DATA_INPUTS/TRAINING_GRAD_BUF_START_IDX/
  TRAINING_NUM_GRAD_INPUTS/TRAINING_NUM_WEIGHT_INPUTS/TRAINING_LEARNING_RATE`, the initial weights (`testInitWeights`),
  and the per-mini-batch data (`testDataVector`).
- `testoutputs.h` — `TRAINING_TOLERANCE_ABS`, `N_LOSS_REFS`, `testLossRef[]` (from `outputs.npz['loss']`).
- `TrainingNetwork.c/.h` and `OptimizerNetwork.c/.h` — the two tiled networks, plus buffer/arena definitions; L3
  tensors are emitted as `.hex` dumps loaded at init.

Each node's C is emitted from a **template** selected by a **binding**
(`Deeploy/Targets/PULPOpen/Bindings.py`) that also type-checks (`TypeCheckers.py`). The binding is where the
concrete **kernel version** is nailed down.

### V.1 Which device kernel version runs per op

The device kernels live in `TargetLibraries/PULPOpen/src/`. Their **provenance** is mixed:
- **Deeploy-native fp32** reference kernels: forward Conv (`Convolution_fp32.c`) and `ConvGradX` scatter
  (`ConvGrad.c`).
- **pulp-trainlib** wrappers: `ConvGradW` calls `pulp_conv2d_fp32_bw_param_grads_cl` (ConvGrad.c:79/148).
- **PULP-NN** proper is the int8/quantized *inference* lineage (im2col+GEMM heritage) — **not** the fp32 training
  path used here.

The **verified bindings** for the SpeechNet fp32 build (`Bindings.py`):

| op | binding → template | kernel called | layout | im2col? | source |
|---|---|---|---|---|---|
| **Forward Conv** | `PULPFloatConv2DBindings` → `reference2DIm2ColTemplate` (Bindings.py:247) | `PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC` | HWC | **yes** | `Convolution_fp32.c:104` (Deeploy-native) |
| **ConvGradW (dW)** | `PULPFloatConvGradW2DBindings` → `referenceConvGradW2DIm2ColTemplate` (Bindings.py:253) | `PULP_ConvGradW2d_fp32_fp32_fp32_CHW_Im2Col` | CHW | **yes** | `ConvGrad.c:82` → pulp-trainlib (`USE_IM2COL=1`) |
| **ConvGradX (dX)** | `PULPFloatConvGradX2DBindings` → `referenceConvGradX2DTemplate` (Bindings.py:258) | `PULP_ConvGradX2d_fp32_fp32_fp32_CHW_scatter_tiled` | CHW | **no** | `ConvGrad.c:158` (Deeploy-native scatter-add) |

So: **forward Conv = im2col+GEMM (HWC); ConvGradW = im2col via pulp-trainlib (CHW); ConvGradX = scatter-add, no
im2col (CHW).** The alternative kernels exist but are *not* the default binding: a direct forward
`PULP_Conv2d_..._HWC` (`Convolution_fp32.c:10`), a naive `PULP_ConvGradW2d_..._CHW` (ConvGrad.c:14), and an
im2col+GEMM `PULP_ConvGradX2d_..._CHW_Im2Col_tiled` (ConvGrad.c:247, stride-1 only, needs dY-column +
weight-transpose buffers).

The pulp-trainlib call for dW (ConvGrad.c:54-79) fills a `Conv2D_args` with `skip_in_grad=1` (weight grad only),
`HWC=0` (CHW blobs), and `USE_IM2COL=0`/`1` for the naive/im2col variant respectively. The dX scatter kernel is
the simple index relation `dX[ci,ih,iw] += dY[co,oh,ow]·W[co,ci,ky,kx]`, parallelized over `Cin` so each core owns
a conflict-free dX slice (ConvGrad.c:152-158).

### V.2 im2col — benefit vs cost, per op (code-grounded)

im2col reformulates a convolution as a **GEMM**: gather each receptive field into a contiguous column, then do a
dot-product against the (contiguous) weights. Benefit: the inner loop is a sequential FMA chain (prefetch-friendly)
that reuses an optimized matmul. Cost: an extra **transient column buffer** in L1. The verdict depends on the
buffer *size*:

- **Forward Conv** (`Convolution_fp32.c:131`): `im2col_size_per_core = C·P·Q` — for block-0, `1·1·4 = 4` floats
  per core, ~128 B total. Negligible ⇒ im2col is a **free win**, it does not shrink tiles.
- **ConvGradW** (`FloatConvGradTemplate.py` im2col dim `Hout·Wout·Cin·kH·kW`): scales with the spatial extent, so
  for large-spatial layers it grows to KB–MB and **forces smaller tiles / more DMA**. It's the right choice only
  when `dY` spatial is already small. (This is exactly the "im2col ⇒ GEMM, more contiguous, but a bigger L1 tensor
  that forces smaller tiles" tradeoff — **true for dW**.)
- **ConvGradX** deliberately **avoids** im2col by default: the scatter-add kernel has **zero transient buffer** and
  is **stride-agnostic**, so it never pressures L1. The im2col+GEMM dX variant is kept for the large-Cin /
  stride-1 regime but is not the default binding.

So the common one-line claim "im2col is always contiguous-but-bigger-buffer" is **correct for ConvGradW, only
partly for the forward (buffer is negligible), and sidestepped entirely for ConvGradX.**

---

## Part VI — On-device simulation & results

The compiled binary runs on GVSoC via the harness `DeeployTest/Platforms/Siracusa/src/deeploytraintest.c`.

### VI.1 The training loop

`main` (deeploytraintest.c:276) initializes the two networks, **zero-inits the gradient-accumulation buffers**
(312-328), copies `testInitWeights` into the persistent weight buffers (348-353), sets `BN_FROZEN_STATS`
(363-368), then runs the nested loop (370-442):

```c
for (update_step = 0; update_step < N_TRAIN_STEPS; update_step++) {      // 540
  for (accum_step = 0; accum_step < N_ACCUM_STEPS; accum_step++) {       // 4
    mb = update_step*N_ACCUM_STEPS + accum_step;
    // ① lazy_reset_grad = (accum_step==0) ? 1 : 0     → reset vs accumulate
    // ② load mini-batch mb%TRAINING_DATA_SIZE (data + label) into inputs
    // ③ RunTrainingNetwork()  → forward + backward + InPlaceAccumulatorV2
    // ④ store loss = DeeployNetwork_outputs[0]
  }
  run_optimizer_step();                                                   // ⑤ SGD update
}
```

**Gradient accumulation** is realized entirely inside `RunTrainingNetwork` via the `InPlaceAccumulatorV2` nodes,
gated by the `lazy_reset_grad` flag (`accum_step==0` ⇒ *reset* the accumulator to this step's gradient; otherwise
*add*). After 4 mini-batches, the accumulator holds the **SUM** of 4 gradients (effective batch 1 under the S2
recipe). There is no gradient loop in Python — the device does the whole 2160-pass schedule itself.

**`run_optimizer_step`** (deeploytraintest.c:147): copy weights + accumulated grads into the optimizer's input
buffers (skipped when codegen shared the buffers — pointer-equality test, 157-168), run `RunOptimizerNetwork`
(the compiled SGD graph) on the cluster, then copy `weight_updated` back (skipped when in-place, 185-208). The SGD
update is `w ← w − lr·Σgradᵢ`.

### VI.2 Verifying against the reference

Losses are compared on the cluster (the fabric controller has no FPU): `CompareLossesOnCluster`
(deeploytraintest.c:252) checks `|computed − reference| ≤ TRAINING_TOLERANCE_ABS` against `testLossRef[]` (the ORT
per-step losses from `outputs.npz`), and `main` prints `Errors: X out of Y` (461) — the pass/fail line the runner
scrapes. A `BENCH train_cycles=… opt_cycles=… weight_sram=…` line (476) reports cycle counts.

> **Loss "breaches" are expected, not failures** (`BP_FLOW.md` A.2/§B.5): after the drift onset (~step 133 for
> round 1) the device-vs-ORT MaxPool argmax can tie-flip due to fp reduction-order differences in the conv/BN
> output. The `[WDUMP]` weights remain valid.

### VI.3 Extracting the trained weights (`DUMP_WEIGHTS`)

With `-D DUMP_WEIGHTS=ON`, after the final optimizer step the harness prints each trainable tensor as raw hex
(`dump_weights`, deeploytraintest.c:218): `[WDUMP s=<step> wi=<i> n=<#floats>] <hex> …`. This is FPU-free and
bit-exact. Off-device, `extract_device_weights.py` (`BP_FLOW.md` A.3) parses the **last** `[WDUMP]` block,
value-matches each dumped tensor to its ORT reference name in `outputs.npz`, reshapes, and writes them over the
checkpoint's trainable tensors (BN running stats stay frozen) → the **carry checkpoint** for round *r+1*.

### VI.4 Closing the chain

The carry checkpoint feeds the **INFER** export (§A.4) → an untiled inference harness evaluates the 180 windows of
batch *r+1* → **balanced accuracy** (`BP_FLOW.md` A.5). Repeating for b1→b5 gives the on-device fine-tuning
accuracy curve, compared against a matched-data PyTorch reference (§A.6).

---

## Part VII — MaxPool argmax-mask memory optimization (§B)

**Goal:** shrink the on-device backprop **L2 peak** so SpeechNet's full-model FT graph fits **GAP9's 1.5 MB L2**.
**Result:** L2 peak **1,793,800 B → 1,511,308 B (−15.7%)** at a loose 2 MB budget, repacking to **1,482,636 B**
under a strict `--l2 1500000` — fits GAP9, **bit-exact (0/16)**.

### VII.1 The idea

MaxPool's backward needs to know *which* input element was the window max, to scatter `dY` to it. Deeploy's stock
`PULP_MaxPoolGrad2d` **recomputes** the argmax from the forward input `X` at backward time — which forces the large
block-0 activation `X` (~314 KB) to be **stashed live from forward all the way to backward**. On an L2-bound MCU
that long-lived stash dominates the peak.

**Fix:** have the forward emit a small **within-window argmax offset mask**; the backward reads the mask and
scatters, so the big activation can be **freed right after the forward pass**.

**Correctness invariant:** store the *within-window* offset `p·Q + q` (the winner's position *inside* its pooling
window), which is **tile-position-independent**; the backward reconstructs the absolute input position **locally**
from the output position + offset. Same tie-break (strict `>`, first-max) as the forward MaxPool ⇒ the stored
winner is exactly the pooled value's source.

**Design choice — a separate single-output op.** A 2-output MaxPool (pooled + mask) is blocked by the tiler's
one-output assert (`TileConstraint.py:134`). So a **separate `MaxPoolArgmax`** op (and matching `MaxPoolGradMask`)
is added, keeping every node single-output → no tiler-core changes.

### VII.2 The implementation

1. **Device kernels** (`TargetLibraries/PULPOpen/src/MaxPool.c`):
   - `PULP_MaxPoolArgmax2d_fp32_fp32_HWC` (MaxPool.c:180) — same window scan + tie-break as `PULP_MaxPool2d`, but
     stores the within-window offset: `best_off = p*Q + q; pMask[…] = (float32_t)best_off;` (MaxPool.c:216-224).
   - `PULP_MaxPoolGradMask2d_fp32_fp32_HWC` (MaxPool.c:236) — reads the offset, reconstructs the position
     **locally** and scatters, *no* forward-activation read:
     ```c
     uint32_t off = (uint32_t)(pMask[out_idx] + 0.5f);      // MaxPool.c:268
     uint32_t p = off / Q;  uint32_t q = off % Q;           // :269-270  tile-invariant
     pGradIn[((h_in)*W_in + w_in)*C + c] += pGradOut[out_idx];  // :275
     ```
   > **Note (Issue A):** the offset is stored as **fp32**, not uint8. The first design used a uint8 mask (4×
   > smaller) but hit type friction in the all-fp32 transpose/tiling pipeline, so it was switched to an fp32 offset
   > mask + a distinct `MaxPoolGradMask` op. A real uint8 mask is deferred (would shrink the long-lived mask ~4×,
   > but is not peak-changing).
2. **Codegen templates** — `Deeploy/Targets/PULPOpen/Templates/FloatMaxPoolTemplate.py` (`argmaxTemplate`,
   `referenceGradMaskTemplate`), same `(H,W)` mapping as the forward MaxPool template.
3. **Op registration** — `TypeCheckers.py`, `Bindings.py` (`PULPMaxPoolArgmaxBindings`,
   `PULPMaxPoolGradMaskBindings`), `Tiler.py`, and `Platform.py` (`MaxPoolArgmaxMapper` at Platform.py:105,
   `MaxPoolGradMaskMapper` at 110; `'MaxPoolArgmax'`/`'MaxPoolGradMask'` layers at 166/170). Parsers are reused
   (`MaxPool2DParser` / `MaxPoolGradParser`).
4. **Graph rewrite** — Onnx4Deeploy `base_exporter.py:912`, behind `--maxpool-argmax-mask`: insert a
   `MaxPoolArgmax` node (input = the MaxPool's forward input; output = the offset mask) and wire
   `MaxPoolGradMask(dY, mask)`; `MaxPool` stays single-output (pooled only).
5. **Layout passes** — `LoweringOptimizationPasses.py`: add `MaxPoolArgmax`/`MaxPoolGradMask` to the NHWC lowering
   op-list (they must be HWC to tile the channel axis), and extend the MaxPoolGrad 2nd-input transpose special-case
   to the mask.
6. **Transpose-dedup (the actual memory win)** — `MergeSiblingTransposesPass` (`LoweringOptimizationPasses.py`),
   wired as the **last** lowering pass in `PULPDeployer` (Deployer.py:64). MaxPool, MaxPoolArgmax and MaxPoolGradMask
   each transpose the **same** 314 KB block-0 activation to NHWC; the PULP pipeline otherwise gives each consumer
   its **own** input transpose (`TransposeSplitPass`), duplicating the buffer. Merging the three siblings into one
   shared transpose is what turns a +1.8% regression into the −15.7% win.

### VII.3 Challenges → root cause → fix (condensed from §B.3)

- **Issue B — tiler `minimizeRectangle` assert (`TilingCodegen.py:537`).** The new ops had no NCHW→NHWC pass, so
  their tensors stayed NCHW and the PULP channel-tiler tiled the wrong (W) axis. Fixed by the layout passes (step 5).
- **Issue C — the correct build first looked like a +1.8% regression.** The correct build had **2 extra 314 KB
  block-0 buffers** = per-op input transposes. Fixed by `MergeSiblingTransposesPass` (step 6).

### VII.4 Result & caveat

| build | L2 peak | vs baseline | fits 1.5 MB? |
|---|--:|--:|:--:|
| recompute `MaxPoolGrad` (pre-dedup baseline) | 1,793,800 B | — | ❌ |
| recompute `MaxPoolGrad` (with transpose-dedup) | 1,735,180 B | −3.3% | ❌ |
| argmax-mask, no transpose-dedup | 1,825,356 B | +1.8% | ❌ |
| **argmax-mask + transpose-dedup** (loose 2 MB) | **1,511,308 B** | **−15.7%** | — |
| **argmax-mask + dedup, strict `--l2 1500000`** | **1,482,636 B** | | ✅ (bit-exact 0/16) |

The dedup pass is general (it also helps the recompute path), so the fair both-with-dedup comparison is
**1,735,180 B → 1,511,308 B (−12.9%)**. All builds are **bit-exact**.

> **Caveat (§B.5):** the argmax-mask does **not** remove the device-vs-ORT MaxPool drift. Both the recompute and
> the argmax path compute `argmax(X)` from the *same* stashed `X`, so the tie-flip susceptibility is unchanged —
> the mask only guarantees device forward/backward agree (which they already did). The optimization is a **memory +
> math-correctness win**; the inherent device-vs-ORT tie-flip drift persists at full round length.

---

## Appendix — file/line index & glossary

### A. File/line index (all in the TrainDeeploy tree unless prefixed `Onnx4Deeploy/`)

**Onnx4Deeploy (export & fixtures)**
- `Onnx4Deeploy/Onnx4Deeploy.py` — argparse (~613–915), `generate_model` (365–578), SpeechNet registry (231–237).
- `Onnx4Deeploy/onnx4deeploy/models/speechnet_exporter.py` — `create_training_test_data` (base 1103), frozen-BN reference.
- `Onnx4Deeploy/onnx4deeploy/models/pytorch_models/speechnet/speechnet.py` — architecture (64–113), Identity-pool (83-92), static reshape (123).
- `Onnx4Deeploy/onnx4deeploy/data/silent_wear_datasource.py` — `.h5` windowing, `load_batches` (seed 42).
- `Onnx4Deeploy/onnx4deeploy/core/base_exporter.py` — `export_inference` (462), `export_training` (715), `_rewire_maxpoolgrad_recompute` + argmax-mask (912), `create_optimizer` (1180).
- `Onnx4Deeploy/onnx4deeploy/core/optimizer_onnx.py` — `create_optimizer_onnx` (35), `derive_optimizer_dir` (129).

**Orchestration**
- `DeeployTest/deeployTrainingRunner_tiled_siracusa.py` → `testUtils/deeployTrainingRunner.py:main` (26).
- `testUtils/core/execution.py` — `run_complete_test` (245), `configure_cmake` (75), `build_binary` (137), `run_simulation` (172).
- `testUtils/trainingUtils.py` — `run_training_codegen` (431), `resolve_optimizer_dir` (243), `_mockScheduler` (171), `add_training_cmake_flags` (259).
- `DeeployTest/testMVPTraining.py` — full codegen pipeline (34–270): load/type (38–105), mapDeployer (110), memory hierarchy (118–123), wrappers (146/152), `prepare` (161), emit (228).

**Deeploy compiler**
- `Deeploy/DeeployTypes.py` — `NetworkContext` (522), `NetworkDeployer` (3202), `lower` (3255), `frontEnd` (3407), `midEnd` (3466), `backEnd` (3483), `prepare` (3503), `generateFunction` (3554), `parse` (2624), `codeTransform` (2125), TileConstraint one-output assert (`TileConstraint.py:134`), offset assert (`TilingCodegen.py:537`).
- `Deeploy/Targets/PULPOpen/Platform.py` — op-map (MaxPoolArgmax 105/166, MaxPoolGradMask 110/170, BN 113/114, SCE 135/136, SGD 138, InPlaceAccumulatorV2 139, ConvGradX/W/B 145/148/151).
- `Deeploy/Targets/PULPOpen/Deployer.py` — `default_channels_first=False` (40), `PULPNCHWtoNHWCPass` (55), `MergeSiblingTransposesPass` (64).
- `Deeploy/Targets/PULPOpen/Bindings.py` — forward Conv (247), ConvGradW (253), ConvGradX (258).
- `Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py` — NHWC dims (348-353), keep-C-whole (371), spatial/stride/halo (374-379/364-367), `computeInputCube` (409).
- `Deeploy/Targets/PULPOpen/TileConstraints/ConvGradConstraint.py` — dX CinSlice (1280), "X, dY are NCHW" (1309), dW tile-along-C (1579).
- `Deeploy/Targets/PULPOpen/TileConstraints/DMASafeTileConstraints.py` — `_MIN_INNERMOST_TILE` = 2.
- `Deeploy/TilingExtension/TilerModel.py` — `Solver('CPSimple')` (40), `addObjective`/maximize (97), search strategy (365/374).
- `Deeploy/TilingExtension/CodeTransformationPasses/{Single,Double}BufferingTilingCodeGeneration.py`, `AsyncDma.py`.
- `Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py` — `_NCHWtoNHWC_fun` Conv-only guard (26/255).

**Device kernels & harness**
- `TargetLibraries/PULPOpen/src/Convolution_fp32.c` — im2col forward (104), im2col loop (131-176), direct forward (10).
- `TargetLibraries/PULPOpen/src/ConvGrad.c` — ConvGradW naive (14) / im2col (82) → `pulp_conv2d_fp32_bw_param_grads_cl` (79/148); ConvGradX scatter (158) / im2col-tiled (247).
- `TargetLibraries/PULPOpen/src/MaxPool.c` — `PULP_MaxPoolArgmax2d…` (180), `PULP_MaxPoolGradMask2d…` (236).
- `Deeploy/Targets/PULPOpen/Templates/{FloatConvTemplate,FloatConvGradTemplate,FloatMaxPoolTemplate}.py`.
- `DeeployTest/Platforms/Siracusa/src/deeploytraintest.c` — loop (370-442), `run_optimizer_step` (147), `dump_weights`/`[WDUMP]` (218-238, gated by `DUMP_WEIGHTS` 212), loss compare (252/461), BN_FROZEN_STATS (363).

### B. Glossary

- **n_accum / N_ACCUM_STEPS** — mini-batches accumulated per optimizer step (4 here → SUM of 4 grads = effective batch 1).
- **n_steps / N_TRAIN_STEPS** — optimizer (SGD) update steps (540 = 40 epochs × 54 windows ÷ 4).
- **lazy_reset_grad** — per-forward flag: `1` on the first accum step (reset accumulator to this grad), `0` after (add). Drives `InPlaceAccumulatorV2`.
- **InPlaceAccumulatorV2** — the on-device gradient-accumulation op; makes device training loop-free in C.
- **BatchNormInternal / BN_FROZEN_STATS** — BN op whose device normalization uses the frozen pretrained running stats (train ≡ inference), matching the `--bn-frozen-stats` export.
- **HWC / CHW** — channels-last / channels-first. Forward Conv runs HWC (im2col); backward ConvGrad runs CHW (pulp-trainlib).
- **im2col** — gather receptive fields into contiguous columns so conv becomes a GEMM; costs a transient L1 column buffer.
- **tiling** — splitting a kernel's tensors into L1-sized tiles; solved as an OR-Tools CP problem maximizing tile size.
- **MiniMalloc** — external lifetime bin-packing allocator that assigns buffer offsets within a memory level.
- **double buffering** — ping-pong L1 buffers overlapping DMA-of-next with compute-of-current (2× L1 cost).
- **[WDUMP]** — device weight dump line (raw hex) parsed off the log to build the carry checkpoint.
- **carry checkpoint** — device-trained weights written over the checkpoint, carried into round *r+1* (b1→b5 protocol).
