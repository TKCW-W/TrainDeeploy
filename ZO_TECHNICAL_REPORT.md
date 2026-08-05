# Zeroth-Order (MeZO) On-Device Fine-Tuning on TrainDeeploy — Technical Report

> Scope: porting and productionising **zeroth-order (ZO / MeZO) fine-tuning** of SpeechNet onto the
> Siracusa / GAP9 (PULP, RISC-V) target, end-to-end from graph export (**Onnx4Deeploy**) through code
> generation + tiling + build + cycle-accurate simulation (**TrainDeeploy / Deeploy**). This document is
> self-contained: it explains the concept, the reference material we started from, every change we made and
> why, the challenges + root causes + fixes, the results, and an exact reproduction flow.
>
> Repos / branches: **TrainDeeploy** `feat/zo-support` (HEAD `efc2540`), **Onnx4Deeploy** `feat/ZO`
> (HEAD `e8b687e`). All our edits are tagged `-- QW` in-source.

---

## 1. Conceptual background — what ZO / MeZO is and why we want it on device

**First-order (BP) fine-tuning** computes `∇L` with a backward pass. On a tiny MCU (GAP9: 128 KiB L1, a few
MB L2, no DRAM) the backward pass is the memory bottleneck — it must stash forward activations and allocate
gradient buffers.

**Zeroth-order optimisation (MeZO, "Memory-efficient ZO")** estimates the gradient from **only forward
evaluations** along a single random direction, using the antithetic (two-sided) finite difference:

```
z   ~ Rademacher(±1), one sample per (step, direction), shared across all params
L+  = L(θ + ε·z)                         # "+ε" perturbed forward
L-  = L(θ − ε·z)                         # "−ε" perturbed forward
g_proj = (L+ − L−) / (2·ε·N_accum)       # scalar projected gradient (SPSA estimator)
θ  ← θ − lr · g_proj · z                  # in-place update, same z
```

Key properties that make it attractive on device and that shape the whole implementation:
- **Forward-only.** No backward graph, no gradient tensors, no activation stash. → large L2 saving (see §9).
- **The whole update is one scalar `g_proj` times the shared direction `z`.** The device never materialises a
  gradient tensor; it re-generates `z` on demand from a seed and applies `θ −= lr·g_proj·z` per weight.
- **`z` must be bit-identical** between the two perturbed forwards (`+ε`/`−ε`), between the loss-eval graph and
  the update graph, and between device and the host reference — otherwise the estimator is inconsistent. This
  drove the "RNG bit-exactness" requirement that recurs throughout.
- **Gradient accumulation** (`N_accum > 1`): average `g_proj` over `N_accum` mini-batches before one update.
- **Multi-step:** step *k*'s forward must read the weights that step *(k−1)*'s update wrote.

---

## 2. What is shipped from the reference vs what is our flow

### 2.1 Reference material we started from
| Source | What it provides | How we use it |
|---|---|---|
| **Deeploy `zo-support`** (upstream) | The FP32 `PerturbRademacher` ZO operator: the RNG-bit-exact **perturb kernel** and its **codegen template** + **tile constraint** (with the per-tile `tile_seed_offset`). | **Ported verbatim** into TrainDeeploy (files tagged "Ported from Deeploy zo-support"). This is the primitive that turns a weight `w` into `w + sign·ε·z` inside the graph. |
| **Onnx4Deeploy** (base) | The model-export framework (`BaseONNXExporter`), the **BP training export** path (weights-as-inputs graphs, `BatchNormInternal`, SoftmaxCrossEntropyLoss artifacts), and the SilentWear data source. | Extended with a **ZO export mode** (`-mode zo-train`) on branch `feat/ZO`. |
| **TrainDeeploy / Deeploy** (base) | The tiler (SB-tiler), memory scheduler, code generation, the BP **training** device harness + runner, and `build_shared_buffer_maps` (BP optimizer-graph buffer sharing). | ZO reuses the BP training codegen path wholesale; we added a parallel ZO harness/runner and a two-graph emission. |

### 2.2 Our flow (high level)
```
Onnx4Deeploy (feat/ZO)                         TrainDeeploy (feat/zo-support)
────────────────────────                       ──────────────────────────────
 SpeechNet + fold3 ckpt                          pack_2step_fixture.py  (split into 2 dirs)
   │  -mode zo-train                                   │
   ▼                                                   ▼
 network_zo_train.onnx  (perturbed fwd → loss)   deeployMezoRunner_tiled_siracusa.py
 network_zo_update.onnx (in-place θ update)        → testMVPTraining.py (--zo)  : zo_train  codegen
 inputs.npz  (input,label, 22 weights, meta)       → testMVPOptimizer.py        : zo_update codegen
 outputs.npz (per-step loss_plus/minus, ...)       → build (LLVM/RISC-V) → GVSoC (deeploymezotest.c)
   │  PyTorch reference (_zo_pytorch_reference)                      │
   └──────────────── bit-exactness check ◄─────────────────────────┘
```
The **two graphs share the trainable weight buffers**: `build_shared_buffer_maps` aliases each
`zo_update` output (`{w}_updated`) onto the corresponding `zo_train` weight **input** buffer, so the update is
in-place and step *(k)* reads step *(k−1)*'s result.

### 2.3 File inventory (all `-- QW`)
**Onnx4Deeploy `feat/ZO`:**
- `onnx4deeploy/transform/zo_transform.py` — the ZO graph augmentation: injects Perturb nodes, builds the
  update graph, promotes trainable weights to graph **inputs**, appends the canonical SCE loss.
- `onnx4deeploy/core/base_exporter.py` — `export_zo_training()` orchestration (reuses the inference export for
  the forward + frozen BN).
- `onnx4deeploy/models/speechnet_exporter.py` — SpeechNet ZO fixture generator + multi-step PyTorch reference
  (`_zo_pytorch_reference`, `create_training_test_data_zo`).
- `onnx4deeploy/utils/onnx_node_implementations.py` — `_perturb_rademacher` (the host RNG replica) + the
  pure-Python graph executor (`run_onnx_graph`) used to compute references.

**TrainDeeploy `feat/zo-support`:**
- `TargetLibraries/PULPOpen/src/RandomNoise.c` — the perturb **kernel** (`ApplyRademacherPerturbation`, RNG,
  the overlap-safe fix). `.../inc/kernel/RandomNoise.h` its header.
- `TargetLibraries/PULPOpen/src/ZORuntime.c`, `.../inc/kernel/ZORuntime.h` — the runtime ZO controls
  (`perturbation_sign`, `perturb_seed_base`, `perturb_eps_(use_)override`).
- `Deeploy/Targets/PULPOpen/Templates/FloatPerturbRademacherTemplate.py` — the perturb op **codegen template**
  (per-core chunking, seed derivation, kernel call).
- `Deeploy/Targets/PULPOpen/TileConstraints/PerturbTileConstraint.py` — the perturb **tile constraint**
  (exposes `tile_seed_offset`).
- `Deeploy/Targets/Generic/{Parsers,Layers,TypeCheckers}.py`, `Deeploy/Targets/PULPOpen/{Bindings,Platform}.py`
  — the op registration (parser/layer/checker/binding/mapper) so Deeploy recognises `PerturbRademacher`.
- `DeeployTest/Platforms/Siracusa/src/deeploymezotest.c` — the **device harness**: the ZO training loop
  (±ε forwards → FP-free loss read → g_proj → in-place update), the two-network init + buffer sharing.
- `DeeployTest/testUtils/deeployMezoRunner.py`, `DeeployTest/deeployMezoRunner_tiled_siracusa.py` — the runner
  (`mezo=True` config, `--eps/--lr/--q/--seed`).
- `DeeployTest/testUtils/trainingUtils.py` — `run_zo_codegen` (drives the two-stage codegen), `deferPerturbNodes`
  (perturb liveness reorder), the fixture consumption (now direct — see §7.5).
- `DeeployTest/testMVPTraining.py`, `DeeployTest/testUtils/codeGenerateTraining.py` — training codegen, extended
  to emit `TRAINING_NUM_WEIGHT_INPUTS`/`testInitWeights` for the no-grad ZO graph (§7.4).
- `DeeployTest/Platforms/Siracusa/CMakeLists.txt` — the `MEZO_TRAINING` build branch + `ZO_EPS/ZO_LR/...` defines.
- `DeeployTest/experiments/zo_smoke/pack_2step_fixture.py` — export-dir → two-dir fixture split.
- Fixtures: `DeeployTest/Tests/Models/Training/SpeechNet/speechnet_zo_{train,update}_{2step,8step,exp5}/`.

---

## 3. The two ONNX graphs in detail

The exporter emits **two** graphs plus a reference (`network_infer.onnx` is a byproduct used to source frozen
weights).

### 3.1 `network_zo_train.onnx` — the perturbed loss-evaluation graph
- **Structure:** for every trainable weight `w`, a `PerturbRademacher` node consumes `w` and produces
  `{idx}_{w}` = `w + sign·ε·z` (element-wise). That perturbed tensor feeds the original op (Conv/Gemm/BN).
  After the forward, a canonical 2-output `SoftmaxCrossEntropyLoss` node emits **`[loss, log_prob]`**
  (loss first, scalar; `reduction="mean"`), structurally identical to the BP artifacts.
- **Weights are graph INPUTS** (24 inputs for SpeechNet: `input`, `label`, then 22 weights). Frozen BN
  running-mean/var stay **initializers** (10 of them). BN is unfolded and frozen (`--bn-frozen-stats`) so the
  device normalises with the pretrained running stats (no live batch-1 statistics), matching the host reference.
- **The 22 trainable params** = 10 conv (`blocks_X_0_weight/bias`) + 10 BN γ/β (`blocks_X_1_weight/bias`) +
  2 fc (`fc_weight/bias`). Every one is a Perturb base.
- **Runtime controls** (globals in `ZORuntime.h`, set by the harness before each dispatch):
  `perturbation_sign` (1=+ε, 0=−ε), `perturb_seed_base` (per-step seed), `perturb_eps_use_override` +
  `perturb_eps_override` (unused for the forward; reserved).

### 3.2 `network_zo_update.onnx` — the in-place weight-update graph
- **Structure:** for each trainable weight `w` (a graph **input**), a `PerturbRademacher`-style node applies
  `w_updated = w − lr·g_proj·z` (the same `z`, re-generated from the same seed), emitting `{w}_updated` as a
  graph **output**.
- The update coefficient `−lr·g_proj` is injected on device via `perturb_eps_override` with
  `perturb_eps_use_override=1` (so the "perturb" kernel doubles as the update kernel: same `z`, scaled by
  `−lr·g_proj` instead of `±ε`).
- **Buffer sharing:** `build_shared_buffer_maps(zo_train, zo_update)` matches `zo_update`'s weight input names
  and `{w}_updated` output names against `zo_train`'s weight input names, and `_patch_shared_buffers` redirects
  `zo_update`'s I/O pointers onto `zo_train`'s already-allocated weight buffers → the update writes in place
  where the next forward reads.

### 3.3 `inputs.npz` / `outputs.npz`
- `inputs.npz`: `arr_0000`=input `(1,1,14,700)`, `arr_0001`=label, `arr_0002..arr_0023`=the 22 initial weights
  (graph-input order), `meta_*` (n_batches, n_accum, data_size, zo_eps/lr/q/seed), and per-mini-batch data
  `mb{k}_arr_*` (the device cycles unique samples by modulo).
- `outputs.npz`: the 22 final weights, `log_prob`, and the **reference `loss_plus`/`loss_minus`** — one entry
  per mini-batch — computed by the PyTorch reference `_zo_pytorch_reference` (`model.eval()` → frozen BN;
  verified == ORT). This is the ground truth the device is checked against.

---

## 4. The RNG and bit-exactness contract (the crux)

`z` is a per-element Rademacher `±1` derived from a seed. The device kernel and the host reference must produce
the **same** `z` for the same tensor.

**Per-core chunking + seed (`FloatPerturbRademacherTemplate.py`, `RandomNoise.c`):** each of `NUM_CORES=8`
cluster cores perturbs a contiguous chunk of the flat tensor:
```
chunk       = (size >> log2(NUM_CORES)) + ((size & (NUM_CORES-1)) != 0)
chunk_start = MIN(chunk*core_id, size);  chunk_stop = MIN(chunk_start+chunk, size)
chunk_seed  = ((baked_seed + perturb_seed_base) + NUM_CORES*node_id + core_id) ^ (tile_seed_offset*0x9E3779B1)
```
The kernel seeds a xorshift RNG per chunk and extracts **1 bit per element, 32 bits per xorshift call**
(`ApplyRademacherPerturbation`): `dest[i] = src[i] + ((bits&1)?+1:−1)·ε`. The host `_perturb_rademacher`
replicates this exactly (same `NUM_CORES`, same per-core chunking, same 32-bit batching), which is why the
device matches to the bit.

- `node_id` (the Perturb node's `idx` attribute) is **consistent across `zo_train` and `zo_update`**
  (Onnx4Deeploy commit `c1826f7`) so the update re-generates the same `z` the forward used.
- `tile_seed_offset` is the flat global element offset of a tile's first element, so a tensor that **tiles**
  gets a different RNG per tile. For SpeechNet every perturbed tensor is **single-tile** (all `tile_seed_offset=0`),
  so this path is currently untriggered but implemented (open item, §10).

---

## 5. The device harness (`deeploymezotest.c`) — the on-device training loop

Conceptually the harness is the ZO optimiser loop; technically it is a Fabric-Controller (FC) program that
dispatches cluster tasks. Per update step:

```
for update_step in 0..N_TRAIN_STEPS:
  acc_bits = 0                                   # FP accumulator held as raw bits (FC has NO FPU — see §7.1)
  seed_base = update_step * ZO_Q
  for accum_step in 0..N_ACCUM_STEPS:
    load mini-batch data+labels (cycle by modulo)
    perturbation_sign=1; perturb_seed_base=seed_base; perturb_eps_use_override=0
    RunTrainingNetwork()                         # +ε forward   → L+
    lp_bits = raw bits of DeeployNetwork_outputs[0]
    perturbation_sign=0
    RunTrainingNetwork()                         # −ε forward   → L−
    lm_bits = raw bits of loss
    acc_bits = AccumulateZODiffOnCluster(acc_bits, lp_bits, lm_bits)   # (L+ − L−) summed on cluster (has FPU)
  coeff = ComputeZOUpdateCoeffOnCluster(acc_bits)  # −ZO_LR·g_proj  (g_proj = Σ(L+−L−)/(2·ZO_EPS·N_ACCUM))
  perturb_eps_use_override=1; perturb_eps_override=coeff
  RunOptimizerNetwork()                          # θ −= lr·g_proj·z, in place, into the shared buffers
```
- **Two-network init:** `InitTrainingNetwork` loads `zo_train`; `InitOptimizerNetwork` loads `zo_update`; the
  buffer-sharing redirection makes `zo_update`'s weight I/O point at `zo_train`'s weight buffers.
- **Weight load:** because the weights are graph **inputs**, the harness copies the initial values from
  `testInitWeights[]` into `DeeployNetwork_inputs[num_data+wi]` at boot (the block guarded by
  `TRAINING_NUM_WEIGHT_INPUTS`).
- **FP-free FC:** all float arithmetic (the loss accumulate, the coeff) is done in **cluster** helper tasks
  (`*OnCluster`); the FC only moves raw `uint32` bits (see §7.1).

---

## 6. Codegen / build / runner wiring

- **Runner:** `deeployMezoRunner_tiled_siracusa.py` → `testUtils/deeployMezoRunner.py::main(tiling_enabled=True)`.
  Near-clone of the BP training runner but sets `mezo=True` on the config and adds `--eps/--lr/--q/--seed`.
- **`mezo=True`** routes `testUtils.core.execution.generate_network` to `trainingUtils.run_zo_codegen`
  (checked before the BP `run_training_codegen`), which:
  1. Stage 1: `testMVPTraining.py --zo -t <zo_train_dir>` — deploys `zo_train`; `--zo` swaps the outputs-header
     emitter to `loss_plus`/`loss_minus`.
  2. Stage 2: `testMVPOptimizer.py -t <zo_update_dir> --training-dir=<zo_train_dir>` — deploys `zo_update` and
     runs `build_shared_buffer_maps` so its weight I/O aliases `zo_train`'s buffers.
- **Build:** `CMakeLists.txt` `MEZO_TRAINING` branch compiles `deeploymezotest.c` and passes
  `-DZO_EPS/-DZO_LR/-DZO_Q/-DZO_SEED/-DN_TRAIN_STEPS/-DN_ACCUM_STEPS/-DTRAINING_NUM_DATA_INPUTS`
  (from `trainingUtils.add_mezo_cmake_flags`). Note `ε` is baked into the ONNX Perturb node; `--lr` only feeds
  the FC-side `g_proj` denominator / update coeff.
- **`deferPerturbNodes`** (`trainingUtils.py`): reorders the toposorted node list so each Perturb runs just
  before its consumer (Deeploy's toposort otherwise hoists all 22 perturbs to the front). Memory-negligible at
  a 2 MB L2 (the tiler's liveness already frees each perturbed copy at consumption — 788,840 B with vs 791,180 B
  without, identical tile counts); it only matters under a tight L2. Kept as a safety valve.

---

## 7. Challenges, root causes, and fixes

Each entry: **symptom → root cause → why the fix works → file/commit.**

### 7.1 FP on the FPU-less Fabric Controller → illegal-instruction trap
- **Symptom:** the sim appeared to hang / run ~500× slow; a killed run showed an empty log.
- **Root cause:** GAP9's **Fabric Controller** (which runs `main()`) has **no FPU**; the cluster cores do. Any
  float op in the harness `main()` — even `float lp; stored[mb]=lp;`, which the compiler lowers to `flw/fsw` —
  traps to `pos_illegal_instr` (PC `0x1c00808c`, a `j`-to-self) and spins forever. GVSoC buffers stdout until
  `main()` returns, so a killed run shows nothing.
- **Fix:** the FC handles **only raw `uint32` bits** (memcpy, never a float op); all FP (loss accumulate,
  `g_proj`, update coeff) is done in **cluster** helper tasks. `printf("%f")` is one instance of the same trap
  (soft-float); avoided.
- **Where:** `deeploymezotest.c` (`AccumulateZODiffOnCluster`, `ComputeZOUpdateCoeffOnCluster`, raw-bit loss
  reads). Commit `430990d`. *General GAP9 gotcha, not ZO-specific.*

### 7.2 `_foldLayoutIntoPerturb` reordered the Rademacher z (single-step correctness bug)
- **Symptom:** single-step device loss diverged from the reference for conv layers with `in_ch>1`
  (conv1–4), while conv0 (`in_ch=1`) matched.
- **Root cause:** an optimisation folded the NCHW→NHWC layout permutation **into the Perturb's constant base
  weight**, so the perturb kernel generated `z` over the *transposed* flat buffer while the reference perturbs
  in logical NCHW order. Harmless for `in_ch=1`, wrong for `in_ch>1`.
- **Fix:** disable the fold — the perturb runs in logical NCHW order followed by a **runtime `Transpose`**
  (exactly the BP path, `_appendTranspose`). The fold's claimed "~100× slowdown" rationale was actually §7.1.
- **Where:** `Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py`. Commit `430990d`.

### 7.3 Multi-step update did not propagate — `fc_bias` perturb L1 **buffer overlap** (the hard one)
- **Symptom:** the **2-step** test (the first test where step-1 reads a post-update weight) diverged only at
  **step 1**; step-1 losses matched a *no-update* reference. Then, after fixing propagation, a **~0.002 bias
  on step-0** appeared — and step-0 has no update, so with identical weights it *must* be bit-exact.
- **Diagnosis (differential trace, all instrumentation later stripped):** dumped, on device, every weight
  input (all 22 correct), the *perturbed* `fc_weight` (matched the reference exactly), the GlobalAveragePool
  input and output features (bit-exact), and finally the **fc Gemm's actual read inputs** — which showed the
  bias `C` **corrupted on exactly the odd indices** `{1,3,5,7}`. Dumping the perturbed `fc_bias` L2 tensor
  proved the **perturb output itself** was wrong (not a downstream DMA).
- **Root cause:** `fc_bias` is the **only** trainable param whose length (9) is not a multiple of
  `NUM_CORES=8`. For that odd-length 1-D tensor the tiler placed the perturb's L1 `data_out` only **+4 B
  (1 float) into `data_in`** — the two 36-byte buffers **overlap**. The element-wise perturb writes
  `data_out[i]` (= `data_in[i+1]`) before reading it (across cores too), so each odd element reads an
  already-perturbed value → odd `fc_bias` wrong → odd fc logits wrong → the whole ~0.002 bias.
  **Input-form specific:** in the initializer form `fc_bias` is a baked constant on a non-tiled path, so it was
  bit-exact — which is exactly why step-0 in init-form gave the confidence that input-form *should* match.
- **Why not fix the tiler:** forcing full-size perturb tiles is **infeasible** — the perturb sees every tensor
  as flat, and both weights (feed the tiled conv) and biases (feed tiled consumers) can't be forced full-size
  (the constraint solver returns infeasible).
- **Fix (kernel, overlap-safe):** in `ApplyRademacherPerturbation`, when `data_out` sits just ahead of
  `data_in` (a small forward overlap — the decision is identical on every core since `dest−src` is a constant
  offset), each core **copies its chunk to a private temp**, a `pi_cl_team_barrier(0)` guarantees **all reads
  finish before any write**, then it perturbs from the temp. Large tensors (`data_out` far from `data_in`)
  take the unchanged fast path. Because the RNG stream/order is untouched, the result is **bit-identical** to
  the non-overlapping path — it just removes the read-after-write hazard.
- **Where:** `TargetLibraries/PULPOpen/src/RandomNoise.c::ApplyRademacherPerturbation`. Commit `186a974`.
  Full trail: `experiments/zo_smoke/STEP0_BIAS_FINDINGS.md`.

### 7.4 Promoted weight inputs were not loaded / update-propagation plumbing
- **Symptom:** with weights as inputs, the device produced **NaN** (weight buffers uninitialised); and the
  buffer-sharing map came back **empty**.
- **Root cause A:** `codeGenerateTraining.generateTrainingTestInputsHeader` only emitted
  `TRAINING_NUM_WEIGHT_INPUTS`/`testInitWeights` when `num_grad_inputs > 0` (BP has grad-accumulation buffers;
  the ZO graph has **none**), so the harness never copied the initial weights.
- **Root cause B:** `testMVPTraining` derived `init_weights` as `npz[num_data:grad_buf_start]`; with no grad
  buffers `grad_buf_start = −1` → `init_weights = []`.
- **Fix:** (A) emit `TRAINING_NUM_WEIGHT_INPUTS` whenever `grad_buf_start_idx > num_data` (not only when grad
  buffers exist); (B) when there are no grad buffers, treat *all* inputs after the data inputs as weights
  (`init_weights = npz[num_data:]`, `grad_buf_start_idx = len(inputs)`).
- **Where:** `DeeployTest/testUtils/codeGenerateTraining.py`, `DeeployTest/testMVPTraining.py`. Commit `186a974`.
- **Buffer sharing itself** already worked once the weights were inputs: `build_shared_buffer_maps` matches
  `zo_update` weight names against `zo_train` **graph inputs** (empty in the initializer form — the reason it
  no-op'd before).

### 7.5 Where the "make weights inputs" step lives (clean flow)
- **History:** during debugging we first had the exporter emit weights as **initializers** (to match the demo
  "reference design") and promoted them to inputs at **deploy time** in TrainDeeploy
  (`_prep_zo_train_for_deploy`). That validated the mechanism but left a promotion step in the runner.
- **Now:** Onnx4Deeploy emits `zo_train` weights as **inputs** directly (`zo_transform._promote_initializers_to_inputs`
  re-enabled), so TrainDeeploy consumes the fixture **directly** (`run_zo_codegen` uses `config.test_dir`; the
  interim `_prep_zo_train_for_deploy` was removed; `pack_2step_fixture.py` is now a pure train/update split).
  We proved the deployed graph is **identical** (bit-exact 2-step/8-step) before and after moving it, so this is
  a no-op functionally — purely a cleaner flow.
- **Where:** Onnx4Deeploy `zo_transform.py` (commit `e8b687e`); TrainDeeploy `trainingUtils.py` + pack script
  (commit `b164df9`).

### 7.6 Miscellaneous
- **Perturb node liveness** (§6, `deferPerturbNodes`): Deeploy's toposort hoists all perturbs to the front;
  we reorder each to just before its consumer to cap peak liveness. Net memory effect is small at 2 MB L2 but
  it is a correctness-neutral safety valve for tight budgets.
- **lr / divergence:** `ε` is baked into the Perturb node (0.01); `--lr` only scales the update. `lr=1e-3`
  diverges to NaN by ~step 11 (fine for 2 steps, borderline at 8) — **use `lr=3e-6` for long runs** (exp5).
- **`BENCH train_cycles` overflow:** the cycle counter is `uint32` and wraps on long runs (~14.2 B → 1.37 B in
  exp5). Cosmetic; correctness unaffected.
- **Fixture data path:** the SilentWear data moved to `/app/SilentWear/SilentWear_data`; the exporter expects
  `/app/SilentWear_data` — restore with `ln -sfn /app/SilentWear/SilentWear_data /app/SilentWear_data`.

---

## 8. Results

All runs: full flow (Onnx4Deeploy fixture gen → TrainDeeploy MeZO runner → GVSoC, cycle-accurate RISC-V),
weights-as-inputs, pass tolerance `1e-3` absolute on each per-mini-batch loss.

| Experiment | Config | Result |
|---|---|---|
| Single-step | 1 step, `n_accum`=1, all 22 params | **bit-exact**, `Errors 0/2`, ~35.6 M cyc/step |
| 2-step | 2 steps, `n_accum`=1, lr 1e-3 | **step-0 bit-exact** (0.016488 / 0.069274), step-1 within **1e-6**, `Errors 0/4` |
| 8-step | 8 steps, `n_accum`=2, lr 3e-6 | `Errors 0/32`, max diff **3e-6** |
| **exp5_zo** | 100 epochs, `n_accum`=4, data_size 4 → 100 steps → 400 mb → **800 forwards**, lr 3e-6 | **`Errors 0/800`**, max diff **3.5e-5** — FP drift stays bounded over the whole run |

**Memory (single-step, `--plotMemAlloc`):** L2 arena **788,840 B (≈39 % of the 2 MB budget)**, L1
**127,232 B (≈99 % of 128 KiB)**. **The win vs BP on-device FT:** ZO uses **~0.79 MB L2 vs ~1.5 MB for BP FT**
(exp4) — roughly **half** — because it is forward-only + perturb (no backward pass ⇒ no gradient-accumulation
buffers, no stashed activations for backprop). L1 is comparably tight (both ~99 %, the largest conv block's
weight+activation+output tiles dominate); the saving is entirely in L2.

**Bit-exactness:** step-0 (no update) is exactly equal; multi-step accumulates only float rounding (≤3.5e-5
over 800 forwards), i.e. the pipeline is numerically faithful, not merely "within tolerance by luck".

---

## 9. Reproduction flow (exact commands)

Two Docker containers (repos are bind-mounted, same files):
- **`agitated_hugle`** — Onnx4Deeploy at `/app/Onnx4Deeploy` (has `onnxscript` / onnxruntime-training for export).
- **`traindeeploy`** — TrainDeeploy at `/app/ETH/TrainDeeploy` (LLVM/RISC-V toolchain + GVSoC).

Constants:
```
CKPT=/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
EXPORT_BASE="--noise-type rademacher --bn-frozen-stats --dataset silentwear --pretrained-weights $CKPT --subject S01 --session 3 --condition vocalized"
```

### Step 0 — data path (once, in agitated_hugle)
```bash
docker exec agitated_hugle bash -lc 'ln -sfn /app/SilentWear/SilentWear_data /app/SilentWear_data'
```

### Step 1 — generate the ZO fixture (Onnx4Deeploy, agitated_hugle)
2-step / n_accum 1 (lr 1e-3):
```bash
docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && \
  python3 Onnx4Deeploy.py -model SpeechNet -mode zo-train $EXPORT_BASE \
    --n-steps 2 --n-accum 1 --lr 0.001 -o ./onnx/model/speechnet_zo_2step"
```
8-step / n_accum 2 (lr 3e-6): `--n-steps 8 --n-accum 2 --lr 3e-6 -o ./onnx/model/speechnet_zo_8step`
exp5 — 100 epochs / n_accum 4 (lr 3e-6): `--n-epochs 100 --data-size 4 --n-accum 4 --lr 3e-6 -o ./onnx/model/speechnet_zo_exp5`
Produces in the `-o` dir: `network_zo_train.onnx` (24 inputs), `network_zo_update.onnx` (22 inputs),
`inputs.npz`, `outputs.npz` (reference losses).

### Step 2 — package into TrainDeeploy's two-dir layout (traindeeploy)
```bash
docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && \
  python3 experiments/zo_smoke/pack_2step_fixture.py \
    /app/ETH/Onnx4Deeploy/onnx/model/speechnet_zo_2step \
    Tests/Models/Training/SpeechNet \
    speechnet_zo_train_2step speechnet_zo_update_2step"
```
(train dir gets `network_zo_train.onnx`→`network.onnx`; update dir gets `network_zo_update.onnx`→`network.onnx`;
both get the same `inputs.npz` + `outputs.npz`.)

> ⚠️ **The fixture is two dirs — name the train dir with `_train` and give the runner BOTH paths.** The MeZO
> runner needs the update (optimizer) graph from a **separate directory**, so Step 4 always passes it explicitly
> via `--optimizer-dir …_update_2step`. If you omit `--optimizer-dir`, the runner auto-derives it by replacing
> `_train`→`_optimizer` in the `-t` name (`resolve_optimizer_dir`), which for a `…_train_…` dir yields a
> `…_optimizer_…` sibling — so the `-t` dir **must** contain `_train` for that fallback to resolve. If the `-t`
> name has no `_train` (e.g. `…/zo_2step`), the auto-derive resolves to the *same* dir and the runner loads the
> perturbed-forward `zo_train` graph as the update graph → generation fails. Keep the `pack` script's
> `<name>_train_… / <name>_update_…` pair and pass both.

### Step 3 — kill any orphan GVSoC before each run (traindeeploy)
```bash
docker exec traindeeploy bash -lc 'pgrep -f gvsoc_launcher | xargs -r kill -9; echo done'
```

### Step 4 — codegen → build → GVSoC (traindeeploy)
2-step:
```bash
docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && \
  python3 deeployMezoRunner_tiled_siracusa.py \
    -t              Tests/Models/Training/SpeechNet/speechnet_zo_train_2step \
    --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_zo_update_2step \
    --n-steps 2 --n-accum 1 --num-data-inputs 2 \
    --eps 0.01 --lr 0.001 --q 1 --seed 42 \
    --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8"
```
8-step: `-t …_train_8step --optimizer-dir …_update_8step --n-steps 8 --n-accum 2 --lr 3e-6` (rest identical).
exp5:   `-t …_train_exp5 --optimizer-dir …_update_exp5 --n-steps 100 --n-accum 4 --lr 3e-6` (rest identical).
Memory plot: append `--plotMemAlloc` (and `--skipsim` for codegen-only) → `memory_alloc.html` under
`TEST_SIRACUSA/Tests/Models/Training/SpeechNet/<train_dir>/deeployStates{,_optimizer}/`.

Flag reference (runner): `--n-steps` (`N_TRAIN_STEPS`), `--n-accum` (`N_ACCUM_STEPS`), `--num-data-inputs`
(data inputs that change per mb; 2 = input+label), `--eps` (bakes nothing — perturb ε is in the ONNX; only feeds
the FC g_proj denominator), `--lr`, `--q` (dirs/step), `--seed`, `--l1`/`--l2` (byte budgets), `--defaultMemLevel`,
`--cores`, `--plotMemAlloc`, `--skipsim`.

**Expected output:** `[loss+ k] computed=… ref=… diff=…`, `[loss- k] …`, `Errors: 0 out of <2·n_batches>`,
`BENCH train_cycles=… opt_cycles=…`. Decode a raw device loss: `struct.unpack('<f', struct.pack('<I', bits))`.

---

## 10. Remaining work / known limitations
1. **`tile_seed_offset` for tensors that tile** — implemented in `PerturbTileConstraint` but untriggered for
   SpeechNet (all perturbed tensors single-tile). Validate before any model with a perturbed tensor large
   enough to tile: the reference `_perturb_rademacher` currently assumes offset 0.
2. **Harness NaN handling** — the loss comparison treats a NaN diff as a pass; flag NaN as an error (surfaces
   when lr is too high and training diverges).
3. **`BENCH` cycle counter is uint32** — wraps on long runs; widen to uint64 for honest cycle reporting.
4. **Strip residual debug** — `ZTRACE` phase markers + loss-bit dumps in `deeploymezotest.c` (all `-- QW`).
5. **Full-accuracy training stays in the PyTorch exp18 sim** — GVSoC is too slow for the 200-epoch recipe at a
   real lr; the on-device tests validate **functional/numeric correctness**, and exp18 reports the accuracy
   (ZO 87.36 ≈ BP 86.11 ≈ paper fold-3 87.64).

---

### Change log (our commits)
- Onnx4Deeploy `feat/ZO`: `c4ed3c0` port MeZO + BN γ/β · `c1826f7` consistent Perturb node_id across graphs ·
  `db51551` multi-step SpeechNet fixture · `f58b0c4` initializer-form + sim param sourcing · `e8b687e` **inputs-form**.
- TrainDeeploy `feat/zo-support`: `430990d` single-step bit-exact (FP-FC trap + fold z-reorder) ·
  `186a974` **fc_bias buffer-overlap fix** + weight-input emission + 2/8-step · `b164df9` consume inputs-form
  directly (remove deploy-prep) · `efc2540` **exp5_zo** (0/800).
