# Quantized Zeroth-Order (QZO) On-Device Training — Wrap-Up

**Status (2026-08-28, branch `feat/QZO` in Onnx4Deeploy + TrainDeeploy):**
- **exp7**: single-step QZO smoke test on Siracusa/GVSoC — device `L+`/`L−` **bit-exact** to the host
  reference (`diff = 0.000000`), real SilentWear data + calibration, pretrained fold-3 checkpoint.
- **exp8**: 8 antithetic pairs, `n_accum=4` → **2 in-place weight updates executed on device** — all 16
  losses match (15 bit-exact, 1 at `1e-6`); the post-training weights were **dumped from the device** and
  20/22 tensors match the host prediction bit-exactly (2 fp32 tensors at 1 ulp).

Deliverables: `exp7_QZO_single_step/`, `exp8_QZO_8steps_4accum/` (fixture + logs + REPRODUCE.md each).
Raw debugging log: `Onnx4Deeploy/QZO_exp/Report.md` (iterations 1–20).

---

## 1. Algorithm level: what "quantized ZO" means

### 1.1 MeZO recap (why ZO on an MCU at all)

Zeroth-order (MeZO-style) training estimates the gradient from **two forward passes** instead of a backward
pass. Per step, draw a random Rademacher direction `z` (each element ±1), and probe the loss antithetically:

```
L+ = L(θ + ε·z ; x)         L− = L(θ − ε·z ; x)
g_proj = (L+ − L−) / (2ε)                       # directional derivative along z
θ ← θ − lr · g_proj · z                          # step along z, scaled by the projection
```

Only the **seed** of `z` must be remembered — the direction is regenerated for the update. No backward
graph, no stored activations, ~zero training memory overhead: this is what makes on-device training on a
GAP9-class MCU feasible (already validated for the fp32 model: float-ZO exp3/exp5/exp6).

### 1.2 What quantization changes

On the deployed network the parameters are not fp32 — they live on **integer grids**:

| parameter group (SpeechNet: 22 tensors) | representation | grid quantum (float space) |
|---|---|---|
| 5 conv weights | **int8** `w_int`, per-output-channel scale `s_w[c]=max(\|W[c]\|)/127` → `w = w_int·s_w[c]` | `s_w ≈ 0.001–0.004` (measured from fixture) |
| 5 conv biases | **int32** at the *accumulator* scale `s_b = s_in·s_w[c]`, folded into the requant `add` (see §3.2) | `s_b ≈ s_w/5` (finer by `1/s_in`, `s_in≈0.2`) |
| 10 BN γ/β + 2 fc (weight, bias) | **fp32** | none |

(Why int32 bias: the conv accumulates int8·int8 over `in_ch·kh·kw` taps into an **int32 accumulator** at
scale `s_in·s_w[c]`; the bias is added there *before* requant, so it must live at that fine scale —
`b_int32 = round(b/(s_in·s_w))` reaches thousands–millions, far past int8's ±127. Universal PTQ convention.)

Three consequences define quantized ZO:

1. **The perturbation must be an integer.** "±ε in float space" becomes a per-channel integer step
   `round(ε/s_w[c])` (≈ ±8 LSBs for our convs — comfortably above the grid, so the *probes* are exact and
   informative). Encoded as a per-channel magnitude vector `mul[c] = round(ε/s_w[c]·2^S)` and applied by an
   integer kernel: `noise = (±mul[c] + 2^(S−1)) >> S`.
2. **The update must also round to the grid** — the weight step in LSBs is `|coeff|/s_w ≈ 9.4e-5/0.0015 ≈
   0.06 LSB`, below the 0.5 rounding threshold. Result (empirically verified by the on-device weight dump,
   exp8): the **conv weights stall at exactly 0 LSB** (sub-LSB stall), the **conv biases** (grid `1/s_in≈5×`
   finer → step `≈0.35–0.6 LSB`) genuinely tick **±1..±7 LSBs**, and the **fp32 params train normally**. The
   stall is *per-parameter, scale-dependent* — not a bug: host and device compute the identical rounding.
   Note the margin is modest (~8×): a ~10× higher lr or more accumulation would start flipping weight LSBs —
   which is what the planned remedy, **master weights** (fp32 shadow copies that accumulate sub-LSB updates
   and re-quantize with the frozen scale), exploits.
3. **All quantization scales are frozen at export**: per-tensor activation scales (from PTQ calibration on
   real data), per-channel weight scales, and the fused requant constants. Nothing is per-sample and
   nothing is updated at runtime (out-of-range activations simply clip at ±127). "Re-calibration" = re-run
   the export with new data — which the inter-session fine-tuning workflow does naturally anyway.

### 1.3 Mixed parameter handling

Because the parameter set is mixed, the graph uses two perturbation op families:
- `RQSPerturbRademacher` (integer): int8 conv weights, int32 conv biases — takes the `mul[c]` vector.
- `PerturbRademacher` (float): BN γ/β, fc weight/bias — takes `ε` directly.

Both share the seed schedule, so the *same* logical `z` covers all 22 tensors of a probe.

---

## 2. What the shipped reference provides — and what it doesn't

The shipped Deeploy (`ETH/Deeploy`, read-only) + the shipped quantized-ZO fixture **QMCUNetZO** give us:

- The **int8 inference machinery**: `Quant`/`Dequant` op (parser/kernel/binding), per-channel
  `RequantShift`, the `Conv+RequantShift → RequantizedConv` merge pass, the PULP int8 conv kernels.
- The **canonical graph convention** (from QMCUNetZO): *quantize the conv layers only* — each conv is
  **2-input** (`data_in, weight`) with the bias absorbed into the requant `add`; the **classifier stays
  fp32** (3-input float Gemm, float-perturbed).
- The **RQSPerturb kernels** (`RandomNoiseQuant.c`) and a validation of a **single perturbed int8 forward**
  (shipped `testMVP.py --run_mode mezo_training` compares one `log_prob` against a reference).

What shipped does **not** provide (all of it built in this project):

| capability | shipped | ours |
|---|---|---|
| perturbed int8 forward (L±) | ✅ (single forward) | ✅ bit-exact vs host, antithetic pair |
| gradient-scaled update (−lr·g_proj) | ❌ no mechanism (update graph re-applies fixed ±ε; no lr anywhere) | ✅ runtime `eps_scale` for integer params (new) |
| on-device training loop (accumulate, update, iterate) | ❌ no harness | ✅ `deeploymezotest.c` (2 updates validated) |
| host reference of the full loop | ❌ | ✅ device-faithful sim in the exporter |
| weight-level validation | ❌ | ✅ on-device dump vs host `updated_*` |

The sub-LSB stall was therefore **never confronted by the shipped work** — it only becomes reachable once a
gradient-scaled multi-step loop exists.

---

## 3. The end-to-end flow

### 3.1 Onnx4Deeploy export (`-mode q-zo-train`)

One CLI command produces the whole fixture:

```bash
python3 Onnx4Deeploy.py -model SpeechNet -mode q-zo-train --noise-type rqs_rademacher \
  --dataset silentwear --data-path .../data_raw_and_filt --pretrained-weights .../fold_3.pt \
  --subject S01 --session 3 --condition vocalized --batch 1 \
  --data-size 8 --n-accum 4 --lr 1e-5 -o QZO_exp/exp8
```

Pipeline (`base_exporter._export_qzo_training`):

1. **Brevitas model** (`QuantSpeechNet`, `speechnet_quant.py`): int8 per-channel conv weights, int32 bias,
   int8 per-tensor activations, **fp32 unfolded BatchNorm**; load the pretrained checkpoint.
2. **PTQ calibration on real windows** — Brevitas observes activation ranges → the per-tensor activation
   scales. (Real data is essential; see Bug B3.)
3. `exportBrevitas` + `create_quant_pipeline` → the base integer graph (`network.onnx`, intermediate).
4. **`build_int8_forward`** rewrites it into the deployable datapath (per-channel integerization, 2-input
   convs, float fc, `BatchNormInternal`) — see §3.2.
5. **`build_qzo_train_graph`** injects the perturbation ops and promotes all 22 params to **graph inputs**;
   appends the `SoftmaxCrossEntropyLoss`. **`build_qzo_update_graph`** builds the in-place update graph
   (22 in → 22 `*_updated` out) with the *same* per-param node ids (so `z` matches between probe and update).
6. **Host reference sim** — replicates the device loop exactly (§3.4) and writes the fixture:
   `inputs.npz` (positional `arr_NNNN` + `mb{k}` windows + `meta_*`), `outputs.npz` (per-pair
   `loss_plus/minus`, per-step `g_proj`, post-training `updated_*` params).

The perturbation injection, showing how the integer magnitude is derived (`qzo_transform.py`):

```python
for p in _iter_qzo_params(model, scale_map):
    if p["kind"] == "rqs":                                  # int8 weight / int32 bias
        mul = np.round(eps / p["scale"] * p["div"]).astype(np.int32)   # per-channel ε in integer LSBs·2^S
        new_nodes.append(helper.make_node(
            "RQSPerturbRademacher", [pname, mk.name], [f"{pname}_pert"],
            name=f"rqsp_{pname}", domain="mezo", idx=idx, seed=seed, signed=1,
            div=p["div"], n_levels=p["nlev"]))
    else:                                                   # fp32 BN γ/β, fc — float Rademacher
        new_nodes.append(helper.make_node(
            "PerturbRademacher", [pname], [f"{pname}_pert"],
            name=f"pert_{pname}", domain="mezo", idx=idx, seed=seed, eps=eps))
    node.input[in_idx] = f"{pname}_pert"                    # consumer reads the PERTURBED tensor
```

`p["scale"]` = the per-channel quantization scale; `div = 2^S` the fixed-point base (2¹⁵ weights, 2¹⁶
biases); `idx` = the per-param node id that keys the RNG stream (consistent across train/update graphs).

### 3.2 The ONNX training graph

Per conv block (24 graph inputs total = `input`, `label`, 22 params):

```
input fp32 ──Quant(s_act)──► int8 ──►┌───────────────────────────────┐
w_int8 (INPUT)──RQSPerturb──► ŵ ────►│ Conv (2-input, int8)          │──int32──► RequantShift ──int8──►
b_int32 (INPUT)─RQSPerturb──► b̂ ─────────────────────────────────────► (as the requant `add`)
                                     └───────────────────────────────┘
──► Dequant(s_out) ──fp32──► BatchNormInternal(γ̂,β̂; frozen stats) ──► ReLU ──► MaxPool ──► next block
...last block ──► GAP ──► float Gemm(ŵ_fc, b̂_fc) ──► SoftmaxCrossEntropyLoss(logits, label) ──► loss
```

Key design points (each is defensible against "why did you do it this way"):

- **2-input conv, bias in the requant `add`.** The PULP int8 kernel has *no bias argument* — its signature
  is `pulp_nn_conv(data, …, weight, mul, add, …)`. The merge pass blindly concatenates
  (`PULPOpen/TopologyOptimizationPasses/Passes.py`):
  ```python
  _inputs = list(conv.inputs) + list(rqs.inputs[1:])   # (data,weight) + (mul,add) = 4-input RequantizedConv
  ```
  so a 3-input conv (bias kept) yields an unmappable 5-input node. Every shipped int8 test is 2-input.
  The bias therefore lives as the **variable** per-channel `add[c] = round(s_w·b_int·2¹⁶)` — which keeps it
  ZO-perturbable (the merge pass only bakes rounding into a *constant* add).
- **Float classifier head.** Matches QMCUNetZO, and for two hard reasons: PULP has no bare int8 GEMM kernel
  (only float GEMM or the 4-input RQS-GEMM which outputs **int8 logits** — quantizing 9-class logits to 256
  levels would destroy the tiny `L+−L−` differences ZO lives on).
- **Weights as graph INPUTS**, not initializers: they must be *mutable persistent buffers*, shared by name
  between the train and update graphs and updated in place across steps. Initializers are baked constants.
- **Layouts**: `RequantizedConv` and `MaxPool` run NHWC (weights transposed OIHW→OHWI at runtime, because
  the perturbed weight is a variable); `Quant`/BN run NCHW with transpose pairs — all verified correct
  (the perms are exact inverses; QMCUNetZO deploys the same machinery).
- **Perturbation runs in logical NCHW order** *before* the weight transpose, so `z` lands on the same
  elements as the host reference.

### 3.3 TrainDeeploy: from ONNX to a GAP9 binary

`deeployMezoRunner_tiled_siracusa.py -t <train_dir> --optimizer-dir <update_dir> --n-steps N --n-accum A
--eps --lr --q --seed --l1 128000 --l2 2000000 --defaultMemLevel L2`

- **Frontend** (`testMVPTraining.py` → Deeploy parse/lower/bind): every graph input is typed **from the ONNX
  element type**, not inferred from values:
  ```python
  _ONNX_ELEM_TO_PTR = {onnx.TensorProto.FLOAT: float32_t, onnx.TensorProto.INT8: int8_t,
                       onnx.TensorProto.INT16: int16_t, onnx.TensorProto.INT32: int32_t,
                       onnx.TensorProto.INT64: int64_t, onnx.TensorProto.UINT8: uint8_t}
  ```
  (value inference mis-typed int8 weights and the int64 label as float32 → bindings rejected; Bug B4).
  Lowering merges `Conv+RequantShift → RequantizedConv` and inserts the layout transposes; binding matches
  each node to a typed kernel template.
- **Midend**: the tiler solves the L1/L2 memory schedule. `--l1 128000` is required — the default 64 kB
  cannot fit the block-0 conv pattern (~88 kB); the tiler's "geometrical constraints infeasible" error means
  *memory*, not geometry.
- **Backend**: templates emit C; the fixture values are baked (`testDataVector` per mini-batch,
  `testInitWeights` for the 22 params — positionally mapped, which is why the npz key convention matters,
  Bug B1); CMake+GVSoC build and simulate.

### 3.4 The device training loop (harness `deeploymezotest.c`)

```c
for (update_step = 0; update_step < N_TRAIN_STEPS; update_step++) {
  seed_base = update_step * ZO_Q;                    // per-step direction z
  acc = 0.0f;                                        // fp32, on the cluster (FC has no FPU)
  for (accum_step = 0; accum_step < N_ACCUM_STEPS; accum_step++) {
    load mini-batch (update_step*N_ACCUM + accum_step);
    perturbation_sign = 1; perturb_seed_base = seed_base;  RunTrainingNetwork();  // L+
    perturbation_sign = 0;                                 RunTrainingNetwork();  // L−
    acc += (Lp − Lm);
  }
  g_proj = acc / (2.0f * ZO_EPS * N_ACCUM_STEPS);
  perturb_eps_override = -ZO_LR * g_proj;            // the runtime update coefficient
  perturb_eps_use_override = 1;  perturb_seed_base = seed_base;   // SAME z as the probes
  RunOptimizerNetwork();                             // in-place θ ← θ − lr·g_proj·z (quantized rounding)
}
```

The runtime controls are ZORuntime globals (`perturbation_sign`, `perturb_seed_base`,
`perturb_eps_override/_use_override`, and the new `perturb_eps_baked = ZO_EPS`). The **integer** update is
the piece we had to build (shipped had no coefficient path for integer params): the RQSPerturb templates
compute

```c
float32_t eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
```

and the kernels scale the per-channel magnitude on the fly (`RandomNoiseQuant.c`):

```c
int32_t m_val = M[(start_offset + i) / channel_width];              // per-OUTPUT-channel (see Bug B5)
if (eps_scale != 1.0f) m_val = (int32_t)lrintf((float32_t)m_val * eps_scale);
int32_t noise_q = (r * m_val + rounding) >> S;                      // r = ±1 (z), S = log2(div)
pweights_dest[i] = CLAMP((int32_t)pweights[i] + noise_q, -127, 127);
```

`channel_width` = elements per output channel (2048/16 = 128 for a conv weight; 1 for a bias vector), so
`i / channel_width` is the output-channel index into the C-entry `M`. With `eps_scale = coeff/ε` the same
kernel serves both the ±ε probes (`eps_scale = 1`) and the −lr·g_proj update.

### 3.5 The host reference (why "bit-exact" is even possible)

The reference is **not PyTorch** — it is `run_onnx_graph`, a pure-Python executor that runs the *identical
ONNX graph* with device-faithful semantics: integer conv, RequantShift with **truncation** (no rounding when
the `add` is a runtime variable — matching the merge pass), Quant with **round-half-away-from-zero**
(matching the C `(int)(v + 0.5·sign)`), the same Rademacher RNG (xorshift32, per-core chunking, 32-bit
batches), and — for multi-step — the device's exact fp32 **operation order**:

```python
_denom = np.float32(np.float32(np.float32(2.0) * np.float32(eps)) * np.float32(n_accum))
g_proj = np.float32(acc / _denom)
coeff  = np.float32(np.float32(-lr) * g_proj)      # device: -(float)ZO_LR * g_proj
ratio  = float(np.float32(coeff) / np.float32(eps))
# integer update mirror of the kernel's lrintf scaling:
mul_flat = np.rint(mul_flat.astype(np.float32) * np.float32(eps_ratio)).astype(np.int64)
```

Because host and device implement the *same arithmetic*, any device deviation is a bug — that discipline is
what let us drive `diff` to exactly 0.000000 and localize every discrepancy to a specific kernel.

---

## 4. Bugs and pitfalls — symptom → diagnosis → fix

Five independent device-vs-host bugs were found and fixed (each masked the next; full log:
`QZO_exp/Report.md` iters 12–20).

**B1 — npz key ordering (weights loaded into the wrong buffers).** Symptom: device logits exploded ~2000×.
Localized by probing layer-by-layer to the first divergent op (BatchNorm) and enabling a kernel debug print:
the BN read `γ=±0.01, β=[60,−4,−127,…]` — *the conv int8 weights*. Root cause: the fixture used
parameter-name npz keys, but `testMVPTraining` loads base keys **sorted** and maps them **positionally** to
graph inputs — alphabetical ≠ graph order. Fix = the float-ZO convention:
```python
_np.savez("inputs.npz", **{f"arr_{gi:04d}": _val_by_name[inp.name]
                           for gi, inp in enumerate(ztrain_graph.input)})   # graph-input order
```

**B2 — Quant scale convention (int8 input saturated).** Symptom: device int8 conv input
`[-31,-128,-128,…]` vs host `[0,-1,-1,…]`. The device `QuantTemplate` computes
`scaled_val = input_val * ${scale}  // Multiply instead of divide` — it expects the **reciprocal**; the
shipped `QuantPatternPass` indeed emits `scale_value = 1.0/divisor`, but our `create_quant_pipeline` graphs
carry the true scale. Fix in `QuantParser`:
```python
self.operatorRepresentation['scale'] = 1.0 / float(node.attrs['scale'])
```
(Defense note: a QuantPatternPass-fused graph would double-invert — documented; long-term fix is emitting
the reciprocal at export.)

**B3 — silent random-calibration fallback (bad activation scales).** The calibration call passed a 3-D
shape where the data source asserts a 4-D per-window shape → the `except` silently fell back to *random*
calibration → saturating activation scales (host loss 9.07 instead of 1.4). Fix: pass `(1,)+ishape[1:]`;
lesson: **never let calibration fail silently** — the export log must not contain "using random calibration".

**B4 — graph-input dtype inference.** `testMVPTraining` inferred buffer types from npz values → int8/int32
params and the int64 label became `float32_t` → the typed bindings (RQSPerturb, RequantizedConv, SCE)
rejected them, failing the frontend mapping at L2. Fix: type every input from the ONNX `elem_type`
(`_ONNX_ELEM_TO_PTR`, §3.3), mirroring `testMVPOptimizer`.

**B5 — RQSPerturb per-channel indexing (the final and subtlest one).** Symptom after all of the above:
losses ~4–10 vs host ~1.3. Found via the earliest-divergence discipline: MaxPool ✓ → Quant ✓ → perturbed
weight ✓ → **perturbed bias ✗**: device `[39,39,−39,…]` (=`M[0]` broadcast, signs correct) vs host
per-channel `[39,31,−31,−26,…]`. Root cause: the kernels indexed `M[(start_offset+i) % channel_width]`
(channels-*last* modulo) while the tile constraint supplies `channel_width` = elements-per-channel
(channels-*first*): for the bias (16 elems / 16 ch → `channel_width=1`) that broadcasts `M[0]`; for the
weight (`channel_width=128`) it reads `M` **out of bounds**. Shipped Deeploy documents this exact class
(`_CHW_ChannelFirst` was added because "the '%' kernel reads M out of bounds"). Fix on both sides:
```c
int32_t m_val = M[(start_offset + i) / channel_width];   // device
```
```python
mul_per_elem = np.repeat(mul_flat, elems_per_channel)[:size]   # host (was np.tile → M[i % len(M)])
```

**Structural fixes that preceded these** (frontend mapping): 2-input conv (B-alignment with the kernel,
§3.2), the vendored **square-padding regression** (`pads[0]==pads[1]` in `PULPConv2DParser` is commented out
in shipped Deeploy — SpeechNet's `[0,K/2,0,K/2]` time-axis padding is legal, the kernel takes all four pads
separately; we re-commented it to mirror shipped), and the float fc head.

**Debugging pitfalls worth institutional memory** (they cost real time):
- *macOS Docker FS cache*: containers serve stale `.py`/bytecode — `docker restart traindeeploy` after any
  TrainDeeploy edit; `__pycache__` purge + `PYTHONDONTWRITEBYTECODE=1` for Onnx4Deeploy.
- *Probe int8 tensors byte-indexed*: reading uint32 words at "offset k" reads elements 4k — an invalid
  comparison that faked a divergence mid-debug.
- *Channel-0 blindness*: probes at flat offsets 0..N of an NCHW tensor sit entirely in **channel 0**, where
  every per-channel indexing bug is invisible — B5 hid behind "bit-exact" channel-0 probes for a full day.
- *One probe output at a time*: extra artificial graph outputs perturb Deeploy's lowering (and can break
  codegen); fp32 graph outputs are layout-safe, int8 intermediates are NHWC-confounded.
- *GVSoC orphans*: kill by explicit PID (`pgrep -f gvsoc_launcher | xargs kill -9`) before every run.

---

## 5. Validation results

**exp7 (single step)** — device vs host on the identical graph:
```
loss+ = 1.245066 vs 1.245066 (diff 0.000000)     loss− = 1.277979 vs 1.277979 (diff 0.000000)
```

**exp8 (8 pairs, 2 device updates)** — 16/16 losses PASS (15 print-identical, `loss+ 6` at `1e-6` = last-ulp
fp32 op-ordering in one float tail). **On-device weight dump** (`-D DUMP_WEIGHTS=ON`) after the 2 updates:

| group | vs initial | vs host `updated_*` |
|---|---|---|
| 5 int8 conv weights | **byte-identical** (sub-LSB stall) | match |
| 5 int32 conv biases | blocks 1–4 changed **±1..±7 LSBs**, block 0 unchanged | **bit-exact** |
| 10 BN γ/β + 2 fc (fp32) | all changed | 10/12 bit-exact, 2 at 1 ulp (2.3e-10) |

So today's device path trains: **conv biases (integer steps) + BN γ/β + fc** on a frozen int8 conv-weight
backbone — every element of it predicted by the host reference.

---

## 6. Open items / roadmap

1. **Master weights** — fp32 shadow copies for the int8 conv weights (~52 kB for SpeechNet) so sub-LSB
   updates accumulate and flip integers at LSB boundaries; the device design = master buffers as persistent
   state + "float-update master → requantize" in the update graph.
2. **PyTorch accuracy simulation** (fast, host-side, using `QuantSpeechNet` — the export source, cos≈1.0
   vs the deployed graph): Mode A = current device semantics (frozen conv weights, trainable
   biases+BN+fc) vs Mode B = master weights (Brevitas's latent-fp32 weights give this for free). The A/B
   accuracy gap decides whether the master-weight device work is warranted.
3. **QZO round-1** (exp4/exp5 analogue): 54 stratified windows, 200 epochs, on-device weight dump → carry
   checkpoint → batch-2 balanced accuracy vs the float-ZO reference (87.4% sim / 80.6% zero-shot).
4. Minor: `q>1` averaging; emit the reciprocal Quant scale at export (retire the parser inversion);
   optional 4-sample cycling identity test (`q=0, lr=0` → round 2 must be bit-identical to round 1).

---

## 7. Anticipated questions (defense notes)

- **Why quantize only the convs and keep the classifier float?** Shipped convention (QMCUNetZO) and two hard
  constraints: PULP has no bare int8 GEMM kernel, and int8 logits (256 levels over 9 classes) would corrupt
  the tiny `L+−L−` differences that carry the entire ZO gradient signal.
- **Is the perturbation itself hurt by quantization?** No — ε=0.01 is ≈8 weight-LSBs, far above the grid;
  the probes are exact integer arithmetic and bit-reproducible.
- **Are the weights actually updated?** Verified at the *state* level by dumping the device buffers: conv
  weights stall (expected, sub-LSB), conv biases and all fp32 params genuinely update, everything matching
  the host prediction. Not inferred — measured.
- **Do quantization scales adapt per sample / during training?** No — static PTQ scales, frozen at export
  (per-sample scales would break the precomputed integer requant constants and cost a runtime range pass).
  Distribution drift is negligible at fine-tuning magnitudes; re-calibration = the next export in the
  inter-session loop.
- **Why trust "bit-exact"?** The reference executes the identical graph with device-faithful arithmetic
  (same truncations, same RNG, same fp32 op order). Every past deviation turned out to be a real bug — five
  of them — and fixing the last one produced `diff = 0.000000` with no tolerance hiding anything.
- **What did the shipped work already solve?** The perturbed int8 *forward*. The training loop, the
  gradient-scaled integer update, multi-step execution, and weight-level validation are new in this project.
