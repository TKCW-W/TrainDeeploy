# Brevitas evidence — who owns the weight, and who updates it

Supporting material for the LSB-stall discussion. Every script here is standalone and runnable in the
`agitated_hugle` container (`docker exec agitated_hugle bash -lc 'python3 <script>'`). Brevitas 0.13.0.

## The claim being evidenced (stated precisely)

> **Brevitas performs no weight update at all.** It never writes `self.weight`; it is a forward-time
> quantizer that *reads* the weight and returns a transient quantized value.
> **But its parameter interface is fp32-only**, so any training loop whose trainable state is Brevitas'
> own parameters necessarily accumulates in fp32 — i.e. it is master-weight training, without the author
> having to choose it.

The corollary that matters for us: **whether the stall happens is decided by the update implementation, not
by Brevitas.** Our device stores only int8 (no fp32 to accumulate into) and therefore stalls; that is a
storage-design choice made in `build_int8_forward`, not a consequence of quantization.

*(An earlier, sloppier phrasing — "Brevitas is intrinsically master-weight" — is wrong and should not be
used: Brevitas does not update anything. Use the precise statement above.)*

## Scripts

| script | shows |
|---|---|
| `state_check.py` | The module holds **no int8 tensor at all**: parameters are all `float32`, buffers `{float32, int64}`. An int8 weight has nowhere to live. |
| `whobrev.py` | **A.** `torch.optim.SGD(m.parameters())` captures the fp32 `nn.Parameter` objects — `conv.weight` is literally one of them. **B.** A real `backward()`+`step()` moves that fp32 tensor (`max|Δ| = 3.1e-4`) via `torch/optim/sgd.py:366 param.add_(grad, alpha=-lr)`. **C.** Trying to store int8 instead fails outright: `RuntimeError: data set to a tensor that requires gradients must be floating point or complex dtype`. |
| `mw_proof2.py` | Sub-LSB accumulation in action: nudging the fp32 master by **0.06 LSB/step** (our real ZO update size) flips the int8 view after 5 steps (`W/s_w 106.2556 → 106.5556 → int8 106 → 107`). No single step could have moved the int8. |
| `chain.py` | The live module chain the weight flows through: `QuantConv2d → WeightQuantProxyFromInjector → RescalingIntQuant → {scaling_impl: StatsFromParameterScaling, int_quant: IntQuant → float_to_int_impl: RoundSte}`. |
| `alias.py` | `m.weight.data = t` **shares storage** (no copy) — an in-place write through the model corrupts the caller's tensor. This is why `loss_with_weights` save/restores. |
| `idem.py` | Why the original methodology was flawed: injecting a *pre-quantized* weight is **not** a no-op — Brevitas re-derived a finer scale and re-rounded, shifting elements by up to 1 LSB. Fixed by pinning `scaling_impl`. |

## Source extracts (`source_extracts/`)

Copies of the relevant Brevitas 0.13.0 files, for reading without entering the container.

| file | key line |
|---|---|
| `nn_quant_conv.py` | `:117` `class QuantConv2d(QuantWBIOL, Conv2d)` — inherits torch `Conv2d`, so `self.weight` is a plain fp32 `nn.Parameter` |
| `nn_quant_layer.py` | `:146` `quant_weight = self.quant_weight(quant_input)` — re-quantized on **every** forward |
| `nn_mixin_parameter.py` | **`:49` `weights_to_quantize = self.weight`** — the key line: reads the fp32 parameter, returns a new tensor, never writes back |
| `core_quant_int.py` | `:157-165` `RescalingIntQuant.forward` — `scale = self.scaling_impl(x, ...)` then `int_quant(...)` |
| `core_quant_int_base.py` | `:54` `y = x / scale` · `:58` `RoundSte` · `:59` clamp ±127 · `:74` `y = y * scale` |
| `core_scaling_runtime.py` | `:25` `StatsFromParameterScaling(tracked_parameter_list: List[nn.Parameter])` — the weight scale is derived **from the parameter**, i.e. data-free, and re-derived every call (which is why we pin it) |
| `proxy_parameter_quant.py` | `:206` `WeightQuantProxyFromInjector` — the proxy between the layer and `tensor_quant` |

**Reading order:** `nn_quant_conv.py:117` → `nn_quant_layer.py:146` → **`nn_mixin_parameter.py:49`** → `core_quant_int.py:157` → `core_quant_int_base.py:54-74`.

## Searches worth repeating

```bash
B=/usr/local/lib/python3.10/site-packages/brevitas
# Brevitas never writes the weight (only 2 hits, both construction: quant_bn init, quant_rnn Parameter creation)
grep -rn "self\.weight\s*=\|self\.weight\.data\s*=" $B/nn/ $B/proxy/ $B/core/ | grep -v weight_quant
# where the fp32 accumulation actually happens in a normal fine-tune
grep -n "param.add_(grad, alpha=-lr)" /usr/local/lib/python3.10/site-packages/torch/optim/sgd.py   # :366, :368
```

## Where the two update rules live in our experiment

`../brevitas_stall_study/brevitas_stall_experiment.py`:
- `:186` `m.weight.data = weight_override[...]` — **the hand-off** (must be fp32; PyTorch enforces it)
- `:189` `logits = model(x)` — the only span where Brevitas is involved
- **`:302` `A_latent[n] = A_latent[n] + coeffA * z[n]`** — master weights: unrounded fp32 accumulation
- **`:307` `delta = torch.round(coeffB * z[n] / s_w)`** — direct int8: sub-LSB discarded → the stall
