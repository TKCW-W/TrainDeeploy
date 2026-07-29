# Test 1 RESULT — B-decompose frozen-BN ORT reference (exporter side)

**Date:** 2026-07-29 · **Branch:** Onnx4Deeploy `feat/BNFRozen_OptionB` (code) · TrainDeeploy `feat/BNFRozen_OptionB` (this dir)

## Verdict: ✅ SUCCESS — ORT computes the frozen reference on the same (decomposed) graph, matching Option A to ~1e-6.

The per-step reference loss is now **ORT-computed** on a `network_train.onnx` that contains **no
`BatchNormInternal`** — each BatchNorm is replaced, before `generate_artifacts`, by an explicit
**frozen affine** that ORT autodiffs natively. This graph is what the device would run (goals i + iii:
ORT reference, same graph). Device-compilability was engineered in (goal ii, see "Device-friendliness").

## What was implemented
New flag **`--bn-decompose-frozen`** (config `bn_decompose_frozen`), parallel to `--bn-frozen-stats`.
When set, `export_training` swaps every `nn.BatchNorm*` → `FrozenAffineBN` **before** the ONNX export:

```
y = (x + neg_running_mean) * inv_running_std * gamma + beta
    neg_running_mean = −running_mean            (constant buffer)
    inv_running_std  = 1/√(running_var + eps)   (constant buffer)
    gamma = weight, beta = bias                 (trainable Parameters, names unchanged)
```

- Uses **Add/Mul only** — deliberately no `Sub` (unbound in PULPOpen), no `Sqrt`/`Reciprocal` (pre-baked
  into `inv_running_std`). torch.onnx therefore emits `Add`/`Mul`/`Reshape`; ORT autodiff adds
  `Mul`/`ReduceSum`/`Reshape` — **all bound in Deeploy's PULPOpen target.**
- γ/β keep the `weight`/`bias` attribute names → ONNX initializer names for the trainable params are
  unchanged → `get_trainable_params` and the grad-accumulation-buffer order are untouched (still the same
  22 trainable tensors: conv + γ/β + fc).
- `_BN_BUFFERS` extended with `neg_running_mean`, `inv_running_std` so both land in `frozen_params`
  (no grad). Confirmed: the 22 grad-accumulation buffers cover exactly the trainable set; the stat
  buffers are frozen.
- Reference routing needs **no change**: decompose uses a distinct config key, so
  `create_training_test_data`'s `_bn_frozen` (which keys on `bn_frozen_stats`) stays False → the **ORT**
  reference path runs (not the Option-A PyTorch path).

## Validation (S01 vocalized sess3 batch1, official ckpt; full model, lr 3e-4, n_accum 4, data-size 18,
n-epochs 2 → 36 steps)

| check | result |
|---|---|
| `BatchNormInternal` in decomposed `network_train.onnx` | **0** (was 5) |
| decomposition ops | `Add`, `Mul`, `ReduceSum`, `Reshape` only — **no `Sub`/`Sqrt`/`Reciprocal`/`Neg`** |
| unbound-op risk | only **`Expand: 1`**, which **also appears in the Option-A graph** → not introduced by the decomposition; device already runs the Option-A graph with it (exp1 chain) |
| **max \|Δloss\| vs Option-A PyTorch frozen ref** (36 steps) | **8.3e-7** (mean 1.2e-7) — pure ORT-vs-PyTorch fp noise |
| γ/β/conv/fc trained (final vs init) | γ 9.2e-4, β 5.8e-4, conv 7.7e-4, fc 6.1e-4 → **all train** |
| frozen stats (final vs init) | `neg_running_mean` 0.0, `inv_running_std` 0.0 → **frozen** |

## Files changed (Onnx4Deeploy, branch feat/BNFRozen_OptionB)
- `onnx4deeploy/core/base_exporter.py` — `FrozenAffineBN` module + `_swap_bn_to_frozen_affine`; swap wired
  into `export_training` (supersedes `bn_frozen_stats`); `_BN_BUFFERS` extended.
- `Onnx4Deeploy.py` — `--bn-decompose-frozen` CLI flag + config override + pass-through (4 spots).

## Why this matters
- The bit-exact training test now compares **ORT vs device on one identical graph** — the residual loss
  difference is purely the on-device numerical effects (fp32 tiling, SUM order, MaxPool argmax drift),
  with **no PyTorch-vs-device engine confound** (the Option-A limitation).
- Retires the `g_bn_frozen_stats` C flag and the Option-A PyTorch reference (both can stay as fallback).

## Next (not in Test 1 scope)
- **Test 2 / device:** actually compile+run the decomposed `network.onnx` on the GVSoC tiled runner and
  confirm the single `Expand` is handled (it is in Option A). Then run the b1→b5 incremental FT and
  record the ORT-vs-device loss residual.
- Reproduction scripts in this dir: `validate_test1.py` (loss match + op audit + weight-trajectory check).
