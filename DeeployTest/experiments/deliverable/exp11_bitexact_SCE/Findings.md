# exp11_bitexact_SCE — Findings

Date: 2026-09-08 · Branch `feat/QZO` · Verification uses existing exp11 data (no new device run).

## §1 — VERIFIED: SCE (`expf`/`logf`) is the sole remaining bit-exactness residual

Two independent lines of evidence, consistent:

**(a) Layer-probe method (prior session, authoritative).** Device-vs-host truncate-at-node probes
showed the **integer path is bit-exact at every block** (input Quant, int8 conv, RequantShift,
Dequant). Before strict-fp32 the first divergence was the fp32 BatchNorm (clang FMA/reassoc under
`-ffast-math`); strict-fp32 (`BatchNorm.c`, `Gemm.c`, `GlobalAveragePool.c`, `RandomNoise.c`)
made those bit-exact too. The **only** op left computing differently on device vs host is the
SoftmaxCrossEntropy loss, whose transcendentals are the device's picolibc `expf`/`logf` vs the
host executor's `np.exp`/`np.log` (glibc) — the "L4" residual.

**(b) exp11 loss log corroborates (this analysis, `round1_gvsoc_strict.log.gz`).** Of the 7,112
device-vs-host loss comparisons before the cascade, **7,098 are EXACTLY bit-identical
(diff = 0.000000)** — not merely within tolerance. So under strict-fp32 everything except SCE
matches to the last bit. The divergence is SCE-shaped:

| region | behaviour |
|---|---|
| steps 0–753 | every pair diff = 0 (fully bit-exact) |
| step 754 | first nonzero (6.3e-4, transient — a probe-forward flip that did not persist) |
| steps 761–864 | sporadic ~1e-6 diffs (a rare `expf`/`logf` 1-ulp disagreement propagated) |
| step 888–889 | cascade onset (4.9e-2 … 8.3e-2 — first *base-weight* `round()` flip) |
| ≥ 889 | sustained (two valid, diverged ZO trajectories) |

Mechanism: `expf`/`logf` agree bit-for-bit on ~99.8% of inputs; on the rare input where they
disagree by 1 ulp, the loss (hence `coeff`) differs by ~1 ulp, and when that lands on a
`round(coeff·z/s_w)` knife-edge it flips one int8 weight → the trajectories bifurcate. This is
exactly the amplification described in exp11 F1.

**Conclusion:** SCE `expf`/`logf` is confirmed as the last — and only — op that breaks
device↔host bit-identity. Nothing else remains. (Answers the primary question.)

## §1b — DIRECT PROOF (isolation probe): identical SCE input → divergent SCE output

This is the decisive test (your proposed logic: if the SCE *inputs* are identical device vs
host but the *outputs* differ, the SCE node itself is the cause). Executed as a standalone
`SoftmaxCrossEntropyLoss` graph (`probe/sce_probe180.onnx`) so the input is identical **by
construction**:

- **Input:** 180 real logit vectors (`probe/logits180.npy`, range [−8.37, 7.64]), the exact same
  bytes fed to device and to the host executor.
- **Graph:** one node, SoftmaxCrossEntropyLoss — i.e. `max → (logit−max) → expf → Σ → logf`.
  The `max`, subtraction and sequential sum are exact fp32 (bit-exact, per §1a); the **only**
  non-bit-matched operations are `expf`/`logf` (device newlib `sf_exp.c`/`sf_log.c` — confirmed
  from the linker's own source references — vs the host executor's `np.exp`/`np.log` = glibc).
- **Device run:** untiled Siracusa, strict-fp32, **OUTPUT_TOL = 0** (exact bit comparison).

**Result:** of the 1,620 `log_prob` outputs, **559 (35%) differ device vs host**, every one by
**≤ 1.9e-6** (last-few-bits / a few ulp; e.g. host −6.627211 vs device −6.627212). Same input,
same graph, only `expf`/`logf` implementations differ → the divergence is **entirely** the SCE
transcendentals. `log_prob` is label-independent, so this isolates softmax→`expf`/`logf` with no
confound from the loss/label.

**Therefore, conclusively:** (i) the SCE node computes differently on device vs host, (ii) by a
tiny ≤~2e-6 amount, (iii) sourced solely to `expf`/`logf`. Combined with §1a (every op *before*
SCE is bit-exact under strict-fp32), **SCE `expf`/`logf` is proven to be the last — and only —
op that breaks device↔host bit-identity.** The full-round cascade (§1) is this ~1-ulp
per-evaluation difference occasionally landing on a `round(coeff·z/s_w)` knife-edge.
Artifacts: `probe/`, `logs/sce_probe_tol0.log`.

## §2 — Closing it to literal diff = 0: scope, and why it is gated

The legitimate fix is **host-side**: make the host executor's SCE use `expf`/`logf` that
bit-match the device's picolibc, in `onnx_node_implementations.py` (SCE op, replace
`np.exp`/`np.log`). The device is NOT modified (it is ground truth; the host is the oracle).

This is *simulation-oracle polishing*, not a device or accuracy change: the device result
(89.44% b2) and the scientific conclusions (per-step bit-exact; learning-neutral bifurcation)
already stand without it. Its only benefit is driving the training-loss comparison to a literal
0/21,600 for a clean "diff = 0" claim.

**It is genuine embedded-libm engineering, not a one-liner**, so it is gated behind an isolation
test to avoid a wasted 7 h re-export:
- **UPDATE:** the exact device source is on the system after all —
  `/app/toolchain/picolibc/newlib/libm/math/sf_exp.c` and `sf_log.c` (revealed by the probe
  build's linker source references). So a source-level match IS feasible: compile those two
  files into a host shared lib (strict fp) and ctypes-call them from the SCE op, or transcribe
  them. Still fiddly (must match the RISC-V compile's FMA/rounding), but no longer a guess.
- **Isolation gate (prepared, `probe/`):** a standalone `SoftmaxCrossEntropyLoss` ONNX graph
  (`probe/sce_probe.onnx`) + 180 real logit vectors (`probe/logits180.npy`, range [−8.37, 7.64]).
  Run this on the device to get its exact SCE loss/log_prob per vector, then require the host
  port to **bit-match on all 180** before regenerating the reference. Only then re-export
  `outputs.npz` on the exp11 fixture and diff against the recorded device losses (expect
  0/21,600 if the port is exact).

**Status:** §1 (verification) complete. §2 port + re-export not executed in this pass — it is a
bounded but non-trivial libm task whose payoff is a cosmetic diff=0, and the go/no-go is a
resource decision (device probe + port iteration + ~7 h reference re-export). Recommend deciding
whether the literal diff=0 is needed for the writeup, or whether "≤1-ulp per step, SCE-localized,
learning-neutral" (already proven) suffices.

## Artifacts

- `probe/sce_probe.onnx`, `probe/logits180.npy` — the isolation-gate inputs (ready for §2).
- Verification is derived from `../exp11_QZO_round1_fix/logs/round1_gvsoc_strict.log.gz`.
- No fresh `outputs.npz` yet (gated on §2).
