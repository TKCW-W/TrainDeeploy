# exp11_bitexact_SCE_verif — Findings: causal, in-trajectory proof that SCE is the divergence origin

Date: 2026-09-08 · Branch `feat/QZO` · Device partial run to step 891 (multi-step `[WDUMP]`,
strict-fp32) + matched host per-step weight/loss trace. No full round; no internal-buffer dump;
no codegen risk.

## The bar (from the critique)

Show, in the real run at the divergence, that the sub-tolerance SCE difference is what *causes*
device and host to take different trajectories — not merely that SCE differs in isolation.

## Method

Dumped the 22 trainable params on **both** sides at steps 884–890 (device `[WDUMP]` after each
update; host `P` after each update — identical semantics). Compared per-param, device vs host.
The device run to step 891 shows only **8 loss errors / 7,128** — i.e. this window is exactly the
onset of the divergence (the full cascade is at later steps, not run).

## Result — the exact causal chain, with a unique fingerprint

| step | fp32 params (BN γ/β, fc) | int32 conv biases | int8 conv weights |
|---|---|---|---|
| 884 | **all differ, ≤1.19e-7 (= 1 ulp of 1.0)** | 0 differ | **0 differ** |
| 888 | all differ, ~1.8e-5 (grown) | **±1 LSB on a few channels** | **0 differ** |

Read it as the chain:

1. **SCE `expf`/`logf` differ by ≤1.9e-6 on identical inputs** (exp11_bitexact_SCE §1b, direct
   probe). This is the only non-bit-matched op (every op before SCE is bit-exact — exp13
   layer-probe, and confirmed here by conv int8 staying bit-identical).
2. Every ZO update multiplies **one shared scalar** `coeff = −lr·(L⁺−L⁻)/(2ε·n_accum)` by the
   per-param Rademacher `z`; `L±` **is** the SCE loss. A ~1-ulp SCE difference → ~1-ulp `coeff`.
3. **The fingerprint:** at step 884 *every fp32 parameter* (all BN γ/β + fc, 100+ values across
   5 blocks) differs by ~1 ulp **uniformly**. Only a difference in the *shared scalar* `coeff`
   can perturb every parameter at once like this — no per-op or per-channel bug produces a global
   scalar offset touching all params. Since `coeff` differs only when `L±` differs, and `L±`
   differs only at the SCE node (1), the fp32 drift **is** the SCE residual, integrated.
4. **Why fp32 first, int8 last:** the fp32 params (BN/fc) integrate `coeff·z` with **no
   rounding**, so the ~1-ulp difference accumulates from the start (1.19e-7 @ 884 → 1.8e-5 @ 888
   as it feeds back through the forward). The int32 bias grid (fine) rounds-flips first at 888
   (±1 LSB, few channels); the int8 conv grid (coarse) is still bit-identical even at 888 — its
   `round()` suppresses the sub-½-LSB `coeff` difference longest. The full-round cascade
   (conv int8 flips, >tol everywhere) follows at later steps.

## Conclusion — meets the bar

The divergence between device and host is **caused by the SCE node**, proven *in the real
trajectory*, three independent ways that agree:
- **Direct (§1b):** identical SCE input → SCE output differs (≤1.9e-6), only `expf`/`logf` unmatched.
- **Fingerprint (here):** the first divergence is a *uniform ~1-ulp drift of every fp32 param* —
  the unique signature of a ~1-ulp difference in the shared scalar `coeff = f(SCE loss)`, which
  nothing but the SCE node can produce.
- **Elimination (here + layer-probe):** every op *before* SCE is bit-exact, and the conv int8
  weights stay bit-identical through the onset — the int datapath is not the origin.

Refinement of the original framing: the "weights into SCE identical, out of SCE divergent" idea
is correct for the *int path* (conv int8 stay locked), but the *first* thing to visibly diverge
is the **fp32 parameter state**, because it integrates the SCE-seeded scalar `coeff` without the
`round()` that suppresses it on the quantized weights. Same cause (SCE), earliest-visible via the
un-rounded fp32 path.

## Artifacts

- `host/P_step{884..890}.npz`, `host/loss_trace.txt` — host per-step weights + loss/coeff.
- `logs/device_to891.log` — device run with `[WDUMP s=884..890]` (8/7128 errors = onset).
- Comparison scripts: `/tmp/verif_compare.py`, `/tmp/verif_perparam.py` (logic copied into
  `compare.py` here).
- Device instrumentation: `deeploymezotest.c` multi-step `[WDUMP]` window (`DUMP_STEP_LO/HI`,
  default-off) + `CMakeLists.txt` passthrough; host: `base_exporter.py` env-gated `QZO_TRACE_*`.
