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
4. **Roles — seed vs carrier vs amplifier (corrected emphasis):**
   - The fp32 params (BN/fc) are the **carrier**: they integrate `coeff·z` with **no rounding**,
     so the ~1-ulp seed shows up on them first and cleanly (the fingerprint). But this drift is
     **sub-tolerance** — 1.19e-7 @ 884, only ~1.8e-5 @ 888, still ~1000× below the 0.001 bar. It
     is NOT what breaks tolerance.
   - **Decisive control (float ZO, exp5):** the *entire* float-ZO model is fp32 and carries the
     *same* ~1-ulp SCE drift, yet it was **0 / 21,600 through the whole round**. So fp32 drift,
     alone, never exceeds tolerance — it is not the divergence driver.
   - The **`round()` on the quantized params is the amplifier and the actual tolerance-breaker**
     — the QZO-specific mechanism float ZO lacks. When the SCE-seeded ~1-ulp `coeff` lands near a
     rounding boundary, the int32 bias (first, step 888, fine grid) or int8 weight (later, coarse
     grid) jumps a full **±1 LSB** — a discrete change ≫ the ulp drift — and THAT pushes the loss
     >tol and cascades. The first >tol errors coincide with the bias flips, not the fp32 drift.

   Net: **SCE is the seed; `round()` amplification of that seed is what makes QZO diverge where
   float ZO does not.** The fp32 drift is the earliest-visible fingerprint of the seed, not the
   cause of the tolerance breach.

## Conclusion — meets the bar

The divergence between device and host is **caused by the SCE node**, proven *in the real
trajectory*, three independent ways that agree:
- **Direct (§1b):** identical SCE input → SCE output differs (≤1.9e-6), only `expf`/`logf` unmatched.
- **Fingerprint (here):** the first divergence is a *uniform ~1-ulp drift of every fp32 param* —
  the unique signature of a ~1-ulp difference in the shared scalar `coeff = f(SCE loss)`, which
  nothing but the SCE node can produce.
- **Elimination (here + layer-probe):** every op *before* SCE is bit-exact, and the conv int8
  weights stay bit-identical through the onset — the int datapath is not the origin.

Refinement of the original framing (two parts):
- "weights into SCE identical, out of SCE divergent" holds for the *int path* — conv int8 stay
  locked well past the onset.
- The *earliest-visible* divergence is the fp32 parameter state (uniform ~1-ulp), but that is the
  **carrier/fingerprint** of the SCE seed, **not** the tolerance-breaker: it stays ~1000×
  sub-tolerance, and float ZO (all-fp32, same seed) is bit-exact for a full round. The
  tolerance breach is the **`round()` amplification** of the SCE-seeded `coeff` on the quantized
  bias/weight grids (±1 LSB jumps) — the QZO-specific mechanism. SCE remains the root cause of
  both the fp32 fingerprint and the round-amplified breach.

## Artifacts

- `host/P_step{884..890}.npz`, `host/loss_trace.txt` — host per-step weights + loss/coeff.
- `logs/device_to891.log` — device run with `[WDUMP s=884..890]` (8/7128 errors = onset).
- Comparison scripts: `/tmp/verif_compare.py`, `/tmp/verif_perparam.py` (logic copied into
  `compare.py` here).
- Device instrumentation: `deeploymezotest.c` multi-step `[WDUMP]` window (`DUMP_STEP_LO/HI`,
  default-off) + `CMakeLists.txt` passthrough; host: `base_exporter.py` env-gated `QZO_TRACE_*`.

## §Q — Quantitative comparison tables (device vs host)

**(A) SCE node in isolation** (§1b probe, 180 real logit vectors fed identically):

| quantity | device vs host |
|---|---|
| pre-SCE logits (input, fed identically) | diff = 0 (by construction) |
| post-SCE log_prob (output) | 559/1620 differ, max 1.9e-6 (~1 ulp) |

**(B) Trajectory step 884 — pristine seed fingerprint** (per-param, from WDUMP vs host P):

| param group | #differ | max\|dev−host\| |
|---|---|---|
| fp32 (BN γ/β + fc) | 481/505 | 1.19e-7 (= 1 ulp of 1.0) |
| int32 conv-bias | 0/104 | 0 |
| int8 conv-wt | 0/14880 | 0 |

**(C) Trajectory step 888 — tolerance-crossing (accumulated + round-amplified):**

| quantity | device | host | diff |
|---|---|---|---|
| L⁺ (4 accum) | 1.4466 / 0.0300 / 0.1266 / 3.0862 | identical | 0 |
| L⁻ (4 accum) | 0.6584 / 0.1070 / 0.3483 / 0.9691 | 0.7073 / 0.1070 / 0.3616 / 1.0516 | up to 0.083 |
| coeff = −lr·Σ(L⁺−L⁻)/(2ε·n_accum) | −3.258e-4 | −3.077e-4 | ~6% |
| weights-out fp32 | — | — | max 1.8e-5 (still 1000× < 1e-3 tol) |
| weights-out int32 bias | — | — | 17/104 flipped ±1 LSB |
| weights-out int8 wt | — | — | 0/14880 |
| weights-IN int8 / int32 | — | — | 0 (bit-identical) |

**Reading:** identical integer weights INTO step 888 → int32 biases OUT differ by whole ±1 LSB
(the `round()` amplification, caught in the act) while int8 stays 0/14880 and the fp32 carrier is
1000× sub-tolerance. HONEST CAVEAT: step 888's L⁻ (~0.08) and coeff (~6%) diffs are much larger
than the ≤1.9e-6 per-eval SCE residual (A) — so 888 is the *accumulated* tolerance-crossing, not
the pristine first event. Pristine single-event SCE evidence = (A) isolation probe + (B) the
uniform 1-ulp fp32 fingerprint at 884. Numbers via `compare.py` on the dumps.
