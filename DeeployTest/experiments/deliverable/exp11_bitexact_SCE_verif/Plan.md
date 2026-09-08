# exp11_bitexact_SCE_verif — causal proof that SCE is why device/host trajectories diverge

Date: 2026-09-08 · Branch `feat/QZO`. Supersedes the §1b *isolation* argument with an
*in-trajectory causal* one, per the (correct) critique below.

## The critique this addresses

We compare device vs host under a **0.001 tolerance**, not diff=0. The isolation probe
(exp11_bitexact_SCE §1b) showed the SCE node differs on identical inputs — but only by ≤1.9e-6,
which is ~1000× **below** tolerance. That alone does NOT prove the SCE difference is what makes
the two trajectories exceed tolerance and diverge. And the exp11 loss log is only 6-dp, so it
cannot show the true per-step state. We must show the causal chain **in the real run, at the
step where the error first explodes (>tol)**.

## Claim to prove (at the explosion step S ≈ 888, from the exp11 log)

1. **Pre-SCE activations (logits into SCE)** are **bit-identical** device vs host at step S
   → everything upstream of SCE still agrees at S.
2. **Post-SCE (log_prob / loss)** **diverge** at step S → the break is AT the SCE node.
3. **The `round()` update increment** (per-weight delta) **diverges** on ≥1 channel at step S
   → the sub-tolerance SCE difference is amplified through rounding into a real weight-state
   divergence — the seed of the >tol explosion.

(1) identical + (2) divergent + (3) divergent ⇒ SCE is *causally* the origin. This is the bar.

## Key simplification (removes the codegen-risk piece) — 2026-09-08

Dumping the internal pre-SCE logits buffer is NOT needed. Instead dump the **weights going into
step S** (the safe `[WDUMP]` mechanism, extended to fire at a few steps, not just the final one).
Then:
- If **weights_in[S] are bit-identical** device vs host (WDUMP compare), the pre-SCE *state* at
  step S is identical (same weights, same input window, same seed). Combined with the exp13
  layer-probe fact that **every op from input through the Gemm is bit-exact**, the pre-SCE
  activations (logits into SCE) are therefore identical — claim (1), without touching an internal
  buffer.
- The device **loss at S differs** from host (from the log / a dump of `outputs[0]`) — claim (2):
  identical weights-in → different loss-out, and the only op that isn't bit-exact between them is
  SCE ⇒ SCE is the divergence.
- **increment[S] = weights[S] − weights[S−1]** differs device vs host (successive WDUMP diff) —
  claim (3): the SCE-seeded coeff difference flipped a `round()`.

So the whole proof reduces to **multi-step `[WDUMP]` (safe) + the already-logged loss + a matched
host per-step weight/loss trace** — no internal-buffer dump, no extra graph output, no codegen
risk.

## Method (no full-round re-run)

Run only to just past S (~step 890) on device, and the host sim to the same steps, with matched
per-step dumps at a small window (e.g. 884–892). Compare the three quantities.

- **Host side (easy, Python, full precision):** instrument the QZO multi-step sim to dump per
  step: logits (SCE input), log_prob + loss (SCE output), coeff, and the per-param `round()`
  increment. Resolves the 6-dp ambiguity on the host trajectory and gives the exact host S.
- **Device side (the crux):** extend the harness dumps to fire at target steps (not just final):
  - loss (post-SCE) — already read as `DeeployNetwork_outputs[0]`.
  - log_prob (post-SCE) — `DeeployNetwork_outputs[1]` (dumpable like the loss).
  - **logits (pre-SCE)** — an *internal* Gemm-output buffer; dumping it needs its symbol from
    the generated `TrainingNetwork.c` (low risk) OR adding it as a graph output (higher risk —
    extra outputs can perturb Deeploy lowering; avoid if possible).
  - increment — obtained by WDUMP of the 22 weight buffers at S-1 and S; delta = increment.
- Run device to ~890 (~1–2 h), host to ~890 (~2–3 h, slow `run_onnx_graph`). Compare at S.

## Honest scope / risk

This is a multi-hour, moderately invasive instrumentation job (device internal-buffer dump +
partial re-runs of BOTH device and host + step targeting under 6-dp log ambiguity). Larger and
riskier than exp11_bitexact_SCE (which was verification-only). Payoff: a causal, in-trajectory
proof rather than an isolation argument. Executed incrementally; if the device logits-dump risks
codegen breakage, fall back to log_prob (post-SCE, output[1]) + the multi-step WDUMP increments,
which already establish (2)+(3), with (1) supported by the exp13 layer-probe (pre-SCE ops
bit-exact) — a slightly weaker but still strong form.

## Structure

`Plan.md` (this), `host/` (host per-step trace script + dumps), `dumps/` (device tensor dumps),
`logs/`, `Findings.md` (after).

## Status

- Design fixed, and simplified to the safe weights-in form (no codegen risk).
- Execution is a multi-hour partial re-run of BOTH sides (device to ~890 ≈ 1–2 h; host sim to
  ~890 ≈ 2–3 h) + multi-step WDUMP (small C change) + a matched host per-step weight/loss trace.
- Concrete next steps: (1) extend `dump_zo_weights` to fire at a target-step list; (2) env-gated
  per-step weight/loss dump in the host QZO sim; (3) run both to ~892; (4) compare weights_in[S]
  (expect identical), loss[S] (expect divergent), increment[S] (expect divergent) at the first
  explosion step S.
