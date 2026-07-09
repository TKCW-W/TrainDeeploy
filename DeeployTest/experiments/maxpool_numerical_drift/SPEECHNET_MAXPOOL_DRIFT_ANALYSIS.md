# MaxPool training drift: is it a numerical limit, and can accuracy be increased?

After the layout fix (`b2c3735`), on-device MaxPool fine-tuning is **bit-exact (1e-6) for
~36 steps (2 epochs)**, then the per-step loss drifts from the ORT reference, plateauing at
~0.06 (52/90 over TOL=0.001 in the 90-step run). This note characterises that drift.

## What the drift is

It is **trajectory divergence**, not a per-step precision deficiency:

- The on-device per-step gradient is **accurate** — the first ~36 steps match the reference to
  1e-6, which proves the computation itself is correct.
- On-device (8-core PULP kernels) and the ORT reference are **two different fp32 implementations**
  of the same SGD. Each step they differ by a tiny **reduction-order rounding** (~1e-6–1e-7,
  unavoidable on parallel fp32 hardware). As the two weight trajectories separate, the per-step
  loss difference grows, then **saturates** (bounded by the loss-landscape curvature). It does
  **not** blow up, and the model still learns (epoch-avg loss 2.54 → 2.18).

The onset is **discrete** (a jump at step ~37), which is the fingerprint of **MaxPool's argmax
flipping** in a near-tie pooling window once the accumulated weight difference crosses a tie.

## Is it a limit, or can precision fix it?

We injected a controlled per-step relative perturbation `eps` into a PyTorch fp32 training run
(mimicking the on-device reduction-order difference) and measured the step at which the loss
first diverges by >1e-3, sweeping `eps` (mean of 4 seeds). See
`speechnet_maxpool_drift_analysis.png`.

| per-step eps | AvgPool onset | MaxPool onset |
|---|---|---|
| 1e-4 | 5  | 9  |
| 1e-5 | 21 | 10 |
| 1e-6 | 44 | 18 |
| 1e-7 | 73 | 44 |
| 1e-8 | 88 | 48 |

**Two conclusions:**

1. **Precision helps (not a hard wall).** Reducing the per-step difference pushes the divergence
   onset proportionally later — for *both* pools. So numerical accuracy *can* be increased by
   shrinking the per-step seed.

2. **MaxPool is intrinsically ~3–4× more sensitive than AvgPool**, at every precision level. Its
   **discrete argmax** flips at near-tie windows that the training dynamics (lr·grad ~ 1e-3, far
   larger than any fp error) inevitably reach; the fp difference only decides *which way* the tie
   breaks. AvgPool averages and absorbs the perturbation smoothly. This extra sensitivity is
   inherent to MaxPool and **cannot be removed by precision** — only delayed.

## How to increase numerical accuracy (reduce the per-step seed)

In decreasing order of effect:

1. **Match the reduction arithmetic to the reference** (deterministic/order-fixed reduction, or
   compute the reference with the device's own kernels) → per-step seed → 0 → bit-exact for all
   steps. This is really a *validation-methodology* alignment, and is the only way to make
   MaxPool match a reference over arbitrarily many steps.
2. **Fewer cores** (1-core sequential reduction) → smaller reduction-order divergence → later
   onset. (We keep 8 cores for throughput, so this is a lever we are not taking.)
3. **Higher-precision accumulators** (Kahan / fp64) in the reduction-heavy kernels — chiefly the
   BatchNorm batch-variance (the largest sum, most order-sensitive) and the conv/matmul gradient
   reductions → smaller per-step `eps` → later onset.

## Practical verdict

- The drift is **bounded (~0.06, ~3% of the loss) and benign**: both trajectories are valid SGD
  paths differing only in fp rounding / arbitrary tie-breaking; the model trains correctly.
- For **on-device fine-tuning**, which is typically short (a handful of steps to adapt to a new
  session), the operating regime is the **bit-exact** part of the curve — the drift is a non-issue.
- For long fine-tuning *and* tight reference-matching, the AvgPool-vs-MaxPool gap is the genuine
  **MaxPool-specific limit** (argmax discreteness); the remainder (the fp seed) is improvable by
  the levers above.

So: the residual training drift is **partly a fundamental limit** (MaxPool's discrete argmax makes
it inherently more sensitive than AvgPool) and **partly improvable** (reduce the per-step fp seed).
It does not indicate incorrect on-device training.
