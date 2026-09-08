# exp11_bitexact_SCE — close the last bit-exactness residual (host-side SCE), verify & re-export

Date: 2026-09-08 · Branch `feat/QZO` · Host reference in `agitated_hugle`; device losses reused
from `exp11_QZO_round1_fix` (NO device re-run).

## Task

The exp11 lr-1e-5 strict-fp32 device round is bit-exact per step but diverges from the host
reference partway through (11,169/21,600) via a `round()`-amplified ≤1-ulp residual. The prior
layer-probe work localized the **only** remaining non-bit-matched op to the SoftmaxCrossEntropy
loss (SCE): the host executor mirrors the device's *order* (L2) but still uses `np.exp`/`np.log`
(glibc) instead of the device's picolibc `expf`/`logf` (the "L4" residual). Goal:

1. **Verify SCE is really the last piece** that affects bit-exactness (primary — the question).
2. **Close it host-side** (mirror picolibc `expf`/`logf` on the host — never touch the device).
3. **Re-export only the reference `outputs.npz`** on the exp11 fixture with the fix, and compare
   pair-by-pair to the **device losses already recorded** in
   `exp11_QZO_round1_fix/logs/round1_gvsoc_strict.log.gz` — no on-device re-run needed, because
   the device is unchanged and its losses are fixed.

## Why no device re-run (confirmed reasoning)

The device trajectory is deterministic and already recorded (device `computed=` values in the
exp11 log). Only the *host reference* depends on the SCE fix. So: regenerate the host reference
losses, diff against the recorded device losses. If SCE is truly the last residual, the new
host trajectory == the device trajectory bit-for-bit → 0/21,600. Any remaining divergence would
expose a further residual. This is the cleanest possible isolation of the SCE fix.

## Legitimacy (direction of the fix)

Host → device only. The device (GVSoC/Siracusa) is ground truth; the host `run_onnx_graph` is a
validation oracle calibrated to it. We mirror picolibc `expf`/`logf` **on the host**; the device
SCE kernel is not modified. (All prior L1/L2 fixes were host-side too.)

## Steps

1. **Verify (from existing data + layer-probe):** confirm every non-SCE op is bit-exact under
   strict-fp32, and the residual is SCE-shaped. (Done — see Findings §1.)
2. **Isolation gate for the port:** get the device's exact `expf`/`logf` (or full-SCE) outputs on
   a broad set of real logit vectors; port picolibc `expf`/`logf` to the host executor; require a
   **bit-match on the gate** before spending the re-export. Only proceed if the gate passes.
3. **Re-export** the host reference on the exp11 fixture (`QZO_exp/exp11_round1_fix`) with the
   fixed SCE → fresh `outputs.npz` saved under `fixture/`.
4. **Compare** to the recorded device losses (all 21,600) → report the error count.
5. Findings + fixture saved here; logs gzipped.

## File paths

- This deliverable: `TrainDeeploy/.../exp11_bitexact_SCE/` (Plan.md, Findings.md, `fixture/`
  fresh `outputs.npz`, `probe/` isolation-gate data, `logs/`)
- Fixture graphs reused (unchanged): `Onnx4Deeploy/QZO_exp/exp11_round1_fix/`
- Device losses reused: `exp11_QZO_round1_fix/logs/round1_gvsoc_strict.log.gz`
- Host SCE to edit: `Onnx4Deeploy/onnx4deeploy/utils/onnx_node_implementations.py` (SCE op,
  lines ~846–870; `np.exp`/`np.log` → picolibc-matched `expf`/`logf`)

## Status / context

- exp11 device round-1 (lr 1e-5, strict-fp32): b2 accuracy 89.44%, per-step bit-exact for the
  first 753 steps, then round()-amplified divergence. Accuracy validated; this experiment is
  the bit-exactness closure/validation of the *training-loss* comparison, not an accuracy change.
- Risk: bit-matching an embedded libm (`expf`/`logf`) on the host is fiddly (exact algorithm,
  FMA usage). The isolation gate (step 2) prevents wasting the re-export on an imperfect port; if
  a perfect match isn't reached, the deliverable is the verification (§1) plus the measured
  closeness and the remaining engineering.
