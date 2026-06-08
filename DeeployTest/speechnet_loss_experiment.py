# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Run SpeechNet training with pretrained weights on tiled Siracusa for 100
optimizer steps and plot the resulting loss curve.

Usage (from DeeployTest/):
    python speechnet_loss_experiment.py
    python speechnet_loss_experiment.py --skipgen   # re-use generated C code
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PREWEIGHTS_DIR = "Tests/Models/Training/SpeechNet/speechnet_train_preweights"
N_STEPS = 100
LOG_PATH = Path("speechnet_100step_run.log")
PLOT_PATH = Path("speechnet_100step_loss.png")

skipgen = "--skipgen" in sys.argv

OPTIMIZER_DIR = "Tests/Models/Training/SpeechNet/speechnet_optimizer"

cmd = [
    sys.executable, "deeployTrainingRunner_tiled_siracusa.py",
    "-t", PREWEIGHTS_DIR,
    "--optimizer-dir", OPTIMIZER_DIR,
    "--l1", "128000",
    "--l2", "2000000",
    "--n-steps", str(N_STEPS),
    "-vv",
]
if skipgen:
    cmd.append("--skipgen")

env = {**os.environ, "PYTHONUNBUFFERED": "1"}

print(f"Running: {' '.join(cmd)}\n")

log_lines = []
with subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, env=env, bufsize=1) as proc:
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_lines.append(line)

stdout_text = "".join(log_lines)
LOG_PATH.write_text(stdout_text)
print(f"\nFull log saved to {LOG_PATH}")

losses = []
for line in stdout_text.splitlines():
    m = re.match(r'LOSS\s+(\d+)\s+(\S+)', line)
    if m:
        losses.append((int(m.group(1)), float(m.group(2))))

if not losses:
    print("ERROR: No '[loss N] computed=X' lines found — check the log.")
    sys.exit(1)

steps, vals = zip(*sorted(losses))
plt.figure(figsize=(8, 4))
plt.plot(steps, vals, marker='o', markersize=3, linewidth=1)
plt.xlabel("Optimizer step")
plt.ylabel("Cross-entropy loss")
plt.title("SpeechNet on-device training — tiled Siracusa, pretrained init (100 steps)")
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"Plot saved to {PLOT_PATH}  ({len(steps)} points, final loss={vals[-1]:.4f})")
