# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
# -- QW: Package an Onnx4Deeploy ZO export dir into the TrainDeeploy two-dir test-fixture layout.
#
# The Onnx4Deeploy `-mode zo-train` export produces a single dir with
#   network_zo_train.onnx  (weights as INITIALIZERS — reference design)
#   network_zo_update.onnx (weights as INPUTS  — optimizer in/out contract)
#   inputs.npz             (slim: input + label + meta + mb{k} batches)
#   outputs.npz            (22 final weights + log_prob + per-step loss_plus/minus)
#
# TrainDeeploy's runner expects two dirs, each `network.onnx` + `inputs.npz` + `outputs.npz`:
#   <name>_zo_train  : network.onnx = zo_train ; inputs.npz = slim (as exported)
#   <name>_zo_update : network.onnx = zo_update; inputs.npz = FULL — the 22 initial weights must be
#                      present as arr_0002..arr_00NN (in the zo_update graph-input order) so the
#                      optimizer graph's weight inputs get their initial values. arr_0000/arr_0001 stay
#                      the data (input, label). The weight values are read straight from the zo_train
#                      INITIALIZERS, so both graphs start from identical weights. NO baking / no weight
#                      mutation — this is pure repackaging. -- QW
import sys, shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

EXPORT = Path(sys.argv[1])                                   # .../onnx/model/speechnet_zo_2step
DEST   = Path(sys.argv[2])                                   # .../Tests/Models/Training/SpeechNet
TRAIN  = DEST / sys.argv[3]                                  # e.g. speechnet_zo_train_2step
UPDATE = DEST / sys.argv[4]                                  # e.g. speechnet_zo_update_2step

zt = onnx.load(str(EXPORT / "network_zo_train.onnx"))
zu = onnx.load(str(EXPORT / "network_zo_update.onnx"))
slim = dict(np.load(EXPORT / "inputs.npz"))

# 22 trainable-weight initial values, keyed by name, from the zo_train initializers.
init_vals = {i.name: numpy_helper.to_array(i) for i in zt.graph.initializer}
upd_input_names = [i.name for i in zu.graph.input]           # weights, in optimizer graph-input order

# ---- train dir : zo_train graph + slim inputs (as exported) ----
TRAIN.mkdir(parents=True, exist_ok=True)
shutil.copy(EXPORT / "network_zo_train.onnx", TRAIN / "network.onnx")
shutil.copy(EXPORT / "inputs.npz",            TRAIN / "inputs.npz")
shutil.copy(EXPORT / "outputs.npz",           TRAIN / "outputs.npz")

# ---- update dir : zo_update graph + FULL inputs (data + 22 initial weights) ----
UPDATE.mkdir(parents=True, exist_ok=True)
shutil.copy(EXPORT / "network_zo_update.onnx", UPDATE / "network.onnx")
shutil.copy(EXPORT / "outputs.npz",            UPDATE / "outputs.npz")

full = {"arr_0000": slim["arr_0000"], "arr_0001": slim["arr_0001"]}   # input, label
for i, nm in enumerate(upd_input_names):
    full[f"arr_{i + 2:04d}"] = np.asarray(init_vals[nm])             # arr_0002.. = weights, update order
for k, v in slim.items():                                            # carry meta_* and mb{k}_* through
    if k.startswith("meta") or k.startswith("mb"):
        full[k] = v
np.savez(UPDATE / "inputs.npz", **full)

print(f"train  : {TRAIN}  (network.onnx {len(zt.graph.input)} inputs / {len(zt.graph.initializer)} inits)")
print(f"update : {UPDATE} (network.onnx {len(zu.graph.input)} inputs; inputs.npz {len(full)} keys, "
      f"{len(upd_input_names)} weights arr_0002..arr_{len(upd_input_names)+1:04d})")
