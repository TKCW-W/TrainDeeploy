# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers used by the training / optimizer code-generation entry points
(generateTrainingNetwork.py, testMVPTraining.py, generateOptimizerNetwork.py,
testMVPOptimizer.py).

Four kinds of helpers live here, all strictly training-specific:

1. inputs.npz / outputs.npz readers (``_load_reference_losses``, ``_infer_*``).
2. The singleton ``_mockScheduler`` the Tiler expects for per-node tiling.
3. Training-only argparse builders (``add_training_inference_args``,
   ``add_optimizer_training_dir_arg``).
4. The core hooks invoked by ``testUtils.core.execution``
   (``resolve_optimizer_dir``, ``run_training_codegen``,
   ``add_training_cmake_flags``).

Generic helpers (``--cores`` / ``--l1`` / ``--l2`` / ``--defaultMemLevel`` /
``--memAllocStrategy`` / ``--searchStrategy`` / ``--plotMemAlloc`` /
``--profileTiling`` / ``--shouldFail`` arg definitions and the ``shouldFail``
try/except handshake) are deliberately *not* wrapped into functions here:
they are not training-specific and belong inline in whichever entry point
needs them, consistent with the upstream inference codegen scripts.
"""

import argparse
import json
import os
import shutil  # -- QW  copy fixture npz into prepped ZO dirs
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.Logging import DEFAULT_LOGGER as log

# Graph input name marker identifying gradient accumulation buffers.
_GRAD_ACC = "_grad.accumulation.buffer"


def _load_reference_losses(train_dir: str) -> Optional[list]:
    """Load reference loss values from outputs.npz.

    Returns the list of per-mini-batch loss values if any key in
    outputs.npz contains 'loss', otherwise None (with a warning).
    """
    outputs_path = os.path.join(train_dir, "outputs.npz")
    if not os.path.exists(outputs_path):
        log.warning(f"outputs.npz not found at {outputs_path} — loss comparison skipped")
        return None

    try:
        outputs = np.load(outputs_path)
    except Exception as e:
        log.warning(f"Failed to load outputs.npz: {e} — loss comparison skipped")
        return None

    for key in outputs.files:
        if 'loss' in key.lower():
            vals = [float(v) for v in np.array(outputs[key]).flatten().tolist()]
            log.info(f"Reference losses loaded from outputs.npz['{key}']: {vals}")
            return vals

    log.warning("No 'loss' key found in outputs.npz — loss comparison skipped")
    return None


def _infer_num_data_inputs(inputs_path: str) -> int:
    """Auto-detect number of data inputs from inputs.npz.

    Data inputs are the base arr_* entries that have per-mini-batch
    variants (mb1_arr_*) in the npz — i.e. entries that actually change
    across mini-batches.

    Raises ValueError if no mb1 entries are found (single-mini-batch case)
    where the data/weight boundary cannot be determined automatically.
    """
    inputs = np.load(inputs_path)
    base_keys = sorted(k for k in inputs.files if not k.startswith('mb') and not k.startswith('meta_'))
    count = sum(1 for k in base_keys if f'mb1_{k}' in inputs.files)
    if count == 0:
        raise ValueError("Cannot auto-detect num_data_inputs: inputs.npz has only one mini-batch "
                         "(no mb1_arr_* entries found). Please pass --num-data-inputs explicitly.")
    return count


def _infer_total_mb(inputs_path: str) -> int:
    """Count total mini-batches from inputs.npz.

    New format: inputs.npz contains meta_n_batches (total training mini-batches)
    and meta_data_size (number of unique samples stored; C harness cycles via modulo).

    Legacy format: count 1 + number of unique mb* indices.
    """
    inputs = np.load(inputs_path)
    if "meta_n_batches" in inputs.files:
        return int(inputs["meta_n_batches"].flat[0])
    mb_indices = set()
    for key in inputs.files:
        if key.startswith('mb'):
            try:
                idx = int(key.split('_')[0][2:])
                mb_indices.add(idx)
            except ValueError:
                pass
    return 1 + len(mb_indices)


def _infer_data_size(inputs_path: str) -> int:
    """Return the number of unique input samples stored in inputs.npz.

    New format: reads meta_data_size.
    Legacy format: same as _infer_total_mb (all batches were unique).
    """
    inputs = np.load(inputs_path)
    if "meta_data_size" in inputs.files:
        return int(inputs["meta_data_size"].flat[0])
    return _infer_total_mb(inputs_path)


def _infer_n_accum(inputs_path: str) -> int:
    """Return the gradient accumulation step count stored in inputs.npz.

    New format: reads meta_n_accum written by the exporter.
    Legacy format: defaults to 1 (no gradient accumulation).
    """
    inputs = np.load(inputs_path)
    if "meta_n_accum" in inputs.files:
        return int(inputs["meta_n_accum"].flat[0])
    return 1


def deferPerturbNodes(nodes: List[gs.Node]) -> List[gs.Node]:  # -- QW
    """Re-order a topologically-sorted node list so each ZO perturbation node runs
    immediately before its earliest consumer (ported from Deeploy zo-support). -- QW

    Perturb* nodes produce a non-in-place perturbed COPY of a weight; that copy is
    live from the perturb until its consumer. Because perturb nodes depend only on
    constants, Deeploy's toposort HOISTS all of them to the front, keeping all ~22
    copies alive across the whole forward. The SB-tiler then has almost no L2 left,
    tiles every conv/BN into tiny pieces, and the (cycle-accurate) sim crawls
    (~min/forward instead of ~s). Moving each perturb to just before its consumer
    caps peak liveness at ~1 copy. No-op for BP graphs (no Perturb nodes). -- QW
    """  # -- QW
    isPerturb = lambda n: "Perturb" in n.op  # -- QW
    perturb = [n for n in nodes if isPerturb(n)]  # -- QW
    if not perturb:  # -- QW  BP fast path: nothing to defer
        return nodes  # -- QW
    base = [n for n in nodes if not isPerturb(n)]  # -- QW
    pos = {id(n): i for i, n in enumerate(base)}  # -- QW
    before = {id(n): [] for n in base}  # -- QW
    leftover = []  # -- QW
    for p in perturb:  # -- QW
        consumers = [c for c in p.outputs[0].outputs if id(c) in pos]  # -- QW
        if not consumers:  # -- QW
            leftover.append(p)  # -- QW
            continue  # -- QW
        earliest = min(consumers, key = lambda c: pos[id(c)])  # -- QW
        before[id(earliest)].append(p)  # -- QW
    out = list(leftover)  # -- QW
    for n in base:  # -- QW
        out.extend(before[id(n)])  # -- QW
        out.append(n)  # -- QW
    return out  # -- QW


def _mockScheduler(graph: gs.Graph) -> List[List[gs.Node]]:
    """Wrap every node in a singleton list for the Tiler pattern interface.

    Applies deferPerturbNodes first so ZO perturb copies don't all live at once
    (BP graphs are unaffected — no Perturb nodes). -- QW
    """
    return [[node] for node in deferPerturbNodes(list(graph.nodes))]  # -- QW


# ---------------------------------------------------------------------------
# argparse builders
#
# The four training / optimizer codegen entry points all define the same
# arguments in their __main__ blocks.  These helpers add the shared groups
# to an existing parser so each entry point only has to compose the groups
# it actually needs.
# ---------------------------------------------------------------------------


def add_training_inference_args(parser: argparse.ArgumentParser) -> None:
    """Arguments consumed by both training codegen entry points."""
    parser.add_argument(
        "--num-data-inputs",
        type = int,
        dest = "num_data_inputs",
        default = None,
        help = "Number of DATA inputs that change per mini-batch. "
        "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--n-steps",
        type = int,
        dest = "n_steps",
        default = None,
        help = "N_TRAIN_STEPS: number of gradient-accumulation update steps. "
        "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--n-accum",
        type = int,
        dest = "n_accum",
        default = None,
        help = "N_ACCUM_STEPS: number of mini-batches per update step. "
        "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--learning-rate",
        type = float,
        dest = "learning_rate",
        default = 0.001,
        help = "SGD learning rate emitted as TRAINING_LEARNING_RATE in testinputs.h. Default: 0.001.",
    )
    parser.add_argument(
        "--tolerance",
        type = float,
        dest = "tolerance_abs",
        default = 1e-3,
        help = "Absolute loss tolerance emitted as TRAINING_TOLERANCE_ABS in testoutputs.h. Default: 1e-3.",
    )


def add_optimizer_training_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training-dir",
        type = str,
        default = None,
        help = "Directory containing the training network.onnx.  When provided, "
        "weight and grad-acc buffers are shared with TrainingNetwork instead "
        "of being allocated independently.",
    )


def resolve_optimizer_dir(test_dir: str, optimizer_dir: Optional[str]) -> str:
    """Return the optimizer ONNX directory for a training test.

    If ``optimizer_dir`` is explicitly set, it is returned as-is.  Otherwise
    fall back to ``<test_dir>/../<model>_optimizer``, where ``<model>`` is
    derived by replacing the ``_train`` suffix of the test directory's base
    name with ``_optimizer`` (e.g. ``simplemlp_train`` → ``simplemlp_optimizer``,
    ``sleepconvit_train`` → ``sleepconvit_optimizer``).
    """
    if optimizer_dir:
        return optimizer_dir
    test_path = Path(test_dir)
    optimizer_name = test_path.name.replace("_train", "_optimizer")
    return str(test_path.parent / optimizer_name)


def add_training_cmake_flags(cmd: List[str], training: bool, n_train_steps: Optional[int], n_accum_steps: Optional[int],
                             training_num_data_inputs: Optional[int]) -> None:
    """Append -DTRAINING=ON/OFF plus any known -DN_TRAIN_STEPS / -DN_ACCUM_STEPS /
    -DTRAINING_NUM_DATA_INPUTS defines to ``cmd``.  In-place."""
    cmd.append(f"-DTRAINING={'ON' if training else 'OFF'}")
    if not training:
        return
    if n_train_steps is not None:
        cmd.append(f"-DN_TRAIN_STEPS={n_train_steps}")
    if n_accum_steps is not None:
        cmd.append(f"-DN_ACCUM_STEPS={n_accum_steps}")
    if training_num_data_inputs is not None:
        cmd.append(f"-DTRAINING_NUM_DATA_INPUTS={training_num_data_inputs}")


def resolve_zo_update_dir(test_dir: str) -> str:  # -- QW
    """Return the zo_update ONNX directory for a ZO (MeZO) test. -- QW

    Derives the update-graph directory from the zo_train test directory by
    replacing the ``_zo_train`` suffix of the base name with ``_zo_update``
    (e.g. ``speechnet_zo_train`` -> ``speechnet_zo_update``). -- QW
    """  # -- QW
    test_path = Path(test_dir)  # -- QW
    update_name = test_path.name.replace("_zo_train", "_zo_update")  # -- QW
    return str(test_path.parent / update_name)  # -- QW


def add_mezo_cmake_flags(cmd: List[str], config) -> None:  # -- QW
    """Append -DMEZO_TRAINING=ON plus ZO hyperparameter -D's to ``cmd``. In-place. -- QW

    ZO_EPS / ZO_LR / ZO_Q / ZO_SEED come from ``config`` (falling back to the C
    harness defaults when unset). The float defines carry an ``f`` suffix so the
    C preprocessor produces float32 literals. -- QW
    """  # -- QW
    cmd.append("-DMEZO_TRAINING=ON")  # -- QW
    # BP TRAINING must be OFF so the MEZO branch is selected; execution.py's
    # add_training_cmake_flags already emitted -DTRAINING=OFF for a mezo config
    # (config.training stays False), so nothing to undo here. -- QW
    eps = config.zo_eps if config.zo_eps is not None else 0.01  # -- QW
    lr = config.zo_lr if config.zo_lr is not None else 3e-6  # -- QW
    cmd.append(f"-DZO_EPS={eps:.10g}f")  # -- QW
    cmd.append(f"-DZO_LR={lr:.10g}f")  # -- QW
    cmd.append(f"-DZO_Q={int(config.zo_q)}")  # -- QW
    cmd.append(f"-DZO_SEED={int(config.zo_seed)}")  # -- QW
    # SpeechNet ZO trains BN gamma/beta but uses FROZEN pretrained running stats
    # (the reference sim runs model.eval()). BatchNormInternal must therefore use
    # frozen stats, not batch stats — enable it by default for the ZO path so the
    # device matches the reference. -- QW
    cmd.append("-DBN_FROZEN_STATS=ON")  # -- QW
    # N_TRAIN_STEPS / N_ACCUM_STEPS / TRAINING_NUM_DATA_INPUTS are consumed by
    # the MEZO branch too; emit them from the (possibly auto-detected) config. -- QW
    if config.n_train_steps is not None:  # -- QW
        cmd.append(f"-DN_TRAIN_STEPS={config.n_train_steps}")  # -- QW
    if config.n_accum_steps is not None:  # -- QW
        cmd.append(f"-DN_ACCUM_STEPS={config.n_accum_steps}")  # -- QW
    if config.training_num_data_inputs is not None:  # -- QW
        cmd.append(f"-DTRAINING_NUM_DATA_INPUTS={config.training_num_data_inputs}")  # -- QW


def run_zo_codegen(config, script_dir: Path) -> None:  # -- QW
    """Drive the two-stage MeZO (ZO) codegen pipeline for one test. -- QW

    Near-clone of ``run_training_codegen`` (BP), but:
      * Stage 1 runs ``testMVPTraining.py --zo`` on the zo_train dir so the
        outputs header emits loss_plus / loss_minus references (not BP loss). -- QW
      * Stage 2 runs ``testMVPOptimizer.py`` on the zo_update dir (resolved via
        ``resolve_zo_update_dir``) with ``--training-dir`` pointing at zo_train,
        so its 22 PerturbRademacher inputs share the training weight buffers
        (in-place update). -- QW

    The BP ``run_training_codegen`` is left untouched. -- QW
    """  # -- QW
    if config.tiling:  # -- QW
        training_script = script_dir / "testMVPTraining.py"  # -- QW
        optimizer_script = script_dir / "testMVPOptimizer.py"  # -- QW
        opt_passthrough = ("--cores", "--l1", "--l2", "--defaultMemLevel", "--memAllocStrategy", "--searchStrategy",  # -- QW
                           "--plotMemAlloc", "--profileTiling", "--profileNodes", "--promoteToL2",  # -- QW
                           "--promoteToL2Strategy", "--promoteToL2IncludeActivations", "--promoteToL2MaxBufferBytes",  # -- QW
                           "--promoteToL2MinBufferBytes", "--promoteToL2Headroom")  # -- QW
        stage = "Tiled MeZO"  # -- QW
    else:  # -- QW
        # Non-tiled ZO codegen scripts are not provided; the ZO smoke bring-up
        # is tiled-only (Siracusa). Fail loudly rather than silently generate BP. -- QW
        raise RuntimeError("run_zo_codegen currently supports the tiled path only "  # -- QW
                           "(use the tiled Siracusa MeZO runner).")  # -- QW

    # --- Onnx4Deeploy emits the ZO train graph in the reference design (trainable weights as
    # INITIALIZERS / Perturb constant bases, BatchNormInternal, canonical 2-output SCE). The device
    # in-place update needs those weights in writable, name-shareable buffers, so we lower them to
    # graph INPUTS here on a COPY (raw fixture + exported artifact untouched); the prepped graph then
    # flows through the EXISTING training + optimizer codegen exactly like a BP graph, and
    # build_shared_buffer_maps can alias zo_update's outputs onto the shared weight inputs. -- QW
    # QW: Onnx4Deeploy's ZO export emits zo_train with the trainable weights already as graph INPUTS
    #     (byte-structurally like a BP training graph), so we consume the fixture DIRECTLY — no deploy-time
    #     promotion. The "make weights inputs" step lives in the exporter (zo_transform), keeping the flow
    #     clean and identical to BP. (build_shared_buffer_maps then aliases zo_update's outputs onto the
    #     zo_train weight inputs for the in-place cross-step update.) -- QW
    train_dir = config.test_dir  # -- QW

    # --- Stage 1: ZO training network (perturb -> fwd -> SoftmaxCE loss). --- -- QW
    cmd = [  # -- QW
        sys.executable,  # -- QW
        str(training_script),  # -- QW
        "-d",  # -- QW
        config.gen_dir,  # -- QW
        "-t",  # -- QW
        train_dir,  # -- QW  prepped zo_train (raw fixture untouched)
        "-p",  # -- QW
        config.platform,  # -- QW
        "--zo",  # -- QW  swap the outputs-header emitter to loss_plus/loss_minus
    ]  # -- QW
    if config.n_train_steps is not None:  # -- QW
        cmd.append(f"--n-steps={config.n_train_steps}")  # -- QW
    if config.n_accum_steps is not None:  # -- QW
        cmd.append(f"--n-accum={config.n_accum_steps}")  # -- QW
    if config.training_num_data_inputs is not None:  # -- QW
        cmd.append(f"--num-data-inputs={config.training_num_data_inputs}")  # -- QW
    if config.verbose > 0:  # -- QW
        cmd.append("-" + "v" * config.verbose)  # -- QW
    if config.debug:  # -- QW
        cmd.append("--debug")  # -- QW
    cmd.extend(config.gen_args)  # -- QW

    log.debug(f"[Execution] {stage} network generation command: {' '.join(cmd)}")  # -- QW
    if subprocess.run(cmd, check = False).returncode != 0:  # -- QW
        raise RuntimeError(f"{stage} network generation failed for {config.test_name}")  # -- QW

    # Read back auto-detected values written by the training generation script. -- QW
    meta_path = Path(config.gen_dir) / "training_meta.json"  # -- QW
    if meta_path.exists():  # -- QW
        with open(meta_path) as f:  # -- QW
            meta = json.load(f)  # -- QW
        config.n_train_steps = meta["n_train_steps"]  # -- QW
        config.n_accum_steps = meta["n_accum_steps"]  # -- QW
        config.training_num_data_inputs = meta["training_num_data_inputs"]  # -- QW
        log.info(f"[Execution] MeZO meta: {meta}")  # -- QW

    # --- Stage 2: ZO update network (22 in-place PerturbRademacher). --- -- QW
    upd_dir = resolve_zo_update_dir(config.test_dir)  # -- QW
    if config.optimizer_dir:  # -- QW  explicit override wins
        upd_dir = config.optimizer_dir  # -- QW
    if not Path(upd_dir).exists():  # -- QW
        log.warning(f"ZO update directory not found: {upd_dir} — skipping update codegen")  # -- QW
        return  # -- QW
    if not optimizer_script.exists():  # -- QW
        log.warning(f"{optimizer_script.name} not found — skipping update codegen")  # -- QW
        return  # -- QW

    upd_prepped = upd_dir  # -- QW  raw native zo_update dir (no prep needed)

    upd_cmd = [  # -- QW
        sys.executable,  # -- QW
        str(optimizer_script),  # -- QW
        "-d",  # -- QW
        config.gen_dir,  # -- QW
        "-t",  # -- QW
        upd_prepped,  # -- QW
        "-p",  # -- QW
        config.platform,  # -- QW
        f"--training-dir={train_dir}",  # -- QW  match prepped train graph names
    ]  # -- QW
    upd_cmd.extend(arg for arg in config.gen_args if any(arg.startswith(p) for p in opt_passthrough))  # -- QW
    if not any(arg.startswith("--defaultMemLevel") for arg in upd_cmd):  # -- QW
        upd_cmd.append("--defaultMemLevel=L2")  # -- QW
    if config.verbose > 0:  # -- QW
        upd_cmd.append("-" + "v" * config.verbose)  # -- QW

    log.debug(f"[Execution] {stage} update network generation command: {' '.join(upd_cmd)}")  # -- QW
    if subprocess.run(upd_cmd, check = False).returncode != 0:  # -- QW
        raise RuntimeError(f"{stage} update network generation failed for {config.test_name}")  # -- QW


def run_training_codegen(config, script_dir: Path) -> None:
    """Drive the two-stage training codegen pipeline for one test.

    Runs the training network codegen script (generateTrainingNetwork.py or
    testMVPTraining.py) followed by the matching optimizer codegen script
    (generateOptimizerNetwork.py or testMVPOptimizer.py), and writes back
    any auto-detected training parameters from ``training_meta.json`` into
    ``config``.

    The single entry point keeps ``testUtils.core.execution.generate_network``
    oblivious to training internals — it only has to call this and return.

    Parameters
    ----------
    config : DeeployTestConfig
        The test configuration (must have ``training=True``).  Training
        fields (``n_train_steps``, ``n_accum_steps``,
        ``training_num_data_inputs``) may be updated in-place from the
        training_meta.json written by the codegen script.
    script_dir : Path
        ``DeeployTest/`` — the directory that hosts the four codegen scripts.
    """
    if config.tiling:
        training_script = script_dir / "testMVPTraining.py"
        optimizer_script = script_dir / "testMVPOptimizer.py"
        opt_passthrough = ("--cores", "--l1", "--l2", "--defaultMemLevel", "--memAllocStrategy", "--searchStrategy",
                           "--plotMemAlloc", "--profileTiling", "--profileNodes", "--promoteToL2",
                           "--promoteToL2Strategy", "--promoteToL2IncludeActivations", "--promoteToL2MaxBufferBytes",
                           "--promoteToL2MinBufferBytes", "--promoteToL2Headroom")
        stage = "Tiled training"
    else:
        training_script = script_dir / "generateTrainingNetwork.py"
        optimizer_script = script_dir / "generateOptimizerNetwork.py"
        opt_passthrough = ("--cores", "--l1", "--l2", "--defaultMemLevel")
        stage = "Training"

    # --- Step 1: Training network (forward + backward + accumulation) ---
    cmd = [
        sys.executable,
        str(training_script),
        "-d",
        config.gen_dir,
        "-t",
        config.test_dir,
        "-p",
        config.platform,
    ]
    if config.n_train_steps is not None:
        cmd.append(f"--n-steps={config.n_train_steps}")
    if config.n_accum_steps is not None:
        cmd.append(f"--n-accum={config.n_accum_steps}")
    if config.training_num_data_inputs is not None:
        cmd.append(f"--num-data-inputs={config.training_num_data_inputs}")
    if config.verbose > 0:
        cmd.append("-" + "v" * config.verbose)
    if config.debug:
        cmd.append("--debug")
    cmd.extend(config.gen_args)

    log.debug(f"[Execution] {stage} network generation command: {' '.join(cmd)}")
    if subprocess.run(cmd, check = False).returncode != 0:
        raise RuntimeError(f"{stage} network generation failed for {config.test_name}")

    # Read back auto-detected values written by the training generation script.
    meta_path = Path(config.gen_dir) / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        config.n_train_steps = meta["n_train_steps"]
        config.n_accum_steps = meta["n_accum_steps"]
        config.training_num_data_inputs = meta["training_num_data_inputs"]
        log.info(f"[Execution] Training meta: {meta}")

    # --- Step 2: Optimizer network (SGD) ---
    opt_dir = resolve_optimizer_dir(config.test_dir, config.optimizer_dir)
    if not Path(opt_dir).exists():
        log.warning(f"Optimizer directory not found: {opt_dir} — skipping optimizer codegen")
        return
    if not optimizer_script.exists():
        log.warning(f"{optimizer_script.name} not found — skipping optimizer codegen")
        return

    opt_cmd = [
        sys.executable,
        str(optimizer_script),
        "-d",
        config.gen_dir,
        "-t",
        opt_dir,
        "-p",
        config.platform,
        f"--training-dir={config.test_dir}",
    ]
    opt_cmd.extend(arg for arg in config.gen_args if any(arg.startswith(p) for p in opt_passthrough))
    if not any(arg.startswith("--defaultMemLevel") for arg in opt_cmd):
        opt_cmd.append("--defaultMemLevel=L2")
    if config.verbose > 0:
        opt_cmd.append("-" + "v" * config.verbose)

    log.debug(f"[Execution] {stage} optimizer network generation command: {' '.join(opt_cmd)}")
    if subprocess.run(opt_cmd, check = False).returncode != 0:
        raise RuntimeError(f"{stage} optimizer network generation failed for {config.test_name}")
