# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Entry point for the Siracusa MeZO (ZO) training test runner. -- QW

Near-clone of testUtils.deeployTrainingRunner.main, but sets ``mezo=True`` on the
DeeployTestConfig (driving the ZO two-graph codegen + MEZO_TRAINING harness) and
adds --eps / --lr / --q / --seed for the ZO hyperparameters.

Usage:
    from testUtils.deeployMezoRunner import main
    sys.exit(main(tiling_enabled=True))    # tiled (SBTiler) — the only supported ZO path
"""

import os
from pathlib import Path

# gapy (gvsoc launcher) uses `#!/usr/bin/env python3`.  On Siracusa /usr/bin/python3 has the
# required packages, so /usr/bin goes first. On GAP9 the SDK tools (kconfigtool / gapy_v2) need
# the SDK venv python3 (kconfiglib + SDK python deps), so when a venv is ACTIVE keep its bin ahead
# of /usr/bin — otherwise the shebang picks /usr/bin/python3 and CMake fails on `No module named
# 'kconfiglib'`. No-op for Siracusa (no VIRTUAL_ENV). -- QW
# os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')   # -- QW original (Siracusa-only)
_venv = os.environ.get('VIRTUAL_ENV')  # -- QW
_venv_pfx = (_venv + '/bin:') if _venv else ''  # -- QW
os.environ['PATH'] = _venv_pfx + '/usr/bin:' + os.environ.get('PATH', '')  # -- QW

from .core import DeeployTestConfig, run_complete_test
from .core.paths import get_test_paths
from .deeployRunner import DeeployRunnerArgumentParser, print_colored_result, print_configuration


def main(tiling_enabled: bool = True, default_platform: str = 'Siracusa', default_simulator: str = 'gvsoc'):
    """Build parser, parse args, create a mezo=True DeeployTestConfig, run the ZO test. -- QW"""

    parser = DeeployRunnerArgumentParser(tiling_arguments = tiling_enabled, platform_required = False)

    parser.add_argument('--cores', type = int, default = 8, help = 'Number of cluster cores (default: 8)\n')
    parser.add_argument('--n-steps',
                        metavar = '<N>',
                        dest = 'n_steps',
                        type = int,
                        default = None,
                        help = 'N_TRAIN_STEPS: ZO update steps (auto-detected if not given)\n')
    parser.add_argument('--n-accum',
                        metavar = '<N>',
                        dest = 'n_accum',
                        type = int,
                        default = None,
                        help = 'N_ACCUM_STEPS: mini-batches per update step (auto-detected if not given)\n')
    parser.add_argument('--num-data-inputs',
                        metavar = '<N>',
                        dest = 'num_data_inputs',
                        type = int,
                        default = None,
                        help = 'Inputs that change each mini-batch (auto-detected if not given)\n')
    parser.add_argument('--optimizer-dir',
                        metavar = '<dir>',
                        dest = 'optimizer_dir',
                        type = str,
                        default = None,
                        help = 'Directory containing the zo_update network.onnx '
                        "(default: auto-derived by replacing '_zo_train' with '_zo_update')\n")
    parser.add_argument('--tolerance',
                        metavar = '<tol>',
                        dest = 'tolerance',
                        type = float,
                        default = None,
                        help = 'Absolute loss tolerance for pass/fail comparison\n')
    # --- ZO hyperparameters --- -- QW
    parser.add_argument('--eps',
                        metavar = '<eps>',
                        dest = 'eps',
                        type = float,
                        default = 0.01,
                        help = 'MeZO perturbation epsilon (default: 0.01)\n')
    parser.add_argument('--lr',
                        metavar = '<lr>',
                        dest = 'lr',
                        type = float,
                        default = 3e-6,
                        help = 'MeZO learning rate (default: 3e-6)\n')
    parser.add_argument('--q',
                        metavar = '<q>',
                        dest = 'q',
                        type = int,
                        default = 1,
                        help = 'MeZO number of random directions per step (default: 1)\n')
    parser.add_argument('--seed',
                        metavar = '<seed>',
                        dest = 'seed',
                        type = int,
                        default = 0,
                        help = 'MeZO base seed (default: 0)\n')

    args = parser.parse_args()

    platform = default_platform
    simulator = args.simulator if args.simulator else default_simulator

    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent

    gen_dir, test_dir_abs, test_name = get_test_paths(args.dir, platform, base_dir = str(base_dir))

    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    build_dir = str(base_dir / f'TEST_{platform.upper()}' / f'build_{worker_id}')

    cmake_args = [f'-DNUM_CORES={args.cores}']
    if args.cmake:
        cmake_args.extend(args.cmake)

    gen_args = [f'--cores={args.cores}']
    if args.tolerance is not None:
        gen_args.append(f'--tolerance={args.tolerance}')
    if args.input_type_map:
        gen_args.extend(['--input-type-map'] + list(args.input_type_map))
    if args.input_offset_map:
        gen_args.extend(['--input-offset-map'] + list(args.input_offset_map))

    if tiling_enabled:
        if getattr(args, 'defaultMemLevel', None):
            gen_args.append(f'--defaultMemLevel={args.defaultMemLevel}')
        if getattr(args, 'l1', None):
            gen_args.append(f'--l1={args.l1}')
        if getattr(args, 'l2', None) and args.l2 != 1024000:
            gen_args.append(f'--l2={args.l2}')
        if getattr(args, 'memAllocStrategy', None):
            gen_args.append(f'--memAllocStrategy={args.memAllocStrategy}')
        if getattr(args, 'searchStrategy', None):
            gen_args.append(f'--searchStrategy={args.searchStrategy}')
        if getattr(args, 'profileTiling', False):
            gen_args.append('--profileTiling')
        if getattr(args, 'profileNodes', None):
            gen_args.append(f"--profileNodes={','.join(args.profileNodes)}")
        if getattr(args, 'plotMemAlloc', False):
            gen_args.append('--plotMemAlloc')
        if getattr(args, 'promoteToL2', False):
            gen_args.append('--promoteToL2')
            gen_args.append(f'--promoteToL2Strategy={args.promoteToL2Strategy}')
            gen_args.append('--promoteToL2IncludeActivations')
            gen_args.append('--promoteToL2MaxBufferBytes=0')
            gen_args.append(f'--promoteToL2Headroom={args.promoteToL2Headroom}')

    config = DeeployTestConfig(
        test_name = test_name,
        test_dir = test_dir_abs,
        platform = platform,
        simulator = simulator,
        tiling = tiling_enabled,
        gen_dir = gen_dir,
        build_dir = build_dir,
        toolchain = args.toolchain,
        toolchain_install_dir = args.toolchain_install_dir,
        cmake_args = cmake_args,
        gen_args = gen_args,
        verbose = args.verbose,
        debug = args.debug,
        training = False,  # -- QW  BP path OFF; ZO path is selected via mezo=True
        mezo = True,  # -- QW
        n_train_steps = args.n_steps,
        n_accum_steps = args.n_accum,
        training_num_data_inputs = args.num_data_inputs,
        optimizer_dir = args.optimizer_dir,
        zo_eps = args.eps,  # -- QW
        zo_lr = args.lr,  # -- QW
        zo_q = args.q,  # -- QW
        zo_seed = args.seed,  # -- QW
    )

    print_configuration(config)

    try:
        result = run_complete_test(config, skipgen = args.skipgen, skipsim = args.skipsim)
        print_colored_result(result, config.test_name)
        return 0 if result.success else 1
    except Exception as e:
        RED = '\033[91m'
        RESET = '\033[0m'
        print(f'\n{RED}✗ Test {config.test_name} FAILED with exception: {e}{RESET}')
        return 1
