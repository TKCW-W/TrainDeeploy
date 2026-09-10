/*
 * SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_HEADER_
#define __DEEPLOY_MATH_HEADER_

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BEGIN_SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0)

#include "DeeployBasicMath.h"

#include "dory_dma.h"
#include "dory_mem.h"

#include "pmsis.h"

// -- QW: the generated ZO training network references the perturb kernels + ZO runtime
//    globals (perturbation_sign / perturb_seed_base / perturb_eps_*). DeeployPULPMath.h
//    pulls these in for Siracusa; mirror it here so the GAP9-generated network compiles.
//    PULPOpen/inc is PUBLIC on deeploygap9, so these resolve. -- QW
#include "kernel/RandomNoise.h"
#include "kernel/ZORuntime.h"

#endif // __DEEPLOY_MATH_HEADER_
