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

/* -- QW (exp16c / blocker 1b): device-side NE16 bit-serial weight encoder. Declared here (rather
 * than in a pulp-nnx header) so TargetLibraries/third_party/pulp-nnx stays pristine -- this is a
 * plain C kernel, not an ISA change. Implementation: TargetLibraries/GAP9/src/NE16WeightEncode.c */
void NE16WeightEncode_i8_u8(const int8_t *__restrict__ src, uint8_t *__restrict__ dst,
                            const uint32_t cout, const uint32_t cin, const uint32_t taps,
                            const uint32_t row_start, const uint32_t row_count);

/* -- QW (exp16c phase 4 / BLOCKER 3): per-output-channel bias correction for signed activations
 * fed to NE16 as x+128. See TargetLibraries/GAP9/src/NE16WeightEncode.c for the derivation. */
void NE16SignedInputBias_i32(const int8_t *__restrict__ weight, const int32_t *__restrict__ mul,
                             const int32_t *__restrict__ add, int32_t *__restrict__ out,
                             const uint32_t cout, const uint32_t cinTaps, const int32_t offset,
                             const uint32_t co_start, const uint32_t co_count);

#endif // __DEEPLOY_MATH_HEADER_
