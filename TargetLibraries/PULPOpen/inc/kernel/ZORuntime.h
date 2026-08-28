/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * ZO (MeZO) runtime perturb controls. -- QW
 * Mutable globals the ZO runner (C main) sets before each perturbed forward / update,
 * so that the Perturb* kernel templates can drive +eps/-eps (perturbation_sign),
 * per-step seeds (perturb_seed_base), and the runtime update coefficient
 * (perturb_eps_override). Defaults are NEUTRAL: sign=+1, seed_base=0, override off ->
 * the templates fall back to the per-node baked seed/eps (inference / isolated-kernel
 * behavior is unchanged).
 */
#ifndef __DEEPLOY_ZO_RUNTIME_HEADER_
#define __DEEPLOY_ZO_RUNTIME_HEADER_

#include "DeeployPULPMath.h"

// +eps (dir=1) / -eps (dir=0), shared across all perturb nodes in a pass.
extern uint32_t perturbation_sign;
// Per-step seed base added onto each node's baked seed:  eff_seed = baked_seed + perturb_seed_base.
extern uint32_t perturb_seed_base;
// Runtime override for the perturb magnitude (used for the ZO update coefficient |lr*g_proj|).
extern float32_t perturb_eps_override;
// When non-zero, kernels use perturb_eps_override instead of the baked ${eps}.
extern uint32_t perturb_eps_use_override;
// The baked export-time eps (the magnitude encoded in the RQSPerturb integer `mul` vectors). The ZO
// runner sets it to ZO_EPS so integer perturb kernels can scale mul by (override/baked) when the
// override is active (the zo_update coefficient -lr*g_proj). Default 1.0 = no scaling. -- QW
extern float32_t perturb_eps_baked;

#endif //__DEEPLOY_ZO_RUNTIME_HEADER_
