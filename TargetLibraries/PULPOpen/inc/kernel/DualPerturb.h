/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp6): the "both signs from one draw" perturbation helpers, shared.
 *
 * Lifted VERBATIM out of DualConv.c (exp5) so the NE16 fused path (NE16DualPrep.c, GAP9) uses the
 * same code as the cluster fused kernel: same per-core chunking as the standalone RQSPerturbRademacher
 * node (`(size >> log2(cores)) + (size % cores != 0)`), same LCG warm-up + Xorshift32 stream (one
 * 32-bit draw per 32 elements, LSB first), same per-output-channel multiplier, same rounding, same
 * int8 clamp -- and the exp5 §7.3 rule: form `prod = r*m` once and derive the -eps noise as
 * `(rounding - prod) >> S`, never as `(-r)*m` (mis-evaluated by the PULP clang-15 build). -- QW
 */
#ifndef __DEEPLOY_MATH_DUALPERTURB_KERNEL_HEADER_
#define __DEEPLOY_MATH_DUALPERTURB_KERNEL_HEADER_

#include <math.h>
#include <stdint.h>

#ifndef DC_CLAMP
#define DC_CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

static inline uint32_t dc_xorshift32(uint32_t s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

/* The perturb nodes' per-core split: chunk = (size >> log2(NUM_CORES)) + (size % NUM_CORES != 0). */
static inline void dc_chunk(uint32_t size, uint32_t num_cores, uint32_t core_id, uint32_t *start,
                            uint32_t *local) {
  uint32_t log2c = 0;
  while ((1u << (log2c + 1u)) <= num_cores)
    log2c++;
  uint32_t chunk = (size >> log2c) + ((size & (num_cores - 1u)) != 0u);
  uint32_t s = chunk * core_id;
  if (s > size)
    s = size;
  uint32_t e = s + chunk;
  if (e > size)
    e = size;
  *start = s;
  *local = e - s;
}

/* int8 weight: both signs from one stream, clamped to [-127, 127] like ApplyPerturbQuantRademacher_CHW. */
static inline void dc_perturb_i8(const int8_t *w, int8_t *wp, int8_t *wm, const int32_t *M,
                                 uint32_t channel_width, uint32_t seed, uint32_t start, uint32_t local,
                                 uint32_t S, float eps_scale) {
  uint32_t rng = seed * 1664525u + 1013904223u;
  const int32_t rounding = (S > 0u) ? (int32_t)(1u << (S - 1u)) : 0;
  uint32_t i = start;
  const uint32_t end = start + local;
  while (i < end) {
    rng = dc_xorshift32(rng);
    uint32_t bits = rng;
    uint32_t n = end - i;
    if (n > 32u)
      n = 32u;
    for (uint32_t k = 0; k < n; k++, i++) {
      const int32_t r = (bits & 1u) ? 1 : -1;
      int32_t m = M[i / channel_width];
      if (eps_scale != 1.0f)
        m = (int32_t)lrintf((float)m * eps_scale);
      const int32_t prod = r * m;
      const int32_t nqp = (prod + rounding) >> (int32_t)S;
      const int32_t nqm = (rounding - prod) >> (int32_t)S;
      const int32_t v = (int32_t)w[i];
      wp[i] = (int8_t)DC_CLAMP(v + nqp, -127, 127);
      wm[i] = (int8_t)DC_CLAMP(v + nqm, -127, 127);
      bits >>= 1;
    }
  }
}

/* int32 requant-add: same stream shape, NO clamp, like ApplyPerturbQuantRademacher_i32. */
static inline void dc_perturb_i32(const int32_t *b, int32_t *bp, int32_t *bm, const int32_t *M,
                                  uint32_t channel_width, uint32_t seed, uint32_t start, uint32_t local,
                                  uint32_t S, float eps_scale) {
  uint32_t rng = seed * 1664525u + 1013904223u;
  const int32_t rounding = (S > 0u) ? (int32_t)(1u << (S - 1u)) : 0;
  uint32_t i = start;
  const uint32_t end = start + local;
  while (i < end) {
    rng = dc_xorshift32(rng);
    uint32_t bits = rng;
    uint32_t n = end - i;
    if (n > 32u)
      n = 32u;
    for (uint32_t k = 0; k < n; k++, i++) {
      const int32_t r = (bits & 1u) ? 1 : -1;
      int32_t m = M[i / channel_width];
      if (eps_scale != 1.0f)
        m = (int32_t)lrintf((float)m * eps_scale);
      const int32_t prod = r * m;
      bp[i] = b[i] + ((prod + rounding) >> (int32_t)S);
      bm[i] = b[i] + ((rounding - prod) >> (int32_t)S);
      bits >>= 1;
    }
  }
}

#endif
