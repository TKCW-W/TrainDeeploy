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

/* QW (dual_pass_fusion exp7, step 3): int8 weight, both signs, for Ci > 1. The source `w` is the RAW CHW
 * weight [co][ci][ky][kx] -- the layout the reference RQSPerturbRademacher node perturbs -- and this core walks
 * ITS CHW chunk [start, start+local) in memory order, so noise bit i lands on CHW element i exactly as in the
 * reference (same stream, chunking, seed, rounding, clamp as dc_perturb_i8). Each result is written to the
 * element's HWC position j = ((co*kh + ky)*kw + kx)*ci + c_i in wp / wm, the layout the conv loop reads.
 * dc_perturb_i8 on an already transposed weight is only correct for ci == 1 (block 0), where the two orders
 * coincide. The (co, ci, ky, kx) position is decoded once and then advanced incrementally (no division per
 * element); the multiplier is reloaded only when co changes. -- QW */
static inline void dc_perturb_i8_chw_v1(const int8_t *w, int8_t *wp, int8_t *wm, const int32_t *M, uint32_t ch_in,
                                        uint32_t kh, uint32_t kw, uint32_t seed, uint32_t start, uint32_t local,
                                        uint32_t S, float eps_scale) {
  /* QW exp7 step 3 version (43-51 cycles per element for both signs); kept as the reference of the step-5b rewrite */
  const uint32_t K = ch_in * kh * kw;
  uint32_t co = start / K;
  uint32_t rem = start - co * K;          /* CHW: rem = (ci*kh + ky)*kw + kx */
  uint32_t kx = rem % kw;
  uint32_t t = rem / kw;
  uint32_t ky = t % kh;
  uint32_t ci = t / kh;
  uint32_t j = co * K + (ky * kw + kx) * ch_in + ci;
  int32_t m = 0;
  if (local) {
    m = M[co];
    if (eps_scale != 1.0f)
      m = (int32_t)lrintf((float)m * eps_scale);
  }
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
      const int32_t prod = r * m;
      const int32_t nqp = (prod + rounding) >> (int32_t)S;
      const int32_t nqm = (rounding - prod) >> (int32_t)S;
      const int32_t v = (int32_t)w[i];
      wp[j] = (int8_t)DC_CLAMP(v + nqp, -127, 127);
      wm[j] = (int8_t)DC_CLAMP(v + nqm, -127, 127);
      bits >>= 1;
      /* advance (co, ci, ky, kx) in CHW order; j follows in HWC */
      if (++kx < kw) {
        j += ch_in;
        continue;
      }
      kx = 0;
      if (++ky == kh) {
        ky = 0;
        if (++ci == ch_in) {
          ci = 0;
          if (i + 1u < end) {
            m = M[++co];
            if (eps_scale != 1.0f)
              m = (int32_t)lrintf((float)m * eps_scale);
          } else {
            ++co;
          }
        }
      }
      j = co * K + ky * kw * ch_in + ci;
    }
  }
}

/* QW (dual_pass_fusion exp7, step 5b): same contract and bit-identical result as dc_perturb_i8_chw_v1, cheaper.
 *  - r is +-1, so per output channel there are only two noise values: up = (m + rnd) >> S and dn = (rnd - m) >> S;
 *    r = +1 gives (+pass, -pass) = (up, dn), r = -1 gives (dn, up). Computed once per channel; the noise bit
 *    selects branch-free (xor mask) -- no multiply, no shift per element.
 *  - within one (co, ci) row the CHW chunk is kh*kw contiguous elements whose HWC positions advance by ch_in, for
 *    1xK and Kx1 kernels alike. The element loop runs over RUNS of min(bits left in the draw, row left, chunk left)
 *    elements, so its body has no branch (a hardware loop on RI5CY); row / channel / draw changes happen between runs.
 *  - out of line (noinline): inlined into the conv kernel it ran short of registers and reloaded its pointers from
 *    the stack every element (step-5b disassembly). -- QW */
static inline __attribute__((noinline)) void dc_perturb_i8_chw(const int8_t *w, int8_t *wp, int8_t *wm, const int32_t *M,
                                                        uint32_t ch_in, uint32_t kh, uint32_t kw, uint32_t seed,
                                                        uint32_t start, uint32_t local, uint32_t S, float eps_scale) {
  if (!local)
    return;
  const uint32_t L = kh * kw;
  const uint32_t K = ch_in * L;
  uint32_t co = start / K;
  uint32_t rem = start - co * K; /* CHW: rem = ci*L + t */
  uint32_t ci = rem / L;
  uint32_t t = rem - ci * L;
  const int32_t rnd = (S > 0u) ? (int32_t)(1u << (S - 1u)) : 0;
  int32_t m = M[co];
  if (eps_scale != 1.0f)
    m = (int32_t)lrintf((float)m * eps_scale);
  int32_t up = (m + rnd) >> (int32_t)S;
  int32_t dn = (rnd - m) >> (int32_t)S;
  const int8_t *src = w + start;
  int8_t *pp = wp + co * K + t * ch_in + ci;
  int8_t *pm = wm + co * K + t * ch_in + ci;
  uint32_t rng = seed * 1664525u + 1013904223u;
  uint32_t bits = 0, nbits = 0, left = local;
  while (left) {
    if (nbits == 0u) {
      rng = dc_xorshift32(rng);
      bits = rng;
      nbits = 32u;
    }
    uint32_t n = L - t;
    if (n > nbits)
      n = nbits;
    if (n > left)
      n = left;
    const int32_t d = up ^ dn;
    for (uint32_t k = 0; k < n; k++) {
      const int32_t v = (int32_t)*src++;
      const int32_t np = dn ^ (d & -(int32_t)(bits & 1u));
      const int32_t nm = np ^ d;
      *pp = (int8_t)DC_CLAMP(v + np, -127, 127);
      *pm = (int8_t)DC_CLAMP(v + nm, -127, 127);
      pp += ch_in;
      pm += ch_in;
      bits >>= 1;
    }
    left -= n;
    nbits -= n;
    t += n;
    if (t == L && left) { /* next (co, ci) row */
      t = 0;
      if (++ci == ch_in) {
        ci = 0;
        ++co;
        m = M[co];
        if (eps_scale != 1.0f)
          m = (int32_t)lrintf((float)m * eps_scale);
        up = (m + rnd) >> (int32_t)S;
        dn = (rnd - m) >> (int32_t)S;
      }
      pp = wp + co * K + ci;
      pm = wm + co * K + ci;
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
