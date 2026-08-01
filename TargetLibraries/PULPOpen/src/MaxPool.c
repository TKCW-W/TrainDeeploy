/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

/* QW: argmax-flip evidence — rolling hash of MaxPoolGrad argmax positions, gated by a
 * runtime enable flag set only by the training harness (zero overhead when off). -- QW */
uint32_t g_maxpool_argmax_sig = 0u;   /* Sum of within-window argmax offsets   */
uint32_t g_maxpool_argmax_sig2 = 0u;  /* Sum of squared offsets (collision guard) */
uint32_t g_maxpool_argmax_en = 0u;

void PULP_MaxPool2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                  uint32_t W, uint32_t H, uint32_t C,
                                  uint32_t Q, uint32_t P, uint32_t SQ,
                                  uint32_t SP, float32_t *__restrict__ pDstC,
                                  uint32_t pad_top, uint32_t pad_bottom,
                                  uint32_t pad_left, uint32_t pad_right) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C);
  uint16_t ch_count = ch_stop - ch_start;

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        float32_t max_val = -inf;

        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;
        for (uint32_t p = 0; p < P; ++p) {
          int32_t h_in = h_in_start + p;

          if (h_in < 0 || h_in >= (int32_t)H) {
            continue;
          }

          for (uint32_t q = 0; q < Q; ++q) {
            int32_t w_in = w_in_start + q;

            if (w_in < 0 || w_in >= (int32_t)W) {
              continue;
            }

            uint32_t input_idx = (h_in * W + w_in) * C + c;
            float32_t val = pSrcA[input_idx];

            if (val > max_val) {
              max_val = val;
            }
          }
        }

        uint32_t output_idx = (h_out * W_out + w_out) * C + c;
        pDstC[output_idx] = max_val;
      }
    }
  }
}

void PULP_MaxPoolGrad2d_fp32_fp32_HWC(
    const float32_t *__restrict__ pGradOut,
    const float32_t *__restrict__ pInput, uint32_t H_out, uint32_t W_out,
    uint32_t C, uint32_t H_in, uint32_t W_in, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, float32_t *__restrict__ pGradIn, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C);

  /* Zero-initialise the gradient input for our channel slice */
  for (uint32_t h = 0; h < H_in; ++h) {
    for (uint32_t w = 0; w < W_in; ++w) {
      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        pGradIn[(h * W_in + w) * C + c] = 0.0f;
      }
    }
  }

  /* Scatter upstream gradient to the argmax position in each pooling window */
  for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {

      int32_t h_in_start = (int32_t)h_out * (int32_t)SP - (int32_t)pad_top;
      int32_t w_in_start = (int32_t)w_out * (int32_t)SQ - (int32_t)pad_left;

      for (uint32_t c = ch_start; c < ch_stop; ++c) {

        /* Find the argmax position within the pooling window */
        float32_t max_val = -inf;
        int32_t max_h = -1;
        int32_t max_w = -1;

        for (uint32_t p = 0; p < P; ++p) {
          int32_t h_in = h_in_start + (int32_t)p;
          if (h_in < 0 || h_in >= (int32_t)H_in)
            continue;

          for (uint32_t q = 0; q < Q; ++q) {
            int32_t w_in = w_in_start + (int32_t)q;
            if (w_in < 0 || w_in >= (int32_t)W_in)
              continue;

            float32_t val =
                pInput[((uint32_t)h_in * W_in + (uint32_t)w_in) * C + c];
            if (val > max_val) {
              max_val = val;
              max_h = h_in;
              max_w = w_in;
            }
          }
        }

        /* Accumulate upstream gradient at the argmax position */
        if (max_h >= 0 && max_w >= 0) {
          uint32_t out_idx = (h_out * W_out + w_out) * C + c;
          uint32_t in_idx = ((uint32_t)max_h * W_in + (uint32_t)max_w) * C + c;
          pGradIn[in_idx] += pGradOut[out_idx];
        }
      }
    }
  }

  /* QW: argmax-flip evidence instrumentation ------------------------------- QW
   * Core 0 re-scans ALL channels and accumulates Sum and SumSq of the *within-window*
   * argmax offset (which element of the pooling window is the max).  That offset is
   * TILE-invariant and the Sum/SumSq combiners are ORDER-invariant, so the device
   * (tiled, multi-tile-call) value equals the ORT host (untiled) value when the argmax
   * agrees, and differs the moment a tie flips.  Dormant unless harness sets the flag. */
  if (core_id == 0 && g_maxpool_argmax_en) {
    for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
      for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
        int32_t h0 = (int32_t)h_out * (int32_t)SP - (int32_t)pad_top;
        int32_t w0 = (int32_t)w_out * (int32_t)SQ - (int32_t)pad_left;
        for (uint32_t c = 0; c < C; ++c) {
          float32_t mv = -inf;
          int32_t mh = -1, mw = -1;
          for (uint32_t p = 0; p < P; ++p) {
            int32_t hi = h0 + (int32_t)p;
            if (hi < 0 || hi >= (int32_t)H_in) continue;
            for (uint32_t q = 0; q < Q; ++q) {
              int32_t wi = w0 + (int32_t)q;
              if (wi < 0 || wi >= (int32_t)W_in) continue;
              float32_t v = pInput[((uint32_t)hi * W_in + (uint32_t)wi) * C + c];
              if (v > mv) { mv = v; mh = hi; mw = wi; }
            }
          }
          /* within-window argmax offset (tile-invariant); +1 so "no max" (=0) differs */
          uint32_t off = (mh >= 0) ? (uint32_t)((mh - h0) * (int32_t)Q + (mw - w0)) + 1u : 0u;
          g_maxpool_argmax_sig  += off;
          g_maxpool_argmax_sig2 += off * off;
        }
      }
    }
  }
  /* QW: end argmax-flip evidence ------------------------------------------- QW */
}

/* QW: Part-4 argmax-mask forward -------------------------------------------- QW
 * Emits the WITHIN-WINDOW argmax offset (p*Q + q, uint8) per output element, using the
 * SAME window scan + tie-break (strict >) as PULP_MaxPool2d so the stored winner matches
 * the pooled value. The offset is tile-invariant (reconstructed locally in backward from
 * the output position), so it survives tiling. This lets MaxPoolGrad scatter WITHOUT the
 * big forward activation, so that activation can be freed right after the forward pass.
 * Same (W,H,C,Q,P,SQ,SP) arg order as PULP_MaxPool2d. -- QW */
void PULP_MaxPoolArgmax2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                        uint32_t W, uint32_t H, uint32_t C,
                                        uint32_t Q, uint32_t P, uint32_t SQ,
                                        uint32_t SP, float32_t *__restrict__ pMask,
                                        uint32_t pad_top, uint32_t pad_bottom,
                                        uint32_t pad_left, uint32_t pad_right) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C);

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        float32_t max_val = -inf;
        uint32_t best_off = 0; /* within-window offset p*Q + q of the winner */

        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;
        for (uint32_t p = 0; p < P; ++p) {
          int32_t h_in = h_in_start + (int32_t)p;
          if (h_in < 0 || h_in >= (int32_t)H) {
            continue;
          }
          for (uint32_t q = 0; q < Q; ++q) {
            int32_t w_in = w_in_start + (int32_t)q;
            if (w_in < 0 || w_in >= (int32_t)W) {
              continue;
            }
            float32_t val = pSrcA[((uint32_t)h_in * W + (uint32_t)w_in) * C + c];
            if (val > max_val) {
              max_val = val;
              best_off = p * Q + q;
            }
          }
        }

        /* store offset as a float32 (exactly representable for small windows) */
        pMask[(h_out * W_out + w_out) * C + c] = (float32_t)best_off;
      }
    }
  }
}

/* QW: Part-4 mask-consuming MaxPoolGrad ------------------------------------- QW
 * Reads the within-window argmax offset from pMask (produced by PULP_MaxPoolArgmax2d)
 * and scatters the upstream gradient to that position — NO recompute from a stored
 * forward activation. The target input position is reconstructed LOCALLY from the output
 * position + offset (p = off/Q, q = off%Q), so it is tiling-invariant. Q here MUST be the
 * same width-kernel dim used to encode the offset in the argmax kernel. -- QW */
void PULP_MaxPoolGradMask2d_fp32_fp32_HWC(
    const float32_t *__restrict__ pGradOut,
    const float32_t *__restrict__ pMask, uint32_t H_out, uint32_t W_out,
    uint32_t C, uint32_t H_in, uint32_t W_in, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, float32_t *__restrict__ pGradIn, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C);

  /* Zero-initialise the gradient input for our channel slice */
  for (uint32_t h = 0; h < H_in; ++h) {
    for (uint32_t w = 0; w < W_in; ++w) {
      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        pGradIn[(h * W_in + w) * C + c] = 0.0f;
      }
    }
  }

  /* Scatter upstream gradient to the stored argmax position in each window */
  for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {

      int32_t h_in_start = (int32_t)h_out * (int32_t)SP - (int32_t)pad_top;
      int32_t w_in_start = (int32_t)w_out * (int32_t)SQ - (int32_t)pad_left;

      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        uint32_t out_idx = (h_out * W_out + w_out) * C + c;
        uint32_t off = (uint32_t)(pMask[out_idx] + 0.5f);
        uint32_t p = off / Q;
        uint32_t q = off % Q;
        int32_t h_in = h_in_start + (int32_t)p;
        int32_t w_in = w_in_start + (int32_t)q;
        if (h_in >= 0 && h_in < (int32_t)H_in && w_in >= 0 &&
            w_in < (int32_t)W_in) {
          pGradIn[((uint32_t)h_in * W_in + (uint32_t)w_in) * C + c] +=
              pGradOut[out_idx];
        }
      }
    }
  }
}