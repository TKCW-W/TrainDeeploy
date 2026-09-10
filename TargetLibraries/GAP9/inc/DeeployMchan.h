/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _DEEPLOY_MCHAN_H
#define _DEEPLOY_MCHAN_H

/*
 * GAP9 MCHAN v7 configuration wrapper for Deeploy
 *
 * This header configures and includes mchan.h with proper GAP9-specific
 * settings. Based on DORY's GAP9 DMA implementation.
 */

#include "pmsis.h"

// Define MCHAN base address if not already defined
#ifndef MCHAN_BASE_ADDR
#define MCHAN_BASE_ADDR (CLUSTER_PERIPHERALS_ADDR + CLUSTER_MCHAN_OFFSET)
#endif

// Define MCHAN version (GAP9 uses v7)
#ifndef MCHAN_VERSION
#define MCHAN_VERSION 7
#endif

// Use event-based synchronization (recommended for GAP9)
#ifndef MCHAN_POLLED
#define MCHAN_EVENT
#endif

// Define event bit for cluster DMA
#ifdef MCHAN_EVENT
#ifndef MCHAN_EVENT_BIT
#define MCHAN_EVENT_BIT (CLUSTER_IRQ_DMA0) // Typically 8
#endif
#endif

// Now include the mchan.h header with all configurations set (GAP9 dory API:
// mchan_transfer_push_1d/2d / get_id / wait / free)
#include "mchan.h"

// -- QW: the generated tiling-DMA + reused-PULPOpen op templates also emit the Deeploy pulp
//    Mchan API (mchan_channel_alloc/free/wait, mchan_transfer_1d/2d_*) which GAP9's mchan.h
//    does NOT provide. GAP9 IS Mchan v7 (MCHAN_VERSION==7 above => MCHAN_TRANSFER_LEN_SIZE 17,
//    identical CMD_FLAG encoding to mchan_v7.h), so mchan_v7.h drives the SAME hardware
//    correctly and its macros redefine identically (legal, no clash) while its function names
//    are disjoint from GAP9's mchan.h. Include it to supply the channel API. mchan_v7.h also
//    defines MCHAN_CHANNEL_ID_MAX, so the manual fallback below is now belt-and-braces. -- QW
// -- QW: mchan_v7.h calls assert() and relies on the includer providing it (Siracusa does via
//    its toolchain; the GAP9 freestanding toolchain does not, giving an implicit-decl error).
//    These are non-essential debug bounds checks — provide a no-op if assert is unavailable. -- QW
#include <assert.h>
#ifndef assert
#define assert(x) ((void)0)
#endif
#include "mchan_v7.h"

#ifndef MCHAN_CHANNEL_ID_MAX
#define MCHAN_CHANNEL_ID_MAX (15)
#endif

#endif // _DEEPLOY_MCHAN_H
