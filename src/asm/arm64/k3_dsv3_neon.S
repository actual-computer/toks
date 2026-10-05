/*
 * src/asm/arm64/k3_dsv3_neon.S: the K3 part of template dsv3, neon tier: k3_neon.inc with K3_O200K 1 and
 * K3_DSV3 1 (docs/templates/dsv3.md §7, kernels.md §3.1). kernels.h's toks_k3_parts runs it with the c twin
 * src/core/k3_dsv3_c.c: the tier's K3.
 *
 *   uint64_t toks_k3_scan_dsv3_neon(const toks_tables *t, toks_k3_args *a)
 */
#define K3_O200K 1
#define K3_DSV3 1
#define K3_FN toks_k3_scan_dsv3_neon
#include "k3_neon.inc"
