/*
 * src/asm/arm64/k3_cl100k_neon.S: the K3 part of template cl100k, neon tier: k3_neon.inc with K3_O200K 0
 * (docs/kernels.md §3.1). kernels.h's toks_k3_parts runs it with the c twin src/core/k3_c.c: the tier's K3.
 *
 *   uint64_t toks_k3_scan_cl100k_neon(const toks_tables *t, toks_k3_args *a)
 */
#define K3_O200K 0
#define K3_FN toks_k3_scan_cl100k_neon
#include "k3_neon.inc"
