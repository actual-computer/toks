/*
 * src/asm/x86_64/k3_cl100k_avx2.S: the K3 part of template cl100k, avx2 tier: k3_avx2.inc with K3_O200K 0
 * (docs/kernels.md §3.1). kernels.h's toks_k3_parts runs it with the c twin src/core/k3_c.c: the tier's K3.
 *
 *   uint64_t toks_k3_scan_cl100k_avx2(const toks_tables *t, toks_k3_args *a)
 */
#define K3_O200K 0
#define K3_FN toks_k3_scan_cl100k_avx2
#include "k3_avx2.inc"
