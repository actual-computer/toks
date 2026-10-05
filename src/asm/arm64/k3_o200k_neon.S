/*
 * src/asm/arm64/k3_o200k_neon.S: the K3 part of template o200k, neon tier: k3_neon.inc with K3_O200K 1
 * (docs/templates/o200k.md §4, kernels.md §3.1). kernels.h's toks_k3_parts runs it with the c twin
 * src/core/k3_o200k_c.c: the tier's K3.
 *
 *   uint64_t toks_k3_scan_o200k_neon(const toks_tables *t, toks_k3_args *a)
 */
#define K3_O200K 1
#define K3_FN toks_k3_scan_o200k_neon
#include "k3_neon.inc"
