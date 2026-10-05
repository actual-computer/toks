/* k3_dsv3_c.c: K3's c twin, template dsv3 (docs/templates/dsv3.md §2-3), docs/notes/c-core.md §k3_dsv3_c.c.1 */
#include "kernels.h"

#define TOKS_D3_KIND (TOKS_C_BASE_MASK | TOKS_C_CJK)

/* the kind of the atom at p (avail >= 1 bytes left) and its byte length *k (kernels.h toks_k3_atom: an ill-formed
 * sequence starts over at each byte, a one-byte P atom without the CJK bit) */
static inline uint8_t toks_d3_atom(const toks_tables *t, const uint8_t *p, uint64_t avail, uint32_t *k)
{
    uint32_t cp;
    return toks_k3_atom(t, p, avail, &cp, k, 0u);
}

/* rationale: docs/notes/c-core.md §k3_dsv3_c.c.2 */
static uint64_t toks_d3_run(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t from, uint8_t want,
                            uint64_t *last, uint8_t *stop)
{
    uint64_t e = from;
    *last = from;
    *stop = 0xFFu;
    while (e < len) {                                   /* bound: len - e (>= 1 byte per atom) */
        uint32_t k;
        uint8_t c = toks_d3_atom(t, text + e, len - e, &k);
        if (c != want) {
            *stop = c;
            break;
        }
        *last = e;
        e += k;
    }
    return e;
}

uint64_t toks_k3_scan_dsv3_c(const toks_tables *t, toks_k3_args *a)
{
    const uint8_t *text = a->text;
    const uint8_t *asc = t->cls_ascii;
    uint64_t len = a->len, pos = a->pos, cap = a->cap;
    uint32_t *ends = a->ends;
    uint64_t n = 0;

    while (pos < len && n < cap) {                      /* bound: min(pieces, cap) */
        uint32_t k1, l1 = 0u;
        uint8_t c = toks_d3_atom(t, text + pos, len - pos, &k1);
        uint8_t b = (uint8_t)(c & TOKS_C_BASE_MASK);
        uint8_t r = (uint8_t)(c & TOKS_C_CJK);
        uint64_t p1 = pos + k1;                         /* start of c1 */
        uint64_t end, last;
        uint8_t c1 = 0xFFu, stop;                       /* 0xFF: no c1 (matches no kind) */

        if (b == TOKS_C_N) {
            end = toks_k3_n13(t, text, len, p1, 0u);    /* D0: up to 3 N atoms (N never carries the CJK bit) */
            goto piece;
        }
        if (b == TOKS_C_L) {
            /* D2: the maximal in-region L run from c */
            end = toks_d3_run(t, text, len, p1, (uint8_t)(TOKS_C_L | r), &last, &stop);
            goto piece;
        }
        if (b == TOKS_C_P) {
            /* D1: an ascii P atom, then ascii letters only */
            if (text[pos] < 0x80u && p1 < len && text[p1] < 0x80u && asc[text[p1]] == TOKS_C_L) {
                end = p1 + 1u;
                while (end < len && text[end] < 0x80u && asc[text[end]] == TOKS_C_L) {   /* bound: len - end */
                    end++;
                }
                goto piece;
            }
            /* D3: the maximal in-region P run from c, then the NL tail */
            end = toks_d3_run(t, text, len, p1, (uint8_t)(TOKS_C_P | r), &last, &stop);
            end = r != 0u ? end : toks_k3_run(t, text, len, end, TOKS_C_NL, 0u);   /* its NL tail (CR LF: never CJK) */
            goto piece;
        }
        if (p1 < len) {
            c1 = toks_d3_atom(t, text + p1, len - p1, &l1);
        }
        if (b == TOKS_C_WS || b == TOKS_C_NL) {
            if (b == TOKS_C_WS && c1 == TOKS_C_L) {
                /* D2: a WS prefix (WS is never CJK: r = 0) */
                end = toks_d3_run(t, text, len, p1 + l1, TOKS_C_L, &last, &stop);
                goto piece;
            }
            if (text[pos] == 0x20u && c1 == TOKS_C_P) {
                /* D3 with its optional space */
                end = toks_d3_run(t, text, len, p1 + l1, TOKS_C_P, &last, &stop);
                end = toks_k3_run(t, text, len, end, TOKS_C_NL, 0u);
                goto piece;
            }
            /* D4-D6 = A5-A7, and D5's region end: an N or CJK atom after the run (kernels.h toks_k3_ws) */
            end = toks_k3_ws(t, text, len, pos, 1, 0u, TOKS_TP_DIGIT_CUT | TOKS_C_CJK);
            goto piece;
        }
        /* X (and any base the tables never hold): D2 with an X prefix, else the D7 gap */
        if (c1 == (uint8_t)(TOKS_C_L | r)) {
            end = toks_d3_run(t, text, len, p1 + l1, (uint8_t)(TOKS_C_L | r), &last, &stop);
            goto piece;
        }
        end = toks_d3_run(t, text, len, p1, (uint8_t)(TOKS_C_X | r), &last, &stop);
        if (stop == (uint8_t)(TOKS_C_L | r) && last > pos) {
            end = last;                                 /* D7: the last X atom prefixes the letters after it */
        }

piece:
        if (end <= pos) { end = p1; }   /* safety: every rule advances >= 1 atom */
        ends[n] = (uint32_t)end;
        n++;
        pos = end;
    }
    a->n = n;
    if (n != 0u) { a->pos = ends[n - 1u]; }
    return n;
}
