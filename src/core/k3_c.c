/* k3_c.c: K3's c twin, template cl100k (kernels.md §2-3), docs/notes/c-core.md §k3_c.c.1 */
#include "kernels.h"

uint64_t toks_k3_scan_cl100k_c(const toks_tables *t, toks_k3_args *a)
{
    const uint8_t *text = a->text;
    uint64_t len = a->len, pos = a->pos, cap = a->cap;
    uint32_t *ends = a->ends;
    uint32_t params = t->tmpl_params;
    uint32_t contr = params & TOKS_TP_CONTR_MASK;
    int lprefix_any = (params & TOKS_TP_LPREFIX_ANY) != 0;
    uint32_t digits = params & TOKS_TP_DIGITS_MASK;
    int punct_nl = (params & TOKS_TP_PUNCT_NL) != 0;
    int ws_nl = (params & TOKS_TP_WS_NL) != 0;
    int nl_cut = (params & TOKS_TP_NL_CUT) != 0;
    uint64_t n = 0;

    while (pos < len && n < cap) {                      /* bound: min(pieces, cap) */
        uint32_t cp, k1;
        uint8_t bc = toks_k3_atom(t, text + pos, len - pos, &cp, &k1, 0u);
        uint8_t bb = (uint8_t)(bc & TOKS_C_BASE_MASK);
        uint64_t p1 = pos + k1;                /* start of the second atom */
        uint64_t end = pos;
        /* rationale: docs/notes/c-core.md §k3_c.c.2 */
        uint32_t cp1 = 0, l1 = 0;
        uint8_t bc1 = 0, bb1 = 0xFFu;          /* no second atom: matches no base class */
        if (p1 < len && ((contr != TOKS_TP_CONTR_NONE && cp == 0x27u) ||
                         (lprefix_any && (bb == TOKS_C_P || bb == TOKS_C_WS)) ||
                         cp == 0x20u)) {
            bc1 = toks_k3_atom(t, text + p1, len - p1, &cp1, &l1, 0u);
            bb1 = (uint8_t)(bc1 & TOKS_C_BASE_MASK);
        }

        /* A1 contractions: c is U+0027 and the next one or two atoms match (§3). */
        if (contr != TOKS_TP_CONTR_NONE && cp == 0x27u && p1 < len) {
            if (contr == TOKS_TP_CONTR_CS) {
                if (cp1 == 0x73u || cp1 == 0x74u || cp1 == 0x6Du || cp1 == 0x64u) {
                    end = p1 + l1;                      /* 's 't 'm 'd */
                    goto piece;
                }
                if (p1 + l1 < len) {
                    uint32_t cp2, l2;
                    toks_k3_atom(t, text + p1 + l1, len - p1 - l1, &cp2, &l2, 0u);
                    if ((cp1 == 0x72u && cp2 == 0x65u) || (cp1 == 0x76u && cp2 == 0x65u) ||
                        (cp1 == 0x6Cu && cp2 == 0x6Cu)) {
                        end = p1 + l1 + l2;             /* 're 've 'll */
                        goto piece;
                    }
                }
            } else {
                /* CI: the ascii cases plus U+017F, the only non-ascii (?i) fold onto a contraction
                 * letter (docs/unicode.md); its class carries TOKS_C_FOLD_S. */
                if (cp1 == 0x73u || cp1 == 0x53u || cp1 == 0x74u || cp1 == 0x54u ||
                    cp1 == 0x6Du || cp1 == 0x4Du || cp1 == 0x64u || cp1 == 0x44u ||
                    (bc1 & TOKS_C_FOLD_S) != 0u) {
                    end = p1 + l1;                      /* 's 'S 't 'T 'm 'M 'd 'D 'ſ */
                    goto piece;
                }
                if (p1 + l1 < len) {
                    uint32_t cp2, l2;
                    toks_k3_atom(t, text + p1 + l1, len - p1 - l1, &cp2, &l2, 0u);
                    if (((cp1 == 0x72u || cp1 == 0x52u) && (cp2 == 0x65u || cp2 == 0x45u)) ||
                        ((cp1 == 0x76u || cp1 == 0x56u) && (cp2 == 0x65u || cp2 == 0x45u)) ||
                        ((cp1 == 0x6Cu || cp1 == 0x4Cu) && (cp2 == 0x6Cu || cp2 == 0x4Cu))) {
                        end = p1 + l1 + l2;             /* 're 'RE ... 'll 'LL */
                        goto piece;
                    }
                }
            }
        }

        /* A2 letters: c is L, or c is an admitted prefix and c1 is L. The prefix class is exactly
         * [^\r\n\p{L}\p{N}] = P ∪ WS (LPREFIX_ANY), or U+0020 only (gpt-2). */
        if (bb == TOKS_C_L) {
            end = toks_k3_run(t, text, len, p1, TOKS_C_L, 0u);
            goto piece;
        }
        if (bb1 == TOKS_C_L &&
            (lprefix_any ? (bb == TOKS_C_P || bb == TOKS_C_WS) : cp == 0x20u)) {
            end = toks_k3_run(t, text, len, p1 + l1, TOKS_C_L, 0u);
            goto piece;
        }

        /* A3 digits. */
        if (bb == TOKS_C_N) {
            if (digits == TOKS_TP_DIGITS_1_3) {
                end = toks_k3_n13(t, text, len, p1, 0u);
            } else if (digits == TOKS_TP_DIGITS_1) {
                end = p1;                                /* \p{N} */
            } else {                                     /* RUN and SP_RUN: \p{N}+ */
                end = toks_k3_run(t, text, len, p1, TOKS_C_N, 0u);
            }
            goto piece;
        }
        if (digits == TOKS_TP_DIGITS_SP_RUN && cp == 0x20u && bb1 == TOKS_C_N) {
            end = toks_k3_run(t, text, len, p1 + l1, TOKS_C_N, 0u);   /* ' ?\p{N}+ */
            goto piece;
        }

        /* A4 punctuation: the maximal P run from c, or from c1 when c is U+0020 and c1 is P;
         * with PUNCT_NL the piece then extends over the NL run that follows. */
        if (bb == TOKS_C_P) {
            end = toks_k3_run(t, text, len, p1, TOKS_C_P, 0u);
        } else if (cp == 0x20u && bb1 == TOKS_C_P) {
            end = toks_k3_run(t, text, len, p1 + l1, TOKS_C_P, 0u);
        } else {
            goto ws;                                     /* c is WS or NL: A5/A6/A7 */
        }
        if (punct_nl) {
            uint64_t q = end;
            end = toks_k3_run(t, text, len, end, TOKS_C_NL, 0u);
            while (nl_cut && q < end && !toks_k3_nlcut(text, len, q)) { q++; }   /* bound: the tail (A9 ends it) */
            end = nl_cut ? q : end;
        }
        goto piece;

ws:
        end = toks_k3_ws(t, text, len, pos, ws_nl, 0u, params);   /* A5/A6/A7, A8-A10 */

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
