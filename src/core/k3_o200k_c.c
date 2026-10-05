/* k3_o200k_c.c: K3's c twin, template o200k (docs/templates/o200k.md §2-3), docs/notes/c-core.md §k3_o200k_c.c.1 */
#include "kernels.h"

/* end of the maximal run of atoms whose class has the flag bit f (TOKS_C_LOWER, TOKS_C_HAN) from `from`. */
static uint64_t toks_o2_flag_run(const toks_tables *t, const uint8_t *text, uint64_t len,
                                 uint64_t from, uint8_t f, uint8_t strip)
{
    uint64_t e = from;
    while (e < len) {                                   /* bound: len - e (>= 1 byte per atom) */
        uint32_t cp, k;
        if ((toks_k3_atom(t, text + e, len - e, &cp, &k, strip) & f) == 0u) { break; }
        e += k;
    }
    return e;
}

/* K(e): the end of the contraction suffix at e, or e itself when there is none (o200k.md §3:
 * (?i:'s|'t|'re|'ve|'m|'ll|'d) -- the ascii cases plus U+017F, whose class carries TOKS_C_FOLD_S). */
static uint64_t toks_o2_contr(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t e,
                              uint32_t contr)
{
    if (contr == TOKS_TP_CONTR_NONE || e + 1u >= len || text[e] != 0x27u) { return e; }
    uint64_t p1 = e + 1u;                               /* ' is one byte */
    uint32_t cp1, l1;
    uint8_t c1 = toks_k3_atom(t, text + p1, len - p1, &cp1, &l1, 0u);
    if (cp1 == 0x73u || cp1 == 0x53u || cp1 == 0x74u || cp1 == 0x54u ||
        cp1 == 0x6Du || cp1 == 0x4Du || cp1 == 0x64u || cp1 == 0x44u || (c1 & TOKS_C_FOLD_S) != 0u) {
        return p1 + l1;                                 /* 's 'S 't 'T 'm 'M 'd 'D 'ſ */
    }
    uint64_t p2 = p1 + l1;
    if (p2 < len) {
        uint32_t cp2, l2;
        toks_k3_atom(t, text + p2, len - p2, &cp2, &l2, 0u);
        if (((cp1 == 0x72u || cp1 == 0x52u) && (cp2 == 0x65u || cp2 == 0x45u)) ||
            ((cp1 == 0x76u || cp1 == 0x56u) && (cp2 == 0x65u || cp2 == 0x45u)) ||
            ((cp1 == 0x6Cu || cp1 == 0x4Cu) && (cp2 == 0x6Cu || cp2 == 0x4Cu))) {
            return p2 + l2;                             /* 're 'RE ... 'll 'LL */
        }
    }
    return e;
}

/* rationale: docs/notes/c-core.md §k3_o200k_c.c.2 */
static uint64_t toks_o2_case(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t s, uint8_t cu,
                             uint32_t k, uint32_t contr, uint8_t strip, uint64_t *l2)
{
    uint64_t u = s, lastlo = 0;
    *l2 = 0;
    while ((cu & TOKS_C_UPPER) != 0u) {                 /* bound: len - u (>= 1 byte per atom) */
        uint32_t cp;
        if ((cu & TOKS_C_LOWER) != 0u) { lastlo = u + k; }
        u += k;
        cu = 0u;                                        /* the atom at u (0: none) */
        if (u < len) { cu = toks_k3_atom(t, text + u, len - u, &cp, &k, strip); }
    }
    if ((cu & TOKS_C_LOWER) != 0u) {                    /* L1: [UP]* then the LO run from u (the atom at u is LO) */
        return toks_o2_contr(t, text, len, toks_o2_flag_run(t, text, len, u + k, TOKS_C_LOWER, strip), contr);
    }
    if (lastlo != 0u) {                                 /* L1: [UP]* gives back to the last LO atom */
        return toks_o2_contr(t, text, len, lastlo, contr);
    }
    if (u > s) {                                        /* L2: [UP]+, the LO run from u is empty here */
        *l2 = toks_o2_contr(t, text, len, u, contr);
    }
    return 0;
}

uint64_t toks_k3_scan_o200k_c(const toks_tables *t, toks_k3_args *a)
{
    const uint8_t *text = a->text;
    uint64_t len = a->len, pos = a->pos, cap = a->cap;
    uint32_t *ends = a->ends;
    uint32_t contr = t->tmpl_params & TOKS_TP_CONTR_MASK;
    int digits1 = (t->tmpl_params & TOKS_TP_DIGITS_MASK) == TOKS_TP_DIGITS_1;
    int han = (t->tmpl_params & TOKS_TP_HAN) != 0u;
    uint8_t strip = han ? (uint8_t)(TOKS_C_UPPER | TOKS_C_LOWER) : 0u;
    uint8_t slash = (t->tmpl_params & TOKS_TP_NO_SLASH) != 0u ? 0x0Du : 0x2Fu;   /* '/' or a repeat of CR */
    uint64_t n = 0;

    while (pos < len && n < cap) {                      /* bound: min(pieces, cap) */
        uint32_t cp, k1;
        uint8_t bc = toks_k3_atom(t, text + pos, len - pos, &cp, &k1, strip);
        uint8_t bb = (uint8_t)(bc & TOKS_C_BASE_MASK);
        uint64_t p1 = pos + k1;                         /* start of the second atom */
        uint64_t end = pos;
        /* the second atom, decoded when c may be a prefix (P or WS): O1 and O3 consult it */
        uint32_t cp1 = 0, l1 = 0;
        uint8_t bc1 = 0, bb1 = 0xFFu;                   /* no second atom: matches no class */
        if (p1 < len && (bb == TOKS_C_P || bb == TOKS_C_WS)) {
            bc1 = toks_k3_atom(t, text + p1, len - p1, &cp1, &l1, strip);
            bb1 = (uint8_t)(bc1 & TOKS_C_BASE_MASK);
        }

        /* O0 Han (TOKS_TP_HAN): the maximal run of Han atoms, whatever their base classes. */
        if (han && (bc & TOKS_C_HAN) != 0u) {
            end = toks_o2_flag_run(t, text, len, p1, TOKS_C_HAN, strip);
            goto piece;
        }

        /* O1 letters. c a prefix: L1(i+1), L1(i), L2(i+1) in that order; c a letter: L1(i), L2(i). */
        if (bb == TOKS_C_P || bb == TOKS_C_WS) {
            uint64_t l2 = 0;
            if ((bc1 & (TOKS_C_UPPER | TOKS_C_LOWER)) != 0u) {     /* else L1(i+1), L2(i+1) fail */
                end = toks_o2_case(t, text, len, p1, bc1, l1, contr, strip, &l2);
                if (end != 0u) { goto piece; }
            }
            if ((bc & (TOKS_C_UPPER | TOKS_C_LOWER)) != 0u) {      /* c is a mark: L1(i) */
                uint64_t unused;
                end = toks_o2_case(t, text, len, pos, bc, k1, contr, strip, &unused);
                if (end != 0u) { goto piece; }
            }
            if (l2 != 0u) {
                end = l2;
                goto piece;
            }
            /* L2(i) never decides: a P/WS atom in UP is a mark, and L1(i) applied to it above */
        } else if (bb == TOKS_C_L) {
            uint64_t l2 = 0;
            end = toks_o2_case(t, text, len, pos, bc, k1, contr, strip, &l2);
            if (end == 0u) { end = l2; }                /* an L atom is UP or LO: one applies */
            goto piece;
        }

        /* O2 digits: \p{N}{1,3}, or \p{N} with DIGITS_1. */
        if (bb == TOKS_C_N) {
            end = digits1 ? p1 : toks_k3_n13(t, text, len, p1, strip);
            goto piece;
        }

        /* O3 punctuation: the maximal P run from c, or from c1 when c is U+0020 and c1 is P; then
         * the maximal run of CR, LF and '/' (ascii bytes are whole atoms). */
        if (bb == TOKS_C_P) {
            end = toks_k3_run(t, text, len, p1, TOKS_C_P, strip);
        } else if (cp == 0x20u && bb1 == TOKS_C_P) {
            end = toks_k3_run(t, text, len, p1 + l1, TOKS_C_P, strip);
        } else {
            goto ws;                                    /* c is WS or NL: O4/O5/O6 */
        }
        while (end < len && (text[end] == 0x0Du || text[end] == 0x0Au || text[end] == slash)) {
            end++;                                      /* bound: len - end */
        }
        goto piece;

ws:
        end = toks_k3_ws(t, text, len, pos, 1, strip, 0u);   /* O4/O5/O6 */

piece:
        /* unreachable while §3's rules hold (each ends past pos). It keeps the loop bound independent of them:
         * a rule bug shows up as one-atom pieces, which the hf / tiktoken differentials report, not a stall. */
        if (end <= pos) { end = p1; }
        ends[n] = (uint32_t)end;
        n++;
        pos = end;
    }
    a->n = n;
    if (n != 0u) { a->pos = ends[n - 1u]; }
    return n;
}
