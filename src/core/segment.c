/* segment.c: the driver's added-token policy (kernels.md §4, §7 step 2) (docs/notes/c-core.md §segment.c.1) */
#include "kernels.h"
#include "../gen/rx_word.h"

/* rationale: docs/notes/c-core.md §segment.c.2 */
static const uint32_t WS_SET[25] = {
    0x0009u, 0x000Au, 0x000Bu, 0x000Cu, 0x000Du, 0x0020u, 0x0085u, 0x00A0u, 0x1680u,
    0x2000u, 0x2001u, 0x2002u, 0x2003u, 0x2004u, 0x2005u, 0x2006u, 0x2007u, 0x2008u,
    0x2009u, 0x200Au, 0x2028u, 0x2029u, 0x202Fu, 0x205Fu, 0x3000u
};

int toks_is_regex_ws(uint32_t cp)
{
    uint32_t lo = 0u, hi = 25u;                    /* bound: ceil(log2(26)) = 5 steps */
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (WS_SET[mid] < cp) { lo = mid + 1u; }
        else { hi = mid; }
    }
    return (lo < 25u) && (WS_SET[lo] == cp);
}

/* the regex-crate \w (hf's single_word): src/gen/rx_word.c, sorted disjoint ranges; binary search. */
int toks_is_regex_word(uint32_t cp)
{
    uint32_t lo = 0u, hi = TOKS_RX_WORD_N;          /* bound: ceil(log2(TOKS_RX_WORD_N + 1)) = 10 steps */
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (toks_rx_word[mid][1] < cp) { lo = mid + 1u; }
        else { hi = mid; }
    }
    return (lo < TOKS_RX_WORD_N) && (toks_rx_word[lo][0] <= cp);
}

#define SEG_NOT_CP 0xFFFFFFFFu                       /* an ill-formed byte: neither \s nor \w */

/* the atom at p (SPEC §3.3): a well-formed utf-8 char (its code point and length), or the single
 * ill-formed byte p[0] (SEG_NOT_CP, length 1). avail >= 1. */
static uint32_t atom_at(const uint8_t *p, uint64_t avail, uint32_t *len)
{
    uint32_t k = toks_utf8_len(p, avail);
    if (k == 0u) { *len = 1u; return SEG_NOT_CP; }
    *len = k;
    return (k == 1u) ? (uint32_t)p[0] : toks_cp_decode(p, k);
}

/* rationale: docs/notes/c-core.md §segment.c.3 */
static uint32_t atom_before(const uint8_t *text, uint64_t end, uint32_t *len)
{
    uint64_t k = end - 1u;
    uint32_t back = 0u;
    while (k > 0u && (text[k] & 0xC0u) == 0x80u && back < 3u) {  /* bound: 3 */
        k--; back++;
    }
    uint32_t l = 0u;
    uint32_t cp = atom_at(text + k, end - k, &l);
    if (cp != SEG_NOT_CP && (uint64_t)l == end - k) { *len = l; return cp; }
    *len = 1u;
    return SEG_NOT_CP;
}

/* start of the maximal \s run ending at end (exclusive), or any j <= floor once the run reaches floor (the caller
 * clamps to it: walking further would only cost time, quadratic over a flood of whitespace tokens); returns end
 * when there is none. */
static uint64_t ws_run_before(const uint8_t *text, uint64_t end, uint64_t floor)
{
    uint64_t j = end;
    while (j > floor) {                              /* bound: end - floor (j strictly decreases) */
        uint32_t l = 0u;
        uint32_t cp = atom_before(text, j, &l);
        if (cp == SEG_NOT_CP || !toks_is_regex_ws(cp)) { break; }
        j -= l;
    }
    return j;
}

/* end of the maximal \s run starting at start; returns start when there is none. */
static uint64_t ws_run_after(const uint8_t *text, uint64_t start, uint64_t len)
{
    uint64_t i = start;
    while (i < len) {                                /* bound: len - start (i advances) */
        uint32_t l = 0u;
        uint32_t cp = atom_at(text + i, len - i, &l);
        if (cp == SEG_NOT_CP || !toks_is_regex_ws(cp)) { break; }
        i += l;
    }
    return i;
}

/* hf's single_word test on the raw match [s, e) of text[0, len): not preceded and not followed by a
 * \w char (start == 0 / stop == len count as space). */
static int single_word_ok(const uint8_t *text, uint64_t s, uint64_t e, uint64_t len)
{
    uint32_t l = 0u;
    if (s > 0u) {
        uint32_t cp = atom_before(text, s, &l);
        if (cp != SEG_NOT_CP && toks_is_regex_word(cp)) { return 0; }
    }
    if (e < len) {
        uint32_t cp = atom_at(text + e, len - e, &l);
        if (cp != SEG_NOT_CP && toks_is_regex_word(cp)) { return 0; }
    }
    return 1;
}

void toks_seg_begin(toks_seg_iter *it, const toks_tables *t, uint32_t mode, uint32_t phase,
                    uint32_t tier, const uint8_t *text, uint64_t len)
{
    it->t = t;
    it->mode = mode;
    it->phase = phase;
    it->tier = tier;
    it->text = text;
    it->len = len;
    it->prev_end = 0;        /* hf's start_offset: the end of the last emitted unit */
    it->pos = 0;             /* the raw resume point (daachorse find_iter's cursor) */
    it->fin = 0;             /* 1 once the final gap has been emitted */
    it->pend = 0;            /* a token whose gap was just emitted is pending */
    it->pend_id = 0u;
    it->pend_start = 0;
    it->pend_end = 0;
    it->mi = it->mn = 0;
    it->rs_from = it->rs_to = UINT64_MAX;   /* no rstrip run walked yet */
}

int toks_seg_next(toks_seg_iter *it, toks_seg_out *u)
{
    if (it->fin && !it->pend) { return 0; }

    /* a token whose gap was the previous unit: emit it now (step 5's order) */
    if (it->pend) {
        it->pend = 0;
        u->kind = TOKS_SEG_TOKEN;
        u->id = it->pend_id;
        u->start = it->pend_start;
        u->end = it->pend_end;
        it->prev_end = it->pend_end;
        return 1;
    }

    const toks_tables *t = it->t;
    const uint8_t *text = it->text;
    uint64_t len = it->len;

    /* match loop over K1's raw matches (find_iter: each from the last one's end, steps 1 and 5, whatever this
     * policy keeps), taken from a batch; dropped matches are skipped in place (no rescan, no recursion) */
    /* bound: matches do not overlap (m_end >= m_start + 1), so at most len over the cursor's life; the loop
     * body only continues on a dropped match */
    while (!it->fin && (t->add_phases & (1ull << it->phase)) != 0u) {
        if (it->mi == it->mn) {
            if (it->pos >= len) { break; }
            toks_k1_args a = { text, len, it->pos, it->phase, it->m, TOKS_SEG_BATCH, 0u, 0u };
            it->mn = (uint32_t)toks_k1(t, &a, it->tier);
            it->mi = 0;
            it->pos = a.next;
            if (it->mn == 0u) { break; }             /* no match: the final gap follows */
        }
        const toks_k1_match *m = &it->m[it->mi++];
        uint64_t m_start = m->start, m_end = m->end;
        const toks_added_entry *e = &t->add_entries[m->entry];

        if (it->mode == TOKS_ADDED_NONSPECIAL && (e->flags & TOKS_AF_SPECIAL) != 0u) {
            continue;                                /* 1. dropped, not rescanned */
        }
        if ((e->flags & TOKS_AF_SINGLE_WORD) != 0u && !single_word_ok(text, m_start, m_end, len)) {
            continue;                                /* 2. dropped, not rescanned */
        }
        if ((e->flags & TOKS_AF_PFX) != 0u) {        /* rationale: docs/notes/c-core.md §segment.c.5 */
            uint64_t k = m_start == 0u ? 0u : text[m_start - 1u] == 0x20u ? 1u : m_start >= 3u &&
                         toks_meta_at(text, len, m_start - 3u) ? 3u : UINT64_MAX;
            if (k == UINT64_MAX) { continue; }       /* no ▁ before it in the normalized gap: dropped */
            m_start -= k;                            /* the ' ' / ▁ is the token's (the gap's ▁ at 0: none) */
        }

        uint64_t start = m_start, end = m_end;
        if ((e->flags & TOKS_AF_LSTRIP) != 0u) {     /* 3. left, clamped to prev_end */
            uint64_t ns = ws_run_before(text, m_start, it->prev_end);
            if (ns < it->prev_end) { ns = it->prev_end; }
            start = ns;
        }
        if ((e->flags & TOKS_AF_RSTRIP) != 0u) {      /* 4. right */
            /* the run walked last time, [rs_from, rs_to), is whole \s atoms, so any non-continuation byte in it starts
             * one and the walk from there ends at rs_to too: one walk per run, not one per match inside it */
            if (m_end >= it->rs_from && m_end <= it->rs_to && (m_end == it->rs_to || (text[m_end] & 0xC0u) != 0x80u)) {
                end = it->rs_to;
            } else {
                end = ws_run_after(text, m_end, len);
                it->rs_from = m_end;
                it->rs_to = end;
            }
        }
        /* rationale: docs/notes/c-core.md §segment.c.4 */
        if (start >= end) { continue; }
        /* 5. the gap first (when non-empty), then the token */
        if (it->prev_end < start) {
            u->kind = TOKS_SEG_GAP;
            u->id = 0u;
            u->start = it->prev_end;
            u->end = start;
            it->pend = 1;                            /* the token is next */
            it->pend_id = e->id;
            it->pend_start = start;
            it->pend_end = end;
            return 1;
        }
        u->kind = TOKS_SEG_TOKEN;
        u->id = e->id;
        u->start = start;
        u->end = end;
        it->prev_end = end;
        return 1;
    }

    /* no more matches: the final gap, once, then done */
    it->fin = 1;
    if (it->prev_end < len) {
        u->kind = TOKS_SEG_GAP;
        u->id = 0u;
        u->start = it->prev_end;
        u->end = len;
        it->prev_end = len;
        return 1;
    }
    return 0;
}
