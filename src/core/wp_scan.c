/* wp_scan.c: BertNormalizer + BertPreTokenizer in one pass over the caller's (docs/notes/c-core.md §wp_scan.c.1) */
#include "wp.h"

/* scanner class of one atom under the flags */
enum { SC_WORD = 0, SC_SPLIT, SC_PUNCT, SC_HOLE, SC_HOLE_MN, SC_CJK, SC_CHANGE, SC_INVALID };

uint8_t toks_wp_ascii_class(uint32_t f, uint32_t b);

uint8_t toks_wp_ascii_class(uint32_t f, uint32_t b)
{
    if (b == 0x09u || b == 0x0Au || b == 0x0Du || b == 0x20u) { return TOKS_WPA_SPLIT; }
    if (b == 0x0Bu || b == 0x0Cu) {                     /* White_Space AND Cc: clean_text removes them first */
        return (f & TOKS_WPF_CLEAN) ? TOKS_WPA_REMOVE : TOKS_WPA_SPLIT;
    }
    if (b < 0x20u || b == 0x7Fu) { return (f & TOKS_WPF_CLEAN) ? TOKS_WPA_REMOVE : TOKS_WPA_WORD; }
    if ((b >= 0x21u && b <= 0x2Fu) || (b >= 0x3Au && b <= 0x40u) || (b >= 0x5Bu && b <= 0x60u) ||
        (b >= 0x7Bu && b <= 0x7Eu)) {
        return TOKS_WPA_PUNCT;
    }
    if (b >= 0x41u && b <= 0x5Au && (f & TOKS_WPF_LOWER)) { return TOKS_WPA_FOLD; }
    return TOKS_WPA_WORD;
}

static uint32_t wp_class(uint32_t f, uint8_t cls)
{
    if ((f & TOKS_WPF_CLEAN) && (cls & TOKS_BC_REMOVE)) { return SC_HOLE; }
    if (cls & TOKS_BC_WS) { return SC_SPLIT; }
    if ((f & TOKS_WPF_CHINESE) && (cls & TOKS_BC_CJK)) { return SC_CJK; }
    /* rationale: docs/notes/c-core.md §wp_scan.c.2 */
    if ((f & TOKS_WPF_STRIP) && (cls & TOKS_BC_MN) && !(cls & TOKS_BC_DECOMP)) { return SC_HOLE_MN; }
    if (((f & TOKS_WPF_STRIP) && (cls & (TOKS_BC_DECOMP | TOKS_BC_NS))) ||
        ((f & TOKS_WPF_LOWER) && (cls & TOKS_BC_LOWER))) {
        return SC_CHANGE;
    }
    if (cls & TOKS_BC_PUNCT) { return SC_PUNCT; }
    return SC_WORD;
}

uint64_t toks_wp_mat_min(const toks_wp_tables *t)
{
    return 8u * ((uint64_t)t->max_chars + 8u) + 64u;    /* a piece's bytes + its run area (wp.h) */
}

/* ---- scan state ----------------------------------------------------------------------------------------- */
typedef struct wps {
    const toks_wp_tables *t;
    toks_wp_scan_args *a;
    uint64_t piece_cap;     /* mat[0, piece_cap): pieces; mat[piece_cap, mat_cap): the run area (u32 cps) */
    uint64_t run_cap;       /* cps the run area holds */
    uint64_t need;          /* mat bytes one materialized piece may take */
    int open, mat, hole, over;
    uint64_t start, end;    /* raw span of an in-place piece */
    uint64_t moff;          /* mat offset of a materialized piece */
    uint64_t nch;           /* chars of the normalized piece */
    uint64_t run_n;         /* pending kept non-starters (mat mode) */
} wps;

static void emit(wps *s, uint32_t off, uint64_t len, uint64_t end, uint32_t flags)
{
    toks_wp_piece *p = &s->a->pieces[s->a->n++];
    p->off = off;
    p->len = (uint32_t)len;
    p->end = (uint32_t)end;
    p->flags = flags;
}

/* a one-char piece (punctuation, a chinese char, an invalid byte): [unk] when max_input_chars_per_word is 0 */
static void emit1(wps *s, uint32_t off, uint64_t len, uint64_t end, uint32_t flags)
{
    emit(s, off, len, end, flags | (s->t->max_chars == 0u ? TOKS_WPP_OVER : 0u));
}

static void mat_cp(wps *s, uint32_t cp)
{
    s->a->mat_len += toks_utf8_put(s->a->mat + s->a->mat_len, cp);
}

static uint32_t run_get(const wps *s, uint64_t k)
{
    uint32_t v;
    memcpy(&v, s->a->mat + s->piece_cap + 4u * k, 4);
    return v;
}

/* the pending run, stable-sorted by combining class, into the piece (wordpiece.md §3.2 N3) */
static void run_flush(wps *s)
{
    uint32_t prev = 0;
    for (uint32_t pass = 0; pass < 256u && s->run_n > 0u; pass++) {   /* bound: 255 classes */
        uint32_t next = 256;
        for (uint64_t k = 0; k < s->run_n; k++) {      /* bound: run_cap */
            uint32_t c = run_get(s, k) >> 24;
            if (c > prev && c < next) { next = c; }
        }
        if (next == 256u) { break; }
        for (uint64_t k = 0; k < s->run_n; k++) {      /* bound: run_cap */
            uint32_t v = run_get(s, k);
            if ((v >> 24) == next) { mat_cp(s, v & 0xFFFFFFu); }
        }
        prev = next;
    }
    s->run_n = 0;
}

/* closes the open piece; end = the raw position after its last char */
static void close_piece(wps *s, uint64_t end)
{
    if (!s->open) { return; }
    uint32_t fl = s->over ? TOKS_WPP_OVER : 0u;
    if (s->mat) {
        run_flush(s);
        emit(s, (uint32_t)s->moff, s->a->mat_len - s->moff, end, fl | TOKS_WPP_MAT);
    } else {
        emit(s, (uint32_t)s->start, s->end - s->start, end, fl);
    }
    s->open = 0;
}

/* switches the open piece to its copy; 0 when the mat buffer is too full (the caller ends the chunk) */
static int to_mat(wps *s)
{
    if (s->mat) { return 1; }
    if (s->piece_cap < s->a->mat_len || s->piece_cap - s->a->mat_len < s->need) { return 0; }
    s->moff = s->a->mat_len;
    memcpy(s->a->mat + s->a->mat_len, s->a->text + s->start, s->end - s->start);   /* <= 4 * max_chars bytes */
    s->a->mat_len += s->end - s->start;
    s->mat = 1;
    s->hole = 0;
    return 1;
}

/* one more normalized char in the open piece (opening it at raw [i, i + l) when none is) */
static void count_char(wps *s)
{
    s->nch++;
    if (s->nch > (uint64_t)s->t->max_chars) { s->over = 1; }
}

/* an unchanged char at raw [i, i + l); 0 when the chunk must end before the open piece */
static int add_raw(wps *s, uint64_t i, uint64_t l)
{
    if (!s->open) {
        s->open = 1; s->mat = 0; s->hole = 0; s->over = 0; s->run_n = 0;
        s->start = i; s->end = i + l; s->nch = 0;
        count_char(s);
        return 1;
    }
    count_char(s);
    if (s->over) { s->end = i + l; return 1; }          /* [unk] whatever the content: nothing to copy */
    if (!s->mat && !s->hole) { s->end = i + l; return 1; }
    if (!to_mat(s)) { return 0; }
    if (s->run_n) { run_flush(s); }                     /* an unchanged char is a starter (NS chars change) */
    memcpy(s->a->mat + s->a->mat_len, s->a->text + i, l);
    s->a->mat_len += l;
    return 1;
}

uint64_t toks_wp_scan_c(const toks_wp_tables *t, toks_wp_scan_args *a)
{
    uint32_t f = t->flags & (uint32_t)a->flags;
    uint8_t own[128];
    const uint8_t *acls = f == t->flags ? t->ascii_cls : f == 0u ? t->ascii_cls0 : own;   /* built at load */
    if (acls == own) {
        for (uint32_t b = 0; b < 128u; b++) { own[b] = toks_wp_ascii_class(f, b); }   /* bound: 128 */
    }
    wps s;
    memset(&s, 0, sizeof s);
    s.t = t;
    s.a = a;
    s.need = 4u * ((uint64_t)t->max_chars + 8u);
    s.piece_cap = a->mat_cap > s.need ? a->mat_cap - s.need : 0u;
    s.run_cap = s.need / 4u;
    a->n = 0;
    const uint8_t *text = a->text;
    uint64_t len = a->len;
    uint64_t i = a->pos;
    uint32_t o[TOKS_NORM_MAX_OUT];
    while (i < len) {                                   /* bound: len - pos (every step consumes >= 1 byte) */
        if (a->n + 2u > a->cap) {                       /* a char emits at most 2 pieces (wordpiece.md §12.2) */
            if (s.open && s.mat) { a->mat_len = s.moff; }
            a->pos = s.open ? s.start : i;
            return a->n;
        }
        uint32_t b = text[i];
        if (b < 0x80u && acls[b] <= TOKS_WPA_FOLD) {     /* an ascii word run: add_raw's steps at once */
            uint64_t j = i + 1u;
            while (j < len && text[j] < 0x80u && acls[text[j]] <= TOKS_WPA_FOLD) { j++; }   /* bound: len - i */
            if (!s.open) {
                s.open = 1; s.mat = 0; s.hole = 0; s.over = 0; s.run_n = 0;
                s.start = i; s.end = i; s.nch = 0;
            }
            if (s.over || (!s.mat && !s.hole)) {        /* in place (or [unk] already): only the span and the count */
                s.nch += j - i;
                if (s.nch > (uint64_t)t->max_chars) { s.over = 1; }
                s.end = j;
                i = j;
                continue;
            }
            for (; i < j; i++) {                        /* bound: j - i (the copy path) */
                if (!add_raw(&s, i, 1u)) { a->pos = s.start; return a->n; }
            }
            continue;
        }
        uint32_t cp = b;
        uint64_t l = 1;
        uint32_t c;
        uint8_t cls = 0;
        if (b < 0x80u) {
            uint8_t ac = acls[b];
            c = ac == TOKS_WPA_SPLIT ? SC_SPLIT : ac == TOKS_WPA_PUNCT ? SC_PUNCT
              : ac == TOKS_WPA_REMOVE ? SC_HOLE : SC_WORD;
        } else {
            l = wp_utf8(text + i, len - i, &cp);
            if (cp == TOKS_WP_INVALID_CP) {
                c = SC_INVALID;
            } else {
                cls = toks_bert_cls(cp);
                c = wp_class(f, cls);
            }
        }
        switch (c) {
        case SC_WORD:
            if (!add_raw(&s, i, l)) { a->pos = s.start; return a->n; }
            break;
        case SC_SPLIT:
            close_piece(&s, i);
            break;
        case SC_HOLE:                                   /* clean_text: gone before N3; a run continues */
            if (s.open && !s.mat) { s.hole = 1; }
            break;
        case SC_HOLE_MN:                                /* strip_accents drops it; a ccc-0 mark ends the run */
            if (s.open) {
                if (s.mat) {
                    if ((cls & TOKS_BC_NS) == 0u && s.run_n) { run_flush(&s); }
                } else {
                    s.hole = 1;
                }
            }
            break;
        case SC_PUNCT:
        case SC_INVALID:
            close_piece(&s, i);
            emit1(&s, (uint32_t)i, l, i + l, c == SC_INVALID ? TOKS_WPP_INVALID : 0u);
            break;
        case SC_CJK: {                                  /* a piece of its own, in place unless N3 rewrites it */
            close_piece(&s, i);
            uint32_t n = toks_norm_char(f, cp, o);
            if (n == 1u && o[0] == cp) {
                emit1(&s, (uint32_t)i, l, i + l, 0u);
                break;
            }
            uint64_t m0 = a->mat_len;
            if (s.piece_cap < m0 || s.piece_cap - m0 < 4u * TOKS_NORM_MAX_OUT) { a->pos = i; return a->n; }
            for (uint32_t k = 0; k < n; k++) {          /* bound: TOKS_NORM_MAX_OUT (all starters) */
                if (o[k] != TOKS_NORM_GHOST) { mat_cp(&s, o[k]); }
            }
            if (a->mat_len > m0) { emit1(&s, (uint32_t)m0, a->mat_len - m0, i + l, TOKS_WPP_MAT); }
            break;
        }
        case SC_CHANGE: {
            uint32_t n = toks_norm_char(f, cp, o);
            int cut = 0, added = 0;
            for (uint32_t k = 0; k < n; k++) {          /* bound: TOKS_NORM_MAX_OUT */
                if (o[k] == TOKS_NORM_GHOST) {
                    if (s.open && s.mat && s.run_n) { run_flush(&s); }
                    continue;
                }
                uint8_t xc = toks_bert_cls(o[k] & 0xFFFFFFu);
                if ((xc & (TOKS_BC_PUNCT | TOKS_BC_WS)) || ((f & TOKS_WPF_CHINESE) && (xc & TOKS_BC_CJK))) {
                    /* the char's only item (asserted by tools/gen/bert_tables.py: a boundary item never shares
                       its char with another): the piece before ends at i, the item is a piece of its own */
                    close_piece(&s, i);
                    if (!(xc & TOKS_BC_WS)) {
                        uint64_t m0 = a->mat_len;
                        if (s.piece_cap < m0 || s.piece_cap - m0 < 4u) { a->pos = i; return a->n; }
                        mat_cp(&s, o[k]);
                        emit1(&s, (uint32_t)m0, a->mat_len - m0, i + l, TOKS_WPP_MAT);
                    }
                    cut = 1;
                    continue;
                }
                if (!s.open) {                          /* a piece starting with a rewritten char */
                    s.open = 1; s.mat = 0; s.hole = 0; s.over = 0; s.run_n = 0;
                    s.start = i; s.end = i; s.nch = 0;
                }
                count_char(&s);
                added = 1;
                if (s.over) { continue; }
                if (!to_mat(&s)) { a->pos = s.start; return a->n; }
                if ((o[k] >> 24) == 0u) {                /* a starter (norm.h: cls << 24 | char) */
                    if (s.run_n) { run_flush(&s); }
                    mat_cp(&s, o[k]);
                } else if (s.run_n < s.run_cap) {
                    memcpy(a->mat + s.piece_cap + 4u * s.run_n, &o[k], 4);
                    s.run_n++;
                }
            }
            if (s.open && !cut) {
                if (s.mat || s.over) {
                    s.end = i + l;                      /* a raw position only: the copy holds the content */
                } else if (!added) {
                    s.hole = 1;                         /* the char normalized to nothing inside the word */
                }
            }
            break;
        }
        default:
            break;
        }
        i += l;
    }
    close_piece(&s, len);
    a->pos = len;
    return a->n;
}
