/* gen.c: the generic pre-tokenizer, every byte-level chain no template compiles (docs/algorithms/generic.md) */
#include "core.h"
#include "config.h"
#include "../gen/ucd_flags.h"

enum { G_CHAR, G_SPLIT, G_JMP, G_ASSERT, G_LOOK, G_MATCH };          /* generic.md §2; x, y relative to the pc */
enum { A_BOL, A_EOL, A_BOS, A_EOS, A_EOSNL, A_WB, A_NWB };           /* ^ $ \A \z \Z \b \B */
enum { GS_RX = 8, GS_PRED = 9 };                                     /* a regex step, a char-predicate step */
#define G_MAX_STEP 512u                  /* code per step: the VM's lists (toks_gen_scr) */
#define G_INF      0xFFFFFFFFu
#define RX_INS     4096u
#define RX_CLS     512u
#define RX_RNG     8192u
#define MISS(what) ("pre_tokenizer Split regex: " what " (generic engine)")

typedef struct gins { uint8_t op, a; uint16_t rsv; int32_t x, y; } gins;         /* G_CHAR: x = class */
typedef struct gcls { uint32_t ascii[4], r0, nr; uint16_t mask, neg; } gcls;     /* ascii bitmap after neg */
typedef struct gstep { uint8_t kind, beh, inv, bytes; uint32_t pmask, pc0, len; } gstep;
struct toks_gen { uint32_t n_steps, n_ins, n_cls, n_rng, max_ins, rsv; gstep step[16]; };   /* then code, classes, ranges */
/* the compiler's work: its own block for the call (a small file's parse arena never holds it), the program alone
 * goes to the arena, at most toks_gen_max_bytes (config.h's bound counts it) */
typedef struct gwork { gins ins[RX_INS]; gcls cls[RX_CLS]; uint32_t rng[2u * RX_RNG]; uint32_t stk[2u * G_MAX_STEP + 2u];
                       uint8_t seen[G_MAX_STEP + 8u]; } gwork;
#define G_INS(g) ((const gins *)(const void *)((g) + 1))
#define G_CLS(g) ((const gcls *)(const void *)(G_INS(g) + (g)->n_ins))
#define G_RNG(g) ((const uint32_t *)(const void *)(G_CLS(g) + (g)->n_cls))

/* ---- the compiler: hf's regex strings (oniguruma's syntax), generic.md §3's constructs, else a named miss ---- */

typedef struct rx {
    const uint8_t *s;
    uint32_t n, i, ni, nc, nr, icase, inlook, depth;
    gins *ins;
    gcls *cls;
    uint32_t *rng;
    const char *bad;
} rx;

static int rx_fail(rx *r, const char *why) { if (r->bad == NULL) { r->bad = why; } return -1; }
static uint32_t peek(const rx *r) { return r->i < r->n ? r->s[r->i] : 0u; }

static uint32_t next_cp(rx *r)                       /* json strings are utf-8 */
{
    uint32_t k = toks_utf8_len(r->s + r->i, r->n - r->i), c = (k <= 1u) ? r->s[r->i] : toks_cp_decode(r->s + r->i, k);
    r->i += (k != 0u) ? k : 1u;
    return c;
}

static int emit(rx *r, uint32_t op, uint32_t a, int32_t x, int32_t y)
{
    if (r->ni >= RX_INS) { return rx_fail(r, MISS("too large")); }
    r->ins[r->ni++] = (gins){ (uint8_t)op, (uint8_t)a, 0u, x, y };
    return 0;
}

/* one instruction at s; the code from s moves down by one (relative targets inside it stay right) */
static int insert(rx *r, uint32_t s, uint32_t op, int32_t x, int32_t y)
{
    if (emit(r, 0u, 0u, 0, 0) != 0) { return -1; }
    for (uint32_t j = r->ni - 1u; j > s; j--) { r->ins[j] = r->ins[j - 1u]; }   /* bound: the code */
    r->ins[s] = (gins){ (uint8_t)op, 0u, 0u, x, y };
    return 0;
}

static int add_range(rx *r, uint32_t lo, uint32_t hi)
{
    if (r->nr >= RX_RNG) { return rx_fail(r, MISS("too many class ranges")); }
    r->rng[2u * r->nr] = lo;
    r->rng[2u * r->nr++ + 1u] = hi;
    return 0;
}

/* a class escape (\s \d \w \p{..}; upper case negated: *mask, *neg) -> 1; a literal escape (*lit) -> 0 */
static int esc(rx *r, uint32_t c, uint16_t *mask, uint16_t *neg, uint32_t *lit)
{
    static const char *const PN[11] = { "L", "Lu", "Ll", "Lt", "Lm", "Lo", "M", "N", "P", "S", "Nd" };
    static const uint16_t PM[11] = { TOKS_UCD_LETTERS, TOKS_UCD_LU, TOKS_UCD_LL, TOKS_UCD_LT, TOKS_UCD_LM, TOKS_UCD_LO,
                                     TOKS_UCD_M, TOKS_UCD_N, TOKS_UCD_P, TOKS_UCD_S, TOKS_UCD_ND };
    static const char CTL[] = "r\rn\nt\tf\fv\va\ae\x1B";
    uint32_t u = c | 0x20u, e = r->i + 1u;
    *neg = (uint16_t)(c >= 'A' && c <= 'Z');
    if (u == 's' || u == 'd' || u == 'w') {
        *mask = u == 's' ? TOKS_UCD_WS : u == 'd' ? TOKS_UCD_ND : TOKS_UCD_WORD;
        return 1;
    }
    for (uint32_t k = 0; CTL[k] != 0; k += 2u) {                        /* bound: 7 pairs */
        if ((uint32_t)CTL[k] == c) { *lit = (uint8_t)CTL[k + 1u]; return 0; }
    }
    while (u == 'p' && peek(r) == '{' && e < r->n && r->s[e] != '}') { e++; }   /* bound: the pattern */
    for (uint32_t k = 0; u == 'p' && peek(r) == '{' && e < r->n && k < 11u; k++) {   /* bound: 11 names */
        uint32_t l = (k >= 1u && k <= 5u) || k == 10u ? 2u : 1u;
        if (l == e - r->i - 1u && memcmp(r->s + r->i + 1u, PN[k], l) == 0) { r->i = e + 1u; *mask = PM[k]; return 1; }
    }
    if ((c >= '0' && c <= '9') || (u >= 'a' && u <= 'z')) { return rx_fail(r, MISS("an escape or property outside the census's set")); }
    *lit = c;                                                           /* \- \[ \\ \. ... */
    return 0;
}

/* under (?i): the ascii letters' other case, and the two non-ascii chars oniguruma folds onto one (s, k) */
static int fold(rx *r, uint32_t lo, uint32_t hi)
{
    if (r->icase && hi >= 0x80u) { return rx_fail(r, MISS("(?i) beyond ascii letters")); }
    for (uint32_t c = lo; r->icase && c <= hi; c++) {                  /* bound: 128 */
        uint32_t u = c | 0x20u;
        if (u < 'a' || u > 'z') { continue; }
        if (add_range(r, c ^ 0x20u, c ^ 0x20u) != 0 || (u == 's' && add_range(r, 0x17Fu, 0x17Fu) != 0) ||
            (u == 'k' && add_range(r, 0x212Au, 0x212Au) != 0)) {
            return -1;
        }
    }
    return 0;
}

/* [...] after its '[': ranges and escapes; a nested class adds its members (not negated); no && */
static int cls_items(rx *r, gcls *k)
{
    for (uint32_t first = 1u;; first = 0u) {                            /* bound: the pattern (>= 1 byte per item) */
        if (r->i >= r->n) { return rx_fail(r, MISS("an unterminated class")); }
        uint32_t c = next_cp(r), lo = c, hi;
        uint16_t m = 0, ng = 0;
        if (c == ']' && !first) { return 0; }
        if (c == '[') {
            if (peek(r) == '^' || peek(r) == ':' || ++r->depth > 8u) { return rx_fail(r, MISS("a negated, POSIX or deep class in a class")); }
            if (cls_items(r, k) != 0) { return -1; }
            r->depth--;
            continue;
        }
        if (c == '&' && peek(r) == '&') { return rx_fail(r, MISS("class intersection")); }
        int t = (c == '\\') ? (r->i < r->n ? esc(r, next_cp(r), &m, &ng, &lo) : rx_fail(r, MISS("a trailing backslash"))) : 0;
        if (t < 0 || (t == 1 && (ng || r->icase))) { return rx_fail(r, MISS("a negated or (?i) class escape in a class")); }
        if (t == 1) {
            k->mask = (uint16_t)(k->mask | m);
            continue;
        }
        hi = lo;
        if (peek(r) == '-' && r->i + 1u < r->n && r->s[r->i + 1u] != ']') {
            r->i++;
            hi = next_cp(r);
            if (hi == '[' || (hi == '\\' && (r->i >= r->n || esc(r, next_cp(r), &m, &ng, &hi) != 0)) || hi < lo) {
                return rx_fail(r, MISS("a class range that is not two characters in order"));
            }
        }
        if (add_range(r, lo, hi) != 0 || fold(r, lo, hi) != 0) { return -1; }
    }
}

/* class k (its ranges from rng[k->r0]) sorted, merged, its ascii bitmap set; a G_CHAR to it */
static int emit_cls(rx *r, gcls *k)
{
    uint32_t *g = r->rng + 2u * k->r0, n = r->nr - k->r0, m = 0;
    for (uint32_t i = 1; i < n; i++) {                                  /* bound: n^2 (insertion sort) */
        uint32_t lo = g[2u * i], hi = g[2u * i + 1u], j = i;
        for (; j > 0u && g[2u * j - 2u] > lo; j--) { g[2u * j] = g[2u * j - 2u]; g[2u * j + 1u] = g[2u * j - 1u]; }
        g[2u * j] = lo;
        g[2u * j + 1u] = hi;
    }
    for (uint32_t i = 0; i < n; i++) {                                  /* bound: n: merge overlaps */
        if (m > 0u && g[2u * i] <= g[2u * m - 1u] + 1u) {
            g[2u * m - 1u] = g[2u * i + 1u] > g[2u * m - 1u] ? g[2u * i + 1u] : g[2u * m - 1u];
            continue;
        }
        g[2u * m] = g[2u * i];
        g[2u * m++ + 1u] = g[2u * i + 1u];
    }
    r->nr = k->r0 + m;
    k->nr = m;
    for (uint32_t c = 0; c < 128u; c++) {                               /* bound: 128 x m */
        uint32_t in = (toks_ucd_flags(c) & k->mask) != 0u;
        for (uint32_t i = 0; i < m && !in; i++) { in = g[2u * i] <= c && c <= g[2u * i + 1u]; }
        k->ascii[c >> 5] |= (in ^ k->neg) << (c & 31u);
    }
    if (r->nc >= RX_CLS) { return rx_fail(r, MISS("too many classes")); }
    r->cls[r->nc] = *k;
    return emit(r, G_CHAR, 0u, (int32_t)r->nc++, 0);
}

/* the code point c (a literal), or the class of an escape */
static int emit_one(rx *r, uint32_t c, uint16_t mask, uint16_t neg)
{
    gcls k = { { 0u, 0u, 0u, 0u }, r->nr, 0u, mask, neg };
    if (mask != 0u && r->icase) { return rx_fail(r, MISS("(?i) on a class escape")); }
    if (mask == 0u && (add_range(r, c, c) != 0 || fold(r, c, c) != 0)) { return -1; }
    return emit_cls(r, &k);
}

static int parse_alt(rx *r);

/* {n}, {n,}, {n,m}, {,m} at '{' -> 1 (consumed); else 0 (the '{' is a literal, oniguruma) */
static int interval(rx *r, uint32_t *lo, uint32_t *hi)
{
    uint32_t i = r->i + 1u, v[2] = { 0u, 0u }, nd[2] = { 0u, 0u }, part = 0u;
    for (; i < r->n && r->s[i] != '}'; i++) {                           /* bound: the pattern */
        uint8_t c = r->s[i];
        if (c == ',' && part == 0u) { part = 1u; continue; }
        if (c < '0' || c > '9' || nd[part] > 4u) { return 0; }
        v[part] = v[part] * 10u + (uint32_t)(c - '0');
        nd[part]++;
    }
    if (i >= r->n || (nd[0] == 0u && (part == 0u || nd[1] == 0u))) { return 0; }
    *lo = v[0];
    *hi = part == 0u ? v[0] : nd[1] == 0u ? G_INF : v[1];
    r->i = i + 1u;
    return 1;
}

/* the atom at [s, ni) repeated lo..hi times, greedy, lazy or possessive (generic.md §4) */
static int quant(rx *r, uint32_t s, uint32_t lo, uint32_t hi, uint32_t lazy, uint32_t poss)
{
    uint32_t L = r->ni - s;
    if (hi < lo || hi == 0u) { return rx_fail(r, MISS("an empty or reversed repetition")); }
    if (poss) {                                    /* c?+ = (?:c|(?!c)), c*+ = c*(?!c), c++ = c+(?!c) */
        gins c = r->ins[s];
        if (L != 1u || c.op != G_CHAR || lo > 1u || (hi != 1u && hi != G_INF) || hi == lo) {
            return rx_fail(r, MISS("a possessive quantifier beyond one class"));
        }
        int bad = (lo == 0u) ? insert(r, s, G_SPLIT, 1, 3) || emit(r, G_JMP, 0u, hi == 1u ? 4 : -2, 0) : emit(r, G_SPLIT, 0u, -1, 1);
        return bad || emit(r, G_LOOK, 1u, 2, 0) || emit(r, G_CHAR, 0u, c.x, 0) ? -1 : emit(r, G_MATCH, 0u, 0, 0);
    }
    if (lo == 1u && hi == 1u) { return 0; }
    if (lo == 1u && hi == G_INF) { return emit(r, G_SPLIT, 0u, lazy ? 1 : -(int32_t)L, lazy ? -(int32_t)L : 1); }
    if (lo == 0u && (hi == 1u || hi == G_INF)) {   /* x? = SPLIT x; x* = SPLIT x JMP */
        int32_t out = (int32_t)L + (hi == 1u ? 1 : 2);
        if (insert(r, s, G_SPLIT, lazy ? out : 1, lazy ? 1 : out) != 0) { return -1; }
        return hi == 1u ? 0 : emit(r, G_JMP, 0u, -(int32_t)(L + 1u), 0);
    }
    uint64_t total = (uint64_t)lo * L + (hi == G_INF ? L + 2u : (uint64_t)(hi - lo) * (L + 1u));
    if (s + total + L > RX_INS) { return rx_fail(r, MISS("too large")); }
    gins *tmp = r->ins + RX_INS - L;               /* the atom, out of the way: lo copies, then the optional ones */
    memcpy(tmp, r->ins + s, (size_t)L * sizeof(gins));
    r->ni = s;
    for (uint32_t k = 0; k < lo + (hi == G_INF ? 1u : hi - lo); k++) {  /* bound: total <= RX_INS */
        int32_t skip = (hi == G_INF) ? (int32_t)L + 2 : (int32_t)((hi - k) * (L + 1u));   /* past every later copy */
        if (k >= lo && emit(r, G_SPLIT, 0u, lazy ? skip : 1, lazy ? 1 : skip) != 0) { return -1; }
        memcpy(r->ins + r->ni, tmp, (size_t)L * sizeof(gins));
        r->ni += L;
    }
    return hi == G_INF ? emit(r, G_JMP, 0u, -(int32_t)(L + 1u), 0) : 0;
}

/* (?:..) (?i:..) (?=..) (?!..) and (..) (a capture splits nothing) */
static int group(rx *r, uint32_t *zero)
{
    uint32_t s = r->ni, icase = r->icase, look = 2u, c = ':';
    if (peek(r) == '?') {
        r->i++;
        c = next_cp(r);
    }
    if (c == '=' || c == '!') {
        if (r->inlook) { return rx_fail(r, MISS("a lookahead inside a lookahead")); }
        look = (c == '!');
        *zero = 1u;
        r->inlook = 1u;
        if (emit(r, G_LOOK, look, 0, 0) != 0) { return -1; }
    } else if (c == 'i' && peek(r) == ':') {
        r->i++;
        r->icase = 1u;
    } else if (c != ':') {
        return rx_fail(r, MISS("a group other than (?: (?i: (?= (?!"));
    }
    if (++r->depth > 32u || parse_alt(r) != 0 || peek(r) != ')') { return rx_fail(r, MISS("an unbalanced or deep group")); }
    r->i++;
    r->depth--;
    r->icase = icase;
    if (look < 2u) {
        r->inlook = 0u;
        r->ins[s].x = (int32_t)(r->ni - s);        /* the body and its G_MATCH */
        return emit(r, G_MATCH, 0u, 0, 0);
    }
    return 0;
}

static int parse_atom(rx *r, uint32_t *zero)
{
    uint32_t c = next_cp(r), lit = 0, a;
    uint16_t m = 0, ng = 0;
    gcls k = { { 0u, 0u, 0u, 0u }, r->nr, 0u, 0u, 0u };
    switch (c) {
    case '(': return group(r, zero);
    case '[':
        if (peek(r) == '^') { k.neg = 1u; r->i++; }
        return (cls_items(r, &k) != 0) ? -1 : emit_cls(r, &k);
    case '.': return emit_one(r, '\n', 0u, 1u);                         /* every char but LF */
    case '^': *zero = 1u; return emit(r, G_ASSERT, A_BOL, 0, 0);
    case '$': *zero = 1u; return emit(r, G_ASSERT, A_EOL, 0, 0);
    case '*': case '+': case '?': return rx_fail(r, MISS("a quantifier with nothing before it"));
    case '\\': break;
    default: return emit_one(r, c, 0u, 0u);
    }
    if (r->i >= r->n) { return rx_fail(r, MISS("a trailing backslash")); }
    c = next_cp(r);
    a = c == 'A' ? A_BOS : c == 'z' ? A_EOS : c == 'Z' ? A_EOSNL : c == 'b' ? A_WB : c == 'B' ? A_NWB : 99u;
    if (a != 99u) { *zero = 1u; return emit(r, G_ASSERT, a, 0, 0); }
    int t = esc(r, c, &m, &ng, &lit);
    return t < 0 ? -1 : emit_one(r, lit, t == 1 ? m : 0u, t == 1 ? ng : 0u);
}

/* an atom and its quantifier, if any (a zero-width atom takes none) */
static int parse_repeat(rx *r)
{
    uint32_t s = r->ni, zero = 0u, lo = 0u, hi = 1u, brace = 0u, c;
    if (parse_atom(r, &zero) != 0) { return -1; }
    c = peek(r);
    if (c == '*' || c == '+') { lo = (uint32_t)(c == '+'); hi = G_INF; }
    else if (c == '{') { brace = (uint32_t)interval(r, &lo, &hi); if (!brace) { return 0; } }
    else if (c != '?') { return 0; }
    r->i += 1u - brace;
    uint32_t q = peek(r), lazy = (uint32_t)(q == '?'), poss = (uint32_t)(q == '+' && !brace);
    r->i += lazy + poss;
    c = peek(r);
    if (zero || c == '?' || c == '*' || c == '+' || (c == '{' && r->i + 1u < r->n && r->s[r->i + 1u] >= '0' && r->s[r->i + 1u] <= '9')) {
        return rx_fail(r, MISS("a quantifier on an assertion or on a quantifier"));
    }
    return quant(r, s, lo, hi, lazy, poss);
}

static int parse_alt(rx *r)
{
    uint32_t s = r->ni;
    while (r->i < r->n && peek(r) != '|' && peek(r) != ')') {           /* bound: >= 1 pattern byte per atom */
        if (parse_repeat(r) != 0) { return -1; }
    }
    if (peek(r) != '|') { return 0; }
    r->i++;
    if (insert(r, s, G_SPLIT, 1, 0) != 0) { return -1; }               /* this branch, else the rest */
    uint32_t j = r->ni;
    if (emit(r, G_JMP, 0u, 0, 0) != 0 || parse_alt(r) != 0) { return -1; }
    r->ins[s].y = (int32_t)(j + 1u - s);
    r->ins[j].x = (int32_t)(r->ni - j);
    return 0;
}

/* 1 when code p[0, n) reaches its G_MATCH without a character (assertions passable): hf's empty matches, which the
 * engine does not model (generic.md §3) */
static int nullable(const gins *p, uint32_t n, uint8_t *seen, uint32_t *stk)
{
    uint32_t sp = 1u;
    memset(seen, 0, n);
    stk[0] = 0u;
    while (sp > 0u) {                                                   /* bound: 2 n pushes (each pc once) */
        uint32_t pc = stk[--sp];
        if (pc >= n || seen[pc]) { continue; }
        const gins *in = &p[pc];
        seen[pc] = 1u;
        if (in->op == G_MATCH) { return 1; }
        if (in->op == G_SPLIT) { stk[sp++] = (uint32_t)((int32_t)pc + in->y); }
        if (in->op == G_SPLIT || in->op == G_JMP) { stk[sp++] = (uint32_t)((int32_t)pc + in->x); }
        if (in->op == G_ASSERT || in->op == G_LOOK) { stk[sp++] = pc + 1u + (in->op == G_LOOK ? (uint32_t)in->x : 0u); }
    }
    return 0;
}

uint64_t toks_gen_max_bytes(void) { return sizeof(struct toks_gen) + RX_INS * sizeof(gins) + RX_CLS * sizeof(gcls) + 8u * RX_RNG; }

int64_t toks_gen_compile(const toks_gen_spec *sp, uint32_t n, toks_arena *ar, const struct toks_gen **out, uint64_t *bytes,
                         toks_err *err)
{
    rx r;
    struct toks_gen h;
    int64_t ret = 0;
    memset(&r, 0, sizeof r);
    memset(&h, 0, sizeof h);
    gwork *w = (gwork *)toks_plat_alloc(sizeof(gwork));
    if (w == NULL) { return toks_fail(err, TOKS_E_NOMEM, "generic pre-tokenizer"); }
    r.ins = w->ins;
    r.cls = w->cls;
    r.rng = w->rng;
    for (uint32_t i = 0, dom = 0u; ret == 0 && i < n; i++) {   /* bound: n <= 16 (read_steps); dom: after a ByteLevel, bytes */
        gstep *q = &h.step[h.n_steps];
        uint32_t lit = sp[i].lit, pl = sp[i].n;
        const uint8_t *pat = sp[i].s;
        *q = (gstep){ GS_RX, (uint8_t)sp[i].beh, (uint8_t)sp[i].inv, (uint8_t)dom, 0u, r.ni, 0u };
        if (sp[i].kind == TOKS_GS_DIGITS || sp[i].kind == TOKS_GS_PUNCT) {
            q->kind = GS_PRED;
            q->pmask = sp[i].kind == TOKS_GS_DIGITS ? TOKS_UCD_RNUM : TOKS_UCD_PUNC;
            h.n_steps++;
            continue;
        }
        if (sp[i].kind == TOKS_GS_BYTELEVEL) {     /* use_regex: gpt-2's regex, Isolated, before its byte mapping */
            if (dom) { ret = toks_fail(err, TOKS_E_UNSUPPORTED, "pre_tokenizer ByteLevel twice (generic engine)"); break; }
            dom = 1u;
            if (!lit) { continue; }
            pat = (const uint8_t *)TOKS_PATTERNS[0].regex;
            for (pl = 0u; pat[pl] != 0; pl++) {}   /* bound: the gpt-2 string */
            lit = 0u;
            q->beh = TOKS_GB_ISOLATED;
            q->inv = 0u;
        }
        r.s = pat;
        r.n = pl;
        r.i = 0u;
        while (lit && r.i < pl) { (void)emit_one(&r, next_cp(&r), 0u, 0u); }   /* a String pattern, literally */
        if (!lit && parse_alt(&r) == 0 && r.i < r.n) { (void)rx_fail(&r, MISS("an unbalanced group")); }
        if (r.bad == NULL && emit(&r, G_MATCH, 0u, 0, 0) == 0 && r.ni - q->pc0 > G_MAX_STEP) { (void)rx_fail(&r, MISS("over 512 instructions a step")); }
        if (r.bad == NULL && (pl == 0u || nullable(r.ins + q->pc0, r.ni - q->pc0, w->seen, w->stk))) { (void)rx_fail(&r, MISS("one that matches the empty string")); }
        if (r.bad != NULL) { ret = toks_fail(err, TOKS_E_UNSUPPORTED, r.bad); break; }
        q->len = r.ni - q->pc0;
        h.max_ins = q->len > h.max_ins ? q->len : h.max_ins;
        h.n_steps++;
    }
    h.n_ins = r.ni;
    h.n_cls = r.nc;
    h.n_rng = r.nr;
    uint64_t ci = sizeof h + (uint64_t)r.ni * sizeof(gins), cr = ci + (uint64_t)r.nc * sizeof(gcls), total = cr + 8u * (uint64_t)r.nr;
    uint8_t *b = (ret == 0) ? (uint8_t *)toks_ar_alloc(ar, total, 8u) : NULL;
    if (ret == 0 && b == NULL) { ret = toks_fail(err, TOKS_E_NOMEM, "generic pre-tokenizer"); }   /* unreachable (config.h) */
    if (ret == 0) {
        memcpy(b, &h, sizeof h);
        memcpy(b + sizeof h, r.ins, (size_t)(ci - sizeof h));
        memcpy(b + ci, r.cls, (size_t)(cr - ci));
        memcpy(b + cr, r.rng, (size_t)(total - cr));
        *out = (const struct toks_gen *)(const void *)b;
        *bytes = total;
    }
    toks_plat_free(w, sizeof(gwork));
    return ret;
}

/* ---- the matcher: a Pike VM (priority threads = onig's leftmost-first order); steps nest as hf's (generic.md §5) */

/* the per-call memory in the scratch's extra region, words of M = the longest step's code: two mark arrays (M
 * each), the step's and the lookahead's thread lists (2M pairs each), two closure stacks (2M + 2 each) */
uint64_t toks_gen_scr(const struct toks_gen *g) { return 64u + 4u * (14u * (uint64_t)g->max_ins + 4u); }

typedef struct gvm {
    const toks_ctx *ctx;
    toks_scratch *h;
    toks_emit *e;
    const uint8_t *t;                  /* the text unit; the current step's piece is t[a, b) */
    uint64_t len, base, a, b, pos, n;
    int ids;
    uint32_t bytes, pc0, stamp[2];
    const struct toks_gen *g;
    const gins *ins;
    const gcls *cls;
    const uint32_t *rng;
    uint32_t *mark[2], *list[2][2], *stk, *ends;
} gvm;

/* the byte-level alphabet's char of byte b (gpt-2's bytes_to_unicode) */
static uint32_t byte_char(uint32_t b) { return b <= 0x20u ? 256u + b : (b >= 0x7Fu && b <= 0xA0u) ? b + 210u : b == 0xADu ? 323u : b; }

/* the code point at p (*k bytes): 0x110000 + byte for an ill-formed byte; in the byte domain a byte's char */
static uint32_t atom(const gvm *v, uint64_t p, uint32_t *k)
{
    uint32_t c = v->t[p], n = v->bytes ? 1u : toks_utf8_len(v->t + p, v->b - p);
    *k = (n != 0u) ? n : 1u;
    return v->bytes ? byte_char(c) : n == 0u ? 0x110000u + c : n == 1u ? c : toks_cp_decode(v->t + p, n);
}

/* the code point that ends at p (> a): a lead or ascii byte starts an atom, so 3 bytes back decide it */
static uint32_t atom_before(const gvm *v, uint64_t p)
{
    uint64_t s = p - 1u;
    uint32_t k;
    while (!v->bytes && s > v->a && s + 3u >= p && (v->t[s] & 0xC0u) == 0x80u) { s--; }   /* bound: 3 */
    uint32_t c = atom(v, s, &k);
    return (s + k == p) ? c : 0x110000u + v->t[p - 1u];
}

/* class k holds code point c; an ill-formed byte is \p{P} (SPEC §3.3: class P) and no literal */
static int has(const gvm *v, const gcls *k, uint32_t c)
{
    if (c < 128u) { return (int)((k->ascii[c >> 5] >> (c & 31u)) & 1u); }
    uint32_t in = (c < 0x110000u ? toks_ucd_flags(c) : TOKS_UCD_P) & k->mask, lo = 0u, hi = c < 0x110000u ? k->nr : 0u;
    const uint32_t *r = v->rng + 2u * k->r0;
    while (in == 0u && lo < hi) {                                       /* bound: log2 nr */
        uint32_t m = (lo + hi) >> 1;
        if (r[2u * m + 1u] < c) { lo = m + 1u; } else if (r[2u * m] > c) { hi = m; } else { in = 1u; }
    }
    return (int)((in != 0u) ^ k->neg);
}

static int word(uint32_t c) { return c < 0x110000u && (toks_ucd_flags(c) & TOKS_UCD_WORD) != 0u; }

static int assert_ok(const gvm *v, uint32_t kind, uint64_t p)
{
    uint32_t k;
    switch (kind) {
    case A_BOL: return p == v->a || atom_before(v, p) == 0x0Au;
    case A_EOL: return p == v->b || atom(v, p, &k) == 0x0Au;
    case A_BOS: return p == v->a;
    case A_EOS: return p == v->b;
    case A_EOSNL: return p == v->b || (atom(v, p, &k) == 0x0Au && p + k == v->b);
    default: return ((p > v->a && word(atom_before(v, p))) != (p < v->b && word(atom(v, p, &k)))) == (kind == A_WB);
    }
}

static int search(gvm *v, uint32_t pc0, uint64_t p, uint32_t lk, uint64_t *ms, uint64_t *me);

/* the threads pc reaches at p without a character, appended (pc, start) to L in priority order; lk: a lookahead's */
static uint32_t add(gvm *v, uint32_t *L, uint32_t n, uint32_t pc, uint32_t st, uint64_t p, uint32_t lk)
{
    uint32_t sp = 0u, *mark = v->mark[lk], stamp = v->stamp[lk];
    uint64_t d0, d1;
    v->stk[sp++] = pc;
    while (sp > 0u) {                                                   /* bound: 2 len pushes (each pc once) */
        pc = v->stk[--sp];
        const gins *in = &v->ins[pc];
        if (mark[pc - v->pc0] == stamp) { continue; }
        mark[pc - v->pc0] = stamp;
        if (in->op == G_SPLIT) { v->stk[sp++] = (uint32_t)((int32_t)pc + in->y); }
        if (in->op == G_SPLIT || in->op == G_JMP) {
            v->stk[sp++] = (uint32_t)((int32_t)pc + in->x);             /* x after y: it pops first */
        } else if (in->op == G_ASSERT) {
            if (assert_ok(v, in->a, p)) { v->stk[sp++] = pc + 1u; }
        } else if (in->op == G_LOOK) {
            if ((uint32_t)search(v, pc + 1u, p, 1u, &d0, &d1) != in->a) { v->stk[sp++] = pc + 1u + (uint32_t)in->x; }
        } else {
            L[2u * n] = pc;
            L[2u * n++ + 1u] = st;
        }
    }
    return n;
}

static void next_stamp(gvm *v, uint32_t lk)
{
    if (++v->stamp[lk] == 0u) {
        memset(v->mark[lk], 0, (size_t)v->g->max_ins * 4u);
        v->stamp[lk] = 1u;
    }
}

/* code pc0 over [p, b): a lookahead body (lk: anchored, 1 when a thread reaches its G_MATCH), else the leftmost-first
 * match [*ms, *me) of the step (1 when there is one) */
static int search(gvm *v, uint32_t pc0, uint64_t p, uint32_t lk, uint64_t *ms, uint64_t *me)
{
    uint32_t *cl = v->list[lk][0], *nl = v->list[lk][1], *t, *stk = v->stk, nc = 0u, nn, k = 0u, cp = 0u, found = 0u;
    v->stk += lk * (2u * v->g->max_ins + 2u);      /* a lookahead's closures above the one that asked */
    next_stamp(v, lk);
    for (uint32_t first = 1u;; first = 0u) {       /* bound: b - p + 1 positions */
        if (!found && (first || !lk)) { nc = add(v, cl, nc, pc0, (uint32_t)p, p, lk); }   /* a start, lowest priority */
        if (p < v->b) { cp = atom(v, p, &k); }
        next_stamp(v, lk);
        nn = 0u;
        for (uint32_t i = 0; i < nc; i++) {        /* bound: nc <= the step's code */
            const gins *in = &v->ins[cl[2u * i]];
            if (in->op == G_MATCH) {               /* every thread after it has a lower priority: cut */
                *ms = cl[2u * i + 1u];
                *me = p;
                found = 1u;
                break;
            }
            if (p < v->b && has(v, &v->cls[in->x], cp)) { nn = add(v, nl, nn, cl[2u * i] + 1u, cl[2u * i + 1u], p + k, lk); }
        }
        if (p >= v->b || (found && (lk || nn == 0u)) || (lk && nn == 0u)) { break; }
        t = cl;
        cl = nl;
        nl = t;
        nc = nn;
        p += k;
    }
    v->stk = stk;
    return (int)found;
}

/* ---- the steps, nested as hf applies them; final pieces leave in rounds of TOKS_CHUNK_PIECES ------------- */

static void flush(gvm *v)
{
    if (v->n != 0u) { toks_round(v->ctx, v->h, v->t, v->len, v->pos, v->ends, v->n, v->base, v->e, v->ids); }
    v->n = 0u;
}

static void piece(gvm *v, uint32_t i, uint64_t s, uint64_t e);

/* hf reports a piece by its original chars (ByteLevel's alignments): a cut inside a char ends at the char's end */
static uint64_t char_end(const gvm *v, uint64_t e)
{
    uint64_t s = e - 1u;
    if (e >= v->len || (v->t[e] & 0xC0u) != 0x80u) { return e; }
    while (s > 0u && e - s < 3u && (v->t[s] & 0xC0u) == 0x80u) { s--; }   /* bound: 3 */
    uint32_t k = toks_utf8_len(v->t + s, v->len - s);
    return (k != 0u && s + k > e) ? s + k : e;
}

typedef struct gpend { uint64_t s, e; uint32_t on, m, prev; } gpend;

static void pend_out(gvm *v, uint32_t i, gpend *q)
{
    if (q->on) { piece(v, i + 1u, q->s, q->e); }
    q->on = 0u;
}

/* one element [s, e) of step i (m: a match, after invert) through hf's split behavior (generic.md §5) */
static void elem(gvm *v, uint32_t i, gpend *q, uint64_t s, uint64_t e, uint32_t m)
{
    uint32_t beh = v->g->step[i].beh;
    m ^= v->g->step[i].inv;
    if (beh == TOKS_GB_ISOLATED || (beh == TOKS_GB_REMOVED && !m)) {
        piece(v, i + 1u, s, e);
    } else if (beh == TOKS_GB_PREV) {              /* a match joins the piece before it, unless that was a match */
        if (m && !q->prev && q->on) { q->e = e; } else { pend_out(v, i, q); *q = (gpend){ s, e, 1u, m, 0u }; }
        q->prev = m;
    } else if (beh == TOKS_GB_NEXT) {              /* a match joins the element after it, unless that is a match */
        if (q->on && !m) {
            piece(v, i + 1u, q->s, e);
            q->on = 0u;
            return;
        }
        pend_out(v, i, q);
        if (m) { *q = (gpend){ s, e, 1u, 1u, 0u }; } else { piece(v, i + 1u, s, e); }
    } else if (beh == TOKS_GB_CONTIGUOUS) {        /* neighbours alike join (only matches can be neighbours) */
        if (q->on && q->m == m) { q->e = e; } else { pend_out(v, i, q); *q = (gpend){ s, e, 1u, m, 0u }; }
    }
}

/* piece [s, e) through steps i.. (a final piece past the last) */
static void piece(gvm *v, uint32_t i, uint64_t s, uint64_t e)
{
    if (s >= e) { return; }
    if (i == v->g->n_steps) {
        if (v->n != 0u && s != v->ends[v->n - 1u]) { flush(v); }   /* a removed span: rounds are contiguous */
        if (v->n == 0u) { v->pos = s; }
        v->ends[v->n++] = (uint32_t)(v->ids ? e : char_end(v, e));
        if (v->n == TOKS_CHUNK_PIECES) { flush(v); }
        return;
    }
    const gstep *st = &v->g->step[i];
    gpend q = { 0u, 0u, 0u, 0u, 0u };
    uint64_t p = s, g = s, ms = 0u, me = 0u;
    uint32_t k = 0u, c;
    for (;;) {                                     /* bound: e - s (every element takes >= 1 atom) */
        v->a = s;
        v->b = e;
        v->bytes = st->bytes;
        v->pc0 = st->pc0;
        if (st->kind == GS_RX && (p >= e || !search(v, st->pc0, p, 0u, &ms, &me))) { break; }
        while (st->kind == GS_PRED && p < e) {     /* bound: e - p: the next char of the predicate's set */
            c = atom(v, p, &k);
            if (c < 0x110000u ? (toks_ucd_flags(c) & st->pmask) != 0u : st->pmask == TOKS_UCD_PUNC) { break; }
            p += k;
        }
        if (st->kind == GS_PRED && p >= e) { break; }
        if (st->kind == GS_PRED) {
            ms = p;
            me = p + k;
        }
        p = me;
        if (ms > g) { elem(v, i, &q, g, ms, 0u); }
        elem(v, i, &q, ms, me, 1u);
        g = me;
    }
    if (g < e) { elem(v, i, &q, g, e, 0u); }
    pend_out(v, i, &q);
}

void toks_gen_run(const toks_ctx *ctx, toks_scratch *h, const uint8_t *seg, uint64_t len, uint64_t base, toks_emit *e,
                  int ids)
{
    const struct toks_gen *g = ctx->gen;
    uint32_t M = g->max_ins, *w = (uint32_t *)(uintptr_t)toks_align64((uint64_t)(uintptr_t)toks_scr_at(h, h->off_bounce - ctx->scr_extra));
    gvm v = { ctx, h, e, seg, len, base, 0u, 0u, 0u, 0u, ids, 0u, 0u, { 0u, 0u }, g, G_INS(g), G_CLS(g), G_RNG(g),
              { w, w + M }, { { w + 2u * M, w + 4u * M }, { w + 6u * M, w + 8u * M } }, w + 10u * M,
              toks_scr_ends(h) };
    memset(w, 0, (size_t)M * 8u);                  /* the two mark arrays (stamps from 1) */
    piece(&v, 0u, 0u, len);
    flush(&v);
}
