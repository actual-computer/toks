/* tools/bench/stall.c: the worst-case (stall) audit of SPEC §7.3 (docs/hardening.md). Adversarial input
 * classes, each aimed at one path (K1 added tokens, K3 templates, K5 / K6 incl. the long-piece and heap paths,
 * whole-segment spm bpe, K7, unigram's Viterbi, wordpiece's greedy match, the normalizers, the generic engine),
 * timed through toks.h against the en text of the same tokenizer at the same size.
 *
 *   stall [-s sizes] [-r reps] [-c class,...] [-e en.txt] [-t auto|scalar] [-o encode|pieces] [-T secs] tok...
 *
 *   -s  comma-separated text sizes in bytes (default 4096,65536,1048576)
 *   -r  reps per cell (default 5; the median is reported)
 *   -c  only these classes (default: all; `stall -l` lists them)
 *   -e  the en reference text (default: a built-in paragraph, tiled; give a real one for receipts)
 *   -T  wall seconds a (class, size) cell may take before it is killed and reported TIMEOUT (default 60);
 *       the larger sizes of that class are skipped
 * Each rep initializes the scratch (outside the timer: the cold state, SPEC §12.3) and times one call. One
 * line per cell: tokenizer, class, size, median ns per input byte, the ratio to en at that size, the growth
 * exponent from the previous size (log(t2 / t1) / log(n2 / n1): 1 = linear, 2 = quadratic) and the call's count.
 * Every cell runs in a child process (a stall or a crash costs one cell, never the audit). */
#define _POSIX_C_SOURCE 200809L
#include "toks.h"

#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint64_t rng(uint64_t *s)                    /* splitmix64 */
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static uint32_t utf8(uint32_t cp, uint8_t *o)
{
    if (cp < 0x80u) { o[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800u) { o[0] = (uint8_t)(0xC0u | (cp >> 6)); o[1] = (uint8_t)(0x80u | (cp & 0x3Fu)); return 2; }
    if (cp < 0x10000u) {
        o[0] = (uint8_t)(0xE0u | (cp >> 12)); o[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)); o[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    o[0] = (uint8_t)(0xF0u | (cp >> 18)); o[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
    o[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)); o[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 4;
}

/* ---- the text under construction: exactly n bytes when done (units that do not fit are cut by pad()) ---- */

typedef struct buf { uint8_t *p; uint64_t n, cap; } buf;

static int room(const buf *b, uint64_t k) { return b->n + k <= b->cap; }
static void put(buf *b, const void *s, uint64_t k) { if (room(b, k)) { memcpy(b->p + b->n, s, k); b->n += k; } }
static void puts_(buf *b, const char *s) { put(b, s, strlen(s)); }
static void putcp(buf *b, uint32_t cp) { uint8_t u[4]; put(b, u, utf8(cp, u)); }
static void pad(buf *b) { while (b->n < b->cap) { b->p[b->n++] = 'a'; } }   /* a unit that did not fit */

/* ---- the tokenizer's markup tokens (<...> / [...], as fuzz.h): the added-token classes' material ----------- */

typedef struct lits { uint8_t *p[512]; uint64_t n[512]; uint32_t k; uint64_t longest; } lits;

static void lits_get(const toks_ctx *c, uint32_t n_ids, lits *l)
{
    memset(l, 0, sizeof *l);
    for (uint32_t id = 0; id < n_ids && l->k < 512u; id++) {
        uint64_t k = 0;
        const uint8_t *p = toks_token(c, id, &k);
        if (p == NULL || k < 3u || k > 255u) { continue; }
        if (!((p[0] == '<' && p[k - 1] == '>') || (p[0] == '[' && p[k - 1] == ']'))) { continue; }
        l->p[l->k] = (uint8_t *)p;
        l->n[l->k] = k;
        if (k > l->n[l->longest]) { l->longest = l->k; }
        l->k++;
    }
}

/* ---- the classes ------------------------------------------------------------------------------------------- */

typedef struct cls { const char *name, *unit, *why; int kind; } cls;
enum { FIX, LET, CASE, B64, DIG, PUN, BYTES, CJKR, HANR, WORDS, LONGW, SCAL, A_WHOLE, A_CUT, A_PFX, A_WS, A_NEST,
       MARKS, MARKS_DESC, EN };

static const cls CLS[] = {
    { "en", "", "the reference: en text", EN },
    { "lt", "<", "K1: '<' flood (every candidate fails at byte 2)", FIX },
    { "lt_pipe", "<|", "K1: '<|' flood", FIX },
    { "lt_fw", "<\xef\xbd\x9c", "K1: '<' + fullwidth bar (DeepSeek's long bucket)", FIX },
    { "added_whole", "", "K1: the tokenizer's markup tokens, whole, glued", A_WHOLE },
    { "added_cut", "", "K1: each markup token cut short, glued (radix walks that fail late)", A_CUT },
    { "added_pfx", "", "K1: the longest markup token minus its last byte, repeated", A_PFX },
    { "added_nest", "", "K1: prefixes of the longest markup token of every length, glued", A_NEST },
    { "added_ws", "", "segment: markup tokens between whitespace runs (lstrip / rstrip)", A_WS },
    { "ws_space", " ", "K3: one whitespace run", FIX },
    { "ws_nl", "\n", "K3: newline run", FIX },
    { "ws_tab", "\t", "K3", FIX },
    { "ws_crlf", "\r\n", "K3", FIX },
    { "ws_sp_nl", " \n", "K3: \\s*[\\r\\n]+ vs \\s+(?!\\S)", FIX },
    { "ws_nbsp", "\xc2\xa0", "K3: non-ascii \\s", FIX },
    { "ws_ideo", "\xe3\x80\x80", "K3: U+3000", FIX },
    { "letters", "a", "K3 / K6: one giant piece of one letter", FIX },
    { "letters_ab", "ab", "K6: one giant piece, two letters", FIX },
    { "letters_rand", "", "K6 heap: one giant piece of random letters", LET },
    { "case_rand", "", "K6: random case letters", CASE },
    { "base64", "", "K6: base64-like giant piece", B64 },
    { "digits", "1", "K3: digit run", FIX },
    { "digits_rand", "", "K3: random digits", DIG },
    { "alt_ln", "a1", "K3: alternating classes", FIX },
    { "alt_lp", "a.", "K3", FIX },
    { "alt_lsp", "a ", "K3 / K5: many 2-byte pieces", FIX },
    { "alt_psp", ". ", "K3", FIX },
    { "punct", "!", "K3: punctuation run", FIX },
    { "punct_rand", "", "K3 / K6: random ascii punctuation", PUN },
    { "apos", "'", "K3: contraction prefixes", FIX },
    { "contr", "'s", "K3: contractions", FIX },
    { "bytes_rand", "", "invalid utf-8 storm", BYTES },
    { "ff", "\xff", "invalid byte run", FIX },
    { "cont", "\x80", "continuation bytes", FIX },
    { "trunc3", "\xe4\xb8", "truncated 3-byte chars", FIX },
    { "cjk", "\xe4\xb8\xad", "K3 / K6 / K7: one CJK char repeated", FIX },
    { "cjk_rand", "", "K6 / K7: random CJK", CJKR },
    { "hangul_rand", "", "NFC / NFKC: random hangul syllables", HANR },
    { "hangul_lvt", "\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8", "NFC: jamo L V T (compose)", FIX },
    { "marks", "", "NFC: one base + combining marks of one class", MARKS },
    { "marks_desc", "", "NFC: one base + marks in descending class order (canonical reorder)", MARKS_DESC },
    { "nfc3", "\xf0\x9d\x85\xa0", "NFC: U+1D160 (3x growth)", FIX },
    { "nfc2", "\xe0\xa5\x98", "NFC: U+0958 (2x)", FIX },
    { "zwj", "\xf0\x9f\x91\xa9\xe2\x80\x8d", "zwj chain", FIX },
    { "emoji", "\xf0\x9f\x98\x80", "4-byte run", FIX },
    { "meta", "\xe2\x96\x81", "spm / unigram: U+2581 run", FIX },
    { "meta_sp", "\xe2\x96\x81 ", "spm: U+2581 + space", FIX },
    { "bytetok", "<0x41>", "spm byte-token look-alike", FIX },
    { "words_rand", "", "K5: random distinct words (cache thrash)", WORDS },
    { "longwords_rand", "", "K6 / long cache: random 16-200 letter words", LONGW },
    { "scalar_rand", "", "any scalar", SCAL },
};
#define NCLS (sizeof CLS / sizeof CLS[0])

static const char EN_BUILTIN[] =
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, "
    "it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of "
    "Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing "
    "before us, we were all going direct to Heaven, we were all going direct the other way -- in short, the period "
    "was so far like the present period, that some of its noisiest authorities insisted on its being received, for "
    "good or for evil, in the superlative degree of comparison only.\n\n";

static uint8_t *en_text;
static uint64_t en_len;

static void gen(const cls *c, const lits *l, uint64_t n, uint64_t seed, buf *b)
{
    uint64_t s = seed;
    b->n = 0;
    b->cap = n;
    static const char P[] = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    static const char B[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static const uint32_t DESC[] = { 0x0345u, 0x0301u, 0x0316u, 0x0328u, 0x05B0u, 0x0334u };   /* ccc 240 230 220 202 10 1 */
    switch (c->kind) {
    case EN:
        while (b->n < n) { put(b, en_text, en_len < n - b->n ? en_len : n - b->n); }
        break;
    case FIX:
        while (room(b, strlen(c->unit))) { puts_(b, c->unit); }
        break;
    case MARKS:
        putcp(b, 'a');
        while (room(b, 2)) { putcp(b, 0x0301u); }
        break;
    case MARKS_DESC:
        putcp(b, 'a');
        for (uint64_t i = 0; room(b, 2); i++) { putcp(b, DESC[i % 6u]); }
        break;
    default:
        while (b->n < n) {
            uint64_t before = b->n;
            switch (c->kind) {
            case LET: putcp(b, 'a' + (uint32_t)(rng(&s) % 26u)); break;
            case CASE: { uint32_t r = (uint32_t)(rng(&s) % 52u); putcp(b, r < 26u ? 'a' + r : 'A' + r - 26u); break; }
            case B64: putcp(b, (uint8_t)B[rng(&s) % 64u]); break;
            case DIG: putcp(b, '0' + (uint32_t)(rng(&s) % 10u)); break;
            case PUN: putcp(b, (uint8_t)P[rng(&s) % (sizeof P - 1u)]); break;
            case BYTES: { uint8_t x = (uint8_t)rng(&s); put(b, &x, 1); break; }
            case CJKR: putcp(b, 0x4E00u + (uint32_t)(rng(&s) % 0x5200u)); break;
            case HANR: putcp(b, 0xAC00u + (uint32_t)(rng(&s) % 11172u)); break;
            case WORDS: case LONGW: {
                uint64_t k = c->kind == WORDS ? 2u + rng(&s) % 11u : 16u + rng(&s) % 185u;
                if (!room(b, k + 1u)) { break; }
                for (uint64_t i = 0; i < k; i++) { putcp(b, 'a' + (uint32_t)(rng(&s) % 26u)); }
                putcp(b, ' ');
                break;
            }
            case SCAL: {
                uint32_t cp = (uint32_t)(rng(&s) % 0x110000u);
                if (cp >= 0xD800u && cp <= 0xDFFFu) { cp = 0xFFFDu; }
                putcp(b, cp);
                break;
            }
            case A_WHOLE: case A_CUT: case A_WS: {
                if (l->k == 0u) { puts_(b, "<|x|>"); break; }
                uint32_t i = (uint32_t)(rng(&s) % l->k);
                uint64_t k = l->n[i];
                if (c->kind == A_CUT) { k = 1u + rng(&s) % (k - 1u); }
                put(b, l->p[i], k);
                if (c->kind == A_WS) {
                    static const char *const W[] = { " ", "  ", "\n", " \n ", "\t", "   \n\n   " };
                    puts_(b, W[rng(&s) % 6u]);
                }
                break;
            }
            case A_PFX:
                if (l->k == 0u) { puts_(b, "<|"); break; }
                put(b, l->p[l->longest], l->n[l->longest] - 1u);
                break;
            case A_NEST:
                if (l->k == 0u) { puts_(b, "<|"); break; }
                for (uint64_t k = 1; k < l->n[l->longest] && room(b, k); k++) { put(b, l->p[l->longest], k); }
                break;
            }
            if (b->n == before) { break; }          /* the unit did not fit */
        }
        break;
    }
    pad(b);
}

/* ---- one cell, in a child: median ns/B of reps cold calls -------------------------------------------------- */

typedef struct res { double ns_b; int64_t count; int ok; } res;

static res cell(const toks_ctx *ctx, const cls *c, const lits *l, uint64_t n, int reps, int pieces, int timeout)
{
    res r = { 0.0, 0, 0 };
    int fd[2];
    if (pipe(fd) != 0) { return r; }
    pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        alarm((unsigned)timeout);
        buf b = { (uint8_t *)malloc(n ? n : 1), 0, 0 };
        gen(c, l, n, 0x5EEDull + n, &b);
        uint64_t sb = toks_scratch_bytes(ctx, n, 0u);
        void *scr = malloc(sb);
        int64_t cnt = pieces ? toks_pieces(ctx, b.p, n, 0u, NULL, 0u, NULL) : 0;
        (void)cnt;
        uint64_t cap = 3u * n + 64u;
        uint32_t *out = (uint32_t *)malloc(cap * 4u);
        double t[64];
        int64_t m = -1;
        if (reps > 64) { reps = 64; }
        for (int i = 0; i < reps; i++) {
            if (toks_scratch_init(ctx, scr, sb, 0u) != 0) { _exit(3); }
            uint64_t t0 = now_ns();
            m = pieces ? toks_pieces(ctx, b.p, n, 0u, out, cap, scr) : toks_encode(ctx, b.p, n, 0u, out, cap, scr);
            t[i] = (double)(now_ns() - t0);
        }
        for (int i = 1; i < reps; i++) {             /* insertion sort: the median */
            for (int j = i; j > 0 && t[j - 1] > t[j]; j--) { double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
        }
        res o = { t[reps / 2] / (double)(n ? n : 1), m, m >= 0 };
        if (write(fd[1], &o, sizeof o) != (ssize_t)sizeof o) { _exit(4); }
        _exit(0);
    }
    close(fd[1]);
    int st = 0;
    ssize_t got = read(fd[0], &r, sizeof r);
    close(fd[0]);
    waitpid(pid, &st, 0);
    if (got != (ssize_t)sizeof r || !WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        r.ok = WIFSIGNALED(st) && WTERMSIG(st) == SIGALRM ? -1 : -2;   /* -1 timeout, -2 crash / error */
        r.count = WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st);
    }
    return r;
}

static int pick(const char *list, const char *name)
{
    if (list == NULL) { return 1; }
    size_t k = strlen(name);
    for (const char *p = list; (p = strstr(p, name)) != NULL; p += k) {
        if ((p == list || p[-1] == ',') && (p[k] == 0 || p[k] == ',')) { return 1; }
    }
    return 0;
}

int main(int argc, char **argv)
{
    uint64_t sizes[16] = { 4096u, 65536u, 1048576u };
    int nsz = 3, reps = 5, pieces = 0, timeout = 60, opt;
    const char *only = NULL, *enf = NULL;
    uint32_t tier = TOKS_TIER_AUTO;
    while ((opt = getopt(argc, argv, "s:r:c:e:t:o:T:l")) != -1) {
        switch (opt) {
        case 's': {
            nsz = 0;
            for (char *p = optarg; *p && nsz < 16; ) { sizes[nsz++] = strtoull(p, &p, 10); if (*p == ',') { p++; } }
            break;
        }
        case 'r': reps = atoi(optarg); break;
        case 'c': only = optarg; break;
        case 'e': enf = optarg; break;
        case 't': tier = strcmp(optarg, "scalar") == 0 ? TOKS_TIER_SCALAR : TOKS_TIER_AUTO; break;
        case 'o': pieces = strcmp(optarg, "pieces") == 0; break;
        case 'T': timeout = atoi(optarg); break;
        case 'l':
            for (size_t i = 0; i < NCLS; i++) { printf("%-16s %s\n", CLS[i].name, CLS[i].why); }
            return 0;
        default:
            fprintf(stderr, "usage: stall [-s sizes] [-r reps] [-c classes] [-e en.txt] [-t auto|scalar] [-o encode|pieces] [-T secs] tok...\n");
            return 2;
        }
    }
    if (enf != NULL) {
        FILE *f = fopen(enf, "rb");
        if (f == NULL) { perror(enf); return 1; }
        fseek(f, 0, SEEK_END);
        long m = ftell(f);
        fseek(f, 0, SEEK_SET);
        en_text = (uint8_t *)malloc((size_t)m + 1u);
        en_len = fread(en_text, 1, (size_t)m, f);
        fclose(f);
    } else {
        en_text = (uint8_t *)(uintptr_t)EN_BUILTIN;
        en_len = sizeof EN_BUILTIN - 1u;
    }
    printf("# stall: op %s tier %s reps %d en %s\n", pieces ? "pieces" : "encode", tier == TOKS_TIER_SCALAR ? "scalar" : "auto",
           reps, enf != NULL ? enf : "built-in");
    printf("%-14s %-16s %9s %10s %9s %6s %10s\n", "tokenizer", "class", "bytes", "ns/B", "x_en", "exp", "count");
    for (int a = optind; a < argc; a++) {
        toks_load_opts o;
        memset(&o, 0, sizeof o);
        o.size = sizeof o;
        o.tier = tier;
        toks_ctx *ctx = NULL;
        int64_t r = toks_load(&ctx, argv[a], &o);
        const char *name = strrchr(argv[a], '/') != NULL ? strrchr(argv[a], '/') + 1 : argv[a];
        if (r != 0) { printf("%-14s load %lld\n", name, (long long)r); continue; }
        toks_info info;
        memset(&info, 0, sizeof info);
        info.size = sizeof info;
        toks_get_info(ctx, &info);
        lits l;
        lits_get(ctx, info.n_ids, &l);
        double en[16] = { 0 };
        for (size_t ci = 0; ci < NCLS; ci++) {
            const cls *c = &CLS[ci];
            if (c->kind != EN && !pick(only, c->name)) { continue; }
            double prev = 0.0;
            for (int si = 0; si < nsz; si++) {
                res x = cell(ctx, c, &l, sizes[si], reps, pieces, timeout);
                if (x.ok != 1) {
                    printf("%-14s %-16s %9llu %10s %9s %6s %10s\n", name, c->name, (unsigned long long)sizes[si],
                           x.ok == -1 ? "TIMEOUT" : "FAIL", "-", "-", "-");
                    fflush(stdout);
                    break;
                }
                if (c->kind == EN) { en[si] = x.ns_b; }
                double e = prev > 0.0 ? log(x.ns_b * (double)sizes[si] / (prev * (double)sizes[si - 1])) /
                                        log((double)sizes[si] / (double)sizes[si - 1]) : 0.0;
                printf("%-14s %-16s %9llu %10.3f %9.2f %6.2f %10lld\n", name, c->name, (unsigned long long)sizes[si], x.ns_b,
                       en[si] > 0.0 ? x.ns_b / en[si] : 0.0, e, (long long)x.count);
                fflush(stdout);
                prev = x.ns_b;
            }
        }
        toks_unload(ctx);
    }
    return 0;
}
