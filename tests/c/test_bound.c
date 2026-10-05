/*
 * test_bound.c: toks_encode_bound (include/toks.h "capacity"; the proof: src/core/compile.c toks_bound_terms_of).
 * The contexts: the tokenizer files under tests/data in FIXTURES (each must load) and the cached tokenizers in CACHED
 * ($TOKS_TOKENIZER_CACHE, else ~/.cache/toks/tokenizers) plus kimik3 ($TOKS_KIMI_DIR, else ~/.cache/toks/kimik3); a
 * SKIP line for each one absent. Both lists are fixed (no directory is listed, so the test is the same on every os):
 * a new fixture or cached file is covered once it is added here. Under all 12 flags encode accepts:
 * toks_encode's count <= toks_encode_bound(ctx, len) on the texts below, runs of 0xFF, '\n' and ' ' (every length
 * 1..33, then the block edges to 4096), a run of the char this test's expansion scan finds worst for the context (the
 * most ids per byte over every char a normalizer table maps, every byte and a sample of each utf-8 length), the
 * shortest added tokens repeated, alone and between text, 4096 random bytes (seeded) and random utf-8, two 8 KiB
 * slices of each bench corpus file (tools/bench/corpus.sha256's names in build/text, when there) and the fuzz corpora
 * (${FUZZ_STATE:-build/fuzz}/corpus/{encode,pieces,par}: tests/fuzz/gen.h's 8-byte header, then the text, encoded on
 * the pinned tokenizer its header selects; the one directory walk, so not on Windows), and each fixture's own texts
 * (the S and A lines of its .expect; an A line's ids, hf 0.23.2's, must be toks'). The scan's worst ratio must not
 * pass r; the fixtures made for a term (PINNED) must keep it; every context with r = 1 and every padded one must reach
 * its bound on an input (tight); the arithmetic (ceil, saturation, NULL, padding's rounding) is checked on constructed
 * terms, Fixed + pad_to_multiple_of also through encodes. The contexts run on a worker per cpu (16 at most; one on
 * Windows), the largest file first, and print in list order with their terms (r, g, the term that sets r). Run from
 * the repository root (make test does).
 */
#include "core.h"
#include "compile.h"
#include "unigram.h"
#include "../../src/gen/norm_nfc.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(_WIN32)
#  include <dirent.h>
#  include <pthread.h>
#  include <unistd.h>
#endif

#define MAXLEN (1u << 18)                                   /* the longest input encoded */
static const uint32_t FLAGS[12] = { 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14 };

/* every tokenizer file under tests/data that loads (the rest are refusal cases); the always-present tight cases among
 * them: spm/llamalike (byte fallback on an unknown byte run), compile/gpt2style (byte-level, one id per byte) and
 * breadth/trunc_pad (a padded context's padded length) */
static const char *const FIXTURES[] = {
    "breadth/added_opts.json", "breadth/digits_gpt2.json", "breadth/digits_p16.json", "breadth/drop_gpt2.json",
    "breadth/drop_nfc.json", "breadth/drop_unk.json", "breadth/drop_unk_fuse.json", "breadth/dup_added.json",
    "breadth/gen_behaviors.json", "breadth/gen_falcon.json", "breadth/gen_groups.json", "breadth/nonalpha.json",
    "breadth/pp_notype.json", "breadth/pp_shapes.json", "breadth/roberta.json", "breadth/roberta_seq.json",
    "breadth/spell_isoinv.json", "breadth/spell_removed.json", "breadth/tmpl_p20.json", "breadth/tmpl_p21.json",
    "breadth/tmpl_p22.json", "breadth/tmpl_p25.json", "breadth/trunc_left.json", "breadth/trunc_pad.json",
    "compile/dsv3style.json", "compile/gpt2style.json", "compile/llama3style.json", "compile/nonalpha.json",
    "compile/nosplit.json", "compile/qwen35style.json", "hardening/ws_lrstrip.json", "hardening/ws_lstrip.json",
    "hardening/ws_rstrip.json", "spm/gemma4like.json", "spm/holes_added.json", "spm/holes_nodec.json",
    "spm/ignore_merges.json", "spm/llamalike.json", "spm/merge_order.json", "spm/meta_always_split.json",
    "spm/meta_never_split.json", "spm/mistrallike.json", "spm/no_unk_no_bf.json", "spm/norm_specials.json",
    "spm/pm_order.json", "spm/replace_chain.json", "spm/space_in_vocab.json", "spm/unk_fused.json",
    "spm/unk_unfused.json", "spm/bound_phase0.json", "spm/bound_phase1.json", "unigram/bound_bf_nometa.json",
    "unigram/bound_bf_meta.json",
};

/* the fixtures made for a term (tests/data/spm/gen.py, tests/data/unigram/gen.py: hf 0.23.2's ids in their .expect):
 * each must keep it. "#a" is 3 ids for 2 bytes where a one-byte added token's match gives the next unit a ▁: the
 * phase-0 term (Prepend) and the phase-1 term (Metaspace always) set r = 2; unigram byte fallback without a ▁ piece
 * makes a space 3 byte ids (f = 3: r 3, the prefix 3 in g), its twin with the piece has r = 1 */
static const struct { const char *path; uint32_t why, num, den, f, g; } PINNED[] = {
    { "tests/data/spm/bound_phase0.json", TOKS_BOUND_PHASE0, 2, 1, 1, 1 },
    { "tests/data/spm/bound_phase1.json", TOKS_BOUND_PHASE1, 2, 1, 1, 1 },
    { "tests/data/unigram/bound_bf_nometa.json", TOKS_BOUND_TEXT, 3, 1, 3, 3 },
    { "tests/data/unigram/bound_bf_meta.json", TOKS_BOUND_TEXT, 1, 1, 1, 1 },
    /* toks.h's r by normalizer (T11 CA6b), a fixture or a cached file (by name; unchecked when absent) for each:
     * byte-level without a normalizer 1, NFC byte-level 3, NFKC 11, a unigram nmt_nfkc charsmap 6 (18/3), NFKD then
     * the charsmap 66 (198/3). No cached file has nmt_nfkc under byte fallback (11) */
    { "tests/data/compile/gpt2style.json", TOKS_BOUND_TEXT, 1, 1, 1, 0 },
    { "tests/data/breadth/drop_nfc.json", TOKS_BOUND_TEXT, 3, 1, 1, 0 },
    { "qwen38", TOKS_BOUND_TEXT, 3, 1, 1, 0 },
    { "dg-exaone35", TOKS_BOUND_TEXT, 11, 1, 1, 0 },
    { "uni_t5base", TOKS_BOUND_TEXT, 18, 3, 1, 2 },
    { "uni_albert", TOKS_BOUND_TEXT, 198, 3, 1, 3 },
};

/* the tokenizer cache's files that load: the fetchers' pins (tools/ci/fetch_tokenizers.py) and a few hub copies */
static const char *const CACHED[] = {
    "Qwen_Qwen3-0.6B", "bpe-distilroberta", "bpe-giga-emb", "bpe-gte-rerank-mbert", "bpe-jina5-omni",
    "bpe-llama32-fp8", "codellama", "deepseek-ai_DeepSeek-V3", "dg-exaone35", "dg-mellum2", "dg-powermoe",
    "dg-smollm-17b-w4", "dg-smollm2", "dg-smolvlm2", "dg-tiny-cohere", "dsr1", "dsv3", "dsv31", "dsv32", "dsv4",
    "dsv41flash", "embeddinggemma", "gemma1", "gemma2", "gemma3", "gemma3n", "gemma4", "gemma4-base", "glm53", "gpt2",
    "gptoss", "granite-emb", "gx-ax-k2", "gx-bloom", "gx-dscoder-67b", "gx-dscoder-7b15", "gx-dscoder2-lite",
    "gx-dsv2-lite", "gx-falcon7b", "gx-laguna-s21", "gx-laguna-xs2", "gx-ling3", "gx-llada2-mini", "gx-llada8b",
    "gx-minicpm5", "gx-natural-sql", "gx-spark-x25", "gx-tiny-cohere2", "gx-zeta21", "llama1", "llama2", "llama3",
    "llama31meta", "llama4", "minimaxm1", "minimaxm2", "minimaxm3", "minimaxt01", "mistral-nemo",
    "mistral-small-2409", "mistral-v0.1", "mistral-v0.3", "modernbert", "nemo3nano30b", "nemonano2", "nemonano2vl",
    "nemotron3-4b", "nemotron3-omni", "o200k", "olmoe", "openai-community_gpt2", "openai_gpt-oss-20b", "phi3",
    "phi3.5", "pythia", "pythia14m", "qwen3", "qwen38", "spm-otel-e4b", "tinyllama", "uni_albert", "uni_arctic2",
    "uni_bgem3", "uni_bgererank", "uni_flant5", "uni_llmjp3", "uni_llmjp4", "uni_me5large", "uni_me5small",
    "uni_mxbaixs", "uni_pmminilm", "uni_ruri3", "uni_t5base", "uni_xlnet", "unsloth_Llama-3.2-1B-Instruct",
    "wp-arctic-l", "wp-bert-base-uncased", "wp-bert-uncased", "wp-bge-base-ft", "wp-bge-small-zh", "wp-distiluse-ml",
    "wp-gte-large", "wp-jina-v2-small", "wp-ko-sroberta", "wp-labse", "wp-labse-setu", "wp-minilm-l6", "wp-mpnet",
    "wp-msmarco-bert", "wp-multiqa-minilm", "wp-multiqa-mpnet", "wp-paraphrase-minilm-l3", "wp-paraphrase-mpnet",
    "wp-pubmedbert", "wp-qdrant-minilm-l6", "wp-specter2", "wp-text2vec-zh", "yi", "yi-dolphin",
};
#define NFIX (sizeof FIXTURES / sizeof FIXTURES[0])
#define NCACHED (sizeof CACHED / sizeof CACHED[0])

/* the fuzz harnesses' pinned set, in tests/fuzz/fuzz.h FZ_PIN_DEF's order: a gen.h input's d[0] mod 16 selects one */
static const char *const FZ_PINS[16] = {
    "gpt2", "llama3", "qwen38", "glm53", "o200k", "dsv3", "gemma4", "mistral-v0.3", "wp-bert-uncased", "uni_t5base",
    "uni_bgem3", "minimaxm2", "nemotron3-4b", "dg-smollm2", "pythia", "tinyllama",
};

static int fails;                                           /* main's own checks, then every job's */
static uint64_t checks;
#define CHECK(c, ...)                                                                                      \
    do {                                                                                                   \
        checks++;                                                                                          \
        if (!(c)) {                                                                                        \
            if (++fails <= 40) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
        }                                                                                                  \
    } while (0)

/* one context's run: written by the worker that runs it, printed by main in list order */
typedef struct job {
    const char *name;
    char        path[1100];
    int         fixture, pin;                               /* a listed fixture (must load); its FZ_PINS index or -1 */
    uint64_t    size;                                       /* its file's bytes: the workers take the largest first */
    int64_t     load;                                       /* toks_load's result */
    int         r1, tight, pad, fails;
    uint64_t    checks, inputs, fz_inputs, ex_texts, ex_ids;
    double      secs;
    char        line[400];                                  /* the census line */
    char        msg[1600];                                  /* its first failures */
    size_t      msg_len;
} job;

/* a failed check of a job: counted, its first messages kept */
static void jfail(job *j, int line, const char *fmt, ...)
{
    j->fails++;
    size_t room = sizeof j->msg - j->msg_len;
    if (room < 160u) { return; }
    int k = snprintf(j->msg + j->msg_len, room, "FAIL %s:%d: ", __FILE__, line);
    if (k < 0 || (size_t)k >= room) { return; }
    j->msg_len += (size_t)k;
    room -= (size_t)k;
    va_list ap;
    va_start(ap, fmt);
    k = vsnprintf(j->msg + j->msg_len, room - 1u, fmt, ap);
    va_end(ap);
    if (k < 0) { k = 0; }
    j->msg_len += (size_t)k < room - 1u ? (size_t)k : room - 2u;
    j->msg[j->msg_len++] = '\n';
    j->msg[j->msg_len] = 0;
}

#define TCHECK(t, c, ...)                                                                                  \
    do {                                                                                                   \
        (t)->j->checks++;                                                                                  \
        if (!(c)) { jfail((t)->j, __LINE__, __VA_ARGS__); }                                                \
    } while (0)

typedef struct blob { const uint8_t *p; uint64_t n; int pin; } blob;
static blob *corp, *fz;                                     /* the bench corpora's slices; the fuzz inputs */
static uint32_t n_corp, n_fz;

typedef struct tctx {
    job        *j;
    toks_ctx   *ctx;
    void       *scr;
    uint32_t   *out;
    uint64_t    out_cap;
    uint8_t    *buf;                                        /* MAXLEN bytes of input */
    uint64_t    rng;
    toks_bound_terms bt;
    uint64_t    worst_num, worst_den;                       /* the scan's worst ids / bytes */
    uint32_t    worst_cp;
} tctx;

/* one input under one flags: the count within the bound, out sized by it */
static int64_t enc1(tctx *t, const uint8_t *s, uint64_t len, uint32_t fl)
{
    uint64_t b = toks_encode_bound(t->ctx, len);
    uint64_t cap = b < t->out_cap ? b : t->out_cap;
    int64_t n = toks_encode(t->ctx, s, len, fl, t->out, cap, t->scr);
    TCHECK(t, n >= 0, "%s: encode len %llu flags %u returned %lld", t->j->name, (unsigned long long)len, fl, (long long)n);
    TCHECK(t, n < 0 || (uint64_t)n <= b, "%s: len %llu flags %u: %lld ids > bound %llu (r %u/%u g %u)", t->j->name,
           (unsigned long long)len, fl, (long long)n, (unsigned long long)b, t->bt.num, t->bt.den, t->bt.g);
    return n;
}

static void enc(tctx *t, const uint8_t *s, uint64_t len)
{
    if (len > MAXLEN) { len = MAXLEN; }
    t->j->inputs++;
    for (uint32_t f = 0; f < 12u; f++) { (void)enc1(t, s, len, FLAGS[f]); }   /* bound: 12 */
}

static uint64_t run_of(tctx *t, const uint8_t *unit, uint64_t k, uint64_t times)
{
    uint64_t n = 0;
    for (uint64_t i = 0; i < times && n + k <= MAXLEN; i++) { memcpy(t->buf + n, unit, k); n += k; }   /* bound: times */
    return n;
}

static uint32_t put_utf8(uint8_t *o, uint32_t cp)
{
    if (cp < 0x80u) { o[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800u) { o[0] = (uint8_t)(0xC0u | cp >> 6); o[1] = (uint8_t)(0x80u | (cp & 63u)); return 2; }
    if (cp < 0x10000u) {
        o[0] = (uint8_t)(0xE0u | cp >> 12); o[1] = (uint8_t)(0x80u | ((cp >> 6) & 63u)); o[2] = (uint8_t)(0x80u | (cp & 63u));
        return 3;
    }
    o[0] = (uint8_t)(0xF0u | cp >> 18); o[1] = (uint8_t)(0x80u | ((cp >> 12) & 63u));
    o[2] = (uint8_t)(0x80u | ((cp >> 6) & 63u)); o[3] = (uint8_t)(0x80u | (cp & 63u));
    return 4;
}

/* a char the scan tries: every byte (cp < 0x100: the byte itself, invalid alone from 0x80), every char a normalizer
 * table maps (NFC / NFKC data: a decomposition, a compatibility one, a mark, not a boundary; the unigram charsmap's
 * keys), U+0130 (lowercase's one 2-char mapping), U+2581, and a sample of 2-, 3- and 4-byte chars */
static int candidate(const tctx *t, uint32_t cp)
{
    if (cp < 0x100u || cp == 0x130u || cp == 0x2581u || cp % 251u == 0u) { return 1; }
    if (cp >= 0xD800u && cp < 0xE000u) { return 0; }
    uint32_t w = toks_nfc_info(cp);
    if ((w & ((TOKS_NFC_DLEN_MASK << TOKS_NFC_DLEN_SHIFT) | TOKS_NFC_KX | TOKS_NFC_MARK | TOKS_NFC_NB | TOKS_NFC_NBK)) != 0u) {
        return 1;
    }
    return t->ctx->uni != NULL && t->ctx->uni->cfg.has_charsmap && toks_pc_char(&t->ctx->uni->pc, cp) != 0u;
}

/* the expansion scan: the char with the most ids per byte in a run of 16 (flags: no template, no added token, a
 * continuation), each run within the bound; its ratio must not pass r */
static void scan(tctx *t)
{
    t->worst_num = 0;
    t->worst_den = 1;
    t->worst_cp = 'a';
    for (uint32_t cp = 0; cp <= 0x10FFFFu; cp++) {          /* bound: every scalar value */
        if (!candidate(t, cp)) { continue; }
        uint8_t u[4];
        uint32_t k = cp < 0x100u ? (u[0] = (uint8_t)cp, 1u) : put_utf8(u, cp);
        uint64_t len = run_of(t, u, k, 16);
        int64_t n = enc1(t, t->buf, len, TOKS_NO_POSTPROCESS | TOKS_ADDED_NONE | TOKS_CONTINUATION);
        if (n > 0 && (uint64_t)n * t->worst_den > t->worst_num * len) {
            t->worst_num = (uint64_t)n;
            t->worst_den = len;
            t->worst_cp = cp;
        }
    }
    /* a run's ids per byte, its prefix (pfirst) taken off, never above r */
    uint64_t wn = t->worst_num > t->bt.pfirst ? t->worst_num - t->bt.pfirst : 0u;
    TCHECK(t, wn * t->bt.den <= (uint64_t)t->bt.num * t->worst_den, "%s: U+%04X: %llu ids per %llu bytes > r %u/%u",
           t->j->name, t->worst_cp, (unsigned long long)wn, (unsigned long long)t->worst_den, t->bt.num, t->bt.den);
}

static uint64_t rng(tctx *t)
{
    t->rng += 0x9E3779B97F4A7C15ull;
    uint64_t z = t->rng;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static const char *TEXTS[] = {
    "", "a", " ", "  ", "\n", "Hello world", "hello, world! 123 4567 89", "  \n\n  hi", "don't I'M 'll",
    "h\xc3\xa9llo w\xc3\xb6rld \xe4\xbd\xa0\xe5\xa5\xbd \xf0\x9f\x98\x80", "\xff\xfe abc \xe2\x82", "\r\n\r\n", "\t\t x",
    "<s>", "</s>", "<unk>", "<|endoftext|>", "[CLS] [SEP] [MASK] [UNK] [PAD]", "<pad><eos><bos>", "a<s>b</s>c",
    "\xe2\x96\x81", "\xe2\x96\x81\xe2\x96\x81 x \xe2\x96\x81y", "\xef\xb7\xba", "\xe3\x8c\x80\xe3\x8c\x80",
    "\xf0\x9d\x85\xa0\xf0\x9d\x85\xa0", "\xc4\xb0stanbul \xc4\xb0", "\xea\xb0\x80\xed\x9e\xa3 \xea\xb0\x81",
    "\xef\xac\x81 \xef\xac\x83 \xe2\x84\xab \xe2\x91\xa0", "\xd8\xa7\xd9\x84\xd8\xb9\xd8\xb1\xd8\xa8\xd9\x8a\xd8\xa9",
    "\xe0\xa4\xb9\xe0\xa4\xbf\xe0\xa4\xa8\xe0\xa5\x8d\xe0\xa4\xa6\xe0\xa5\x80", "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82",
    "e\xcc\x81\xcc\x81\xcc\x81 a\xcc\x8a", "\xc0\xaf\xed\xa0\x80\xf5\x80\x80\x80", "x\x00y\x01z\x7f",
    "def f(x):\n    return x**2  # \xe2\x9c\x93\n", "    \t\n   \r\n\n\n   x",
};

/* the byte runs' lengths: every one to 33, then each side of the 16 / 32 / 64-byte blocks and on to 4096 */
static const uint16_t RUNLEN[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 47, 48, 63, 64, 65, 95, 127, 128, 129, 255, 256, 257, 511, 512, 1023, 1024, 2047, 2048, 4095, 4096,
};

/* every input class against one context */
static void exercise(tctx *t)
{
    static const uint8_t RUNB[3] = { 0xFFu, '\n', ' ' };
    for (size_t i = 0; i < sizeof TEXTS / sizeof TEXTS[0]; i++) {   /* bound: the texts */
        enc(t, (const uint8_t *)TEXTS[i], strlen(TEXTS[i]));
        uint64_t n = run_of(t, (const uint8_t *)TEXTS[i], strlen(TEXTS[i]), 64);
        enc(t, t->buf, n);
    }
    for (uint32_t b = 0; b < 3u; b++) {                     /* bound: 3 bytes x the lengths */
        memset(t->buf, RUNB[b], 4096);
        for (size_t i = 0; i < sizeof RUNLEN / sizeof RUNLEN[0]; i++) { enc(t, t->buf, RUNLEN[i]); }   /* bound: 53 */
    }
    uint8_t u[4];
    uint32_t k = t->worst_cp < 0x100u ? (u[0] = (uint8_t)t->worst_cp, 1u) : put_utf8(u, t->worst_cp);
    static const uint64_t TIMES[] = { 1, 2, 3, 7, 64, 1024, 4096 };
    for (size_t i = 0; i < sizeof TIMES / sizeof TIMES[0]; i++) { enc(t, t->buf, run_of(t, u, k, TIMES[i])); }
    /* the shortest added tokens (each phase), every token within 2 bytes of the shortest phase-0 one (16 at most):
     * alone, repeated, and between text units that take a prefix */
    const toks_tables *tb = &t->ctx->t;
    uint32_t done = 0;
    for (uint32_t e = 0; e < tb->add_n && done < 16u; e++) {        /* bound: add_n */
        const toks_added_entry *a = &tb->add_entries[e];
        if (!(e == t->bt.e0 || e == t->bt.e1 || (a->phase == 0u && a->len <= t->bt.l0 + 2u))) { continue; }
        done++;
        const uint8_t *s = tb->add_bytes + a->off;
        uint8_t unit[256 + 8];
        static const char *const TAIL[] = { "", "a", " ", "\n", "\xe2\x96\x81", "x y" };
        for (size_t j = 0; j < sizeof TAIL / sizeof TAIL[0]; j++) {   /* bound: 6 */
            uint32_t tl = (uint32_t)strlen(TAIL[j]);
            memcpy(unit, s, a->len);
            memcpy(unit + a->len, TAIL[j], tl);
            enc(t, t->buf, run_of(t, unit, a->len + tl, 1));
            enc(t, t->buf, run_of(t, unit, a->len + tl, 3));
            enc(t, t->buf, run_of(t, unit, a->len + tl, 64));
        }
        memcpy(unit, s, a->len);
        memcpy(unit + a->len, u, k);
        enc(t, t->buf, run_of(t, unit, a->len + k, 64));
    }
    for (uint32_t seed = 1; seed <= 4u; seed++) {           /* bound: 4 seeds */
        t->rng = seed;
        for (uint32_t i = 0; i < 4096u; i++) { t->buf[i] = (uint8_t)rng(t); }   /* bound: 4096 */
        enc(t, t->buf, 4096);
        uint64_t n = 0;
        while (n + 4u <= 4096u) {                           /* bound: 4096 bytes */
            uint64_t r = rng(t);
            uint32_t cp = (r & 3u) == 0u ? (uint32_t)(r >> 8) % 0x80u : (r & 3u) == 1u ? 0x80u + (uint32_t)(r >> 8) % 0x780u
                        : (r & 3u) == 2u ? 0x800u + (uint32_t)(r >> 8) % 0xF800u : 0x10000u + (uint32_t)(r >> 8) % 0x100000u;
            if (cp >= 0xD800u && cp < 0xE000u) { continue; }
            n += put_utf8(t->buf + n, cp);
        }
        enc(t, t->buf, n);
    }
    for (uint32_t i = 0; i < n_corp; i++) { enc(t, corp[i].p, corp[i].n); }   /* bound: the slices */
    for (uint32_t i = 0; t->j->pin >= 0 && i < n_fz; i++) {  /* bound: the fuzz inputs; this pin's */
        if (fz[i].pin == t->j->pin) { enc(t, fz[i].p, fz[i].n); t->j->fz_inputs++; }
    }
}

static int hexv(int c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; }

/* the fixture's own texts: its .expect's S and A lines (tests/data/spm/gen.py's format: "S|A <0|1> <text hex or ->
 * <ids...>"), each within the bound under every flags; an A line's ids (hf 0.23.2 encoding the file as is: 0 = flags
 * 0, 1 = NONSPECIAL without pp) must be toks' */
static void expect_texts(tctx *t)
{
    char p[1200];
    size_t l = strlen(t->j->path);
    if (l < 5u || l + 3u > sizeof p || strcmp(t->j->path + l - 5u, ".json") != 0) { return; }
    memcpy(p, t->j->path, l - 5u);
    memcpy(p + l - 5u, ".expect", 8u);
    FILE *f = fopen(p, "rb");
    if (f == NULL) { return; }
    char *line = (char *)malloc(1u << 16);
    uint32_t *want = (uint32_t *)malloc((1u << 16) * 4u);
    while (line != NULL && want != NULL && fgets(line, 1 << 16, f) != NULL) {   /* bound: the file's lines */
        if ((line[0] != 'S' && line[0] != 'A') || line[1] != ' ' || line[2] == 0 || line[3] != ' ') { continue; }
        const char *h = line + 4;
        uint64_t n = 0;
        if (*h == '-') { h++; }
        while (n < MAXLEN && hexv(h[0]) >= 0 && hexv(h[1]) >= 0) {   /* bound: MAXLEN */
            t->buf[n++] = (uint8_t)(hexv(h[0]) << 4 | hexv(h[1]));
            h += 2;
        }
        enc(t, t->buf, n);
        t->j->ex_texts++;
        if (line[0] != 'A') { continue; }
        uint64_t nw = 0;
        while (*h == ' ' && nw < (1u << 16)) {              /* bound: 2^16 ids */
            h++;
            uint32_t v = 0;
            while (*h >= '0' && *h <= '9') { v = v * 10u + (uint32_t)(*h - '0'); h++; }   /* bound: the digits */
            want[nw++] = v;
        }
        uint32_t fl = line[2] == '0' ? 0u : (TOKS_ADDED_NONSPECIAL | TOKS_NO_POSTPROCESS);
        int64_t k = toks_encode(t->ctx, t->buf, n, fl, t->out, t->out_cap, t->scr);
        TCHECK(t, k == (int64_t)nw && memcmp(t->out, want, (size_t)nw * 4u) == 0, "%s: A %c (%llu bytes): %lld ids, hf %llu",
               t->j->name, line[2], (unsigned long long)n, (long long)k, (unsigned long long)nw);
        t->j->ex_ids++;
    }
    free(line);
    free(want);
    fclose(f);
}

/* r = 1: a text of one repeated byte whose every byte is one id reaches the bound (len + g); 1 when found */
static int tight(tctx *t)
{
    for (uint32_t b = 0; b < 256u; b++) {                   /* bound: 256 bytes */
        memset(t->buf, (int)b, 64);
        int64_t n = toks_encode(t->ctx, t->buf, 64, 0u, t->out, t->out_cap, t->scr);
        if (n >= 0 && (uint64_t)n == toks_encode_bound(t->ctx, 64)) { return 1; }
    }
    return 0;
}

static const char *WHY[3] = { "text", "phase-0 token", "phase-1 token" };
static const char *FAM[5] = { "?", "bytelevel", "spm", "unigram", "wordpiece" };

static double now(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* one job: load, scan, every input class, tightness; the census line */
static void run_job(job *j, uint8_t *buf)
{
    double t0 = now();
    tctx t;
    memset(&t, 0, sizeof t);
    t.j = j;
    t.buf = buf;
    j->load = toks_load(&t.ctx, j->path, NULL);
    if (j->load != 0) { return; }
    toks_bound_terms_of(t.ctx, &t.bt);
    uint64_t sb = toks_scratch_bytes(t.ctx, MAXLEN, 0u);
    t.scr = malloc((size_t)sb);
    t.out_cap = toks_encode_bound(t.ctx, MAXLEN);
    t.out = (uint32_t *)malloc((size_t)t.out_cap * 4u);
    if (t.scr == NULL || t.out == NULL || toks_scratch_init(t.ctx, t.scr, sb, 0u) != 0) {
        TCHECK(&t, 0, "%s: scratch / out", j->name);
    } else {
        TCHECK(&t, t.bt.num == t.ctx->bound_num && t.bt.den == t.ctx->bound_den && t.bt.g == t.ctx->bound_g,
               "%s: the context's terms are not the load's", j->name);
        TCHECK(&t, toks_encode_bound(t.ctx, 0) >= t.bt.g, "%s: bound(0)", j->name);
        scan(&t);
        exercise(&t);
        expect_texts(&t);
        for (size_t i = 0; i < sizeof PINNED / sizeof PINNED[0]; i++) {   /* bound: the pinned fixtures */
            if (strcmp(j->path, PINNED[i].path) != 0 && strcmp(j->name, PINNED[i].path) != 0) { continue; }
            TCHECK(&t, t.bt.why == PINNED[i].why && t.bt.num == PINNED[i].num && t.bt.den == PINNED[i].den &&
                   t.bt.f == PINNED[i].f && t.bt.g == PINNED[i].g, "%s: terms r %u/%u f %u g %u (%s), want r %u/%u f %u g %u (%s)",
                   j->name, t.bt.num, t.bt.den, t.bt.f, t.bt.g, WHY[t.bt.why], PINNED[i].num, PINNED[i].den, PINNED[i].f,
                   PINNED[i].g, WHY[PINNED[i].why]);
        }
        j->r1 = t.bt.num == t.bt.den;
        j->pad = t.ctx->o.pad_on != 0u;
        j->tight = j->r1 && tight(&t);
        if (j->pad) {                                       /* padded: a short text gives the padded length */
            int64_t n = toks_encode(t.ctx, "a", 1, 0u, t.out, t.out_cap, t.scr);
            j->tight = n >= 0 && (uint64_t)n == toks_encode_bound(t.ctx, 1);
            TCHECK(&t, n >= 0 && (uint64_t)n <= toks_encode_bound(t.ctx, 1), "%s: padded", j->name);
        }
        TCHECK(&t, j->tight || (!j->r1 && !j->pad), "%s: r = 1%s, and no input reaches the bound", j->name,
               j->pad ? " or padded" : "");
        int k = snprintf(j->line, sizeof j->line,
                         "  %-28s %-9s r %3u/%-3u g %2u  x %2u/%u f %u p0 %u p1 %u l0 %3u l1 %3u  sets r: %-13s worst U+%04X "
                         "%llu/%llu%s%s", j->name, FAM[t.ctx->t.algo <= 4u ? t.ctx->t.algo : 0u], t.bt.num, t.bt.den,
                         t.bt.g, t.bt.x, t.bt.xd, t.bt.f, t.bt.p0, t.bt.p1, t.bt.l0, t.bt.l1, WHY[t.bt.why], t.worst_cp,
                         (unsigned long long)t.worst_num, (unsigned long long)t.worst_den, j->pad ? "  padded" : "",
                         j->tight ? "  tight" : "");
        if (j->ex_texts != 0u && k > 0 && (size_t)k < sizeof j->line) {
            k += snprintf(j->line + k, sizeof j->line - (size_t)k, "  expect %llu (ids %llu)",
                          (unsigned long long)j->ex_texts, (unsigned long long)j->ex_ids);
        }
        if (j->fz_inputs != 0u && k > 0 && (size_t)k < sizeof j->line) {
            snprintf(j->line + k, sizeof j->line - (size_t)k, "  fuzz %llu", (unsigned long long)j->fz_inputs);
        }
    }
    free(t.scr);
    free(t.out);
    toks_unload(t.ctx);
    j->secs = now() - t0;
}

static job *jobs;
static uint32_t *order;                                     /* the jobs by size, largest first (the loads dominate) */
static uint32_t n_jobs, next_job;
#if !defined(_WIN32)
static pthread_mutex_t job_mu = PTHREAD_MUTEX_INITIALIZER;
#endif

static int by_size(const void *a, const void *b)
{
    uint64_t x = jobs[*(const uint32_t *)a].size, y = jobs[*(const uint32_t *)b].size;
    return x < y ? 1 : x > y ? -1 : 0;
}

/* a worker: the next job until none is left */
static void *worker(void *arg)
{
    (void)arg;
    uint8_t *buf = (uint8_t *)malloc(MAXLEN + 16u);
    for (;;) {                                              /* bound: n_jobs claims */
#if !defined(_WIN32)
        pthread_mutex_lock(&job_mu);
#endif
        uint32_t i = next_job++;
#if !defined(_WIN32)
        pthread_mutex_unlock(&job_mu);
#endif
        if (i >= n_jobs) { break; }
        job *j = &jobs[order[i]];
        if (buf == NULL) { j->load = -1; j->fails++; continue; }
        run_job(j, buf);
    }
    free(buf);
    return NULL;
}

/* toks_encode_bound's arithmetic on constructed terms: no shipped context has a non-integral r or pads to a multiple
 * of a Fixed length, so ceil, saturation and the rounding are set by hand on a loaded context here */
static void arithmetic(void)
{
    CHECK(toks_encode_bound(NULL, 0) == 0u && toks_encode_bound(NULL, 12345) == 0u, "bound(NULL, n) is not 0");
    toks_ctx *c = NULL;
    if (toks_load(&c, "tests/data/compile/gpt2style.json", NULL) != 0) { CHECK(0, "gpt2style.json does not load"); return; }
    CHECK(c->o.pad_on == 0u, "gpt2style.json pads");
    /* want: ceil(num len / den) + g, UINT64_MAX where that is above it; the saturation test may answer UINT64_MAX up
     * to num + g early (still an upper bound), never a wrapped value */
    static const struct { uint32_t num, den, g; uint64_t len, want; } AR[] = {
        { 3, 2, 0, 0, 0 }, { 3, 2, 0, 1, 2 }, { 3, 2, 0, 2, 3 }, { 3, 2, 0, 3, 5 }, { 3, 2, 0, 4, 6 },   /* 3/2: ceil */
        { 3, 2, 7, 3, 12 }, { 18, 3, 1, 5, 31 }, { 198, 3, 2, 1, 68 },                         /* + g; unreduced r */
        { 198, 3, 2, 1ull << 40, 66ull * (1ull << 40) + 2u }, { 11, 1, 3, 1000, 11003 },
        { 1, 1, 0, UINT64_MAX, UINT64_MAX }, { 1, 1, 1, UINT64_MAX - 2u, UINT64_MAX - 1u },   /* the top, exact */
        { 1, 1, 1, UINT64_MAX, UINT64_MAX }, { 3, 2, 7, UINT64_MAX, UINT64_MAX },             /* saturated */
        { 198, 3, 2, UINT64_MAX, UINT64_MAX }, { 11, 1, 3, UINT64_MAX / 11u, UINT64_MAX - 1u },
        { 3, 2, 0, UINT64_MAX / 3u * 2u, UINT64_MAX },
    };
    for (size_t i = 0; i < sizeof AR / sizeof AR[0]; i++) { /* bound: the cases */
        c->bound_num = AR[i].num, c->bound_den = AR[i].den, c->bound_g = AR[i].g;
        uint64_t b = toks_encode_bound(c, AR[i].len);
        CHECK(b == AR[i].want || (b == UINT64_MAX && UINT64_MAX - AR[i].want <= (uint64_t)AR[i].num + AR[i].g),
              "r %u/%u g %u: bound(%llu) = %llu, want %llu", AR[i].num, AR[i].den, AR[i].g,
              (unsigned long long)AR[i].len, (unsigned long long)b, (unsigned long long)AR[i].want);
    }
    /* padding, r = 1 g = 0: Fixed pads to its length rounded up to the multiple and never truncates; BatchLongest pads
     * one sequence to its own length rounded up */
    static const struct { uint32_t fixed, len, mult; uint64_t in, want; } PD[] = {
        { 1, 12, 0, 1, 12 }, { 1, 12, 0, 20, 20 }, { 1, 12, 8, 1, 16 }, { 1, 12, 8, 16, 16 }, { 1, 12, 8, 17, 17 },
        { 0, 0, 8, 1, 8 }, { 0, 0, 8, 8, 8 }, { 0, 0, 8, 9, 16 }, { 0, 0, 8, UINT64_MAX, UINT64_MAX },
        { 1, 12, 8, UINT64_MAX, UINT64_MAX },
    };
    c->bound_num = 1, c->bound_den = 1, c->bound_g = 0;
    for (size_t i = 0; i < sizeof PD / sizeof PD[0]; i++) { /* bound: the cases */
        c->o.pad_on = 1u, c->o.pad_fixed = PD[i].fixed, c->o.pad_len = PD[i].len, c->o.pad_multiple = PD[i].mult;
        uint64_t b = toks_encode_bound(c, PD[i].in);
        CHECK(b == PD[i].want, "pad fixed %u len %u multiple %u: bound(%llu) = %llu, want %llu", PD[i].fixed, PD[i].len,
              PD[i].mult, (unsigned long long)PD[i].in, (unsigned long long)b, (unsigned long long)PD[i].want);
    }
    toks_unload(c);
    /* Fixed + pad_to_multiple_of through encodes: trunc_pad.json (Fixed 12) with the multiple set to 8 pads "a" to 16,
     * the bound; every length to 40 stays within it */
    if (toks_load(&c, "tests/data/breadth/trunc_pad.json", NULL) != 0) { CHECK(0, "trunc_pad.json does not load"); return; }
    CHECK(c->o.pad_on != 0u && c->o.pad_fixed != 0u && c->o.pad_len == 12u && c->o.pad_multiple == 0u,
          "trunc_pad.json is not Fixed 12 without a multiple");
    c->o.pad_multiple = 8u;
    uint64_t sb = toks_scratch_bytes(c, 64u, 0u);
    void *scr = malloc((size_t)sb);
    uint32_t out[128];
    uint8_t text[64];
    memset(text, 'a', sizeof text);
    if (scr != NULL && toks_scratch_init(c, scr, sb, 0u) == 0) {
        int64_t n = toks_encode(c, text, 1, 0u, out, 128, scr);
        CHECK(n == 16 && toks_encode_bound(c, 1) == 16u, "Fixed 12 multiple 8: \"a\" gives %lld ids, bound %llu", (long long)n,
              (unsigned long long)toks_encode_bound(c, 1));
        for (uint64_t len = 0; len <= 40u; len++) {         /* bound: 41 lengths x 12 flags */
            for (uint32_t f = 0; f < 12u; f++) {
                n = toks_encode(c, text, len, FLAGS[f], out, 128, scr);
                CHECK(n >= 0 && (uint64_t)n <= toks_encode_bound(c, len), "Fixed 12 multiple 8: len %llu flags %u: %lld > %llu",
                      (unsigned long long)len, FLAGS[f], (long long)n, (unsigned long long)toks_encode_bound(c, len));
            }
        }
    } else {
        CHECK(0, "trunc_pad.json: scratch");
    }
    free(scr);
    toks_unload(c);
}

/* the bench corpora: tools/bench/corpus.sha256's files in build/text, an 8 KiB slice at the start and one in the
 * middle of the first 256 KiB (prose is the mildest input here, ~0.25 ids per byte; more of it only makes make test
 * slower) */
static void corpora(void)
{
    FILE *l = fopen("tools/bench/corpus.sha256", "rb");
    uint32_t listed = 0, found = 0;
    char line[512], name[300], sha[80], p[400];
    corp = (blob *)calloc(256, sizeof *corp);
    uint8_t *tmp = (uint8_t *)malloc(MAXLEN);
    while (l != NULL && corp != NULL && tmp != NULL && fgets(line, sizeof line, l) != NULL && n_corp + 2u <= 256u) {
        if (sscanf(line, "%79s %299s", sha, name) != 2) { continue; }
        listed++;
        snprintf(p, sizeof p, "build/text/%s", name);
        FILE *f = fopen(p, "rb");
        if (f == NULL) { continue; }
        size_t n = fread(tmp, 1, MAXLEN, f);
        fclose(f);
        found++;
        size_t at[2] = { 0, n / 2u }, len[2] = { n < 8192u ? n : 8192u, n > 16384u ? 8192u : 0u };
        for (int s = 0; s < 2; s++) {                       /* bound: 2 slices */
            if (len[s] == 0u) { continue; }
            uint8_t *c = (uint8_t *)malloc(len[s]);
            if (c == NULL) { continue; }
            memcpy(c, tmp + at[s], len[s]);
            corp[n_corp].p = c, corp[n_corp].n = len[s], corp[n_corp].pin = -1;
            n_corp++;
        }
    }
    if (l != NULL) { fclose(l); }
    free(tmp);
    if (found == 0u) { printf("bench corpora: SKIP (none of tools/bench/corpus.sha256's %u files in build/text)\n", listed); }
    else { printf("bench corpora: %u of %u files in build/text, %u slices of <= 8 KiB\n", found, listed, n_corp); }
}

/* the fuzz corpora: gen.h's inputs of the text harnesses, each to the pinned tokenizer fz_pin(d[0]) picks (d[0] mod
 * 16, the next present one where that file is missing); 32 MiB of text at most */
static void fuzz_corpora(const char *root)
{
#if defined(_WIN32)
    (void)root;
    printf("fuzz corpora: SKIP (Windows: no directory walk here; the corpora live on the linux hosts)\n");
#else
    static const char *const H[3] = { "encode", "pieces", "par" };
    const char *st = getenv("FUZZ_STATE");
    char dir[1024], p[1400], f[1800];
    snprintf(dir, sizeof dir, "%s/corpus", st != NULL && st[0] != 0 ? st : "build/fuzz");
    int present[16];
    for (int i = 0; i < 16; i++) {                          /* bound: 16 pins */
        snprintf(p, sizeof p, "%s/%s", root, FZ_PINS[i]);
        FILE *x = fopen(p, "rb");
        present[i] = x != NULL;
        if (x != NULL) { fclose(x); }
    }
    uint32_t cap = 0, dirs = 0, pinless = 0;
    uint64_t bytes = 0;
    uint8_t *tmp = (uint8_t *)malloc(8u + 65536u);
    for (int h = 0; h < 3 && tmp != NULL; h++) {            /* bound: 3 harnesses */
        snprintf(p, sizeof p, "%s/%s", dir, H[h]);
        DIR *d = opendir(p);
        if (d == NULL) { continue; }
        dirs++;
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {                 /* bound: the directory */
            if (de->d_name[0] == '.') { continue; }
            snprintf(f, sizeof f, "%s/%s", p, de->d_name);
            FILE *x = fopen(f, "rb");
            if (x == NULL) { continue; }
            size_t n = fread(tmp, 1, 8u + 65536u, x);
            fclose(x);
            if (n < 8u) { continue; }                       /* gen.h: too short for the header, no text */
            int pin = -1;
            for (int k = 0; k < 16 && pin < 0; k++) { if (present[(tmp[0] + k) % 16]) { pin = (tmp[0] + k) % 16; } }   /* bound: 16 */
            if (pin < 0) { pinless++; continue; }
            if (bytes + n > (32u << 20) || n_fz % 4096u == 0u) {
                if (bytes + n > (32u << 20)) { cap++; continue; }
                blob *g = (blob *)realloc(fz, (n_fz + 4096u) * sizeof *fz);
                if (g == NULL) { cap++; continue; }
                fz = g;
            }
            uint8_t *c = (uint8_t *)malloc(n - 8u + 1u);
            if (c == NULL) { cap++; continue; }
            memcpy(c, tmp + 8, n - 8u);
            fz[n_fz].p = c, fz[n_fz].n = n - 8u, fz[n_fz].pin = pin;
            n_fz++;
            bytes += n - 8u;
        }
        closedir(d);
    }
    free(tmp);
    if (dirs == 0u) { printf("fuzz corpora (%s): SKIP (none here)\n", dir); }
    else {
        printf("fuzz corpora (%s): %u harness dirs, %u inputs read (%llu bytes of text)%s%u over the 32 MiB cap, %u "
               "with no pinned file here\n", dir, dirs, n_fz, (unsigned long long)bytes, ", ", cap, pinless);
    }
#endif
}

int main(void)
{
    arithmetic();
    char root[1024], kimi[1024];
    const char *env = getenv("TOKS_TOKENIZER_CACHE"), *home = getenv("HOME"), *kd = getenv("TOKS_KIMI_DIR");
    if (env != NULL && env[0] != 0) { snprintf(root, sizeof root, "%s", env); }
    else { snprintf(root, sizeof root, "%s/.cache/toks/tokenizers", home != NULL ? home : "."); }
    if (kd != NULL && kd[0] != 0) { snprintf(kimi, sizeof kimi, "%s", kd); }
    else { snprintf(kimi, sizeof kimi, "%s/.cache/toks/kimik3", home != NULL ? home : "."); }
    corpora();
    fuzz_corpora(root);

    n_jobs = (uint32_t)(NFIX + NCACHED + 1u);
    jobs = (job *)calloc(n_jobs, sizeof *jobs);
    order = (uint32_t *)calloc(n_jobs, sizeof *order);
    if (jobs == NULL || order == NULL) { return 1; }
    for (uint32_t i = 0; i < n_jobs; i++) {                 /* bound: n_jobs */
        job *j = &jobs[i];
        j->pin = -1;
        if (i < NFIX) {
            snprintf(j->path, sizeof j->path, "tests/data/%s", FIXTURES[i]);
            j->name = j->path;
            j->fixture = 1;
        } else if (i < NFIX + NCACHED) {
            j->name = CACHED[i - NFIX];
            snprintf(j->path, sizeof j->path, "%s/%s", root, j->name);
            for (int k = 0; k < 16; k++) { if (strcmp(j->name, FZ_PINS[k]) == 0) { j->pin = k; } }   /* bound: 16 */
        } else {
            j->name = "kimik3";
            snprintf(j->path, sizeof j->path, "%s", kimi);
        }
        FILE *f = i + 1u < n_jobs ? fopen(j->path, "rb") : NULL;
        if (f != NULL && fseek(f, 0, SEEK_END) == 0) { long z = ftell(f); j->size = z > 0 ? (uint64_t)z : 0u; }
        if (f != NULL) { fclose(f); }
        if (i + 1u == n_jobs) { j->size = UINT64_MAX; }       /* kimik3, a directory: one of the heaviest loads */
        order[i] = i;
    }
    qsort(order, n_jobs, sizeof *order, by_size);
    uint32_t nw = 1;
    double t0 = now();
#if !defined(_WIN32)
    long cpus = 1;
#  if defined(_SC_NPROCESSORS_ONLN)
    cpus = sysconf(_SC_NPROCESSORS_ONLN);
#  endif
    nw = cpus < 1 ? 1u : cpus > 16 ? 16u : (uint32_t)cpus;
    pthread_t th[16];
    uint32_t started = 0;
    for (uint32_t i = 0; i < nw; i++) { if (pthread_create(&th[i], NULL, worker, NULL) == 0) { started++; } }   /* bound: 16 */
    if (started < nw) { (void)worker(NULL); }               /* a thread that did not start: main works too */
    for (uint32_t i = 0; i < started; i++) { pthread_join(th[i], NULL); }   /* bound: 16 */
    nw = started > 0u ? started : 1u;
#else
    (void)worker(NULL);
#endif
    double wall = now() - t0;

    uint64_t n_ctx = 0, n_r1 = 0, n_tight = 0, n_skip = 0, n_inputs = 0, fz_done = 0, fz_pinned = 0;
    const job *slow = NULL;
    printf("fixtures (tests/data):\n");
    for (uint32_t i = 0; i < n_jobs; i++) {                 /* bound: n_jobs */
        const job *j = &jobs[i];
        if (i == NFIX) { printf("cached tokenizers (%s):\n", root); }
        if (j->load != 0) {
            if (j->fixture) { CHECK(0, "%s: a listed fixture does not load (%lld)", j->name, (long long)j->load); }
            else { printf("  SKIP %s (not here, or not loaded: %lld)\n", j->name, (long long)j->load); n_skip++; }
            continue;
        }
        printf("%s", j->line);
        if (j->fails != 0) { printf("  FAILED %d checks", j->fails); }
        printf("\n");
        if (j->msg_len != 0u) { printf("%s", j->msg); }
        fails += j->fails;
        checks += j->checks;
        n_ctx++;
        n_r1 += (uint64_t)j->r1;
        n_tight += (uint64_t)j->tight;
        n_inputs += j->inputs;
        fz_done += j->fz_inputs;
        if (slow == NULL || j->secs > slow->secs) { slow = j; }
    }
    for (uint32_t i = 0; i < n_fz; i++) {                   /* bound: n_fz; the inputs a loaded pin took */
        for (uint32_t k = NFIX; k < NFIX + NCACHED; k++) {  /* bound: NCACHED */
            if (jobs[k].pin == fz[i].pin && jobs[k].load == 0) { fz_pinned++; break; }
        }
    }
    CHECK(fz_done == fz_pinned, "fuzz: %llu inputs encoded, %llu whose pin loaded", (unsigned long long)fz_done,
          (unsigned long long)fz_pinned);
    if (n_fz != 0u) { printf("fuzz corpora: %llu of %u inputs encoded on their pin\n", (unsigned long long)fz_done, n_fz); }
    printf("test_bound: %llu contexts (%llu with r = 1, %llu reach the bound exactly), %llu skipped, %llu inputs x 12 "
           "flags, %llu checks, %d failures; %u workers, %.2f s (slowest %s %.2f s)\n", (unsigned long long)n_ctx,
           (unsigned long long)n_r1, (unsigned long long)n_tight, (unsigned long long)n_skip,
           (unsigned long long)n_inputs, (unsigned long long)checks, fails, nw, wall, slow != NULL ? slow->name : "-",
           slow != NULL ? slow->secs : 0.0);
    for (uint32_t i = 0; i < n_corp; i++) { free((void *)corp[i].p); }   /* bound: n_corp */
    for (uint32_t i = 0; i < n_fz; i++) { free((void *)fz[i].p); }       /* bound: n_fz */
    free(corp);
    free(fz);
    free(jobs);
    free(order);
    return fails != 0;
}
