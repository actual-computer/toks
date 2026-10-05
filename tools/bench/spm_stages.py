#!/usr/bin/env python3
"""tools/bench/spm_stages.py <variant> <spm_c.c>: rewrite a COPY of src/core/spm_c.c into one of the stage-profile
variants tools/bench/spm_stages.sh builds (the shipping sources are never touched):
  ship  unchanged
  prof  counters (sp_cnt, tools/bench/spm_stages.c) + a timer around toks_spm_encode and around every model call
  scan  word() stubbed to one store per word: the scan + its flush loops alone
  drv   toks_spm_encode stubbed to one store per unit: the driver alone
  words every word the cache path sees and every unit start, to the file SP_WORDS names, in one pass (counts only)
Every anchor must match exactly once (a drifted source fails loudly instead of profiling something else)."""
import sys

TICK = r'''
extern uint64_t sp_cnt[16];
static inline uint64_t sp_tick(void)
{
#if defined(__x86_64__)
    uint32_t lo, hi, aux;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) : : "memory");
    return ((uint64_t)hi << 32) | lo;
#else
    uint64_t v;
    __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v) : : "memory");
    return v;
#endif
}
static inline int sp_cut_between(const toks_spm *s, uint32_t a, uint32_t b)
{
    uint32_t ia = a & TOKS_SPM_E_ID, ib = b & TOKS_SPM_E_ID;
    uint32_t xa = (a >> TOKS_SPM_E_SI_SHIFT) & 0xFFu, xb = (b >> TOKS_SPM_E_SI_SHIFT) & 0xFFu;
    if (ia != TOKS_SPM_E_NOID && ib != TOKS_SPM_E_NOID && !(xa != TOKS_SPM_E_SI_NONE && xb != TOKS_SPM_E_SI_NONE)
        && (a & TOKS_SPM_E_PAIRED)) { sp_cnt[12]++; }
    return toks_spm_cut_between(s, a, b);
}
'''

ENC_SIG = ("uint64_t toks_spm_encode(const toks_tables *t, const toks_spm *s, const uint8_t *text, uint64_t len, int at_start,\n"
           "                         uint32_t *out, uint64_t cap, uint64_t n, uint8_t *cache, uint64_t cm, uint64_t tag,\n"
           "                         uint8_t *work, uint32_t flags)\n{\n")


WORD_K = "    uint8_t *bucket = NULL;\n    if (cache != NULL && len <= (uint64_t)TOKS_KEY_MAXLEN) {\n"   # word_k's first lines


def sub(src, old, new):
    if src.count(old) != 1:
        sys.exit("spm_stages.py: anchor found %d times: %r" % (src.count(old), old[:80]))
    return src.replace(old, new)


def main():
    var, path = sys.argv[1], sys.argv[2]
    src = open(path).read()
    if var == "ship":
        return
    if var == "prof":
        src = sub(src, '#include "bpe.h"\n', '#include "bpe.h"\n' + TICK)
        src = sub(src, "    uint64_t m = model(t, s, p, len, wp, work);\n",
                  "    uint64_t sp_t0 = sp_tick();\n    uint64_t m = model(t, s, p, len, wp, work);\n"
                  "    sp_cnt[2] += sp_tick() - sp_t0;\n    sp_cnt[3]++;\n    sp_cnt[11] += len > 15u;\n"
                  "    sp_cnt[14] += len > 30u;\n    sp_cnt[15] += len <= 30u && m > 7u;\n")
        src = sub(src, "            return put_wide(wb + 32, out, cap, n);\n",
                  "            sp_cnt[4]++;\n            return put_wide(wb + 32, out, cap, n);\n")
        src = sub(src, "        const uint8_t *v = bpe_cache_get(bucket, k, tw);\n        if (v != NULL) {\n",
                  "        const uint8_t *v = bpe_cache_get(bucket, k, tw);\n        if (v != NULL) {\n            sp_cnt[5]++;\n")
        src = sub(src, "toks_spm_whash(k.lo, k.hi), k);\n        if (v != NULL) {\n",
                  "toks_spm_whash(k.lo, k.hi), k);\n        if (v != NULL) {\n            sp_cnt[13]++;\n")
        src = sub(src, WORD_K, WORD_K.replace("    uint8_t *bucket = NULL;\n",
                                              "    uint8_t *bucket = NULL;\n    sp_cnt[6]++;\n    sp_cnt[7] += len;\n"))
        src = sub(src, "                if (n < cap) { toks_st32(out + n, o & TOKS_SPM_E_ID); }",
                  "                sp_cnt[8]++;\n                if (n < cap) { toks_st32(out + n, o & TOKS_SPM_E_ID); }")
        src = sub(src, "                m |= cut8(s->cut_ab, text + i - 1u) << (i - base);\n",
                  "                m |= cut8(s->cut_ab, text + i - 1u) << (i - base);\n                sp_cnt[9]++;\n")
        src = sub(src, "        uint32_t e = entry_at(s, text + i, len - i, &k);\n        one[i - base]",
                  "        uint32_t e = entry_at(s, text + i, len - i, &k);\n        sp_cnt[10]++;\n        one[i - base]")
        src = sub(src, "(cuts && toks_spm_cut_between(s, prev, e))", "(cuts && sp_cut_between(s, prev, e))")
        # the timer around the whole unit: the real body becomes spm_encode_real
        src = sub(src, ENC_SIG, ENC_SIG.replace("uint64_t toks_spm_encode(", "static uint64_t spm_encode_real("))
        src += ("\nuint64_t toks_spm_encode(const toks_tables *t, const toks_spm *s, const uint8_t *text, uint64_t len, int at_start,\n"
                "                         uint32_t *out, uint64_t cap, uint64_t n, uint8_t *cache, uint64_t cm, uint64_t tag,\n"
                "                         uint8_t *work, uint32_t flags)\n{\n"
                "    uint64_t t0 = sp_tick();\n"
                "    uint64_t r = spm_encode_real(t, s, text, len, at_start, out, cap, n, cache, cm, tag, work, flags);\n"
                "    sp_cnt[0] += sp_tick() - t0;\n    sp_cnt[1]++;\n    return r;\n}\n")
    elif var == "scan":
        src = "#pragma clang diagnostic ignored \"-Wunused-function\"\n" + src
        src = sub(src, WORD_K,
                  "    (void)t; (void)s; (void)wp; (void)cache; (void)cm; (void)tw; (void)work; (void)miss; (void)k;\n"
                  "    if (n < cap) { toks_st32(out + n, (uint32_t)(end - p) ^ (uint32_t)len); }\n    return n + 1u;\n"
                  + WORD_K)
    elif var == "drv":
        src = sub(src, ENC_SIG, ENC_SIG + "    if (len != 0u) {\n        if (n < cap) { toks_st32(out + n, (uint32_t)len); }\n"
                  "        return n + 1u;\n    }\n")
    elif var == "words":
        # every word the cache path sees (word_k: a one-vocab-char word never gets there) and every unit start,
        # to the file SP_WORDS names, during one pass (tools/bench/spm_stages.c sp_word / sp_unit)
        src = sub(src, '#include "bpe.h"\n', '#include "bpe.h"\n'
                  "void sp_word(const uint8_t *p, uint64_t len, int wp);\nvoid sp_unit(void);\n")
        src = sub(src, WORD_K, "    sp_word(p, len, wp);\n" + WORD_K)
        src = sub(src, ENC_SIG, ENC_SIG + "    sp_unit();\n")
    else:
        sys.exit("spm_stages.py: unknown variant " + var)
    open(path, "w").write(src)


main()
