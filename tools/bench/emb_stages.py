#!/usr/bin/env python3
"""tools/bench/emb_stages.py <algo> <variant> <src/core dir>: rewrite COPIES of the embedders' sources into the
stage-profile variants tools/bench/emb_stages.sh builds (the shipping sources are never touched). Counters and
timers go to tools/bench/spm_stages.c's sp_cnt (0/1 the first stage's ticks / calls, 2/3 the model's, 6/7 pieces /
piece bytes, 5 whole-piece hits, 11 misses):
  uni   unigram.c: the normalizer walk + pre-tokenizers per segment, the model per piece (piece cache + Viterbi)
        ship  unchanged
        prof  ticks around toks_uni_encode_segment (0/1) and every Viterbi run (2/3), pieces / bytes (6/7)
        walk  the model stubbed at piece_close (one id per piece): the walk + driver alone
        drv   toks_uni_encode_segment stubbed (one id per segment): the driver alone
  wp    wp_api.c / wp.c: toks_wp_scan_c = BertNormalizer + BertPreTokenizer, toks_wp_encode_c = the model
        ship  unchanged
        prof  ticks around every toks_wp_scan_c (0/1) and toks_wp_encode_c (2/3), pieces (6), hits / misses (5/11)
        scan  toks_wp_encode_c stubbed (one id per piece): scan + driver alone
        drv   toks_wp_scan_c stubbed (no pieces): the driver alone
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
'''


def sub(src, old, new):
    if src.count(old) != 1:
        sys.exit("emb_stages.py: anchor found %d times: %r" % (src.count(old), old[:80]))
    return src.replace(old, new)


def patch(path, fn):
    src = open(path).read()
    open(path, "w").write(fn(src))


SEG_SIG = ("int64_t toks_uni_encode_segment(const toks_uni *u, const uint8_t *text, uint64_t len, int pre_normalized,\n"
           "                                int gap_start, toks_uni_call *c)\n{\n")
PIECE_SIG = "static void uni_piece(const toks_uni *u, int virt, const uint8_t *p, uint64_t len, toks_emit *em, uint8_t *delta)\n{\n"
CLOSE = "static void piece_close(seg *g)\n{\n    if (!g->p_open) { return; }\n"


def uni(var, d):
    path = d + "/unigram.c"
    if var == "ship":
        return

    def f(src):
        if var == "prof":
            src = sub(src, '#include "unigram.h"\n', '#include "unigram.h"\n' + TICK)
            # every Viterbi run (a cache miss, or each piece without a cache) is timed: uni_piece becomes a wrapper
            real = PIECE_SIG.replace("static void uni_piece(", "static void uni_piece_real(")
            src = sub(src, PIECE_SIG, real[:-2] + ";\n" + PIECE_SIG +
                      "    uint64_t sp_t0 = sp_tick();\n    uni_piece_real(u, virt, p, len, em, delta);\n"
                      "    sp_cnt[2] += sp_tick() - sp_t0;\n    sp_cnt[3]++;\n}\n\n" + real)
            src = sub(src, CLOSE, CLOSE + "    sp_cnt[6]++;\n    sp_cnt[7] += g->p_mat ? g->p_ml : g->p_e - g->p_s;\n")
            src = sub(src, SEG_SIG, SEG_SIG.replace("int64_t toks_uni_encode_segment(", "static int64_t uni_seg_real("))
            src += ("\nint64_t toks_uni_encode_segment(const toks_uni *u, const uint8_t *text, uint64_t len, int pre_normalized,\n"
                    "                                int gap_start, toks_uni_call *c)\n{\n"
                    "    uint64_t t0 = sp_tick();\n"
                    "    int64_t r = uni_seg_real(u, text, len, pre_normalized, gap_start, c);\n"
                    "    sp_cnt[0] += sp_tick() - t0;\n    sp_cnt[1]++;\n    return r;\n}\n")
        elif var == "walk":
            # the model (cache and Viterbi) stubbed at the piece: one id per piece
            src = "#pragma clang diagnostic ignored \"-Wunused-function\"\n" + src
            src = sub(src, CLOSE, CLOSE + "    if (!g->c->pieces && (g->p_virt || g->p_mat || g->p_e > g->p_s)) {\n"
                      "        toks_put(&g->c->e, 1u);\n        g->p_open = 0;\n        g->p_virt = 0;\n        g->p_mat = 0;\n"
                      "        g->p_ml = 0u;\n        g->p_s = g->p_e = 0u;\n        return;\n    }\n")
        elif var == "drv":
            src = "#pragma clang diagnostic ignored \"-Wunused-function\"\n" + src
            src = sub(src, SEG_SIG, SEG_SIG + "    (void)u; (void)text; (void)pre_normalized; (void)gap_start;\n"
                      "    if (len != 0u) { toks_put(&c->e, (uint32_t)len); }\n    return (int64_t)len;\n")
        else:
            sys.exit("emb_stages.py: unknown variant " + var)
        return src
    patch(path, f)


def wp(var, d):
    if var == "ship":
        return
    if var == "prof":
        def f(src):
            src = sub(src, '#include "wp.h"\n', '#include "wp.h"\n' + TICK)
            src = sub(src, "    uint64_t np = toks_wp_scan_c(w->t, &a);\n",
                      "    uint64_t sp_t0 = sp_tick();\n    uint64_t np = toks_wp_scan_c(w->t, &a);\n"
                      "    sp_cnt[0] += sp_tick() - sp_t0;\n    sp_cnt[1]++;\n    sp_cnt[6] += np;\n")
            src = sub(src, "    uint64_t m = toks_wp_encode_c(w->t, &k);\n",
                      "    uint64_t sp_t0 = sp_tick();\n    uint64_t m = toks_wp_encode_c(w->t, &k);\n"
                      "    sp_cnt[2] += sp_tick() - sp_t0;\n    sp_cnt[3]++;\n    sp_cnt[5] += k.hits;\n"
                      "    sp_cnt[11] += k.misses;\n")
            return src
        patch(d + "/wp_api.c", f)
    elif var == "scan":
        def f(src):
            return sub(src, "uint64_t toks_wp_encode_c(const toks_wp_tables *t, toks_wp_encode_args *a)\n{\n",
                       "uint64_t toks_wp_encode_c(const toks_wp_tables *t, toks_wp_encode_args *a)\n{\n"
                       "    (void)t;\n    for (uint64_t i = 0; i < a->n; i++) { toks_st32(a->out + i, a->pieces[i].len); }\n"
                       "    a->n_out = a->n;\n    return a->n;\n")
        patch(d + "/wp.c", f)
    elif var == "drv":
        def f(src):
            src = "#pragma clang diagnostic ignored \"-Wunused-function\"\n" + src
            return sub(src, "uint64_t toks_wp_scan_c(const toks_wp_tables *t, toks_wp_scan_args *a)\n{\n",
                       "uint64_t toks_wp_scan_c(const toks_wp_tables *t, toks_wp_scan_args *a)\n{\n"
                       "    (void)t;\n    a->n = 0;\n    a->pos = a->len;\n    return 0;\n")
        patch(d + "/wp_scan.c", f)
    else:
        sys.exit("emb_stages.py: unknown variant " + var)


def main():
    algo, var, d = sys.argv[1], sys.argv[2], sys.argv[3]
    if algo == "uni":
        uni(var, d)
    elif algo == "wp":
        wp(var, d)
    else:
        sys.exit("emb_stages.py: unknown algo " + algo)


main()
