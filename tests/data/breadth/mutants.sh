#!/bin/sh
# tests/data/breadth/mutants.sh: one-line mutants of the breadth code; each must make test_breadth or
# test_compile fail (the suites' teeth). Runs on a lab host from the repository root:
#   tools/remote.sh <host> 'sh tests/data/breadth/mutants.sh'
# Each mutant is a copy of src/ + tests/ under build/mut/<k>/ with one perl substitution applied; the
# substitution must change the file (a mutant that does not apply counts as an error, not a kill).
set -u
root=$(pwd)
out=build/mut
mkdir -p "$out"
uptime
n=0 killed=0 bad=0
mutant() {   # mutant <file> <perl substitution> <what>
    n=$((n + 1))
    d="$out/$n"
    rm -rf "$d" && mkdir -p "$d"
    cp -R "$root/src" "$root/tests" "$root/include" "$root/Makefile" "$d/"
    cp "$d/$1" "$d/$1.orig"
    perl -0pi -e "$2" "$d/$1"
    if cmp -s "$d/$1" "$d/$1.orig"; then
        echo "mutant $n ($3): DID NOT APPLY"
        bad=$((bad + 1))
        return
    fi
    if ! (cd "$d" && make -s -j8 BUILD_DIR=b b/tests/test_breadth b/tests/test_compile >/dev/null 2>&1); then
        echo "mutant $n ($3): killed (does not build)"
        killed=$((killed + 1))
        return
    fi
    if (cd "$d" && ./b/tests/test_breadth >/dev/null 2>&1 && ./b/tests/test_compile >/dev/null 2>&1); then
        echo "mutant $n ($3): SURVIVED"
    else
        echo "mutant $n ($3): killed"
        killed=$((killed + 1))
    fi
}
mutant src/core/segment.c 's/\(e->flags & TOKS_AF_SINGLE_WORD\) != 0u &&/0 \&\&/' "single_word never applied"
mutant src/core/segment.c 's/if \(e < len\) \{/if (0) {/' "single_word ignores the char after"
mutant src/core/segment.c 's/if \(s > 0u\) \{/if (0) {/' "single_word ignores the char before"
mutant src/core/segment.c 's/if \(start >= end\) \{ continue; \}//' "the empty token split is emitted"
mutant src/core/segment.c 's/    \*len = 1u;\n    return SEG_NOT_CP;\n\}/    *len = 1u;\n    return text[end - 1u];\n}/' "an ill-formed byte before a match counts as its code point"
mutant src/core/segment.c 's/if \(k == 0u\) \{ \*len = 1u; return SEG_NOT_CP; \}/if (k == 0u) { *len = 1u; return p[0]; }/' "an ill-formed byte counts as its code point"
mutant src/core/segment.c 's/\(toks_rx_word\[lo\]\[0\] <= cp\)/(toks_rx_word[lo][0] < cp)/' "\\\\w loses each range's first code point"
mutant src/core/segment.c 's/if \(ns < it->prev_end\) \{ ns = it->prev_end; \}//' "lstrip not clamped to prev_end"
mutant src/core/compile.c 's/\(a->lstrip != 0u \? TOKS_AF_LSTRIP : 0u\)/0u/' "lstrip flag dropped"
mutant src/core/compile.c 's/\(a->rstrip != 0u \? TOKS_AF_RSTRIP : 0u\)/0u/' "rstrip flag dropped"
mutant src/core/bpe_build.c 's/return cfg->n_vocab_raw == 0u \|\|/return 1 ||/' "decode-only ids enter byte2id / vhash / words"
mutant src/core/bpe_build.c 's/== 1u && model_token\(cfg, id\)\)/== 1u)/' "decode-only ids enter byte2id"
mutant src/core/config.c 's/flat\[0\]\.id = tp_cls;/flat[0].id = tp_sep;/' "Roberta cls / sep swapped"
mutant src/core/config.c 's/if \(c >= .1. && c <= .9.\) \{ return 0; \}/if (c >= 0x32 \&\& c <= 0x39) { return 0; }/' "dropout 0.1 read as zero"
mutant src/core/config.c 's/if \(all_ws != 0u\) \{/if (0) {/' "the hf-panic token set is accepted"
mutant src/core/config.c 's/\{ return PP_CLS_SEP; \}/{ (void)0; }/' "cls / sep shapes not recognized"
mutant src/core/config.c 's/!\(jv_eq_str\(beh, "Removed"\) && inv->num != 0\)/!jv_eq_str(beh, "Removed")/' "Removed accepted without invert"
mutant src/core/config.c 's/if \(!jv_absent\(mul\) && mul->num != 0\)/if (0)/' "pad_to_multiple_of ignored"
mutant src/core/config.c 's/a->special = \(uint8_t\)\(a->special \| \(special->num != 0\)\);/a->special = (uint8_t)(special->num != 0);/' "a repeated content: specialness not sticky"
mutant src/core/config.c 's/            a->lstrip = \(uint8_t\)\(ls->num != 0\);/            (void)0;/' "a repeated content: the first lstrip kept"
mutant src/core/config.c 's/            a->normalized = \(uint8_t\)\(norm->num != 0\);/            (void)0;/' "a repeated content: the first phase kept"
mutant src/core/api.c 's/uint64_t s_nb = nb - \(\(re > u.start\) \? re - u.start : 0u\), e_nb;/uint64_t s_nb = nb, e_nb;/' "pieces: an rstrip overlap's bytes counted twice"
echo "mutants: $n, killed $killed, did not apply $bad"
uptime
