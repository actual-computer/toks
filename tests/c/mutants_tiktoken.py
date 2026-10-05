#!/usr/bin/env python3
"""tests/c/mutants_tiktoken.py: one-line mutants of src/core/tiktoken.c; each must make test_tiktoken fail (or crash)."""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
SRC = "src/core/tiktoken.c"
MUTANTS = [
    ("if (pad == 2u && (b64v(p[n - 3u]) & 0x0F) != 0) { return -1; }", "if (0) { return -1; }"),
    ("if (pad == 1u && (b64v(p[n - 2u]) & 0x03) != 0) { return -1; }", "if (0) { return -1; }"),
    ("if (d[i] == '\\r' || d[i] == '\\t' || (d[i] == ' ' && i != sp)) {", "if (d[i] == '\\t' || (d[i] == ' ' && i != sp)) {"),
    ("if (dn == 0u || dn > 7u || (d[ds] == '0' && dn > 1u)) {", "if (dn == 0u || dn > 7u) {"),
    ('if (rl[l.rank] != 0u) { return fail(err, TOKS_E_FORMAT, "tiktoken.model: a rank appears twice"); }', ""),
    ("for (uint32_t k = 1u; k < rl[id]; k++) {        /* bound: the token's length */",
     "for (uint32_t k = 2u; k < rl[id]; k++) {        /* bound: the token's length */"),
    ("cfg->ids_as_rank = 1u;", "cfg->ids_as_rank = 0u;"),
    ("cfg->ignore_merges = 1u;", "cfg->ignore_merges = 0u;"),
    ("b2u[b] = self ? (uint16_t)b : (uint16_t)(0x100u + n++);", "b2u[b] = self ? (uint16_t)b : (uint16_t)(0x101u + n++);"),
    ("run = (hit >= 1u && hit <= 8u && hit == run + 1u) ? run + 1u : (hit == 1u ? 1u : 0u);",
     "run = (hit >= 1u && hit <= 8u) ? run + 1u : 0u;"),
    ("if (n_ranks != cfg->n_vocab) {", "if (0) {"),
    ("if (memcmp(sp[a].content + p, sp[b].content, k) == 0) { return 0; }",
     "if (p == 0u && memcmp(sp[a].content + p, sp[b].content, k) == 0) { return 0; }"),
    ("if (in_dec[i] != 0u && dec_special[i] != named[i]) {", "if (0) {"),
    ("if (in_dec[i] != 0u || named[i] != 0u) {", "if (in_dec[i] != 0u) {"),
    ("for (uint32_t v = first + i; nd == 0u || v != 0u; v /= 10u)", "for (uint32_t v = first + i + 1u; nd == 0u || v != 0u; v /= 10u)"),
    ("if (f == NULL || f->type != JV_BOOL || f->num != 0) {", "if (f != NULL && f->num != 0) {"),
    ("cfg->n_ids = first + TOKS_TIKTOKEN_RESERVED;", "cfg->n_ids = first + TOKS_TIKTOKEN_RESERVED - 1u;"),
    ("if (bidx_find(&x, raw + ro, l.raw_n) >= 0) { return fail(err, TOKS_E_FORMAT, \"tiktoken.model: a token appears twice\"); }", ""),
    ("if (bidx_find(&x, &one, 1u) < 0) { return fail(err, TOKS_E_UNSUPPORTED, \"tiktoken.model lacks a one-byte token\"); }", ""),
    ("    *e = (i > s && d[i - 1u] == '\\r') ? i - 1u : i;", "    *e = i;"),
]


def main():
    lib = "build/macos-arm64/libtoks.a"
    src = open(SRC).read()
    killed = 0
    for i, (a, b) in enumerate(MUTANTS, 1):
        if src.count(a) != 1:
            print(f"m{i}: pattern found {src.count(a)} times: {a[:60]}")
            continue
        d = f"build/mut/m{i}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        open(f"{d}/tiktoken.c", "w").write(src.replace(a, b, 1))
        cc = ["clang", "-std=c17", "-O2", "-fno-strict-aliasing", "-fwrapv", "-Iinclude", "-Isrc/core", "-Isrc/platform"]
        if subprocess.run(cc + ["-c", f"{d}/tiktoken.c", "-o", f"{d}/tiktoken.o"], capture_output=True).returncode:
            print(f"m{i}: does not compile")
            continue
        shutil.copy(lib, f"{d}/libtoks.a")
        subprocess.run(["ar", "d", f"{d}/libtoks.a", "tiktoken.o"], check=True)
        subprocess.run(["ar", "rcs", f"{d}/libtoks.a", f"{d}/tiktoken.o"], check=True)
        subprocess.run(cc + ["-Itests/common", "-o", f"{d}/t", "tests/c/test_tiktoken.c", "tests/common/guard.c",
                             "tests/common/abicheck_arm64.S", f"{d}/libtoks.a"], check=True, capture_output=True)
        r = subprocess.run([f"{d}/t"], capture_output=True, text=True)
        fails = r.stdout.count("FAIL")
        if r.returncode == 0:
            print(f"m{i} SURVIVED: {a[:70]!r} -> {b[:50]!r}")
        else:
            killed += 1
            print(f"m{i} killed ({'crash ' if r.returncode < 0 else ''}{fails} FAIL lines): {a[:70]!r}")
    print(f"{killed} of {len(MUTANTS)} killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
