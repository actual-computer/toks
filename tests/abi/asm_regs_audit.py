#!/usr/bin/env python3
"""tests/abi/asm_regs_audit.py: the asm abi lint, run by `make asmcheck` (part of make test) over what asmcheck built.
Every asm function preserves what its abi makes callee-saved, and says so where the os unwinder reads it:

  x86_64, from the coff objects (x86_64-pc-windows-msvc, the strictest abi: rbx rbp rdi rsi r12-r15 and xmm6-15
          callee-saved): every such register an instruction names (llvm-objdump -d, after macro expansion: a register
          named through a .macro argument counts) must be saved by the function's win64 unwind codes (llvm-objdump
          -u: PushNonVol / SaveNonVol / SaveXMM128), which asm.h's PROLOGUE ngpr, nxmm emits; a LEAF saves none.
  arm64,  from the sources (asm.h): x18 never; under PROLOGUE nx, nd only x19 .. x(18 + 2 nx) and d8 .. d(7 + 2 nd)
          of the callee-saved; a LEAF uses x0-x17, v0-v7, v16-v31 only; every EPILOGUE repeats its PROLOGUE. A
          register named only inside a .macro body is unseen here; test_k1 / test_k7's abicheck calls see it at run time.

It found toks_k7_spm_avx2's xmm6 / xmm7 (the win64 clobber fixed in commit e6d9760). Its own teeth: selftest.S's
toks_k0_frame_x86 (PROLOGUE 6, 10) must be seen using and saving xmm6-15 and the eight callee-saved gprs, else the
lint fails: an objdump whose output it no longer reads must not pass everything. Needs python3 and an llvm objdump
(llvm-objdump beside $CC or on PATH, or an objdump that is LLVM's, as on macos); without one it fails, never skips.

  asm_regs_audit.py ASMCHECK_DIR CC        (from the repository root; prints violations, exit 1 if any)
"""
import glob
import os
import re
import shutil
import subprocess
import sys

COFF = "x86_64-pc-windows-msvc"
GPR = {}
for r, names in (("RBX", "rbx ebx bx bl bh"), ("RBP", "rbp ebp bp bpl"), ("RDI", "rdi edi di dil"),
                 ("RSI", "rsi esi si sil")):
    for n in names.split():
        GPR[n] = r
for k in range(12, 16):
    for s in ("", "d", "w", "b"):
        GPR[f"r{k}{s}"] = f"R{k}"
GPR_RE = re.compile(r"\b(" + "|".join(sorted(GPR, key=len, reverse=True)) + r")\b")
VEC_RE = re.compile(r"\b[xyz]mm(\d+)\b")


def objdump_tool(cc):
    cands = []
    p = shutil.which(cc)
    if p:
        for d in {os.path.dirname(p), os.path.dirname(os.path.realpath(p))}:
            cands.append(os.path.join(d, "llvm-objdump"))
    cands += [shutil.which("llvm-objdump") or "", shutil.which("objdump") or ""]
    for c in cands:
        if not c or not os.access(c, os.X_OK):
            continue
        v = subprocess.run([c, "--version"], capture_output=True, text=True)
        if v.returncode == 0 and "LLVM" in v.stdout:
            return c
    return None


def run(tool, *args):
    r = subprocess.run([tool, *args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"asm_regs_audit: {os.path.basename(tool)} {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def coff_functions(tool, obj):
    """per function in .text: (name, start, end, used vector numbers, used callee-saved gprs, saved xmm, saved gprs)"""
    syms, ins = [], []
    for ln in run(tool, "-d", "--no-show-raw-insn", "--x86-asm-syntax=intel", obj).splitlines():
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:$", ln)
        if m:
            syms.append((int(m.group(1), 16), m.group(2)))
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(.*)$", ln)
        if m:
            ins.append((int(m.group(1), 16), m.group(2)))
    addr = {name: a for a, name in syms}
    unwind, cur = {}, None
    for ln in run(tool, "-u", obj).splitlines():
        m = re.match(r"^\s*Start Address:\s*(\S+)(?:\s*\+\s*0x([0-9a-f]+))?", ln)
        if m:
            base = 0 if m.group(1).startswith(".") else addr.get(m.group(1), -1)
            cur = {"xmm": set(), "gpr": set()}
            unwind[base + int(m.group(2) or "0", 16)] = cur
            continue
        if cur is None:
            continue
        m = re.search(r"UOP_(PushNonVol|SaveNonVol\w*)\s+(\w+)", ln)
        if m:
            cur["gpr"].add(m.group(2).upper())
        m = re.search(r"UOP_SaveXMM128\w*\s+XMM(\d+)", ln)
        if m:
            cur["xmm"].add(int(m.group(1)))
    out = []
    syms.sort()
    for i, (start, name) in enumerate(syms):
        end = syms[i + 1][0] if i + 1 < len(syms) else 1 << 62
        vec, gpr = set(), set()
        for a, text in ins:
            if start <= a < end:
                vec |= {int(v) for v in VEC_RE.findall(text)}
                gpr |= {GPR[g] for g in GPR_RE.findall(text)}
        u = unwind.get(start, {"xmm": set(), "gpr": set()})
        out.append((name, vec, gpr, u["xmm"], u["gpr"]))
    return out


def audit_x86(asmcheck, cc):
    tool = objdump_tool(cc)
    if tool is None:
        sys.exit("asm_regs_audit: no llvm objdump (llvm-objdump beside " + cc + " or on PATH): the x86_64 rule needs one")
    bad, n, teeth = [], 0, False
    objs = sorted(glob.glob(os.path.join(asmcheck, COFF, "**", "*.o"), recursive=True))
    if not objs:
        sys.exit(f"asm_regs_audit: no coff objects under {os.path.join(asmcheck, COFF)}")
    for obj in objs:
        for name, vec, gpr, sx, sg in coff_functions(tool, obj):
            n += 1
            over = sorted(v for v in vec if 6 <= v <= 15 and v not in sx)
            ogpr = sorted(gpr - sg)
            if over or ogpr:
                bad.append(f"{os.path.relpath(obj, asmcheck)} {name}: names callee-saved "
                           f"{', '.join(['xmm%d' % v for v in over] + [g.lower() for g in ogpr])}; its win64 unwind codes "
                           f"save {', '.join(['xmm%d' % v for v in sorted(sx)] + [g.lower() for g in sorted(sg)]) or 'nothing'}")
            if name == "toks_k0_frame_x86":
                full = set(range(6, 16))
                teeth = vec >= full and sx >= full and sg >= set(GPR.values())
    if not teeth:
        bad.append("toks_k0_frame_x86 (selftest.S, PROLOGUE 6, 10) not seen using and saving xmm6-15 and the eight "
                   "callee-saved gprs: the objdump output is no longer read right")
    return n, bad


def strip(src):
    src = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def funcs(path):
    """(name, line, kind, args, epilogue args, body lines) per FUNC .. ENDFUNC"""
    cur = None
    for i, ln in enumerate(strip(open(path).read()).split("\n"), 1):
        m = re.match(r"\s*FUNC\s+(\w+)", ln)
        if m:
            cur = {"name": m.group(1), "line": i, "kind": None, "args": None, "epi": [], "body": []}
            continue
        if cur is None:
            continue
        if re.match(r"\s*ENDFUNC\b", ln):
            yield cur
            cur = None
            continue
        m = re.match(r"\s*(PROLOGUE|EPILOGUE)\s+(\d+)\s*,\s*(\d+)", ln)
        if m:
            a = (int(m.group(2)), int(m.group(3)))
            if m.group(1) == "PROLOGUE" and cur["kind"] is None:
                cur["kind"], cur["args"] = "framed", a
            elif m.group(1) == "EPILOGUE":
                cur["epi"].append(a)
            continue
        if re.match(r"\s*LEAF\b", ln) and cur["kind"] is None:
            cur["kind"] = "leaf"
            continue
        cur["body"].append(ln)


def audit_arm64(path):
    bad = []
    for f in funcs(path):
        body = "\n".join(f["body"])
        vec = {int(r) for r in re.findall(r"\b[vqdsbh](\d+)\b", body) if int(r) < 32}
        gpr = {int(r) for r in re.findall(r"\b[xw](\d+)\b", body)}
        where = f"{os.path.basename(path)}:{f['line']} {f['name']}"
        if f["kind"] == "leaf":
            v = sorted(r for r in vec if 8 <= r <= 15)
            g = sorted(r for r in gpr if 18 <= r <= 30)
            if v or g:
                bad.append(f"{where}: LEAF uses v{v} x{g}")
        elif f["kind"] == "framed":
            nx, nd = f["args"]
            v = sorted(r for r in vec if 8 + 2 * nd <= r <= 15)
            g = sorted(r for r in gpr if 19 + 2 * nx <= r <= 28 or r == 18)
            if v or g:
                bad.append(f"{where}: PROLOGUE {nx}, {nd} saves x19..x{18 + 2 * nx} d8..d{7 + 2 * nd}, uses v{v} x{g}")
            if any(e != f["args"] for e in f["epi"]):
                bad.append(f"{where}: EPILOGUE {f['epi']} != PROLOGUE {f['args']}")
        else:
            bad.append(f"{where}: neither LEAF nor PROLOGUE after FUNC")
    return bad


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: asm_regs_audit.py ASMCHECK_DIR CC")
    nx, bad = audit_x86(sys.argv[1], sys.argv[2])
    na = 0
    for path in sorted(glob.glob(os.path.join("src/asm/arm64", "*.S"))):
        na += sum(1 for _ in funcs(path))
        bad += audit_arm64(path)
    print(f"asm_regs_audit: {nx} x86_64 functions (coff: register use vs win64 unwind codes), {na} arm64 functions "
          f"(source: PROLOGUE / LEAF), {len(bad)} violations")
    for b in bad:
        print("  " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
