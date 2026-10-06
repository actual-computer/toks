#!/usr/bin/env python3
"""tests/abi/cf_audit.py: SPEC §9's strict subset and §4.1's after-load rule, checked on the BUILT objects (make test,
beside the abi lint), because the source can be clean and the object not: clang's loop-idiom pass turns a counting
loop into strlen, memcmp == 0 into bcmp, a page-sized frame into a stack-probe call. Read: every object under
BUILD_DIR/obj (the library's real link inputs for this target: src/core, src/gen, src/platform, src/par,
src/asm/<isa>) and under BUILD_DIR/asmcheck (the asm of both isas as mach-o, elf and coff), through llvm-objdump -d -r
and llvm-nm -P. The rules (rulings of 2026-10-05):

  R1 indirect call   no call through a register or memory operand (SPEC §0.5: dispatch is a switch compiled to direct
                     calls; §9: no function pointers; §10.4: direct calls only). Darwin's stack probe (a frame of a
                     page or more calls ___chkstk_darwin through x16) is the compiler's, not a pointer of the source.
                     A jump through a table (a switch) is not a §9 item: counted per object, reported, not gated.
  R2 address taken   no function's address is used but by a branch: a pointer-sized relocation from data, or any
                     relocation from a non-branch instruction, to a function's entry (§11: no address-taken function,
                     so no guard-cf table). toks_par's thread entry, which the os calls, is the one named exception.
  R3 libc            the c core (src/core, src/gen) calls no external symbol but memcpy, memset and memcmp (§9),
                     Darwin's memset lowerings (bzero from the backend, memset_pattern16 from loop idioms) and the
                     compiler's runtime (stack probe, stack protector); everything else it calls is defined in the
                     library. The Makefile's -fno-builtin-strlen and -fno-builtin-bcmp keep clang from writing
                     strlen calls the source never made and from spelling memcmp == 0 as bcmp (linux).
  R4 recursion       no cycle in the c core's static call graph (§9: the json reader and the regex compiler use
                     explicit bounded stacks). A cycle in ALLOW is a dated debt with its bound and the PR removing it.
  R5 dynamic stack   no stack-pointer move by a register (a vla, alloca: §9) but a fixed frame's probe.
  R6 after load      from the entry points a loaded tokenizer is used through (ENTRY_AFTER_LOAD), nothing reaches an
                     external symbol but memcpy / memset / memcmp, nor the platform layer (§4.1: no allocation,
                     syscall, lock, recursion or callback in the core after load), but the one named exception.
  R7 writable data   the c core (src/core, src/gen) has no writable data: no byte in a data, bss, common or
                     thread-local section (toks.h: no global is written; a context is read-only after load, so one
                     context serves every thread). Read-only data is fine: .rodata, mach-o __const (also
                     __DATA,__const, which only the loader's relocations write) and elf .data.rel.ro.

Information (no gate): jump tables per object; stack frames of 4 KiB or more (§7.2's 4 KiB is the run-time chain:
a frame on the after-load path is marked so). Teeth: tests/abi/cf_teeth.c, compiled with the library's own flags (the
CC and flags after BUILD_DIR), breaks each rule exactly once and has one jump table; the audit must report exactly
that, else it fails (an objdump whose output it no longer reads must not pass everything). Needs python3 and an LLVM
objdump and nm (beside CC, on PATH, or the system's when they are LLVM's, as on macos); without them it fails, never
skips (asm_regs_audit.py's rule).

  cf_audit.py BUILD_DIR CC [CFLAGS...]      (from the repository root; prints the report, exit 1 on any violation)
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

# ---- what is permitted: (rule, function, detail) -> (date, reason) ---------------------------------------------
# R2 detail: the layer of the object taking the address. R4 detail: the cycle's functions, sorted, ' -> '-joined.
# R6 detail: the platform function or external symbol reached.
ALLOW = {
    ("R2", "thr_main", "par"): ("2026-10-05",
        "src/par/par.c: the thread entry handed to pthread_create / CreateThread; toks_par is outside the proof "
        "boundary (SPEC §2.4): the os calls it, toks never calls through it"),
    ("R4", "rx_node", "rx_node"): ("2026-10-05",
        "compile.c, the K1 radix tree's build (load time): depth <= 255, an added token's bytes. Debt: removed by "
        "the explicit-stack PR (toks/hardening-norec)"),
    ("R4", "parse_alt", "parse_alt"): ("2026-10-05",
        "gen.c, the generic regex compiler (load time): depth <= 32 groups (r->depth). Debt: removed by the "
        "explicit-stack PR (toks/hardening-norec)"),
    ("R4", "cls_items", "cls_items"): ("2026-10-05",
        "gen.c, a class inside a class (load time): depth <= 8 (r->depth). Debt: removed by the explicit-stack PR "
        "(toks/hardening-norec)"),
    ("R4", "add", "add -> search"): ("2026-10-05",
        "gen.c, the Pike VM's closure and a lookahead's search (after load): depth 2, a lookahead inside a "
        "lookahead is refused at compile. Debt: removed by the explicit-stack PR (toks/hardening-norec)"),
    ("R4", "elem", "elem -> piece"): ("2026-10-05",
        "gen.c, the generic program's nested steps (after load): depth <= 16 steps (toks_gen step[16]). Debt: "
        "removed by the explicit-stack PR (toks/hardening-norec)"),
    ("R6", "toks_scratch_init", "toks_plat_hint_huge"): ("2026-10-05",
        "madvise(MADV_HUGEPAGE) on the caller's scratch: the first init of a scratch only, only when "
        "TOKS_SCRATCH_CACHE_MIB(n) != 0, never on the default scratch; a syscall after load, tied to decision 8 "
        "(pending)"),
}

ENTRY_AFTER_LOAD = ("toks_encode toks_pieces toks_split_points toks_decode toks_token toks_stream_init toks_stream_push "
                    "toks_stream_flush toks_stream_bound toks_stream_hold toks_token_to_id toks_id_flags "
                    "toks_encode_bound toks_scratch_bytes toks_scratch_init toks_get_info toks_version").split()
TEETH_ENTRY = ["cft_entry"]
LIBC_OK = {"memcpy", "memset", "memcmp"}
LOWERED = {"bzero": "memset", "__bzero": "memset", "memset_pattern16": "memset"}   # Darwin's; bcmp is not (-fno-builtin-bcmp)
RUNTIME = {"__chkstk_darwin", "__chkstk", "__stack_chk_fail", "__stack_chk_guard"}
CORE = ("core", "gen", "teeth")
SKIP_SECTIONS = re.compile(r"^(\.rela?)?(\.eh_frame|\.debug|__debug|__compact_unwind|__eh_frame|\.pdata|\.xdata|"
                           r"\.llvm_addrsig|\.note|\.gnu|\.comment|\.llvm\.|__LD|__DWARF|\.drectve)")
ARM_IND = {"br", "blr", "braa", "braaz", "brab", "brabz", "blraa", "blraaz", "blrab", "blrabz"}
ARM_BRANCH = {"b", "bl", "cbz", "cbnz", "tbz", "tbnz"}
PREFIX = {"notrack", "bnd", "rep", "repne", "lock", "data16"}
# R7: the sections a program writes (elf, coff, and mach-o by the section name objdump -h prints, without its segment)
WRITABLE = re.compile(r"^(?:\.(?:data(?!\.rel\.ro)|bss|tdata|tbss|tls)(?:[.$].*)?|__(?:data|bss|common|thread_data|"
                      r"thread_bss|thread_vars))$")


def wkind(sec):
    """a writable section's kind: data, bss (common included) or tls"""
    s = sec.split(",")[-1]
    if re.match(r"^(\.t(?:data|bss)|\.tls|__thread)", s):
        return "tls"
    return "bss" if re.match(r"^(\.bss|__bss|__common|\*COM\*)", s) else "data"


def llvm_tool(cc, name):
    cands = []
    p = shutil.which(cc)
    if p:
        for d in {os.path.dirname(p), os.path.dirname(os.path.realpath(p))}:
            cands += [os.path.join(d, "llvm-" + name), os.path.join(d, "llvm-" + name + ".exe")]
    cands += [shutil.which("llvm-" + name) or "", shutil.which(name) or ""]
    for c in cands:
        if c and os.access(c, os.X_OK):
            v = subprocess.run([c, "--version"], capture_output=True, text=True)
            if v.returncode == 0 and "LLVM" in v.stdout:
                return c
    return None


def run(tool, *args):
    r = subprocess.run([tool, *args], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        sys.exit(f"cf_audit: {os.path.basename(tool)} {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


class Insn:
    __slots__ = ("addr", "func", "mn", "ops", "cmt", "rels")

    def __init__(self, addr, func, mn, ops, cmt):
        self.addr, self.func, self.mn, self.ops, self.cmt, self.rels = addr, func, mn, ops, cmt, []


class Obj:
    """one object file: its functions (entry -> name), instructions with their relocations, data relocations"""

    def __init__(self, path, layer, tools):
        objdump, nm = tools
        self.path, self.layer = path, layer
        hdr = run(objdump, "-h", path)
        m = re.search(r"file format (.+)$", hdr, re.M)
        self.fmt = m.group(1).strip() if m else "?"
        self.macho = "mach-o" in self.fmt
        self.arch = "arm64" if re.search(r"aarch64|arm64", self.fmt) else "x86_64"
        self.wsec = {}                                                     # R7: writable section -> bytes
        for ln in hdr.splitlines():
            m = re.match(r"^\s*\d+\s+(\S+)\s+([0-9a-fA-F]+)\s", ln)
            if m and int(m.group(2), 16) > 0 and WRITABLE.match(m.group(1)):
                self.wsec[m.group(1)] = self.wsec.get(m.group(1), 0) + int(m.group(2), 16)
        self.wsym = []                                                     # R7: (section, symbol) of written data
        for ln in run(objdump, "-t", path).splitlines():
            f = ln.split()
            if len(f) < 3 or not re.match(r"^[0-9a-fA-F]+$", f[0]):
                continue
            sec = next((t for t in f[1:-1] if t == "*COM*" or (t.startswith((".", "__")) and
                        t not in (".hidden", ".protected", ".internal"))), None)
            if sec is not None and (sec == "*COM*" or sec.split(",")[-1] in self.wsec):
                self.wsym.append((sec.split(",")[-1], self.norm(f[-1])))
        self.text, self.glob, self.undef, self.data = {}, set(), set(), set()
        for ln in run(nm, "-P", path).splitlines():
            f = ln.split()
            if len(f) < 3:
                continue
            name, t = self.norm(f[0]), f[1]
            if t == "U":
                self.undef.add(name)
            elif t in "tT":
                if not name.startswith(("$", "ltmp", ".L", "L")):
                    self.text.setdefault(int(f[2], 16), name)
                    if t == "T":
                        self.glob.add(name)
            else:
                self.data.add(name)
        self.funcs = set(self.text.values())
        self.starts = sorted(self.text)
        self.insns = []
        args = ["-d", "-r", "--no-show-raw-insn"] + (["--x86-asm-syntax=intel"] if self.arch == "x86_64" else [])
        for ln in run(objdump, *args, path).splitlines():
            if ln.startswith("\t\t"):
                m = re.match(r"^\t\t[0-9a-f]+:\s+(\S+)\s+(\S+)", ln)
                if m and self.insns:
                    self.insns[-1].rels.append((m.group(1), self.target(m.group(2))))
                continue
            m = re.match(r"^\s*([0-9a-f]+):\s+(\S+)(?:\s+(.*))?$", ln)
            if not m:
                continue
            mn, ops = m.group(2), (m.group(3) or "").strip()
            while mn in PREFIX and ops:
                mn, _, ops = ops.partition(" ")
                ops = ops.strip()
            ops, _, cmt = ops.partition(";" if self.arch == "arm64" else "#")   # arm64 immediates are #n
            addr = int(m.group(1), 16)
            self.insns.append(Insn(addr, self.func_at(addr), mn, ops.strip(), cmt.strip()))
        self.drel = []
        sec = None
        for ln in run(objdump, "-r", path).splitlines():
            m = re.match(r"^RELOCATION RECORDS FOR \[(.*)\]:", ln)
            if m:
                sec = m.group(1)
                continue
            m = re.match(r"^([0-9a-f]+)\s+(\S+)\s+(\S+)", ln)
            if m and sec and not self.is_text(re.sub(r"^\.rela?(?=\.)", "", sec)) and not SKIP_SECTIONS.match(sec):
                self.drel.append((sec, int(m.group(1), 16), m.group(2), self.target(m.group(3))))

    @staticmethod
    def is_text(sec):
        return sec in (".text", "__text", "__TEXT,__text") or sec.startswith(".text.")

    def norm(self, name):
        return name[1:] if self.macho and name.startswith("_") else name

    def target(self, t):
        t = re.sub(r"@\w+", "", t)
        m = re.match(r"^(.*?)(?:([+-])0x([0-9a-f]+))?$", t)
        base, sign, off = m.group(1), m.group(2), m.group(3)
        return self.norm(base), ((-1 if sign == "-" else 1) * int(off, 16) if off else 0)

    def func_at(self, addr):
        best = None
        for s in self.starts:
            if s > addr:
                break
            best = s
        return self.text[best] if best is not None else "?"


def is_branch(o, mn):
    if o.arch == "arm64":
        return mn in ARM_BRANCH or mn.startswith("b.")
    return mn in ("call", "jmp") or (mn.startswith("j") and len(mn) <= 4)


def is_call(o, mn):
    return mn.startswith("bl") if o.arch == "arm64" else mn == "call"


def is_indirect(o, mn, ops):
    if o.arch == "arm64":
        return mn in ARM_IND
    return mn in ("call", "jmp") and not re.match(r"^0x[0-9a-f]+(\s+<[^>]*>)?$", ops)


def probe_ref(b):
    return any(t[0] in ("__chkstk_darwin", "__chkstk") for _, t in b.rels)


def probe_near(o, i):
    """the instruction at i is a stack probe's call: a reference to __chkstk(_darwin) in it or the 3 before it"""
    return any(probe_ref(b) for b in o.insns[max(0, i - 3):i + 1])


def probe_size(o, i):
    """the probe called at i covers a fixed frame: its bytes from the immediate the compiler loads before the call
    (mach-o arm64: x15 in 16-byte units, or x9 in bytes for an over-aligned frame; coff x86-64: eax in bytes), else
    None (a probe of a vla / alloca, whose size is computed)"""
    n, reg = None, None
    for b in o.insns[max(0, i - 6):i]:
        m = re.match(r"^([xw](?:9|15)|[er]ax),\s*#?(0x[0-9a-f]+|\d+)(?:,\s*lsl\s*#(\d+))?$", b.ops)
        if m and b.mn in ("mov", "movz", "movk"):
            v = int(m.group(2), 0) << int(m.group(3) or 0)
            n, reg = (v if b.mn != "movk" else (n or 0) | v), m.group(1)[1:]
    if n is None:
        return None
    return n * 16 if reg == "15" else n


def audit(objs, entries):
    """(violations [(rule, obj, func, detail, text)], jump tables {obj path: n}, frames [(obj, func, bytes)],
    ALLOW keys used, the (obj path, func) set reachable from entries)"""
    viol, used, jumps, frames = [], set(), {}, []
    defined = {}
    for o in objs:
        for name in o.glob:
            defined.setdefault(name, (o, name))
    datas = set().union(*(o.data for o in objs))
    byp = {o.path: o for o in objs}

    def resolve(o, name):
        if name in o.funcs and name not in o.glob:
            return (o.path, name)
        d = defined.get(name)
        return (d[0].path, d[1]) if d else None

    def check(rule, func, detail, o, text):
        if (rule, func, detail) in ALLOW and o.layer != "teeth":
            used.add((rule, func, detail))
        else:
            viol.append((rule, o, func, detail, text))

    edges, fsize, fprobe, probe_at = {}, {}, {}, {}
    for o in objs:
        for f in o.funcs:
            edges.setdefault((o.path, f), set())
        for i, n in enumerate(o.insns):
            node = (o.path, n.func)
            edges.setdefault(node, set())
            if is_indirect(o, n.mn, n.ops):                               # R1
                if probe_near(o, i):
                    size = probe_size(o, i)
                    probe_at[node] = (i, size is not None)
                    fprobe[node] = max(fprobe.get(node, 0), size or 0)
                elif is_call(o, n.mn) or any(t[0] in o.undef for _, t in n.rels):
                    check("R1", n.func, "call", o, f"{n.mn} {n.ops}")
                else:
                    jumps[o.path] = jumps.get(o.path, 0) + 1
                continue
            if is_call(o, n.mn) and probe_ref(n):                          # coff: `call __chkstk`
                size = probe_size(o, i)
                probe_at[node] = (i, size is not None)
                fprobe[node] = max(fprobe.get(node, 0), size or 0)
            branch = is_branch(o, n.mn)
            for rtype, (tname, toff) in n.rels:
                if branch:                                                 # a call or a tail call
                    tgt = resolve(o, tname)
                    edges[node].add(tgt if tgt else ("ext", tname))
                elif tname not in RUNTIME and tname not in datas:         # R2: an address in a register
                    tgt = resolve(o, tname)
                    if tgt is not None and toff == 0:
                        check("R2", tname, o.layer, o, f"{n.func}: {n.mn} {n.ops} ({rtype} {tname})")
                    elif tgt is None and tname in o.undef:
                        viol.append(("R2", o, n.func, tname, f"{n.mn} {n.ops}: {rtype} to external {tname}"))
            if not n.rels:
                for a, _ in re.findall(r"0x([0-9a-f]+)\s+<([^>+]+)>", n.ops + " " + n.cmt):
                    a = int(a, 16)
                    if a not in o.text:
                        continue
                    tf = o.text[a]
                    if branch:
                        if tf != n.func or is_call(o, n.mn):
                            edges[node].add((o.path, tf))
                    else:
                        check("R2", tf, o.layer, o, f"{n.func}: {n.mn} {n.ops}")
            if o.arch == "arm64":                                          # R5 and the frame's size
                dyn = (n.mn in ("sub", "add") and re.match(r"^sp,\s*sp,\s*[xw]\d+", n.ops)) or \
                    (n.mn == "mov" and re.match(r"^sp,\s*x\d+$", n.ops) and not n.ops.endswith("x29"))
                m = re.match(r"^sp,\s*sp,\s*#(0x[0-9a-f]+|\d+)(?:,\s*lsl\s*#(\d+))?$", n.ops) if n.mn == "sub" else \
                    re.search(r"\[sp,\s*#-(0x[0-9a-f]+|\d+)\]!$()", n.ops) if n.mn in ("stp", "str") else None
            else:
                dyn = (n.mn in ("sub", "add") and re.match(r"^rsp,\s*[re]\w+$", n.ops)) or \
                    (n.mn == "mov" and re.match(r"^rsp,\s*[re]\w+$", n.ops) and not n.ops.endswith("bp"))
                m = re.match(r"^rsp,\s*(0x[0-9a-f]+|\d+)()$", n.ops) if n.mn == "sub" else None
            if m:
                fsize[node] = fsize.get(node, 0) + (int(m.group(1), 0) << int(m.group(2) or 0))
            if dyn:
                p = probe_at.get(node)
                if p is None or not p[1] or i - p[0] > 6:
                    viol.append(("R5", o, n.func, "dynamic stack", f"{n.mn} {n.ops}"))
        for sec, off, rtype, (tname, toff) in o.drel:                     # R2 from data: pointer-sized absolute
            if not re.search(r"_64$|ABS64|ADDR64|UNSIGNED", rtype):       # (pc-relative 32: a jump table's entry)
                continue
            tgt = resolve(o, tname)
            fn = tname if tgt is not None and toff == 0 else (o.text.get(toff) if o.is_text(tname) else None)
            if fn is not None:
                check("R2", fn, o.layer, o, f"[{sec} + {off:#x}] {rtype} {tname}")
    for (p, func) in set(fsize) | set(fprobe):
        size = max(fsize.get((p, func), 0), fprobe.get((p, func), 0))      # a probe only touches; immediates move sp
        if size >= 4096:
            frames.append((byp[p], func, size))

    lib = set(defined) | datas                                             # R3
    for o in objs:
        if o.layer not in CORE:
            continue
        for sym in sorted(o.undef - lib - LIBC_OK - set(LOWERED) - RUNTIME):
            callers = sorted({n.func for n in o.insns for _, (t, _) in n.rels if t == sym})
            viol.append(("R3", o, ",".join(callers) or "-", sym, f"calls {sym} (§9: libc is memcpy / memset / memcmp)"))

    for o in objs:                                                         # R7: writable data, by section
        if o.layer not in CORE:
            continue
        named = {}
        for sec, sym in o.wsym:
            if not sym.startswith(("$", "ltmp", ".L", "L", ".", "l_")):    # labels and section symbols
                named.setdefault(sec, set()).add(sym)
        for sec in sorted(set(o.wsec) | {s for s, _ in o.wsym if s == "*COM*"}):
            for sym in sorted(named.get(sec, ())) or ["-"]:
                viol.append(("R7", o, sym, wkind(sec), f"{sec} ({wkind(sec)}, {o.wsec.get(sec, 0)} bytes): a writable "
                                                       f"global (toks.h: no global is written)"))

    core = {v for v in edges if byp[v[0]].layer in CORE}                   # R4: Tarjan over the c core
    index, low, stack, on, counter = {}, {}, [], set(), [0]
    for root in sorted(core):
        if root in index:
            continue
        work = [(root, iter(sorted(e for e in edges[root] if e in core)))]
        index[root] = low[root] = counter[0]; counter[0] += 1; stack.append(root); on.add(root)
        while work:
            v, it = work[-1]
            pushed = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]; counter[0] += 1; stack.append(w); on.add(w)
                    work.append((w, iter(sorted(e for e in edges[w] if e in core))))
                    pushed = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if pushed:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop(); on.discard(w); comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1 or v in edges[v]:
                    names = sorted(f for _, f in comp)
                    check("R4", names[0], " -> ".join(names), byp[comp[0][0]], "a call cycle: " + " -> ".join(names))

    seen, todo = set(), [(defined[e][0].path, e) for e in entries if e in defined]   # R6
    while todo:
        v = todo.pop()
        if v in seen:
            continue
        seen.add(v)
        for x in edges.get(v, ()):
            if x[0] == "ext":
                if x[1] not in LIBC_OK and x[1] not in LOWERED and x[1] not in RUNTIME:
                    check("R6", v[1], x[1], byp[v[0]], f"after load, {v[1]} calls {x[1]}")
            elif byp[x[0]].layer == "platform":
                check("R6", v[1], x[1], byp[v[0]], f"after load, {v[1]} calls the platform's {x[1]}")
            else:
                todo.append(x)
    return viol, jumps, frames, used, seen


TEETH_EXPECT = {("R1", "cft_call", "call"), ("R2", "cft_leaf", "teeth"), ("R3", "cft_libc", "strlen"),
                ("R3", "cft_alloc", "malloc"), ("R3", "cft_entry", "getenv"), ("R4", "cft_rec", "cft_rec"),
                ("R5", "cft_vla", "dynamic stack"), ("R6", "cft_entry", "getenv"), ("R7", "cft_table", "data"),
                ("R7", "cft_sink", "bss")}


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: cf_audit.py BUILD_DIR CC [CFLAGS...]")
    bdir, cc, cflags = sys.argv[1], sys.argv[2], sys.argv[3:]
    tools = (llvm_tool(cc, "objdump"), llvm_tool(cc, "nm"))
    if None in tools:
        sys.exit(f"cf_audit: no llvm objdump / nm (llvm-objdump and llvm-nm beside {cc} or on PATH): it needs both")
    src = {k: {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"src/{k}/*.c")}
           for k in ("core", "gen", "platform", "par")}
    asm = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob("src/asm/*/*.S")}
    lib = []
    for p in sorted(glob.glob(os.path.join(bdir, "obj", "**", "*.o"), recursive=True)):
        stem = re.sub(r"(\.S)?\.o$", "", os.path.basename(p))
        layer = "asm" if stem in asm else next((k for k in ("core", "gen", "platform", "par") if stem in src[k]), None)
        if layer is not None:
            lib.append(Obj(p, layer, tools))
    if not lib:
        sys.exit(f"cf_audit: no library objects under {os.path.join(bdir, 'obj')}")
    other = [Obj(p, "asm", tools) for p in sorted(glob.glob(os.path.join(bdir, "asmcheck", "**", "*.o"), recursive=True))
             if "/src/asm/" in p.replace("\\", "/")]
    viol, jumps, frames, used, after = audit(lib, ENTRY_AFTER_LOAD)
    for o in other:                                                         # the other formats' asm, one at a time
        v, j, f, u, _ = audit([o], [])
        viol, frames, used = viol + v, frames + f, used | u
        jumps.update(j)
    with tempfile.TemporaryDirectory() as td:
        to = os.path.join(td, "cf_teeth.o")
        r = subprocess.run([cc, *cflags, "-c", "tests/abi/cf_teeth.c", "-o", to], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"cf_audit: tests/abi/cf_teeth.c does not compile: {r.stderr.strip()}")
        tv, tj, _, _, _ = audit([Obj(to, "teeth", tools)], TEETH_ENTRY)
        got = {(v[0], v[2], v[3]) for v in tv}
    missing, extra = TEETH_EXPECT - got, got - TEETH_EXPECT

    fmts = sorted({o.fmt for o in lib + other})
    print(f"cf_audit: {len(lib)} library objects ({lib[0].fmt}), {sum(len(o.funcs) for o in lib)} functions, "
          f"{len(after)} reachable after load; {len(other)} asmcheck objects ({', '.join(fmts)}); "
          f"{len(viol)} violations, {len(used)} allowed")
    for rule, o, func, detail, text in sorted(viol, key=lambda v: (v[0], v[1].path, v[2], v[4])):
        print(f"  {rule} {os.path.relpath(o.path, bdir)} {func}: {text}")
    for key in sorted(used):
        print(f"  allowed {key[0]} {key[1]} ({key[2]}): {ALLOW[key][1]} [{ALLOW[key][0]}]")
    for key in sorted(set(ALLOW) - used):
        print(f"  note: ALLOW {key} matched nothing in this build")
    if jumps:
        print("  jump tables (information, not a §9 item): " +
              ", ".join(f"{os.path.basename(p)} {n}" for p, n in sorted(jumps.items())))
    for o, func, size in sorted(frames, key=lambda f: -f[2]):
        tag = "AFTER LOAD (§7.2: the run-time chain <= 4 KiB)" if (o.path, func) in after else "load time"
        print(f"  frame: {os.path.relpath(o.path, bdir)} {func}: {size} bytes, {tag}")
    bad = bool(viol)
    if missing or extra or sum(tj.values()) != 1:
        bad = True
        print(f"  teeth: tests/abi/cf_teeth.c must give exactly {sorted(TEETH_EXPECT)} and one jump table; missing "
              f"{sorted(missing)}, unexpected {sorted(extra)}, {sum(tj.values())} jump tables: the objects are no "
              f"longer read right")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
