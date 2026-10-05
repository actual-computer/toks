# /// script
# requires-python = ">=3.10"
# ///
"""tools/release/rc_rent.py <asm e2e log> <scalar e2e log>: the cells where the asm tier was slower than the c twin
(cold or pass) in two separate tools/bench/e2e.sh runs, one line each: "<tier> <tokenizer> <corpus> <chunk>".
tools/release/rc_host.sh's rent step re-measures exactly these cells with tools/bench/e2e_ab.sh (abba, one build);
tools/release/rc_report.py decides the rent rule by those paired medians where they exist."""
import importlib.util
import os
import sys


def main():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "e2e_table.py")
    spec = importlib.util.spec_from_file_location("e2e_table", p)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    asm = [T.Cell("asm", "", c) for c in T.parse(sys.argv[1])[1] if "toks" in c]
    twin = {(x.tk, x.corp, x.chunk): x for x in (T.Cell("c", "", c) for c in T.parse(sys.argv[2])[1] if "toks" in c)}
    for x in asm:
        y = twin.get((x.tk, x.corp, x.chunk))
        if y is not None and (x.s["cold"] > y.s["cold"] or x.s["pass"] > y.s["pass"]):
            print(x.tier, x.tk, x.corp, x.chunk)


if __name__ == "__main__":
    main()
