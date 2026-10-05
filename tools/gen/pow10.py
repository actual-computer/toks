#!/usr/bin/env python3
"""toks: serde_json's POW10 table (1e0 .. 1e308, each the double nearest its decimal literal), for unigram.c's score
parser (docs/algorithms/unigram.md §2: serde_json 1.0.151 without float_roundtrip multiplies or divides the
significand by POW10[|e|]). The C compiler rounds each literal exactly as rust does.

    uv run tools/gen/pow10.py      -> src/gen/pow10.c
"""
from pathlib import Path

out = Path(__file__).resolve().parents[2] / "src/gen/pow10.c"
lits = [f"1e{k}" for k in range(309)]
lines = ["/* toks: serde_json 1.0.151's POW10 (de.rs), 1e0 .. 1e308.  GENERATED FILE -- DO NOT EDIT.",
         " * Regenerate with: uv run tools/gen/pow10.py",
         " */",
         "",
         "const double toks_pow10[309] = {"]
row = "   "
for lit in lits:
    if len(row) + len(lit) + 2 > 116:
        lines.append(row)
        row = "   "
    row += " " + lit + ","
lines.append(row)
lines.append("};")
out.write_text("\n".join(lines) + "\n")
print(out)
