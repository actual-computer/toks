"""toks_oracle: the hf/tiktoken reference oracle for toks parity (SPEC §3, §13).

Loads an hf tokenizer.json (or tiktoken encoding) and produces, for any text:
  - ids in modes ALL / NONSPECIAL / NONE, each with post-processing on/off
  - pieces as end offsets in the NORMALIZED byte stream (SPEC §3.5)
  - decode with/without skip_special_tokens
  - a stream-decode reference (incremental lossy utf-8, U+FFFD per maximal subpart)
  - a tiktoken oracle (cl100k_base / o200k_base)

Run tests:  uv run --with tokenizers==0.23.2 --with tiktoken --with pytest \\
              pytest tests/parity/test_oracle.py -q
"""
from .oracle import (
    AddedToken,
    ModeError,
    MODES,
    TikOracle,
    Tok,
    TokSpec,
    Trie,
    download,
    find_matches,
    held,
    load,
    load_cached,
    load_synthetic,
    stream_reference,
    stream_steps,
)

__all__ = [
    "AddedToken",
    "ModeError",
    "MODES",
    "TikOracle",
    "Tok",
    "TokSpec",
    "Trie",
    "download",
    "find_matches",
    "held",
    "load",
    "load_cached",
    "load_synthetic",
    "stream_reference",
    "stream_steps",
]
