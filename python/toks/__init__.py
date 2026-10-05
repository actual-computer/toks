"""toks: Actual Computer's tokenizer for Python.

Exactly hf tokenizers 0.23.2's ids, from a small C library with asm kernels (this package links it in).

    import toks
    tok = toks.Tokenizer.from_file("tokenizer.json")   # or a model directory, or .from_str / .from_buffer
    ids = tok.encode("Hello world")                    # == hf Tokenizer.encode("Hello world").ids
    tok.decode(ids)                                    # == hf Tokenizer.decode(ids)

A Tokenizer is read-only and can be shared by any number of threads; encode_batch, and every call on a
text of 2 KiB or more, runs without the GIL. Errors from the library raise toks.Error, whose .code and
.name are the TOKS_E_* value and name of include/toks.h.
"""
from ._toks import ABI, ID_ADDED, ID_BYTE, ID_SPECIAL, MAX_TEXT, DecodeStream, Error, Tokenizer, __version__
__all__ = ["ABI", "ID_ADDED", "ID_BYTE", "ID_SPECIAL", "MAX_TEXT", "DecodeStream", "Error", "Tokenizer", "__version__"]
