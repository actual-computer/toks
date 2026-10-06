"""toks: Actual Computer's tokenizer for Python.

Exactly hf tokenizers 0.23.2's ids, from a small C library with asm kernels (this package links it in).

    import toks
    tok = toks.Tokenizer.from_file("tokenizer.json")   # or a model directory, or .from_str / .from_buffer
    ids = tok.encode("Hello world")                    # == hf Tokenizer.encode("Hello world").ids
    enc = tok.encode_ex("Hello world")                 # == hf Tokenizer.encode(...): ids, attention_mask, ...
    tok.decode(ids)                                    # == hf Tokenizer.decode(ids)
    tok.encode(text, allowed_special="all")            # tiktoken's Encoding.encode (and n_vocab, eot_token, ...)

A Tokenizer is read-only and can be shared by any number of threads; encode_batch, and every call on a
text of 2 KiB or more, runs without the GIL. Errors from the library raise toks.Error, whose .code and
.name are the TOKS_E_* value and name of include/toks.h.
"""
from ._toks import (ABI, ID_ADDED, ID_BYTE, ID_SPECIAL, MAX_TEXT, DecodeStream, Encoding, Error, Tokenizer,
                    __version__)
from ._vocab import AddedToken
__all__ = ["ABI", "ID_ADDED", "ID_BYTE", "ID_SPECIAL", "MAX_TEXT", "AddedToken", "DecodeStream", "Encoding", "Error",
           "Tokenizer", "__version__"]
