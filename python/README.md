# toks for Python

The Python package of [toks](../README.md): a CPython extension module over the C ABI
([`include/toks.h`](../include/toks.h)), with the library linked in statically. Ids are exactly hf
`tokenizers` 0.23.2's; the API follows hf's `tokenizers.Tokenizer` so it drops in.

```python
import toks

tok = toks.Tokenizer.from_file("path/to/tokenizer.json")   # a model directory works too
ids = tok.encode("Hello world")                            # == hf tok.encode("Hello world").ids
tok.encode("Hello world", add_special_tokens=False)        # no bos / eos / template
tok.encode(text, added_tokens="nonspecial")                # == hf with encode_special_tokens=True
tok.encode(text, added_tokens="none")                      # no added token recognized
tok.encode_batch(texts)                                    # list of lists, without the GIL
n = tok.encode_into(text, out)                             # ids straight into a uint32 / int32 buffer
tok.encode_bound(len(text.encode()))                       # the most ids any text of that many bytes gives
tok.decode(ids)                                            # == hf tok.decode(ids) (skip_special_tokens=True)
tok.pieces(text)                                           # piece end offsets (utf-8 bytes)
tok.token_to_id("<|endoftext|>"), tok.id_to_token(50256), tok.token_bytes(50256)
tok.token_to_id(b" hello")                                 # bytes: the id they decode from (the C index)
tok.id_flags(50256) & toks.ID_SPECIAL                      # ID_ADDED / ID_SPECIAL / ID_BYTE per id
st = tok.decode_stream(); st.push(ids); st.flush()         # incremental decode
tok.info()                                                 # algorithm, tier, n_ids, sha256, ...
```

`token_to_id` takes two forms of a token:
- A `str` is hf's written form, read from the tokenizer.json, so a byte-level model's `"Ġhello"` works.
- `bytes` are what the id decodes to, looked up in the library's own index (`toks_token_to_id`), so the same token is
  `b" hello"`.

Don't pass an hf spelling's UTF-8 as bytes. It can name another valid id: gpt2's `"¢"` is id 95, but `"¢".encode()`
(C2 A2) is id 44359.

An added token's content is found either way: `"<s>"` and `b"<s>"` both give llama2's id 1.

`id_flags(id)` gives hf's `added_tokens_decoder` facts as bits:
- `ID_ADDED`: an added token holds the id.
- `ID_SPECIAL`: that token's own special flag, not the rule `skip_special_tokens` uses.
- `ID_BYTE`: a `<0xHH>` byte-fallback token.

An id beyond the table raises `toks.Error` (`TOKS_E_ID`).

Build (from the repository root; needs clang, make and [uv](https://github.com/astral-sh/uv)):

```sh
python3 tools/ci/fetch_tokenizers.py              # the pinned tokenizer files the tests read (else they SKIP)
uv build --wheel python --python 3.12 --out-dir build/wheels
uv run --no-project --python 3.12 --with build/wheels/toks-*-cp312-*.whl --with pytest --with tokenizers==0.23.2 \
    pytest python/tests                            # one wheel: pick your python's tag
```

`python/build.sh` builds and tests the wheels for every CPython on a machine. The wheels are attached to the GitHub
releases, not on PyPI yet; the license is BUSL-1.1 ([LICENSING.md](../LICENSING.md)).
