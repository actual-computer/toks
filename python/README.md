# toks for Python

The Python package of [toks](../README.md): a CPython extension module over the C ABI
([`include/toks.h`](../include/toks.h)), with the library linked in statically. Ids are exactly hf
`tokenizers` 0.23.2's; the API follows hf's `tokenizers.Tokenizer` so it drops in, and the same object answers
tiktoken's `Encoding` calls.

```python
import toks

tok = toks.Tokenizer.from_file("path/to/tokenizer.json")   # a model directory works too
ids = tok.encode("Hello world")                            # == hf tok.encode("Hello world").ids
tok.encode("Hello world", add_special_tokens=False)        # no bos / eos / template
tok.encode(text, added_tokens="nonspecial")                # == hf with encode_special_tokens=True
tok.encode(text, added_tokens="none")                      # no added token recognized
enc = tok.encode_ex(text)                                  # == hf tok.encode(text): an Encoding
enc.ids, enc.attention_mask, enc.type_ids, enc.special_tokens_mask, enc.tokens
tok.encode_batch(texts)                                    # list of lists, without the GIL, padded as hf pads
tok.encode_batch_ex(texts)                                 # == hf tok.encode_batch(texts): a list of Encoding
n = tok.encode_into(text, out)                             # ids straight into a uint32 / int32 buffer
tok.encode_bound(len(text.encode()))                       # the most ids any text of that many bytes gives
tok.decode(ids)                                            # == hf tok.decode(ids) (skip_special_tokens=True)
tok.pieces(text)                                           # piece end offsets (utf-8 bytes)
tok.token_to_id("<|endoftext|>"), tok.id_to_token(50256), tok.token_bytes(50256)
tok.token_to_id(b" hello")                                 # bytes: the id they decode from (the C index)
tok.id_flags(50256) & toks.ID_SPECIAL                      # ID_ADDED / ID_SPECIAL / ID_BYTE per id
tok.num_special_tokens_to_add(), tok.get_added_tokens_decoder()
st = tok.decode_stream(); st.push(ids); st.flush()         # incremental decode
tok.info()                                                 # algorithm, tier, n_ids, sha256, truncation, padding, ...

tok.encode(text, allowed_special="all")                    # tiktoken's Encoding.encode
tok.encode_ordinary(text), tok.decode_bytes(ids)           # tiktoken's
tok.n_vocab, tok.eot_token, tok.special_tokens_set, tok.token_byte_values()
```

## The API and what each call equals

Every call below is checked against the reference named beside it by python/tests (through the built wheel), on
every tokenizer.json under tests/data and in the tokenizer cache, and on the tiktoken models.

| call | gives | equals |
|---|---|---|
| `Tokenizer.from_file(path, *, tier, cache_mib)`, `from_str(json)`, `from_buffer(data)` | a tokenizer | (toks_load) |
| `encode(text, *, add_special_tokens=True, added_tokens=None, continuation=False)` | `list[int]` | hf `Tokenizer.encode(text).ids` |
| `encode_ex(text, *, add_special_tokens, added_tokens, continuation)` | `Encoding` | hf `Tokenizer.encode(text)`: `ids`, `type_ids`, `tokens`, `attention_mask`, `special_tokens_mask`, `n_sequences`, `len()` |
| `encode_batch(texts, ...)` | `list[list[int]]` | hf `Tokenizer.encode_batch(texts)`'s ids: the file's padding over the batch (BatchLongest: to its longest member) |
| `encode_batch_ex(texts, ...)` | `list[Encoding]` | hf `Tokenizer.encode_batch(texts)`, field by field |
| `encode_into(text, out, ...)` | the count | `encode`'s ids, written into `out` |
| `encode_bound(n)` | `int` | toks_encode_bound |
| `pieces(text, *, added_tokens, continuation)` | `list[int]` | hf's normalizer + pre-tokenizer, as byte offsets |
| `decode(ids, skip_special_tokens=True)`, `decode_batch(...)` | `str` | hf `Tokenizer.decode` |
| `decode_stream(skip_special_tokens=False)` | `DecodeStream` | hf `decode` of all the ids pushed |
| `num_special_tokens_to_add(is_pair=False)` | `int` | hf `num_special_tokens_to_add(False)` (toks_template); a pair raises `toks.Error` |
| `get_added_tokens_decoder()` | `dict[int, AddedToken]` | hf `get_added_tokens_decoder()`, every field (toks_added) |
| `token_to_id(token)`, `id_to_token(id)`, `get_vocab(...)`, `get_vocab_size(...)` | | hf's, read from the tokenizer.json |
| `token_bytes(id)`, `id_flags(id)`, `info()` | | toks_token, toks_id_flags, toks_get_info |
| `encode(text, *, allowed_special=set(), disallowed_special="all")` | `list[int]` | tiktoken 0.14.0 `Encoding.encode`, its `ValueError` included |
| `encode_ordinary(text)` | `list[int]` | tiktoken `encode_ordinary` |
| `decode_bytes(ids)`, `decode_bytes_batch(batch)` | `bytes` | tiktoken `decode_bytes` (TOKS_DECODE_RAW; an id beyond the table is its `KeyError`) |
| `n_vocab`, `max_token_value`, `eot_token`, `special_tokens_set`, `token_byte_values()`, `name` | | tiktoken's (`eot_token`: `KeyError` when the file has no `<|endoftext|>`) |

`encode(text)` with no tiktoken keyword stays hf's call (added tokens recognized, the template, the file's
truncation and padding). With `allowed_special` or `disallowed_special` it is tiktoken's: no template, truncation or
padding, the allowed specials recognized, and `ValueError` (tiktoken's message) when the text holds a disallowed one.
`decode` stays hf's; tiktoken's `decode(ids)` is `decode_bytes(ids).decode("utf-8", "replace")`.

What is special is the file's own definition:
- a tokenizer.json: the added tokens it marks special (hf's `special_tokens_set`). Its other added tokens are
  ordinary vocabulary, recognized by `encode_ordinary` too, as hf's `encode_special_tokens=True` recognizes them.
- a tiktoken model: every special token it defines (Kimi K3's 256, Qwen-1's 208), none of them in
  `encode_ordinary`.

On a tokenizer.json, where tiktoken has no reference, each call equals hf:
- `encode(text, allowed_special="all")` is hf's encode without template, truncation or padding.
- `encode_ordinary` is the same with `encode_special_tokens=True`.
- `n_vocab` is the highest id + 1.
- `token_byte_values()` is the bytes of every id outside the added tokens, sorted.

Where these differ from the reference:
- `Encoding.overflowing` is always `[]`. hf fills it with the rest of a text its truncation cut; toks never
  computes that overflow.
- `Encoding.tokens` raises `toks.Error` (`TOKS_E_UNSUPPORTED`) where the ids and the pieces do not determine hf's
  string, rather than guess:
  - a Unigram unknown piece: hf writes the normalized text the piece covers (albert's `'日本語'`), and toks never
    materializes that text;
  - an id the model writes under one string and an added token under another, in a text that holds the token (a
    test fixture; no cached tokenizer gives a model id to an added token of another string);
  - whitespace lstrip / rstrip tokens no piece separates: hf's rstrip swallows a run and writes the next match inside
    it again (`"\t\t"` is `'\t\t'`, `'\t'`), or one run could be either of two whitespace tokens (test fixtures).

  Every other string is hf's, including the whitespace an lstrip / rstrip token takes (`'<|user|>\n'` in phi-3) and a
  normalized token's normalized content (llama's `'▁<s>'`).
- Kimi K3 follows its own tokenizer, which cuts a run of 25,000 non-space characters into chunks; the bare
  tiktoken engine does not.
- Qwen-1's wrapper NFCs the text before tiktoken sees it, and so does toks.

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
    --with tiktoken==0.14.0 --with transformers==5.18.0 \
    pytest python/tests                            # one wheel: pick your python's tag
```

`python/build.sh` builds and tests the wheels for every CPython on a machine. The wheels are attached to the GitHub
releases, not on PyPI yet; the license is BUSL-1.1 ([LICENSING.md](../LICENSING.md)).
