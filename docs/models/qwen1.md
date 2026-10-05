Qwen-1 (Qwen-7B / 14B / 72B): a tiktoken model directory
============================================================

Qwen-1 ships no tokenizer.json: `qwen.tiktoken` (the ranks, tiktoken's base64 format), `tokenization_qwen.py`
(transformers remote code, `QWenTokenizer`) and a three-key `tokenizer_config.json`. toks_load takes the directory
(or the `qwen.tiktoken` path inside it) and reads the three files (src/core/tiktoken.c `qwen`, src/core/load.c).

1. the reference
-----------------

QWenTokenizer overrides `tokenize`, so `encode()` never runs transformers' added-token trie: it is

    tiktoken.Encoding("Qwen", PAT_STR, ranks, SPECIAL_TOKENS).encode(unicodedata.normalize("NFC", text),
                                                                 allowed_special="all", disallowed_special=())

with PAT_STR = the qwen 2 template's string (kernels.md §3: cl100k parameters, `\p{N}` single), SPECIAL_TOKENS =
`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|extra_0|>` .. `<|extra_204|>` numbered from SPECIAL_START_ID =
151643 = len(ranks). No bos / eos is added. `decode(ids, skip_special_tokens=True)` keeps the ids below `eod_id`
(151643), i.e. drops every special; errors="replace" is the caller's (toks returns the bytes).

So toks builds: the ranks as a tiktoken model (ignore_merges: tiktoken's whole-piece lookup), the qwen 2 template,
NFC (K2), and the 208 specials as phase-1 added tokens (matched in the NFC'd text, all special): mode ALL is the
wrapper's encode, NONSPECIAL and NONE are `encode_ordinary` of the NFC'd text; skip-special decode is the wrapper's.

2. what the reader checks
--------------------------

The wrapper must hold every line of tiktoken.c `QWEN_LINES` (blanks around a line ignored): the class line, PAT_STR
exactly, the three special names, EXTRAS = range(205), SPECIAL_START_ID = 151643, the specials' order
(`ENDOFTEXT,` `IMSTART,` `IMEND,` one after another, `+ EXTRAS`, `start=SPECIAL_START_ID,`), the NFC line, the
tokenize defaults (allowed_special "all", disallowed_special ()), the encode call and the skip filter. The ranks must
be 0 .. 151642. tokenizer_config.json: tokenizer_class QWenTokenizer, auto_map AutoTokenizer
tokenization_qwen.QWenTokenizer, model_max_length, nothing else. Anything else is a named refusal
("tokenization_qwen.py: not the Qwen-1 wrapper toks reads ...", "qwen.tiktoken: not ranks 0 .. 151642 ...",
"tokenizer_config.json: not QWenTokenizer's ...").

3. proof
---------

tests/parity/qwen1_check.py: the wrapper's own constants (exec'd from the file) + tiktoken 0.14.0 against toks's
driver on random texts over the specials, broken specials (`<|extra_205|>`, `<|im_`), NFC-composing and
NFC-stable sequences and the pattern's edge chars: encode ALL / NONSPECIAL / NONE, decode, skip-special decode.
Pinned files: tools/corpora/fetch_tokenizers.py qwen1-72b.* (Qwen/Qwen-72B at b8e18ac); test_targets loads them.
