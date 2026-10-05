# Third-party notices

toks's C and assembly core (`src/`, `include/`) carries no third-party code. The only third-party material in
this repository is data: the Unicode tables below, derived from the Unicode Character Database, partly through the
generated tables of three Rust crates (below).

## Unicode Character Database

The generated tables `src/gen/norm_nfc.{c,h}` (NFC / NFKC), `src/gen/bert_tables.{c,h}` (categories, case,
whitespace), `src/gen/grapheme17.c` (extended grapheme clusters), `src/gen/ucd_flags.{c,h}` (per-code-point
properties), `src/gen/rx_word.{c,h}` (`\w`), `src/gen/han_ranges.{c,h}` (Script=Han) and the test data
`tests/data/unigram/grapheme17.txt`, `tests/data/wordpiece/unicode.json`, `tests/norm/nfc_golden.txt` and
`tests/norm/nfkc_golden.txt` are derived from the Unicode Character Database (Unicode versions 8.0, 9.0, 16.0
and 17.0), Copyright © 1991-2026 Unicode, Inc., and are distributed under the Unicode License v3, reproduced
below as that license requires (https://www.unicode.org/license.txt).

```
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```

## Rust crates the table generators read

Some of the tables above are extracted from the generated `tables.rs` of the Rust crates hf tokenizers 0.23.2 runs
on, so that toks matches its Unicode versions exactly (each generator pins its source by sha256):

- `src/gen/norm_nfc.{c,h}` and `src/gen/bert_tables.{c,h}`: unicode-normalization-alignments 0.1.12 (Unicode 9.0.0;
  MIT or Apache-2.0)
- `src/gen/bert_tables.{c,h}`: unicode_categories 0.1.1 (Unicode 8.0.0; MIT or Apache-2.0), and the Rust standard
  library's `char::to_lowercase` / `char::is_whitespace` (Unicode 17.0.0; MIT or Apache-2.0)
- `src/gen/grapheme17.c` and `tests/data/unigram/grapheme17.txt`: unicode-segmentation 1.13.3 (Unicode 17.0.0; MIT or
  Apache-2.0)

They are used here under the MIT license:

```
Copyright (c) 2015 The Rust Project Developers
Copyright (c) 2015 The unicode-categories Developers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Not redistributed

Tokenizer files (`tokenizer.json`, `tiktoken.model` and the like) and the bench corpora are downloaded by the
test and bench scripts, pinned by sha256, under their own licenses; none is redistributed here, only toks's own
test texts and the resulting ids. The Python oracle and parity tooling under `python/` and `tests/` use
Hugging Face `tokenizers` and `tiktoken` as test-time dependencies to check toks against them; none of their
code ships with toks.
