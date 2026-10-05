# Licensing

toks is source-available under the Business Source License 1.1 (BUSL-1.1). The license text with toks's
parameters is [LICENSE](LICENSE); this page is the plain-language summary. Where the two differ, LICENSE governs.

## In short

- **Under 10^15 tokens a year: you need nothing.** Use toks in production, modify it, vendor it, ship it inside
  your product. The grant covers any organization that, together with its affiliates, processes fewer than
  1,000,000,000,000,000 (one quadrillion) tokens in total per rolling twelve months. That is about 32 million
  tokens a second, around the clock, all year; nearly everyone is under it.
- **At or above 10^15 tokens a year: a commercial license.** It is inexpensive and a short form, not a
  negotiation. Contact Actual Computer Inc. through [actual.inc](https://actual.inc).
- **Every version becomes Apache-2.0 four years after its first public release.** Each tagged version's LICENSE
  states its own Change Date (toks 0.3.0: 2030-10-05), and BUSL's own rule, the fourth anniversary of the first
  public release of that version, applies whichever comes first. On that date the BUSL terms for that version end
  and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) takes over.
- **Source-available, not open source.** BUSL-1.1 is not an OSI-approved license. Until its Change Date, each
  version of toks is source-available: you can read, build, modify and redistribute it under the terms above.
- **For license scanners:** the SPDX identifier is `BUSL-1.1`. `include/toks.h` carries
  `SPDX-License-Identifier: BUSL-1.1` and the Python wheel declares `License-Expression: BUSL-1.1`.

## FAQ

- *Can I vendor it?* Yes, under the grant: copy the source or the built library into your tree and keep the
  LICENSE file with it.
- *Can I ship it inside my product?* Yes, under the grant, closed products included; the LICENSE travels with
  the copies, as it says.
- *What counts toward the threshold?* Your organization's total token processing, together with its
  affiliates, across all of your systems and by any software or service, not only tokens processed with toks.
  Self-assessed, in good faith.
- toks's C and assembly core carries no third-party code; its Unicode tables are derived from the Unicode
  Character Database, partly through the generated tables of three Rust crates (notices in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Contributing

Contributions land under [CONTRIBUTING.md](CONTRIBUTING.md): for now every pull request that merges is run by an
Actual Computer engineer, who cherry-picks contributors' commits with their authorship kept, and every commit
carries a sign-off plus a grant that lets Actual Computer Inc. relicense the contribution. That grant is what
keeps the commercial and Apache-2.0 sides of this model consistent.

## Trademarks

"toks" and the Actual Computer name are trademarks of Actual Computer Inc. The license grants no trademark
rights (its text says so); you may use the names to refer to the software truthfully.
