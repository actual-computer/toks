# Licensing

toks is open source under the Apache License, Version 2.0 (Apache-2.0). The license text is [LICENSE](LICENSE)
and the copyright notice is [NOTICE](NOTICE); this page is the plain-language summary. Where they differ, LICENSE
governs.

## In short

- **Use it for anything.** Use toks in production, modify it, vendor it, ship it inside your product, closed
  products included, at any scale. There is no token threshold, no commercial license and nothing to buy.
- **Keep the notices.** Every copy of toks you pass on, modified or not, carries the LICENSE and NOTICE files,
  and a file you changed says so (Apache-2.0 §4).
- **Patents.** Actual Computer Inc. and every contributor grant you a patent license covering their contribution
  (§3); it ends for anyone who sues over toks (§3).
- **No warranty.** toks is provided as is (§7, §8).
- **For license scanners:** the SPDX identifier is `Apache-2.0`. `include/toks.h` carries
  `SPDX-License-Identifier: Apache-2.0` and the Python wheel declares `License-Expression: Apache-2.0`.

## FAQ

- *Can I vendor it?* Yes: copy the source or the built library into your tree and keep LICENSE and NOTICE with
  it.
- *Can I ship it inside my product?* Yes, closed products included; the notices travel with the copies, as §4
  says.
- *Earlier releases.* toks 0.3.0 and 0.3.1 shipped under the Business Source License 1.1 with Apache-2.0 as
  their Change License, and their tags still carry that text. Every release after 0.3.1 is Apache-2.0 from the
  day it ships.
- toks's C and assembly core carries no third-party code; its Unicode tables are derived from the Unicode
  Character Database, partly through the generated tables of three Rust crates (notices in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Contributing

Contributions land under [CONTRIBUTING.md](CONTRIBUTING.md): for now every pull request that merges is run by an
Actual Computer engineer, who cherry-picks contributors' commits with their authorship kept, and every commit
carries a sign-off plus a grant that lets Actual Computer Inc. relicense the contribution, so toks stays one work
under one licensor.

## Trademarks

"toks" and the Actual Computer name are trademarks of Actual Computer Inc. The license grants no trademark
rights (§6); you may use the names to refer to the software truthfully.
