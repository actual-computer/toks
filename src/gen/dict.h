/* toks: the piece dictionary of the byte-level bpe words table (kernels.md §6 "Static table"; tools/gen/dict.py).
 *
 * toks_dict holds toks_dict_n pieces of 2..15 bytes in score order, each one length byte then its bytes: pieces
 * that common text cuts and that are no single token of the reference byte-level bpe models. toks_bpe_build seats
 * them in the words table after the model's own tokens, each valued by K6's c twin on the tables just built
 * (SPEC §2.7): the list only picks which pieces get a certified entry, never what an entry says. */

#ifndef TOKS_DICT_H
#define TOKS_DICT_H

#include <stdint.h>

extern const uint32_t toks_dict_n;
extern const uint8_t toks_dict[];

#endif /* TOKS_DICT_H */
