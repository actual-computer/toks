tests/proof/alarms.md: the Eva alarm ledger (SPEC §14.1, T4; docs/proof.md)
=========================================================================

Every property Eva leaves open in a report of tests/proof/eva.sh (a status other than Valid, Considered valid or Dead)
must match one class below; tests/proof/ledger.py checks it and fails on any that does not. A class is in one of
SPEC §14.1's two categories:

  proven: discharged by a separate proof, named in the class (counted as proven)
  review: accepted by review: an open, unproven obligation, with the argument a reviewer checked against the code

A line is `<id> | <jobs> | <file> | <function or *> | <property kind> | <regex on the property text> | <category>: <why>`;
<jobs> is a glob on the report's <job>.<model> name: a class whose argument was checked only for some callers (a
helper of core.h, say) names those jobs, and the helper's alarms in any other job need their own class. Frama-C's
temporaries (tmp_<n>) are folded to `tmp` before the match. The analysis is unsound nowhere by design
(no -eva-* option drops a check); what it cannot prove it reports, and every report lands here.

The source and its placement (entries.c): a source of len bytes, len in [1, 256 MiB], of unknown contents, placed
flush against the end (EVA_PLACE=0) or the start (EVA_PLACE=1) of a 256 MiB block, so a read past either end of the
source leaves the block in one of the two runs. Eva's domains are non-relational: they keep the offset of a pointer
and the value of len apart, so they lose `end == data + len`; the classes S* are that loss.


json reader (src/core/json.c; entries eva_json, and eva_config through toks_config_parse)
------------------------------------------------------------------------------------------

S1 | * | json.c | toks_json_parse | pointer_value | data \+ len | review: §4.1's precondition: data points to len readable bytes, so data + len is the one-past-the-end pointer; entries.c places the source so that data + len is the block's end exactly; Eva keeps offset(data) = MAX - len and len apart and cannot add them back
S2 | * | json.c | skip_ws | mem_access | \\valid_read\(p\) | review: the read is guarded by p < end on the same line; every p handed to skip_ws is in [data, end]: data itself, q + 1 for a q < end that matched a byte (json.c:226, parse_string's closing quote at json.c:124), parse_num's q <= end, or p + 4 / p + 5 after end - p >= 4 / 5 (json.c:245-259); the loss of end == data + len (S1) makes p < end prove nothing to Eva
S3 | * | json.c | skip_ws | ptr_comparison | \\pointer_comparable | review: p and end are both pointers into the source (S2), so the comparison is between pointers of one object
S4 | * | json.c | toks_json_parse | mem_access | \\valid_read\(p\) | review: json.c:217 reads *p in phase JF_VALUE; every transition into JF_VALUE checks p < end first (json.c:209 the root, :234 a pushed container, :284 a member's value, :295 after ','), and the phase is left before p moves; json.c:271 reads *p after p >= end failed on the same line; p is in [data, end] as in S2
S5 | * | json.c | toks_json_parse | ptr_comparison | \\pointer_comparable | review: as S3
S6 | * | json.c | parse_string | pointer_value | dst \+ n | review: n <= r - p <= raw (S7), and dst is a block of raw bytes, so dst + n is at most its one-past-the-end
S7 | * | json.c | parse_string | mem_access | \\valid\(dst \+ tmp\) | review: pass 2 writes n < raw: each step consumes k >= 1 bytes of [p, q) and writes at most k (a plain byte 1 for 1, a simple escape 1 for 2, a \u escape at most 3 for 6, a surrogate pair 4 for 12), so before every write n + (bytes written) <= r_after - p <= q - p = raw = dst's block size; pass 1 guarantees every backslash in [p, q) is followed by its escape byte inside [p, q), and json.c:96 / :100 check the \u digits against q
S8 | json-* | core.h | toks_utf8_put | mem_access | \\valid\(o \+ [0-3]\) | review: called only from parse_string (json.c:109) with o = dst + n; the escape it replaces is 6 bytes (BMP, writes <= 3) or 12 (pair, writes 4), so o + 3 < dst + raw (S7)
S9 | json-* | core.h | toks_utf8_len | mem_access | \\valid_read\(p \+ [0-3]\) | review: toks_utf8_valid(dst, n) calls it at p = dst + i with avail = n - i >= 1 for i < n; it reads p[0] and p[k] only after avail >= k + 1 (core.h:117-137), so every read is in dst[0, n) and n <= raw = the block's size (S7)
S10 | json-* | core.h | toks_utf8_len | initialization | \\initialized\(p \+ [0-3]\) | review: dst[0, n) is written by pass 2 before toks_utf8_valid reads it (json.c:89-121); reads stay in [0, n) (S9)
S11 | json-* | core.h | toks_utf8_valid | pointer_value | p \+ i | review: i <= n <= raw (S9), so p + i is at most the block's one-past-the-end
S12 | * | json.c | parse_num | signed_downcast | mag ≤ 9223372036854775807 | review: mag only grows by mag = mag * 10 + dg when mag <= (2^63 - 1 - dg) / 10 (json.c:146), so mag <= 2^63 - 1; on overflow mag is set to 0 (json.c:163); both casts at json.c:165 are in range


the frame stack and the node tree
---------------------------------

F1 | * | json.c | toks_json_parse | pointer_value | &st\[\(uint32_t\)\(depth - 1u\)\] | review: depth starts at 1 (json.c:206), grows only after depth < 64 (json.c:221), and falls only in phase JF_MEMBER_END / JF_ELEM_END (json.c:297), which only a frame with obj != NULL takes (jf_after_value gives the root frame, obj == NULL, JF_DONE); the root frame st[0] never takes them, so depth >= 2 before the decrement and depth - 1 >= 0 at json.c:212 and :299
F2 | * | json.c | * | initialization | \\initialized\(&f->(obj|mem|last|phase)\) | review: f is &st[depth - 1] (F1); st[0] is written at json.c:205 and st[depth] in full at json.c:230-231 before depth++, so st[0, depth) is initialized; Eva's writes to st[depth] at an imprecise depth are weak updates and keep "uninitialized" in every cell
F3 | * | json.c | * | mem_access | \\valid_read\(&f->phase\)|\\valid_read\(&f->obj\) | review: f = &st[depth - 1] with 1 <= depth <= 64 (F1)
F4 | * | json.c | toks_json_parse | mem_access | \\valid(_read)?\(&\(f->obj\)->(child|type)\) | review: json.c:278 runs in phase JF_KEY and :289 in JF_MEMBER_END / JF_ELEM_END; a frame takes those phases only when pushed for a container (json.c:230-231: obj = v, a fresh node) or after ',' in one, so f->obj is a live JV_OBJ / JV_ARR node
F5 | * | json.c | * | initialization | \\initialized\(&\(f->obj\)->type\) | review: f->obj is a node from jv_new, which zeroes the node and sets type before returning it (json.c:23-30)
N1 | * | json.c | toks_jv_get | initialization | \\initialized\(&(obj|m)->(type|next|child|s|s_len)\) | review: every node comes from jv_new (memset to 0, then type); parse_string / parse_num set s and s_len, jf_link / json.c:278 set child and next; the arena's per-call model (prelude.h) is a malloc per node and Eva merges nodes of one call site into a weak base, which keeps "uninitialized" from the fresh blocks
N2 | * | json.c | toks_jv_get | precondition of memcmp | (valid_read_or_empty|\\initialized|non_escaping)\(\(char \*\)a|valid_s1|initialization: s1|danglingness: s1 | review: memcmp(m->s, key, kl) runs only for a JV_MEM node with s_len == kl; a member key's s is parse_string's dst, a block of raw >= s_len bytes, written for [0, s_len) (S7), and the arena keeps it for the context's load (no free before toks_config_parse returns)
