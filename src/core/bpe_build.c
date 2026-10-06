/* bpe_build.c: the byte-level bpe tables of layout.h built from a toks_config (docs/notes/c-core.md §bpe_build.c.1) */
#include "bpe.h"
#include "kernels.h"
#include "../gen/dict.h"

/* ---- sizing (shared by the builder and the estimate) --------------------------------------------- */

static uint64_t merge_buckets(uint64_t n_merges)           /* 8 slots each: load <= 0.625 */
{
    uint64_t b = bpe_pow2(n_merges / 5u + 1u);
    return b < 2u ? 2u : b;
}

static uint64_t vhash_slots(uint64_t n_tokens)             /* load < 0.5 */
{
    return bpe_pow2(2u * n_tokens + 1u);
}

static uint64_t words_buckets(uint64_t n_keys)             /* 2 ways each */
{
    return bpe_pow2(n_keys / 2u + 1u);
}

/* the premerge table's size for nblk blocks (0: no table) */
static uint64_t premerge_size(uint64_t nblk)
{
    return nblk == 0u ? 0u : (uint64_t)TOKS_PM_HDR + (1u + nblk) * TOKS_PM_BLOCK;
}

/* the most blocks premerge_build can use: the stage1 indexes of the model tokens that are one char
 * pattern (bpe_pm_index); every index when cfg carries no token strings (unit tests building tables
 * from raw bytes alone) */
static uint64_t premerge_blocks(const toks_config *cfg)
{
    if (cfg->vocab == NULL || cfg->vocab_len == NULL) { return TOKS_PM_STAGE1; }
    uint8_t seen[(TOKS_PM_STAGE1 + 7) / 8];
    memset(seen, 0, sizeof seen);
    uint64_t nblk = 0u;
    for (uint32_t id = 0; id < cfg->n_vocab; id++) {       /* bound: n_vocab */
        uint8_t b[6];
        uint32_t s1 = 0u, s2 = 0u, n = cfg->vocab_len[id];
        if (n < 2u || n > 6u) { continue; }                /* a raw 2..3-byte token: 2..6 alphabet bytes */
        int64_t l = toks_alpha_bytes(cfg->vocab[id], n, b);
        if (!((l == 2 || l == 3) && bpe_pm_index(b, (uint64_t)l, &s1, &s2) == (uint32_t)l)) { continue; }
        if ((seen[s1 >> 3] & (1u << (s1 & 7u))) == 0u) { seen[s1 >> 3] |= (uint8_t)(1u << (s1 & 7u)); nblk++; }
    }
    return nblk;
}

uint64_t toks_bpe_tables_bytes(const toks_config *cfg)
{
    uint64_t nm = cfg->n_merges, nv = cfg->n_vocab;        /* the counts below never exceed these */
    return (256u * 4u + 64u)                               /* byte2id */
         + (merge_buckets(nm) * 64u + 64u)                 /* merge slots */
         + (merge_buckets(nm) * 4u + 64u)                  /* pair filter */
         + (4u * nm + 64u)                                 /* rank2id */
         + (65536u * 4u + 64u)                             /* bytepair */
         + (vhash_slots(nv) * 8u + 1024u + 64u)            /* vhash + its longest-token table */
         + (premerge_size(premerge_blocks(cfg)) + 64u)     /* premerge */
         + ((uint64_t)TOKS_APM_BYTES + 64u)                /* ascii premerge */
         + (words_buckets(nv) * TOKS_BUCKET + 64u);        /* words */
}

/* ---- premerge (kernels.md §5.1: the candidates, R, the edge chains and their bounds, the risk bits) ---- */

#define PM_NONE UINT32_MAX
#define PM_ASCII 0x10000u                                  /* s1 of an ascii candidate (s2: its pair index) */

typedef struct pm_cand {
    uint32_t id, rb, s1, s2;
    uint32_t le[2], leb[2], re[2], reb[2];                 /* proper edge symbols (PM_NONE: unused), bounds */
} pm_cand;

/* merge-only BPE of b[0, n) (n = 2 or 3) on t's tables: 1 when it ends as [id], with c's R and edges */
static int pm_sim(const toks_tables *t, const uint8_t *b, uint32_t n, uint32_t id, pm_cand *c)
{
    const bpe_mt mt = bpe_mt_of(t);
    uint32_t sym[3], k = n, nl = 0u, nr = 0u, lbd = 0u, rbd = 0u;
    for (uint32_t i = 0; i < n; i++) { sym[i] = t->byte2id[b[i]]; }   /* bound: 3 */
    uint32_t le = sym[0], re = sym[n - 1u];
    c->rb = 0u;
    for (uint32_t j = 0; j < 2u; j++) { c->le[j] = c->re[j] = PM_NONE; c->leb[j] = c->reb[j] = 0u; }   /* bound: 2 */
    while (k > 1u) {                                       /* bound: n - 1 merges */
        uint32_t best = PM_NONE, at = 0u, pr = 0u;
        for (uint32_t j = 0; j + 1u < k; j++) {            /* bound: 2 pairs */
            if ((pr = bpe_mt_find(&mt, bpe_pair_key(sym[j], sym[j + 1u]))) < best) { best = pr; at = j; }
        }
        if (best == PM_NONE) { return 0; }                 /* BPE(B) keeps several symbols */
        lbd = lbd > best ? lbd : best;
        rbd = rbd > best ? rbd : best;
        c->rb = c->rb > best ? c->rb : best;
        int left = at == 0u, right = at + 2u == k;
        sym[at] = bpe_mt_id(&mt, best);
        for (uint32_t j = at + 1u; j + 1u < k; j++) { sym[j] = sym[j + 1u]; }   /* bound: 1 */
        k -= 1u;
        if (left) { c->le[nl] = le; c->leb[nl] = lbd; nl++; le = sym[0]; lbd = 0u; }
        if (right) { c->re[nr] = re; c->reb[nr] = rbd; nr++; re = sym[k - 1u]; rbd = 0u; }
    }
    return sym[0] == id;
}

/* the risk bits from one operand list (entries prio << 9 | the neighbour's byte, 256: none, always risky), every
 * byte with prio <= bound: side 0 / 1 the char table's bits for the byte before / after, 2 the ascii classes */
static uint64_t pm_risk(const uint32_t *off, const uint32_t *ent, uint32_t sym, uint32_t bound, int side)
{
    uint64_t m = 0u;
    if (sym == PM_NONE) { return 0u; }
    for (uint32_t i = off[sym]; i < off[sym + 1u]; i++) {  /* bound: the merges with sym as that operand */
        uint32_t b = ent[i] & 511u;
        if ((ent[i] >> 9) > bound) { continue; }
        if (side == 2) { m |= b > 255u ? 0xFFFFFFFFu : UINT64_C(1) << bpe_apm_class(b); continue; }
        m |= UINT64_C(1) << (b > 255u ? (uint32_t)TOKS_PM_ALWAYS : side == 0 ? bpe_pm_lbit(b) : bpe_pm_rbit(b));
    }
    return m;
}

/* premerge_build's rel: list i is read up to prio bound */
static void pm_need(uint32_t *rel, uint32_t i, uint32_t bound) { rel[i] = rel[i] > bound ? rel[i] : bound + 1u; }

static int model_token(const toks_config *cfg, uint32_t id);

static int64_t premerge_build(toks_tables *t, toks_arena *ar, const toks_config *cfg)
{
    uint32_t nv = cfg->n_vocab;
    t->premerge = NULL;
    t->apm = NULL;
    uint32_t nc = 0u;
    for (uint32_t id = 0; id < nv; id++) {                 /* bound: nv */
        uint32_t l = t->tok_off[id + 1u] - t->tok_off[id];
        nc += l == 2u || l == 3u;                          /* the candidates' bound */
    }
    if (nc == 0u) { return 0; }
    uint64_t cb = (uint64_t)nc * sizeof(pm_cand), rb = 8u * ((uint64_t)nv + 1u), ob = 2u * 4u * ((uint64_t)nv + 2u);
    pm_cand *cand = (pm_cand *)toks_plat_alloc(cb);
    uint32_t *rel = (uint32_t *)toks_plat_alloc(rb);
    uint32_t *off = (uint32_t *)toks_plat_alloc(ob);      /* [0, nv + 2): right-operand lists, then left */
    uint32_t *ent = NULL;
    uint64_t eb = 0u;
    int64_t rc = TOKS_E_NOMEM;
    if (cand == NULL || rel == NULL || off == NULL) { goto done; }
    memset(rel, 0, rb);
    memset(off, 0, ob);
    uint32_t *offr = off, *offl = off + nv + 2u;

    /* candidates: BPE(B) = [t]; mark what their risk reads: rel[2 id] / rel[2 id + 1] = 1 + the highest prio of
     * id's list by right / left operand (0: none) */
    uint32_t k = 0u;
    for (uint32_t id = 0; id < nv; id++) {                 /* bound: nv */
        uint32_t o = t->tok_off[id], l = t->tok_off[id + 1u] - o, s1 = 0u, s2 = 0u;
        const uint8_t *b = t->tok_bytes + o;
        int ok = (l == 2u || l == 3u) && bpe_pm_index(b, l, &s1, &s2) == l;   /* one char pattern, or two ascii bytes */
        if (!ok && l == 2u && ((b[0] | b[1]) & 0x80u) == 0u) { s1 = PM_ASCII; s2 = (uint32_t)b[0] << 7 | b[1]; ok = 1; }
        if (!ok || !model_token(cfg, id)) { continue; }   /* (or decode-only) */
        pm_cand *c = cand + k;
        if (!pm_sim(t, b, l, id, c)) { continue; }
        c->id = id;
        c->s1 = s1;
        c->s2 = s2;
        pm_need(rel, 2u * id, c->rb);
        pm_need(rel, 2u * id + 1u, c->rb);
        for (uint32_t j = 0; j < 2u; j++) {                /* bound: 2 */
            if (c->le[j] != PM_NONE) { pm_need(rel, 2u * c->le[j], c->leb[j]); }
            if (c->re[j] != PM_NONE) { pm_need(rel, 2u * c->re[j] + 1u, c->reb[j]); }
        }
        k += 1u;
    }
    nc = k;

    /* the merges up to the prio their right operand needs (entry: prio, the left operand's last byte) and their
     * left operand needs (prio, the right operand's first byte): counts, offsets, entries, over the merge table's
     * slots (the surviving pairs with their prios) */
    const uint64_t ns = (t->merge_mask + 1u) * 8u;
    for (int pass = 0; pass < 2; pass++) {                 /* bound: 2 (count, then fill) */
        if (pass == 1) {
            for (uint32_t i = 0; i < nv + 1u; i++) { offr[i + 1u] += offr[i]; offl[i + 1u] += offl[i]; }   /* bound: nv + 1 */
            eb = 4u * ((uint64_t)offr[nv + 1u] + offl[nv + 1u] + 1u);
            ent = (uint32_t *)toks_plat_alloc(eb);
            if (ent == NULL) { goto done; }
        }
        for (uint64_t si = 0; si < ns; si++) {             /* bound: the slots */
            uint64_t sl = t->merge_slots[si];
            if (sl == BPE_EMPTY_SLOT) { continue; }
            uint64_t key = sl >> TOKS_PRIO_BITS;
            uint32_t pr = (uint32_t)(sl & BPE_PRIO_MASK), left = (uint32_t)(key >> 21), right = (uint32_t)(key & 0x1FFFFFu);
            if (left >= nv || right >= nv) { continue; }
            uint32_t ol = t->tok_off[left], ll = t->tok_off[left + 1u] - ol;
            uint32_t orr = t->tok_off[right], lr = t->tok_off[right + 1u] - orr;
            if (pr < rel[2u * right]) {
                uint32_t b = ll > 0u ? t->tok_bytes[ol + ll - 1u] : 256u;
                if (pass == 0) { offr[right + 2u] += 1u; } else { ent[offr[right + 1u]++] = (pr << 9) | b; }
            }
            if (pr < rel[2u * left + 1u]) {
                uint32_t b = lr > 0u ? t->tok_bytes[orr] : 256u;
                if (pass == 0) { offl[left + 2u] += 1u; } else { ent[offr[nv + 1u] + offl[left + 1u]++] = (pr << 9) | b; }
            }
        }
    }
    /* after the fill, offr[i + 1] / offl[i + 1] are list i's ends and offr[i] / offl[i] its starts */

    /* the char table: one block per stage1 index any certified char uses */
    const uint32_t *el = ent + offr[nv + 1u];
    uint32_t nblk = 0u, na = 0u;
    uint32_t s1blk[TOKS_PM_STAGE1];
    for (uint32_t i = 0; i < (uint32_t)TOKS_PM_STAGE1; i++) { s1blk[i] = 0u; }   /* bound: TOKS_PM_STAGE1 */
    for (uint32_t i = 0; i < nc; i++) {                    /* bound: nc */
        if (cand[i].s1 == PM_ASCII) { na++; } else if (s1blk[cand[i].s1] == 0u) { s1blk[cand[i].s1] = ++nblk; }
    }
    uint8_t *pm = nblk > 0u ? (uint8_t *)toks_tab_ar(ar, premerge_size(nblk), 64u, TOKS_X_PREMERGE) : NULL;
    uint8_t *apm = na > 0u ? (uint8_t *)toks_tab_ar(ar, TOKS_APM_BYTES, 64u, TOKS_X_APM) : NULL;
    if ((nblk > 0u && pm == NULL) || (na > 0u && apm == NULL)) { goto done; }
    if (pm != NULL) {
        memset(pm, 0, premerge_size(nblk));
        for (uint32_t i = 0; i < (uint32_t)TOKS_PM_STAGE1; i++) {   /* bound: TOKS_PM_STAGE1 */
            ((uint32_t *)(void *)pm)[i] = (uint32_t)TOKS_PM_HDR + s1blk[i] * (uint32_t)TOKS_PM_BLOCK;   /* 0: the empty block */
        }
    }
    if (apm != NULL) {                                     /* layout.h TOKS_APM_*: no pair entry is NONE */
        memset(apm + TOKS_APM_PAIRS, 0xFF, TOKS_APM_BYTES - TOKS_APM_PAIRS);
        for (uint32_t b = 0; b < 256u; b++) { apm[b] = (uint8_t)bpe_apm_class(b); }   /* bound: 256 */
    }
    for (uint32_t i = 0; i < nc; i++) {                    /* bound: nc */
        const pm_cand *c = cand + i;
        int a = c->s1 == PM_ASCII, ls = a ? 2 : 0, rs = a ? 2 : 1;   /* pm_risk's side for the bytes before, after */
        uint64_t before = pm_risk(offr, ent, c->id, c->rb, ls), after = pm_risk(offl, el, c->id, c->rb, rs);
        for (uint32_t j = 0; j < 2u; j++) {                /* bound: 2 */
            before |= pm_risk(offr, ent, c->le[j], c->leb[j], ls);
            after |= pm_risk(offl, el, c->re[j], c->reb[j], rs);
        }
        uint32_t v[4] = { c->id, (uint32_t)after, (uint32_t)before, 0u };
        if (a) { memcpy(apm + TOKS_APM_PAIRS + 16u * c->s2, v, 16); continue; }
        uint64_t e = (uint64_t)c->id | (UINT64_C(1) << TOKS_PM_ALWAYS) | before | after;
        *(uint64_t *)(void *)(pm + ((const uint32_t *)(const void *)pm)[c->s1] + 8u * c->s2) = e;
    }
    t->premerge = pm;
    t->apm = apm;
    rc = 0;
done:
    if (cand != NULL) { toks_plat_free(cand, cb); }
    if (rel != NULL) { toks_plat_free(rel, rb); }
    if (off != NULL) { toks_plat_free(off, ob); }
    if (ent != NULL) { toks_plat_free(ent, eb); }
    return rc;
}

/* ---- the merge table (layout.h, kernels.md §5 "Tables"; spm_build.c's too) ------------------------------------- */

uint64_t toks_merge_slots(toks_tables *t, uint64_t *slots, uint64_t *pf, uint64_t nb, const uint32_t *ml, const uint32_t *mr,
                          uint32_t n, const uint32_t *reach)
{
    uint64_t lg = 0u, out = 0u, w = 0u;
    while ((UINT64_C(1) << lg) < nb) { lg++; }             /* bound: 64 */
    memset(slots, 0xFF, nb * 64u);
    memset(pf, 0, nb * 4u);
    t->merge_slots = slots;
    t->pairf = pf;
    t->merge_mask = nb - 1u;
    t->merge_shift = 64u - lg;
    t->merge_maxprobe = 1u;
    t->n_merges = n;
    for (uint32_t i = 0; i < n; i++) {                     /* bound: n */
        if (reach != NULL && !((reach[ml[i] >> 5] >> (ml[i] & 31u)) & (reach[mr[i] >> 5] >> (mr[i] & 31u)) & 1u)) {
            out++;
            continue;
        }
        uint64_t key = bpe_pair_key(ml[i], mr[i]), home = (key * TOKS_FIB64) >> t->merge_shift, d = 0u, *slot = NULL;
        for (; d < nb && slot == NULL; d++) {              /* bound: nb (8 nb slots > the keys: one is free) */
            uint64_t *bucket = slots + ((home + d) & t->merge_mask) * 8u;
            for (uint32_t j = 0; j < 8u && slot == NULL; j++) {   /* bound: 8 */
                if (bucket[j] == BPE_EMPTY_SLOT || (bucket[j] >> TOKS_PRIO_BITS) == key) { slot = bucket + j; }
            }
        }
        *slot = (key << TOKS_PRIO_BITS) | i;               /* d = buckets probed to land */
        uint64_t m = bpe_pf_bits(key, t->merge_shift, &w);
        pf[w] |= m;
        t->merge_maxprobe = d > t->merge_maxprobe ? d : t->merge_maxprobe;
    }
    return out;
}

/* ---- the builder ---------------------------------------------------------------------------------- */

/* ignore_merges: seat key k with [id] (a token whose own bytes do not bpe back to it) in a free way of its buckets,
 * else in the place of an entry that moves to a free way of its other bucket, else of an entry whose own bytes bpe
 * back to its one id (K6 answers that one without the probe; K6 runs here with the probe off). 1 when seated. */
static int words_seat(const toks_tables *t, uint8_t *words, uint64_t mask, bpe_key k, uint32_t id, uint8_t *work,
                      uint64_t wbytes)
{
    uint32_t val[4], h = bpe_key_hash(k);
    bpe_val_pack_tag(val, &id, 1u, 0u);
    if (bpe_words_put(words, mask, h, k, val)) { return 1; }
    for (uint32_t pass = 0; pass < 2u; pass++) {           /* bound: move, then drop */
        for (uint32_t i = 0; i < 4u; i++) {                 /* bound: 2 buckets x 2 ways */
            uint8_t *e = words + ((uint64_t)((i >> 1) == 0u ? h : (h >> 16) | (h << 16)) & mask) * TOKS_BUCKET + 16u * (i & 1u);
            bpe_key ok;
            uint32_t ov[4], got[TOKS_KEY_MAXLEN];
            memcpy(&ok.lo, e, 8);
            memcpy(&ok.hi, e + 8, 8);
            memcpy(ov, e + 32, 16);
            if (pass == 1u) {
                toks_k6_args k6 = { e, e[15], got, work, wbytes, 0u, 0u, 0u };
                if (bpe_val_count(ov) != 1u || toks_k6_bpe_c(t, &k6) != 1u || got[0] != (ov[0] & TOKS_ID_MASK)) { continue; }
            }
            memcpy(e, &k.lo, 8);
            memcpy(e + 8, &k.hi, 8);
            memcpy(e + 32, val, 16);
            if (bpe_words_put(words, mask, bpe_key_hash(ok), ok, ov) || pass == 1u) { return 1; }
            memcpy(e, &ok.lo, 8);                           /* no room for it: back */
            memcpy(e + 8, &ok.hi, 8);
            memcpy(e + 32, ov, 16);
        }
    }
    return 0;
}

/* 1 when bucket h or rotr32(h, 16) of the words table has a free way (bpe_words_put would seat a key) */
static int words_room(const uint8_t *words, uint64_t mask, uint32_t h)
{
    const uint8_t *b0 = words + ((uint64_t)h & mask) * TOKS_BUCKET;
    const uint8_t *b1 = words + ((uint64_t)((h >> 16) | (h << 16)) & mask) * TOKS_BUCKET;
    return b0[15] == 0u || b0[31] == 0u || b1[15] == 0u || b1[31] == 0u;
}

/* 1 when id is a model token (its vocab string is an alphabet image), 0 for a decode-only id. */
static int model_token(const toks_config *cfg, uint32_t id)
{
    return cfg->n_vocab_raw == 0u || toks_alpha_bytes(cfg->vocab[id], cfg->vocab_len[id], NULL) >= 0;
}

int64_t toks_bpe_build(toks_tables *t, toks_arena *ar, const toks_config *cfg)
{
    uint32_t nv = cfg->n_vocab, nm = cfg->n_merges;
    const uint32_t *ml = cfg->m_left_id, *mr = cfg->m_right_id, *mo = cfg->m_out_id;
    if (nv == 0u || nv > t->n_ids) { return TOKS_E_FORMAT; }
    if (nv > TOKS_MAX_IDS || (uint64_t)nm > BPE_PRIO_MASK + 1u) { return TOKS_E_UNSUPPORTED; }
    t->flags &= ~(uint32_t)(TOKS_TF_IGNORE_MERGES | TOKS_TF_IDS_AS_RANK | TOKS_TF_PROBE_LONG | TOKS_TF_PROBE_ASCII);
    if (cfg->ignore_merges != 0u) { t->flags |= TOKS_TF_IGNORE_MERGES; }

    /* ---- byte2id: the token whose raw form is the byte ----------------------------------------- */
    uint32_t *byte2id = (uint32_t *)toks_tab_ar(ar, 256u * 4u, 64u, TOKS_X_BYTE2ID);
    if (byte2id == NULL) { return TOKS_E_NOMEM; }
    memset(byte2id, 0xFF, 256u * 4u);
    for (uint32_t id = 0; id < nv; id++) {                 /* bound: nv */
        if (t->tok_off[id + 1u] - t->tok_off[id] == 1u && model_token(cfg, id)) { byte2id[t->tok_bytes[t->tok_off[id]]] = id; }
    }
    for (uint32_t b = 0; b < 256u; b++) {                  /* bound: 256 */
        if (byte2id[b] == UINT32_MAX && ((cfg->drop[b >> 3] >> (b & 7u)) & 1u) == 0u) {
            return TOKS_E_FORMAT;                          /* the alphabet is incomplete (a dropped byte: config.c) */
        }
    }
    t->byte2id = byte2id;

    /* ---- the merge table (kernels.md §5 "Tables") ----------------------------------------------------- */
    for (uint32_t i = 0; i < nm; i++) {                    /* bound: nm */
        if (ml[i] >= nv || mr[i] >= nv || mo[i] >= nv) { return TOKS_E_FORMAT; }
    }
    uint64_t nb = merge_buckets(nm);
    uint64_t *slots = (uint64_t *)toks_tab_ar(ar, nb * 64u, 64u, TOKS_X_MERGE_SLOTS);
    uint64_t *pf = (uint64_t *)toks_tab_ar(ar, nb * 4u, 64u, TOKS_X_PAIRF);
    if (slots == NULL || pf == NULL) { return TOKS_E_NOMEM; }
    (void)toks_merge_slots(t, slots, pf, nb, ml, mr, nm, NULL);
    bpe_mt mt = bpe_mt_of(t);                              /* its probe (the prios change in place below) */

    /* do the surviving pairs' merged ids rise strictly in rank order? (docs/notes/c-core.md §bpe_build.c.2) */
    uint32_t as_rank = 1u, prev = 0u, have = 0u;
    for (uint32_t i = 0; i < nm && as_rank != 0u && cfg->ids_as_rank == 0u; i++) {   /* bound: nm */
        if (bpe_mt_find(&mt, bpe_pair_key(ml[i], mr[i])) != i) { continue; }   /* a later duplicate survives */
        if (have != 0u && mo[i] <= prev) { as_rank = 0u; }
        prev = mo[i];
        have = 1u;
    }
    if (as_rank != 0u) {
        for (uint64_t s = 0; s < nb * 8u; s++) {           /* bound: the slots */
            if (slots[s] != BPE_EMPTY_SLOT) { slots[s] = (slots[s] & ~BPE_PRIO_MASK) | mo[slots[s] & BPE_PRIO_MASK]; }
        }
        t->flags |= TOKS_TF_IDS_AS_RANK;
        t->rank2id = NULL;
    } else {
        uint32_t *r2i = (uint32_t *)toks_tab_ar(ar, 4u * (uint64_t)nm, 64u, TOKS_X_RANK2ID);
        if (r2i == NULL) { return TOKS_E_NOMEM; }
        memcpy(r2i, mo, 4u * (uint64_t)nm);
        t->rank2id = r2i;
    }

    /* ---- bytepair: the prio of (byte2id[b0], byte2id[b1]) --------------------------------------- */
    uint32_t *bytepair = (uint32_t *)toks_tab_ar(ar, 65536u * 4u, 64u, TOKS_X_BYTEPAIR);
    if (bytepair == NULL) { return TOKS_E_NOMEM; }
    for (uint32_t bp = 0; bp < 65536u; bp++) {             /* bound: 65536 */
        bytepair[bp] = bpe_mt_find(&mt, bpe_pair_key(byte2id[bp >> 8], byte2id[bp & 0xFFu]));   /* NONE: absent */
    }
    t->bytepair = bytepair;

    /* ---- vhash: the tokens a whole piece encodes to (SPEC §2.7): under ignore_merges every model token
     * with a raw form; otherwise the all-ascii tokens over TOKS_KEY_MAXLEN bytes whose own bytes bpe back to
     * the token, certified here by K6's c twin (no vhash yet, so no shortcut; no premerge yet, so plain merges).
     * TOKS_TF_IGNORE_MERGES: the probe is on. TOKS_TF_PROBE_LONG: only for a piece over 15 bytes, K5's words
     * answering the shorter ones (under ignore_merges once the words fill below has seated every token of 2..15
     * bytes whose own bytes do not bpe back to it); TOKS_TF_PROBE_ASCII: only for one ending in an ascii byte
     * (without ignore_merges vhash's tokens are all-ascii; with it, when every longer token that ends otherwise
     * bpes back to itself). A probe that misses costs a hash of the piece and a vhash line; a long non-ascii
     * piece is a letter run that almost never is one token (qwen 3.8 zh 8 hits in 45,090). A token over 256
     * bytes is not certified. The region keeps its full size (every model token): a smaller one moves the
     * arena's other tables, and that alone cost gpt2 1% (kernels.md §5). ----------------------------------- */
    uint32_t n_raw = 0u, n_keys = 0u, n_in = 0u, lmin = cfg->ignore_merges != 0u ? 1u : (uint32_t)TOKS_KEY_MAXLEN + 1u;
    uint32_t ascii = 1u;                                   /* ignore_merges: TOKS_TF_PROBE_ASCII holds */
    for (uint32_t id = 0; id < nv; id++) {                 /* bound: nv */
        uint32_t l = model_token(cfg, id) ? t->tok_off[id + 1u] - t->tok_off[id] : 0u;
        n_raw += l > 0u ? 1u : 0u;                         /* the region's size: every model token (see below) */
        n_keys += (l >= 2u && l <= (uint32_t)TOKS_KEY_MAXLEN) ? 1u : 0u;
    }
    t->vhash = NULL;
    t->vhash_mask = 0u;
    if (n_raw > 0u) {
        uint64_t vs = vhash_slots(n_raw);
        uint64_t *vh = (uint64_t *)toks_tab_ar(ar, vs * 8u + 1024u, 64u, TOKS_X_VHASH);
        if (vh == NULL) { return TOKS_E_NOMEM; }
        memset(vh, 0xFF, vs * 8u);
        uint32_t *ml = (uint32_t *)(void *)(vh + vs);       /* layout.h TT_VHASH: the longest token by first byte */
        memset(ml, 0, 1024u);
        _Alignas(64) uint8_t cw[TOKS_BPE_WORK_BYTES(256)];
        uint32_t cids[256u + 4u];
        t->premerge = NULL;
        t->apm = NULL;
        for (uint32_t id = 0; id < nv; id++) {             /* bound: nv */
            uint32_t o = t->tok_off[id], l = t->tok_off[id + 1u] - o;
            if (l < lmin || !model_token(cfg, id)) { continue; }
            if (cfg->ignore_merges == 0u) {                /* ascii only (below), then certify: bpe(bytes) == [id] */
                uint32_t hi = 0u;
                for (uint32_t i = 0; i < l; i++) { hi |= t->tok_bytes[o + i]; }   /* bound: l */
                if (hi >= 0x80u) { continue; }
                toks_k6_args k6 = { t->tok_bytes + o, l, cids, cw, sizeof cw, 0u, 0u, 0u };
                if (l > 256u || toks_k6_bpe_c(t, &k6) != 1u || cids[0] != id) { continue; }
            } else if (l > (uint32_t)TOKS_KEY_MAXLEN && t->tok_bytes[o + l - 1u] >= 0x80u && ascii != 0u) {
                toks_k6_args k6 = { t->tok_bytes + o, l, cids, cw, sizeof cw, 0u, 0u, 0u };   /* no vhash yet: bpe */
                ascii = l <= 256u && toks_k6_bpe_c(t, &k6) == 1u && cids[0] == id;
            }
            uint8_t b0 = t->tok_bytes[o];
            ml[b0] = ml[b0] > l ? ml[b0] : l;
            uint32_t h = bpe_vhash_h(t->tok_bytes + o, l);
            uint64_t s = (uint64_t)h & (vs - 1u);
            while (vh[s] != BPE_EMPTY_SLOT) { s = (s + 1u) & (vs - 1u); }   /* bound: vs (load < 0.5) */
            vh[s] = ((uint64_t)id << 32) | h;
            n_in += 1u;
        }
        t->vhash = vh;
        t->vhash_mask = vs - 1u;
        if (cfg->ignore_merges == 0u && n_in > 0u) {
            t->flags |= TOKS_TF_IGNORE_MERGES | TOKS_TF_PROBE_LONG | TOKS_TF_PROBE_ASCII;
        }
    }

    /* ---- premerge: K6's certified initial symbols (also speeds up the words' certification runs) ------ */
    int64_t prc = premerge_build(t, ar, cfg);
    if (prc != 0) { return prc; }

    /* ---- words: the certified shortcuts, valued by this lane's own K6 ----------------------------- */
    t->words = NULL;
    t->words_mask = 0u;
    if (n_keys > 0u) {
        uint64_t wb = words_buckets(n_keys);
        uint8_t *words = (uint8_t *)toks_tab_ar(ar, wb * TOKS_BUCKET, 64u, TOKS_X_WORDS);
        if (words == NULL) { return TOKS_E_NOMEM; }
        memset(words, 0, wb * TOKS_BUCKET);
        t->words = words;
        t->words_mask = wb - 1u;
        _Alignas(64) uint8_t work[TOKS_BPE_WORK_BYTES(TOKS_KEY_MAXLEN)];
        uint32_t ids[TOKS_KEY_MAXLEN];
        uint64_t placed = 0u;                              /* entries put in a free way: the list stops at a full table */
        uint32_t unseated = 0u;                            /* ignore_merges: left out, and K6 needs the probe for it */
        for (uint32_t id = 0; id < nv; id++) {             /* bound: nv */
            uint32_t o = t->tok_off[id], l = t->tok_off[id + 1u] - o;
            if (l < 2u || l > (uint32_t)TOKS_KEY_MAXLEN || !model_token(cfg, id)) { continue; }
            bpe_key k = bpe_key_at(t->tok_bytes + o, l, 0u, l);
            uint32_t h = bpe_key_hash(k);
            toks_k6_args k6 = { t->tok_bytes + o, l, ids, work, sizeof work, 0u, 0u, 0u };
            if (words_room(words, wb - 1u, h)) {
                uint64_t n = toks_k6_bpe_c(t, &k6);        /* the certification run */
                if (n > 4u) { continue; }                  /* an entry holds 1..4 ids */
                uint32_t val[4];
                bpe_val_pack_tag(val, ids, (uint32_t)n, 0u);
                placed += (uint64_t)bpe_words_put(words, wb - 1u, h, k, val);
                continue;
            }
            if (cfg->ignore_merges == 0u) { continue; }
            /* a token left out under ignore_merges (both buckets full): K6 answers it without the probe when its own
             * bytes bpe back to it; else words_seat makes room for it, so K6 never needs the probe for 2..15 bytes */
            t->flags &= ~(uint32_t)TOKS_TF_IGNORE_MERGES;
            if (toks_k6_bpe_c(t, &k6) != 1u || ids[0] != id) {
                unseated += words_seat(t, words, wb - 1u, k, id, work, sizeof work) != 0 ? 0u : 1u;
            }
            t->flags |= TOKS_TF_IGNORE_MERGES;
        }
        /* the piece dictionary (src/gen/dict.h): pieces common text cuts that are no single model token, in score
         * order, each valued by K6's c twin like the tokens above (SPEC §2.7: the list only picks which pieces get
         * an entry), into the free ways the tokens left; a piece with a byte the model drops is never one K5 sees.
         * The increment reads the length byte of the piece just done and steps d to the next piece; after the last
         * one d rests on the literal's terminating 0, which the loop's test (i < toks_dict_n) stops before reading.
         * test_bpe's test_dict pins every length to 2..15 and the walk's end to that 0. */
        const uint8_t *d = toks_dict;
        for (uint32_t i = 0; i < toks_dict_n && placed < 2u * wb; i++, d += 1u + d[0]) {   /* bound: toks_dict_n */
            uint32_t l = d[0], h, val[4], ok = 1u;
            for (uint32_t j = 0; j < l; j++) { ok &= byte2id[d[1u + j]] != UINT32_MAX ? 1u : 0u; }   /* bound: 15 */
            bpe_key k = bpe_key_at(d + 1, l, 0u, l);
            h = bpe_key_hash(k);
            if (ok == 0u || bpe_words_probe(words, wb - 1u, h, k) != NULL || !words_room(words, wb - 1u, h)) { continue; }
            toks_k6_args k6 = { d + 1, l, ids, work, sizeof work, 0u, 0u, 0u };
            uint64_t n = toks_k6_bpe_c(t, &k6);            /* the certification run */
            if (n > 4u) { continue; }
            bpe_val_pack_tag(val, ids, (uint32_t)n, 0u);
            placed += (uint64_t)bpe_words_put(words, wb - 1u, h, k, val);
        }
        if (cfg->ignore_merges != 0u && unseated == 0u) { t->flags |= TOKS_TF_PROBE_LONG | (ascii != 0u ? TOKS_TF_PROBE_ASCII : 0u); }
    }
    return 0;
}
