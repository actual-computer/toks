/* alloc.c: the os-free part of the load path: the crc32c table (docs/notes/c-core.md §alloc.c.1) */
#include "core.h"

/* ---- crc32c table (reflected 0x82F63B78, standard sse4.2 polynomial) ---------------------------------- */

const uint32_t TOKS_CRC32C_TAB[256] = {
    0x00000000u, 0xF26B8303u, 0xE13B70F7u, 0x1350F3F4u, 0xC79A971Fu, 0x35F1141Cu, 0x26A1E7E8u, 0xD4CA64EBu,
    0x8AD958CFu, 0x78B2DBCCu, 0x6BE22838u, 0x9989AB3Bu, 0x4D43CFD0u, 0xBF284CD3u, 0xAC78BF27u, 0x5E133C24u,
    0x105EC76Fu, 0xE235446Cu, 0xF165B798u, 0x030E349Bu, 0xD7C45070u, 0x25AFD373u, 0x36FF2087u, 0xC494A384u,
    0x9A879FA0u, 0x68EC1CA3u, 0x7BBCEF57u, 0x89D76C54u, 0x5D1D08BFu, 0xAF768BBCu, 0xBC267848u, 0x4E4DFB4Bu,
    0x20BD8EDEu, 0xD2D60DDDu, 0xC186FE29u, 0x33ED7D2Au, 0xE72719C1u, 0x154C9AC2u, 0x061C6936u, 0xF477EA35u,
    0xAA64D611u, 0x580F5512u, 0x4B5FA6E6u, 0xB93425E5u, 0x6DFE410Eu, 0x9F95C20Du, 0x8CC531F9u, 0x7EAEB2FAu,
    0x30E349B1u, 0xC288CAB2u, 0xD1D83946u, 0x23B3BA45u, 0xF779DEAEu, 0x05125DADu, 0x1642AE59u, 0xE4292D5Au,
    0xBA3A117Eu, 0x4851927Du, 0x5B016189u, 0xA96AE28Au, 0x7DA08661u, 0x8FCB0562u, 0x9C9BF696u, 0x6EF07595u,
    0x417B1DBCu, 0xB3109EBFu, 0xA0406D4Bu, 0x522BEE48u, 0x86E18AA3u, 0x748A09A0u, 0x67DAFA54u, 0x95B17957u,
    0xCBA24573u, 0x39C9C670u, 0x2A993584u, 0xD8F2B687u, 0x0C38D26Cu, 0xFE53516Fu, 0xED03A29Bu, 0x1F682198u,
    0x5125DAD3u, 0xA34E59D0u, 0xB01EAA24u, 0x42752927u, 0x96BF4DCCu, 0x64D4CECFu, 0x77843D3Bu, 0x85EFBE38u,
    0xDBFC821Cu, 0x2997011Fu, 0x3AC7F2EBu, 0xC8AC71E8u, 0x1C661503u, 0xEE0D9600u, 0xFD5D65F4u, 0x0F36E6F7u,
    0x61C69362u, 0x93AD1061u, 0x80FDE395u, 0x72966096u, 0xA65C047Du, 0x5437877Eu, 0x4767748Au, 0xB50CF789u,
    0xEB1FCBADu, 0x197448AEu, 0x0A24BB5Au, 0xF84F3859u, 0x2C855CB2u, 0xDEEEDFB1u, 0xCDBE2C45u, 0x3FD5AF46u,
    0x7198540Du, 0x83F3D70Eu, 0x90A324FAu, 0x62C8A7F9u, 0xB602C312u, 0x44694011u, 0x5739B3E5u, 0xA55230E6u,
    0xFB410CC2u, 0x092A8FC1u, 0x1A7A7C35u, 0xE811FF36u, 0x3CDB9BDDu, 0xCEB018DEu, 0xDDE0EB2Au, 0x2F8B6829u,
    0x82F63B78u, 0x709DB87Bu, 0x63CD4B8Fu, 0x91A6C88Cu, 0x456CAC67u, 0xB7072F64u, 0xA457DC90u, 0x563C5F93u,
    0x082F63B7u, 0xFA44E0B4u, 0xE9141340u, 0x1B7F9043u, 0xCFB5F4A8u, 0x3DDE77ABu, 0x2E8E845Fu, 0xDCE5075Cu,
    0x92A8FC17u, 0x60C37F14u, 0x73938CE0u, 0x81F80FE3u, 0x55326B08u, 0xA759E80Bu, 0xB4091BFFu, 0x466298FCu,
    0x1871A4D8u, 0xEA1A27DBu, 0xF94AD42Fu, 0x0B21572Cu, 0xDFEB33C7u, 0x2D80B0C4u, 0x3ED04330u, 0xCCBBC033u,
    0xA24BB5A6u, 0x502036A5u, 0x4370C551u, 0xB11B4652u, 0x65D122B9u, 0x97BAA1BAu, 0x84EA524Eu, 0x7681D14Du,
    0x2892ED69u, 0xDAF96E6Au, 0xC9A99D9Eu, 0x3BC21E9Du, 0xEF087A76u, 0x1D63F975u, 0x0E330A81u, 0xFC588982u,
    0xB21572C9u, 0x407EF1CAu, 0x532E023Eu, 0xA145813Du, 0x758FE5D6u, 0x87E466D5u, 0x94B49521u, 0x66DF1622u,
    0x38CC2A06u, 0xCAA7A905u, 0xD9F75AF1u, 0x2B9CD9F2u, 0xFF56BD19u, 0x0D3D3E1Au, 0x1E6DCDEEu, 0xEC064EEDu,
    0xC38D26C4u, 0x31E6A5C7u, 0x22B65633u, 0xD0DDD530u, 0x0417B1DBu, 0xF67C32D8u, 0xE52CC12Cu, 0x1747422Fu,
    0x49547E0Bu, 0xBB3FFD08u, 0xA86F0EFCu, 0x5A048DFFu, 0x8ECEE914u, 0x7CA56A17u, 0x6FF599E3u, 0x9D9E1AE0u,
    0xD3D3E1ABu, 0x21B862A8u, 0x32E8915Cu, 0xC083125Fu, 0x144976B4u, 0xE622F5B7u, 0xF5720643u, 0x07198540u,
    0x590AB964u, 0xAB613A67u, 0xB831C993u, 0x4A5A4A90u, 0x9E902E7Bu, 0x6CFBAD78u, 0x7FAB5E8Cu, 0x8DC0DD8Fu,
    0xE330A81Au, 0x115B2B19u, 0x020BD8EDu, 0xF0605BEEu, 0x24AA3F05u, 0xD6C1BC06u, 0xC5914FF2u, 0x37FACCF1u,
    0x69E9F0D5u, 0x9B8273D6u, 0x88D28022u, 0x7AB90321u, 0xAE7367CAu, 0x5C18E4C9u, 0x4F48173Du, 0xBD23943Eu,
    0xF36E6F75u, 0x0105EC76u, 0x12551F82u, 0xE03E9C81u, 0x34F4F86Au, 0xC69F7B69u, 0xD5CF889Du, 0x27A40B9Eu,
    0x79B737BAu, 0x8BDCB4B9u, 0x988C474Du, 0x6AE7C44Eu, 0xBE2DA0A5u, 0x4C4623A6u, 0x5F16D052u, 0xAD7D5351u,
};

/* ---- byte-level alphabet (gpt-2 bytes_to_unicode) ------------------------------------------------------ */

/* rationale: docs/notes/c-core.md §alloc.c.2 */
int32_t toks_char_byte(uint32_t cp)
{
    if (cp < 0x80u) {
        if (cp >= 0x21u && cp <= 0x7Eu) { return (int32_t)cp; }
        return -1;
    }
    if (cp >= 0xA1u && cp != 0xADu && cp <= 0xFFu) { return (int32_t)cp; }
    if (cp < 0x100u || cp > 0x143u) { return -1; }
    /* the remapped byte with this index: walk the remapped set in byte order */
    uint32_t idx = cp - 0x100u;
    uint32_t seen = 0;
    for (uint32_t i = 0; i < 256u; i++) {                /* bound: 256 */
        if ((i >= 0x21u && i <= 0x7Eu) || (i >= 0xA1u && i != 0xADu)) { continue; }
        if (seen == idx) { return (int32_t)i; }
        seen++;
    }
    return -1;
}

/* ---- sha256 (load-time only; for toks_info.source_sha256) --------------------------------------------- */

typedef struct s256 { uint32_t h[8]; uint64_t len; uint32_t buf[16]; uint32_t n; } s256;

static const uint32_t S256_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

static uint32_t ror32(uint32_t v, uint32_t n) { return (v >> n) | (v << (32u - n)); }

static void s256_block(s256 *s, const uint32_t w_in[16])
{
    uint32_t w[64];
    for (uint32_t i = 0; i < 16u; i++) { w[i] = w_in[i]; }             /* bound: 16 */
    for (uint32_t i = 16u; i < 64u; i++) {                            /* bound: 48 */
        uint32_t a = ror32(w[i-15u],7u) ^ ror32(w[i-15u],18u) ^ (w[i-15u] >> 3);
        uint32_t b = ror32(w[i-2u],17u) ^ ror32(w[i-2u],19u) ^ (w[i-2u] >> 10);
        w[i] = w[i-16u] + a + w[i-7u] + b;
    }
    uint32_t h[8];
    for (uint32_t i = 0; i < 8u; i++) { h[i] = s->h[i]; }             /* bound: 8 */
    for (uint32_t i = 0; i < 64u; i++) {                             /* bound: 64 */
        uint32_t t1 = h[7] + (ror32(h[4],6u) ^ ror32(h[4],11u) ^ ror32(h[4],25u)) +
                      ((h[4] & h[5]) ^ ((~h[4]) & h[6])) + S256_K[i] + w[i];
        uint32_t t2 = (ror32(h[0],2u) ^ ror32(h[0],13u) ^ ror32(h[0],22u)) +
                      ((h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]));
        h[7]=h[6]; h[6]=h[5]; h[5]=h[4]; h[4]=h[3]+t1; h[3]=h[2]; h[2]=h[1]; h[1]=h[0]; h[0]=t1+t2;
    }
    for (uint32_t i = 0; i < 8u; i++) { s->h[i] += h[i]; }            /* bound: 8 */
}

static void s256_write(s256 *s, const uint8_t *p, uint64_t n)
{
    s->len += n;
    uint64_t i = 0;
    while (i < n) {                                    /* bound: n input bytes */
        uint32_t nb = s->n & 63u;              /* bytes buffered in the current block */
        uint32_t w = nb >> 2, b = nb & 3u;
        if (b == 0u && i + 4u <= n) {
            s->buf[w] = ((uint32_t)p[i] << 24) | ((uint32_t)p[i+1u] << 16) |
                        ((uint32_t)p[i+2u] << 8) | (uint32_t)p[i+3u];
            i += 4u;
            s->n += 4u;
        } else {
            if (b == 0u) { s->buf[w] = 0u; }
            s->buf[w] |= ((uint32_t)p[i]) << (24u - 8u * b);
            i++;
            s->n++;
        }
        if ((s->n & 63u) == 0u && s->n != 0u) { s256_block(s, s->buf); s->n = 0u; }
    }
}

void toks_sha256(const uint8_t *data, uint64_t len, uint8_t out[32])
{
    s256 s;
    s.h[0]=0x6a09e667u; s.h[1]=0xbb67ae85u; s.h[2]=0x3c6ef372u; s.h[3]=0xa54ff53au;
    s.h[4]=0x510e527fu; s.h[5]=0x9b05688cu; s.h[6]=0x1f83d9abu; s.h[7]=0x5be0cd19u;
    s.len = 0; s.n = 0;
    memset(s.buf, 0, sizeof(s.buf));
    s256_write(&s, data, len);
    uint64_t bits = s.len * 8u;
    uint8_t pad = 0x80u;
    s256_write(&s, &pad, 1u);
    while ((s.n & 63u) != 56u) { pad = 0u; s256_write(&s, &pad, 1u); }   /* bound: 63 */
    uint8_t lenb[8];
    for (uint32_t i = 0; i < 8u; i++) { lenb[i] = (uint8_t)(bits >> (56u - 8u * i)); }  /* bound: 8 */
    s256_write(&s, lenb, 8u);
    for (uint32_t i = 0; i < 8u; i++) {                 /* bound: 8 */
        out[i*4u]   = (uint8_t)(s.h[i] >> 24);
        out[i*4u+1u] = (uint8_t)(s.h[i] >> 16);
        out[i*4u+2u] = (uint8_t)(s.h[i] >> 8);
        out[i*4u+3u] = (uint8_t)(s.h[i]);
    }
}
