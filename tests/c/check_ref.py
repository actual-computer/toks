#!/usr/bin/env python3
"""check_ref.py: an independent reference of check.c's memo check (CLNH x2 over 4 KiB blocks + poly mod 2^127 - 1),
for the known-answer values in tests/c/test_check.c. Key words: k[i] = splitmix64 stream from seed (as the test sets).
"""
import sys

M64 = (1 << 64) - 1
P = (1 << 127) - 1
BLOCK = 4096
KEY_W = BLOCK // 8 + 6


def clmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def splitmix(seed, n):
    out, x = [], seed
    for _ in range(n):
        x = (x + 0x9E3779B97F4A7C15) & M64
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        out.append(z ^ (z >> 31))
    return out


def words(b):
    return [int.from_bytes(b[i:i + 8], 'little') for i in range(0, len(b), 8)]


def check(key, g):
    n = len(g)
    r = ((key[KEY_W - 1] << 64) | key[KEY_W - 2]) >> 2
    h = 0
    at = 0
    while at < n:
        blk = g[at:at + BLOCK]
        pad = (-len(blk)) % 32
        m = words(blk + b'\0' * pad)
        s0 = s1 = 0
        for i in range(len(m) // 4):
            a = key[4 * i:4 * i + 4]
            b = key[4 * i + 4:4 * i + 8]
            w = m[4 * i:4 * i + 4]
            s0 ^= clmul(w[0] ^ a[0], w[2] ^ a[2]) ^ clmul(w[1] ^ a[1], w[3] ^ a[3])
            s1 ^= clmul(w[0] ^ b[0], w[2] ^ b[2]) ^ clmul(w[1] ^ b[1], w[3] ^ b[3])
        for c in (s0 & M64, s0 >> 64, s1 & M64, s1 >> 64):
            h = (h * r + c) % P
        at += BLOCK
    h = (h * r + n) % P
    return h & M64, h >> 64


def text(n, seed):
    s = splitmix(seed, (n + 7) // 8)
    b = b''.join(x.to_bytes(8, 'little') for x in s)
    return b[:n]


if __name__ == '__main__':
    key = splitmix(0x746F6B73, KEY_W)    # "toks"
    for n in [0, 1, 7, 8, 31, 32, 33, 255, 256, 1000, 4095, 4096, 4097, 8191, 8192, 12345]:
        lo, hi = check(key, text(n, 0x5EED + n))
        print('    { %5d, 0x%016Xull, 0x%016Xull },' % (n, lo, hi))
