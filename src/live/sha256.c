/*
 * SHA-256 implementation — FIPS 180-4 compliant.
 * Adapted from Brad Conte's public-domain crypto-algorithms
 * (https://github.com/B-Con/crypto-algorithms/blob/master/sha256.c)
 * API renamed to QcfpSha256 / qcfp_sha256_* for quakecore_live.
 * Public domain — no attribution required, but credit goes to Brad Conte.
 */

#include "sha256.h"
#include <string.h>

/* ---- helpers ---- */
#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define CH(x, y, z)   (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z)  (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)  (ROTRIGHT(x, 2)  ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x)  (ROTRIGHT(x, 6)  ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7)  ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

static const uint32_t k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static void sha256_transform(QcfpSha256* ctx, const uint8_t data[64])
{
    uint32_t a, b, c, d, e, f, g, h, t1, t2, m[64];
    int i, j;

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = ((uint32_t)data[j]     << 24)
              | ((uint32_t)data[j + 1] << 16)
              | ((uint32_t)data[j + 2] <<  8)
              | ((uint32_t)data[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx->s[0]; b = ctx->s[1]; c = ctx->s[2]; d = ctx->s[3];
    e = ctx->s[4]; f = ctx->s[5]; g = ctx->s[6]; h = ctx->s[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->s[0] += a; ctx->s[1] += b; ctx->s[2] += c; ctx->s[3] += d;
    ctx->s[4] += e; ctx->s[5] += f; ctx->s[6] += g; ctx->s[7] += h;
}

void qcfp_sha256_init(QcfpSha256* ctx)
{
    ctx->buf_len = 0;
    ctx->len     = 0;
    ctx->s[0] = 0x6a09e667u;
    ctx->s[1] = 0xbb67ae85u;
    ctx->s[2] = 0x3c6ef372u;
    ctx->s[3] = 0xa54ff53au;
    ctx->s[4] = 0x510e527fu;
    ctx->s[5] = 0x9b05688cu;
    ctx->s[6] = 0x1f83d9abu;
    ctx->s[7] = 0x5be0cd19u;
}

void qcfp_sha256_update(QcfpSha256* ctx, const void* data, size_t n)
{
    const uint8_t* p = (const uint8_t*)data;
    size_t i;

    for (i = 0; i < n; ++i) {
        ctx->buf[ctx->buf_len++] = p[i];
        if (ctx->buf_len == 64) {
            sha256_transform(ctx, ctx->buf);
            ctx->len += 512;
            ctx->buf_len = 0;
        }
    }
}

void qcfp_sha256_final(QcfpSha256* ctx, uint8_t out[32])
{
    size_t i;
    uint64_t bit_len;

    i = ctx->buf_len;

    /* Pad: append 0x80 then zeros to reach 56 bytes in the block */
    if (ctx->buf_len < 56) {
        ctx->buf[i++] = 0x80;
        while (i < 56)
            ctx->buf[i++] = 0x00;
    } else {
        ctx->buf[i++] = 0x80;
        while (i < 64)
            ctx->buf[i++] = 0x00;
        sha256_transform(ctx, ctx->buf);
        memset(ctx->buf, 0, 56);
    }

    /* Append bit length as big-endian 64-bit */
    bit_len = ctx->len + (uint64_t)ctx->buf_len * 8u;
    ctx->buf[63] = (uint8_t)(bit_len);
    ctx->buf[62] = (uint8_t)(bit_len >> 8);
    ctx->buf[61] = (uint8_t)(bit_len >> 16);
    ctx->buf[60] = (uint8_t)(bit_len >> 24);
    ctx->buf[59] = (uint8_t)(bit_len >> 32);
    ctx->buf[58] = (uint8_t)(bit_len >> 40);
    ctx->buf[57] = (uint8_t)(bit_len >> 48);
    ctx->buf[56] = (uint8_t)(bit_len >> 56);
    sha256_transform(ctx, ctx->buf);

    /* Produce big-endian output */
    for (i = 0; i < 4; ++i) {
        out[i]      = (uint8_t)(ctx->s[0] >> (24 - i * 8));
        out[i +  4] = (uint8_t)(ctx->s[1] >> (24 - i * 8));
        out[i +  8] = (uint8_t)(ctx->s[2] >> (24 - i * 8));
        out[i + 12] = (uint8_t)(ctx->s[3] >> (24 - i * 8));
        out[i + 16] = (uint8_t)(ctx->s[4] >> (24 - i * 8));
        out[i + 20] = (uint8_t)(ctx->s[5] >> (24 - i * 8));
        out[i + 24] = (uint8_t)(ctx->s[6] >> (24 - i * 8));
        out[i + 28] = (uint8_t)(ctx->s[7] >> (24 - i * 8));
    }
}
