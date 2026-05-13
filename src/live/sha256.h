#ifndef QCFP_SHA256_H
#define QCFP_SHA256_H
#include <stddef.h>
#include <stdint.h>
typedef struct { uint32_t s[8]; uint64_t len; uint8_t buf[64]; size_t buf_len; } QcfpSha256;
void qcfp_sha256_init(QcfpSha256* c);
void qcfp_sha256_update(QcfpSha256* c, const void* data, size_t n);
void qcfp_sha256_final(QcfpSha256* c, uint8_t out[32]);
#endif
