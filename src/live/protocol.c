#include "quakecore_live/frame_protocol.h"
#include "sha256.h"
#include <stdio.h>

int qcfp_hash_file(const char* path, uint8_t out32[32]) {
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  QcfpSha256 ctx; qcfp_sha256_init(&ctx);
  uint8_t buf[8192];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) qcfp_sha256_update(&ctx, buf, n);
  int err = ferror(f) ? -1 : 0;
  fclose(f);
  if (err) return err;
  qcfp_sha256_final(&ctx, out32);
  return 0;
}
