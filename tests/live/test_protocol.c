#include "quakecore_live/frame_protocol.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void hex(const uint8_t* in, size_t n, char* out) {
  static const char* H = "0123456789abcdef";
  for (size_t i = 0; i < n; ++i) { out[2*i] = H[in[i]>>4]; out[2*i+1] = H[in[i]&0xF]; }
  out[2*n] = 0;
}

int main(void) {
  /* sha256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad */
  const char* tmp_path = "/tmp/qcfp_test_abc.bin";
  FILE* f = fopen(tmp_path, "wb"); assert(f); fwrite("abc", 1, 3, f); fclose(f);

  uint8_t out[32];
  int rc = qcfp_hash_file(tmp_path, out);
  assert(rc == 0);
  char hexbuf[65];
  hex(out, 32, hexbuf);
  const char* expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  if (strcmp(hexbuf, expected) != 0) {
    fprintf(stderr, "sha256 mismatch: got %s\n", hexbuf);
    return 1;
  }
  printf("sha256 ok\n");
  return 0;
}
