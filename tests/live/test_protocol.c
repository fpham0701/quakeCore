#include "quakecore_live/frame_protocol.h"
#include "../../src/live/transport_internal.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

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

  {
    QcfpSpec s;
    assert(qcfp_parse_spec("shm:foo", &s) == 0);
    assert(s.kind == QCFP_SPEC_SHM && strcmp(s.shm_name, "foo") == 0);
    assert(qcfp_parse_spec("tcp:1.2.3.4:5000", &s) == 0);
    assert(s.kind == QCFP_SPEC_TCP_CONNECT && strcmp(s.host, "1.2.3.4") == 0 && s.port == 5000);
    assert(qcfp_parse_spec("tcp-listen:5000", &s) == 0);
    assert(s.kind == QCFP_SPEC_TCP_LISTEN && s.port == 5000);
    assert(qcfp_parse_spec("bogus", &s) != 0);
    printf("spec parser ok\n");
  }

  {
  /* Fork: child = producer, parent = consumer. */
  pid_t pid = fork();
  if (pid == 0) {
    /* child: connect after small delay */
    usleep(200 * 1000);
    QcfpTransport* prod = qcfp_open_producer("tcp:127.0.0.1:23456");
    if (!prod) _exit(2);
    QcfpFramePacket pkt; memset(&pkt, 0, sizeof pkt);
    pkt.hdr.magic = QCFP_MAGIC; pkt.hdr.version = QCFP_VERSION;
    pkt.hdr.type = QCFP_TYPE_FRAME; pkt.hdr.length = sizeof pkt - sizeof pkt.hdr;
    pkt.frame_id = 42;
    QcfpStatus s = qcfp_send(prod, &pkt, sizeof pkt);
    qcfp_close(prod);
    _exit(s == QCFP_OK ? 0 : 3);
  }
  QcfpTransport* cons = qcfp_open_consumer("tcp-listen:23456");
  assert(cons);
  QcfpFramePacket got; size_t bytes = 0;
  QcfpStatus s = qcfp_recv(cons, &got, sizeof got, &bytes);
  assert(s == QCFP_OK && bytes == sizeof got);
  assert(got.frame_id == 42);
  int status; waitpid(pid, &status, 0); assert(WEXITSTATUS(status) == 0);
  qcfp_close(cons);
  printf("tcp roundtrip ok\n");
  }

  return 0;
}
