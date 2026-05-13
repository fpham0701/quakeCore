#include "quakecore_live/frame_protocol.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
  /* Consumer first (cleans up any leftover segment). */
  QcfpTransport* cons = qcfp_open_consumer("shm:qcfp_test_ring");
  assert(cons);
  QcfpTransport* prod = qcfp_open_producer("shm:qcfp_test_ring");
  assert(prod);

  for (uint32_t i = 0; i < 100; ++i) {
    QcfpFramePacket pkt; memset(&pkt, 0, sizeof pkt);
    pkt.hdr.magic = QCFP_MAGIC; pkt.hdr.version = QCFP_VERSION;
    pkt.hdr.type = QCFP_TYPE_FRAME; pkt.hdr.length = sizeof pkt - sizeof pkt.hdr;
    pkt.frame_id = i;
    assert(qcfp_send(prod, &pkt, sizeof pkt) == QCFP_OK);
  }
  for (uint32_t i = 0; i < 100; ++i) {
    QcfpFramePacket got; size_t bytes = 0;
    QcfpStatus s = qcfp_recv(cons, &got, sizeof got, &bytes);
    assert(s == QCFP_OK); assert(bytes == sizeof got);
    assert(got.frame_id == i);
  }

  /* Overrun: ring is 1024 slots. Send 2000 without consuming; expect ~1000 drops. */
  for (uint32_t i = 0; i < 2000; ++i) {
    QcfpFramePacket pkt; memset(&pkt, 0, sizeof pkt);
    pkt.hdr.magic = QCFP_MAGIC; pkt.hdr.version = QCFP_VERSION;
    pkt.hdr.type = QCFP_TYPE_FRAME; pkt.hdr.length = sizeof pkt - sizeof pkt.hdr;
    qcfp_send(prod, &pkt, sizeof pkt);
  }
  uint64_t drops = qcfp_dropped(prod);
  printf("drops=%llu (expected ~976)\n", (unsigned long long)drops);
  assert(drops >= 900 && drops <= 1100);

  qcfp_close(prod); qcfp_close(cons);
  printf("shm ring ok\n");
  return 0;
}
