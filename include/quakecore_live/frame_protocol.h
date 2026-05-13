#ifndef QUAKECORE_LIVE_FRAME_PROTOCOL_H
#define QUAKECORE_LIVE_FRAME_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QCFP_MAGIC   0x50464351u  /* 'Q','C','F','P' little-endian */
#define QCFP_VERSION 1u

enum { QCFP_TYPE_HANDSHAKE = 1, QCFP_TYPE_FRAME = 2 };

#pragma pack(push, 1)
typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t type;
  uint32_t length;  /* payload bytes following this header */
} QcfpHeader;

typedef struct {
  QcfpHeader hdr;
  uint8_t  bsp_sha256[32];
  uint16_t bsp_path_len;
  /* followed by bsp_path_len bytes, not null-terminated */
} QcfpHandshakePacket;

typedef struct { float n[3]; float d; } QcfpPlane;

typedef struct {
  QcfpHeader hdr;
  uint64_t  frame_id;
  uint64_t  t_game_cull_start_ns;
  uint64_t  t_game_cull_end_ns;
  uint32_t  game_visited_nodes;
  uint32_t  game_accepted_leafs;
  uint8_t   bsp_sha256[32];
  float     cam_origin[3];
  float     cam_forward[3];
  float     cam_right[3];
  float     cam_up[3];
  uint8_t   n_planes;          /* always 6 in v1 */
  uint8_t   _pad[3];
  QcfpPlane planes[6];
} QcfpFramePacket;
#pragma pack(pop)

typedef enum {
  QCFP_OK = 0,
  QCFP_DROPPED = 1,
  QCFP_EMPTY = 2,
  QCFP_EOF = 3,
  QCFP_ERROR = -1
} QcfpStatus;

typedef struct QcfpTransport QcfpTransport;

/* Transport spec strings:
 *   "shm:<name>"           — POSIX SHM SPSC ring
 *   "tcp:<host>:<port>"    — TCP client (Ironwail side)
 *   "tcp-listen:<port>"    — TCP listener (sidecar side)
 */
QcfpTransport* qcfp_open_producer(const char* spec);
QcfpTransport* qcfp_open_consumer(const char* spec);
void           qcfp_close(QcfpTransport* t);

QcfpStatus qcfp_send(QcfpTransport* t, const void* packet, size_t bytes);
QcfpStatus qcfp_recv(QcfpTransport* t, void* buf, size_t buf_bytes, size_t* out_bytes);

/* Drops accumulated on producer side (SHM only). 0 for TCP. */
uint64_t qcfp_dropped(const QcfpTransport* t);

/* sha256 of file contents; out32 must be 32 bytes. Returns 0 on success. */
int qcfp_hash_file(const char* path, uint8_t out32[32]);

#ifdef __cplusplus
}
#endif

#endif
