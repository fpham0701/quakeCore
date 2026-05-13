#include "frame_probe.h"
#include "quakecore_live/frame_protocol.h"

/* Forward declarations of Ironwail globals — actual headers added during patch apply. */
extern struct { float vieworg[3]; float vpn[3]; float vright[3]; float vup[3]; } r_refdef;
extern struct cplane_s { float normal[3]; float dist; unsigned char type; unsigned char signbits; unsigned char pad[2]; } frustum[4];
extern struct { float time; struct model_s* worldmodel; } cl;
struct model_s { char name[64]; /* ... */ };

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static QcfpTransport* s_tx = NULL;
static uint64_t s_frame_id = 0;
static uint64_t s_cull_start_ns = 0;
static uint64_t s_cull_end_ns   = 0;
static int      s_handshake_sent = 0;
static uint8_t  s_bsp_sha256[32];
static char     s_bsp_path[1024];

static uint64_t now_ns(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void FrameProbe_Init(const char* spec, int handshake_once) {
  if (!spec || !*spec) return;
  s_tx = qcfp_open_producer(spec);
  if (!s_tx) { fprintf(stderr, "[frame_probe] cannot open %s\n", spec); return; }
  /* Locate BSP path. Ironwail stores cl.worldmodel->name like "maps/e1m1.bsp". */
  snprintf(s_bsp_path, sizeof s_bsp_path, "id1/%s", cl.worldmodel ? cl.worldmodel->name : "unknown");
  qcfp_hash_file(s_bsp_path, s_bsp_sha256);
  if (handshake_once) {
    uint8_t hsbuf[sizeof(QcfpHandshakePacket) + 1024];
    QcfpHandshakePacket* hs = (QcfpHandshakePacket*)hsbuf;
    size_t plen = strlen(s_bsp_path);
    hs->hdr.magic = QCFP_MAGIC; hs->hdr.version = QCFP_VERSION;
    hs->hdr.type = QCFP_TYPE_HANDSHAKE;
    hs->hdr.length = (uint32_t)(sizeof(QcfpHandshakePacket) + plen - sizeof(QcfpHeader));
    memcpy(hs->bsp_sha256, s_bsp_sha256, 32);
    hs->bsp_path_len = (uint16_t)plen;
    memcpy(hsbuf + sizeof(QcfpHandshakePacket), s_bsp_path, plen);
    qcfp_send(s_tx, hsbuf, sizeof(QcfpHandshakePacket) + plen);
    s_handshake_sent = 1;
  }
}

void FrameProbe_Shutdown(void) { if (s_tx) { qcfp_close(s_tx); s_tx = NULL; } }
void FrameProbe_BeginCull(void) { s_cull_start_ns = now_ns(); }

uint32_t g_frame_probe_visited = 0;
uint32_t g_frame_probe_accepted = 0;

void FrameProbe_EndCull(uint32_t vn, uint32_t al) {
  s_cull_end_ns = now_ns();
  g_frame_probe_visited = vn; g_frame_probe_accepted = al;
}

void FrameProbe_Emit(void) {
  if (!s_tx) return;
  if (!s_handshake_sent) {
    FrameProbe_Init(NULL, 1); /* won't reopen tx — guarded above by *spec check */
  }
  QcfpFramePacket p;
  memset(&p, 0, sizeof p);
  p.hdr.magic = QCFP_MAGIC; p.hdr.version = QCFP_VERSION;
  p.hdr.type = QCFP_TYPE_FRAME;
  p.hdr.length = sizeof p - sizeof(QcfpHeader);
  p.frame_id = s_frame_id++;
  p.t_game_cull_start_ns = s_cull_start_ns;
  p.t_game_cull_end_ns   = s_cull_end_ns;
  p.game_visited_nodes   = g_frame_probe_visited;
  p.game_accepted_leafs  = g_frame_probe_accepted;
  memcpy(p.bsp_sha256, s_bsp_sha256, 32);
  memcpy(p.cam_origin,  r_refdef.vieworg, sizeof p.cam_origin);
  memcpy(p.cam_forward, r_refdef.vpn,     sizeof p.cam_forward);
  memcpy(p.cam_right,   r_refdef.vright,  sizeof p.cam_right);
  memcpy(p.cam_up,      r_refdef.vup,     sizeof p.cam_up);
  p.n_planes = 6;
  for (int i = 0; i < 4; ++i) {
    p.planes[i].n[0] = frustum[i].normal[0];
    p.planes[i].n[1] = frustum[i].normal[1];
    p.planes[i].n[2] = frustum[i].normal[2];
    p.planes[i].d    = frustum[i].dist;
  }
  /* Synthetic near plane: forward-facing, ~1e-3 in front of eye */
  p.planes[4].n[0] = r_refdef.vpn[0]; p.planes[4].n[1] = r_refdef.vpn[1]; p.planes[4].n[2] = r_refdef.vpn[2];
  p.planes[4].d    = r_refdef.vieworg[0]*r_refdef.vpn[0] + r_refdef.vieworg[1]*r_refdef.vpn[1] + r_refdef.vieworg[2]*r_refdef.vpn[2] + 1e-3f;
  /* Synthetic far plane: backward-facing, ~1e9 behind eye — never culls anything inside the world */
  p.planes[5].n[0] = -r_refdef.vpn[0]; p.planes[5].n[1] = -r_refdef.vpn[1]; p.planes[5].n[2] = -r_refdef.vpn[2];
  p.planes[5].d    = -(r_refdef.vieworg[0]*r_refdef.vpn[0] + r_refdef.vieworg[1]*r_refdef.vpn[1] + r_refdef.vieworg[2]*r_refdef.vpn[2]) + 1e9f;
  (void)qcfp_send(s_tx, &p, sizeof p);  /* fire-and-forget; DROPPED is fine */
}
