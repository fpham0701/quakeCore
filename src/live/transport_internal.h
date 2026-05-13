#ifndef QCFP_TRANSPORT_INTERNAL_H
#define QCFP_TRANSPORT_INTERNAL_H
#include "quakecore_live/frame_protocol.h"

typedef enum { QCFP_SPEC_SHM, QCFP_SPEC_TCP_CONNECT, QCFP_SPEC_TCP_LISTEN } QcfpSpecKind;
typedef struct { QcfpSpecKind kind; char shm_name[128]; char host[128]; int port; } QcfpSpec;

int qcfp_parse_spec(const char* spec, QcfpSpec* out);

/* Internal transport object — opaque to header. */
struct QcfpTransport {
  QcfpSpecKind kind;
  void* impl;       /* shm ring or tcp socket state */
  uint64_t dropped; /* producer-only */
};

QcfpTransport* qcfp_shm_open_producer(const char* name);
QcfpTransport* qcfp_shm_open_consumer(const char* name);
void qcfp_shm_close(QcfpTransport* t);
QcfpStatus qcfp_shm_send(QcfpTransport* t, const void* p, size_t n);
QcfpStatus qcfp_shm_recv(QcfpTransport* t, void* p, size_t cap, size_t* out);

QcfpTransport* qcfp_tcp_connect(const char* host, int port);
QcfpTransport* qcfp_tcp_listen(int port);
void qcfp_tcp_close(QcfpTransport* t);
QcfpStatus qcfp_tcp_send(QcfpTransport* t, const void* p, size_t n);
QcfpStatus qcfp_tcp_recv(QcfpTransport* t, void* p, size_t cap, size_t* out);

#endif
