#include "transport_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int qcfp_parse_spec(const char* spec, QcfpSpec* out) {
  if (!spec || !out) return -1;
  memset(out, 0, sizeof *out);
  if (strncmp(spec, "shm:", 4) == 0) {
    out->kind = QCFP_SPEC_SHM;
    snprintf(out->shm_name, sizeof out->shm_name, "%s", spec + 4);
    return out->shm_name[0] ? 0 : -1;
  }
  if (strncmp(spec, "tcp-listen:", 11) == 0) {
    out->kind = QCFP_SPEC_TCP_LISTEN;
    out->port = atoi(spec + 11);
    return out->port > 0 ? 0 : -1;
  }
  if (strncmp(spec, "tcp:", 4) == 0) {
    const char* rest = spec + 4;
    const char* colon = strrchr(rest, ':');
    if (!colon) return -1;
    size_t hlen = (size_t)(colon - rest);
    if (hlen == 0 || hlen >= sizeof out->host) return -1;
    memcpy(out->host, rest, hlen); out->host[hlen] = 0;
    out->port = atoi(colon + 1);
    out->kind = QCFP_SPEC_TCP_CONNECT;
    return out->port > 0 ? 0 : -1;
  }
  return -1;
}

QcfpTransport* qcfp_open_producer(const char* spec) {
  QcfpSpec s; if (qcfp_parse_spec(spec, &s) != 0) return NULL;
  if (s.kind == QCFP_SPEC_SHM) return qcfp_shm_open_producer(s.shm_name);
  if (s.kind == QCFP_SPEC_TCP_CONNECT) return qcfp_tcp_connect(s.host, s.port);
  return NULL;
}
QcfpTransport* qcfp_open_consumer(const char* spec) {
  QcfpSpec s; if (qcfp_parse_spec(spec, &s) != 0) return NULL;
  if (s.kind == QCFP_SPEC_SHM) return qcfp_shm_open_consumer(s.shm_name);
  if (s.kind == QCFP_SPEC_TCP_LISTEN) return qcfp_tcp_listen(s.port);
  return NULL;
}
void qcfp_close(QcfpTransport* t) {
  if (!t) return;
  if (t->kind == QCFP_SPEC_SHM) qcfp_shm_close(t); else qcfp_tcp_close(t);
}
QcfpStatus qcfp_send(QcfpTransport* t, const void* p, size_t n) {
  return t->kind == QCFP_SPEC_SHM ? qcfp_shm_send(t,p,n) : qcfp_tcp_send(t,p,n);
}
QcfpStatus qcfp_recv(QcfpTransport* t, void* p, size_t cap, size_t* out) {
  return t->kind == QCFP_SPEC_SHM ? qcfp_shm_recv(t,p,cap,out) : qcfp_tcp_recv(t,p,cap,out);
}
uint64_t qcfp_dropped(const QcfpTransport* t) { return t ? t->dropped : 0; }
