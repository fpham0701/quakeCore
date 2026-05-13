#define _GNU_SOURCE
#include "transport_internal.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

typedef struct { int fd; int listen_fd; } QcfpTcpImpl;

static int set_nodelay(int fd) {
  int one = 1; return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

QcfpTransport* qcfp_tcp_connect(const char* host, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return NULL;
  struct sockaddr_in sa; memset(&sa, 0, sizeof sa);
  sa.sin_family = AF_INET; sa.sin_port = htons((uint16_t)port);
  if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) { close(fd); return NULL; }
  if (connect(fd, (struct sockaddr*)&sa, sizeof sa) != 0) { close(fd); return NULL; }
  set_nodelay(fd);
  QcfpTransport* t = calloc(1, sizeof *t);
  QcfpTcpImpl* impl = calloc(1, sizeof *impl);
  impl->fd = fd; impl->listen_fd = -1;
  t->kind = QCFP_SPEC_TCP_CONNECT; t->impl = impl;
  return t;
}

QcfpTransport* qcfp_tcp_listen(int port) {
  int lfd = socket(AF_INET, SOCK_STREAM, 0);
  if (lfd < 0) return NULL;
  int reuse = 1; setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof reuse);
  struct sockaddr_in sa; memset(&sa, 0, sizeof sa);
  sa.sin_family = AF_INET; sa.sin_port = htons((uint16_t)port); sa.sin_addr.s_addr = INADDR_ANY;
  if (bind(lfd, (struct sockaddr*)&sa, sizeof sa) != 0) { close(lfd); return NULL; }
  if (listen(lfd, 1) != 0) { close(lfd); return NULL; }
  int afd = accept(lfd, NULL, NULL);
  if (afd < 0) { close(lfd); return NULL; }
  set_nodelay(afd);
  QcfpTransport* t = calloc(1, sizeof *t);
  QcfpTcpImpl* impl = calloc(1, sizeof *impl);
  impl->fd = afd; impl->listen_fd = lfd;
  t->kind = QCFP_SPEC_TCP_LISTEN; t->impl = impl;
  return t;
}

void qcfp_tcp_close(QcfpTransport* t) {
  if (!t) return;
  QcfpTcpImpl* impl = t->impl;
  if (impl) { if (impl->fd >= 0) close(impl->fd); if (impl->listen_fd >= 0) close(impl->listen_fd); free(impl); }
  free(t);
}

static int write_all(int fd, const void* p, size_t n) {
  const uint8_t* b = p; while (n) {
    ssize_t k = send(fd, b, n, MSG_NOSIGNAL);
    if (k < 0) { if (errno == EINTR) continue; return -1; }
    if (k == 0) return -1;
    b += k; n -= (size_t)k;
  } return 0;
}
static int read_all(int fd, void* p, size_t n) {
  uint8_t* b = p; while (n) {
    ssize_t k = recv(fd, b, n, 0);
    if (k < 0) { if (errno == EINTR) continue; return -1; }
    if (k == 0) return 1; /* EOF */
    b += k; n -= (size_t)k;
  } return 0;
}

QcfpStatus qcfp_tcp_send(QcfpTransport* t, const void* p, size_t n) {
  QcfpTcpImpl* impl = t->impl;
  uint32_t len = (uint32_t)n;
  if (write_all(impl->fd, &len, sizeof len) != 0) return QCFP_ERROR;
  if (write_all(impl->fd, p, n) != 0) return QCFP_ERROR;
  return QCFP_OK;
}

QcfpStatus qcfp_tcp_recv(QcfpTransport* t, void* p, size_t cap, size_t* out) {
  QcfpTcpImpl* impl = t->impl;
  uint32_t len = 0;
  int r = read_all(impl->fd, &len, sizeof len);
  if (r == 1) return QCFP_EOF;
  if (r != 0) return QCFP_ERROR;
  if (len > cap) return QCFP_ERROR;
  r = read_all(impl->fd, p, len);
  if (r == 1) return QCFP_EOF;
  if (r != 0) return QCFP_ERROR;
  if (out) *out = len;
  return QCFP_OK;
}
