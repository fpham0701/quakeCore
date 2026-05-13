#include "transport_internal.h"
QcfpTransport* qcfp_tcp_connect(const char* host, int port) { (void)host; (void)port; return NULL; }
QcfpTransport* qcfp_tcp_listen(int port) { (void)port; return NULL; }
void qcfp_tcp_close(QcfpTransport* t) { (void)t; }
QcfpStatus qcfp_tcp_send(QcfpTransport* t, const void* p, size_t n) { (void)t; (void)p; (void)n; return QCFP_ERROR; }
QcfpStatus qcfp_tcp_recv(QcfpTransport* t, void* p, size_t cap, size_t* out) { (void)t; (void)p; (void)cap; (void)out; return QCFP_ERROR; }
