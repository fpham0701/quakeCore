#include "transport_internal.h"
QcfpTransport* qcfp_shm_open_producer(const char* name) { (void)name; return NULL; }
QcfpTransport* qcfp_shm_open_consumer(const char* name) { (void)name; return NULL; }
void qcfp_shm_close(QcfpTransport* t) { (void)t; }
QcfpStatus qcfp_shm_send(QcfpTransport* t, const void* p, size_t n) { (void)t; (void)p; (void)n; return QCFP_ERROR; }
QcfpStatus qcfp_shm_recv(QcfpTransport* t, void* p, size_t cap, size_t* out) { (void)t; (void)p; (void)cap; (void)out; return QCFP_ERROR; }
