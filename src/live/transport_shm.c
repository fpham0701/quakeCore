#define _GNU_SOURCE
#include "transport_internal.h"
#include <fcntl.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define QCFP_SHM_CAPACITY 1024u
#define QCFP_SHM_SLOT_BYTES 256u

typedef struct {
  _Atomic uint64_t head;
  _Atomic uint64_t tail;
  uint32_t capacity;
  uint32_t slot_bytes;
  sem_t sem;
  uint8_t slots[QCFP_SHM_CAPACITY * QCFP_SHM_SLOT_BYTES];
  /* each slot: [uint32_t len][payload bytes] */
} QcfpShmRegion;

typedef struct {
  char name[128];
  QcfpShmRegion* region;
  int is_owner;  /* consumer owns the segment lifetime */
} QcfpShmImpl;

static char* prefix_name(const char* name, char out[256]) {
  snprintf(out, 256, "/%s", name);
  return out;
}

QcfpTransport* qcfp_shm_open_consumer(const char* name) {
  char shm_name[256]; prefix_name(name, shm_name);
  shm_unlink(shm_name);  /* clear any leaked segment */
  int fd = shm_open(shm_name, O_CREAT | O_RDWR | O_EXCL, 0600);
  if (fd < 0) return NULL;
  if (ftruncate(fd, sizeof(QcfpShmRegion)) != 0) { close(fd); shm_unlink(shm_name); return NULL; }
  void* m = mmap(NULL, sizeof(QcfpShmRegion), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (m == MAP_FAILED) { shm_unlink(shm_name); return NULL; }
  QcfpShmRegion* r = (QcfpShmRegion*)m;
  memset(r, 0, sizeof *r);
  atomic_store(&r->head, 0);
  atomic_store(&r->tail, 0);
  r->capacity = QCFP_SHM_CAPACITY;
  r->slot_bytes = QCFP_SHM_SLOT_BYTES;
  sem_init(&r->sem, 1, 0);

  QcfpTransport* t = calloc(1, sizeof *t);
  QcfpShmImpl* impl = calloc(1, sizeof *impl);
  snprintf(impl->name, sizeof impl->name, "%s", shm_name);
  impl->region = r; impl->is_owner = 1;
  t->kind = QCFP_SPEC_SHM; t->impl = impl;
  return t;
}

QcfpTransport* qcfp_shm_open_producer(const char* name) {
  char shm_name[256]; prefix_name(name, shm_name);
  int fd = shm_open(shm_name, O_RDWR, 0600);
  if (fd < 0) return NULL;
  void* m = mmap(NULL, sizeof(QcfpShmRegion), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (m == MAP_FAILED) return NULL;
  QcfpTransport* t = calloc(1, sizeof *t);
  QcfpShmImpl* impl = calloc(1, sizeof *impl);
  snprintf(impl->name, sizeof impl->name, "%s", shm_name);
  impl->region = (QcfpShmRegion*)m; impl->is_owner = 0;
  t->kind = QCFP_SPEC_SHM; t->impl = impl;
  return t;
}

void qcfp_shm_close(QcfpTransport* t) {
  if (!t) return;
  QcfpShmImpl* impl = t->impl;
  if (impl) {
    if (impl->is_owner && impl->region) { sem_destroy(&impl->region->sem); }
    if (impl->region) munmap(impl->region, sizeof(QcfpShmRegion));
    if (impl->is_owner) shm_unlink(impl->name);
    free(impl);
  }
  free(t);
}

QcfpStatus qcfp_shm_send(QcfpTransport* t, const void* p, size_t n) {
  QcfpShmImpl* impl = t->impl; QcfpShmRegion* r = impl->region;
  if (n + sizeof(uint32_t) > r->slot_bytes) return QCFP_ERROR;
  uint64_t head = atomic_load_explicit(&r->head, memory_order_relaxed);
  uint64_t tail = atomic_load_explicit(&r->tail, memory_order_acquire);
  if (head - tail >= r->capacity) { t->dropped++; return QCFP_DROPPED; }
  uint8_t* slot = &r->slots[(head & (r->capacity - 1)) * r->slot_bytes];
  uint32_t len = (uint32_t)n;
  memcpy(slot, &len, sizeof len);
  memcpy(slot + sizeof len, p, n);
  atomic_store_explicit(&r->head, head + 1, memory_order_release);
  sem_post(&r->sem);
  return QCFP_OK;
}

QcfpStatus qcfp_shm_recv(QcfpTransport* t, void* p, size_t cap, size_t* out) {
  QcfpShmImpl* impl = t->impl; QcfpShmRegion* r = impl->region;
  while (sem_wait(&r->sem) != 0) { /* EINTR retry */ }
  uint64_t tail = atomic_load_explicit(&r->tail, memory_order_relaxed);
  uint64_t head = atomic_load_explicit(&r->head, memory_order_acquire);
  if (tail >= head) return QCFP_EMPTY;
  uint8_t* slot = &r->slots[(tail & (r->capacity - 1)) * r->slot_bytes];
  uint32_t len; memcpy(&len, slot, sizeof len);
  if (len > cap) return QCFP_ERROR;
  memcpy(p, slot + sizeof len, len);
  if (out) *out = len;
  atomic_store_explicit(&r->tail, tail + 1, memory_order_release);
  return QCFP_OK;
}
