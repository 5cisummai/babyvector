#ifndef BABYVECTOR_H
#define BABYVECTOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BvHandle BvHandle;

/* Returns handle or NULL on error (see bv_last_error). */
BvHandle *bv_create(const char *dir, uint32_t dim);
BvHandle *bv_open(const char *dir);

/* 0 on success, -1 on error. */
int32_t bv_close(BvHandle *handle);

/* Remove index directory at dir. 0 on success. */
int32_t bv_drop(const char *dir);

/* ids: n * 16 bytes. vectors: n * dim floats. */
int32_t bv_add_vectors(BvHandle *handle, const uint8_t *ids, const float *vectors, uint32_t n);
int32_t bv_publish(BvHandle *handle);

/* out_ids: k * 16 bytes. out_scores: k floats. Returns hit count (<= k). */
uint32_t bv_search(BvHandle *handle, const float *query, uint32_t k, uint8_t *out_ids, float *out_scores);

const char *bv_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
