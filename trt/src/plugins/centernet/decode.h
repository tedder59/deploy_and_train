#ifndef TRT_SRC_PLUGINS_CENTERNET_DECODE_H_
#define TRT_SRC_PLUGINS_CENTERNET_DECODE_H_

#include <cuda_runtime_api.h>

int centernet_decode(const int* topk_idx,
                     const float* topk_score,
                     const float* reg,
                     const float* wh,
                     int b, int n, int h, int w,
                     float* dets,
                     cudaStream_t stream);

#endif  // TRT_SRC_PLUGINS_CENTERNET_DECODE_H_