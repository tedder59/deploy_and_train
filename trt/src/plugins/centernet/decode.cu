/***
 * Author: ZhuXiaolong
 * Date: 2022-11-14 20:48:48
 * LastEditTime: 2022-11-15 19:20:29
 * FilePath: /deploy_and_train/trt/src/plugins/centernet/decode.cu
 * Description: 
 * Copyright (c) 2022 by ZhuXiaolong, All Rights Reserved.
 * Licensed under the BSD 3-Clause License (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * https://opensource.org/licenses/BSD-3-Clause
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed
 * under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*/
#include "decode.h"

__global__
void decode_kernel(const int* topk_idx, const float* topk_score,
                   const float* reg, const float* wh,
                   int b, int n, int spatial_stride,
                   int row_stride, float* out)
{
    if (threadIdx.x >= n) return;
    topk_idx += blockIdx.x * n;
    topk_score += blockIdx.x * n;

    int idx = topk_idx[threadIdx.x];
    float score = topk_score[threadIdx.x];

    int c = idx / spatial_stride;
    idx = idx % spatial_stride;
    
    int y = idx / row_stride;
    int x = idx % row_stride;

    size_t offset = blockIdx.x * spatial_stride * 2;
    reg += offset;
    wh += offset;

    offset = y * row_stride + x;
    float xc = x + reg[offset];
    float yc = y + reg[offset + spatial_stride];
    float w = wh[offset] * 0.5f;
    float h = wh[offset + spatial_stride] * 0.5f;

    out += (blockIdx.x * n + threadIdx.x) * 6;
    out[0] = c;
    out[1] = score;
    out[2] = xc - w;
    out[3] = yc - h;
    out[4] = xc + w;
    out[5] = yc + h;
}

int centernet_decode(const int* topk_idx,
                     const float* topk_score,
                     const float* reg,
                     const float* wh,
                     int b, int n, int h, int w,
                     float* dets,
                     cudaStream_t stream)
{
    int THREAD_NUM = 128;
    int BLOCK_NUM = b;

    dim3 block_dim(THREAD_NUM);
    dim3 grid_dim(BLOCK_NUM);
    decode_kernel<<<grid_dim, block_dim, 0, stream>>>(
        topk_idx, topk_score, reg, wh, b, n, h * w, w, dets
    );

    cudaError_t err = cudaGetLastError();
    return err;
}
