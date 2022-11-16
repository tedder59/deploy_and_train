/***
 * Author: ZhuXiaolong
 * Date: 2022-11-15 17:32:44
 * LastEditTime: 2022-11-16 17:08:00
 * FilePath: /deploy_and_train/trt/src/plugins/centernet/deformable_conv2d.cu
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
#include "centernet/deformable_conv2d.h"
#include "common.h"

template <typename T>
__global__
void fill_bias_kernel(const T* bias, int s, T* out)
{
    __shared__ T b[1];
    if (threadIdx.x == 0) {
        b[0] = bias[blockIdx.x];
    }
    __syncthreads();
    
    out += blockIdx.x * s;
    for (int i = threadIdx.x; i < s; i += blockDim.x) {
        out[i] = b[0];
    }
}

template <typename T>
int fill_bias(const T* bias, int channels, int spatial,
              T* out, cudaStream_t stream)
{
    int THREAD_NUM = 512;
    int BLOCK_NUM = channels;

    dim3 block_dim(THREAD_NUM);
    dim3 grid_dim(BLOCK_NUM);
    fill_bias_kernel<<<grid_dim, block_dim, 0, stream>>>(
        bias, spatial, out
    );

    KERNEL_LAUNCH_CHECK();
    return 0;
}

template int fill_bias<float>(const float*, int, int, float*, cudaStream_t);
template int fill_bias<half>(const half*, int, int, half*, cudaStream_t);

template <typename T>
__device__
T _bilinear(const T* im, int h, int w, T x, T y)
{
    int x1 = floor(x);
    int y1 = floor(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    T v1 = 0;
    if (x1 >= 0 && y1 >= 0) v1 = im[y1 * w + x1];

    T v2 = 0;
    if (x2 < w && y1 >= 0) v2 = im[y1 * w + x2];

    T v3 = 0;
    if (x1 >= 0 && y2 < h) v3 = im[y2 * w + x1];

    T v4 = 0;
    if (x2 < w && y2 < h) v4 = im[y2 * w + x2];

    T alpha = y - y1;
    T beta = x - x1;
    T v = (1 - alpha) * (1 - beta) * v1
            + (1 - alpha) * beta * v2
            + alpha * (1 - beta) * v3
            + alpha * beta * v4;
    return v;
}

template <>
__device__
half _bilinear(const half* im, int h, int w, half x, half y)
{
    int x1 = hfloor(x);
    int y1 = hfloor(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    half v1 = 0;
    if (x1 >= 0 && y1 >= 0) v1 = im[y1 * w + x1];

    half v2 = 0;
    if (x2 < w && y1 >= 0) v2 = im[y1 * w + x2];

    half v3 = 0;
    if (x1 >= 0 && y2 < h) v3 = im[y2 * w + x1];

    half v4 = 0;
    if (x2 < w && y2 < h) v4 = im[y2 * w + x2];

    half alpha = y - half(y1);
    half beta = x - half(x1);
    half one(1);
    half v = (one - alpha) * (one - beta) * v1
           + (one - alpha) * beta * v2
           + alpha * (one - beta) * v3
           + alpha * beta * v4;
    return v;
}

template <typename T>
__global__
void deformable_im2col_kernel(const T* input, const T* offset,
                              int in_h, int in_w, int out_h, int out_w,
                              int kernel_size, int stride,
                              int padding, int dilation,
                              T* out)
{
    offset += blockIdx.y * kernel_size * kernel_size * 2 * out_h * out_w;
    input += blockIdx.x * in_h * in_w;
    out += blockIdx.x * kernel_size * kernel_size * out_h * out_w;

    for (int i = threadIdx.x; i < out_h * out_w; i += blockDim.x) {
        int ox = i % out_w;
        int oy = i / out_w;

        for (int j = 0; j < kernel_size * kernel_size; ++j) {
            int kx = j % kernel_size;
            int ky = j / kernel_size;

            T off_y = offset[((2 * j + 0) * out_h + oy) * out_w + ox];
            T off_x = offset[((2 * j + 1) * out_h + oy) * out_w + ox];

            T y = T(oy * stride - padding + ky * dilation) + off_y;
            T x = T(ox * stride - padding + kx * dilation) + off_x;

            T val = 0;
            if (y > T(-1) && y < T(in_h) && x > T(-1) && x < T(in_w))
                val = _bilinear(input, in_h, in_w, x, y);

            out[(j * out_h + oy) * out_w + ox] = val;
        }
    }
}

template <typename T>
int deformable_im2col(const T* input, const T* offset,
                      int in_channels, int in_h, int in_w,
                      int offset_grps, int kernel, int stride,
                      int dilation, int padding, T* col,
                      cudaStream_t stream)
{
    int out_h = (in_h + 2 * padding - ((kernel - 1) * dilation + 1)) / stride + 1;
    int out_w = (in_w + 2 * padding - ((kernel - 1) * dilation + 1)) / stride + 1;
    int THREAD_NUM = 1024;
    int channels_per_group = in_channels / offset_grps;

    dim3 block_dim(THREAD_NUM);
    dim3 grid_dim(channels_per_group, offset_grps);
    deformable_im2col_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input, offset, in_h, in_w, out_h, out_w, kernel, stride,
        padding, dilation, col
    );

    KERNEL_LAUNCH_CHECK();
    return 0;
}

template int deformable_im2col<float>(const float*, const float*, int, int, int, int, int, int, int, int, float*, cudaStream_t);
template int deformable_im2col<half>(const half*, const half*, int, int, int, int, int, int, int, int, half*, cudaStream_t);

int deformable_mask_im2col(const float* input, const float* offset,
                           const float* mask,
                           int in_channels, int in_h, int in_w,
                           int offset_grps, int kernel,
                           int stride, int dilation, int padding,
                           float* col, cudaStream_t stream)
{
    KERNEL_LAUNCH_CHECK();
    return 0;
}
