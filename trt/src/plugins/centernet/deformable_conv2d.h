/***
 * Author: ZhuXiaolong
 * Date: 2022-11-15 17:19:52
 * LastEditTime: 2022-11-16 17:04:23
 * FilePath: /deploy_and_train/trt/src/plugins/centernet/deformable_conv2d.h
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
#ifndef TRT_SRC_PLUGINS_CENTERNET_DEFORMABLE_CONV2D_H_
#define TRT_SRC_PLUGINS_CENTERNET_DEFORMABLE_CONV2D_H_

#include <cuda_runtime_api.h>

template <typename T>
int fill_bias(const T* bias, int channels, int spatial,
              T* out, cudaStream_t stream);

template <typename T>
int deformable_im2col(const T* input, const T* offset,
                      int in_channels, int in_h, int in_w,
                      int offset_grps,
                      int kernel, int stride, int dilation,
                      int padding, T* col,
                      cudaStream_t stream);

#endif  // TRT_SRC_PLUGINS_CENTERNET_DEFORMABLE_CONV2D_H_