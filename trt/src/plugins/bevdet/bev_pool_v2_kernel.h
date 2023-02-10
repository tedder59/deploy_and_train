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

void bev_pool_v2_set_zero(int n_points, float* out);
void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat, const int* ranks_depth, const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* out, cudaStream_t stream);

#endif  // TRT_SRC_PLUGINS_CENTERNET_DEFORMABLE_CONV2D_H_
