/***
 * Author: ZhuXiaolong
 * Date: 2022-11-14 21:04:54
 * LastEditTime: 2022-11-15 17:17:29
 * FilePath: /deploy_and_train/trt/src/plugins/centernet/deformable_conv2d_plugin.h
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
#ifndef TRT_SRC_PLUGINS_CENTERNET_DEFORMABLE_CONV2D_PLUGIN_H_
#define TRT_SRC_PLUGINS_CENTERNET_DEFORMABLE_CONV2D_PLUGIN_H_

#include "common.h"
#include <cublas_v2.h>

namespace dt_plugin {
namespace centernet {

class DeformableConv2dPlugin : public DynamicPlugin
{
public:
    DeformableConv2dPlugin(int stride, int dilation,
                           int padding, int weight_groups,
                           int offset_groups,
                           bool use_mask);
    DeformableConv2dPlugin(const void* buffer, size_t length);
    ~DeformableConv2dPlugin();
    
    size_t getWorkspaceSize(
        const PluginTensorDesc* inputs, int num_inputs,
        const PluginTensorDesc* outputs, int num_outputs
    ) const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int output_idx, const DimsExprs* inputs,
        int num_inputs, IExprBuilder& expr_builder
    ) noexcept override;
    
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* in_out,
        int num_inputs, int num_outputs
    ) noexcept override;
    
    int enqueue(const PluginTensorDesc* input_descs,
                const PluginTensorDesc* output_descs,
                const void* const* inputs,
                void* const* outputs, void* workspace,
                cudaStream_t stream) noexcept override;

    DataType getOutputDataType(
        int idx, const DataType* input_types,
        int num_inputs
    ) const noexcept override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;

private:
    int stride_;
    int dilation_;
    int padding_;
    int weight_grps_;
    int offset_grps_;
    bool use_mask_;

    cublasHandle_t handle_;
};

class DeformableConv2dPluginCreator : public DynamicPluginCreator
{
public:
    DeformableConv2dPluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(
        const char* name, const PluginFieldCollection* fc
    ) noexcept override;
    IPluginV2* deserializePlugin(
        const char* name, const void* buffer, size_t length
    ) noexcept override;

private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attrs_;
};
    
} // namespace centernet
} // namespace dt_plugin

extern "C" bool registerDeformableConv2dPlugin();

#endif  // TRT_SRC_PLUGINS_CENTERNET_DEFORMABLE_CONV2D_PLUGIN_H_