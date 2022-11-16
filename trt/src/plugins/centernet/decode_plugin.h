/***
 * Author: ZhuXiaolong
 * Date: 2022-11-14 13:13:17
 * LastEditTime: 2022-11-15 10:30:04
 * FilePath: /deploy_and_train/trt/src/plugins/centernet/decode_plugin.h
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
#ifndef TRT_SRC_PLUGINS_CENTERNET_DECODE_PLUGIN_H_
#define TRT_SRC_PLUGINS_CENTERNET_DECODE_PLUGIN_H_

#include "common.h"

namespace dt_plugin {
namespace centernet{

class DecodePlugin : public DynamicPlugin
{
public:
    DecodePlugin();
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
                const void* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) noexcept override;

    DataType getOutputDataType(
        int idx, const DataType* input_types, int num_inputs
    ) const noexcept override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
};

class DecodePluginCreator : public DynamicPluginCreator
{
public:
    DecodePluginCreator();
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

extern "C" bool registerCenterNetDecodePlugin();

#endif  // TRT_SRC_PLUGINS_CENTERNET_DECODE_PLUGIN_H_