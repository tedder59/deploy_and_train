/***
 * Author: ZhuXiaolong
 * Date: 2022-11-14 15:23:33
 * LastEditTime: 2022-11-15 14:14:35
 * FilePath: /deploy_and_train/trt/src/plugins/centernet/decode_plugin.cpp
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
#include "centernet/decode_plugin.h"
#include "centernet/decode.h"

static constexpr char PLUGIN_VERSION[] = "1";
static constexpr char PLUGIN_NAME[] = "CenterNetDecode";

namespace dt_plugin {
namespace centernet {

DecodePlugin::DecodePlugin()
{

}

IPluginV2DynamicExt* DecodePlugin::clone() const noexcept
{
    return new DecodePlugin();
}

DimsExprs DecodePlugin::getOutputDimensions(
    int output_idx, const DimsExprs* inputs,
    int num_inputs, IExprBuilder& expr_builder
) noexcept
{
    DimsExprs out;
    out.nbDims = 3;

    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[1];
    out.d[2] = expr_builder.constant(6);

    return out;
}
    
bool DecodePlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* in_out,
    int num_inputs, int num_outputs
) noexcept
{
    if (num_inputs != 4 || num_outputs != 1)
        return false;

    auto desc = in_out[pos];
    switch (pos)
    {
    case 0:
        return (desc.type == DataType::kINT32) &&
               (desc.format == TensorFormat::kLINEAR);
    case 1:
    case 2:
    case 3:
    case 4:
        return (desc.type == DataType::kFLOAT) &&
               (desc.format == TensorFormat::kLINEAR);
    default:
        return false;
    }
}

int DecodePlugin::enqueue(
    const PluginTensorDesc* input_descs,
    const PluginTensorDesc* output_descs,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream
) noexcept
{
    int b = input_descs[0].dims.d[0];
    int n = input_descs[0].dims.d[1];
    int h = input_descs[2].dims.d[2];
    int w = input_descs[2].dims.d[3];

    const int* topk_idxes = static_cast<const int*>(inputs[0]);
    const float* topk_scores = static_cast<const float*>(inputs[1]);
    const float* reg = static_cast<const float*>(inputs[2]);
    const float* wh = static_cast<const float*>(inputs[3]);
    float* dets = static_cast<float*>(outputs[0]);

    int ret = centernet_decode(topk_idxes, topk_scores, reg, wh,
                               b, n, h, w, dets, stream);
    return ret;
}

DataType DecodePlugin::getOutputDataType(
    int idx, const DataType* input_types, int num_inputs
) const noexcept
{
    return DataType::kFLOAT;
}

const char* DecodePlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* DecodePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

PluginFieldCollection DecodePluginCreator::fc_{0, nullptr};
std::vector<PluginField> DecodePluginCreator::attrs_;

DecodePluginCreator::DecodePluginCreator()
{

}

const char* DecodePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* DecodePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* DecodePluginCreator::getFieldNames() noexcept
{
    return &fc_;
}

IPluginV2* DecodePluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc
) noexcept
{
    return new DecodePlugin();
}

IPluginV2* DecodePluginCreator::deserializePlugin(
    const char* name, const void* buffer, size_t length
) noexcept
{
    return new DecodePlugin();
}
    
} // namespace centernet
} // namespace dt_plugin

DEFINE_DT_PLUGIN(CenterNetDecodePlugin,
                 dt_plugin::centernet::DecodePluginCreator)
