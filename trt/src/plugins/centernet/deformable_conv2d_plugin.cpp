/***
 * Author: ZhuXiaolong
 * Date: 2022-11-14 21:01:32
 * LastEditTime: 2022-11-16 18:39:17
 * FilePath: /deploy_and_train/trt/src/plugins/centernet/deformable_conv2d_plugin.cpp
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

#include "deformable_conv2d_plugin.h"
#include "deformable_conv2d.h"
#include <iostream>

static constexpr char PLUGIN_VERSION[] = "1";
static constexpr char PLUGIN_NAME[] = "DeformConv2d";

namespace dt_plugin {
namespace centernet {

DeformableConv2dPlugin::DeformableConv2dPlugin(
    int stride, int dilation, int padding,
    int weight_groups, int offset_groups,
    bool use_mask
) : stride_(stride), dilation_(dilation), padding_(padding)
  , weight_grps_(weight_groups), offset_grps_(offset_groups)
  , use_mask_(use_mask)
{
    cublasStatus_t stat = cublasCreate(&handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        handle_ = nullptr;
    }

    assert(!use_mask);
}

DeformableConv2dPlugin::DeformableConv2dPlugin(const void* buffer, size_t length)
{
    stride_ = readBuffer<int>(buffer);
    dilation_ = readBuffer<int>(buffer);
    padding_ = readBuffer<int>(buffer);
    weight_grps_ = readBuffer<int>(buffer);
    offset_grps_ = readBuffer<int>(buffer);
    use_mask_ = readBuffer<bool>(buffer);

    cublasStatus_t stat = cublasCreate(&handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        handle_ = nullptr;
    }
}

DeformableConv2dPlugin::~DeformableConv2dPlugin()
{
    if (handle_) {
        cublasDestroy(handle_);
    }
}

size_t DeformableConv2dPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * 5 + sizeof(bool);
}

void DeformableConv2dPlugin::serialize(void* buffer) const noexcept
{
    writeBuffer(buffer, stride_);
    writeBuffer(buffer, dilation_);
    writeBuffer(buffer, padding_);
    writeBuffer(buffer, weight_grps_);
    writeBuffer(buffer, offset_grps_);
    writeBuffer(buffer, use_mask_);
}

IPluginV2DynamicExt* DeformableConv2dPlugin::clone() const noexcept
{
    return new DeformableConv2dPlugin(*this);
}

DimsExprs DeformableConv2dPlugin::getOutputDimensions(
    int output_idx, const DimsExprs* inputs,
    int num_inputs, IExprBuilder& expr_builder
) noexcept
{
    DimsExprs out;
    out.nbDims = 4;

    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[1].d[0];
    out.d[2] = expr_builder.operation(DimensionOperation::kSUM, *expr_builder.operation(DimensionOperation::kFLOOR_DIV, *expr_builder.operation(DimensionOperation::kSUB, *expr_builder.operation(DimensionOperation::kSUM, *(inputs[0].d[2]), *expr_builder.constant(2 * padding_)), 
                                                                                                                                                                          *expr_builder.operation(DimensionOperation::kSUM, *expr_builder.operation(DimensionOperation::kPROD, *expr_builder.operation(DimensionOperation::kSUB, *(inputs[1].d[2]),
                                                                                                                                                                                                                                                                                                                                 *expr_builder.constant(1)),
                                                                                                                                                                                                                                                                               *expr_builder.constant(dilation_)),
                                                                                                                                                                                                                            *expr_builder.constant(1))),
                                                                                                                        *expr_builder.constant(stride_)),
                                                                *expr_builder.constant(1));
    out.d[3] = expr_builder.operation(DimensionOperation::kSUM, *expr_builder.operation(DimensionOperation::kFLOOR_DIV, *expr_builder.operation(DimensionOperation::kSUB, *expr_builder.operation(DimensionOperation::kSUM, *(inputs[0].d[3]), *expr_builder.constant(2 * padding_)),
                                                                                                                                                                          *expr_builder.operation(DimensionOperation::kSUM, *expr_builder.operation(DimensionOperation::kPROD, *expr_builder.operation(DimensionOperation::kSUB, *(inputs[1].d[3]),
                                                                                                                                                                                                                                                                                                                                 *expr_builder.constant(1)),
                                                                                                                                                                                                                                                                               *expr_builder.constant(dilation_)),
                                                                                                                                                                                                                            *expr_builder.constant(1))),
                                                                                                                        *expr_builder.constant(stride_)),
                                                                *expr_builder.constant(1));

    return out;
}

bool DeformableConv2dPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* in_out,
    int num_inputs, int num_outputs
) noexcept
{
    if (num_inputs != 5 || num_outputs != 1)
        return false;

    auto desc = in_out[pos];
    return (desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) && desc.format == TensorFormat::kLINEAR;
}

size_t DeformableConv2dPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int num_inputs,
    const PluginTensorDesc* outputs, int num_outputs
) const noexcept
{ 
    int batch = inputs[0].dims.d[0];
    int in_channels = inputs[0].dims.d[1];
    int in_h = inputs[0].dims.d[2];
    int in_w = inputs[0].dims.d[3];

    int kernel_size = inputs[1].dims.d[2];
    int out_h = (in_h + 2 * padding_ - ((kernel_size - 1) * dilation_ + 1)) / stride_ + 1;
    int out_w = (in_w + 2 * padding_ - ((kernel_size - 1) * dilation_ + 1)) / stride_ + 1;
    
    size_t workspace = in_channels * kernel_size * kernel_size * out_h * out_w;
    workspace *= (inputs[0].type == DataType::kFLOAT) ? sizeof(float) : sizeof(half);
    return WORKSPACE_ALGIN(workspace, CUDA_MEM_ALIGN);
}

int DeformableConv2dPlugin::enqueue(
    const PluginTensorDesc* input_descs,
    const PluginTensorDesc* output_descs,
    const void* const* inputs,
    void* const* outputs, void* workspace,
    cudaStream_t stream
) noexcept
{
    int batch = input_descs[0].dims.d[0];
    int in_channels = input_descs[0].dims.d[1];
    int out_channels = input_descs[1].dims.d[0];
    int in_h = input_descs[0].dims.d[2];
    int in_w = input_descs[0].dims.d[3];

    int kernel_size = input_descs[1].dims.d[2];
    int out_h = output_descs[0].dims.d[2];
    int out_w = output_descs[0].dims.d[3];

    if (cublasSetStream(handle_, stream) != CUBLAS_STATUS_SUCCESS)
        return CUBLAS_STATUS_NOT_INITIALIZED;

    if (input_descs[0].type == DataType::kFLOAT) {
        const float* input = static_cast<const float*>(inputs[0]);
        const float* weight = static_cast<const float*>(inputs[1]);
        const float* offset = static_cast<const float*>(inputs[2]);
        const float* mask = static_cast<const float*>(inputs[3]);
        const float* bias = static_cast<const float*>(inputs[4]);
        float* out = static_cast<float*>(outputs[0]);
        float* col = static_cast<float*>(workspace);

        float alpha = 1.f, beta = 1.f;
        for (int i = 0; i < batch; ++i) {
            float *ptr_out = out + i * out_channels * out_h * out_w;
            int err = fill_bias(bias, out_channels, out_h * out_w, ptr_out, stream);
            if (err != 0) return err;

            if (!use_mask_) {
                err = deformable_im2col(input + i * in_channels * in_h * in_w,
                                        offset + i * kernel_size * kernel_size * 2 * offset_grps_ * out_h * out_w,
                                        in_channels, in_h, in_w, offset_grps_,
                                        kernel_size, stride_, dilation_, padding_,
                                        col, stream);
            }
            if (err != 0) return err;

            int m = out_channels / weight_grps_;
            int n = out_h * out_w;
            int k = in_channels * kernel_size * kernel_size / weight_grps_;

            for (int j = 0; j < weight_grps_; ++j) {
                cublasStatus_t ret = cublasSgemm(
                    handle_, CUBLAS_OP_N, CUBLAS_OP_N, 
                                  m, n, k, &alpha,
                                  weight + j * m * k, m,
                                  col + j * k * n, k,
                                  &beta, ptr_out, m);
                if (CUBLAS_STATUS_SUCCESS != ret) {
                    return ret;
                }
            }
        }
    }
    else {
        const half* input = static_cast<const half*>(inputs[0]);
        const half* weight = static_cast<const half*>(inputs[1]);
        const half* offset = static_cast<const half*>(inputs[2]);
        const half* mask = static_cast<const half*>(inputs[3]);
        const half* bias = static_cast<const half*>(inputs[4]);
        half* out = static_cast<half*>(outputs[0]);
        half* col = static_cast<half*>(workspace);

        half alpha = 1.f, beta = 1.f;
        for (int i = 0; i < batch; ++i) {
            half *ptr_out = out + i * out_channels * out_h * out_w;
            int err = fill_bias(bias, out_channels, out_h * out_w, ptr_out, stream);
            if (err != 0) return err;

            if (!use_mask_) {
                err = deformable_im2col(input + i * in_channels * in_h * in_w,
                                        offset + i * kernel_size * kernel_size * 2 * offset_grps_ * out_h * out_w,
                                        in_channels, in_h, in_w, offset_grps_,
                                        kernel_size, stride_, dilation_, padding_,
                                        col, stream);
            }
            if (err != 0) return err;

            int m = out_channels / weight_grps_;
            int n = out_h * out_w;
            int k = in_channels * kernel_size * kernel_size / weight_grps_;

            for (int j = 0; j < weight_grps_; ++j) {
                cublasStatus_t ret = cublasHgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, 
                                  m, n, k, &alpha,
                                  weight + j * m * k, m,
                                  col + j * k * n, k,
                                  &beta, ptr_out, m);
                if (CUBLAS_STATUS_SUCCESS != ret) {
                    return ret;
                }
            }
        }
    }

    

    

    return 0;
}

DataType DeformableConv2dPlugin::getOutputDataType(
    int idx, const DataType* input_types,
    int num_inputs
) const noexcept
{
    return input_types[0];
}

const char* DeformableConv2dPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* DeformableConv2dPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

std::vector<PluginField> DeformableConv2dPluginCreator::attrs_{
    {"dil_h", nullptr, PluginFieldType::kINT32, 1},
    {"dil_w", nullptr, PluginFieldType::kINT32, 1},
    {"stride_h", nullptr, PluginFieldType::kINT32, 1},
    {"stride_w", nullptr, PluginFieldType::kINT32, 1},
    {"pad_h", nullptr, PluginFieldType::kINT32, 1},
    {"pad_w", nullptr, PluginFieldType::kINT32, 1},
    {"n_offset_grps", nullptr, PluginFieldType::kINT32, 1},
    {"n_weight_grps", nullptr, PluginFieldType::kINT32, 1},
    {"use_mask", nullptr, PluginFieldType::kINT32, 1}
};

PluginFieldCollection DeformableConv2dPluginCreator::fc_{
    static_cast<int>(attrs_.size()), attrs_.data()
};

DeformableConv2dPluginCreator::DeformableConv2dPluginCreator()
{

}

const char* DeformableConv2dPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* DeformableConv2dPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* DeformableConv2dPluginCreator::getFieldNames() noexcept
{
    return &fc_;
}

IPluginV2* DeformableConv2dPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc
) noexcept
{
    const PluginField* fields = fc->fields;
    struct {
        int dil_h, dil_w, stride_h, stride_w;
        int pad_h, pad_w, n_offset_grps;
        int n_weight_grps, use_mask;
    } cfgs;

    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "dil_h")) {
            cfgs.dil_h = *static_cast<const int*>(fields[i].data);
        }
        else if (!strcmp(attr_name, "dil_w")) {
            cfgs.dil_w = *static_cast<const int*>(fields[i].data);
        }
        else if (!strcmp(attr_name, "stride_h")) {
            cfgs.stride_h = *static_cast<const int*>(fields[i].data);
        }
        else if (!strcmp(attr_name, "stride_w")) {
            cfgs.stride_w = *static_cast<const int*>(fields[i].data);
        }
        else if (!strcmp(attr_name, "pad_h")) {
            cfgs.pad_h = *static_cast<const int*>(fields[i].data);
        }
        else if (!strcmp(attr_name, "pad_w")) {
            cfgs.pad_w = *static_cast<const int*>(fields[i].data);
        }
        else if (!strcmp(attr_name, "n_offset_grps")) {
            cfgs.n_offset_grps = *static_cast<const int*>(fields[i].data);
        }
        else if (!strcmp(attr_name, "n_weight_grps")) {
            cfgs.n_weight_grps = *static_cast<const int*>(fields[i].data);
        }
        else if (!strcmp(attr_name, "use_mask")) {
            cfgs.use_mask = *static_cast<const int*>(fields[i].data);
        }
    }

    assert(cfgs.dil_h == cfgs.dil_w);
    assert(cfgs.stride_h == cfgs.stride_w);
    assert(cfgs.pad_h == cfgs.pad_w);

    return new DeformableConv2dPlugin(cfgs.stride_h, cfgs.dil_h, cfgs.pad_h,
                                      cfgs.n_weight_grps, cfgs.n_offset_grps,
                                      cfgs.use_mask == 1);
}

IPluginV2* DeformableConv2dPluginCreator::deserializePlugin(
    const char* name, const void* buffer, size_t length
) noexcept
{
    return new DeformableConv2dPlugin(buffer, length);
}

} // namespace centernet
} // namespace dt_plugin

DEFINE_DT_PLUGIN(DeformableConv2dPlugin,
                 dt_plugin::centernet::DeformableConv2dPluginCreator)
