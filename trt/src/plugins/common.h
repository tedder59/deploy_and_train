/***
 * Author: ZhuXiaolong
 * Date: 2022-11-14 13:10:23
 * LastEditTime: 2022-11-16 10:21:14
 * FilePath: /deploy_and_train/trt/src/plugins/common.h
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
#ifndef TRT_SRC_PLUGINS_COMMON_H_
#define TRT_SRC_PLUGINS_COMMON_H_

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <NvInferPlugin.h>
#include <memory>
#include <vector>
#include <cstring>
#include <cassert>
#include <iostream>

using namespace nvinfer1;

#define PLUGIN_NAMESPACE ""

#define DEFINE_DT_PLUGIN(op, creator)                               \
    bool register##op() {                                           \
        static std::unique_ptr<creator> _creator(new creator());    \
        bool ret = getPluginRegistry()->registerCreator(*_creator, PLUGIN_NAMESPACE);   \
        return ret;                                                 \
    }

#define WORKSPACE_ALGIN(x, n) ((x + n - 1) & (-n))
#define CUDA_MEM_ALIGN 256

#define KERNEL_LAUNCH_CHECK()                           \
    do {                                                \
        cudaError_t err = cudaGetLastError();           \
        if (cudaSuccess != err) {                       \
            std::cerr << __FILE__ << ": " << __LINE__   \
                      << " [" << __func__ << "]: "      \
                      << cudaGetErrorString(err);       \
        }                                               \
        assert(cudaSuccess == err);                      \
    } while(0)

namespace dt_plugin {

template <typename T>
T readBuffer(const void*& buffer)
{
    const T* ptr = static_cast<const T*>(buffer);
    buffer = static_cast<const void*>(ptr + 1);
    return *ptr;
}

template <typename T>
void writeBuffer(void*& buffer, T v)
{
    T* ptr = static_cast<T*>(buffer);
    buffer = static_cast<void*>(ptr + 1);
    *ptr = v;
}

class DynamicPlugin : public IPluginV2DynamicExt
{
public:
    virtual ~DynamicPlugin() noexcept = default;

    IPluginV2DynamicExt* clone() const noexcept override = 0;
    
    DimsExprs getOutputDimensions(
        int output_idx, const DimsExprs* inputs,
        int num_inputs, IExprBuilder& expr_builder
    ) noexcept override = 0;
    
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* in_out,
        int num_inputs, int num_outputs
    ) noexcept override = 0;
    
    int enqueue(const PluginTensorDesc* input_descs,
                const PluginTensorDesc* output_descs,
                const void* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) noexcept override = 0;

    DataType getOutputDataType(
        int idx, const DataType* input_types, int num_inputs
    ) const noexcept override=0;

    const char* getPluginType() const noexcept override = 0;
    const char* getPluginVersion() const noexcept override = 0;

    virtual void configurePlugin(
        const DynamicPluginTensorDesc* inputs, int num_inputs,
        const DynamicPluginTensorDesc* outputs, int num_outputs
    ) noexcept override {}
    virtual size_t getWorkspaceSize(
        const PluginTensorDesc* inputs, int num_inputs,
        const PluginTensorDesc* outputs, int num_outputs
    ) const noexcept override { return 0; }
    virtual size_t getSerializationSize() const noexcept override { return 0; }
    virtual void serialize(void* buffer) const noexcept override {}
    virtual int getNbOutputs() const noexcept override { return 1; }
    virtual int initialize() noexcept override { return 0; }
    virtual void terminate() noexcept override {}
    virtual void destroy() noexcept override {}

    void setPluginNamespace(const char* plugin_namespace) noexcept override {}
    const char* getPluginNamespace() const noexcept override { 
        return PLUGIN_NAMESPACE; 
    }
};

class DynamicPluginCreator : public IPluginCreator
{
public:
    virtual ~DynamicPluginCreator() = default;
    const char* getPluginName() const noexcept override = 0;
    const char* getPluginVersion() const noexcept override = 0;
    const PluginFieldCollection* getFieldNames() noexcept override = 0;
    IPluginV2* createPlugin(
        const char* name, const PluginFieldCollection* fc
    ) noexcept override = 0;
    IPluginV2* deserializePlugin(
        const char* name, const void* buffer, size_t length
    ) noexcept override = 0;

    void setPluginNamespace(const char* plugin_namespace) noexcept override {}
    const char* getPluginNamespace() const noexcept override {
        return PLUGIN_NAMESPACE;
    }
};

} // dt_plugin

#endif  // TRT_SRC_PLUGINS_COMMON_H_
