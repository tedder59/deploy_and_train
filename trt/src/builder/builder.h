/***
 * Author: ZhuXiaolong
 * Date: 2022-11-17 11:11:09
 * LastEditTime: 2022-11-18 22:18:08
 * FilePath: /deploy_and_train/trt/src/builder/builder.h
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
#ifndef TRT_SRC_BUILDER_BUILDER_H_
#define TRT_SRC_BUILDER_BUILDER_H_

#include <NvOnnxParser.h>
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>

namespace dt_builder
{

class BuildLogger : public nvinfer1::ILogger
{
public:
    static BuildLogger& getInstance();

    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override;

private:
    BuildLogger(std::ostream& stream=std::cout);
    std::ostream& output_;
};

struct BuilderParam
{
    bool sparsity{false};
    bool int8{false};
    bool fp16{false};
    uint64_t workspace{2048};

    struct DLAConfig {
        bool enable;
        bool standalone;
        bool allow_gpu_fallback;
        uint8_t core;

        DLAConfig();
        void setStandalone(bool yes);
        void setAllowGPUFallback(bool yes);
        void setDLACore(uint8_t core_id);

    } dla_cfg;

    struct CalibConfig {
        bool enable;
        std::string calib_cache_;

        CalibConfig();
    } calib_cfg;

    std::string onnx_filename;

    void setCalibCache(const std::string& cache);
};

class Builder
{
public:
    Builder(nvinfer1::ILogger& logger);

    bool SaveEngine(const BuilderParam& params,
                    const std::string& filename);

    bool buildEngine(const BuilderParam& params,
                     nvinfer1::ICudaEngine*& engine);

private:
    nvinfer1::ICudaEngine* build(const BuilderParam& params);

private:
    nvinfer1::ILogger& logger_;
    static std::once_flag once_;
};

} // namespace dt_builder

#endif  // TRT_SRC_BUILDER_BUILDER_