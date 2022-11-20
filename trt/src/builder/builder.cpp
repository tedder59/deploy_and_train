/***
 * Author: ZhuXiaolong
 * Date: 2022-11-17 11:11:12
 * LastEditTime: 2022-11-18 22:19:29
 * FilePath: /deploy_and_train/trt/src/builder/builder.cpp
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
#include "builder/builder.h"
#include "builder/calibrator.h"
#include "plugins/plugin.h"
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <memory>
#include <ctime>

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

#define CUDA_CHECK(status)          \
    do                              \
    {                               \
        auto ret = (status);        \
        assert(cudaSuccess == ret); \
    } while (0)
    
static auto StreamDeleter = [](cudaStream_t* s) {
    if (s) {
        CUDA_CHECK(cudaStreamDestroy(*s));
        delete s;
    }
};

using StreamUniquePtr =\
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)>;

inline StreamUniquePtr makeCudaStream()
{
    StreamUniquePtr stream(new cudaStream_t, StreamDeleter);
    if (cudaSuccess != cudaStreamCreate(stream.get())) {
        stream.reset();
    }
    
    return stream;
}

using Calibrator =\
    dt_builder::Int8Calibrator<dt_builder::DataLoader, nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION_2>;
using CalibratorPtr = std::unique_ptr<Calibrator>;

namespace dt_builder
{

BuildLogger::BuildLogger(std::ostream& stream)
    : output_(stream)
{

}

BuildLogger& BuildLogger::getInstance()
{
    static BuildLogger _logger;
    return _logger;
}

void BuildLogger::log(nvinfer1::ILogger::Severity severity,
                      const char* msg) noexcept
{
    std::time_t ts = std::time(nullptr);
    tm* t_local = std::localtime(&ts);

    output_ << "[";
    output_ << std::setw(4) << 1900 + t_local->tm_year << "/";
    output_ << std::setw(2) << std::setfill('0') << 1 + t_local->tm_mon << "/";
    output_ << std::setw(2) << std::setfill('0') << t_local->tm_mday << "-";
    output_ << std::setw(2) << std::setfill('0') << t_local->tm_hour << ":";
    output_ << std::setw(2) << std::setfill('0') << t_local->tm_min << ":";
    output_ << std::setw(2) << std::setfill('0') << t_local->tm_sec << "] ";

    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        output_ << "[FATAL] ";
        break;
    case nvinfer1::ILogger::Severity::kERROR:
        output_ << "[ERROR] ";
        break;
    case nvinfer1::ILogger::Severity::kWARNING:
        output_ << "[WARNING] ";
        break;
    case nvinfer1::ILogger::Severity::kINFO:
        output_ << "[INFO] ";
        break;
    case nvinfer1::ILogger::Severity::kVERBOSE:
        output_ << "[VERBOSE] ";
        break;
    default:
        output_ << "[UNKNOWN] ";
        break;
    }

    output_ << msg << std::endl;
    output_.flush();
}

BuilderParam::DLAConfig::DLAConfig()
    : enable(false)
    , standalone(false)
    , allow_gpu_fallback(false)
    , core(0)
{

}

void BuilderParam::DLAConfig::setStandalone(bool yes)
{
    enable = true;
    standalone = yes;
}

void BuilderParam::DLAConfig::setAllowGPUFallback(bool yes)
{
    enable = true;
    allow_gpu_fallback = yes;
}

void BuilderParam::DLAConfig::setDLACore(uint8_t core_id)
{
    enable = true;
    core = core_id;
}

BuilderParam::CalibConfig::CalibConfig()
    : enable(false)
{

}

void BuilderParam::setCalibCache(
    const std::string& cache
)
{
    calib_cfg.calib_cache_ = cache;
    calib_cfg.enable = true;
    int8 = true;
}

std::once_flag Builder::once_;

Builder::Builder(nvinfer1::ILogger& logger)
    : logger_(logger)
{
    std::call_once(once_, [&logger](){
        if (!initLibNvInferPlugins(&logger, "")) {
            logger.log(nvinfer1::ILogger::Severity::kWARNING,
                    "nvinfer_plugin register failed.");
        }

        if (!dt_plugin_status) {
            logger.log(nvinfer1::ILogger::Severity::kWARNING,
                    "dt_plugin register failed.");
        }
    });
}

bool Builder::SaveEngine(const BuilderParam& params,
                         const std::string& filename)
{
    TrtUniquePtr<nvinfer1::ICudaEngine> engine(build(params));
    if (!engine) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR,
                    "build engine failed.");
        return false;
    }
    
    TrtUniquePtr<nvinfer1::IHostMemory> plan(engine->serialize());

    if (!plan) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR,
                    "build serialized engine failed.");
        return false;
    }

    std::ofstream ofs(filename.c_str(), std::ofstream::binary);
    if (!ofs.is_open()) {
        std::stringstream ss;
        ss << "open output file [" << filename <<  "] failed.";
        logger_.log(nvinfer1::ILogger::Severity::kERROR, ss.str().c_str());
        return false;
    }

    ofs.write(static_cast<char*>(plan->data()), plan->size());
    ofs.close();
    return true;
}

bool Builder::buildEngine(const BuilderParam& params,
                          nvinfer1::ICudaEngine*& engine)
{
    engine = build(params);
    return engine != nullptr;
}

nvinfer1::ICudaEngine* Builder::build(const BuilderParam& params)
{
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_)
    );
    if (!builder) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR,
                    "create builder failed.");
        return nullptr;
    }

    const auto network_flags
        = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(network_flags)
    );
    if (!network) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR,
                    "create network failed.");
        return nullptr;
    }

    auto parser = TrtUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_)
    );
    if (!parser) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR,
                    "create onnx parser failed.");
        return nullptr;
    }

    if (!parser->parseFromFile(params.onnx_filename.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::stringstream ss;
        ss << "parse onnx model [" << params.onnx_filename << "] failed.";
        logger_.log(nvinfer1::ILogger::Severity::kERROR, ss.str().c_str());
        return nullptr;
    }

    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig()
    );
    if (!config) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR,
                    "create builder config failed.");
        return nullptr;
    }

    if (params.int8)
        config->setFlag(nvinfer1::BuilderFlag::kINT8);

    if (params.fp16)
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // DLA config
    if (params.dla_cfg.enable && builder->getNbDLACores() > 0) {
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(params.dla_cfg.core);

        if (params.dla_cfg.standalone) {
            config->setEngineCapability(
                nvinfer1::EngineCapability::kDLA_STANDALONE
            );
        }
        else if (params.dla_cfg.allow_gpu_fallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }

        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    }

    // Int8Calib
    if (params.calib_cfg.enable) {
        DataLoader dataloader;
        CalibratorPtr calibrator (new Calibrator(
            params.calib_cfg.calib_cache_, dataloader
        ));
        config->setInt8Calibrator(calibrator.get());
    }
    
    if (params.sparsity)
        config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);

#ifdef TENSORRT_MEMORY_POOL
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               params.workspace << 20);
#else
    config->setMaxWorkspaceSize(params.workspace << 20);
#endif

    auto stream = makeCudaStream();
    if (!stream) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR,
                    "create profile stream failed.");
        return nullptr;
    }
    config->setProfileStream(*stream);

    return builder->buildEngineWithConfig(*network, *config);
}
    
} // namespace dt_builder
