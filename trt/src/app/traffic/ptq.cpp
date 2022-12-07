/***
 * Author: ZhuXiaolong
 * Date: 2022-11-22 15:04:17
 * LastEditTime: 2022-12-07 11:34:48
 * FilePath: /deploy_and_train/trt/src/app/traffic/ptq.cpp
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
#include "infer/trt_engine.h"
#include "builder/calibrator.h"
#include "builder/builder.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <getopt.h>
#include <cstdlib>
#include <cassert>

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

#define CUDA_CHECK(status)          \
    do                              \
    {                               \
        auto ret = (status);        \
        assert(cudaSuccess == ret); \
    } while (0)

using std::cout;

void printUsage()
{
    cout << "TensorRT model PTQ" << "\n";
    cout << "Usage: ptq" << "\n"
         << "   [--onnx onnx_model.pb] (input onnx pb file)" << "\n"
         << "   [-o, --output engine_file] (output serialized engine file)" << "\n"
         << "   [--dlaStandalone] (use DLA Standalone model)" << "\n"
         << "   [--dla 0/1] (chose DLA Core id)" << "\n"
         << "   [--allowGPUFallback] (allow gpu fallback)" << "\n"
         << "   [--sparsity] (use sparse tensor core)" << "\n"
         << "   [--fp16] (use fp16 tensor core)" << "\n"
         << "   [--int8] (user int8 tensor core)" << "\n"
         << "   [--workspace] (workspace size: MB)" << "\n"
         << "   [--calibInputs] (int8 ptq calib image directory)" << "\n"
         << "   [--calibCache] (int8 ptq calib cache)" << "\n"
         << std::endl;
}

struct BuilderParam
{
    bool sparsity{false};
    bool int8{false};
    bool fp16{false};
    uint64_t workspace{2048};

    struct DLAConfig {
        bool enable{false};
        bool standalone{false};
        bool allow_gpu_fallback{false};
        bool core{false};

        void setStandalone(bool yes) {
            standalone = yes;
            enable = true;
        }

        void setAllowGPUFallback(bool yes) {
            allow_gpu_fallback = yes;
            enable = true;
        }

        void setDLACore(int core_id) {
            core = core_id > 0;
            enable = true;
        }

    } dla_cfg;

    std::string calib_dir;
    std::string calib_cache;
    std::string onnx_filename;
    std::string engine_filename;
};

void parseOptions(int argc, char**argv, BuilderParam& options)
{
    struct option long_options[] = {
        {"dlaStandalone", no_argument, 0, 1},
        {"allowGPUFallback", no_argument, 0, 2},
        {"dla", required_argument, 0, 3},
        {"output", required_argument, 0, 'o'},
        {"onnx",  required_argument, 0, 5},
        {"sparsity", no_argument, 0, 6},
        {"fp16", no_argument, 0, 7},
        {"int8", no_argument, 0, 8},
        {"workspace", required_argument, 0, 9},
        {"calibInputs", required_argument, 0, 10},
        {"calibCache", required_argument, 0, 11},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int arg = 0, option_index = 0;
    while ((arg = ::getopt_long(argc, argv, "o:h", long_options, &option_index)) != -1) {
        switch (arg)
        {
        case 1:
            options.dla_cfg.setStandalone(true);
            break;
        case 2:
            options.dla_cfg.setAllowGPUFallback(true);
            break;
        case 3:
            options.dla_cfg.setDLACore((std::stoi(optarg) == 0) ? 0 : 1);
            break;
        case 'o':
            options.engine_filename = optarg;
            break;
        case 5:
            options.onnx_filename = optarg;
            break;
        case 6:
            options.sparsity = true;
            break;
        case 7:
            options.fp16 = true;
            break;
        case 8:
            options.int8 = true;
            break;
        case 9:
            options.workspace = std::stoi(optarg);
            break;
        case 10:
            options.calib_dir = optarg;
            break;
        case 11:
            options.calib_cache = optarg;
            break;
        case 'h':
            printUsage();
            break;
        default:
            printUsage();
            abort();
        }
    }
}

class ImageDataLoader : public dt_builder::DataLoader
{
public:
    ImageDataLoader(const std::string& d, int r, int c)
        : input_(nvinfer1::DataType::kFLOAT, std::vector<int>{1, 3, r, c})
        , rows_(r), cols_(c)
    {
        std::filesystem::path directory = d.c_str();
        if (directory.empty())
            iter_ = std::filesystem::directory_iterator();
        else
            iter_ = std::filesystem::directory_iterator(directory);
    }

    bool next(int n, const char* names[],
              void* bindings[]) noexcept override {
        if (iter_ == std::filesystem::directory_iterator()) {
            return false;
        }

        auto p = iter_->path();
        iter_++;
        
        cv::Mat im = cv::imread(p.c_str(), cv::IMREAD_COLOR);
        cv::Mat resized;
        cv::resize(im, resized, cv::Size(cols_, rows_), cv::INTER_AREA);

        std::vector<cv::Mat> splits;
        cv::split(resized, splits);

        uint8_t* ps_b = splits[0].ptr();
        uint8_t* ps_g = splits[1].ptr();
        uint8_t* ps_r = splits[2].ptr();

        float* po_r = static_cast<float*>(input_.data());
        float* po_g = po_r + rows_ * cols_;
        float* po_b = po_g + rows_ * cols_;

        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                int offset = i * cols_ + j;

                po_r[offset] = static_cast<float>(ps_r[offset]) / 255.f;
                po_g[offset] = static_cast<float>(ps_g[offset]) / 255.f;
                po_b[offset] = static_cast<float>(ps_b[offset]) / 255.f;
            }
        }
        
        input_.copyIn();
        names[0] = input_name_.c_str();
        bindings[0] = input_.dev_ptr();
        return true;
    }

private:
    std::filesystem::directory_iterator iter_;
    dt_infer::SyncBlob input_;
    std::string input_name_;
    int rows_, cols_;
};

using Calibrator =\
    dt_builder::Int8Calibrator<ImageDataLoader, nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION_2>;
using CalibratorPtr = std::unique_ptr<Calibrator>;

bool saveEngine(const BuilderParam& params)
{
    dt_builder::BuildLogger& logger =\
        dt_builder::BuildLogger::getInstance();
    
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger)
    );
    if (!builder) {
        logger.log(nvinfer1::ILogger::Severity::kERROR,
                    "create builder failed.");
        return false;
    }

    const auto network_flags
        = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(network_flags)
    );
    if (!network) {
        logger.log(nvinfer1::ILogger::Severity::kERROR,
                    "create network failed.");
        return false;
    }

    auto parser = TrtUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger)
    );
    if (!parser) {
        logger.log(nvinfer1::ILogger::Severity::kERROR,
                    "create onnx parser failed.");
        return false;
    }

    if (!parser->parseFromFile(params.onnx_filename.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::stringstream ss;
        ss << "parse onnx model [" << params.onnx_filename << "] failed.";
        logger.log(nvinfer1::ILogger::Severity::kERROR, ss.str().c_str());
        return false;
    }

    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig()
    );
    if (!config) {
        logger.log(nvinfer1::ILogger::Severity::kERROR,
                    "create builder config failed.");
        return false;
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
    }

    ImageDataLoader dataloader(params.calib_dir, 544, 960);
    CalibratorPtr calib(new Calibrator(params.calib_cache, dataloader));

    if (!params.calib_dir.empty())
        config->setInt8Calibrator(calib.get());

    TrtUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        logger.log(nvinfer1::ILogger::Severity::kERROR,
                   "build serialized network failed.");
        return false;
    }

    std::ofstream ofs(params.engine_filename.c_str());
    if (!ofs) {
        std::stringstream ss;
        ss << "open output file [" << params.engine_filename << "] failed.";
        logger.log(nvinfer1::ILogger::Severity::kERROR, ss.str().c_str());
        return false;
    }
    
    ofs.write((char*)plan->data(), plan->size());
    ofs.close();
    return true;
}
                      
int main(int argc, char* argv[])
{
    BuilderParam options;
    parseOptions(argc, argv, options);
    return !saveEngine(options);
}
