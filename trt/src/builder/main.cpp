/***
 * Author: ZhuXiaolong
 * Date: 2022-11-17 17:41:34
 * LastEditTime: 2022-11-24 14:46:20
 * FilePath: /deploy_and_train/trt/src/builder/main.cpp
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
#include <cuda_runtime_api.h>
#include <getopt.h>
#include <cstdlib>
#include <cassert>

#define CUDA_CHECK(status)          \
    do                              \
    {                               \
        auto ret = (status);        \
        assert(cudaSuccess == ret); \
    } while (0)

using std::cout;

void print_usage()
{
    cout << "ONNX to TensorRT model parser" << "\n";
    cout << "Usage: build.out" << "\n"
         << "   [-onnx onnx_model.pb] (input onnx pb file)" << "\n"
         << "   [-o, --output engine_file] (output serialized engine file)" << "\n"
         << "   [--dlaStandalone] (use DLA Standalone model)" << "\n"
         << "   [--dla 0/1] (chose DLA Core id)" << "\n"
         << "   [--allowGPUFallback] (allow gpu fallback)" << "\n"
         << "   [--sparsity] (use sparse tensor core)" << "\n"
         << "   [--fp16] (use fp16 tensor core)" << "\n"
         << "   [--int8] (user int8 tensor core)" << "\n"
         << "   [--workspace] (workspace size: MB)" << "\n"
         << "   [--calibCache] (int8 ptq calib cache file)" << "\n"
         << std::endl;
}

void parseOptions(int argc, char**argv, dt_builder::BuilderParam& options,
                   std::string& output_file) {
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
        {"calibCache", required_argument, 0, 10},
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
            output_file = optarg;
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
            options.setCalibCache(optarg);
            break;
        case 'h':
            print_usage();
            break;
        default:
            print_usage();
            abort();
        }
    }
}

inline void setCudaDevice(int device, std::ostream& os)
{
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    // clang-format off
    os << "=== Device Information ===" << "\n"
       << "Selected Device: "      << properties.name << "\n"
       << "Compute Capability: "   << properties.major << "." << properties.minor << "\n"
       << "SMs: "                  << properties.multiProcessorCount << "\n"
       << "Compute Clock Rate: "   << properties.clockRate / 1000000.0F << " GHz\n"
       << "Device Global Memory: " << (properties.totalGlobalMem >> 20) << " MiB\n"
       << "Shared Memory per SM: " << (properties.sharedMemPerMultiprocessor >> 10) << " KiB\n"
       << "Memory Bus Width: "     << properties.memoryBusWidth << " bits (ECC "
                                   << (properties.ECCEnabled != 0 ? "enabled" : "disabled") << ")\n"
       << "Memory Clock Rate: "    << properties.memoryClockRate / 1000000.0F << " GHz\n"
       << std::endl;
    // clang-format on
}

int main(int argc, char* argv[])
{
    dt_builder::BuildLogger& logger = dt_builder::BuildLogger::getInstance();
    setCudaDevice(0, std::cout);

    dt_builder::BuilderParam options;
    std::string output;

    parseOptions(argc, argv, options, output);
    if (options.onnx_filename.empty() || output.empty()) {
        return 0;
    }

    dt_builder::Builder builder(logger);
    bool ret = builder.SaveEngine(options, output);
    return (ret) ? 0 : 1;
}
