/***
 * Author: ZhuXiaolong
 * Date: 2022-11-20 08:01:26
 * LastEditTime: 2022-11-20 08:25:28
 * FilePath: /deploy_and_train/trt/src/app/benchmark.cpp
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
#include <benchmark/benchmark.h>
#include <cassert>
#include <chrono>
#include <memory>
#include "infer/trt_engine.h"

using dt_infer::InferLogger;
using dt_infer::SyncManager;
using dt_infer::AsyncManager;
using dt_infer::UniformManager;
using dt_infer::ZeroCopyManager;
using dt_infer::TrtEngine;

const char* engine_filename = "cd.trt";
const char* image_filename = "";
std::unique_ptr<TrtEngine<SyncManager>> trt;

static void DoSetup(const benchmark::State& state)
{
    InferLogger& logger = InferLogger::getLogger();
    auto stream = dt_infer::makeCudaStream(logger);
    auto* manager = new SyncManager();
    trt.reset(new TrtEngine<SyncManager>(logger, stream, manager));
    assert(trt->loadEngine(engine_filename));

    const int warmups = 10;
    for (int i = 0; i < warmups; ++i) {
        trt->infer();
    }
}

static void DoTeardown(const benchmark::State& state)
{
    trt.reset();
}

static void infer(benchmark::State& s)
{
    for (auto _ : s) {
        auto start = std::chrono::high_resolution_clock::now();

        trt->infer();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = \
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end - start
            );
        s.SetIterationTime(elapsed_seconds.count());
    }
}

BENCHMARK(infer)->Threads(1)->Iterations(100)\
                ->UseManualTime()\
                ->Unit(benchmark::kMillisecond)\
                ->Setup(DoSetup)->Teardown(DoTeardown);
BENCHMARK_MAIN();
