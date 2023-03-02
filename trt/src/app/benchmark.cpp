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

const char* engine_filename = "cc.fp16.trt";
std::unique_ptr<TrtEngine<SyncManager>> sync_trt;
std::unique_ptr<TrtEngine<AsyncManager>> async_trt;
std::unique_ptr<TrtEngine<UniformManager>> uniform_trt;

#ifdef JETSON
std::unique_ptr<TrtEngine<ZeroCopyManager>> zerocopy_trt;
#endif	// JETSON

__attribute((constructor)) void setDevice()
{
#ifdef JETSON
    ZeroCopyManager m;
#endif	// JETSON
}

static void syncSetup(const benchmark::State& state)
{
    InferLogger& logger = InferLogger::getLogger();
    auto stream = dt_infer::makeCudaStream(logger);

    auto* manager = new SyncManager();

    sync_trt.reset(new TrtEngine<SyncManager>(logger, stream, manager));
    
    assert(sync_trt->loadEngine(engine_filename));

    const int warmups = 10;
    for (int i = 0; i < warmups; ++i) {
        sync_trt->infer();
    }
}

static void asyncSetup(const benchmark::State& state)
{
    InferLogger& logger = InferLogger::getLogger();
    auto stream = dt_infer::makeCudaStream(logger);

    auto* manager = new AsyncManager(stream);

    async_trt.reset(new TrtEngine<AsyncManager>(logger, stream, manager));
    
    assert(async_trt->loadEngine(engine_filename));

    const int warmups = 10;
    for (int i = 0; i < warmups; ++i) {
        async_trt->infer();
    }
}

static void uniformSetup(const benchmark::State& state)
{
    InferLogger& logger = InferLogger::getLogger();
    auto stream = dt_infer::makeCudaStream(logger);

    auto* manager = new UniformManager();

    uniform_trt.reset(new TrtEngine<UniformManager>(logger, stream, manager));
    
    assert(uniform_trt->loadEngine(engine_filename));

    const int warmups = 10;
    for (int i = 0; i < warmups; ++i) {
        uniform_trt->infer();
    }
}

#ifdef JETSON

static void zerocopySetup(const benchmark::State& state)
{
    InferLogger& logger = InferLogger::getLogger();
    auto stream = dt_infer::makeCudaStream(logger);

    auto* manager = new ZeroCopyManager();

    zerocopy_trt.reset(new TrtEngine<ZeroCopyManager>(logger, stream, manager));
    
    assert(zerocopy_trt->loadEngine(engine_filename));

    const int warmups = 10;
    for (int i = 0; i < warmups; ++i) {
        zerocopy_trt->infer();
    }
}

#endif	// JETSON

static void syncTeardown(const benchmark::State& state)
{
    sync_trt.reset();
}

static void asyncTeardown(const benchmark::State& state)
{
    async_trt.reset();
}

static void uniformTeardown(const benchmark::State& state)
{
    uniform_trt.reset();
}

#ifdef JETSON
static void zerocopyTeardown(const benchmark::State& state)
{
    zerocopy_trt.reset();
}
#endif	// JETSON

static void syncInfer(benchmark::State& s)
{
    if (s.thread_index == 0) {
        syncSetup(s);
    }

    for (auto _ : s) {
        auto start = std::chrono::high_resolution_clock::now();

        sync_trt->infer();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = \
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end - start
            );
        s.SetIterationTime(elapsed_seconds.count());
    }

    if (s.thread_index == 0) {
        syncTeardown(s);
    }
}

static void asyncInfer(benchmark::State& s)
{
    if (s.thread_index == 0) {
        asyncSetup(s);
    }

    for (auto _ : s) {
        auto start = std::chrono::high_resolution_clock::now();

        async_trt->infer();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = \
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end - start
            );
        s.SetIterationTime(elapsed_seconds.count());
    }

    if (s.thread_index == 0) {
        asyncTeardown(s);
    }
}

static void uniformInfer(benchmark::State& s)
{
    if (s.thread_index == 0) {
        uniformSetup(s);
    }

    for (auto _ : s) {
        auto start = std::chrono::high_resolution_clock::now();

        uniform_trt->infer();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = \
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end - start
            );
        s.SetIterationTime(elapsed_seconds.count());
    }

    if (s.thread_index == 0) {
        uniformTeardown(s);
    }
}

#ifdef JETSON

static void zerocopyInfer(benchmark::State& s)
{
    if (s.thread_index == 0) {
        zerocopySetup(s);
    }

    for (auto _ : s) {
        auto start = std::chrono::high_resolution_clock::now();

        zerocopy_trt->infer();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = \
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end - start
            );
        s.SetIterationTime(elapsed_seconds.count());
    }

    if (s.thread_index == 0) {
        zerocopyTeardown(s);
    }
}

#endif	// JETSON

BENCHMARK(syncInfer)->Threads(1)->Iterations(100)\
                    ->UseManualTime()\
                    ->Unit(benchmark::kMillisecond);

BENCHMARK(asyncInfer)->Threads(1)->Iterations(100)\
                     ->UseManualTime()\
                     ->Unit(benchmark::kMillisecond);

BENCHMARK(uniformInfer)->Threads(1)->Iterations(100)\
                       ->UseManualTime()\
                       ->Unit(benchmark::kMillisecond);

#ifdef JETSON
BENCHMARK(zerocopyInfer)->Threads(1)->Iterations(100)\
                        ->UseManualTime()\
                        ->Unit(benchmark::kMillisecond);
#endif	// JETSON

BENCHMARK_MAIN();
