/***
 * Author: ZhuXiaolong
 * Date: 2022-11-18 16:37:53
 * LastEditTime: 2022-11-20 07:42:32
 * FilePath: /deploy_and_train/trt/src/infer/trt_engine.cpp
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
#include <functional>
#include <iomanip>
#include <numeric>
#include <cstring>

namespace dt_infer
{

LogStreamComsumerBuffer::LogStreamComsumerBuffer(
    std::ostream& stream,
    const std::string& prefix,
    bool should_log
)
    : ostream_(stream)
    , prefix_(prefix)
    , should_log_(should_log)
{

}

LogStreamComsumerBuffer::~LogStreamComsumerBuffer()
{
    if (pbase() != pptr()) {
        put();
    }
}

int32_t LogStreamComsumerBuffer::sync()
{
    put();
    return 0;
}

void LogStreamComsumerBuffer::put()
{
    if (should_log_) {
        std::time_t ts = std::time(nullptr);
        tm* tm_local = std::localtime(&ts);

        ostream_ << "[";
        ostream_ << std::setw(4) << 1900 + tm_local->tm_year << "/";
        ostream_ << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
        ostream_ << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "-";
        ostream_ << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
        ostream_ << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
        ostream_ << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
        ostream_ << prefix_ << str();
    }

    str("");
    ostream_.flush();
}

bool LogStreamComsumerBuffer::getShouldLog() const
{
    return should_log_;
}

LogStreamConsumerBase::LogStreamConsumerBase(
    std::ostream& stream, const std::string& prefix,
    bool should_log
)   : buffer_(stream, prefix, should_log)
{

}

LogStreamConsumer::LogStreamConsumer(
    Severity reportable_severity, Severity severity
)   : LogStreamConsumerBase(severityOStream(severity),
                            severityPrefix(severity),
                            severity <= reportable_severity)
    , std::ostream(&buffer_)
{

}

std::mutex& LogStreamConsumer::getMutex()
{
    return mutex_;
}

bool LogStreamConsumer::getShouldLog() const
{
    return buffer_.getShouldLog();
}

std::ostream& LogStreamConsumer::severityOStream(Severity s)
{
    return s >= Severity::kINFO ? std::cout : std::cerr;
}

std::string LogStreamConsumer::severityPrefix(Severity s)
{
    switch (s)
    {
    case Severity::kINTERNAL_ERROR: return "[FATAL] ";
    case Severity::kERROR: return "[ERROR] ";
    case Severity::kWARNING: return "[WARNING] ";
    case Severity::kINFO: return "[INFO] ";
    case Severity::kVERBOSE: return "[VERBOSE] ";
    default: assert(0); return "";
    }
}

InferLogger::InferLogger(Severity severity)
    : reportable_severity_(severity)
{

}

InferLogger& InferLogger::getLogger()
{
    static InferLogger _instance;
    return _instance;
}

void InferLogger::setReportableSeverity(Severity severity)
{
    reportable_severity_ = severity;
}

Severity InferLogger::getReportableSeverity() const
{
    return reportable_severity_;
}

void InferLogger::log(Severity severity, const char* msg) noexcept
{
    LogStreamConsumer(reportable_severity_, severity) << std::string(msg)
                                                      << std::endl;
}

static auto hostAlloc = [](void** ptr, size_t size) {
    *ptr = malloc(size);
    return *ptr != nullptr;
};

static auto hostDealloc = [](void* ptr) {
    free(ptr);
};

static auto cudaAlloc = [](void** ptr, size_t size) {
    return cudaMalloc(ptr, size) == cudaSuccess;
};

static auto uniformAlloc = [](void** ptr, size_t size) {
    return cudaMallocManaged(ptr, size) == cudaSuccess;
};

static auto cudaDealloc = [](void* ptr) {
    cudaFree(ptr);
};

static auto pinnedAlloc = [](void** ptr, size_t size) {
    return cudaMallocHost(ptr, size) == cudaSuccess;
};

static auto zeroCopyAlloc = [](void** ptr, size_t size) {
    return cudaHostAlloc(ptr, size, cudaHostAllocMapped) == cudaSuccess;
};

static auto cudaHostDealloc = [](void* ptr) {
    cudaFreeHost(ptr);
};

Blob::Blob(nvinfer1::DataType dtype, const std::vector<int>& dims)
    : dtype_(dtype), dims_(dims), size_(Blob::size(dtype, dims))
{

}

size_t Blob::size(nvinfer1::DataType dtype,
                  const std::vector<int>& dims) {
    size_t ans = 1;
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
        ans = 4;
        break;
    case nvinfer1::DataType::kHALF:
        ans = 2;
        break;
    default:    // kINT8 && kBOOL
        break;
    }

    return std::accumulate(dims.begin(), dims.end(), ans,
                           std::multiplies<size_t>());
}

SyncBlob::SyncBlob(nvinfer1::DataType dtype,
                   const std::vector<int>& dims)
    : Blob(dtype, dims)
{
    size_t size = Blob::size(dtype, dims);
    void* ptr = nullptr;

    assert(hostAlloc(&ptr, size));
    host_.reset(ptr, hostDealloc);

    ptr = nullptr;
    assert(cudaAlloc(&ptr, size));
    dev_.reset(ptr, cudaDealloc);
}

void* SyncBlob::data()
{
    return host_.get();
}

void* SyncBlob::dev_ptr()
{
    return dev_.get();
}

void SyncBlob::copyIn()
{
    CUDA_CHECK(cudaMemcpy(dev_.get(), host_.get(), size_,
                          cudaMemcpyHostToDevice));
}

void SyncBlob::copyOut()
{
    CUDA_CHECK(cudaMemcpy(host_.get(), dev_.get(), size_,
                          cudaMemcpyDeviceToHost));
}

void SyncBlob::copyIn(const void* ptr, size_t len)
{
    memcpy(host_.get(), ptr, len);
}

void SyncBlob::copyOut(void* const ptr, size_t len)
{
    memcpy(ptr, host_.get(), len);
}

AsyncBlob::AsyncBlob(nvinfer1::DataType dtype,
                     const std::vector<int>& dims)
    : Blob(dtype, dims)
{
    size_t size = Blob::size(dtype, dims);
    void* ptr = nullptr;

    assert(pinnedAlloc(&ptr, size));
    host_.reset(ptr, cudaHostDealloc);

    ptr = nullptr;
    assert(cudaAlloc(&ptr, size));
    dev_.reset(ptr, cudaDealloc);
}

void* AsyncBlob::data()
{
    return host_.get();
}

void* AsyncBlob::dev_ptr()
{
    return dev_.get();
}

void AsyncBlob::copyIn(cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(dev_.get(), host_.get(), size_,
                               cudaMemcpyHostToDevice,
                               stream));
}

void AsyncBlob::copyOut(cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(host_.get(), dev_.get(), size_,
                               cudaMemcpyDeviceToHost,
                               stream));
}

void AsyncBlob::copyIn(const void* ptr, size_t len)
{
    memcpy(host_.get(), ptr, len);
}

void AsyncBlob::copyOut(void* const ptr, size_t len)
{
    memcpy(ptr, host_.get(), len);
}

UniformBlob::UniformBlob(nvinfer1::DataType dtype,
                         const std::vector<int>& dims)
    : Blob(dtype, dims)
{
    size_t size = Blob::size(dtype, dims);
    void* ptr = nullptr;
    assert(uniformAlloc(&ptr, size));
    buffer_.reset(ptr, cudaDealloc);
}

void* UniformBlob::data()
{
    return buffer_.get();
}

void UniformBlob::copyIn(const void* ptr, size_t len)
{
    memcpy(buffer_.get(), ptr, len);
}

void UniformBlob::copyOut(void* const ptr, size_t len)
{
    memcpy(ptr, buffer_.get(), len);
}

ZeroCopyBlob::ZeroCopyBlob(nvinfer1::DataType dtype,
                           const std::vector<int>& dims)
    : Blob(dtype, dims)
{
    size_t size = Blob::size(dtype, dims);
    void* ptr = nullptr;
    assert(zeroCopyAlloc(&ptr, size));
    buffer_.reset(ptr, cudaHostDealloc);
}

void* ZeroCopyBlob::data()
{
    return buffer_.get();
}

void* ZeroCopyBlob::dev_ptr()
{
    void* ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&ptr, buffer_.get(), 0));
    return ptr;
}

void ZeroCopyBlob::copyIn(const void* ptr, size_t len)
{
    memcpy(buffer_.get(), ptr, len);
}

void ZeroCopyBlob::copyOut(void* const ptr, size_t len)
{
    memcpy(ptr, buffer_.get(), len);
}

SyncManager::SyncManager()
{

}

SyncManager::~SyncManager()
{
    for (auto* ptr : all_blobs_) {
        delete ptr;
    }
}

bool SyncManager::init(nvinfer1::ICudaEngine* engine,
                       nvinfer1::IExecutionContext* context)
{
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        std::string name = engine->getBindingName(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);

        auto dims = engine->getBindingDimensions(i);
        std::vector<int> shape(dims.d, dims.d + dims.nbDims);
        
        SyncBlob* blob = new SyncBlob(dtype, shape);
        all_blobs_.push_back(blob);
        bindings_.push_back(blob->dev_ptr());

        blobs_dict_[name] = blob;
        if (engine->bindingIsInput(i)) {
            input_blobs_.push_back(blob);
        }
        else {
            output_blobs_.push_back(blob);
        }
    }

    return true;
}

void SyncManager::copyIn()
{
    for (auto* ptr : input_blobs_) {
        ptr->copyIn();
    }
}

void SyncManager::copyOut()
{
    for (auto* ptr : output_blobs_) {
        ptr->copyOut();
    }
}

bool SyncManager::getBindings(void* const*& bindings)
{
    bindings = bindings_.data();
    return !bindings_.empty();
}

Blob* SyncManager::getBlob(const std::string& name)
{
    if (blobs_dict_.find(name) != blobs_dict_.end()) {
        return blobs_dict_[name];
    }
    else {
        return nullptr;
    }
}

AsyncManager::AsyncManager(cudaStream_t stream)
    : stream_(stream)
{

}

AsyncManager::~AsyncManager()
{
    for (auto* ptr : all_blobs_) {
        delete ptr;
    }
}

bool AsyncManager::init(nvinfer1::ICudaEngine* engine,
                        nvinfer1::IExecutionContext* context)
{
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        std::string name = engine->getBindingName(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);

        auto dims = engine->getBindingDimensions(i);
        std::vector<int> shape(dims.d, dims.d + dims.nbDims);
        
        AsyncBlob* blob = new AsyncBlob(dtype, shape);
        all_blobs_.push_back(blob);
        bindings_.push_back(blob->dev_ptr());

        blobs_dict_[name] = blob;
        if (engine->bindingIsInput(i)) {
            input_blobs_.push_back(blob);
        }
        else {
            output_blobs_.push_back(blob);
        }
    }

    return true;
}

void AsyncManager::copyIn()
{
    for (auto* ptr : input_blobs_) {
        ptr->copyIn(stream_);
    }
}

void AsyncManager::copyOut()
{
    for (auto* ptr : output_blobs_) {
        ptr->copyOut(stream_);
    }
}

bool AsyncManager::getBindings(void* const*& bindings)
{
    bindings = bindings_.data();
    return !bindings_.empty();
}

Blob* AsyncManager::getBlob(const std::string& name)
{
    if (blobs_dict_.find(name) != blobs_dict_.end()) {
        return blobs_dict_[name];
    }
    else {
        return nullptr;
    }
}

UniformManager::UniformManager()
{

}

UniformManager::~UniformManager()
{
    for (auto* ptr : all_blobs_) {
        delete ptr;
    }
}

bool UniformManager::init(nvinfer1::ICudaEngine* engine,
                          nvinfer1::IExecutionContext* context)
{
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        std::string name = engine->getBindingName(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);

        auto dims = engine->getBindingDimensions(i);
        std::vector<int> shape(dims.d, dims.d + dims.nbDims);
        
        UniformBlob* blob = new UniformBlob(dtype, shape);
        all_blobs_.push_back(blob);
        bindings_.push_back(blob->data());
        blobs_dict_[name] = blob;
    }

    return true;
}

void UniformManager::copyIn()
{

}

void UniformManager::copyOut()
{

}

bool UniformManager::getBindings(void* const*& bindings)
{
    bindings = bindings_.data();
    return !bindings_.empty();
}

Blob* UniformManager::getBlob(const std::string& name)
{
    if (blobs_dict_.find(name) != blobs_dict_.end()) {
        return blobs_dict_[name];
    }
    else {
        return nullptr;
    }
}

std::once_flag ZeroCopyManager::once_;

ZeroCopyManager::ZeroCopyManager()
{
    std::call_once(once_, []{
        CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    });
}

ZeroCopyManager::~ZeroCopyManager()
{
    for (auto* ptr : all_blobs_) {
        delete ptr;
    }
}

bool ZeroCopyManager::init(nvinfer1::ICudaEngine* engine,
                           nvinfer1::IExecutionContext* context)
{
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        std::string name = engine->getBindingName(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);

        auto dims = engine->getBindingDimensions(i);
        std::vector<int> shape(dims.d, dims.d + dims.nbDims);
        
        ZeroCopyBlob* blob = new ZeroCopyBlob(dtype, shape);
        all_blobs_.push_back(blob);
        bindings_.push_back(blob->dev_ptr());
        blobs_dict_[name] = blob;
    }

    return true;
}

void ZeroCopyManager::copyIn()
{

}

void ZeroCopyManager::copyOut()
{

}

bool ZeroCopyManager::getBindings(void* const*& bindings)
{
    bindings = bindings_.data();
    return !bindings_.empty();
}

Blob* ZeroCopyManager::getBlob(const std::string& name)
{
    if (blobs_dict_.find(name) != blobs_dict_.end()) {
        return blobs_dict_[name];
    }
    else {
        return nullptr;
    }
}

} // namespace dt_infer