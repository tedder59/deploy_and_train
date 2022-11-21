/***
 * Author: ZhuXiaolong
 * Date: 2022-11-18 11:35:25
 * LastEditTime: 2022-11-20 08:08:38
 * FilePath: /deploy_and_train/trt/src/infer/trt_engine.h
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
#ifndef TRT_SRC_INFER_TRT_ENGINE_H_
#define TRT_SRC_INFER_TRT_ENGINE_H_

#include "builder/builder.h"
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <cassert>
#include <sstream>
#include <vector>
#include <memory>
#include <mutex>
#include <map>


template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

using Severity = nvinfer1::ILogger::Severity;

#define NOCOPY(X)                           \
    X(const X& othr) = delete;              \
    X& operator=(const X& othr) = delete

namespace dt_infer
{

class LogStreamComsumerBuffer : public std::stringbuf
{
public:
    LogStreamComsumerBuffer(std::ostream& stream,
                            const std::string& prefix,
                            bool should_log);
    ~LogStreamComsumerBuffer();

    int32_t sync() override;
    void put();
    bool getShouldLog() const;

    NOCOPY(LogStreamComsumerBuffer);

private:
    std::ostream& ostream_;
    std::string prefix_;
    bool should_log_;
};

class LogStreamConsumerBase
{
public:
    LogStreamConsumerBase(std::ostream& stream,
                          const std::string& prefix,
                          bool should_log);

protected:
    std::mutex mutex_;
    LogStreamComsumerBuffer buffer_;
};

class LogStreamConsumer : protected LogStreamConsumerBase, public std::ostream
{
public:
    LogStreamConsumer(Severity reportable_severity,
                      Severity severity);
    ~LogStreamConsumer() = default;

    std::mutex& getMutex();
    bool getShouldLog() const;

    NOCOPY(LogStreamConsumer);

private:
    static std::ostream& severityOStream(Severity s);
    static std::string severityPrefix(Severity s);
};

template <typename T>
LogStreamConsumer& operator<<(LogStreamConsumer& logger,
                              const T& obj)
{
    if (logger.getShouldLog()) {
        std::lock_guard<std::mutex> guard(logger.getMutex());
        auto& os = static_cast<std::ostream&>(logger);
        os << obj;
    }
    return logger;
}

class InferLogger : public nvinfer1::ILogger
{
public:
    static InferLogger& getLogger();

    void setReportableSeverity(Severity severity);
    Severity getReportableSeverity() const;
    void log(Severity severity, const char* msg) noexcept override;

private:
    InferLogger(Severity severity=Severity::kWARNING);
    Severity reportable_severity_;
};

class Blob {
public:
    Blob(nvinfer1::DataType dtype, const std::vector<int>& dims);
    virtual ~Blob() = default;

    // host mem ptr
    virtual void* data() = 0;

    // outer <-> host
    virtual void copyIn(const void* ptr, size_t len) = 0;
    virtual void copyOut(void* const ptr, size_t len) = 0;

    static size_t size(nvinfer1::DataType dtype,
                       const std::vector<int>& dims);

protected:
    nvinfer1::DataType dtype_;
    std::vector<int> dims_;
    size_t size_;
};

class SyncBlob : public Blob
{
public:
    SyncBlob(nvinfer1::DataType dtype,
             const std::vector<int>& dims);
    ~SyncBlob() = default;

    void* data() override;
    void* dev_ptr();

    // host <-> dev
    void copyIn();
    void copyOut();

    void copyIn(const void* ptr, size_t len) override;
    void copyOut(void* const ptr, size_t len) override;

private:
    std::shared_ptr<void> host_;
    std::shared_ptr<void> dev_;
};

class AsyncBlob : public Blob
{
public:
    AsyncBlob(nvinfer1::DataType dtype,
              const std::vector<int>& dims);
    ~AsyncBlob() = default;

    void* data() override;
    void* dev_ptr();

    // host <-> dev
    void copyIn(cudaStream_t stream);
    void copyOut(cudaStream_t stream);

    void copyIn(const void* ptr, size_t len) override;
    void copyOut(void* const ptr, size_t len) override;
    
private:
    std::shared_ptr<void> host_;
    std::shared_ptr<void> dev_;
};

class UniformBlob : public Blob
{
public:
    UniformBlob(nvinfer1::DataType dtype,
                const std::vector<int>& dims);
    ~UniformBlob() = default;

    void* data() override;
    void copyIn(const void* ptr, size_t len) override;
    void copyOut(void* const ptr, size_t len) override;
    
private:
    std::shared_ptr<void> buffer_;
};

class ZeroCopyBlob : public Blob
{
public:
    ZeroCopyBlob(nvinfer1::DataType dtype,
                 const std::vector<int>& dims);
    ~ZeroCopyBlob() = default;

    void* data() override;
    void* dev_ptr();

    void copyIn(const void* ptr, size_t len) override;
    void copyOut(void* const ptr, size_t len) override;
    
private:
    std::shared_ptr<void> buffer_;
};

#define CUDA_CHECK(status)          \
    do                              \
    {                               \
        auto ret = (status);        \
	if (cudaSuccess != ret) {   \
	    std::cerr << cudaGetErrorString(ret) << std::endl;	\
	}			    \
        assert(cudaSuccess == ret); \
    } while (0)
    
static auto streamDeleter = [](cudaStream_t* s) {
    if (s) {
        CUDA_CHECK(cudaStreamDestroy(*s));
        delete s;
    }
};

using CudaStreamPtr = std::shared_ptr<cudaStream_t>;

CudaStreamPtr makeCudaStream(nvinfer1::ILogger& logger)
{
    CudaStreamPtr stream(new cudaStream_t(), streamDeleter);
    if (cudaSuccess != cudaStreamCreate(stream.get())) {
        logger.log(Severity::kERROR, "create stream failed.");
        stream.reset();
    }

    return stream;
}

class MemManager
{
public:
    virtual ~MemManager() = default;

    virtual bool init(nvinfer1::ICudaEngine* engine,
                      nvinfer1::IExecutionContext* context) = 0;
    virtual void copyIn() {}
    virtual void copyOut() {}

    virtual bool getBindings(void* const*& bindings) = 0;
    virtual Blob* getBlob(const std::string& name) = 0;
};

class SyncManager : public MemManager
{
public:
    SyncManager();
    ~SyncManager();

    bool init(nvinfer1::ICudaEngine* engine,
              nvinfer1::IExecutionContext* context) override;
    void copyIn() override;
    void copyOut() override;

    bool getBindings(void* const*& bindings) override;
    Blob* getBlob(const std::string& name) override;

private:
    std::map<std::string, SyncBlob*> blobs_dict_;
    std::vector<SyncBlob*> input_blobs_;
    std::vector<SyncBlob*> output_blobs_;
    std::vector<void*> bindings_;

    std::vector<SyncBlob*> all_blobs_;
};

class AsyncManager : public MemManager
{
public:
    AsyncManager(CudaStreamPtr stream);
    ~AsyncManager();

    bool init(nvinfer1::ICudaEngine* engine,
              nvinfer1::IExecutionContext* context) override;
    void copyIn() override;
    void copyOut() override;

    bool getBindings(void* const*& bindings) override;
    Blob* getBlob(const std::string& name) override;

private:
    CudaStreamPtr stream_;

    std::map<std::string, AsyncBlob*> blobs_dict_;
    std::vector<AsyncBlob*> input_blobs_;
    std::vector<AsyncBlob*> output_blobs_;
    std::vector<void*> bindings_;

    std::vector<AsyncBlob*> all_blobs_;
};

class UniformManager : public MemManager
{
public:
    UniformManager();
    ~UniformManager();

    bool init(nvinfer1::ICudaEngine* engine,
              nvinfer1::IExecutionContext* context) override;
    void copyIn() override;
    void copyOut() override;

    bool getBindings(void* const*& bindings) override;
    Blob* getBlob(const std::string& name) override;

private:
    std::map<std::string, UniformBlob*> blobs_dict_;
    std::vector<void*> bindings_;
    std::vector<UniformBlob*> all_blobs_;
};

class ZeroCopyManager : public MemManager
{
public:
    ZeroCopyManager();
    ~ZeroCopyManager();

    bool init(nvinfer1::ICudaEngine* engine,
              nvinfer1::IExecutionContext* context) override;
    void copyIn() override;
    void copyOut() override;

    bool getBindings(void* const*& bindings) override;
    Blob* getBlob(const std::string& name) override;

private:
    std::map<std::string, ZeroCopyBlob*> blobs_dict_;
    std::vector<void*> bindings_;
    std::vector<ZeroCopyBlob*> all_blobs_;

    static std::once_flag once_;
};

template <class MemManager>
class TrtEngine
{
public:
    TrtEngine(nvinfer1::ILogger& logger, CudaStreamPtr stream,
              MemManager* manager)
        : mem_manager_(manager)
        , logger_(logger)
        , stream_(stream) {}

    bool loadOnnx(const std::string& filename,
                  const dt_builder::BuilderParam& params) {
        dt_builder::Builder builder(logger_);
        
        nvinfer1::ICudaEngine* engine;
        if (!builder.buildEngine(params, engine)) {
            logger_.log(Severity::kERROR,
                        "load onnx and build engine failed.");
            return false;
        }

        engine_.reset(engine);
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            logger_.log(Severity::kERROR,
                        "create onnx execution context failed.");
            return false;
        }

        return mem_manager_->init(engine_.get(), context_.get());
    }

    bool loadEngine(const std::string& filename) {
        dt_builder::Builder builder(logger_);

        std::ifstream engine_file(filename.c_str(),
                                  std::ifstream::binary);
        engine_file.seekg(0, engine_file.end);
        int64_t fsize = engine_file.tellg();
        std::vector<char> engine_bytes(fsize);

        engine_file.seekg(0, std::ifstream::beg);
        engine_file.read(engine_bytes.data(), fsize);
        engine_file.close();

        TrtUniquePtr<nvinfer1::IRuntime> runtime(
            nvinfer1::createInferRuntime(logger_)
        );
        if (!runtime) {
            logger_.log(Severity::kERROR,
                        "create runtime failed.");
            return false;
        }
           
        engine_.reset(
            runtime->deserializeCudaEngine(engine_bytes.data(),
                                           fsize)
        );
        if (!engine_) {
            logger_.log(Severity::kERROR,
                        "deserialize cuda engine failed.");
            return false;
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            logger_.log(Severity::kERROR,
                       "create context failed.");
            return false;
        }

        return mem_manager_->init(engine_.get(), context_.get());
    }

    bool infer() {
        copyIn();
        bool ret = enqueue();
        copyOut();

        cudaStreamSynchronize(*stream_);
        return ret;
    }

    void copyIn() {
        mem_manager_->copyIn();
    }

    void copyOut() {
        mem_manager_->copyOut();
    }

    bool enqueue() {
        void* const* bindings;
        if (!mem_manager_->getBindings(bindings))
        {
            return false;
        }

        return context_->enqueueV2(bindings, *stream_, nullptr);
    }

    Blob* getBlob(const std::string& name) {
        return mem_manager_->getBlob(name);
    }

    TrtEngine(const TrtEngine& othr) = delete;
    TrtEngine& operator=(const TrtEngine& othr) = delete;

private:
    std::unique_ptr<MemManager> mem_manager_;
    nvinfer1::ILogger& logger_;
    CudaStreamPtr stream_;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;
};

} // namespace dt_infer

#endif  // TRT_SRC_INFER_TRT_ENGINE_H_
