#include "infer/trt_engine.h"

using dt_infer::InferLogger;
using dt_infer::SyncManager;
using dt_infer::AsyncManager;
using dt_infer::UniformManager;
using dt_infer::ZeroCopyManager;
using dt_infer::TrtEngine;

int main(int argc, char** argv)
{
    const char* engine_filename = argv[1];
    std::unique_ptr<TrtEngine<SyncManager>> trt;

    InferLogger& logger = InferLogger::getLogger();
    auto stream = dt_infer::makeCudaStream(logger);
    auto* manager = new SyncManager();
    trt.reset(new TrtEngine<SyncManager>(logger, stream, manager));
    if (!trt->loadEngine(engine_filename)) {
        return 1;
    }

    trt->infer();
    return 0;
}