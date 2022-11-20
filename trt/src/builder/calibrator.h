/***
 * Author: ZhuXiaolong
 * Date: 2022-11-17 13:26:08
 * LastEditTime: 2022-11-18 10:52:18
 * FilePath: /deploy_and_train/trt/src/builder/calibrator.h
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
#ifndef TRT_SRC_BUILDER_CALIBRATOR_H_
#define TRT_SRC_BUILDER_CALIBRATOR_H_

#include <NvInfer.h>
#include <fstream>
#include <string>
#include <vector>

namespace dt_builder
{

class DataLoader
{
public:
    DataLoader(int batch=1);
    virtual int getBatchSize() const noexcept;
    virtual bool next(int n, const char* names[],
                      void* bindings[]) noexcept;
                      
protected:
    int batch_size_;
};

template <class 
, nvinfer1::CalibrationAlgoType algo>
class Int8Calibrator : public nvinfer1::IInt8Calibrator
{
public:
    Int8Calibrator(const std::string& cache, DataLoader& loader)
        : dataloader_(loader)
        , cache_filename_(cache) {}

    int getBatchSize() const noexcept override {
        return dataloader_.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[],
                  int nbBindings) noexcept override {
        return dataloader_.next(nbBindings, names, bindings);
    }

    const void* readCalibrationCache(size_t& len) noexcept override {
        std::ifstream ifs(cache_filename_.c_str(),
                          std::ifstream::binary);
        if (!ifs.is_open()) {
            len = 0;
            return nullptr;
        }

        ifs.seekg(0, ifs.end);
        len = ifs.tellg();
        buffer_.resize(len);

        ifs.seekg(0, ifs.beg);
        ifs.read(buffer_.data(), len);
        ifs.close();

        return buffer_.data();
    }

    void writeCalibrationCache(const void* ptr, size_t len) noexcept override {
        std::ofstream ofs(cache_filename_.c_str(),
                          std::ofstream::binary);
        ofs.write(static_cast<const char*>(ptr), len);
        ofs.close();
    }

    nvinfer1::CalibrationAlgoType getAlgorithm () noexcept override {
        return algo;
    }

private:
    DataLoader dataloader_;
    std::vector<char> buffer_;
    std::string cache_filename_;
};
    
} // namespace dt_builder


#endif  // TRT_SRC_BUILDER_CALIBRATOR_H_