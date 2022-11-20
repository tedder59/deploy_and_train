/***
 * Author: ZhuXiaolong
 * Date: 2022-11-18 10:18:12
 * LastEditTime: 2022-11-18 10:18:13
 * FilePath: /deploy_and_train/trt/src/builder/calibrator.cpp
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
#include "builder/calibrator.h"

namespace dt_builder
{

DataLoader::DataLoader(int batch)
    : batch_size_(batch)
{

}

int DataLoader::getBatchSize() const noexcept
{
    return batch_size_;
}

bool DataLoader::next(int n, const char* names[],
                      void* bindings[]) noexcept
{
    return false;
}

} // namespace dt_builder
