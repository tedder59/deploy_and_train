/***
 * Author: ZhuXiaolong
 * Date: 2022-11-13 14:29:45
 * LastEditTime: 2022-11-15 10:39:30
 * FilePath: /deploy_and_train/trt/src/plugins/plugin.cpp
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

#include "centernet/decode_plugin.h"
#include "centernet/deformable_conv2d_plugin.h"

using namespace dt_plugin;

bool registerDtPlugins()
{
    bool ret = true;
    ret = ret && registerCenterNetDecodePlugin();
    ret = ret && registerDeformableConv2dPlugin();
    return ret;
}

bool dt_plugin_status = registerDtPlugins();
