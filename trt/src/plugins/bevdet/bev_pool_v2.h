// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_BEV_POOL_HPP
#define TRT_BEV_POOL_HPP

#include "common.h"

namespace dt_plugin {
namespace bevdet {

class TRTBEVPoolV2 : public DynamicPlugin {
 public:
  TRTBEVPoolV2();
  TRTBEVPoolV2(const void *data, size_t length);
  ~TRTBEVPoolV2() noexcept override = default;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                          int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
      noexcept override;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs,
                                 int nbOutputs) noexcept override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
              void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const noexcept override;

  // IPluginV2 Methods
  const char *getPluginType() const noexcept override;

  const char *getPluginVersion() const noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void *buffer) const noexcept override;

};

class TRTBEVPoolV2Creator : public DynamicPluginCreator {
 public:
  TRTBEVPoolV2Creator();

  const char *getPluginName() const noexcept override;

  const char *getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc
    ) noexcept override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                         size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection fc_;
    static std::vector<nvinfer1::PluginField> attrs_;

};
}
}  // namespace mmdeploy

extern "C" bool registerTRTBEVPoolV2();
#endif  // TRT_GRID_SAMPLER_HPP
