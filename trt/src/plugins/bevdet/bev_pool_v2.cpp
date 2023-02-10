// Copyright (c) OpenMMLab. All rights reserved.
#include "bev_pool_v2.h"
#include "bev_pool_v2_kernel.h"


using namespace nvinfer1;

static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"bev_pool_v2"};

namespace dt_plugin {
namespace bevdet {

std::vector<PluginField> TRTBEVPoolV2Creator::attrs_;
PluginFieldCollection TRTBEVPoolV2Creator::fc_ {
  0, attrs_.data()
};

TRTBEVPoolV2::TRTBEVPoolV2() {}

TRTBEVPoolV2::TRTBEVPoolV2(const void *data, size_t length)
{
}

nvinfer1::IPluginV2DynamicExt *TRTBEVPoolV2::clone() const noexcept {
  return new TRTBEVPoolV2();
}

nvinfer1::DimsExprs TRTBEVPoolV2::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  // input[0] == depth
  // input[1] == feat
  // input[2] == ranks_depth
  // input[3] == ranks_feat
  // input[4] == ranks_bev
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = exprBuilder.constant(1); //Todo support batch>1
  ret.d[1] = exprBuilder.constant(128);
  ret.d[2] = exprBuilder.constant(128);
  ret.d[3] = inputs[1].d[3];
  return ret;
}

bool TRTBEVPoolV2::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
                                               int nbInputs, int nbOutputs) noexcept {
  // input[0] == depth->kFLOAT
  // input[1] == feat->kFLOAT
  // input[2] == ranks_depth->kINT32
  // input[3] == ranks_feat->kINT32
  // input[4] == ranks_bev->kINT32
  // input[5] == interval_starts->kINT32
  // input[6] == interval_lengths->kINT32
  // output[0] == bev_feat->kFLOAT
  if (pos == 0 || pos==1 || pos == 7) {
    return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return (ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  }
}

size_t TRTBEVPoolV2::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                        const nvinfer1::PluginTensorDesc *outputs,
                                        int nbOutputs) const noexcept {
  return 0;
}


int TRTBEVPoolV2::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                            const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                            void *const *outputs, void *workSpace,
                            cudaStream_t stream) noexcept {
  return 0;

  int c = inputDesc[1].dims.d[3];
  int n_intervals = inputDesc[5].dims.d[0];
  int n_points = outputDesc[0].dims.d[0] * outputDesc[0].dims.d[1] * outputDesc[0].dims.d[2] * outputDesc[0].dims.d[3];

  bev_pool_v2_set_zero(n_points, static_cast<float*>(outputs[0]));
  bev_pool_v2(c, n_intervals,
              static_cast<const float*>(inputs[0]),
              static_cast<const float*>(inputs[1]),
              static_cast<const int*>(inputs[2]),
              static_cast<const int*>(inputs[3]),
              static_cast<const int*>(inputs[4]),
              static_cast<const int*>(inputs[5]),
              static_cast<const int*>(inputs[6]),
              static_cast<float*>(outputs[0]),
	      stream);
  return 0;
}

nvinfer1::DataType TRTBEVPoolV2::getOutputDataType(int index,
                                                     const nvinfer1::DataType *inputTypes,
                                                     int nbInputs) const noexcept {
  return nvinfer1::DataType::kFLOAT;
}

const char *TRTBEVPoolV2::getPluginType() const noexcept { return PLUGIN_NAME; }

const char *TRTBEVPoolV2::getPluginVersion() const noexcept { return PLUGIN_VERSION; }


size_t TRTBEVPoolV2::getSerializationSize() const noexcept {
  return 0;
}

void TRTBEVPoolV2::serialize(void *buffer) const noexcept {
}


TRTBEVPoolV2Creator::TRTBEVPoolV2Creator() {

}

const char *TRTBEVPoolV2Creator::getPluginName() const noexcept { return PLUGIN_NAME; }

const char *TRTBEVPoolV2Creator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }

const PluginFieldCollection* TRTBEVPoolV2Creator::getFieldNames() noexcept
{
    return &fc_;
}

nvinfer1::IPluginV2 *TRTBEVPoolV2Creator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  const PluginField* fields = fc->fields;
    struct {
        int outWidth;
        int outHeight;
    } cfgs;

    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attr_name = fields[i].name;

        if (!strcmp(attr_name, "out_height"))
        {
            cfgs.outHeight = *static_cast<const int*>(fields[i].data);
        }
        else if (!strcmp(attr_name, "out_width"))
        {
            cfgs.outWidth = *static_cast<const int*>(fields[i].data);
        }
    }

    return new TRTBEVPoolV2();

}

nvinfer1::IPluginV2 *TRTBEVPoolV2Creator::deserializePlugin(const char *name,
                                                              const void *serialData,
                                                              size_t serialLength) noexcept {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TRTBEVPoolV2(serialData, serialLength);
}

}  // namespace mmdeploy
}

DEFINE_DT_PLUGIN(TRTBEVPoolV2, dt_plugin::bevdet::TRTBEVPoolV2Creator)
