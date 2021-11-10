//
// Created by lxf on 2021/9/16.
//

#ifndef YOLOV5_INVOLUTION2DPLUGIN_H
#define YOLOV5_INVOLUTION2DPLUGIN_H

#include "cuda_utils.h"
#include "involution2d.h"
#include <NvInferRuntime.h>
#include <vector>
#include <string>
#include "torch_unfold.h"
namespace nvinfer1 {

#if NV_TENSORRT_MAJOR > 7
#define PLUGIN_NOEXCEPT noexcept
#else
#define PLUGIN_NOEXCEPT
#endif
    class Involution2dPlugin final : public nvinfer1::IPluginV2DynamicExt {


    public:

        Involution2dPlugin(
                const std::vector<int32_t> &in_data_dims,
                const std::vector<int32_t> &weight_dims,
                const std::vector<int32_t> &kernel_size,
                const std::vector<int32_t> &stride,
                const std::vector<int32_t> &padding,
                const std::vector<int32_t> &dilation,
                int32_t groups
        );

        Involution2dPlugin(void const *data, size_t length);

        Involution2dPlugin() = delete;


        ~Involution2dPlugin() override;

        int getNbOutputs()  const PLUGIN_NOEXCEPT override;

        nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                                nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT override  ;

        int initialize() PLUGIN_NOEXCEPT override;

        void terminate() PLUGIN_NOEXCEPT override;

        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const PLUGIN_NOEXCEPT override;

        int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                    const void *const *inputs, void *const *outputs,
                    void *workspace,
                    cudaStream_t stream) PLUGIN_NOEXCEPT override;

        size_t getSerializationSize() const PLUGIN_NOEXCEPT  override;

        void serialize(void *buffer) const PLUGIN_NOEXCEPT override;

        bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
                                       int nbOutputs)PLUGIN_NOEXCEPT override;

        const char *getPluginType() const PLUGIN_NOEXCEPT override;

        const char *getPluginVersion() const  PLUGIN_NOEXCEPT override;

        void destroy() PLUGIN_NOEXCEPT override;

        nvinfer1::IPluginV2DynamicExt *clone() const PLUGIN_NOEXCEPT override;

        void setPluginNamespace(const char *pluginNamespace) PLUGIN_NOEXCEPT override;

        const char *getPluginNamespace()  const PLUGIN_NOEXCEPT override ;

        nvinfer1::DataType
        getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const PLUGIN_NOEXCEPT  override;

        void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                             const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) PLUGIN_NOEXCEPT override;

    protected:
        const char *mPluginNamespace;
    private:

        //2*4
        std::vector<int32_t> kernel_size;
        std::vector<int32_t> stride;
        std::vector<int32_t> padding;
        std::vector<int32_t> dilation;
        //4*2
        std::vector<int32_t> in_data_dims;
        std::vector<int32_t> weight_dims;
        //1
        int32_t groups;


    };

    class Involution2dPluginCreator : public IPluginCreator {
    public:
        Involution2dPluginCreator();

        ~Involution2dPluginCreator() override = default;

        const char *getPluginName()  const PLUGIN_NOEXCEPT override;

        const char *getPluginVersion() const PLUGIN_NOEXCEPT override;

        const PluginFieldCollection *getFieldNames() PLUGIN_NOEXCEPT  override;

        IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT override;

        IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) PLUGIN_NOEXCEPT  override;

        void setPluginNamespace(const char *libNamespace)  PLUGIN_NOEXCEPT override {
            mNamespace = libNamespace;
        }

        const char *getPluginNamespace() const PLUGIN_NOEXCEPT  override {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        PluginFieldCollection mFC;
        std::vector<PluginField> mPluginAttributes;
    };

    REGISTER_TENSORRT_PLUGIN(Involution2dPluginCreator);
}


#endif //YOLOV5_INVOLUTION2DPLUGIN_H
