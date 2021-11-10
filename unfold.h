#pragma once
#include "macros.h"
namespace nvinfer1{

#if NV_TENSORRT_MAJOR > 7
#define PLUGIN_NOEXCEPT noexcept
#else
#define PLUGIN_NOEXCEPT
#endif

    class PluginDynamicBase : public nvinfer1::IPluginV2DynamicExt {
    public:
        PluginDynamicBase(const std::string &name) : mLayerName(name) {}
        // IPluginV2 Methods
        const char *getPluginVersion() const PLUGIN_NOEXCEPT override { return "1"; }
        int initialize() PLUGIN_NOEXCEPT override { return 0; }
        void terminate() PLUGIN_NOEXCEPT override {}
        void destroy() PLUGIN_NOEXCEPT override { delete this; }
        void setPluginNamespace(const char *pluginNamespace)
        PLUGIN_NOEXCEPT override {
            mNamespace = pluginNamespace;
        }
        const char *getPluginNamespace() const PLUGIN_NOEXCEPT override {
            return mNamespace.c_str();
        }

    protected:
        const std::string mLayerName;
        std::string mNamespace;

#if NV_TENSORRT_MAJOR < 8
    protected:
        using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
        using nvinfer1::IPluginV2DynamicExt::configurePlugin;
        using nvinfer1::IPluginV2DynamicExt::enqueue;
        using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
        using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
        using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
        using nvinfer1::IPluginV2DynamicExt::supportsFormat;
#endif

        // To prevent compiler warnings.
    };

    class PluginCreatorBase : public nvinfer1::IPluginCreator {
    public:
        const char *getPluginVersion() const PLUGIN_NOEXCEPT override { return "1"; };

        const nvinfer1::PluginFieldCollection *getFieldNames()
        PLUGIN_NOEXCEPT override {
            return &mFC;
        }

        void setPluginNamespace(const char *pluginNamespace)
        PLUGIN_NOEXCEPT override {
            mNamespace = pluginNamespace;
        }

        const char *getPluginNamespace() const PLUGIN_NOEXCEPT override {
            return mNamespace.c_str();
        }

    protected:
        nvinfer1::PluginFieldCollection mFC;
        std::vector<nvinfer1::PluginField> mPluginAttributes;
        std::string mNamespace;
    };

    /*class TorchUnfoldPlugin:public IPluginV2IOExt{

    public:
        TorchUnfoldPlugin(const std::string &name,
                                 const std::vector<int32_t> &kernelSize,
                                 const std::vector<int32_t> &dilation,
                                 const std::vector<int32_t> &padding,
                                 const std::vector<int32_t> &stride);
        TorchUnfoldPlugin(const std::string name, const void *data,
                                 size_t length);


        // It doesn't make sense to make TorchUnfoldPlugin without arguments,
        // so we delete default constructor.
        TorchUnfoldPlugin() = delete;

        int getNbOutputs() const TRT_NOEXCEPT override
                {
                        return 1;
                }

        DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                 nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT override;

        int initialize() TRT_NOEXCEPT override;

        virtual void terminate() TRT_NOEXCEPT override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0; }

        virtual int enqueue(int batchSize, const void* const* inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

        virtual size_t getSerializationSize() const TRT_NOEXCEPT override;

        virtual void serialize(void* buffer) const TRT_NOEXCEPT override;


        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const TRT_NOEXCEPT override;

        const char* getPluginVersion() const TRT_NOEXCEPT override;

        void destroy() TRT_NOEXCEPT override;

        IPluginV2IOExt* clone() const TRT_NOEXCEPT override;

        void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

        const char* getPluginNamespace() const TRT_NOEXCEPT override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

        void attachToContext(
                cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT override;

        void detachFromContext() TRT_NOEXCEPT override;




    };*/

    class TorchUnfoldPluginDynamic : public PluginDynamicBase {
    public:
        TorchUnfoldPluginDynamic(const std::string &name,
                                 const std::vector<int32_t> &kernelSize,
                                 const std::vector<int32_t> &dilation,
                                 const std::vector<int32_t> &padding,
                                 const std::vector<int32_t> &stride);

        TorchUnfoldPluginDynamic(const std::string name, const void *data,
                                 size_t length);

        // It doesn't make sense to make TorchUnfoldPluginDynamic without arguments,
        // so we delete default constructor.
        TorchUnfoldPluginDynamic() = delete;

        // IPluginV2DynamicExt Methods
        nvinfer1::IPluginV2DynamicExt *clone() const PLUGIN_NOEXCEPT override;
        nvinfer1::DimsExprs getOutputDimensions(
                int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT override;
        bool supportsFormatCombination(int pos,
                                       const nvinfer1::PluginTensorDesc *inOut,
                                       int nbInputs,
                                       int nbOutputs) PLUGIN_NOEXCEPT override;
        void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                             int nbInputs,
                             const nvinfer1::DynamicPluginTensorDesc *out,
                             int nbOutputs) PLUGIN_NOEXCEPT override;
        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                int nbInputs,
                                const nvinfer1::PluginTensorDesc *outputs,
                                int nbOutputs) const PLUGIN_NOEXCEPT override;
        int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                    const nvinfer1::PluginTensorDesc *outputDesc,
                    const void *const *inputs, void *const *outputs, void *workspace,
                    cudaStream_t stream) PLUGIN_NOEXCEPT override;

        // IPluginV2Ext Methods
        nvinfer1::DataType getOutputDataType(
                int index, const nvinfer1::DataType *inputTypes,
                int nbInputs) const PLUGIN_NOEXCEPT override;

        // IPluginV2 Methods
        const char *getPluginType() const PLUGIN_NOEXCEPT override;
        const char *getPluginVersion() const PLUGIN_NOEXCEPT override;
        int getNbOutputs() const PLUGIN_NOEXCEPT override;
        size_t getSerializationSize() const PLUGIN_NOEXCEPT override;
        void serialize(void *buffer) const PLUGIN_NOEXCEPT override;

    private:
        std::vector<int32_t> mKernelSize;
        std::vector<int32_t> mDilation;
        std::vector<int32_t> mPadding;
        std::vector<int32_t> mStride;
    };

    class TorchUnfoldPluginDynamicCreator : public PluginCreatorBase {
    public:
        TorchUnfoldPluginDynamicCreator();

        const char *getPluginName() const PLUGIN_NOEXCEPT override;

        const char *getPluginVersion() const PLUGIN_NOEXCEPT override;

        nvinfer1::IPluginV2 *createPlugin(const char *name,
                                          const nvinfer1::PluginFieldCollection *fc)
        PLUGIN_NOEXCEPT override;

        nvinfer1::IPluginV2 *deserializePlugin(
                const char *name, const void *serialData,
                size_t serialLength) PLUGIN_NOEXCEPT override;
    };
    REGISTER_TENSORRT_PLUGIN(TorchUnfoldPluginDynamicCreator);

}