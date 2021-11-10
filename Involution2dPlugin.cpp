#include <algorithm>
#include "involution2d.h"
#include <cuda_runtime_api.h>
#include <cuda_utils.h>
#include <iostream>
#include "Involution2dPlugin.h"
#include "utils.h"
//析构函数，释放GPU资源
nvinfer1::Involution2dPlugin::~Involution2dPlugin() {
    terminate();
}

nvinfer1::Involution2dPlugin::Involution2dPlugin(const std::vector<int32_t> &in_data_dims,
                                                 const std::vector<int32_t> &weight_dims,
                                                 const std::vector<int32_t> &kernel_size,
                                                 const std::vector<int32_t> &stride,
                                                 const std::vector<int32_t> &padding,
                                                 const std::vector<int32_t> &dilation,
                                                 const int32_t groups) : kernel_size(kernel_size), stride(stride),
                                                                         padding(padding), dilation(dilation),
                                                                          in_data_dims(in_data_dims),
                                                                         weight_dims(weight_dims),groups(groups) {

}


int
nvinfer1::Involution2dPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                      const nvinfer1::PluginTensorDesc *outputDesc,
                                      const void *const *inputs, void *const *outputs, void *workspace,
                                      cudaStream_t stream) PLUGIN_NOEXCEPT {


     auto *in_data = (float *) inputs[0];
     auto *weight_data = (float *) inputs[1];
     auto *out_data=(float *)outputs[0];

    const auto batch_size =this->in_data_dims[0];
     auto channels = this->in_data_dims[1];
     auto in_height = this->in_data_dims[2];
     auto in_width =this->in_data_dims[3];

     auto weight_height=this->weight_dims[2];
     auto weight_width=this->weight_dims[3];


    //weight dims-> batch_size, groups, kernel_size[0], kernel_size[1], weight_height, weight_width
    //out dims ->batch_size, channels, out_height, out_width

    auto out_height = (in_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1;
    auto out_width = (in_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1;

    auto num_elements=batch_size* channels*out_height* out_width;
    involution_cuda_forward(in_data, weight_data, out_data,weight_height,weight_width,num_elements,channels,this->groups,in_height,in_width,out_height,out_width,kernel_size[0],kernel_size[1],padding[0],padding[1],stride[0],stride[1],dilation[0],dilation[1],stream);

    return 0;
}

size_t nvinfer1::Involution2dPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                                      const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const PLUGIN_NOEXCEPT {

    return 0;
}

int nvinfer1::Involution2dPlugin::getNbOutputs() const PLUGIN_NOEXCEPT {
    return 1;
}

int nvinfer1::Involution2dPlugin::initialize() PLUGIN_NOEXCEPT {

    return 0;
}

void nvinfer1::Involution2dPlugin::terminate() PLUGIN_NOEXCEPT {

}

nvinfer1::DimsExprs
nvinfer1::Involution2dPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                                  nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT {


    //只有一个输出
    assert(outputIndex==0);
    //有两个输入
    assert(nbInputs==2);

    nvinfer1::DimsExprs ret{};
    ret.nbDims = 4;
    ret.d[0]=inputs[0].d[0];
    ret.d[1]=inputs[0].d[1];


    const auto in_height = this->in_data_dims[2];
    const auto in_width =this->in_data_dims[3];
    const auto out_height = (in_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1;
    const auto out_width = (in_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1;

    ret.d[2]=exprBuilder.constant(out_height);
    ret.d[3]=exprBuilder.constant(out_width);
    return ret;
}

size_t nvinfer1::Involution2dPlugin::getSerializationSize() const PLUGIN_NOEXCEPT {
    return 17*sizeof (int32_t);
}

void nvinfer1::Involution2dPlugin::serialize(void *buffer) const PLUGIN_NOEXCEPT {


 /*   //2*4
    const std::vector<int32_t> kernel_size;
    const std::vector<int32_t> stride;
    const std::vector<int32_t> padding;
    const std::vector<int32_t> dilation;
    //10
    const std::vector<int32_t> in_data_dims;
    const std::vector<int32_t> weight_dims;
    //1
    const int32_t groups;*/



 std::cout<<"call seri"<<std::endl;
    using namespace Tn;
    char* d = static_cast<char*>(buffer), *a = d;
    write(d,kernel_size[0]);
    write(d,kernel_size[1]);
    write(d,stride[0]);
    write(d,stride[1]);
    write(d,padding[0]);
    write(d,padding[1]);
    write(d,dilation[0]);
    write(d,dilation[1]);

    memcpy(d, &in_data_dims[0], 4*sizeof (int32_t));
    d += 4*sizeof (int32_t);
    memcpy(d, &weight_dims[0], 4*sizeof (int32_t));
    d += 4*sizeof (int32_t);
    write(d,groups);
    assert(d == a + getSerializationSize());


}

nvinfer1::Involution2dPlugin::Involution2dPlugin(const void *data, size_t length) {

    /*   //2*4
   const std::vector<int32_t> kernel_size;
   const std::vector<int32_t> stride;
   const std::vector<int32_t> padding;
   const std::vector<int32_t> dilation;
   //10
   const std::vector<int32_t> in_data_dims;
   const std::vector<int32_t> weight_dims;
   //1
   const int32_t groups;*/
    std::cout<<"call desr construct "<<std::endl;
    using namespace Tn;

    const int *data_start = static_cast<const int *>(data);
    const int *a=data_start;
    this->kernel_size=std::vector<int32_t>(data_start,data_start+2);
    data_start+=2;
    this->stride=std::vector<int32_t>(data_start,data_start+2);
    data_start+=2;
    this->padding=std::vector<int32_t>(data_start,data_start+2);
    data_start+=2;
    this->dilation=std::vector<int32_t>(data_start,data_start+2);
    data_start+=2;
    this->in_data_dims=std::vector<int32_t>(data_start,data_start+4);
    data_start+=4;
    this->weight_dims=std::vector<int32_t>(data_start,data_start+4);
    data_start+=4;

    std::vector<int32_t> temp=std::vector<int32_t>(data_start,data_start+1);
    this->groups=temp[0];
    data_start+=1;
    assert(data_start == a + length/sizeof (int32_t));
    std::cout<<"check"<<std::endl;
}

nvinfer1::IPluginV2DynamicExt *nvinfer1::Involution2dPlugin::clone()  const PLUGIN_NOEXCEPT  {


    auto *plugin = new Involution2dPlugin(in_data_dims,weight_dims,kernel_size,stride,padding,dilation,groups);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;

}

const char *nvinfer1::Involution2dPlugin::getPluginVersion() const PLUGIN_NOEXCEPT {
    return "1";
}

void nvinfer1::Involution2dPlugin::setPluginNamespace(const char *pluginNamespace)PLUGIN_NOEXCEPT {

    this->mPluginNamespace=pluginNamespace;
}

const char *nvinfer1::Involution2dPlugin::getPluginNamespace() const PLUGIN_NOEXCEPT {
    return mPluginNamespace;
}

void nvinfer1::Involution2dPlugin::destroy() PLUGIN_NOEXCEPT{
delete this;
}

bool
nvinfer1::Involution2dPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
                                                        int nbOutputs) PLUGIN_NOEXCEPT{

    switch (pos) {
        case 0:
            return (inOut[0].type == nvinfer1::DataType::kFLOAT &&
                    inOut[0].format == nvinfer1::TensorFormat::kLINEAR) ||
                   (inOut[0].type == nvinfer1::DataType::kINT32 &&
                    inOut[0].format == nvinfer1::TensorFormat::kLINEAR);
        case 1:(inOut[1].type == nvinfer1::DataType::kFLOAT &&
                inOut[1].format == nvinfer1::TensorFormat::kLINEAR) ||
               (inOut[1].type == nvinfer1::DataType::kINT32 &&
                inOut[1].format == nvinfer1::TensorFormat::kLINEAR);
        case 2:return inOut[1].type == inOut[0].type &&
                      inOut[1].format == inOut[0].format;
    }

}

const char *nvinfer1::Involution2dPlugin::getPluginType() const PLUGIN_NOEXCEPT {

    return "Involution2dPlugin";
}

nvinfer1::DataType
nvinfer1::Involution2dPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const PLUGIN_NOEXCEPT {

    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

void nvinfer1::Involution2dPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                                   const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs)  PLUGIN_NOEXCEPT{

    assert(nbInputs==2);
    assert(nbOutputs==1);
}



nvinfer1::Involution2dPluginCreator::Involution2dPluginCreator() {

    /*   //2*4
const std::vector<int32_t> kernel_size;
const std::vector<int32_t> stride;
const std::vector<int32_t> padding;
const std::vector<int32_t> dilation;
//10
const std::vector<int32_t> in_data_dims;
const std::vector<int32_t> weight_dims;
//1
const int32_t groups;*/

    this->mPluginAttributes = std::vector<PluginField>(
            {PluginField("kernel_size"),PluginField("stride"),
             PluginField("padding"), PluginField("dilation"),PluginField("in_data_dims"),PluginField("weight_dims"),PluginField("groups")});
    this->mFC.nbFields = mPluginAttributes.size();
    this->mFC.fields = mPluginAttributes.data();

}

const char *nvinfer1::Involution2dPluginCreator::getPluginName() const  PLUGIN_NOEXCEPT{
    return "Involution2dPlugin";
}

const char *nvinfer1::Involution2dPluginCreator::getPluginVersion() const PLUGIN_NOEXCEPT {
    return "1";
}

const nvinfer1::PluginFieldCollection *nvinfer1::Involution2dPluginCreator::getFieldNames() PLUGIN_NOEXCEPT {
    return &mFC;
}



nvinfer1::IPluginV2 *
nvinfer1::Involution2dPluginCreator::createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) PLUGIN_NOEXCEPT{

       //2*4
 std::vector<int32_t> kernel_size;
 std::vector<int32_t> stride;
 std::vector<int32_t> padding;
 std::vector<int32_t> dilation;
//10
 std::vector<int32_t> in_data_dims;
 std::vector<int32_t> weight_dims;
//1
 int32_t groups;

    for (int i = 0; i < fc->nbFields; i++) {
        if (fc->fields[i].data == nullptr) {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("kernel_size") == 0) {
            int data_size = fc->fields[i].length;
            const int *data_start = static_cast<const int *>(fc->fields[i].data);
            kernel_size = std::vector<int32_t>(data_start, data_start + data_size);
        }

        if (field_name.compare("dilation") == 0) {
            int data_size = fc->fields[i].length;
            const int *data_start = static_cast<const int *>(fc->fields[i].data);
            dilation = std::vector<int32_t>(data_start, data_start + data_size);
        }

        if (field_name.compare("padding") == 0) {
            int data_size = fc->fields[i].length;
            const int *data_start = static_cast<const int *>(fc->fields[i].data);
            padding = std::vector<int32_t>(data_start, data_start + data_size);
        }

        if (field_name.compare("stride") == 0) {
            int data_size = fc->fields[i].length;
            const int *data_start = static_cast<const int *>(fc->fields[i].data);
            stride = std::vector<int32_t>(data_start, data_start + data_size);
        }

        if (field_name.compare("in_data_dims") == 0) {
            int data_size = fc->fields[i].length;
            const int *data_start = static_cast<const int *>(fc->fields[i].data);
            in_data_dims = std::vector<int32_t>(data_start, data_start + data_size);
        }
        if (field_name.compare("weight_dims") == 0) {
            int data_size = fc->fields[i].length;
            const int *data_start = static_cast<const int *>(fc->fields[i].data);
            weight_dims = std::vector<int32_t>(data_start, data_start + data_size);
        }
        if (field_name.compare("groups") == 0) {
            int data_size = fc->fields[i].length;
            const int *data_start = static_cast<const int *>(fc->fields[i].data);
            groups = std::vector<int32_t>(data_start, data_start + data_size)[0];
        }
    }

    Involution2dPlugin *plugin =
            new Involution2dPlugin(in_data_dims,weight_dims,kernel_size,stride,padding,dilation,groups);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::IPluginV2 *
nvinfer1::Involution2dPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)PLUGIN_NOEXCEPT {

    std::cout<<"call deserializePlugin"<<std::endl;
    auto* obj = new Involution2dPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;

}
