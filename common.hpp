#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yololayer.h"

using namespace nvinfer1;

cv::Rect get_rect(cv::Mat &img, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), //left
            (std::min)(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
            (std::max)(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
            (std::min)(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection &a, const Yolo::Detection &b) {
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection> &res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto &item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        std::cout<<name<<std::endl;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}
IPluginV2Layer* addInvolution2d(INetworkDefinition *network,ITensor &input,ITensor &weight,int kernelHeight, int kernelWidth, int strideHeight,
                                int strideWidth,
                                int dilationHeight, int dilationWidth, int paddingHeight, int paddingWidth,int groups){
    auto creator = getPluginRegistry()->getPluginCreator("Involution2dPlugin", "1");
    PluginField pluginField[7];
    int kernelSize[2] = {kernelHeight, kernelWidth};
    pluginField[0].type = PluginFieldType::kINT32;
    pluginField[0].name = "kernel_size";
    pluginField[0].data = kernelSize;
    pluginField[0].length = 2;

    int dilationSize[2] = {dilationHeight, dilationWidth};
    pluginField[1].type = PluginFieldType::kINT32;
    pluginField[1].name = "dilation";
    pluginField[1].data = dilationSize;
    pluginField[1].length = 2;

    int strideSize[2] = {strideHeight, strideWidth};
    pluginField[2].type = PluginFieldType::kINT32;
    pluginField[2].name = "stride";
    pluginField[2].data = strideSize;
    pluginField[2].length = 2;

    int paddingSize[2] = {paddingHeight, paddingWidth};
    pluginField[3].type = PluginFieldType::kINT32;
    pluginField[3].name = "padding";
    pluginField[3].data = paddingSize;
    pluginField[3].length = 2;

    pluginField[4].type=PluginFieldType::kINT32;
    assert(input.getDimensions().nbDims==4);
    int inDataDims[4];
    for (int i=0;i<4;i++){
        inDataDims[i]=input.getDimensions().d[i];
    }
    pluginField[4].name = "in_data_dims";
    pluginField[4].data = inDataDims;
    pluginField[4].length = 4;


    pluginField[5].type=PluginFieldType::kINT32;
    assert(weight.getDimensions().nbDims==4);
    int weightDims[4];
    for (int i=0;i<4;i++){
        weightDims[i]=weight.getDimensions().d[i];
    }
    pluginField[5].name = "weight_dims";
    pluginField[5].data = weightDims;
    pluginField[5].length = 4;

    int groupsSize[1] = {groups};
    pluginField[6].type=PluginFieldType::kINT32;
    pluginField[6].name="groups";
    pluginField[6].length=1;
    pluginField[6].data=groupsSize;


    PluginFieldCollection pluginData;
    pluginData.nbFields = 7;
    pluginData.fields = pluginField;
    IPluginV2 *plugin_obj = creator->createPlugin("involution2dlayer", &pluginData);

    std::vector<ITensor *> input_tensors;
    input_tensors.push_back(&input);
    input_tensors.push_back(&weight);
    auto involution2d = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return involution2d;
}

IPluginV2Layer *
addUnfoldLayer(INetworkDefinition *network, ITensor &input, int kernelHeight, int kernelWidth, int strideHeight,
               int strideWidth,
               int dilationHeight, int dilationWidth, int paddingHeight, int paddingWidth) {
    auto creator = getPluginRegistry()->getPluginCreator("TorchUnfoldPluginDynamic", "1");

    Dims dims = input.getDimensions();

    PluginField pluginField[4];
    int kernelSize[2] = {kernelHeight, kernelWidth};
    pluginField[0].type = PluginFieldType::kINT32;
    pluginField[0].name = "kernel_size";
    pluginField[0].data = kernelSize;
    pluginField[0].length = 2;

    int dilationSize[2] = {dilationHeight, dilationWidth};
    pluginField[1].type = PluginFieldType::kINT32;
    pluginField[1].name = "dilation";
    pluginField[1].data = dilationSize;
    pluginField[1].length = 2;

    int strideSize[2] = {strideHeight, strideWidth};
    pluginField[2].type = PluginFieldType::kINT32;
    pluginField[2].name = "stride";
    pluginField[2].data = strideSize;
    pluginField[2].length = 2;

    int paddingSize[2] = {paddingHeight, paddingWidth};
    pluginField[3].type = PluginFieldType::kINT32;
    pluginField[3].name = "padding";
    pluginField[3].data = paddingSize;
    pluginField[3].length = 2;

    PluginFieldCollection pluginData;
    pluginData.nbFields = 4;
    pluginData.fields = pluginField;
    IPluginV2 *plugin_obj = creator->createPlugin("unfoldlayer", &pluginData);

    std::vector<ITensor *> input_tensors;
    input_tensors.push_back(&input);
    auto unfold = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return unfold;
}

IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input,
                            std::string lname, float eps) {
    float *gamma = (float *) weightMap[lname + ".weight"].values;
    float *beta = (float *) weightMap[lname + ".bias"].values;
    float *mean = (float *) weightMap[lname + ".running_mean"].values;
    float *var = (float *) weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }

    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}
ILayer *inConvNew(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch,
                  int ksize, int s,  int dilation, const std::string& lname){
    int p = ksize / 2;
    int groups = 1;
    int reduceRatio = 4;

    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int inChannel = input.getDimensions().d[1];

    //------分割线-------- weight
    //o_mapping:
    ITensor *afterOMapping = nullptr;
    if (s != 1) {
        IPoolingLayer *avgPool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{s, s});
        avgPool->setStrideNd(Dims2{s, s});
        afterOMapping = avgPool->getOutput(0);
    } else {
        IIdentityLayer *identityLayer = network->addIdentity(input);
        afterOMapping = identityLayer->getOutput(0);
    }
    //reduce_mapping:

    IConvolutionLayer *reduceLayer = network->addConvolutionNd(*afterOMapping, outch / reduceRatio, DimsHW{1, 1},
                                                               weightMap[lname+".conv.reduce_mapping.weight"], emptywts);
    assert(reduceLayer);
    //sigma_mapping
    IScaleLayer *bn2d = addBatchNorm2d(network, weightMap, *reduceLayer->getOutput(0), lname+".conv.sigma_mapping.0", 1e-5);
    assert(bn2d);
    //IActivationLayer *reluLayer = network->addActivation(*bn2d->getOutput(0), ActivationType::kRELU);
    //assert(reluLayer);
    // silu = x * sigmoid
    auto sig = network->addActivation(*bn2d->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(*bn2d->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);


    //span_mapping
    IConvolutionLayer *spanLayer = network->addConvolutionNd(*ew->getOutput(0), ksize * ksize * groups,
                                                             DimsHW{1, 1}, weightMap[lname+".conv.span_mapping.weight"], emptywts);
    assert(spanLayer);
    ITensor* weight=spanLayer->getOutput(0);

    //------分割线--------input_init

    ITensor *input_init;
    if (inChannel != outch) {
        Weights initialMappingWeight = weightMap[lname + ".conv.initial_mapping.weight"];
        IConvolutionLayer * initConvLayer= network->addConvolutionNd(input, outch, Dims2{1, 1}, initialMappingWeight,
                                                                     emptywts);
        initConvLayer->setPaddingNd(Dims2{0, 0});
        input_init=initConvLayer->getOutput(0);
    } else{
        input_init=network->addIdentity(input)->getOutput(0);
    }
    //------分割线--------input_init

   return addInvolution2d(network,*input_init,*weight,ksize,ksize,s,s,dilation,dilation,p,p,groups);
}


ILayer *inConvBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch,
                    int ksize, int s,  int dilation, const std::string& lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = ksize / 2;
    int groups = 1;
    int reduceRatio = 4;

    int batchSize = input.getDimensions().d[0];
    int inChannel = input.getDimensions().d[1];
    assert(inChannel%groups==0);
    int inputHeight = input.getDimensions().d[2];
    int inputWidth = input.getDimensions().d[3];

    int outHeight = (inputHeight + 2 * p - dilation * (ksize - 1) - 1) / s + 1;
    int outWidth = outHeight;


    //输入输出通道不匹配
    IPluginV2Layer *unfoldLayer;
    if (inChannel != outch){
        Weights initialMappingWeight=weightMap[lname+".conv.initial_mapping.weight"];
        IConvolutionLayer *initConvLayer = network->addConvolutionNd(input, outch, Dims2{1, 1}, initialMappingWeight,emptywts);
        //initConvLayer->setName(std::string(lname+".conv.init_mapping").c_str());
        initConvLayer->setPaddingNd(Dims2{0,0});
        ITensor *afterInit = initConvLayer->getOutput(0);
        unfoldLayer = addUnfoldLayer(network, *afterInit, ksize, ksize, s, s, 1, 1, p, p);
    }else{
        unfoldLayer= addUnfoldLayer(network,input,ksize, ksize, s, s, 1, 1, p, p);
        unfoldLayer->setName(std::string(lname+"unfold op").c_str());
    }

    ITensor *afterUnfold = unfoldLayer->getOutput(0);
    assert(afterUnfold);
    IShuffleLayer *shuffleLayer=network->addShuffle(*afterUnfold);
    shuffleLayer->setReshapeDimensions(Dims{6, {batchSize, groups, outch / groups, ksize*ksize, outHeight, outWidth}});
    afterUnfold=shuffleLayer->getOutput(0);

    //------分割线--------
    //o_mapping:
    ITensor *afterOMapping = nullptr;
    if (s != 1) {
        IPoolingLayer *avgPool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{s, s});
        avgPool->setStrideNd(Dims2{s, s});
        afterOMapping = avgPool->getOutput(0);
    } else {
        IIdentityLayer *identityLayer = network->addIdentity(input);
        afterOMapping = identityLayer->getOutput(0);
    }
    //reduce_mapping:

    IConvolutionLayer *reduceLayer = network->addConvolutionNd(*afterOMapping, outch / reduceRatio, DimsHW{1, 1},
                                                               weightMap[lname+".conv.reduce_mapping.weight"], emptywts);
    assert(reduceLayer);
    //sigma_mapping
    IScaleLayer *bn2d = addBatchNorm2d(network, weightMap, *reduceLayer->getOutput(0), lname+".conv.sigma_mapping.0", 1e-5);
    assert(bn2d);
    IActivationLayer *reluLayer = network->addActivation(*bn2d->getOutput(0), ActivationType::kRELU);
    assert(reluLayer);
    //span_mapping
    IConvolutionLayer *spanLayer = network->addConvolutionNd(*reluLayer->getOutput(0), ksize * ksize * groups,
                                                             DimsHW{1, 1}, weightMap[lname+".conv.span_mapping.weight"], emptywts);
    assert(spanLayer);



    ITensor *kernel = spanLayer->getOutput(0);
    assert(kernel->getDimensions().nbDims == 4);

    IShuffleLayer *kernelShuffleLayer=network->addShuffle(*kernel);

    kernelShuffleLayer->setReshapeDimensions(Dims{6,{batchSize,groups,1,ksize*ksize,kernel->getDimensions().d[2],kernel->getDimensions().d[3]}});
    kernel = kernelShuffleLayer->getOutput(0);

    //IMatrixMultiplyLayer *matrixMultiplyLayer = network->addMatrixMultiply(*afterUnfold, MatrixOperation::kVECTOR,*kernel, MatrixOperation::kVECTOR);

    IElementWiseLayer *elementWiseLayer= network->addElementWise(*afterUnfold,*kernel,ElementWiseOperation::kPROD);
    IReduceLayer *resReduceLayer = network->addReduce(*elementWiseLayer->getOutput(0), ReduceOperation::kSUM, 8,
                                                      false);
    ITensor*afterResReduce=resReduceLayer->getOutput(0);
    IShuffleLayer *lastShuffleLayer=network->addShuffle(*afterResReduce);
    lastShuffleLayer->setReshapeDimensions(Dims4{batchSize,-1,afterResReduce->getDimensions().d[3],afterResReduce->getDimensions().d[4]});
    return lastShuffleLayer;

}

ILayer *
convBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int ksize,
          int s, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = ksize / 2;
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize},
                                                         weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);
    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

ILayer *focusWithBs(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch,
                    int ksize, std::string lname){

    int bs=input.getDimensions().d[0];
    assert(bs);
    ISliceLayer *s1 = network->addSlice(input, Dims4{0,0, 0, 0}, Dims4{1,inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2},
                                        Dims4{1,1, 2, 2});
    ISliceLayer *s2 = network->addSlice(input, Dims4{0,0, 1, 0}, Dims4{1,inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2},
                                        Dims4{1,1, 2, 2});
    ISliceLayer *s3 = network->addSlice(input, Dims4{0,0, 0, 1}, Dims4{1,inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2},
                                        Dims4{1,1, 2, 2});
    ISliceLayer *s4 = network->addSlice(input, Dims4{0,0, 1, 1}, Dims4{1,inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2},
                                        Dims4{1,1, 2, 2});
    ITensor *inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer *
focus(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch,
      int ksize, std::string lname) {

    ISliceLayer *s1 = network->addSlice(input, Dims3{0, 0, 0}, Dims3{inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2},
                                        Dims3{1, 2, 2});
    ISliceLayer *s2 = network->addSlice(input, Dims3{0, 1, 0}, Dims3{inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2},
                                        Dims3{1, 2, 2});
    ISliceLayer *s3 = network->addSlice(input, Dims3{0, 0, 1}, Dims3{inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2},
                                        Dims3{1, 2, 2});
    ISliceLayer *s4 = network->addSlice(input, Dims3{0, 1, 1}, Dims3{inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2},
                                        Dims3{1, 2, 2});
    ITensor *inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer *
bottleneck(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2,
           bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, (int) ((float) c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer *
bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2,
              int n, bool shortcut, int g, float e, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int c_ = (int) ((float) c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{1, 1}, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{1, 1}, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor *inputTensors[] = {cv3->getOutput(0), cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer *bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}

ILayer *
C3(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int n,
   bool shortcut, int g, float e, std::string lname) {
    int c_ = (int) ((float) c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    ITensor *inputTensors[] = {y1, cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    return cv3;
}

ILayer *
SPP(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int k1,
    int k2, int k3, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k1, k1});
    pool1->setPaddingNd(DimsHW{k1 / 2, k1 / 2});
    pool1->setStrideNd(DimsHW{1, 1});
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k2, k2});
    pool2->setPaddingNd(DimsHW{k2 / 2, k2 / 2});
    pool2->setStrideNd(DimsHW{1, 1});
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k3, k3});
    pool3->setPaddingNd(DimsHW{k3 / 2, k3 / 2});
    pool3->setStrideNd(DimsHW{1, 1});

    ITensor *inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights> &weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = Yolo::CHECK_COUNT * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto *p = (const float *) wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}



IPluginV2Layer *addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, std::string lname,
                             std::vector<IConvolutionLayer *> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);
    PluginField plugin_fields[2];
    int netinfo[4] = {Yolo::CLASS_NUM, Yolo::INPUT_W, Yolo::INPUT_H, Yolo::MAX_OUTPUT_BBOX_COUNT};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    int scale = 8;
    std::vector<Yolo::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Yolo::YoloKernel kernel;
        kernel.width = Yolo::INPUT_W / scale;
        kernel.height = Yolo::INPUT_H / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor *> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}

#endif

