#pragma once
#include <cstdint>
#include <NvInferRuntimeCommon.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include "cuda_utils.h"
namespace nvinfer1 {


 /*   void involution_cuda_forward(float const* in_data,
                                 float const* weight_data,
                                 float const* out_data,
                                 const int64_t *in_data_dims,
                                 const int64_t *weight_data_dims,
                                 const int64_t num_elements,
                                 const int64_t channels,
                                 const int64_t groups,
                                 const int64_t in_height, const int64_t in_width,
                                 const int64_t out_height, const int64_t out_width,
                                 const int64_t kernel_height, const int64_t kernel_width,
                                 const int64_t pad_h, const int64_t pad_w,
                                 const int64_t stride_h, const int64_t stride_w,
                                 const int64_t dilation_h, const int64_t dilation_w,
                                 cudaStream_t stream);*/


    template<typename T>
    void involution_cuda_forward(T * in_data,
                                 T * weight_data,
                                 T * out_data,
                                 const int64_t weight_height,
                                 const int64_t weight_width,
                                 const int64_t num_elements,
                                 const int64_t channels,
                                 const int64_t groups,
                                 const int64_t in_height, const int64_t in_width,
                                 const int64_t out_height, const int64_t out_width,
                                 const int64_t kernel_height, const int64_t kernel_width,
                                 const int64_t pad_h, const int64_t pad_w,
                                 const int64_t stride_h, const int64_t stride_w,
                                 const int64_t dilation_h, const int64_t dilation_w,
                                 cudaStream_t stream);


}