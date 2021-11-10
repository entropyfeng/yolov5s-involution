#include "involution2d.h"

namespace nvinfer1 {


    template<typename T>
    __global__ void
    involution_2d_kernel(T *const in_data,
                         T *const weight_data,
                         T *const out_data,
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
                         const int64_t dilation_h, const int64_t dilation_w) {

        int64_t in_data_1 = channels;
        int64_t in_data_2 = in_height;
        int64_t in_data_3 = in_width;
        int64_t weight_1 = groups;
        int64_t weight_2 = kernel_height;
        int64_t weight_3 = kernel_width;
        int64_t weight_4 = weight_height;
        int64_t weight_5 = weight_width;
        CUDA_KERNEL_LOOP(idx, num_elements) {
            const int64_t w = idx % out_width;
            const int64_t h = (idx / out_width) % out_height;
            int64_t divisor = out_width * out_height;
            const int64_t c = (idx / divisor) % channels;
            divisor *= channels;
            const int64_t n = idx / divisor;
            const int64_t g = c / (channels / groups);

            T value = 0;

            for (int64_t kh = 0l; kh < kernel_height; kh++) {
                const int64_t h_in = h * stride_h + kh * dilation_h - pad_h;

                if ((0l <= h_in) && (h_in < in_height)) {
                    for (int64_t kw = 0l; kw < kernel_width; kw++) {
                        const int64_t w_in = w * stride_w + kw * dilation_w - pad_w;
                        if ((0l <= w_in) && (w_in < in_width)) {
                            auto in_data_pos = n * (in_data_1 * in_data_2 * in_data_3) +
                                               c * (in_data_2 * in_data_3) + h_in * in_data_3 + w_in;
                            auto weight_data_pos = n *
                                                   (weight_1 * weight_2 * weight_3 *
                                                    weight_4 * weight_5) + g *
                                                                           (weight_2 *
                                                                            weight_3 *
                                                                            weight_4 *
                                                                            weight_5) +
                                                   kh *
                                                   (weight_3 * weight_4 * weight_5) +
                                                   kw * (weight_4 * weight_5) +
                                                   h * weight_5 + w;
                            value += weight_data[weight_data_pos] * in_data[in_data_pos];
                        }
                    }
                }
            }
            out_data[idx] = value;
        }
    }


    template void involution_cuda_forward<float>(float *in_data,
                                                 float *weight_data,
                                                 float *out_data,
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


    template<typename T>
    void nvinfer1::involution_cuda_forward( T *in_data,
                                            T *weight_data,
                                            T *out_data,
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
                                           cudaStream_t stream) {



        involution_2d_kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
                in_data,
                weight_data,
                out_data,
                weight_height,
                weight_width,
                num_elements,
                channels,
                groups,
                in_height,
                in_width,
                out_height,
                out_width,
                kernel_height,
                kernel_width,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w);

    }


}