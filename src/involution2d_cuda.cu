#include <involution2d_cuda.cuh>

namespace involution {
namespace cuda {

static u_int32_t ceildiv(u_int32_t num_elements, u_int32_t threads) {
    return (num_elements + threads - 1) / threads;
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS_PER_BLOCK)
__global__ static void involution2d_forward_kernel(
    const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, int64_t> in_data,
    const at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, int64_t> weight_data,
    scalar_t* const out_data,
    const int64_t num_elements,
    const int64_t channels,
    const int64_t groups,
    const int64_t in_height,        const int64_t in_width,
    const int64_t out_height,       const int64_t out_width,
    const int64_t kernel_height,    const int64_t kernel_width,
    const int64_t pad_h,            const int64_t pad_w,
    const int64_t stride_h,         const int64_t stride_w,
    const int64_t dilation_h,       const int64_t dilation_w
) {
    CUDA_KERNEL_LOOP(idx, num_elements) {
        const int64_t w = idx % out_width;
        const int64_t h = (idx / out_width) % out_height;
        int64_t divisor = out_width * out_height;
        const int64_t c = (idx / divisor) % channels;
        divisor *= channels;
        const int64_t n = idx / divisor;
        const int64_t g = c / (channels / groups);

        scalar_t value = 0;

        for (int64_t kh = 0l; kh < kernel_height; kh++) {
            const int64_t h_in = h * stride_h + kh * dilation_h - pad_h;

            if ((0l <= h_in) && (h_in < in_height)) {
                for (int64_t kw = 0l; kw < kernel_width; kw++) {
                    const int64_t w_in = w * stride_w + kw * dilation_w - pad_w;

                    if ((0l <= w_in) && (w_in < in_width)) {
                        value += weight_data[n][g][kh][kw][h][w] * in_data[n][c][h_in][w_in];
                    }
                }
            }
        }

        out_data[idx] = value;
    }
}

at::Tensor involution2d_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    const int64_t groups
) {
    AT_ASSERTM(input.device().is_cuda(), "\"input\" must be a CUDA tensor.");
    AT_ASSERTM(weight.device().is_cuda(), "\"weight\" must be a CUDA tensor.");

    at::TensorArg input_t{input, "input", 1}, weight_t{weight, "weight", 2};

    at::CheckedFrom c = __func__;
    at::checkAllSameGPU(c, {input_t, weight_t});
    at::checkAllSameType(c, {input_t, weight_t});

    at::cuda::CUDAGuard device_guard(input.device());

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);

    const auto weight_height = weight.size(2);
    const auto weight_width = weight.size(3);

    const at::Tensor weight_ = weight.view({batch_size, groups, kernel_size[0], kernel_size[1], weight_height, weight_width});

    const auto out_height = (in_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1;
    const auto out_width = (in_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1;

    at::Tensor output = at::zeros({batch_size, channels, out_height, out_width}, input.options());
    const auto num_elements = output.numel();

    if (num_elements == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return output;
    }

    const auto threads = std::min(static_cast<u_int32_t>(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock), CUDA_MAX_THREADS_PER_BLOCK);
    const dim3 num_blocks(ceildiv(num_elements, threads), 1u, 1u);
    const dim3 threads_per_block(threads, 1u, 1u);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        input.scalar_type(),
        "involution2d_forward_kernel", [&] {
            involution2d_forward_kernel<scalar_t><<<num_blocks, threads_per_block, 0, stream>>>(
                input.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int64_t>(),
                weight_.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, int64_t>(),
                output.data_ptr<scalar_t>(),
                num_elements,
                channels,
                groups,
                in_height, in_width,
                out_height, out_width,
                kernel_size[0], kernel_size[1],
                padding[0], padding[1],
                stride[0], stride[1],
                dilation[0], dilation[1]
            );
        }
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace cuda
} // namespace involution
