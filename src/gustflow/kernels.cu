__device__ __forceinline__ float wrap_phase_cuda(const float value) {
    constexpr float pi = 3.14159265358979323846f;
    constexpr float two_pi = 6.2831853071795864769f;
    return value - floorf((value + pi) / two_pi) * two_pi;
}

__device__ __forceinline__ float signed_l1(const float value) {
    return (value > 0.0f) - (value < 0.0f);
}

__device__ __forceinline__ float edge_residual(
    const float* __restrict__ phase,
    const float* __restrict__ wrapped,
    const long long current,
    const long long neighbor) {



    const float target = wrap_phase_cuda(
        wrapped[neighbor] - wrapped[current]);
    return wrap_phase_cuda(
        (phase[neighbor] - phase[current]) - target);
}




extern "C" __global__ void phase_l1_forward(
    const float* __restrict__ phase,
    const float* __restrict__ wrapped,
    const float* __restrict__ confidence,
    float* __restrict__ loss,
    const long long total,
    const int time_size,
    const int fe_size,
    const int pe_size,
    const int spe_size,
    const float wt,
    const float wfe,
    const float wpe,
    const float wspe) {
    const long long spe_stride = 1;
    const long long pe_stride = spe_size;
    const long long fe_stride = (long long)pe_size * spe_size;
    const long long time_stride = (long long)fe_size * pe_size * spe_size;
    const long long spatial_stride = time_stride;

    float local = 0.0f;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += (long long)blockDim.x * gridDim.x) {
        const long long within_channel = index % ((long long)time_size * spatial_stride);
        const int time = within_channel / time_stride;
        const long long within_frame = within_channel - (long long)time * time_stride;
        const int fe = within_frame / fe_stride;
        const long long within_fe = within_frame - (long long)fe * fe_stride;
        const int pe = within_fe / pe_stride;
        const int spe = within_fe - (long long)pe * pe_stride;
        const float voxel_weight = confidence[index];


        if (time_size > 1) {
            const long long next_time = (time + 1 < time_size)
                ? index + time_stride
                : index - (long long)(time_size - 1) * time_stride;
            local += wt * fabsf(edge_residual(
                phase, wrapped, index, next_time)) * voxel_weight;
        }
        if (fe + 1 < fe_size) {
            local += wfe * fabsf(edge_residual(
                phase, wrapped, index, index + fe_stride)) * voxel_weight;
        }
        if (pe + 1 < pe_size) {
            local += wpe * fabsf(edge_residual(
                phase, wrapped, index, index + pe_stride)) * voxel_weight;
        }
        if (spe + 1 < spe_size) {
            local += wspe * fabsf(edge_residual(
                phase, wrapped, index, index + spe_stride)) * voxel_weight;
        }
    }

    extern __shared__ float reduction[];
    reduction[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(loss, reduction[0]);
    }
}




extern "C" __global__ void phase_l1_backward(
    const float* __restrict__ phase,
    const float* __restrict__ wrapped,
    const float* __restrict__ confidence,
    const float* __restrict__ grad_loss,
    float* __restrict__ grad_phase,
    const long long total,
    const int time_size,
    const int fe_size,
    const int pe_size,
    const int spe_size,
    const float wt,
    const float wfe,
    const float wpe,
    const float wspe) {
    const long long pe_stride = spe_size;
    const long long fe_stride = (long long)pe_size * spe_size;
    const long long time_stride = (long long)fe_size * pe_size * spe_size;
    const long long spatial_stride = time_stride;

    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += (long long)blockDim.x * gridDim.x) {
        const long long within_channel = index % ((long long)time_size * spatial_stride);
        const int time = within_channel / time_stride;
        const long long within_frame = within_channel - (long long)time * time_stride;
        const int fe = within_frame / fe_stride;
        const long long within_fe = within_frame - (long long)fe * fe_stride;
        const int pe = within_fe / pe_stride;
        const int spe = within_fe - (long long)pe * pe_stride;
        float gradient = 0.0f;



        if (time_size > 1) {
            const long long next_time = (time + 1 < time_size)
                ? index + time_stride
                : index - (long long)(time_size - 1) * time_stride;
            gradient -= wt * confidence[index] * signed_l1(edge_residual(
                phase, wrapped, index, next_time));
        }
        if (fe + 1 < fe_size) {
            gradient -= wfe * confidence[index] * signed_l1(edge_residual(
                phase, wrapped, index, index + fe_stride));
        }
        if (pe + 1 < pe_size) {
            gradient -= wpe * confidence[index] * signed_l1(edge_residual(
                phase, wrapped, index, index + pe_stride));
        }
        if (spe + 1 < spe_size) {
            gradient -= wspe * confidence[index] * signed_l1(edge_residual(
                phase, wrapped, index, index + 1));
        }



        if (time_size > 1) {
            const long long previous = (time > 0)
                ? index - time_stride
                : index + (long long)(time_size - 1) * time_stride;
            gradient += wt * confidence[previous] * signed_l1(edge_residual(
                phase, wrapped, previous, index));
        }
        if (fe > 0) {
            const long long previous = index - fe_stride;
            gradient += wfe * confidence[previous] * signed_l1(edge_residual(
                phase, wrapped, previous, index));
        }
        if (pe > 0) {
            const long long previous = index - pe_stride;
            gradient += wpe * confidence[previous] * signed_l1(edge_residual(
                phase, wrapped, previous, index));
        }
        if (spe > 0) {
            const long long previous = index - 1;
            gradient += wspe * confidence[previous] * signed_l1(edge_residual(
                phase, wrapped, previous, index));
        }
        grad_phase[index] = gradient * grad_loss[0];
    }
}
