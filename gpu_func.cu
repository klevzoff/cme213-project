#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_result));
    return result;
}

// all matrices assumed column major
template<typename T>
__global__ void simple_gemm_kernel(T const * __restrict__ A,
                                   T const * __restrict__ B,
                                   T * __restrict__ C,
                                   T const alpha,
                                   T const beta,
                                   int M, int N, int K)
{
    int const offset = blockDim.x * blockIdx.x + threadIdx.x;
    
    int const row = offset % M;
    int const col = offset / M;
    
    if (col < N)
    {
        T res = beta * C[col * M + row];
        for (int k = 0; k < K; ++k)
        {
            res += alpha * A[k * M + row] * B[col * K + k];
        }
        C[col * M + row] = res;
    }
}

template<typename T>
void simple_gemm_wrapper(T const * A, T const * B, T * C,
                         T const alpha, T const beta,
                         int M, int N, int K) 
{
    int const threads = 192;
    int const blocks = (M * N + threads - 1) / threads;
    
    simple_gemm_kernel<<<blocks, threads>>>(A, B, C, alpha, beta, M, N, K);
    
    check_launch("simple_gemm");
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) 
{
    /* TODO: Write an efficient GEMM implementation on GPU */
    simple_gemm_wrapper(A, B, C, *alpha, *beta, M, N, K);

    return 0;
}

namespace un_ops
{
    
struct exponent
{
    template<typename T>
    static inline __device__ T apply(T const & x) 
    { 
        return exp(x);
    }
};
    
struct sigmoid
{
    template<typename T>
    static inline __device__ T apply(T const & x) 
    { 
        return T(1) / (T(1) + exp(-x));
    }
};
    
struct x_times_1_minus_x // does this have a special name?
{
    template<typename T>
    static inline __device__ T apply(T const & x) 
    { 
        return x * (1 - x);
    }
};
    
}

namespace bin_ops
{

struct add 
{
    template<typename T>
    static inline __device__ T apply(T const & x0, T const & x1)
    {
        return x0 + x1;
    }
    
    template<typename T>
    static inline __device__ void atomic(T & x0, T const & x1)
    {
        atomicAdd(&x0, x1);
    }
};
    
struct mult 
{
    template<typename T>
    static inline __device__ T apply(T const & x0, T const & x1)
    {
        return x0 * x1;
    }
    
    // no atomic version in CUDA
};
 
struct greater_of
{
    template<typename T>
    static inline __device__ T apply(T const & x0, T const & x1)
    {
        return max(x0, x1);
    }
    
    template<typename T>
    static inline __device__ void atomic(T & x0, T const & x1)
    {
        atomicMax(&x0, x1); // only supported for unsigned int and long long 
    }
};
    
}

template<int warp_size, int num_warps, typename T>
__global__ void transpose_kernel(T const * __restrict__ src,  T * __restrict__ dst, int M, int N)
{
    // transpose a square block of size warp_size
    T __shared__ sdata[warp_size][warp_size+1];
    
    {
        int const lr = threadIdx.x;
        int const gr = blockIdx.x * warp_size + lr;

        if (gr < M)
        {
            int lc = threadIdx.y;
            int gc = blockIdx.y * warp_size + lc;

            for (int pass = 0; pass < warp_size / num_warps; ++pass, gc += num_warps, lc += num_warps)
            {
                if (gc < N)
                {
                    sdata[lc][lr] = src[gc * M + gr];
                }
            }
        }
    }

    __syncthreads();

    {
        int const lc = threadIdx.x;
        int const gc = blockIdx.y * warp_size + lc;
        
        if (gc < N)
        {
            int lr = threadIdx.y;
            int gr = blockIdx.x * warp_size + lr;

            for (int pass = 0; pass < warp_size / num_warps; ++pass, gr += num_warps, lr += num_warps)
            {
                if (gr < M)
                {
                    dst[gr * N + gc] = sdata[lc][lr];
                }
            }
        }
    }
}

template<typename T>
void transpose_wrapper(T const * src, T * dst, int M, int N)
{
    int const warp_size = 32;
    int const num_warps = 256 / 32;
    
    dim3 const threads(warp_size, num_warps);
    dim3 const blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    transpose_kernel<warp_size, num_warps><<<blocks, threads>>>(src, dst, M, N);
    
    check_launch("transpose");
}

template<int block_size, typename OP, typename T>
__global__ void block_reduce_kernel(T const * __restrict__ data, T * __restrict__ res, int M, int N)
{
    int const tid = threadIdx.x;
    int const col_offset = blockIdx.y * M;
    int const row_offset = 2 * block_size * blockIdx.x + threadIdx.x;
    
    T __shared__ sdata[block_size];
    
    sdata[tid] = (row_offset < M) ? data[col_offset + row_offset] : T(0);
    
    if (row_offset + block_size < M)
    {
        sdata[tid] = OP::template apply(sdata[tid], data[col_offset + row_offset + block_size]);
    }
    
    __syncthreads();
    
    for (int i = block_size/2; i > 0; i >>= 1)
    {
        if (tid < i)
        {
            sdata[tid] = OP::template apply(sdata[tid], sdata[tid + i]);
        }
        
        __syncthreads();
    }
    
    if (tid == 0)
    {
        res[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0];
    }
}

template<typename OP, typename T>
void reduce_wrapper(T const * data, T * res, int M, int N)
{
    int const threads = 256;
    int const nblocks = (M + 2 * threads - 1) / (2 * threads);
    dim3 const blocks(nblocks, N);
    
    T * block_res;
    cudaMalloc((void**) &block_res, nblocks * N * sizeof(T));
    
    block_reduce_kernel<threads, OP><<<blocks, threads>>>(data, block_res, M, N);
    
    check_launch("reduce");
    
    if (nblocks > 1)
    {
        reduce_wrapper<OP>(block_res, res, nblocks, N);
    }
    else
    {
        cudaMemcpy(res, block_res, N * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(block_res);
}



template<typename T>
__global__ void coldiv_kernel(T * __restrict__ data, T const * __restrict__ divs, int M, int N)
{
    int const offset = blockDim.x * blockIdx.x + threadIdx.x;
    
    int const row = offset % M;
    int const col = offset / M;
    
    if (col < N)
    {
        data[col * M + row] /= divs[col];
    }
}

template<typename T>
void coldiv_wrapper(T * data, T const * divs, int M, int N)
{
    int const threads = 192;
    int const blocks = (M * N + threads - 1) / threads;
    
    coldiv_kernel<<<blocks, threads>>>(data, divs, M, N);
    
    check_launch("coldiv");
}


template<typename OP, typename T>
__global__ void apply_kernel(T const * src, T * dst, int N)
{
    int const offset = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (offset < N)
    {
        dst[offset] = OP::template apply(src[offset]);
    }
}

template<typename OP, typename T>
void apply_wrapper(T const * src, T * dst, int N)
{
    int const threads = 192;
    int const blocks = (N + threads - 1) / threads;
    
    apply_kernel<OP><<<blocks, threads>>>(src, dst, N);
    
    check_launch("apply");
}

template<typename OP, typename T>
__global__ void combine_kernel(T const * src1, T const * src2, T * dst, int N)
{
    int const offset = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (offset < N)
    {
        dst[offset] = OP::template apply(src1[offset], src2[offset]);
    }
}

template<typename OP, typename T>
void combine_wrapper(T const * src1, T const * src2, T * dst, int N)
{
    int const threads = 192;
    int const blocks = (N + threads - 1) / threads;
    
    combine_kernel<OP><<<blocks, threads>>>(src1, src2, dst, N);
    
    check_launch("combine");
}


template<typename T>
__global__ void axpby_kernel(T const a, T const * X, T const b, T const * Y, T * Z, int N)
{
    int const offset = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (offset < N)
    {
        Z[offset] = a * X[offset] + b * Y[offset];
    }
}

template<typename T>
void axpby_wrapper(T a, T const * X, T b, T const * Y, T * Z, int N)
{
    int const threads = 192;
    int const blocks = (N + threads - 1) / threads;
    
    axpby_kernel<<<blocks, threads>>>(a, X, b, Y, Z, N);
    
    check_launch("axpby");
}

/**
 * Invoke template instantiations for required parameter types
 */

#define INST_WRAPPER_TEMPLATES(T) \
template void simple_gemm_wrapper<T>(T const * A, T const * B, T * C, T const alpha, T const beta, int M, int N, int K); \
template void transpose_wrapper<T>(T const * src, T * dst, int M, int N); \
template void reduce_wrapper<bin_ops::add,T>(T const * data, T * res, int M, int N); \
template void reduce_wrapper<bin_ops::greater_of,T>(T const * data, T * res, int M, int N); \
template void coldiv_wrapper<T>(T * data, T const * divs, int M, int N); \
template void apply_wrapper<un_ops::exponent,T>(T const * src, T * dst, int N); \
template void apply_wrapper<un_ops::sigmoid,T>(T const * src, T * dst, int N); \
template void apply_wrapper<un_ops::x_times_1_minus_x,T>(T const * src, T * dst, int N); \
template void combine_wrapper<bin_ops::mult,T>(T const * src1, T const * src2, T * dst, int N); \
template void axpby_wrapper<T>(T a, T const * X, T b, T const * Y, T * Z, int N);

INST_WRAPPER_TEMPLATES(double)
    
#undef INST_WRAPPER_TEMPLATES
