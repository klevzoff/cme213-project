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
__global__ void naive_gemm_kernel(T const * __restrict__ A,
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
void naive_gemm(T const * A, T const * B, T * C,
                T const alpha, T const beta,
                int M, int N, int K) 
{
    int constexpr threads = 192;
    int const blocks = (M * N + threads - 1) / threads;
    
    naive_gemm_kernel<<<blocks, threads>>>(A, B, C, alpha, beta, M, N, K);
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) 
{
    /* TODO: Write an efficient GEMM implementation on GPU */
    naive_gemm(A, B, C, *alpha, *beta, M, N, K);

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
};
 
struct max 
{
    template<typename T>
    static inline __device__ T apply(T const & x0, T const & x1)
    {
        return max(x0, x1); // TODO CUDA intrinsic?
    }
    
    template<typename T>
    static inline __device__ void atomic(T & x0, T const & x1)
    {
        atomicMax(&x0, x1); // only supported for unsigned int and long long 
    }
};
    
}

template<int warp_size, int num_warps, typename T>
__global__ void transpose2D_kernel(T const * __restrict__ src,
                                   T * __restrict__ dst,
                                   int M, int N)
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
void transpose2D(T const * src, T * dst, int M, int N)
{
    int constexpr warp_size = 32;
    int constexpr num_warps = 256 / 32;
    
    dim3 const threads = { warp_size, num_warps };
    dim3 const blocks  = { (M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y };
    
    transpose2D_kernel<warp_size, num_warps><<<blocks, threads>>>(src, dst, M, N);
}

template<typename T>
void transpose(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    assert(dst.nrow() == src.ncol() && dst.ncol() == src.nrow());
    transpose2D(src.data(), dst.data(), src.nrow(), src.ncol());
}

template<typename T>
DeviceMat<T> transpose(DeviceMat<T> const & src)
{
    DeviceMat<T> dst(src.ncol(), src.nrow());
    transpose(src, dst);
    return dst;
}

template<int block_size, typename OP, typename T>
__global__ void block_reduce_kernel(T const * __restrict__ data, T * __restrict__ res, int N)
{
    int const tid    = threadIdx.x;
    int const offset = 2 * block_size * blockIdx.x + threadIdx.x;
    
    T __shared__ sdata[block_size];
    
    sdata[tid] = (offset < N) ? data[offset] : T(0);
    
    if (offset + block_size < N)
    {
        sdata[tid] = OP::template apply(sdata[tid], data[offset + block_size]);
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
        res[blockIdx.x] = sdata[0];
    }
}

template<typename OP, typename T>
void block_reduce(T const * data, T * res, int N)
{
    int constexpr threads = 256;
    int const blocks = (N + 2 * threads - 1) / (2 * threads);
    
    T * block_res;
    cudaMalloc((void**) &block_res, blocks * sizeof(T));
    
    block_reduce_kernel<threads, OP><<<blocks, threads>>>(data, block_res, N);
    
    if (blocks > 1)
    {
        block_reduce<OP>(block_res, res, blocks);
    }
    else
    {
        cudaMemcpy(res, block_res, sizeof(T), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(block_res);
}

template<typename OP, typename T>
void colreduce(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    assert(dst.nrow() == 1 && dst.ncol() == src.ncol());
    
    // TODO run all column reductions in one kernel
    for (int j = 0; j < src.ncol(); ++j)
    {
        block_reduce<OP>(src.col(j), dst.col(j), src.nrow());
    }
}

template<typename OP, typename T>
DeviceMat<T> colreduce(DeviceMat<T> const & src)
{
    DeviceMat<T> dst(1, src.ncol());
    colreduce<OP>(src, dst);
    return dst;
}

template<typename OP, typename T>
void rowreduce(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    assert(dst.ncol() == 1 && dst.nrow() == src.nrow());
    
    // TODO proper row reduction without transpose
    DeviceMat<T> src_temp = transpose(src);
    DeviceMat<T> dst_temp(1, src_temp.ncol());
    colreduce<OP>(src_temp, dst_temp);
    dst = transpose(dst_temp);
}

template<typename OP, typename T>
DeviceMat<T> rowreduce(DeviceMat<T> const & src)
{
    DeviceMat<T> dst(src.nrow(), 1);
    rowreduce<OP>(src, dst);
    return dst;
}

template<typename T>
void colsum(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    colreduce<bin_ops::add>(src, dst);
}

template<typename T>
DeviceMat<T> colsum(DeviceMat<T> const & src)
{
    return colreduce<bin_ops::add>(src);
}

template<typename T>
void colmax(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    colreduce<bin_ops::max>(src, dst);
}

template<typename T>
DeviceMat<T> colmax(DeviceMat<T> const & src)
{
    return colreduce<bin_ops::max>(src);
}

template<typename T>
void rowsum(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    rowreduce<bin_ops::add>(src, dst);
}

template<typename T>
DeviceMat<T> rowsum(DeviceMat<T> const & src)
{
    return rowreduce<bin_ops::add>(src);
}

template<typename T>
void rowmax(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    rowreduce<bin_ops::max>(src, dst);
}

template<typename T>
DeviceMat<T> rowmax(DeviceMat<T> const & src)
{
    return rowreduce<bin_ops::max>(src);
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
void coldiv(DeviceMat<T> & mat, DeviceMat<T> const & divs)
{
    assert(divs.nrow() == 1 && divs.ncol() == mat.ncol());
    
    int constexpr threads = 192;
    int const blocks = (mat.size() + threads - 1) / threads;
    
    coldiv_kernel<<<blocks, threads>>>(mat.data(), divs.data(), mat.nrow(), mat.ncol());
}

template<typename T>
void normalize(DeviceMat<T> & mat)
{
    // 1-norms of each column (assuming non-negative elements)
    DeviceMat<T> norms(1, mat.ncol());
    
    colsum(mat, norms);
    coldiv(mat, norms);
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
void apply(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    assert(dst.nrow() == src.nrow() && dst.ncol() == src.ncol());
    
    int constexpr threads = 192;
    int const blocks = (src.size() + threads - 1) / threads;
    
    apply_kernel<OP><<<blocks, threads>>>(src.data(), dst.data(), src.size());
}

template<typename OP, typename T>
DeviceMat<T> apply(DeviceMat<T> const & src)
{
    DeviceMat<T> dst(src.nrow(), src.ncol());
    apply<OP>(src, dst);
    return dst;
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
void combine(DeviceMat<T> const & src1, DeviceMat<T> const & src2, DeviceMat<T> & dst)
{
    assert(dst.nrow() == src1.nrow() && dst.ncol() == src1.ncol() && src2.nrow() == src1.nrow() && src2.ncol() == src1.ncol());
    
    int constexpr threads = 192;
    int const blocks = (src1.size() + threads - 1) / threads;
    
    combine_kernel<OP><<<blocks, threads>>>(src1.data(), src2.data(), dst.data(), src1.size());
}

template<typename OP, typename T>
DeviceMat<T> combine(DeviceMat<T> const & src1, DeviceMat<T> const & src2)
{
    DeviceMat<T> dst(src1.nrow(), src1.ncol());
    combine<OP>(src1, src2, dst);
    return dst;
}

template<typename T>
void exp(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    apply<un_ops::exponent>(src, dst);
}

template<typename T>
DeviceMat<T> exp(DeviceMat<T> const & src)
{
    return apply<un_ops::exponent>(src);
}

template<typename T>
void sigmoid(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    apply<un_ops::sigmoid>(src, dst);
}

template<typename T>
DeviceMat<T> sigmoid(DeviceMat<T> const & src)
{
    return apply<un_ops::sigmoid>(src);
}

template<typename T>
void sigmoid_derivative(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    apply<un_ops::x_times_1_minus_x>(src, dst);
}

template<typename T>
DeviceMat<T> sigmoid_derivative(DeviceMat<T> const & src)
{
    return apply<un_ops::x_times_1_minus_x>(src);
}

template<typename T>
void softmax(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    exp(src, dst);
    normalize(dst);
}

template<typename T>
DeviceMat<T> softmax(DeviceMat<T> const & src)
{
    DeviceMat<T> dst(src.nrow(), src.ncol());
    softmax(src, dst);
    return dst;
}

template<typename T>
void gemm(T alpha, DeviceMat<T> const & A, DeviceMat<T> const & B, T beta, DeviceMat<T> & C)
{
    assert(C.nrow() == A.nrow() && C.ncol() == B.ncol());
    naive_gemm(A.data(), B.data(), C.data(), alpha, beta, A.nrow(), B.ncol(), A.ncol());
}

template<typename T>
DeviceMat<T> matmult(DeviceMat<T> const & A, DeviceMat<T> const & B)
{
    DeviceMat<T> C(A.nrow(), B.ncol());
    gemm(T(1), A, B, T(0), C);
    return C;
}

template<typename T>
void hadamard(DeviceMat<T> const & A, DeviceMat<T> const & B, DeviceMat<T> & C)
{
    combine<bin_ops::mult>(A, B, C);
}

template<typename T>
DeviceMat<T> hadamard(DeviceMat<T> const & A, DeviceMat<T> const & B)
{
    return combine<bin_ops::mult>(A, B);
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
void axpby(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y, DeviceMat<T> & Z)
{
    assert(Z.nrow() == X.nrow() && Z.ncol() == X.ncol() && X.nrow() == Y.nrow() && X.ncol() == Y.ncol());
    
    int constexpr threads = 192;
    int const blocks = (X.size() + threads - 1) / threads;
    
    axpby_kernel<<<blocks, threads>>>(a, X.data(), b, Y.data(), Z.data(), X.size());
}

template<typename T>
DeviceMat<T> axpby(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y)
{
    DeviceMat<T> Z(X.nrow(), X.ncol());
    axpby(a, X, b, Y, Z);
    return Z;
}

template<typename T>
DeviceMat<T> sum(DeviceMat<T> const & src, int dim)
{
    if (dim == 0)
    {
        return colsum(src);
    }
    else if (dim == 1)
    {
        return rowsum(src);
    }
    else
    {
        std::cerr << "repmat(): invalid dim: " << dim << std::endl;
        exit(1);
    }
}

#define INST_TMP_DMAT_FUNCS(T) \
template void gemm<T>(T alpha, DeviceMat<T> const & A, DeviceMat<T> const & B, T beta, DeviceMat<T> & C); \
template void hadamard<T>(DeviceMat<T> const & A, DeviceMat<T> const & B, DeviceMat<T> & C); \
template DeviceMat<T> matmult(DeviceMat<T> const & A, DeviceMat<T> const & B); \
template DeviceMat<T> sigmoid<T>(DeviceMat<T> const & A); \
template DeviceMat<T> sigmoid_derivative<T>(DeviceMat<T> const & A); \
template DeviceMat<T> softmax<T>(DeviceMat<T> const & A); \
template DeviceMat<T> transpose<T>(DeviceMat<T> const & A); \
template DeviceMat<T> axpby<T>(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y); \
template void axpby<T>(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y, DeviceMat<T> & Z); \
template DeviceMat<T> sum<T>(DeviceMat<T> const & A, int dim);

INST_TMP_DMAT_FUNCS(double)
    
#undef INST_TMP_DMAT_FUNCS