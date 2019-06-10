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

/* ================================ operators =============================== */

namespace un_ops
{

struct identity
{
    template<typename T>
    static __forceinline__ __device__ T apply(T const & x) 
    { 
        return x;
    }
};
    
struct exponent
{
    template<typename T>
    static __forceinline__ __device__ T apply(T const & x) 
    { 
        return exp(x);
    }
};
    
struct sigmoid
{
    template<typename T>
    static __forceinline__ __device__ T apply(T const & x) 
    { 
        return T(1) / (T(1) + exp(-x));
    }
};
    
struct x_times_1_minus_x // does this have a special name?
{
    template<typename T>
    static __forceinline__ __device__ T apply(T const & x) 
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
    static __forceinline__ __device__ T apply(T const & x0, T const & x1)
    {
        return x0 + x1;
    }
    
    template<typename T>
    static __forceinline__ __device__ void atomic(T & x0, T const & x1)
    {
        atomicAdd(&x0, x1);
    }
};
    
struct mult 
{
    template<typename T>
    static __forceinline__ __device__ T apply(T const & x0, T const & x1)
    {
        return x0 * x1;
    }
    
    // no atomic version in CUDA
};
 
struct greater_of
{
    template<typename T>
    static __forceinline__ __device__ T apply(T const & x0, T const & x1)
    {
        return max(x0, x1);
    }
    
    template<typename T>
    static __forceinline__ __device__ void atomic(T & x0, T const & x1)
    {
        atomicMax(&x0, x1); // only supported for unsigned int and long long 
    }
};
    
}

/* =============================== simple GEMM ============================== */

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
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
        T res = beta * OP_C::template apply(C[col * M + row]);
        for (int k = 0; k < K; ++k)
        {
            res += alpha * OP_A::template apply(A[k * M + row]) * OP_B::template apply(B[col * K + k]);
        }
        C[col * M + row] = OP_R::template apply(res);
    }
}

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void simple_gemm_wrapper(T const * __restrict__ A,
                         T const * __restrict__ B,
                         T * __restrict__ C,
                         T const alpha, T const beta,
                         int M, int N, int K) 
{
    int const threads = 192;
    int const blocks = (M * N + threads - 1) / threads;
    
    simple_gemm_kernel<OP_A, OP_B, OP_C, OP_R><<<blocks, threads>>>(A, B, C, alpha, beta, M, N, K);
    
    check_launch("simple_gemm");
}

// gemmpv = general matrix-matrix plus vector
// i.e. C = alpha * A * B + beta * repmat(d, 1, N)
template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
__global__ void simple_gemmpv_kernel(T const * __restrict__ A,
                                     T const * __restrict__ B,
                                     T const * __restrict__ d,
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
        T res = beta * OP_C::template apply(d[row]);
        for (int k = 0; k < K; ++k)
        {
            res += alpha * OP_A::template apply(A[k * M + row]) * OP_B::template apply(B[col * K + k]);
        }
        C[col * M + row] = OP_R::template apply(res);
    }
}

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void simple_gemmpv_wrapper(T const * __restrict__ A,
                           T const * __restrict__ B,
                           T const * __restrict__ d,
                           T * __restrict__ C,
                           T const alpha, T const beta,
                           int M, int N, int K) 
{
    int const threads = 192;
    int const blocks = (M * N + threads - 1) / threads;
    
    simple_gemmpv_kernel<OP_A,OP_B,OP_C,OP_R><<<blocks, threads>>>(A, B, d, C, alpha, beta, M, N, K);
    
    check_launch("simple_gemmpv");
}

/* ============================== shared GEMM 1 ============================= */

template<int warp_size, int num_warps, typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
__global__ void shared_gemm_kernel(T const * __restrict__ A,
                                   T const * __restrict__ B,
                                   T * __restrict__ C,
                                   T const alpha,
                                   T const beta,
                                   int M, int N, int K)
{
    T __shared__ sA[warp_size][warp_size];
    T __shared__ sB[warp_size][warp_size];
    
    int const num_passes = warp_size / num_warps;
    T lC[num_passes];
    
    int const local_row = threadIdx.x;
    int const row = warp_size * blockIdx.x + local_row;
    
    // get enough tiles to cover the width of A / height of B
    int const num_tiles = (K + warp_size - 1) / warp_size; 
    
    // store previous C values updated by this thread
    for (int pass = 0; pass < num_passes; ++pass)
    {
        int const local_col = pass * num_warps + threadIdx.y;
        int const col = warp_size * blockIdx.y + local_col;
        if (row < M && col < N)
        {
            lC[pass] = beta * OP_C::template apply(C[col * M + row]);
        }
    }
    
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        int const tile_offset = tile * warp_size;
        int const num_k = min(warp_size, K - tile_offset);
        
        for (int pass = 0; pass < num_passes; ++pass)
        {
            int const local_col = pass * num_warps + threadIdx.y;
            int const col = warp_size * blockIdx.y + local_col;

            int const inner_row = tile_offset + local_row;
            int const inner_col = tile_offset + local_col;
            
            if (row < M && inner_col < K)
            {
                sA[local_col][local_row] = OP_A::template apply(A[inner_col * M + row]);
            }
            if (col < N && inner_row < K)
            {
                sB[local_row][local_col] = OP_B::template apply(B[col * K + inner_row]);
            }
        }

        __syncthreads();

        for (int pass = 0; pass < num_passes; ++pass)
        {
            int const local_col = pass * num_warps + threadIdx.y;
            int const col = warp_size * blockIdx.y + local_col;

            if (row < M && col < N)
            {               
                for (int k = 0; k < num_k; ++k)
                {
                    lC[pass] += alpha * sA[k][local_row] * sB[k][local_col];
                }
            }       
        }
        
        __syncthreads();
    }
    
    // write C values updated by this thread
    for (int pass = 0; pass < num_passes; ++pass)
    {
        int const local_col = pass * num_warps + threadIdx.y;
        int const col = warp_size * blockIdx.y + local_col;
        if (row < M && col < N)
        {
            C[col * M + row] = OP_R::template apply(lC[pass]);
        }
    }
}

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void shared_gemm_wrapper(T const * __restrict__ A,
                         T const * __restrict__ B,
                         T * __restrict__ C,
                         T const alpha, T const beta,
                         int M, int N, int K) 
{
    int const warp_size = 32;
    int const num_warps = 1024 / warp_size;
    
    dim3 const threads(warp_size, num_warps);
    dim3 const blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    shared_gemm_kernel<warp_size,num_warps,OP_A,OP_B,OP_C,OP_R><<<blocks, threads>>>(A, B, C, alpha, beta, M, N, K);
    
    check_launch("shared_gemm");
}

template<int warp_size, int num_warps, typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
__global__ void shared_gemmpv_kernel(T const * __restrict__ A,
                                     T const * __restrict__ B,
                                     T const * __restrict__ d,
                                     T * __restrict__ C,
                                     T const alpha,
                                     T const beta,
                                     int M, int N, int K)
{
    T __shared__ sA[warp_size][warp_size];
    T __shared__ sB[warp_size][warp_size];
    
    int const num_passes = warp_size / num_warps;
    T lC[num_passes];
    
    int const local_row = threadIdx.x;
    int const row = warp_size * blockIdx.x + local_row;
    
    // get enough tiles to cover the width of A / height of B
    int const num_tiles = (K + warp_size - 1) / warp_size; 
    
    // store previous C values updated by this thread
    for (int pass = 0; pass < num_passes; ++pass)
    {
        int const local_col = pass * num_warps + threadIdx.y;
        int const col = warp_size * blockIdx.y + local_col;
        if (row < M && col < N)
        {
            lC[pass] = beta * OP_C::template apply(d[row]);
        }
    }
    
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        int const tile_offset = tile * warp_size;
        int const num_k = min(warp_size, K - tile_offset);
        
        for (int pass = 0; pass < num_passes; ++pass)
        {
            int const local_col = pass * num_warps + threadIdx.y;
            int const col = warp_size * blockIdx.y + local_col;

            int const inner_row = tile_offset + local_row;
            int const inner_col = tile_offset + local_col;
            
            if (row < M && inner_col < K)
            {
                sA[local_col][local_row] = OP_A::template apply(A[inner_col * M + row]);
            }
            if (col < N && inner_row < K)
            {
                sB[local_row][local_col] = OP_B::template apply(B[col * K + inner_row]);
            }
        }

        __syncthreads();

        for (int pass = 0; pass < num_passes; ++pass)
        {
            int const local_col = pass * num_warps + threadIdx.y;
            int const col = warp_size * blockIdx.y + local_col;

            if (row < M && col < N)
            {               
                for (int k = 0; k < num_k; ++k)
                {
                    lC[pass] += alpha * sA[k][local_row] * sB[k][local_col];
                }
            }       
        }
        
        __syncthreads();
    }
    
    // write C values updated by this thread
    for (int pass = 0; pass < num_passes; ++pass)
    {
        int const local_col = pass * num_warps + threadIdx.y;
        int const col = warp_size * blockIdx.y + local_col;
        if (row < M && col < N)
        {
            C[col * M + row] = OP_R::template apply(lC[pass]);
        }
    }
}

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void shared_gemmpv_wrapper(T const * __restrict__ A,
                           T const * __restrict__ B,
                           T const * __restrict__ d,
                           T * __restrict__ C,
                           T const alpha, T const beta,
                           int M, int N, int K) 
{
    int const warp_size = 32;
    int const num_warps = 1024 / warp_size;
    
    dim3 const threads(warp_size, num_warps);
    dim3 const blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    shared_gemmpv_kernel<warp_size,num_warps,OP_A,OP_B,OP_C,OP_R><<<blocks, threads>>>(A, B, d, C, alpha, beta, M, N, K);
    
    check_launch("shared_gemmpv");
}

/* ============================== shared GEMM 2 ============================= */

template<int Mtile, int Ktile, typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
__global__ void shared2_gemm_kernel(T const * __restrict__ A,
                                    T const * __restrict__ B,
                                    T * __restrict__ C,
                                    T const alpha,
                                    T const beta,
                                    int M, int N, int K)
{
    int const Ntile = Mtile / Ktile;
    
    T __shared__ sB[Ntile][Ktile];
    T lA[Ktile];
    T lC[Ntile];
    
    for (int lc = 0; lc < Ntile; ++lc)
    {
        lC[lc] = T(0);
    }
    
    int const num_tiles = (K + Ktile - 1) / Ktile;
    
    int const row = Mtile * blockIdx.y + Ktile * threadIdx.y + threadIdx.x;
    int const col_offset = Ntile * blockIdx.x;
    
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        int const tile_offset = tile * Ktile;
        
        if ((col_offset + threadIdx.y) < N && (tile_offset + threadIdx.x) < K)
        {
            sB[threadIdx.y][threadIdx.x] = OP_B::template apply(B[(col_offset + threadIdx.y) * K + tile_offset + threadIdx.x]);
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = T(0);
        }
        
        __syncthreads(); 

        if (row < M)
        {
            #pragma unroll
            for (int k = 0; k < Ktile; ++k)
            {
                if (tile_offset + k < K)
                {
                    lA[k] = OP_A::template apply(A[(tile_offset + k) * M + row]);
                }
                else
                {
                    lA[k] = T(0);
                }
            }

            for (int lc = 0; lc < Ntile; ++lc)
            {
                if (col_offset + lc < N)
                {
                    #pragma unroll
                    for (int k = 0; k < Ktile; ++k)
                    {
                        lC[lc] += lA[k] * sB[lc][k];
                    }
                }
            }
        }
        
        __syncthreads();   
    }
    
    if (row < M)
    {
        for (int lc = 0; lc < Ntile; ++lc)
        {
            if (col_offset + lc < N)
            {
                C[(col_offset + lc) * M + row] = OP_R::template apply(beta * OP_C::template apply(C[(col_offset + lc) * M + row]) + alpha * lC[lc]);
            }
        }
    }
}

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void shared2_gemm_wrapper(T const * __restrict__ A,
                          T const * __restrict__ B,
                          T * __restrict__ C,
                          T const alpha, T const beta,
                          int M, int N, int K) 
{
    if (M < 512 || N < 128)
    {
        int const Mtile = 64;
        int const Ktile = 4;
        int const Ntile = Mtile / Ktile;

        dim3 const threads(Ktile, Ntile);
        dim3 const blocks((N + Ntile - 1) / Ntile, (M + Mtile - 1) / Mtile);

        shared2_gemm_kernel<Mtile,Ktile,OP_A,OP_B,OP_C,OP_R><<<blocks, threads>>>(A, B, C, alpha, beta, M, N, K);
    }
    else // this gives slightly better (5-10%) results on large enough matrices
    {
        int const Mtile = 128;
        int const Ktile = 8;
        int const Ntile = Mtile / Ktile;

        dim3 const threads(Ktile, Ntile);
        dim3 const blocks((N + Ntile - 1) / Ntile, (M + Mtile - 1) / Mtile);

        shared2_gemm_kernel<Mtile,Ktile,OP_A,OP_B,OP_C,OP_R><<<blocks, threads>>>(A, B, C, alpha, beta, M, N, K);
    }
    
    check_launch("shared2_gemm");
}

template<int Mtile, int Ktile, typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
__global__ void shared2_gemmpv_kernel(T const * __restrict__ A,
                                      T const * __restrict__ B,
                                      T const * __restrict__ d,
                                      T * __restrict__ C,
                                      T const alpha,
                                      T const beta,
                                      int M, int N, int K)
{
    int const Ntile = Mtile / Ktile;
    
    T __shared__ sB[Ntile][Ktile];
    T lA[Ktile];
    T lC[Ntile];
    
    for (int lc = 0; lc < Ntile; ++lc)
    {
        lC[lc] = T(0);
    }
    
    int const num_tiles = (K + Ktile - 1) / Ktile;
    
    int const row = Mtile * blockIdx.y + Ktile * threadIdx.y + threadIdx.x;
    int const col_offset = Ntile * blockIdx.x;
    
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        int const tile_offset = tile * Ktile;
        
        if ((col_offset + threadIdx.y) < N && (tile_offset + threadIdx.x) < K)
        {
            sB[threadIdx.y][threadIdx.x] = OP_B::template apply(B[(col_offset + threadIdx.y) * K + tile_offset + threadIdx.x]);
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = T(0);
        }
        
        __syncthreads(); 

        if (row < M)
        {
            #pragma unroll
            for (int k = 0; k < Ktile; ++k)
            {
                if (tile_offset + k < K)
                {
                    lA[k] = OP_A::template apply(A[(tile_offset + k) * M + row]);
                }
                else
                {
                    lA[k] = T(0);
                }
            }

            for (int lc = 0; lc < Ntile; ++lc)
            {
                if (col_offset + lc < N)
                {
                    #pragma unroll
                    for (int k = 0; k < Ktile; ++k)
                    {
                        lC[lc] += lA[k] * sB[lc][k];
                    }
                }
            }
        }
        
        __syncthreads();   
    }
    
    if (row < M)
    {
        for (int lc = 0; lc < Ntile; ++lc)
        {
            if (col_offset + lc < N)
            {
                C[(col_offset + lc) * M + row] = OP_R::template apply(beta * OP_C::template apply(d[row]) + alpha * lC[lc]);
            }
        }
    }
}

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void shared2_gemmpv_wrapper(T const * __restrict__ A, 
                            T const * __restrict__ B, 
                            T const * __restrict__ d, 
                            T * __restrict__ C,
                            T const alpha, T const beta,
                            int M, int N, int K) 
{    
    if (M < 512 || N < 128)
    {
        int const Mtile = 64;
        int const Ktile = 4;
        int const Ntile = Mtile / Ktile;

        dim3 const threads(Ktile, Ntile);
        dim3 const blocks((N + Ntile - 1) / Ntile, (M + Mtile - 1) / Mtile);

        shared2_gemmpv_kernel<Mtile,Ktile,OP_A,OP_B,OP_C,OP_R><<<blocks, threads>>>(A, B, d, C, alpha, beta, M, N, K);
    }
    else // this gives slightly better (5-10%) results on large enough matrices
    {
        int const Mtile = 128;
        int const Ktile = 8;
        int const Ntile = Mtile / Ktile;

        dim3 const threads(Ktile, Ntile);
        dim3 const blocks((N + Ntile - 1) / Ntile, (M + Mtile - 1) / Mtile);

        shared2_gemmpv_kernel<Mtile,Ktile,OP_A,OP_B,OP_C,OP_R><<<blocks, threads>>>(A, B, d, C, alpha, beta, M, N, K);   
    }
    
    check_launch("shared2_gemmpv");
}

/* ================================= myGEMM ================================ */

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double const * __restrict__ A, 
           double const * __restrict__ B,
           double * __restrict__ C, 
           double * alpha, double * beta,
           int M, int N, int K) 
{
    typedef un_ops::identity eye; // shortcut
    
    //simple_gemm_wrapper<eye,eye,eye,eye>(A, B, C, *alpha, *beta, M, N, K);
    //shared_gemm_wrapper<eye,eye,eye,eye>(A, B, C, *alpha, *beta, M, N, K);
    shared2_gemm_wrapper<eye,eye,eye,eye>(A, B, C, *alpha, *beta, M, N, K);

    return 0;
}

/* ============================== transpose ============================= */

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
    int const num_warps = 256 / warp_size;
    
    dim3 const threads(warp_size, num_warps);
    dim3 const blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    transpose_kernel<warp_size, num_warps><<<blocks, threads>>>(src, dst, M, N);
    
    check_launch("transpose");
}

/* ============================== reductions ============================= */

template<int block_size, typename OP, typename T>
__global__ void colreduce_kernel(T const * __restrict__ data, T * __restrict__ res, int M, int N)
{
    int const tid = threadIdx.x;
    int const col = blockIdx.y;
    int const row = 2 * block_size * blockIdx.x + threadIdx.x;
    
    T __shared__ sdata[block_size];
    
    sdata[tid] = (row < M) ? data[col * M + row] : T(0);
    
    if (row + block_size < M)
    {
        sdata[tid] = OP::template apply(sdata[tid], data[col * M + row + block_size]);
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
        res[col * gridDim.x + blockIdx.x] = sdata[0];
    }
}

template<typename OP, typename T>
void colreduce_wrapper(T const * data, T * res, int M, int N)
{
    int const threads = 256;
    int const nblocks = (M + 2 * threads - 1) / (2 * threads);
    dim3 const blocks(nblocks, N);
    
    T * block_res;
    cudaMalloc((void**) &block_res, nblocks * N * sizeof(T));
    
    colreduce_kernel<threads, OP><<<blocks, threads>>>(data, block_res, M, N);
    
    check_launch("colreduce");
    
    if (nblocks > 1)
    {
        colreduce_wrapper<OP>(block_res, res, nblocks, N);
    }
    else
    {
        cudaMemcpy(res, block_res, N * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(block_res);
}

template<int block_size, typename OP, typename T>
__global__ void rowreduce_kernel(T const * __restrict__ data, T * __restrict__ res, int M, int N)
{
    int const tid = threadIdx.x;
    int const col = 2 * block_size * blockIdx.x + threadIdx.x;
    int const row = blockIdx.y;
    
    T __shared__ sdata[block_size];
    
    sdata[tid] = (col < N) ? data[col * M + row] : T(0);
    
    if (col + block_size < N)
    {
        sdata[tid] = OP::template apply(sdata[tid], data[(col + block_size) * M + row]);
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
        res[blockIdx.x * M + row] = sdata[0];
    }
}

template<typename OP, typename T>
void rowreduce_wrapper(T const * data, T * res, int M, int N)
{
    int const threads = 256;
    int const nblocks = (N + 2 * threads - 1) / (2 * threads);
    dim3 const blocks(nblocks, M);
    
    T * block_res;
    cudaMalloc((void**) &block_res, nblocks * M * sizeof(T));
    
    rowreduce_kernel<threads, OP><<<blocks, threads>>>(data, block_res, M, N);
    
    check_launch("rowreduce");
    
    if (nblocks > 1)
    {
        rowreduce_wrapper<OP>(block_res, res, M, nblocks);
    }
    else
    {
        cudaMemcpy(res, block_res, M * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(block_res);
}

/* ============================== other kernels ============================= */

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
template void shared2_gemm_wrapper<un_ops::identity,un_ops::identity,un_ops::identity,un_ops::identity,T>(T const * A, T const * B, T * C, T const alpha, T const beta, int M, int N, int K); \
template void shared2_gemmpv_wrapper<un_ops::identity,un_ops::identity,un_ops::identity,un_ops::identity,T>(T const * A, T const * B, T const * d, T * C, T const alpha, T const beta, int M, int N, int K); \
template void shared2_gemmpv_wrapper<un_ops::identity,un_ops::identity,un_ops::identity,un_ops::sigmoid,T>(T const * A, T const * B, T const * d, T * C, T const alpha, T const beta, int M, int N, int K); \
template void shared2_gemmpv_wrapper<un_ops::identity,un_ops::identity,un_ops::identity,un_ops::exponent,T>(T const * A, T const * B, T const * d, T * C, T const alpha, T const beta, int M, int N, int K); \
template void transpose_wrapper<T>(T const * src, T * dst, int M, int N); \
template void colreduce_wrapper<bin_ops::add,T>(T const * data, T * res, int M, int N); \
template void rowreduce_wrapper<bin_ops::add,T>(T const * data, T * res, int M, int N); \
template void coldiv_wrapper<T>(T * data, T const * divs, int M, int N); \
template void apply_wrapper<un_ops::exponent,T>(T const * src, T * dst, int N); \
template void apply_wrapper<un_ops::sigmoid,T>(T const * src, T * dst, int N); \
template void apply_wrapper<un_ops::x_times_1_minus_x,T>(T const * src, T * dst, int N); \
template void combine_wrapper<bin_ops::mult,T>(T const * src1, T const * src2, T * dst, int N); \
template void axpby_wrapper<T>(T a, T const * X, T b, T const * Y, T * Z, int N);

INST_WRAPPER_TEMPLATES(double)
    
#undef INST_WRAPPER_TEMPLATES
