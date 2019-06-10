#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <iomanip>

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

int myGEMM(double const * A, double const * B, double * C, 
           double * alpha, double * beta, 
           int M, int N, int K);

/**
 * Forward declare operations for explicit template instantiation
 */

namespace un_ops
{
struct identity;
struct exponent;
struct sigmoid;
struct x_times_1_minus_x;
}

namespace bin_ops
{
struct add;  
struct mult;
struct greater_of;
}

/**
 * Kernel wrapper declarations
 */

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void simple_gemm_wrapper(T const * A, T const * B, T * C, T const alpha, T const beta, int M, int N, int K);

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void simple_gemmpv_wrapper(T const * A, T const * B, T const * d, T * C, T const alpha, T const beta, int M, int N, int K);

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void shared_gemm_wrapper(T const * A, T const * B, T * C, T const alpha, T const beta, int M, int N, int K);
    
template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void shared_gemmpv_wrapper(T const * A, T const * B, T const * d, T * C, T const alpha, T const beta, int M, int N, int K);

template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void shared2_gemm_wrapper(T const * A, T const * B, T * C, T const alpha, T const beta, int M, int N, int K);
    
template<typename OP_A, typename OP_B, typename OP_C, typename OP_R, typename T>
void shared2_gemmpv_wrapper(T const * A, T const * B, T const * d, T * C, T const alpha, T const beta, int M, int N, int K);

template<typename T>
void transpose_wrapper(T const * src, T * dst, int M, int N);

template<typename OP, typename T>
void colreduce_wrapper(T const * data, T * res, int M, int N);

template<typename OP, typename T>
void rowreduce_wrapper(T const * data, T * res, int M, int N);

template<typename T>
void coldiv_wrapper(T * data, T const * divs, int M, int N);

template<typename OP, typename T>
void apply_wrapper(T const * src, T * dst, int N);

template<typename OP, typename T>
void combine_wrapper(T const * src1, T const * src2, T * dst, int N);

template<typename T>
void axpby_wrapper(T a, T const * X, T b, T const * Y, T * Z, int N);

/**
 * Declare explicit template instantiations for required parameter types
 */

#define DECL_INST_WRAPPER_TEMPLATES(T) \
extern template void shared2_gemm_wrapper<un_ops::identity,un_ops::identity,un_ops::identity,un_ops::identity,T>(T const * A, T const * B, T * C, T const alpha, T const beta, int M, int N, int K); \
extern template void shared2_gemmpv_wrapper<un_ops::identity,un_ops::identity,un_ops::identity,un_ops::identity,T>(T const * A, T const * B, T const * d, T * C, T const alpha, T const beta, int M, int N, int K); \
extern template void shared2_gemmpv_wrapper<un_ops::identity,un_ops::identity,un_ops::identity,un_ops::sigmoid,T>(T const * A, T const * B, T const * d, T * C, T const alpha, T const beta, int M, int N, int K); \
extern template void shared2_gemmpv_wrapper<un_ops::identity,un_ops::identity,un_ops::identity,un_ops::exponent,T>(T const * A, T const * B, T const * d, T * C, T const alpha, T const beta, int M, int N, int K); \
extern template void transpose_wrapper<T>(T const * src, T * dst, int M, int N); \
extern template void colreduce_wrapper<bin_ops::add,T>(T const * data, T * res, int M, int N); \
extern template void rowreduce_wrapper<bin_ops::add,T>(T const * data, T * res, int M, int N); \
extern template void coldiv_wrapper<T>(T * data, T const * divs, int M, int N); \
extern template void apply_wrapper<un_ops::exponent,T>(T const * src, T * dst, int N); \
extern template void apply_wrapper<un_ops::sigmoid,T>(T const * src, T * dst, int N); \
extern template void apply_wrapper<un_ops::x_times_1_minus_x,T>(T const * src, T * dst, int N); \
extern template void combine_wrapper<bin_ops::mult,T>(T const * src1, T const * src2, T * dst, int N); \
extern template void axpby_wrapper<T>(T a, T const * X, T b, T const * Y, T * Z, int N);

DECL_INST_WRAPPER_TEMPLATES(double)
    
#undef DECL_INST_WRAPPER_TEMPLATES

#endif
