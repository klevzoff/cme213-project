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

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);

/**
 * @class DeviceMat
 * @brief Lightweight device matrix class to simplify memory management and argument passing
 *
 * @note Supports copy and move constructions/assignment, safe to return by value
 *
 * @note To be used only with primitive types. Does not call object constructors when copying!
 */
template<typename T>
class DeviceMat
{
public:

    DeviceMat() : DeviceMat(0, 0) {}

    DeviceMat(int M, int N) : m_data(nullptr)
    {
        allocate(M, N);
    }
    
    ~DeviceMat()
    {
        deallocate();
    }
    
    DeviceMat(DeviceMat<T> const & other)
    {
        allocate(other.nrow(), other.ncol());
        copy_from_device(other.data());
    }
    
    DeviceMat(DeviceMat<T> && other)
    {
        setsize(other.nrow(), other.ncol());
        std::swap(m_data, other.m_data);
    }
    
    DeviceMat<T> & operator=(DeviceMat<T> const & other)
    {
        if (&other != this)
        {
            resize(other.nrow(), other.ncol());
            copy_from_device(other.data());
        }
        return *this;
    }
    
    DeviceMat<T> & operator=(DeviceMat<T> && other)
    {
        if (&other != this)
        {
            setsize(other.nrow(), other.ncol());
            std::swap(m_data, other.m_data);
        }
        return *this;
    }
    
    void resize(int M, int N)
    {
        if (M != nrow() || N != ncol())
        {
            deallocate();
            allocate(M, N);
        }
    }
    
    void setzero()
    {
        if (size() > 0)
        {
            cudaMemset(m_data, 0, bytes());
        }
    }
    
    int nrow() const  { return m_nrow; }
    int ncol() const  { return m_ncol; }
    int size() const  { return m_nrow * m_ncol; }
    int bytes() const { return size() * sizeof(T); }
    
    T       * data()       { return m_data; };
    T const * data() const { return m_data; };
    
    T       * col(int j)       { assert(j < m_ncol); return m_data + j * nrow(); }
    T const * col(int j) const { assert(j < m_ncol); return m_data + j * nrow(); }
    
    void copy_from_device(T const * src)
    {
        cudaMemcpy(m_data, src, bytes(), cudaMemcpyDeviceToDevice);
    }
    
    void copy_from_host(T const * src)
    {
        cudaMemcpy(m_data, src, bytes(), cudaMemcpyHostToDevice);
    }
    
    void copy_to_host(T * dst) const
    {
        cudaMemcpy(dst, m_data, bytes(), cudaMemcpyDeviceToHost);
    }
    
    void print(std::ostream & os = std::cout) const
    {
        T * temp = new T[size()];
        copy_to_host(temp);
        
        os << std::scientific;
        for (int i = 0; i < m_nrow; ++i)
        {
            for (int j = 0; j < m_ncol; ++j)
            {
                os << std::right << std::setw(13) << std::setprecision(4) << temp[j * m_nrow + i];
            }
            os << std::endl;
        }
        
        delete[] temp;
    }

private:

    void setsize(int M, int N)
    {
        assert(M >= 0 && N >= 0);
        m_nrow = M;
        m_ncol = N;
    }

    void allocate(int M, int N)
    {
        setsize(M, N);
        if (size() > 0)
        {
            cudaMalloc((void**) &m_data, bytes());
        }
    }
    
    void deallocate()
    {
        cudaFree(m_data);
    }

    int m_nrow;
    int m_ncol;
    T * m_data;
};

template<typename T>
DeviceMat<T> repmat(DeviceMat<T> const & src, int dim, int times)
{
    DeviceMat<T> dst;
    
    if (dim == 0)
    {
        dst.resize(src.nrow() * times, src.ncol());
        for (int j = 0; j < src.ncol(); ++j)
        {
            for (int rep = 0; rep < times; ++rep)
            {
                cudaMemcpy(dst.col(j) + rep * src.nrow(), src.col(j), src.nrow() * sizeof(T), cudaMemcpyDeviceToDevice);
            }
        }
        
    }
    else if (dim == 1)
    {
        dst.resize(src.nrow(), src.ncol() * times);
        for (int rep = 0; rep < times; ++rep)
        {
            cudaMemcpy(dst.data() + rep * src.size(), src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        std::cerr << "repmat(): invalid dim: " << dim << std::endl;
        exit(1);
    }
    
    return dst;
}

template<typename T>
void gemm(T alpha, DeviceMat<T> const & A, DeviceMat<T> const & B, T beta, DeviceMat<T> & C);

template<typename T>
void hadamard(DeviceMat<T> const & A, DeviceMat<T> const & B, DeviceMat<T> & C);

template<typename T>
DeviceMat<T> matmult(DeviceMat<T> const & A, DeviceMat<T> const & B);

template<typename T>
DeviceMat<T> sigmoid(DeviceMat<T> const & A);

template<typename T>
DeviceMat<T> sigmoid_derivative(DeviceMat<T> const & A);

template<typename T>
DeviceMat<T> softmax(DeviceMat<T> const & A);

template<typename T>
DeviceMat<T> transpose(DeviceMat<T> const & A);

template<typename T>
DeviceMat<T> axpby(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y);

template<typename T>
void axpby(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y, DeviceMat<T> & Z);

template<typename T>
DeviceMat<T> sum(DeviceMat<T> const & A, int dim);

#define EXT_TMP_DMAT_FUNCS(T) \
extern template void gemm<T>(T alpha, DeviceMat<T> const & A, DeviceMat<T> const & B, T beta, DeviceMat<T> & C); \
extern template void hadamard<T>(DeviceMat<T> const & A, DeviceMat<T> const & B, DeviceMat<T> & C); \
extern template DeviceMat<T> matmult(DeviceMat<T> const & A, DeviceMat<T> const & B); \
extern template DeviceMat<T> sigmoid<T>(DeviceMat<T> const & A); \
extern template DeviceMat<T> sigmoid_derivative<T>(DeviceMat<T> const & A); \
extern template DeviceMat<T> softmax<T>(DeviceMat<T> const & A); \
extern template DeviceMat<T> transpose<T>(DeviceMat<T> const & A); \
extern template DeviceMat<T> axpby<T>(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y); \
extern template void axpby<T>(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y, DeviceMat<T> & Z); \
extern template DeviceMat<T> sum<T>(DeviceMat<T> const & A, int dim);

EXT_TMP_DMAT_FUNCS(double)
    
#undef EXT_TMP_DMAT_FUNCS

#endif
