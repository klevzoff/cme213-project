#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"
#include <vector>
#include <cmath>
#include <numeric>

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)


#define DEBUG_PRINT(A) \
    std::cout << "--------------------------------------------------------------------" << std::endl; \
    std::cout << #A << " = " << std::endl; \
    (A).print(); \
    std::cout << "--------------------------------------------------------------------" << std::endl


double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;
    
    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);
    
    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}


/* GPU IMPLEMENTATIONS */

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
void transpose(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    assert(dst.nrow() == src.ncol() && dst.ncol() == src.nrow());
    transpose_wrapper(src.data(), dst.data(), src.nrow(), src.ncol());
}

template<typename T>
DeviceMat<T> transpose(DeviceMat<T> const & src)
{
    DeviceMat<T> dst(src.ncol(), src.nrow());
    transpose(src, dst);
    return dst;
}

template<typename OP, typename T>
void colreduce(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    assert(dst.nrow() == 1 && dst.ncol() == src.ncol());
    colreduce_wrapper<OP>(src.data(), dst.data(), src.nrow(), src.ncol());
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
    rowreduce_wrapper<OP>(src.data(), dst.data(), src.nrow(), src.ncol());
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
    colreduce<bin_ops::greater_of>(src, dst);
}

template<typename T>
DeviceMat<T> colmax(DeviceMat<T> const & src)
{
    return colreduce<bin_ops::greater_of>(src);
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
    rowreduce<bin_ops::greater_of>(src, dst);
}

template<typename T>
DeviceMat<T> rowmax(DeviceMat<T> const & src)
{
    return rowreduce<bin_ops::greater_of>(src);
}

template<typename T>
void coldiv(DeviceMat<T> & mat, DeviceMat<T> const & divs)
{
    assert(divs.nrow() == 1 && divs.ncol() == mat.ncol());
    coldiv_wrapper(mat.data(), divs.data(), mat.nrow(), mat.ncol());
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
void apply(DeviceMat<T> const & src, DeviceMat<T> & dst)
{
    assert(dst.nrow() == src.nrow() && dst.ncol() == src.ncol());
    apply_wrapper<OP>(src.data(), dst.data(), src.size());
}

template<typename OP, typename T>
DeviceMat<T> apply(DeviceMat<T> const & src)
{
    DeviceMat<T> dst(src.nrow(), src.ncol());
    apply<OP>(src, dst);
    return dst;
}

template<typename OP, typename T>
void combine(DeviceMat<T> const & src1, DeviceMat<T> const & src2, DeviceMat<T> & dst)
{
    assert(dst.nrow() == src1.nrow() && dst.ncol() == src1.ncol() && src2.nrow() == src1.nrow() && src2.ncol() == src1.ncol());
    combine_wrapper<OP>(src1.data(), src2.data(), dst.data(), src1.size());
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
    using eye = un_ops::identity; // shortcut
    assert(C.nrow() == A.nrow() && C.ncol() == B.ncol());
    shared2_gemm_wrapper<eye,eye,eye,eye>(A.data(), B.data(), C.data(), alpha, beta, A.nrow(), B.ncol(), A.ncol());
}

template<typename T>
void gemmpv(T alpha, DeviceMat<T> const & A, DeviceMat<T> const & B, T beta, DeviceMat<T> const & d, DeviceMat<T> & C)
{
    using eye = un_ops::identity; // shortcut
    assert(C.nrow() == A.nrow() && C.ncol() == B.ncol() && d.nrow() == A.nrow());
    shared2_gemmpv_wrapper<eye,eye,eye,eye>(A.data(), B.data(), d.data(), C.data(), alpha, beta, A.nrow(), B.ncol(), A.ncol());
}

template<typename T>
void gemmpv_sigmoid(T alpha, DeviceMat<T> const & A, DeviceMat<T> const & B, T beta, DeviceMat<T> const & d, DeviceMat<T> & C)
{
    using eye = un_ops::identity; // shortcut
    assert(C.nrow() == A.nrow() && C.ncol() == B.ncol() && d.nrow() == A.nrow());
    shared2_gemmpv_wrapper<eye,eye,eye,un_ops::sigmoid>(A.data(), B.data(), d.data(), C.data(), alpha, beta, A.nrow(), B.ncol(), A.ncol());
}

template<typename T>
void gemmpv_exponent(T alpha, DeviceMat<T> const & A, DeviceMat<T> const & B, T beta, DeviceMat<T> const & d, DeviceMat<T> & C)
{
    using eye = un_ops::identity; // shortcut
    assert(C.nrow() == A.nrow() && C.ncol() == B.ncol() && d.nrow() == A.nrow());
    shared2_gemmpv_wrapper<eye,eye,eye,un_ops::exponent>(A.data(), B.data(), d.data(), C.data(), alpha, beta, A.nrow(), B.ncol(), A.ncol());
}

template<typename T>
void matmult(DeviceMat<T> const & A, DeviceMat<T> const & B, DeviceMat<T> & C)
{
    gemm(T(1), A, B, T(0), C);
}

template<typename T>
DeviceMat<T> matmult(DeviceMat<T> const & A, DeviceMat<T> const & B)
{
    DeviceMat<T> C(A.nrow(), B.ncol());
    matmult(A, B, C);
    return C;
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
void axpby(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y, DeviceMat<T> & Z)
{
    assert(Z.nrow() == X.nrow() && Z.ncol() == X.ncol() && X.nrow() == Y.nrow() && X.ncol() == Y.ncol()); 
    axpby_wrapper(a, X.data(), b, Y.data(), Z.data(), X.size());
}

template<typename T>
DeviceMat<T> axpby(T a, DeviceMat<T> const & X, T b, DeviceMat<T> const & Y)
{
    DeviceMat<T> Z(X.nrow(), X.ncol());
    axpby(a, X, b, Y, Z);
    return Z;
}

/**
 * NN-specific stuff - helper structs and methods
 */

using DMat = DeviceMat<double>;

struct device_nn
{
    int num_layers;
    std::vector<int> H;
    std::vector<DMat> W;
    std::vector<DMat> b;
};

struct device_grads
{
    std::vector<DMat> dW;
    std::vector<DMat> db;
};

struct device_cache
{
    std::vector<DMat> a;
};

void to_host(DMat const & dmat, arma::mat & hmat)
{
    hmat.resize(dmat.nrow(), dmat.ncol());
    dmat.copy_to_host(hmat.memptr());
}

void to_device(arma::mat const & hmat, DMat & dmat)
{
    dmat.resize(hmat.n_rows, hmat.n_cols);
    dmat.copy_from_host(hmat.memptr());
}

void to_device(NeuralNetwork const & hnn, device_nn & dnn)
{
    dnn.num_layers = hnn.num_layers;
    dnn.H = hnn.H;
    dnn.W.resize(hnn.num_layers);
    dnn.b.resize(hnn.num_layers);
    for (int i = 0; i < hnn.num_layers; ++i)
    {
        to_device(hnn.W[i], dnn.W[i]);
        to_device(hnn.b[i], dnn.b[i]);
    }
}

void to_host(device_nn const & dnn, NeuralNetwork & hnn)
{
    hnn.H = dnn.H;
    hnn.W.resize(dnn.num_layers);
    hnn.b.resize(dnn.num_layers);
    for (int i = 0; i < dnn.num_layers; ++i)
    {
        to_host(dnn.W[i], hnn.W[i]);
        to_host(dnn.b[i], hnn.b[i]);
    }
}

void to_device(grads const & hgrads, device_grads & dgrads)
{
    int const num_layers = hgrads.dW.size();
    dgrads.dW.resize(num_layers);
    dgrads.db.resize(num_layers);
    for (int i = 0; i < num_layers; ++i)
    {
        to_device(hgrads.dW[i], dgrads.dW[i]);
        to_device(hgrads.db[i], dgrads.db[i]);
    }
}

void to_host(device_grads const & dgrads, grads & hgrads)
{
    int const num_layers = dgrads.dW.size();
    hgrads.dW.resize(num_layers);
    hgrads.db.resize(num_layers);
    for (int i = 0; i < num_layers; ++i)
    {
        to_host(dgrads.dW[i], hgrads.dW[i]);
        to_host(dgrads.db[i], hgrads.db[i]);
    }
}

void allreduce(DMat & dA, arma::mat & hA)
{
    to_host(dA, hA);
    MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, hA.memptr(), hA.n_rows * hA.n_cols, 
                                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    to_device(hA, dA);
}

/**
 * Actual implementation of NN training
 */ 

void device_feedforward(device_nn const & nn, DMat const & X, device_cache & cache)
{  
    cache.a[0].resize(nn.W[0].nrow(), X.ncol()); // use fused kernel with sigmoid
    gemmpv_sigmoid(1.0, nn.W[0], X, 1.0, nn.b[0], cache.a[0]); // M = 100(0), N = 800, K = 784

    cache.a[1].resize(nn.W[1].nrow(), X.ncol()); // use fused kernel with exponent
    gemmpv_exponent(1.0, nn.W[1], cache.a[0], 1.0, nn.b[1], cache.a[1]); // M = 10, N = 800, K = 100(0)
    normalize(cache.a[1]);
}

void device_backprop(device_nn const & nn, DMat const & Xt, DMat const & y, double reg,
                     device_cache const & bpcache, device_grads & bpgrads, int N, int num_procs)
{   
    static DMat a0t;
    a0t.resize(bpcache.a[0].ncol(), bpcache.a[0].nrow());
    transpose(bpcache.a[0], a0t);

    static DMat diff;
    diff.resize(y.nrow(), y.ncol());
    axpby(1.0 / N, bpcache.a[1], -1.0 / N, y, diff);
    
    bpgrads.dW[1] = nn.W[1];
    gemm(1.0, diff, a0t, reg / num_procs, bpgrads.dW[1]); // M = 10, N = 100(0), K = 800
    bpgrads.db[1] = sum(diff, 1);
    
    static DMat W1t;
    W1t.resize(nn.W[1].ncol(), nn.W[1].nrow());
    transpose(nn.W[1], W1t);
    
    static DMat da1;
    da1.resize(W1t.nrow(), diff.ncol());  
    matmult(W1t, diff, da1); // M = 100(0), N = 800, K = 10
    
    static DMat dz1;
    dz1.resize(da1.nrow(), da1.ncol());
    sigmoid_derivative(bpcache.a[0], dz1);
    hadamard(da1, dz1, dz1);
    
    bpgrads.dW[0] = nn.W[0];
    gemm(1.0, dz1, Xt, reg / num_procs, bpgrads.dW[0]); // M = 100(0), N = 784, K = 800
    
    bpgrads.db[0] = sum(dz1, 1);
}


void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug)
{
 
    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0) ? X.n_cols : 0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;
    
    int const num_batches = (N + batch_size - 1) / batch_size;
    
    std::vector<DMat> X_batch(num_batches);
    std::vector<DMat> Xt_batch(num_batches);
    std::vector<DMat> y_batch(num_batches);
    std::vector<int> num_col(num_batches);
    
    // distribute the input
    for (int batch = 0; batch < num_batches; ++batch)
    {         
        int const first_col = batch * batch_size;
        int const last_col = std::min((batch + 1)*batch_size-1, N-1);
        num_col[batch] = last_col - first_col + 1;
        
        // trying to get as fair a distribution as possible
        double const nc_per_rank = double(num_col[batch]) / num_procs;
        std::vector<int> num_col_rank(num_procs);
        std::vector<int> sendcounts_X(num_procs);
        std::vector<int> sendcounts_y(num_procs);

        for (int irank = 0; irank < num_procs; ++irank)
        {
            num_col_rank[irank] = static_cast<int>(std::lround((irank+1) * nc_per_rank) - std::lround(irank * nc_per_rank));
            sendcounts_X[irank] = num_col_rank[irank] * nn.H[0];
            sendcounts_y[irank] = num_col_rank[irank] * nn.H[nn.num_layers];
        }
        
        std::vector<int> senddispls_X(num_procs, 0);
        std::vector<int> senddispls_y(num_procs, 0);
        std::partial_sum(sendcounts_X.begin(), sendcounts_X.end()-1, senddispls_X.begin()+1);
        std::partial_sum(sendcounts_y.begin(), sendcounts_y.end()-1, senddispls_y.begin()+1);
       
        arma::mat X_batch_host(nn.H[0], num_col_rank[rank]);
        arma::mat y_batch_host(nn.H[nn.num_layers], num_col_rank[rank]);

        double const * const X_sendbuf = (rank == 0) ? X.colptr(first_col) : nullptr;
        double const * const y_sendbuf = (rank == 0) ? y.colptr(first_col) : nullptr;

        MPI_SAFE_CALL(MPI_Scatterv(X_sendbuf, sendcounts_X.data(), senddispls_X.data(), MPI_DOUBLE, 
                                   X_batch_host.memptr(), sendcounts_X[rank], MPI_DOUBLE, 
                                   0, MPI_COMM_WORLD));

        MPI_SAFE_CALL(MPI_Scatterv(y_sendbuf, sendcounts_y.data(), senddispls_y.data(), MPI_DOUBLE, 
                                   y_batch_host.memptr(), sendcounts_y[rank], MPI_DOUBLE, 
                                   0, MPI_COMM_WORLD));

        to_device(X_batch_host, X_batch[batch]);
        to_device(y_batch_host, y_batch[batch]);
        Xt_batch[batch] = transpose(X_batch[batch]);
    }
    
    // copy the whole network onto device
    device_nn dnn;
    to_device(nn, dnn);

    int iter = 0;

    device_cache bpcache;
    device_grads bpgrads;
    grads host_grads;
    
    bpcache.a.resize(2);
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    host_grads.dW.resize(2);
    host_grads.db.resize(2);
    
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (int batch = 0; batch < num_batches; ++batch)
        {                  
            DMat const & dX_batch  = X_batch[batch];
            DMat const & dXt_batch = Xt_batch[batch];
            DMat const & dy_batch  = y_batch[batch];

            device_feedforward(dnn, dX_batch, bpcache);
            device_backprop(dnn, dXt_batch, dy_batch, reg, bpcache, bpgrads, num_col[batch], num_procs);
            
            // the code below works properly, but commented out to avoid losing time  
            /*
            if (print_every > 0 && iter % print_every == 0)
            {
                to_host(dnn, nn);
                
                arma::mat hy_batch;
                to_host(dy_batch, hy_batch);
                
                if (grad_check)
                {
                    arma::mat hX_batch;
                    to_host(dX_batch, hX_batch);
                
                    grads numgrads;
                    numgrad(nn, hX_batch, hy_batch, reg, numgrads);
                    
                    grads host_bpgrads;
                    to_host(bpgrads, host_bpgrads);
                    
                    assert(gradcheck(numgrads, host_bpgrads));
                }
                
                arma::mat a1;
                to_host(bpcache.a[1], a1);

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, a1, hy_batch, reg) << "\n";
            }
            */
            
            // Gradient descent step
            for (int i = 0; i < dnn.num_layers; ++i)
            {
                if (num_procs > 1)
                {
                    allreduce(bpgrads.dW[i], host_grads.dW[i]);
                    allreduce(bpgrads.db[i], host_grads.db[i]);
                }
                axpby(1.0, dnn.W[i], -learning_rate, bpgrads.dW[i], dnn.W[i]);
                axpby(1.0, dnn.b[i], -learning_rate, bpgrads.db[i], dnn.b[i]);
            }

            if (print_every <= 0)
            {
                print_flag = batch == 0;
            }
            else
            {
                print_flag = iter % print_every == 0;
            }

            // the code below works properly, but commented out to avoid losing time  
            /*
            if (debug && rank == 0 && print_flag)
            {
                to_host(dnn, nn); // copy network onto host for debugging output
                write_diff_gpu_cpu(nn, iter, error_file);
            }
            */

            iter++;
        }
    }
    
    to_host(dnn, nn); // copy the network back onto host
    error_file.close();
}
