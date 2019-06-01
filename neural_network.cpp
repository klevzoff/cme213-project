#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"
#include <vector>

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
    DMat X;
    std::vector<DMat> z;
    std::vector<DMat> a;
    DMat yc;
};

arma::mat to_host(DMat const & dmat)
{
    arma::mat hmat(dmat.nrow(), dmat.ncol());
    dmat.copy_to_host(hmat.memptr());
    return hmat;
}

DMat to_device(arma::mat const & hmat)
{
    DMat dmat(hmat.n_rows, hmat.n_cols);
    dmat.copy_from_host(hmat.memptr());
    return dmat;
}

void to_device(NeuralNetwork const & hnn, device_nn & dnn)
{
    dnn.num_layers = hnn.num_layers;
    dnn.H = hnn.H;
    dnn.W.resize(hnn.num_layers);
    dnn.b.resize(hnn.num_layers);
    for (int i = 0; i < hnn.num_layers; ++i)
    {
        dnn.W[i] = to_device(hnn.W[i]);
        dnn.b[i] = to_device(hnn.b[i]);
    }
}

void to_host(device_nn const & dnn, NeuralNetwork & hnn)
{
    hnn.H = dnn.H;
    hnn.W.resize(dnn.num_layers);
    hnn.b.resize(dnn.num_layers);
    for (int i = 0; i < dnn.num_layers; ++i)
    {
        hnn.W[i] = to_host(dnn.W[i]);
        hnn.b[i] = to_host(dnn.b[i]);
    }
}

void to_device(grads const & hgrads, device_grads & dgrads)
{
    int const num_layers = hgrads.dW.size();
    dgrads.dW.resize(num_layers);
    dgrads.db.resize(num_layers);
    for (int i = 0; i < num_layers; ++i)
    {
        dgrads.dW[i] = to_device(hgrads.dW[i]);
        dgrads.db[i] = to_device(hgrads.db[i]);
    }
}

void to_host(device_grads const & dgrads, grads & hgrads)
{
    int const num_layers = dgrads.dW.size();
    hgrads.dW.resize(num_layers);
    hgrads.db.resize(num_layers);
    for (int i = 0; i < num_layers; ++i)
    {
        hgrads.dW[i] = to_host(dgrads.dW[i]);
        hgrads.db[i] = to_host(dgrads.db[i]);
    }
}

void device_feedforward(device_nn const & nn, DMat const & X, device_cache & cache)
{
    cache.z.resize(2);
    cache.a.resize(2);

    assert(X.nrow() == nn.W[0].ncol());
    cache.X = X;
    int N = X.ncol();

    cache.z[0] = repmat(nn.b[0], 1, N);
    gemm(1.0, nn.W[0], X, 1.0, cache.z[0]);
   
    cache.a[0] = sigmoid(cache.z[0]);

    assert(cache.a[0].nrow() == nn.W[1].ncol());
    cache.z[1] = repmat(nn.b[1], 1, N);
    gemm(1.0, nn.W[1], cache.a[0], 1.0, cache.z[1]);

    cache.a[1] = softmax(cache.z[1]);
    cache.yc = cache.a[1];
}

void device_backprop(device_nn const & nn, DMat const & y, double reg,
                     device_cache const & bpcache, device_grads & bpgrads)
{
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.ncol();

    DMat diff = axpby(1.0 / N, bpcache.yc, -1.0 / N, y);
    bpgrads.dW[1] = nn.W[1];
    gemm(1.0, diff, transpose(bpcache.a[0]), reg, bpgrads.dW[1]);
    bpgrads.db[1] = sum(diff, 1);
    
    DMat da1 = matmult(transpose(nn.W[1]), diff);
    DMat dz1 = sigmoid_derivative(bpcache.a[0]);
    hadamard(da1, dz1, dz1);
    
    bpgrads.dW[0] = nn.W[0];
    gemm(1.0, dz1, transpose(bpcache.X), reg, bpgrads.dW[0]);
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
    
    // copy the whole network onto device
    device_nn dnn;
    to_device(nn, dnn);

    int iter = 0;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        int num_batches = (N + batch_size - 1) / batch_size;

        for (int batch = 0; batch < num_batches; ++batch)
        {         
            int const first_col = batch * batch_size;
            int const last_col = std::min((batch + 1)*batch_size-1, N-1);
            int const num_col = last_col - first_col + 1;
            int const num_col_rank = num_col / num_procs; // assume equal for now
            
            arma::mat X_batch(nn.H[0], num_col_rank);
            arma::mat y_batch(nn.H[nn.num_layers], num_col_rank);
            
            double const * const X_sendbuf = (rank == 0) ? X.colptr(first_col) : nullptr;
            double const * const y_sendbuf = (rank == 0) ? y.colptr(first_col) : nullptr;
                            
            MPI_SAFE_CALL(MPI_Scatter(X_sendbuf, num_col_rank * nn.H[0], MPI_DOUBLE, 
                                      X_batch.memptr(), num_col_rank * nn.H[0], MPI_DOUBLE, 
                                      0, MPI_COMM_WORLD));
            
            MPI_SAFE_CALL(MPI_Scatter(y_sendbuf, num_col_rank * nn.H[nn.num_layers], MPI_DOUBLE, 
                                      y_batch.memptr(), num_col_rank * nn.H[nn.num_layers], MPI_DOUBLE, 
                                      0, MPI_COMM_WORLD));
            
            DMat dX_batch = to_device(X_batch);
            DMat dy_batch = to_device(y_batch);

            device_cache bpcache;
            device_feedforward(dnn, dX_batch, bpcache);

            device_grads bpgrads;
            device_backprop(dnn, dy_batch, reg, bpcache, bpgrads);
            
            if (print_every > 0 && iter % print_every == 0)
            {
                to_host(dnn, nn);
                
                if (grad_check)
                {
                    grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    
                    grads host_bpgrads;
                    to_host(bpgrads, host_bpgrads);
                    
                    assert(gradcheck(numgrads, host_bpgrads));
                }

                arma::mat yc = to_host(bpcache.yc);
                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, yc, y_batch, reg) << "\n";
            }
            
            // Gradient descent step
            for (int i = 0; i < dnn.num_layers; ++i)
            {
                arma::mat dW = to_host(bpgrads.dW[i]);
                MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, dW.memptr(), dW.n_rows * dW.n_cols, 
                                            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
                dW /= num_procs;
                bpgrads.dW[i] = to_device(dW);

                arma::mat db = to_host(bpgrads.db[i]);
                MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, db.memptr(), db.n_rows * db.n_cols, 
                                            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
                db /= num_procs;
                bpgrads.db[i] = to_device(db);
                              
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

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if (debug && rank == 0 && print_flag)
            {
                to_host(dnn, nn); // copy network onto host for debugging output
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }
    
    to_host(dnn, nn); // copy the network back onto host
    error_file.close();
}
