#include "complex_matrix_ops.h"

#include <limits>

namespace
{

Status StatusFromCuda(cudaError_t status)
{
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

Status StatusFromCublas(cublasStatus_t status)
{
    return status == CUBLAS_STATUS_SUCCESS ? Status::kSuccess : Status::kCublasError;
}

Status StatusFromCusolver(cusolverStatus_t status)
{
    return status == CUSOLVER_STATUS_SUCCESS ? Status::kSuccess : Status::kCusolverError;
}

__global__ void AddKernel(const Complex* A, const Complex* B, Complex* C, int64_t count)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = complex_add(A[idx], B[idx]);
    }
}

__global__ void SubKernel(const Complex* A, const Complex* B, Complex* C, int64_t count)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = complex_sub(A[idx], B[idx]);
    }
}

bool CheckBlasSize(int64_t value) { return value > 0 && value <= std::numeric_limits<int>::max(); }

} // namespace

ComplexMatrixOps::ComplexMatrixOps() : ComplexMatrixOps(nullptr) {}

ComplexMatrixOps::ComplexMatrixOps(cudaStream_t stream) : stream_(stream)
{
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS)
    {
        cublas_handle_ = nullptr;
    }
    if (cusolverDnCreate(&cusolver_handle_) != CUSOLVER_STATUS_SUCCESS)
    {
        cusolver_handle_ = nullptr;
    }
    if (cublas_handle_)
    {
        cublasSetStream(cublas_handle_, stream_);
    }
    if (cusolver_handle_)
    {
        cusolverDnSetStream(cusolver_handle_, stream_);
    }
}

ComplexMatrixOps::~ComplexMatrixOps()
{
    if (cublas_handle_)
    {
        cublasDestroy(cublas_handle_);
    }
    if (cusolver_handle_)
    {
        cusolverDnDestroy(cusolver_handle_);
    }
}

Status ComplexMatrixOps::SetStream(cudaStream_t stream)
{
    if (!cublas_handle_ || !cusolver_handle_)
    {
        return Status::kInvalidValue;
    }
    stream_ = stream;
    Status status = StatusFromCublas(cublasSetStream(cublas_handle_, stream_));
    if (status == Status::kSuccess)
    {
        return StatusFromCusolver(cusolverDnSetStream(cusolver_handle_, stream_));
    }
    return status;
}

Status ComplexMatrixOps::MatMul(const Complex* A,
                                const Complex* B,
                                Complex* C,
                                int64_t m,
                                int64_t n,
                                int64_t k,
                                cublasOperation_t transA,
                                cublasOperation_t transB)
{
    if (!A || !B || !C || !cublas_handle_)
    {
        return Status::kInvalidValue;
    }
    if (!CheckBlasSize(m) || !CheckBlasSize(n) || !CheckBlasSize(k))
    {
        return Status::kInvalidValue;
    }

    int m_int = static_cast<int>(m);
    int n_int = static_cast<int>(n);
    int k_int = static_cast<int>(k);

    int lda = (transA == CUBLAS_OP_N) ? m_int : k_int;
    int ldb = (transB == CUBLAS_OP_N) ? k_int : n_int;
    int ldc = m_int;

    Complex alpha = make_complex(static_cast<Real>(1.0), static_cast<Real>(0.0));
    Complex beta = make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0));

    return StatusFromCublas(CUBLAS_GEMM(cublas_handle_,
                                        transA,
                                        transB,
                                        m_int,
                                        n_int,
                                        k_int,
                                        &alpha,
                                        A,
                                        lda,
                                        B,
                                        ldb,
                                        &beta,
                                        C,
                                        ldc));
}

Status ComplexMatrixOps::MatMul(const Complex* A,
                                const Complex* B,
                                Complex* C,
                                int64_t n,
                                cublasOperation_t transA,
                                cublasOperation_t transB)
{
    return MatMul(A, B, C, n, n, n, transA, transB);
}

Status
ComplexMatrixOps::MatAdd(const Complex* A, const Complex* B, Complex* C, int64_t m, int64_t n)
{
    if (!A || !B || !C)
    {
        return Status::kInvalidValue;
    }
    if (m <= 0 || n <= 0 || m > (std::numeric_limits<int64_t>::max() / n))
    {
        return Status::kInvalidValue;
    }
    int64_t count = m * n;
    int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    AddKernel<<<blocks, threads, 0, stream_>>>(A, B, C, count);
    return StatusFromCuda(cudaGetLastError());
}

Status ComplexMatrixOps::MatAdd(const Complex* A, const Complex* B, Complex* C, int64_t n)
{
    return MatAdd(A, B, C, n, n);
}

Status
ComplexMatrixOps::MatSub(const Complex* A, const Complex* B, Complex* C, int64_t m, int64_t n)
{
    if (!A || !B || !C)
    {
        return Status::kInvalidValue;
    }
    if (m <= 0 || n <= 0 || m > (std::numeric_limits<int64_t>::max() / n))
    {
        return Status::kInvalidValue;
    }
    int64_t count = m * n;
    int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    SubKernel<<<blocks, threads, 0, stream_>>>(A, B, C, count);
    return StatusFromCuda(cudaGetLastError());
}

Status ComplexMatrixOps::MatSub(const Complex* A, const Complex* B, Complex* C, int64_t n)
{
    return MatSub(A, B, C, n, n);
}

Status
ComplexMatrixOps::Solve(const Complex* A, const Complex* B, Complex* X, int64_t n, int64_t nrhs)
{
    if (!A || !B || !X || !cusolver_handle_)
    {
        return Status::kInvalidValue;
    }
    if (!CheckBlasSize(n) || !CheckBlasSize(nrhs))
    {
        return Status::kInvalidValue;
    }
    if (n > (std::numeric_limits<int64_t>::max() / n) ||
        n > (std::numeric_limits<int64_t>::max() / nrhs))
    {
        return Status::kInvalidValue;
    }

    int n_int = static_cast<int>(n);
    int nrhs_int = static_cast<int>(nrhs);

    size_t matrix_elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    size_t rhs_elems = static_cast<size_t>(n) * static_cast<size_t>(nrhs);

    Complex* A_work = nullptr;
    int* dev_ipiv = nullptr;
    int* dev_info = nullptr;
    Complex* work = nullptr;
    int lwork = 0;

    auto cleanup = [&]()
    {
        if (work)
        {
            cudaFree(work);
        }
        if (dev_info)
        {
            cudaFree(dev_info);
        }
        if (dev_ipiv)
        {
            cudaFree(dev_ipiv);
        }
        if (A_work)
        {
            cudaFree(A_work);
        }
    };

    if (StatusFromCuda(cudaMalloc(&A_work, matrix_elems * sizeof(Complex))) != Status::kSuccess)
    {
        cleanup();
        return Status::kCudaError;
    }
    if (StatusFromCuda(cudaMemcpyAsync(
            A_work, A, matrix_elems * sizeof(Complex), cudaMemcpyDeviceToDevice, stream_)) !=
        Status::kSuccess)
    {
        cleanup();
        return Status::kCudaError;
    }
    if (X != B)
    {
        if (StatusFromCuda(cudaMemcpyAsync(
                X, B, rhs_elems * sizeof(Complex), cudaMemcpyDeviceToDevice, stream_)) !=
            Status::kSuccess)
        {
            cleanup();
            return Status::kCudaError;
        }
    }

    if (StatusFromCuda(cudaMalloc(&dev_ipiv, n_int * sizeof(int))) != Status::kSuccess)
    {
        cleanup();
        return Status::kCudaError;
    }
    if (StatusFromCuda(cudaMalloc(&dev_info, sizeof(int))) != Status::kSuccess)
    {
        cleanup();
        return Status::kCudaError;
    }

    if (StatusFromCusolver(CUSOLVER_GETRF_BUFFER_SIZE(
            cusolver_handle_, n_int, n_int, A_work, n_int, &lwork)) != Status::kSuccess)
    {
        cleanup();
        return Status::kCusolverError;
    }

    if (StatusFromCuda(cudaMalloc(&work, static_cast<size_t>(lwork) * sizeof(Complex))) !=
        Status::kSuccess)
    {
        cleanup();
        return Status::kCudaError;
    }

    if (StatusFromCusolver(CUSOLVER_GETRF(
            cusolver_handle_, n_int, n_int, A_work, n_int, work, dev_ipiv, dev_info)) !=
        Status::kSuccess)
    {
        cleanup();
        return Status::kCusolverError;
    }

    if (StatusFromCusolver(CUSOLVER_GETRS(cusolver_handle_,
                                          CUBLAS_OP_N,
                                          n_int,
                                          nrhs_int,
                                          A_work,
                                          n_int,
                                          dev_ipiv,
                                          X,
                                          n_int,
                                          dev_info)) != Status::kSuccess)
    {
        cleanup();
        return Status::kCusolverError;
    }

    {
        int info_host = 0;
        if (StatusFromCuda(cudaMemcpyAsync(
                &info_host, dev_info, sizeof(int), cudaMemcpyDeviceToHost, stream_)) !=
            Status::kSuccess)
        {
            cleanup();
            return Status::kCudaError;
        }
        if (StatusFromCuda(cudaStreamSynchronize(stream_)) != Status::kSuccess)
        {
            cleanup();
            return Status::kCudaError;
        }
        if (info_host != 0)
        {
            cleanup();
            return Status::kCusolverError;
        }
    }

    cleanup();
    return Status::kSuccess;
}

Status ComplexMatrixOps::Solve(const Complex* A, const Complex* B, Complex* X, int64_t n)
{
    return Solve(A, B, X, n, n);
}
