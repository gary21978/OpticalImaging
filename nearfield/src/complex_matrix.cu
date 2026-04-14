#include "complex_matrix.h"

#include <limits>

static Status StatusFromCuda(cudaError_t status)
{
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

static Status StatusFromCublas(cublasStatus_t status)
{
    return status == CUBLAS_STATUS_SUCCESS ? Status::kSuccess : Status::kCublasError;
}

static Status StatusFromCusolver(cusolverStatus_t status)
{
    return status == CUSOLVER_STATUS_SUCCESS ? Status::kSuccess : Status::kCusolverError;
}

__global__ void AddKernel(const Complex* A, const Complex* B, Complex* C, int64_t count)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = ComplexAdd(A[idx], B[idx]);
    }
}

__global__ void SubKernel(const Complex* A, const Complex* B, Complex* C, int64_t count)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = ComplexSub(A[idx], B[idx]);
    }
}

__global__ void
LeftDiagMulKernel(const Complex* diag, const Complex* B, Complex* C, int64_t rows, int64_t count)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        int64_t row = idx % rows;
        C[idx] = ComplexMul(diag[row], B[idx]);
    }
}

__global__ void
RightDiagMulKernel(const Complex* A, const Complex* diag, Complex* C, int64_t rows, int64_t count)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        int64_t col = idx / rows;
        C[idx] = ComplexMul(A[idx], diag[col]);
    }
}

__global__ void ScaleKernel(const Complex* A, Complex* C, int64_t count, Complex scalar)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = ComplexMul(A[idx], scalar);
    }
}

__global__ void IdentityKernel(Complex* A, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * n;
    if (idx < total)
    {
        int64_t row = idx % n;
        int64_t col = idx / n;
        if (row == col)
        {
            A[idx] = make_complex(1.0, 0.0);
        }
        else
        {
            A[idx] = make_complex(0.0, 0.0);
        }
    }
}

__global__ void DiagFromVectorKernel(const Complex* diag, Complex* out, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * n;
    if (idx < total)
    {
        int64_t row = idx % n;
        int64_t col = idx / n;
        if (row == col)
        {
            out[idx] = diag[row];
        }
        else
        {
            out[idx] = make_complex(0.0, 0.0);
        }
    }
}

__global__ void ConstantDiagKernel(Complex* out, int64_t n, Complex value)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * n;
    if (idx < total)
    {
        int64_t row = idx % n;
        int64_t col = idx / n;
        if (row == col)
        {
            out[idx] = value;
        }
        else
        {
            out[idx] = make_complex(0.0, 0.0);
        }
    }
}

__global__ void CopyBlockKernel(const Complex* src,
                                int64_t n,
                                Complex* dst,
                                int64_t dst_n,
                                int64_t row_offset,
                                int64_t col_offset,
                                Complex scale)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * n;
    if (idx < total)
    {
        int64_t row = idx % n;
        int64_t col = idx / n;
        const Complex value = ComplexMul(src[col * n + row], scale);
        dst[(col + col_offset) * dst_n + (row + row_offset)] = value;
    }
}

__global__ void CopyBlockDirectKernel(const Complex* src,
                                      int64_t n,
                                      Complex* dst,
                                      int64_t dst_n,
                                      int64_t row_offset,
                                      int64_t col_offset)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * n;
    if (idx < total)
    {
        int64_t row = idx % n;
        int64_t col = idx / n;
        dst[(col + col_offset) * dst_n + (row + row_offset)] = src[col * n + row];
    }
}

__global__ void CopyBlockNegatedKernel(const Complex* src,
                                       int64_t n,
                                       Complex* dst,
                                       int64_t dst_n,
                                       int64_t row_offset,
                                       int64_t col_offset)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * n;
    if (idx < total)
    {
        int64_t row = idx % n;
        int64_t col = idx / n;
        const Complex value = src[col * n + row];
        dst[(col + col_offset) * dst_n + (row + row_offset)] = make_complex(-value.x, -value.y);
    }
}

__global__ void AddDiagonalKernel(Complex* A, const Complex* diag, int64_t n, Complex scale)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const int64_t offset = idx * n + idx;
        A[offset] = ComplexAdd(A[offset], ComplexMul(diag[idx], scale));
    }
}

__global__ void AddDiagonalConstantKernel(Complex* A, int64_t n, Complex value)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const int64_t offset = idx * n + idx;
        A[offset] = ComplexAdd(A[offset], value);
    }
}

__global__ void CopyDiagonalBlockKernel(const Complex* diag,
                                        int64_t n,
                                        Complex* dst,
                                        int64_t dst_n,
                                        int64_t row_offset,
                                        int64_t col_offset,
                                        Complex scale)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const int64_t row = row_offset + idx;
        const int64_t col = col_offset + idx;
        dst[col * dst_n + row] = ComplexMul(diag[idx], scale);
    }
}

__global__ void ElementwiseMulKernel(const Complex* a, const Complex* b, Complex* out, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = ComplexMul(a[idx], b[idx]);
    }
}

static constexpr int kReduceBlockSize = 256;

__global__ void ColumnSumKernel(const Complex* A, int64_t n, Real* col_sums)
{
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if (col >= n)
    {
        return;
    }

    Real sum = 0.0;
    for (int64_t row = threadIdx.x; row < n; row += blockDim.x)
    {
        sum += ComplexAbs(A[col * n + row]);
    }

    __shared__ Real shared[kReduceBlockSize];
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        col_sums[col] = shared[0];
    }
}

__global__ void MaxReduceKernel(const Real* input, int64_t n, Real* output)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    Real value = 0.0;
    if (idx < n)
    {
        value = input[idx];
    }

    __shared__ Real shared[kReduceBlockSize];
    shared[threadIdx.x] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            const Real other = shared[threadIdx.x + stride];
            shared[threadIdx.x] = shared[threadIdx.x] > other ? shared[threadIdx.x] : other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = shared[0];
    }
}

static bool CheckBlasSize(int64_t value)
{
    return value > 0 && value <= std::numeric_limits<int>::max();
}

struct DeviceRealVector
{
    Real* ptr = nullptr;
    size_t elements = 0;

    ~DeviceRealVector()
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    Real* get() const { return ptr; }

    Status Allocate(size_t count)
    {
        if (count == 0)
        {
            return Status::kInvalidValue;
        }
        if (count > (std::numeric_limits<size_t>::max() / sizeof(Real)))
        {
            return Status::kInvalidValue;
        }
        if (ptr && elements == count)
        {
            return Status::kSuccess;
        }
        if (ptr)
        {
            cudaFree(ptr);
        }
        Real* out = nullptr;
        if (cudaMalloc(&out, count * sizeof(Real)) != cudaSuccess)
        {
            return Status::kCudaError;
        }
        ptr = out;
        elements = count;
        return Status::kSuccess;
    }
};

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

    Complex alpha = make_complex(1.0, 0.0);
    Complex beta = make_complex(0.0, 0.0);

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

Status ComplexMatrixOps::DiagLeftMatMul(const Complex* diag,
                                        const Complex* B,
                                        Complex* C,
                                        int64_t m,
                                        int64_t n)
{
    if (!diag || !B || !C)
    {
        return Status::kInvalidValue;
    }
    if (m <= 0 || n <= 0 || m > (std::numeric_limits<int64_t>::max() / n))
    {
        return Status::kInvalidValue;
    }
    int64_t count = m * n;
    int blocks = static_cast<int>((count + CUDA_THREADS - 1) / CUDA_THREADS);
    LeftDiagMulKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(diag, B, C, m, count);
    return StatusFromCuda(cudaGetLastError());
}

Status
ComplexMatrixOps::DiagLeftMatMul(const Complex* diag, const Complex* B, Complex* C, int64_t n)
{
    return DiagLeftMatMul(diag, B, C, n, n);
}

Status ComplexMatrixOps::DiagRightMatMul(const Complex* A,
                                         const Complex* diag,
                                         Complex* C,
                                         int64_t m,
                                         int64_t n)
{
    if (!A || !diag || !C)
    {
        return Status::kInvalidValue;
    }
    if (m <= 0 || n <= 0 || m > (std::numeric_limits<int64_t>::max() / n))
    {
        return Status::kInvalidValue;
    }
    int64_t count = m * n;
    int blocks = static_cast<int>((count + CUDA_THREADS - 1) / CUDA_THREADS);
    RightDiagMulKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(A, diag, C, m, count);
    return StatusFromCuda(cudaGetLastError());
}

Status
ComplexMatrixOps::DiagRightMatMul(const Complex* A, const Complex* diag, Complex* C, int64_t n)
{
    return DiagRightMatMul(A, diag, C, n, n);
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
    int blocks = static_cast<int>((count + CUDA_THREADS - 1) / CUDA_THREADS);
    AddKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(A, B, C, count);
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
    int blocks = static_cast<int>((count + CUDA_THREADS - 1) / CUDA_THREADS);
    SubKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(A, B, C, count);
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


    cleanup();
    return Status::kSuccess;
}

Status ComplexMatrixOps::Solve(const Complex* A, const Complex* B, Complex* X, int64_t n)
{
    return Solve(A, B, X, n, n);
}

Status MakeIdentity(Complex* A, int64_t n, cudaStream_t stream)
{
    if (!A || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int blocks = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    IdentityKernel<<<blocks, CUDA_THREADS, 0, stream>>>(A, n);
    return StatusFromCuda(cudaGetLastError());
}

Status FillDiagonalFromVector(const Complex* diag, int64_t n, Complex* out, cudaStream_t stream)
{
    if (!diag || !out || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int blocks = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    DiagFromVectorKernel<<<blocks, CUDA_THREADS, 0, stream>>>(diag, out, n);
    return StatusFromCuda(cudaGetLastError());
}

Status FillDiagonalConstant(int64_t n, Complex value, Complex* out, cudaStream_t stream)
{
    if (!out || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int blocks = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    ConstantDiagKernel<<<blocks, CUDA_THREADS, 0, stream>>>(out, n, value);
    return StatusFromCuda(cudaGetLastError());
}

Status AddMatrix(ComplexMatrixOps& ops, const Complex* A, const Complex* B, Complex* C, int64_t n)
{
    if (!A || !B || !C || n <= 0)
    {
        return Status::kInvalidValue;
    }
    return ops.MatAdd(A, B, C, n);
}

Status SubMatrix(ComplexMatrixOps& ops, const Complex* A, const Complex* B, Complex* C, int64_t n)
{
    if (!A || !B || !C || n <= 0)
    {
        return Status::kInvalidValue;
    }
    return ops.MatSub(A, B, C, n);
}

Status AddInPlace(ComplexMatrixOps& ops, Complex* A, const Complex* B, int64_t n)
{
    if (!A || !B || n <= 0)
    {
        return Status::kInvalidValue;
    }
    return ops.MatAdd(A, B, A, n);
}

Status SubInPlace(ComplexMatrixOps& ops, Complex* A, const Complex* B, int64_t n)
{
    if (!A || !B || n <= 0)
    {
        return Status::kInvalidValue;
    }
    return ops.MatSub(A, B, A, n);
}

Status MatMul(ComplexMatrixOps& ops, const Complex* A, const Complex* B, Complex* C, int64_t n)
{
    if (!A || !B || !C || n <= 0)
    {
        return Status::kInvalidValue;
    }
    return ops.MatMul(A, B, C, n);
}

Status
AddDiagonalInPlace(Complex* A, const Complex* diag, int64_t n, Complex scale, cudaStream_t stream)
{
    if (!A || !diag || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int blocks = static_cast<int>((n + CUDA_THREADS - 1) / CUDA_THREADS);
    AddDiagonalKernel<<<blocks, CUDA_THREADS, 0, stream>>>(A, diag, n, scale);
    return StatusFromCuda(cudaGetLastError());
}

Status AddDiagonalConstantInPlace(Complex* A, int64_t n, Complex value, cudaStream_t stream)
{
    if (!A || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int blocks = static_cast<int>((n + CUDA_THREADS - 1) / CUDA_THREADS);
    AddDiagonalConstantKernel<<<blocks, CUDA_THREADS, 0, stream>>>(A, n, value);
    return StatusFromCuda(cudaGetLastError());
}

Status CopyDiagonalBlock(const Complex* diag,
                         int64_t n,
                         Complex* dst,
                         int64_t dst_n,
                         int64_t row_offset,
                         int64_t col_offset,
                         Complex scale,
                         cudaStream_t stream)
{
    if (!diag || !dst || n <= 0 || dst_n <= 0)
    {
        return Status::kInvalidValue;
    }
    int blocks = static_cast<int>((n + CUDA_THREADS - 1) / CUDA_THREADS);
    CopyDiagonalBlockKernel<<<blocks, CUDA_THREADS, 0, stream>>>(
        diag, n, dst, dst_n, row_offset, col_offset, scale);
    return StatusFromCuda(cudaGetLastError());
}

Status MultiplyDiagonalVectors(const Complex* a,
                               const Complex* b,
                               Complex* out,
                               int64_t n,
                               cudaStream_t stream)
{
    if (!a || !b || !out || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int blocks = static_cast<int>((n + CUDA_THREADS - 1) / CUDA_THREADS);
    ElementwiseMulKernel<<<blocks, CUDA_THREADS, 0, stream>>>(a, b, out, n);
    return StatusFromCuda(cudaGetLastError());
}

Status ScaleMatrix(const Complex* A, Complex* C, int64_t n, Complex scalar, cudaStream_t stream)
{
    if (!A || !C || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t count = n * n;
    int blocks = static_cast<int>((count + CUDA_THREADS - 1) / CUDA_THREADS);
    ScaleKernel<<<blocks, CUDA_THREADS, 0, stream>>>(A, C, count, scalar);
    return StatusFromCuda(cudaGetLastError());
}

Status ScaleInPlace(Complex* A, int64_t n, Complex scalar, cudaStream_t stream)
{
    return ScaleMatrix(A, A, n, scalar, stream);
}

Status CopyMatrix(const Complex* src, Complex* dst, int64_t n, cudaStream_t stream)
{
    if (!src || !dst || n <= 0)
    {
        return Status::kInvalidValue;
    }
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
    return StatusFromCuda(
        cudaMemcpyAsync(dst, src, count * sizeof(Complex), cudaMemcpyDeviceToDevice, stream));
}

Status CopyBlock(const Complex* src,
                 int64_t n,
                 Complex* dst,
                 int64_t dst_n,
                 int64_t row_offset,
                 int64_t col_offset,
                 Complex scale,
                 cudaStream_t stream)
{
    if (!src || !dst || n <= 0 || dst_n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int blocks = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    CopyBlockKernel<<<blocks, CUDA_THREADS, 0, stream>>>(
        src, n, dst, dst_n, row_offset, col_offset, scale);
    return StatusFromCuda(cudaGetLastError());
}

Status CopyBlockDirect(const Complex* src,
                       int64_t n,
                       Complex* dst,
                       int64_t dst_n,
                       int64_t row_offset,
                       int64_t col_offset,
                       cudaStream_t stream)
{
    if (!src || !dst || n <= 0 || dst_n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int blocks = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    CopyBlockDirectKernel<<<blocks, CUDA_THREADS, 0, stream>>>(
        src, n, dst, dst_n, row_offset, col_offset);
    return StatusFromCuda(cudaGetLastError());
}

Status CopyBlockNegated(const Complex* src,
                        int64_t n,
                        Complex* dst,
                        int64_t dst_n,
                        int64_t row_offset,
                        int64_t col_offset,
                        cudaStream_t stream)
{
    if (!src || !dst || n <= 0 || dst_n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int blocks = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    CopyBlockNegatedKernel<<<blocks, CUDA_THREADS, 0, stream>>>(
        src, n, dst, dst_n, row_offset, col_offset);
    return StatusFromCuda(cudaGetLastError());
}

Status ComputeMatrixNorm1(const Complex* A, int64_t n, double* out_norm, cudaStream_t stream)
{
    if (!A || !out_norm || n <= 0)
    {
        return Status::kInvalidValue;
    }

    if (n > std::numeric_limits<int>::max())
    {
        return Status::kInvalidValue;
    }

    const size_t count = static_cast<size_t>(n);
    const int block = kReduceBlockSize;
    const int grid = static_cast<int>(count);
    const size_t temp_count = (count + block - 1) / block;

    DeviceRealVector col_sums;
    DeviceRealVector temp;
    CHECK(col_sums.Allocate(count));
    CHECK(temp.Allocate(temp_count));

    ColumnSumKernel<<<grid, block, 0, stream>>>(A, n, col_sums.get());
    CHECK(StatusFromCuda(cudaGetLastError()));

    size_t current = count;
    Real* input = col_sums.get();
    Real* output = temp.get();

    while (current > 1)
    {
        const size_t blocks = (current + block - 1) / block;
        MaxReduceKernel<<<static_cast<int>(blocks), block, 0, stream>>>(
            input, static_cast<int64_t>(current), output);
        CHECK(StatusFromCuda(cudaGetLastError()));
        current = blocks;
        Real* swap = input;
        input = output;
        output = swap;
    }

    Real host_value = 0.0;
    CHECK(StatusFromCuda(
        cudaMemcpyAsync(&host_value, input, sizeof(Real), cudaMemcpyDeviceToHost, stream)));
    CHECK(StatusFromCuda(cudaStreamSynchronize(stream)));

    *out_norm = static_cast<double>(host_value);
    return Status::kSuccess;
}
