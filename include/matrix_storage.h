#pragma once

#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "common_utils.h"

// RAM/file allocations have host addresses, not CUDA device pointers.
// They may only be accessed by the storage-aware matrix primitives below.
namespace rcwa_memory
{
struct Stats
{
    uint64_t file_bytes = 0, peak_file_bytes = 0;
    uint64_t bytes_read = 0, bytes_written = 0, gemms = 0, solves = 0;
    uint64_t cached_lu_solves = 0, rhs_slabs = 0;
    uint64_t ram_bytes = 0, peak_ram_bytes = 0;
    uint64_t ram_bytes_read = 0, ram_bytes_written = 0;
    uint64_t file_allocations = 0, ram_allocations = 0;
};
void Configure(int order_x, int order_y);
bool Enabled();
bool RamOnly();
bool IsHost(const void* ptr);
bool IsFile(const void* ptr);
bool IsRam(const void* ptr);
Stats GetStats();
int TileSize();
cudaError_t Allocate(Complex** ptr, size_t bytes);
cudaError_t Free(void* ptr);
cudaError_t MemcpyAsync(void* dst,
                        const void* src,
                        size_t bytes,
                        cudaMemcpyKind kind,
                        cudaStream_t stream = nullptr);
cudaError_t MemsetAsync(void* dst, int value, size_t bytes, cudaStream_t stream = nullptr);
Status Read(const Complex* src, Complex* host, size_t count, cudaStream_t stream = nullptr);
Status Write(Complex* dst, const Complex* host, size_t count, cudaStream_t stream = nullptr);
Status ReadRect(const Complex* src,
                int64_t ld,
                int64_t row,
                int64_t col,
                int64_t rows,
                int64_t cols,
                Complex* host,
                cudaStream_t stream = nullptr);
Status WriteRect(Complex* dst,
                 int64_t ld,
                 int64_t row,
                 int64_t col,
                 int64_t rows,
                 int64_t cols,
                 const Complex* host,
                 cudaStream_t stream = nullptr);
Status Gemm(cublasHandle_t handle,
            const Complex* A,
            int64_t lda,
            const Complex* B,
            int64_t ldb,
            Complex* C,
            int64_t ldc,
            int64_t m,
            int64_t n,
            int64_t k,
            cublasOperation_t ta,
            cublasOperation_t tb,
            Complex alpha,
            Complex beta,
            cudaStream_t stream);
Status Solve(cublasHandle_t blas,
             cusolverDnHandle_t solver,
             const Complex* A,
             const Complex* B,
             Complex* X,
             int64_t n,
             int64_t nrhs,
             cublasOperation_t trans,
             cudaStream_t stream);
enum class ElementOp
{
    Add,
    Sub,
    Scale,
    ScaleAdd,
    LeftDiag,
    RightDiag,
    Inverse
};
Status Elementwise(ElementOp op,
                   const Complex* A,
                   const Complex* B,
                   Complex* C,
                   int64_t rows,
                   int64_t cols,
                   Complex alpha,
                   cudaStream_t stream);
Status Diagonal(Complex* A,
                int64_t ld,
                int64_t row,
                int64_t col,
                int64_t n,
                const Complex* diag,
                Complex value,
                bool add,
                cudaStream_t stream);
Status Block(const Complex* src,
             int64_t n,
             Complex* dst,
             int64_t ld,
             int64_t row,
             int64_t col,
             Complex alpha,
             bool add,
             cudaStream_t stream);
Status Norm1(const Complex* A, int64_t n, Complex offset, double* norm, cudaStream_t stream);
Status Transpose(const Complex* A, Complex* C, int64_t n, cudaStream_t stream);
} // namespace rcwa_memory
