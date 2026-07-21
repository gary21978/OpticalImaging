#pragma once

#include <cstdint>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "common_utils.h"

class ComplexMatrixOps
{
  public:
    ComplexMatrixOps();
    explicit ComplexMatrixOps(cudaStream_t stream);
    ~ComplexMatrixOps();

    ComplexMatrixOps(const ComplexMatrixOps&) = delete;
    ComplexMatrixOps& operator=(const ComplexMatrixOps&) = delete;

    Status SetStream(cudaStream_t stream);

    Status MatMul(const Complex* A,
                  const Complex* B,
                  Complex* C,
                  int64_t m,
                  int64_t n,
                  int64_t k,
                  cublasOperation_t transA = CUBLAS_OP_N,
                  cublasOperation_t transB = CUBLAS_OP_N);

    Status MatMul(const Complex* A,
                  const Complex* B,
                  Complex* C,
                  int64_t n,
                  cublasOperation_t transA = CUBLAS_OP_N,
                  cublasOperation_t transB = CUBLAS_OP_N);

    Status DiagLeftMatMul(const Complex* diag, const Complex* B, Complex* C, int64_t m, int64_t n);
    Status DiagLeftMatMul(const Complex* diag, const Complex* B, Complex* C, int64_t n);

    Status DiagRightMatMul(const Complex* A, const Complex* diag, Complex* C, int64_t m, int64_t n);

    Status DiagRightMatMul(const Complex* A, const Complex* diag, Complex* C, int64_t n);

    Status MatAdd(const Complex* A, const Complex* B, Complex* C, int64_t m, int64_t n);
    Status MatAdd(const Complex* A, const Complex* B, Complex* C, int64_t n);

    Status MatSub(const Complex* A, const Complex* B, Complex* C, int64_t m, int64_t n);
    Status MatSub(const Complex* A, const Complex* B, Complex* C, int64_t n);

    Status Solve(const Complex* A,
                 const Complex* B,
                 Complex* X,
                 int64_t n,
                 int64_t nrhs,
                 cublasOperation_t trans = CUBLAS_OP_N);
    Status Solve(const Complex* A, const Complex* B, Complex* X, int64_t n);

  private:
    cublasHandle_t cublas_handle_ = nullptr;
    cusolverDnHandle_t cusolver_handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

Status MakeIdentity(Complex* A, int64_t n, cudaStream_t stream = nullptr);
Status
FillDiagonalFromVector(const Complex* diag, int64_t n, Complex* out, cudaStream_t stream = nullptr);
Status FillDiagonalConstant(int64_t n, Complex value, Complex* out, cudaStream_t stream = nullptr);
Status AddMatrix(ComplexMatrixOps& ops, const Complex* A, const Complex* B, Complex* C, int64_t n);
Status SubMatrix(ComplexMatrixOps& ops, const Complex* A, const Complex* B, Complex* C, int64_t n);
Status AddInPlace(ComplexMatrixOps& ops, Complex* A, const Complex* B, int64_t n);
Status SubInPlace(ComplexMatrixOps& ops, Complex* A, const Complex* B, int64_t n);
Status MatMul(ComplexMatrixOps& ops, const Complex* A, const Complex* B, Complex* C, int64_t n);
Status AddDiagonalInPlace(Complex* A,
                          const Complex* diag,
                          int64_t n,
                          Complex scale,
                          cudaStream_t stream = nullptr);
Status
AddDiagonalConstantInPlace(Complex* A, int64_t n, Complex value, cudaStream_t stream = nullptr);
Status CopyDiagonalBlock(const Complex* diag,
                         int64_t n,
                         Complex* dst,
                         int64_t dst_n,
                         int64_t row_offset,
                         int64_t col_offset,
                         Complex scale,
                         cudaStream_t stream = nullptr);
Status MultiplyDiagonalVectors(const Complex* a,
                               const Complex* b,
                               Complex* out,
                               int64_t n,
                               cudaStream_t stream = nullptr);
Status
ScaleMatrix(const Complex* A, Complex* C, int64_t n, Complex scalar, cudaStream_t stream = nullptr);
Status ScaleInPlace(Complex* A, int64_t n, Complex scalar, cudaStream_t stream = nullptr);
Status TransposeMatrix(const Complex* src, Complex* dst, int64_t n, cudaStream_t stream = nullptr);
Status CopyMatrix(const Complex* src, Complex* dst, int64_t n, cudaStream_t stream = nullptr);
Status CopyBlock(const Complex* src,
                 int64_t n,
                 Complex* dst,
                 int64_t dst_n,
                 int64_t row_offset,
                 int64_t col_offset,
                 Complex scale,
                 cudaStream_t stream = nullptr);

// Helper functions for CopyBlock with scale of 1.0 or -1.0
Status CopyBlockDirect(const Complex* src,
                       int64_t n,
                       Complex* dst,
                       int64_t dst_n,
                       int64_t row_offset,
                       int64_t col_offset,
                       cudaStream_t stream = nullptr);

Status CopyBlockNegated(const Complex* src,
                        int64_t n,
                        Complex* dst,
                        int64_t dst_n,
                        int64_t row_offset,
                        int64_t col_offset,
                        cudaStream_t stream = nullptr);

Status ScaleAddBlockInPlace(Complex* dst,
                            int64_t dst_n,
                            const Complex* src,
                            int64_t n,
                            int64_t row_offset,
                            int64_t col_offset,
                            Complex alpha,
                            cudaStream_t stream = nullptr);

Status ScaleAddInPlace(Complex* A,
                       const Complex* B,
                       int64_t n,
                       Complex alpha,
                       cudaStream_t stream = nullptr);

Status ElementwiseInverse(const Complex* input,
                          Complex* output,
                          int64_t n,
                          cudaStream_t stream = nullptr);

Status
ComputeMatrixNorm1(const Complex* A, int64_t n, double* out_norm, cudaStream_t stream = nullptr);

Status ComputeMatrixNorm1WithDiagOffset(const Complex* A,
                                        int64_t n,
                                        Complex diag_offset,
                                        double* out_norm,
                                        cudaStream_t stream = nullptr);
