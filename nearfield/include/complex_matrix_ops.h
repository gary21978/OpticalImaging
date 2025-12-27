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

    Status MatAdd(const Complex* A, const Complex* B, Complex* C, int64_t m, int64_t n);
    Status MatAdd(const Complex* A, const Complex* B, Complex* C, int64_t n);

    Status MatSub(const Complex* A, const Complex* B, Complex* C, int64_t m, int64_t n);
    Status MatSub(const Complex* A, const Complex* B, Complex* C, int64_t n);

    Status Solve(const Complex* A, const Complex* B, Complex* X, int64_t n, int64_t nrhs);
    Status Solve(const Complex* A, const Complex* B, Complex* X, int64_t n);

  private:
    cublasHandle_t cublas_handle_ = nullptr;
    cusolverDnHandle_t cusolver_handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
};
