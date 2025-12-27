#pragma once

#include <cstdint>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#ifndef USE_DOUBLE_PRECISION
#define USE_DOUBLE_PRECISION 1
#endif

#if USE_DOUBLE_PRECISION
using Real = double;
using Complex = cuDoubleComplex;
#define CUBLAS_GEMM cublasZgemm
#define CUSOLVER_GETRF_BUFFER_SIZE cusolverDnZgetrf_bufferSize
#define CUSOLVER_GETRF cusolverDnZgetrf
#define CUSOLVER_GETRS cusolverDnZgetrs
#define make_complex make_cuDoubleComplex
#define complex_add cuCadd
#define complex_sub cuCsub
#define complex_real cuCreal
#define complex_imag cuCimag
#define PRINT_PRECISION std::setprecision(15)
#else
using Real = float;
using Complex = cuComplex;
#define CUBLAS_GEMM cublasCgemm
#define CUSOLVER_GETRF_BUFFER_SIZE cusolverDnCgetrf_bufferSize
#define CUSOLVER_GETRF cusolverDnCgetrf
#define CUSOLVER_GETRS cusolverDnCgetrs
#define make_complex make_cuComplex
#define complex_add cuCaddf
#define complex_sub cuCsubf
#define complex_real cuCrealf
#define complex_imag cuCimagf
#define PRINT_PRECISION std::setprecision(6)
#endif

// Matrices are column-major and stored in device memory.
enum class Status
{
    kSuccess = 0,
    kInvalidValue,
    kCudaError,
    kCublasError,
    kCusolverError,
};

const char* StatusToString(Status status);

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
