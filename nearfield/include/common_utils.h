#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <iomanip>

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

inline const char* StatusToString(Status status)
{
    switch (status)
    {
    case Status::kSuccess:
        return "success";
    case Status::kInvalidValue:
        return "invalid value";
    case Status::kCudaError:
        return "cuda error";
    case Status::kCublasError:
        return "cublas error";
    case Status::kCusolverError:
        return "cusolver error";
    default:
        return "unknown";
    }
}

#ifndef SAFEFREE
#define SAFEFREE(ptr)             \
    do                            \
    {                             \
        if (ptr)                  \
        {                         \
            cudaFree(ptr);        \
            (ptr) = nullptr;      \
        }                         \
    } while (false)
#endif

inline Status CheckResult(Status status, const char* file, int line)
{
    if (status != Status::kSuccess)
    {
        std::fprintf(stderr, "ERROR: %s(%d), %s\n", file, line, StatusToString(status));
    }
    return status;
}

inline Status CheckResult(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess)
    {
        std::fprintf(stderr, "ERROR: %s(%d), %s\n", file, line, cudaGetErrorString(error));
        return Status::kCudaError;
    }
    return Status::kSuccess;
}

#define CHECK(call)                                                                                \
    do                                                                                             \
    {                                                                                              \
        Status status_ = CheckResult((call), __FILE__, __LINE__);                                   \
        if (status_ != Status::kSuccess)                                                           \
        {                                                                                          \
            return status_;                                                                        \
        }                                                                                          \
    } while (false)
