#pragma once

#include <cmath>
#include <cstdio>
#include <cuComplex.h>
#include <cuda_runtime.h>
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

constexpr Real kZero = static_cast<Real>(0.0);
constexpr Real kOne = static_cast<Real>(1.0);
constexpr Real kTwo = static_cast<Real>(2.0);
constexpr Real kHalf = static_cast<Real>(0.5);
constexpr Real kNegOne = static_cast<Real>(-1.0);

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

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

HOST_DEVICE inline Complex ComplexMul(Complex a, Complex b)
{
    const Real ar = complex_real(a);
    const Real ai = complex_imag(a);
    const Real br = complex_real(b);
    const Real bi = complex_imag(b);
    return make_complex(ar * br - ai * bi, ar * bi + ai * br);
}

HOST_DEVICE inline Complex ComplexAdd(Complex a, Complex b)
{
    return complex_add(a, b);
}

HOST_DEVICE inline Complex ComplexSub(Complex a, Complex b)
{
    return complex_sub(a, b);
}

HOST_DEVICE inline Complex ComplexConj(Complex a)
{
    return make_complex(complex_real(a), -complex_imag(a));
}

HOST_DEVICE inline Real ComplexAbs(Complex value)
{
    const Real real = complex_real(value);
    const Real imag = complex_imag(value);
#if USE_DOUBLE_PRECISION
    return sqrt(real * real + imag * imag);
#else
    return sqrtf(real * real + imag * imag);
#endif
}

HOST_DEVICE inline Complex ComplexSqrt(Complex z)
{
    const Real a = complex_real(z);
    const Real b = complex_imag(z);
    if (b == kZero && a >= kZero)
    {
#if USE_DOUBLE_PRECISION
        return make_complex(sqrt(a), kZero);
#else
        return make_complex(sqrtf(a), kZero);
#endif
    }
    const Real r = ComplexAbs(z);
#if USE_DOUBLE_PRECISION
    const Real t = sqrt((r + a) / kTwo);
    Real u = sqrt((r - a) / kTwo);
#else
    const Real t = sqrtf((r + a) / kTwo);
    Real u = sqrtf((r - a) / kTwo);
#endif
    if (b < kZero)
    {
        u = -u;
    }
    return make_complex(t, u);
}

HOST_DEVICE inline Complex ComplexReciprocal(Complex z)
{
    const Real ar = complex_real(z);
    const Real ai = complex_imag(z);
    const Real denom = ar * ar + ai * ai;
    if (denom == kZero)
    {
        return make_complex(kZero, kZero);
    }
    return make_complex(ar / denom, -ai / denom);
}

HOST_DEVICE inline Complex ComplexExpI(Complex z)
{
    const Real a = complex_real(z);
    const Real b = complex_imag(z);
#if USE_DOUBLE_PRECISION
    const Real exp_r = exp(-b);
    const Real c = cos(a);
    const Real s = sin(a);
#else
    const Real exp_r = expf(-b);
    const Real c = cosf(a);
    const Real s = sinf(a);
#endif
    return make_complex(exp_r * c, exp_r * s);
}

#undef HOST_DEVICE

#ifndef SAFEFREE
#define SAFEFREE(ptr)                                                                              \
    do                                                                                             \
    {                                                                                              \
        if (ptr)                                                                                   \
        {                                                                                          \
            cudaFree(ptr);                                                                         \
            (ptr) = nullptr;                                                                       \
        }                                                                                          \
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
        Status status_ = CheckResult((call), __FILE__, __LINE__);                                  \
        if (status_ != Status::kSuccess)                                                           \
        {                                                                                          \
            return status_;                                                                        \
        }                                                                                          \
    } while (false)
