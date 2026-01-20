#pragma once
#include <cstdio>
#include <cmath>
#include <cuComplex.h>
#include <cuda_runtime.h>

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
#define RealSqrt sqrt
#define RealAbs abs
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
#define RealSqrt sqrtf
#define RealAbs fabsf
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

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

HOST_DEVICE inline bool isnan(Complex value)
{
    const Real real = complex_real(value);
    const Real imag = complex_imag(value);
    return isnan(real) || isnan(imag);
}

HOST_DEVICE inline Complex ComplexMul(Complex a, Complex b)
{
    const Real ar = complex_real(a);
    const Real ai = complex_imag(a);
    const Real br = complex_real(b);
    const Real bi = complex_imag(b);
    return make_complex(ar * br - ai * bi, ar * bi + ai * br);
}

HOST_DEVICE inline Complex ComplexAdd(Complex a, Complex b) { return complex_add(a, b); }

HOST_DEVICE inline Complex ComplexSub(Complex a, Complex b) { return complex_sub(a, b); }

HOST_DEVICE inline Complex operator+(Complex a, Complex b) { return ComplexAdd(a, b); }

HOST_DEVICE inline Complex operator-(Complex a, Complex b) { return ComplexSub(a, b); }

HOST_DEVICE inline Complex operator-(Complex a)
{
    return make_complex(-complex_real(a), -complex_imag(a));
}

HOST_DEVICE inline Complex operator*(Complex a, Complex b) { return ComplexMul(a, b); }

HOST_DEVICE inline Complex ComplexConj(Complex a)
{
    return make_complex(complex_real(a), -complex_imag(a));
}

HOST_DEVICE inline Real ComplexAbs(Complex value)
{
    const Real real = complex_real(value);
    const Real imag = complex_imag(value);
    return RealSqrt(real * real + imag * imag);
}

HOST_DEVICE inline Complex ComplexSqrt(Complex z)
{
    const Real a = complex_real(z);
    const Real b = complex_imag(z);

    if (b == 0.0)
    {
        if (a >= 0.0)
        {
            return make_complex(RealSqrt(a), 0.0);
        }
        else
        {
            return make_complex(0.0, RealSqrt(-a));
        }
    }

    if (a == 0.0)
    {
        Real root = RealSqrt(0.5 * RealAbs(b));
        return b > 0.0 ? make_complex(root, root) : make_complex(root, -root);
    }

    Real r = ComplexAbs(z);
    Real t = RealSqrt(0.5 * (r + RealAbs(a)));
    Real u = 0.5 * b / t;

    if (a < 0.0)
    {
        if (b >= 0.0)
        {
            return make_complex(u, t);
        }
        else
        {
            return make_complex(RealAbs(u), -t);
        }
    }
    return make_complex(t, u);
}

HOST_DEVICE inline Complex ComplexReciprocal(Complex z)
{
    const Real ar = complex_real(z);
    const Real ai = complex_imag(z);
    const Real denom = ar * ar + ai * ai;
    if (denom == 0.0)
    {
        return make_complex(0.0, 0.0);
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
