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
#define ComplexAdd cuCadd
#define ComplexSub cuCsub
#define ComplexMul cuCmul
#define ComplexDiv cuCdiv
#define ComplexAbs cuCabs
#define ComplexReal cuCreal
#define ComplexImag cuCimag
#define ComplexConj cuConj
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
#define ComplexAdd cuCaddf
#define ComplexSub cuCsubf
#define ComplexMul cuCmulf
#define ComplexDiv cuCdivf
#define ComplexAbs cuCabsf
#define ComplexReal cuCrealf
#define ComplexImag cuCimagf
#define ComplexConj cuConjf
#define RealSqrt sqrtf
#define RealAbs fabsf
#endif

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
    const Real real = ComplexReal(value);
    const Real imag = ComplexImag(value);
    return isnan(real) || isnan(imag);
}

HOST_DEVICE inline Complex operator+(Complex a, Complex b) { return ComplexAdd(a, b); }

HOST_DEVICE inline Complex operator-(Complex a, Complex b) { return ComplexSub(a, b); }

HOST_DEVICE inline Complex operator-(Complex a)
{
    return make_complex(-ComplexReal(a), -ComplexImag(a));
}

HOST_DEVICE inline Complex operator*(Complex a, Complex b) { return ComplexMul(a, b); }

HOST_DEVICE inline Complex operator/(Complex a, Complex b) { return ComplexDiv(a, b); }
/*
HOST_DEVICE inline Complex ComplexSqrt(Complex z)
{
    const Real a = ComplexReal(z);
    const Real b = ComplexImag(z);

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
}*/

// branchless implementation of complex square root

HOST_DEVICE inline Complex ComplexSqrt(Complex z)
{
    Real a = ComplexReal(z);
    Real b = ComplexImag(z);
    Real r = ComplexAbs(z);
    Real t = RealSqrt(Real(0.5) * (r + RealAbs(a)));
    Real inv_t = (t > Real(0)) ? Real(1) / t : Real(0);
    Real u = Real(0.5) * b * inv_t;
    // sign masks
    Real sign_a = Real(a < Real(0));   // 0 or 1
    Real sign_b = copysign(Real(1), b);// 1 or -1
    // a >= 0
    Real re_pos = t;
    Real im_pos = u;
    // a < 0
    Real re_neg = RealAbs(u);
    Real im_neg = sign_b * t;
    // select
    Real re = re_pos + sign_a * (re_neg - re_pos);
    Real im = im_pos + sign_a * (im_neg - im_pos);
    return make_complex(re, im);
}

HOST_DEVICE inline Complex ComplexSqrtImPos(Complex z)
{
    Complex w = ComplexSqrt(z);
    Real re = ComplexReal(w);
    Real im = ComplexImag(w);
    return make_complex(re, RealAbs(im));
}

HOST_DEVICE inline Complex ComplexReciprocal(Complex z)
{
    const Real ar = ComplexReal(z);
    const Real ai = ComplexImag(z);
    const Real denom = ar * ar + ai * ai;
    if (denom == 0.0)
    {
        return make_complex(0.0, 0.0);
    }
    return make_complex(ar / denom, -ai / denom);
}

HOST_DEVICE inline Complex ComplexExpI(Complex z)
{
    const Real a = ComplexReal(z);
    const Real b = ComplexImag(z);
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
