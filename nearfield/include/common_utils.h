#pragma once
#include <cmath>
#include <cstdio>
#include <cuComplex.h>
#include <cuda_runtime.h>


#ifndef USE_DOUBLE_PRECISION
#define USE_DOUBLE_PRECISION 1
#endif

#ifndef CUDA_THREADS
#define CUDA_THREADS 256
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

__host__ __device__ inline bool isnan(Complex value)
{
    const Real real = ComplexReal(value);
    const Real imag = ComplexImag(value);
    return isnan(real) || isnan(imag);
}

__host__ __device__ inline Complex operator+(Complex a, Complex b) { return ComplexAdd(a, b); }

__host__ __device__ inline Complex operator-(Complex a, Complex b) { return ComplexSub(a, b); }

__host__ __device__ inline Complex operator-(Complex a)
{
    return make_complex(-ComplexReal(a), -ComplexImag(a));
}

__host__ __device__ inline Complex operator*(Complex a, Complex b) { return ComplexMul(a, b); }

__host__ __device__ inline Complex operator/(Complex a, Complex b) { return ComplexDiv(a, b); }

// branchless implementation of complex square root

__host__ __device__ inline Complex ComplexSqrt(Complex z)
{
    Real a = ComplexReal(z);
    Real b = ComplexImag(z);
    Real r = ComplexAbs(z);
    Real t = RealSqrt(Real(0.5) * (r + RealAbs(a)));
    Real inv_t = (t > Real(0)) ? Real(1) / t : Real(0);
    Real u = Real(0.5) * b * inv_t;
    // sign masks
    Real sign_a = Real(a < Real(0));    // 0 or 1
    Real sign_b = copysign(Real(1), b); // 1 or -1
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

__device__ inline Complex ComplexSqrtImPos(Complex z)
{
    Complex w = ComplexSqrt(z);
    Real re = ComplexReal(w);
    Real im = ComplexImag(w);
    return make_complex(re, RealAbs(im));
}

__device__ inline Complex ComplexSqrt_RCWA(Complex z)
{
    Real x = ComplexReal(z);
    Real y = ComplexImag(z);
    Real r = ComplexAbs(z);
    Real p = RealSqrt(Real(0.5) * (r + RealAbs(x)));
    Real q = (p > Real(0)) ? Real(0.5) * y / p : Real(0);
    Real s = Real(x >= Real(0)) * (RealAbs(q) - p);
    Real u = q - copysign(s, q);
    Real v = p + s;
    return make_complex(u, v);
}

__device__ inline Complex ComplexReciprocal(Complex z)
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

__device__ inline Complex ComplexExpI(Complex z)
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

// ============================================================================
// 2x2 Complex Matrix Structure
// ============================================================================
/**
 * @brief Structure representing a 2x2 complex matrix
 * 
 * Storage layout (column-major order, CUDA convention):
 *   [m00 m01]   stored as   {m00, m10, m01, m11}
 *   [m10 m11]
 * 
 * This provides a cleaner interface for 2x2 matrix operations on GPU.
 */
struct Complex2x2
{
    Complex m00, m10;  // First column
    Complex m01, m11;  // Second column

    // Constructor
    __host__ __device__ Complex2x2()
        : m00(make_complex(0.0, 0.0)), m10(make_complex(0.0, 0.0)),
          m01(make_complex(0.0, 0.0)), m11(make_complex(0.0, 0.0))
    {
    }

    // Constructor with values
    __host__ __device__ Complex2x2(Complex a00, Complex a10, Complex a01, Complex a11)
        : m00(a00), m10(a10), m01(a01), m11(a11)
    {
    }

    // Access as pointer to underlying data (column-major)
    __device__ inline Complex* data() { return &m00; }
    __device__ inline const Complex* data() const { return &m00; }
};

// ============================================================================
// 2x2 Complex Matrix Device Functions
// ============================================================================

/**
 * @brief Add two 2x2 complex matrices: C = A + B
 * @param A First 2x2 matrix (left operand)
 * @param B Second 2x2 matrix (right operand)
 * @return Result matrix C = A + B
 */
__device__ inline Complex2x2 Mat2x2Add(const Complex2x2& A, const Complex2x2& B)
{
    return Complex2x2(
        ComplexAdd(A.m00, B.m00),
        ComplexAdd(A.m10, B.m10),
        ComplexAdd(A.m01, B.m01),
        ComplexAdd(A.m11, B.m11)
    );
}

/**
 * @brief Subtract two 2x2 complex matrices: C = A - B
 * @param A First 2x2 matrix (left operand)
 * @param B Second 2x2 matrix (right operand)
 * @return Result matrix C = A - B
 */
__device__ inline Complex2x2 Mat2x2Sub(const Complex2x2& A, const Complex2x2& B)
{
    return Complex2x2(
        ComplexSub(A.m00, B.m00),
        ComplexSub(A.m10, B.m10),
        ComplexSub(A.m01, B.m01),
        ComplexSub(A.m11, B.m11)
    );
}

/**
 * @brief Multiply two 2x2 complex matrices: C = A * B
 * @param A First 2x2 matrix (left operand)
 * @param B Second 2x2 matrix (right operand)
 * @return Result matrix C = A * B
 * 
 * Computation:
 * c00 = a00*b00 + a01*b10
 * c10 = a10*b00 + a11*b10
 * c01 = a00*b01 + a01*b11
 * c11 = a10*b01 + a11*b11
 */
__device__ inline Complex2x2 Mat2x2Mul(const Complex2x2& A, const Complex2x2& B)
{
    Complex c00 = ComplexAdd(ComplexMul(A.m00, B.m00), ComplexMul(A.m01, B.m10));
    Complex c10 = ComplexAdd(ComplexMul(A.m10, B.m00), ComplexMul(A.m11, B.m10));
    Complex c01 = ComplexAdd(ComplexMul(A.m00, B.m01), ComplexMul(A.m01, B.m11));
    Complex c11 = ComplexAdd(ComplexMul(A.m10, B.m01), ComplexMul(A.m11, B.m11));
    
    return Complex2x2(c00, c10, c01, c11);
}

/**
 * @brief Scale a 2x2 complex matrix by a complex scalar: C = scalar * A
 * @param A Input 2x2 matrix
 * @param scalar Complex scaling factor
 * @return Scaled matrix
 */
__device__ inline Complex2x2 Mat2x2Scale(const Complex2x2& A, Complex scalar)
{
    return Complex2x2(ComplexMul(A.m00, scalar),
                      ComplexMul(A.m10, scalar),
                      ComplexMul(A.m01, scalar),
                      ComplexMul(A.m11, scalar));
}

/**
 * @brief Invert a 2x2 complex matrix: inv_A = A^(-1)
 * @param A Input 2x2 matrix to invert
 * @param success Output boolean indicating if inversion succeeded
 * @return Inverse matrix A^(-1). On failure (singular matrix), returns zero matrix.
 *         Check success flag to determine if operation succeeded.
 * 
 * For matrix:
 *   [a b]
 *   [c d]
 * 
 * Inverse is:
 *   1/(ad-bc) * [d  -b]
 *               [-c  a]
 */
__device__ inline Complex2x2 Mat2x2Inv(const Complex2x2& A, bool& success)
{
    // Compute determinant: det = a00*a11 - a01*a10
    Complex det = ComplexSub(
        ComplexMul(A.m00, A.m11),
        ComplexMul(A.m01, A.m10)
    );
    
    // Check if determinant is close to zero (singular matrix)
    Real det_magnitude = ComplexAbs(det);
    const Real epsilon = 1e-15;
    
    success = (det_magnitude >= epsilon);
    
    if (!success)
    {
        return Complex2x2();  // Return zero matrix on failure
    }
    
    // Compute 1/det
    Complex inv_det = ComplexDiv(make_complex(1.0, 0.0), det);
    
    // Compute inverse: 1/det * [a11 -a01; -a10 a00]
    Complex inv_m00 = ComplexMul(A.m11, inv_det);
    Complex inv_m10 = ComplexMul(make_complex(-1.0, 0.0), ComplexMul(A.m10, inv_det));
    Complex inv_m01 = ComplexMul(make_complex(-1.0, 0.0), ComplexMul(A.m01, inv_det));
    Complex inv_m11 = ComplexMul(A.m00, inv_det);
    
    return Complex2x2(inv_m00, inv_m10, inv_m01, inv_m11);
}

