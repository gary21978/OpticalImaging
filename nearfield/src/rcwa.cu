#include "rcwa.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

#include "complex_matrix_ops.h"

namespace
{

Status StatusFromCuda(cudaError_t status)
{
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

__device__ inline Complex Multiply(Complex a, Complex b)
{
    const Real ar = complex_real(a);
    const Real ai = complex_imag(a);
    const Real br = complex_real(b);
    const Real bi = complex_imag(b);
    return make_complex(ar * br - ai * bi, ar * bi + ai * br);
}

__device__ inline Complex Conj(Complex a)
{
    return make_complex(complex_real(a), -complex_imag(a));
}

__device__ inline Real Abs(Complex a)
{
    const Real ar = complex_real(a);
    const Real ai = complex_imag(a);
#if USE_DOUBLE_PRECISION
    return sqrt(ar * ar + ai * ai);
#else
    return sqrtf(ar * ar + ai * ai);
#endif
}

__device__ inline Complex SqrtComplex(Complex z)
{
    const Real a = complex_real(z);
    const Real b = complex_imag(z);
    if (b == static_cast<Real>(0.0) && a >= static_cast<Real>(0.0))
    {
#if USE_DOUBLE_PRECISION
        return make_complex(sqrt(a), static_cast<Real>(0.0));
#else
        return make_complex(sqrtf(a), static_cast<Real>(0.0));
#endif
    }
    const Real r = Abs(z);
#if USE_DOUBLE_PRECISION
    const Real t = sqrt((r + a) / static_cast<Real>(2.0));
    Real u = sqrt((r - a) / static_cast<Real>(2.0));
#else
    const Real t = sqrtf((r + a) / static_cast<Real>(2.0));
    Real u = sqrtf((r - a) / static_cast<Real>(2.0));
#endif
    if (b < static_cast<Real>(0.0))
    {
        u = -u;
    }
    return make_complex(t, u);
}

__device__ inline Complex Reciprocal(Complex z)
{
    const Real ar = complex_real(z);
    const Real ai = complex_imag(z);
    const Real denom = ar * ar + ai * ai;
    if (denom == static_cast<Real>(0.0))
    {
        return make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0));
    }
    return make_complex(ar / denom, -ai / denom);
}

__device__ inline Complex ExpI(Complex z)
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

struct DeviceMatrix
{
    Complex* ptr = nullptr;
    size_t elements = 0;

    ~DeviceMatrix()
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    Complex* get() const { return ptr; }
    Complex* Release()
    {
        Complex* out = ptr;
        ptr = nullptr;
        elements = 0;
        return out;
    }

    Status Allocate(int64_t rows, int64_t cols)
    {
        if (rows <= 0 || cols <= 0)
        {
            return Status::kInvalidValue;
        }
        const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (count > (std::numeric_limits<size_t>::max() / sizeof(Complex)))
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
        Complex* out = nullptr;
        if (cudaMalloc(&out, count * sizeof(Complex)) != cudaSuccess)
        {
            return Status::kCudaError;
        }
        ptr = out;
        elements = count;
        return Status::kSuccess;
    }
};

struct DeviceVector
{
    Complex* ptr = nullptr;
    size_t elements = 0;

    ~DeviceVector()
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    Complex* get() const { return ptr; }

    Status Allocate(size_t count)
    {
        if (count == 0)
        {
            return Status::kInvalidValue;
        }
        if (count > (std::numeric_limits<size_t>::max() / sizeof(Complex)))
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
        Complex* out = nullptr;
        if (cudaMalloc(&out, count * sizeof(Complex)) != cudaSuccess)
        {
            return Status::kCudaError;
        }
        ptr = out;
        elements = count;
        return Status::kSuccess;
    }
};

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
            A[idx] = make_complex(static_cast<Real>(1.0), static_cast<Real>(0.0));
        }
        else
        {
            A[idx] = make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0));
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
            out[idx] = make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0));
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
            out[idx] = make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0));
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
        const Complex value = Multiply(src[col * n + row], scale);
        dst[(col + col_offset) * dst_n + (row + row_offset)] = value;
    }
}

__global__ void BuildV0iVPhaseKernel(const Complex* kx_diag,
                                     const Complex* ky_diag,
                                     const Complex* v0_11,
                                     const Complex* v0_12,
                                     const Complex* v0_21,
                                     const Complex* v0_22,
                                     Complex eps,
                                     Real omega,
                                     Real thickness,
                                     Complex* v0iv,
                                     Complex* phase_diag,
                                     int64_t n,
                                     int64_t n2)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    const Complex kx = kx_diag[idx];
    const Complex ky = ky_diag[idx];
    const Complex kx2 = Multiply(kx, kx);
    const Complex ky2 = Multiply(ky, ky);
    const Complex sum = complex_add(kx2, ky2);
    const Complex diff = complex_sub(eps, sum);
    Complex kz = SqrtComplex(diff);
    if (complex_imag(kz) < static_cast<Real>(0.0))
    {
        kz = Conj(kz);
    }

    const Complex kz_inv = Reciprocal(kz);
    const Complex kxky = Multiply(kx, ky);
    const Complex v11 = Multiply(make_complex(static_cast<Real>(-1.0), static_cast<Real>(0.0)),
                                 Multiply(kxky, kz_inv));
    const Complex v12 = complex_add(kz, Multiply(kx2, kz_inv));
    const Complex v21 = Multiply(make_complex(static_cast<Real>(-1.0), static_cast<Real>(0.0)),
                                 complex_add(kz, Multiply(ky2, kz_inv)));
    const Complex v22 = Multiply(kxky, kz_inv);

    const Complex term11 = Multiply(v0_11[idx], v11);
    const Complex term12 = Multiply(v0_12[idx], v21);
    const Complex term21 = Multiply(v0_21[idx], v11);
    const Complex term22 = Multiply(v0_22[idx], v21);
    const Complex term31 = Multiply(v0_11[idx], v12);
    const Complex term32 = Multiply(v0_12[idx], v22);
    const Complex term41 = Multiply(v0_21[idx], v12);
    const Complex term42 = Multiply(v0_22[idx], v22);

    const Complex neg_one = make_complex(static_cast<Real>(-1.0), static_cast<Real>(0.0));
    const Complex b11 = Multiply(neg_one, complex_add(term11, term12));
    const Complex b12 = Multiply(neg_one, complex_add(term31, term32));
    const Complex b21 = Multiply(neg_one, complex_add(term21, term22));
    const Complex b22 = Multiply(neg_one, complex_add(term41, term42));

    const int64_t row0 = idx;
    const int64_t row1 = idx + n;
    const int64_t col0 = idx;
    const int64_t col1 = idx + n;
    v0iv[col0 * n2 + row0] = b11;
    v0iv[col1 * n2 + row0] = b12;
    v0iv[col0 * n2 + row1] = b21;
    v0iv[col1 * n2 + row1] = b22;

    const Complex kzt = Multiply(kz, make_complex(omega * thickness, static_cast<Real>(0.0)));
    const Complex phase = ExpI(kzt);
    phase_diag[idx] = phase;
    phase_diag[idx + n] = phase;
}

__global__ void BuildV0Kernel(const Complex* kx_diag,
                              const Complex* ky_diag,
                              Complex eps,
                              Complex* v0_11,
                              Complex* v0_12,
                              Complex* v0_21,
                              Complex* v0_22,
                              int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    const Complex kx = kx_diag[idx];
    const Complex ky = ky_diag[idx];
    const Complex kx2 = Multiply(kx, kx);
    const Complex ky2 = Multiply(ky, ky);
    const Complex sum = complex_add(kx2, ky2);
    const Complex diff = complex_sub(eps, sum);
    Complex kz = SqrtComplex(diff);
    if (complex_imag(kz) < static_cast<Real>(0.0))
    {
        kz = Conj(kz);
    }

    const Complex kz_inv = Reciprocal(kz);
    const Complex kxky = Multiply(kx, ky);
    v0_11[idx] =
        Multiply(make_complex(static_cast<Real>(-1.0), static_cast<Real>(0.0)),
                 Multiply(kxky, kz_inv));
    v0_12[idx] = complex_add(kz, Multiply(kx2, kz_inv));
    v0_21[idx] =
        Multiply(make_complex(static_cast<Real>(-1.0), static_cast<Real>(0.0)),
                 complex_add(kz, Multiply(ky2, kz_inv)));
    v0_22[idx] = Multiply(kxky, kz_inv);
}

__global__ void BuildKxKyDiagKernel(Complex* kx_diag,
                                    Complex* ky_diag,
                                    int order_x,
                                    int order_y,
                                    Real kx0_norm,
                                    Real ky0_norm,
                                    Real gx_norm,
                                    Real gy_norm,
                                    int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    const int ny = 2 * order_y + 1;
    const int64_t ix = idx / ny;
    const int64_t iy = idx - ix * ny;
    const int ox = static_cast<int>(ix) - order_x;
    const int oy = static_cast<int>(iy) - order_y;
    const Real kx = kx0_norm + static_cast<Real>(ox) * gx_norm;
    const Real ky = ky0_norm + static_cast<Real>(oy) * gy_norm;
    kx_diag[idx] = make_complex(kx, static_cast<Real>(0.0));
    ky_diag[idx] = make_complex(ky, static_cast<Real>(0.0));
}

__global__ void ScaleKernel(const Complex* A, Complex* C, int64_t count, Complex scalar)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = Multiply(A[idx], scalar);
    }
}

Status MakeIdentity(int64_t n, Complex* out, cudaStream_t stream)
{
    if (!out || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    IdentityKernel<<<blocks, threads, 0, stream>>>(out, n);
    return StatusFromCuda(cudaGetLastError());
}

Status FillDiagonalFromVector(const Complex* diag, int64_t n, Complex* out, cudaStream_t stream)
{
    if (!diag || !out || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    DiagFromVectorKernel<<<blocks, threads, 0, stream>>>(diag, out, n);
    return StatusFromCuda(cudaGetLastError());
}

Status FillDiagonalConstant(int64_t n, Complex value, Complex* out, cudaStream_t stream)
{
    if (!out || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    ConstantDiagKernel<<<blocks, threads, 0, stream>>>(out, n, value);
    return StatusFromCuda(cudaGetLastError());
}

Status ScaleInPlace(Complex* A, int64_t n, Complex scalar, cudaStream_t stream)
{
    if (!A || n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t count = n * n;
    int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    ScaleKernel<<<blocks, threads, 0, stream>>>(A, A, count, scalar);
    return StatusFromCuda(cudaGetLastError());
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
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    CopyBlockKernel<<<blocks, threads, 0, stream>>>(
        src, n, dst, dst_n, row_offset, col_offset, scale);
    return StatusFromCuda(cudaGetLastError());
}

Status AddMatrix(ComplexMatrixOps& ops,
                 const Complex* A,
                 const Complex* B,
                 int64_t n,
                 DeviceMatrix* out)
{
    CHECK(out->Allocate(n, n));
    return ops.MatAdd(A, B, out->get(), n);
}

Status SubMatrix(ComplexMatrixOps& ops,
                 const Complex* A,
                 const Complex* B,
                 int64_t n,
                 DeviceMatrix* out)
{
    CHECK(out->Allocate(n, n));
    return ops.MatSub(A, B, out->get(), n);
}

Status MatMul(ComplexMatrixOps& ops,
              const Complex* A,
              const Complex* B,
              int64_t n,
              DeviceMatrix* out)
{
    CHECK(out->Allocate(n, n));
    return ops.MatMul(A, B, out->get(), n);
}

} // namespace

Rcwa::Rcwa(Real freq, int order_x, int order_y, Real Lx, Real Ly, cudaStream_t stream)
    : freq_(freq),
      omega_(static_cast<Real>(6.28318530717958647692) * freq),
      Lx_(Lx),
      Ly_(Ly),
      order_x_(std::max(order_x, 0)),
      order_y_(std::max(order_y, 0)),
      stream_(stream)
{
    const int ox = std::max(order_x_, 0);
    const int oy = std::max(order_y_, 0);
    order_N_ = static_cast<int64_t>((2 * ox + 1) * (2 * oy + 1));
    if (freq_ != static_cast<Real>(0.0) && Lx_ != static_cast<Real>(0.0) &&
        Ly_ != static_cast<Real>(0.0))
    {
        Gx_norm_ = static_cast<Real>(1.0) / (Lx_ * freq_);
        Gy_norm_ = static_cast<Real>(1.0) / (Ly_ * freq_);
    }
}

Rcwa::~Rcwa()
{
    FreeLayers();
    FreeKVectors();
}

void Rcwa::FreeLayers()
{
    for (auto& layer : layer_smatrices_)
    {
        SAFEFREE(layer.S_11);
        SAFEFREE(layer.S_12);
        SAFEFREE(layer.S_21);
        SAFEFREE(layer.S_22);
    }
    layer_smatrices_.clear();
}

void Rcwa::FreeKVectors()
{
    SAFEFREE(kx_diag_);
    SAFEFREE(ky_diag_);
    SAFEFREE(kx_mat_);
    SAFEFREE(ky_mat_);
    SAFEFREE(V0_.V_11);
    SAFEFREE(V0_.V_12);
    SAFEFREE(V0_.V_21);
    SAFEFREE(V0_.V_22);
    kvectors_ready_ = false;
}

Status Rcwa::SetIncidentAngles(Real inc_ang, Real azi_ang, Real eps_ref)
{
    return BuildKxKy(inc_ang, azi_ang, eps_ref);
}

Status Rcwa::BuildKxKy(Real inc_ang, Real azi_ang, Real eps_ref)
{
    if (order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }

    FreeKVectors();

    const Real sqrt_eps = std::sqrt(static_cast<double>(eps_ref));
    const Real kx0_norm = sqrt_eps * std::sin(inc_ang) * std::cos(azi_ang);
    const Real ky0_norm = sqrt_eps * std::sin(inc_ang) * std::sin(azi_ang);

    CHECK(StatusFromCuda(
        cudaMalloc(&kx_diag_, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&ky_diag_, static_cast<size_t>(order_N_) * sizeof(Complex))));

    CHECK(StatusFromCuda(
        cudaMalloc(&kx_mat_, static_cast<size_t>(order_N_) * order_N_ * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&ky_mat_, static_cast<size_t>(order_N_) * order_N_ * sizeof(Complex))));
    const int threads = 256;
    const int blocks = static_cast<int>((order_N_ + threads - 1) / threads);
    BuildKxKyDiagKernel<<<blocks, threads, 0, stream_>>>(kx_diag_,
                                                        ky_diag_,
                                                        order_x_,
                                                        order_y_,
                                                        kx0_norm,
                                                        ky0_norm,
                                                        Gx_norm_,
                                                        Gy_norm_,
                                                        order_N_);
    CHECK(StatusFromCuda(cudaGetLastError()));
    CHECK(FillDiagonalFromVector(kx_diag_, order_N_, kx_mat_, stream_));
    CHECK(FillDiagonalFromVector(ky_diag_, order_N_, ky_mat_, stream_));

    CHECK(BuildV0(static_cast<Real>(1.0)));

    kvectors_ready_ = true;
    return Status::kSuccess;
}

Status Rcwa::BuildV0(Real eps_ref)
{
    if (!kx_diag_ || !ky_diag_ || order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }

    SAFEFREE(V0_.V_11);
    SAFEFREE(V0_.V_12);
    SAFEFREE(V0_.V_21);
    SAFEFREE(V0_.V_22);

    CHECK(StatusFromCuda(
        cudaMalloc(&V0_.V_11, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&V0_.V_12, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&V0_.V_21, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&V0_.V_22, static_cast<size_t>(order_N_) * sizeof(Complex))));

    const int threads = 256;
    const int blocks = static_cast<int>((order_N_ + threads - 1) / threads);
    BuildV0Kernel<<<blocks, threads, 0, stream_>>>(kx_diag_,
                                                   ky_diag_,
                                                   make_complex(eps_ref, static_cast<Real>(0.0)),
                                                   V0_.V_11,
                                                   V0_.V_12,
                                                   V0_.V_21,
                                                   V0_.V_22,
                                                   order_N_);
    CHECK(StatusFromCuda(cudaGetLastError()));

    return Status::kSuccess;
}

Status Rcwa::AddLayer(Real thickness, Real eps)
{
    return AddLayer(thickness, make_complex(eps, static_cast<Real>(0.0)));
}

Status Rcwa::AddLayer(Real thickness, Complex eps)
{
    if (order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }

    DeviceMatrix E;
    CHECK(E.Allocate(order_N_, order_N_));
    CHECK(FillDiagonalConstant(order_N_, eps, E.get(), stream_));
    return AddLayer(thickness, E.get());
}

Status Rcwa::AddLayerHomogeneous(Real thickness, Real eps)
{
    return AddLayerHomogeneous(thickness, make_complex(eps, static_cast<Real>(0.0)));
}

Status Rcwa::AddLayerHomogeneous(Real thickness, Complex eps)
{
    if (order_N_ <= 0 || !kvectors_ready_ || !kx_diag_ || !ky_diag_ || !V0_.V_11 || !V0_.V_12 ||
        !V0_.V_21 || !V0_.V_22)
    {
        return Status::kInvalidValue;
    }

    const int64_t n = order_N_;
    const int64_t n2 = 2 * n;
    DeviceMatrix V0iV;
    CHECK(V0iV.Allocate(n2, n2));
    DeviceVector phase_diag;
    CHECK(phase_diag.Allocate(static_cast<size_t>(n2)));
    CHECK(StatusFromCuda(
        cudaMemsetAsync(V0iV.get(), 0, static_cast<size_t>(n2) * n2 * sizeof(Complex), stream_)));

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    BuildV0iVPhaseKernel<<<blocks, threads, 0, stream_>>>(kx_diag_,
                                                         ky_diag_,
                                                         V0_.V_11,
                                                         V0_.V_12,
                                                         V0_.V_21,
                                                         V0_.V_22,
                                                         eps,
                                                         omega_,
                                                         thickness,
                                                         V0iV.get(),
                                                         phase_diag.get(),
                                                         n,
                                                         n2);
    CHECK(StatusFromCuda(cudaGetLastError()));

    ComplexMatrixOps ops(stream_);
    CHECK(ops.SetStream(stream_));

    DeviceMatrix I;
    CHECK(I.Allocate(n2, n2));
    CHECK(MakeIdentity(n2, I.get(), stream_));

    DeviceMatrix A;
    CHECK(AddMatrix(ops, I.get(), V0iV.get(), n2, &A));

    DeviceMatrix I_minus;
    CHECK(SubMatrix(ops, I.get(), V0iV.get(), n2, &I_minus));

    DeviceMatrix B;
    CHECK(B.Allocate(n2, n2));
    CHECK(ops.DiagRightMatMul(I_minus.get(), phase_diag.get(), B.get(), n2));

    DeviceMatrix Ainv;
    CHECK(Ainv.Allocate(n2, n2));
    CHECK(ops.Solve(A.get(), I.get(), Ainv.get(), n2));

    DeviceMatrix BAi;
    CHECK(MatMul(ops, B.get(), Ainv.get(), n2, &BAi));

    DeviceMatrix BAiB;
    CHECK(MatMul(ops, BAi.get(), B.get(), n2, &BAiB));

    DeviceMatrix T;
    CHECK(SubMatrix(ops, A.get(), BAiB.get(), n2, &T));

    DeviceMatrix C1;
    CHECK(C1.Allocate(n2, n2));
    CHECK(ops.Solve(T.get(), I.get(), C1.get(), n2));
    CHECK(ScaleInPlace(C1.get(), n2, make_complex(static_cast<Real>(2.0), static_cast<Real>(0.0)),
                       stream_));

    DeviceMatrix C2;
    CHECK(MatMul(ops, C1.get(), BAi.get(), n2, &C2));
    CHECK(ScaleInPlace(
        C2.get(), n2, make_complex(static_cast<Real>(-1.0), static_cast<Real>(0.0)), stream_));

    DeviceMatrix phaseC1;
    CHECK(phaseC1.Allocate(n2, n2));
    CHECK(ops.DiagLeftMatMul(phase_diag.get(), C1.get(), phaseC1.get(), n2));

    DeviceMatrix S11;
    CHECK(AddMatrix(ops, phaseC1.get(), C2.get(), n2, &S11));

    DeviceMatrix phaseC2;
    CHECK(phaseC2.Allocate(n2, n2));
    CHECK(ops.DiagLeftMatMul(phase_diag.get(), C2.get(), phaseC2.get(), n2));

    DeviceMatrix tmp;
    CHECK(AddMatrix(ops, phaseC2.get(), C1.get(), n2, &tmp));

    DeviceMatrix S12;
    CHECK(SubMatrix(ops, tmp.get(), I.get(), n2, &S12));

    DeviceMatrix S21;
    CHECK(S21.Allocate(n2, n2));
    CHECK(CopyMatrix(S12.get(), S21.get(), n2, stream_));

    DeviceMatrix S22;
    CHECK(S22.Allocate(n2, n2));
    CHECK(CopyMatrix(S11.get(), S22.get(), n2, stream_));

    SMatrix S{};
    S.S_11 = S11.Release();
    S.S_12 = S12.Release();
    S.S_21 = S21.Release();
    S.S_22 = S22.Release();
    layer_smatrices_.push_back(S);
    return Status::kSuccess;
}

Status Rcwa::AddLayer(Real thickness, const Complex* eps_conv)
{
    if (!eps_conv || !kvectors_ready_ || order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }

    const int64_t n = order_N_;
    const int64_t n2 = 2 * n;

    int n_repeated = 0;
    Real d_block = thickness;
    if (order_x_ > 0 && order_y_ > 0)
    {
        const Real dx = Lx_ / static_cast<Real>(order_x_);
        const Real dy = Ly_ / static_cast<Real>(order_y_);
        const Real thickness_thres = static_cast<Real>(2.0) * std::min(dx, dy);
        if (thickness_thres > static_cast<Real>(0.0) && thickness > thickness_thres)
        {
            const Real ratio = thickness / thickness_thres;
            n_repeated = static_cast<int>(std::ceil(std::log2(ratio)));
            d_block = thickness / std::pow(static_cast<Real>(2.0), n_repeated);
        }
    }

    const Real k0d = omega_ * d_block;

    DeviceMatrix P;
    DeviceMatrix Q;
    CHECK(P.Allocate(n2, n2));
    CHECK(Q.Allocate(n2, n2));
    CHECK(BuildPQ(eps_conv, n, P.get(), Q.get()));

    TMatrix T{};
    CHECK(ComputeTFromPQ(P.get(), Q.get(), n2, k0d, &T, stream_));

    SMatrix S{};
    CHECK(ComputeSFromT(T, V0_, n2, &S, stream_));

    SAFEFREE(T.T_11);
    SAFEFREE(T.T_12);
    SAFEFREE(T.T_21);
    SAFEFREE(T.T_22);

    if (n_repeated > 0)
    {
        CHECK(RepeatSMatrix(&S, n_repeated, n2));
    }

    layer_smatrices_.push_back(S);
    return Status::kSuccess;
}

Status Rcwa::BuildPQ(const Complex* eps_conv, int64_t n, Complex* P, Complex* Q)
{
    if (!eps_conv || !P || !Q || n <= 0)
    {
        return Status::kInvalidValue;
    }

    ComplexMatrixOps ops(stream_);
    CHECK(ops.SetStream(stream_));

    DeviceMatrix KxKx;
    DeviceMatrix KyKy;
    DeviceMatrix KxKy;
    CHECK(KxKx.Allocate(n, n));
    CHECK(KyKy.Allocate(n, n));
    CHECK(KxKy.Allocate(n, n));
    CHECK(ops.DiagLeftMatMul(kx_diag_, kx_mat_, KxKx.get(), n));
    CHECK(ops.DiagLeftMatMul(ky_diag_, ky_mat_, KyKy.get(), n));
    CHECK(ops.DiagLeftMatMul(kx_diag_, ky_mat_, KxKy.get(), n));

    DeviceMatrix EiKx;
    DeviceMatrix EiKy;
    CHECK(EiKx.Allocate(n, n));
    CHECK(EiKy.Allocate(n, n));

    const Status solve_kx = ops.Solve(eps_conv, kx_mat_, EiKx.get(), n);
    const Status solve_ky = ops.Solve(eps_conv, ky_mat_, EiKy.get(), n);
    const bool fallback = (solve_kx != Status::kSuccess || solve_ky != Status::kSuccess);

    DeviceMatrix KxEiKy;
    DeviceMatrix KyEiKy;
    DeviceMatrix KxEiKx;
    DeviceMatrix KyEiKx;
    CHECK(KxEiKy.Allocate(n, n));
    CHECK(KyEiKy.Allocate(n, n));
    CHECK(KxEiKx.Allocate(n, n));
    CHECK(KyEiKx.Allocate(n, n));

    if (fallback)
    {
        CHECK(CopyMatrix(KxKy.get(), KxEiKy.get(), n, stream_));
        CHECK(CopyMatrix(KxKy.get(), KyEiKx.get(), n, stream_));
        CHECK(CopyMatrix(KxKx.get(), KxEiKx.get(), n, stream_));
        CHECK(CopyMatrix(KyKy.get(), KyEiKy.get(), n, stream_));
    }
    else
    {
        CHECK(ops.DiagLeftMatMul(kx_diag_, EiKy.get(), KxEiKy.get(), n));
        CHECK(ops.DiagLeftMatMul(ky_diag_, EiKx.get(), KyEiKx.get(), n));
        CHECK(ops.DiagLeftMatMul(kx_diag_, EiKx.get(), KxEiKx.get(), n));
        CHECK(ops.DiagLeftMatMul(ky_diag_, EiKy.get(), KyEiKy.get(), n));
    }

    DeviceMatrix I;
    CHECK(I.Allocate(n, n));
    CHECK(MakeIdentity(n, I.get(), stream_));

    DeviceMatrix P12;
    DeviceMatrix P21;
    DeviceMatrix Q12;
    DeviceMatrix Q21;
    CHECK(SubMatrix(ops, KyEiKy.get(), I.get(), n, &P12));
    CHECK(SubMatrix(ops, I.get(), KxEiKx.get(), n, &P21));
    CHECK(SubMatrix(ops, eps_conv, KyKy.get(), n, &Q12));
    CHECK(SubMatrix(ops, KxKx.get(), eps_conv, n, &Q21));

    const Complex one = make_complex(static_cast<Real>(1.0), static_cast<Real>(0.0));
    const Complex neg_one = make_complex(static_cast<Real>(-1.0), static_cast<Real>(0.0));

    CHECK(CopyBlock(KxEiKy.get(), n, P, 2 * n, 0, 0, one, stream_));
    CHECK(CopyBlock(P12.get(), n, P, 2 * n, 0, n, one, stream_));
    CHECK(CopyBlock(P21.get(), n, P, 2 * n, n, 0, one, stream_));
    CHECK(CopyBlock(KyEiKx.get(), n, P, 2 * n, n, n, neg_one, stream_));

    CHECK(CopyBlock(KxKy.get(), n, Q, 2 * n, 0, 0, neg_one, stream_));
    CHECK(CopyBlock(Q12.get(), n, Q, 2 * n, 0, n, one, stream_));
    CHECK(CopyBlock(Q21.get(), n, Q, 2 * n, n, 0, one, stream_));
    CHECK(CopyBlock(KxKy.get(), n, Q, 2 * n, n, n, one, stream_));

    return Status::kSuccess;
}

Status Rcwa::RepeatSMatrix(SMatrix* S, int repeats, int64_t n)
{
    if (!S || repeats <= 0 || n <= 0)
    {
        return Status::kInvalidValue;
    }
    if (!S->S_11 || !S->S_12 || !S->S_21 || !S->S_22)
    {
        return Status::kInvalidValue;
    }

    ComplexMatrixOps ops(stream_);
    CHECK(ops.SetStream(stream_));

    DeviceMatrix I;
    CHECK(I.Allocate(n, n));
    CHECK(MakeIdentity(n, I.get(), stream_));

    DeviceMatrix S12S12;
    DeviceMatrix R;
    DeviceMatrix S12S11;
    DeviceMatrix X;
    DeviceMatrix S11X;
    DeviceMatrix newS12;
    DeviceMatrix Y;
    DeviceMatrix newS11;

    for (int i = 0; i < repeats; ++i)
    {
        CHECK(MatMul(ops, S->S_12, S->S_12, n, &S12S12));
        CHECK(SubMatrix(ops, I.get(), S12S12.get(), n, &R));

        CHECK(MatMul(ops, S->S_12, S->S_11, n, &S12S11));
        CHECK(X.Allocate(n, n));
        CHECK(ops.Solve(R.get(), S12S11.get(), X.get(), n));
        CHECK(MatMul(ops, S->S_11, X.get(), n, &S11X));
        CHECK(AddMatrix(ops, S->S_12, S11X.get(), n, &newS12));

        CHECK(Y.Allocate(n, n));
        CHECK(ops.Solve(R.get(), S->S_11, Y.get(), n));
        CHECK(MatMul(ops, S->S_11, Y.get(), n, &newS11));

        CHECK(CopyMatrix(newS11.get(), S->S_11, n, stream_));
        CHECK(CopyMatrix(newS12.get(), S->S_12, n, stream_));
    }

    CHECK(CopyMatrix(S->S_12, S->S_21, n, stream_));
    CHECK(CopyMatrix(S->S_11, S->S_22, n, stream_));
    return Status::kSuccess;
}
