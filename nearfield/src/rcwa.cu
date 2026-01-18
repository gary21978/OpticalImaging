#include "rcwa.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

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

__global__ void FillDiagBlockKernel(const Complex* diag,
                                    Complex* dst,
                                    int64_t n,
                                    int64_t dst_n,
                                    int64_t row_offset,
                                    int64_t col_offset)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * n;
    if (idx < total)
    {
        int64_t row = idx % n;
        int64_t col = idx / n;
        const int64_t out_row = row + row_offset;
        const int64_t out_col = col + col_offset;
        if (row == col)
        {
            dst[out_col * dst_n + out_row] = diag[row];
        }
        else
        {
            dst[out_col * dst_n + out_row] =
                make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0));
        }
    }
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

Status FillDiagBlock(const Complex* diag,
                     int64_t n,
                     Complex* dst,
                     int64_t dst_n,
                     int64_t row_offset,
                     int64_t col_offset,
                     cudaStream_t stream)
{
    if (!diag || !dst || n <= 0 || dst_n <= 0)
    {
        return Status::kInvalidValue;
    }
    int64_t total = n * n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    FillDiagBlockKernel<<<blocks, threads, 0, stream>>>(
        diag, dst, n, dst_n, row_offset, col_offset);
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

std::complex<Real> ToStd(Complex value)
{
    return {static_cast<Real>(complex_real(value)), static_cast<Real>(complex_imag(value))};
}

Complex ToCu(const std::complex<Real>& value)
{
    return make_complex(static_cast<Real>(value.real()), static_cast<Real>(value.imag()));
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
    kx_host_.clear();
    ky_host_.clear();
    v0_11_host_.clear();
    v0_12_host_.clear();
    v0_21_host_.clear();
    v0_22_host_.clear();
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

    std::vector<Real> order_x_vals(static_cast<size_t>(2 * order_x_ + 1));
    std::vector<Real> order_y_vals(static_cast<size_t>(2 * order_y_ + 1));
    for (int i = 0; i < static_cast<int>(order_x_vals.size()); ++i)
    {
        order_x_vals[static_cast<size_t>(i)] = static_cast<Real>(i - order_x_);
    }
    for (int i = 0; i < static_cast<int>(order_y_vals.size()); ++i)
    {
        order_y_vals[static_cast<size_t>(i)] = static_cast<Real>(i - order_y_);
    }

    std::vector<Complex> kx_diag_host(static_cast<size_t>(order_N_));
    std::vector<Complex> ky_diag_host(static_cast<size_t>(order_N_));
    size_t idx = 0;
    for (size_t ix = 0; ix < order_x_vals.size(); ++ix)
    {
        for (size_t iy = 0; iy < order_y_vals.size(); ++iy)
        {
            const Real kx = kx0_norm + order_x_vals[ix] * Gx_norm_;
            const Real ky = ky0_norm + order_y_vals[iy] * Gy_norm_;
            kx_diag_host[idx] = make_complex(kx, static_cast<Real>(0.0));
            ky_diag_host[idx] = make_complex(ky, static_cast<Real>(0.0));
            ++idx;
        }
    }

    kx_host_ = kx_diag_host;
    ky_host_ = ky_diag_host;

    CHECK(StatusFromCuda(
        cudaMalloc(&kx_diag_, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&ky_diag_, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(cudaMemcpyAsync(kx_diag_,
                                         kx_diag_host.data(),
                                         static_cast<size_t>(order_N_) * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(ky_diag_,
                                         ky_diag_host.data(),
                                         static_cast<size_t>(order_N_) * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));

    CHECK(StatusFromCuda(
        cudaMalloc(&kx_mat_, static_cast<size_t>(order_N_) * order_N_ * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&ky_mat_, static_cast<size_t>(order_N_) * order_N_ * sizeof(Complex))));
    CHECK(FillDiagonalFromVector(kx_diag_, order_N_, kx_mat_, stream_));
    CHECK(FillDiagonalFromVector(ky_diag_, order_N_, ky_mat_, stream_));

    CHECK(BuildV0(kx_diag_host, ky_diag_host, static_cast<Real>(1.0)));

    kvectors_ready_ = true;
    return Status::kSuccess;
}

Status Rcwa::BuildV0(const std::vector<Complex>& kx_diag,
                     const std::vector<Complex>& ky_diag,
                     Real eps_ref)
{
    if (kx_diag.size() != static_cast<size_t>(order_N_) ||
        ky_diag.size() != static_cast<size_t>(order_N_))
    {
        return Status::kInvalidValue;
    }

    std::vector<Complex> v11(kx_diag.size());
    std::vector<Complex> v12(kx_diag.size());
    std::vector<Complex> v21(kx_diag.size());
    std::vector<Complex> v22(kx_diag.size());

    const Real eps = eps_ref;
    for (size_t i = 0; i < kx_diag.size(); ++i)
    {
        const std::complex<Real> kx(complex_real(kx_diag[i]), complex_imag(kx_diag[i]));
        const std::complex<Real> ky(complex_real(ky_diag[i]), complex_imag(ky_diag[i]));
        std::complex<Real> kz = std::sqrt(std::complex<Real>(eps) - kx * kx - ky * ky);
        if (kz.imag() < static_cast<Real>(0.0))
        {
            kz = std::conj(kz);
        }
        const Real kz_abs = std::abs(kz);
        const std::complex<Real> kz_inv =
            kz_abs == static_cast<Real>(0.0)
                ? std::complex<Real>(0.0, 0.0)
                : (std::complex<Real>(1.0, 0.0) / kz);

        const std::complex<Real> v11_val = -kx * ky * kz_inv;
        const std::complex<Real> v12_val = kz + kx * kx * kz_inv;
        const std::complex<Real> v21_val = -(kz + ky * ky * kz_inv);
        const std::complex<Real> v22_val = kx * ky * kz_inv;

        v11[i] = make_complex(static_cast<Real>(v11_val.real()),
                              static_cast<Real>(v11_val.imag()));
        v12[i] = make_complex(static_cast<Real>(v12_val.real()),
                              static_cast<Real>(v12_val.imag()));
        v21[i] = make_complex(static_cast<Real>(v21_val.real()),
                              static_cast<Real>(v21_val.imag()));
        v22[i] = make_complex(static_cast<Real>(v22_val.real()),
                              static_cast<Real>(v22_val.imag()));
    }

    v0_11_host_ = v11;
    v0_12_host_ = v12;
    v0_21_host_ = v21;
    v0_22_host_ = v22;

    CHECK(StatusFromCuda(
        cudaMalloc(&V0_.V_11, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&V0_.V_12, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&V0_.V_21, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(
        cudaMalloc(&V0_.V_22, static_cast<size_t>(order_N_) * sizeof(Complex))));

    CHECK(StatusFromCuda(cudaMemcpyAsync(V0_.V_11,
                                         v11.data(),
                                         v11.size() * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(V0_.V_12,
                                         v12.data(),
                                         v12.size() * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(V0_.V_21,
                                         v21.data(),
                                         v21.size() * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(V0_.V_22,
                                         v22.data(),
                                         v22.size() * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));

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
    if (order_N_ <= 0 || !kvectors_ready_ || kx_host_.empty() || ky_host_.empty() ||
        v0_11_host_.empty())
    {
        return Status::kInvalidValue;
    }

    const int64_t n = order_N_;
    const int64_t n2 = 2 * n;
    const std::complex<Real> eps_c = ToStd(eps);

    std::vector<std::complex<Real>> kz(static_cast<size_t>(n));
    std::vector<Complex> v11(static_cast<size_t>(n));
    std::vector<Complex> v12(static_cast<size_t>(n));
    std::vector<Complex> v21(static_cast<size_t>(n));
    std::vector<Complex> v22(static_cast<size_t>(n));

    for (int64_t i = 0; i < n; ++i)
    {
        const std::complex<Real> kx = ToStd(kx_host_[static_cast<size_t>(i)]);
        const std::complex<Real> ky = ToStd(ky_host_[static_cast<size_t>(i)]);
        std::complex<Real> kz_val = std::sqrt(eps_c - kx * kx - ky * ky);
        if (kz_val.imag() < static_cast<Real>(0.0))
        {
            kz_val = std::conj(kz_val);
        }
        kz[static_cast<size_t>(i)] = kz_val;
        const Real kz_abs = std::abs(kz_val);
        const std::complex<Real> kz_inv =
            kz_abs == static_cast<Real>(0.0)
                ? std::complex<Real>(0.0, 0.0)
                : (std::complex<Real>(1.0, 0.0) / kz_val);

        const std::complex<Real> v11_val = -kx * ky * kz_inv;
        const std::complex<Real> v12_val = kz_val + kx * kx * kz_inv;
        const std::complex<Real> v21_val = -(kz_val + ky * ky * kz_inv);
        const std::complex<Real> v22_val = kx * ky * kz_inv;

        v11[static_cast<size_t>(i)] = ToCu(v11_val);
        v12[static_cast<size_t>(i)] = ToCu(v12_val);
        v21[static_cast<size_t>(i)] = ToCu(v21_val);
        v22[static_cast<size_t>(i)] = ToCu(v22_val);
    }

    std::vector<Complex> v0iv11(static_cast<size_t>(n));
    std::vector<Complex> v0iv12(static_cast<size_t>(n));
    std::vector<Complex> v0iv21(static_cast<size_t>(n));
    std::vector<Complex> v0iv22(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        const std::complex<Real> v0_11 = ToStd(v0_11_host_[static_cast<size_t>(i)]);
        const std::complex<Real> v0_12 = ToStd(v0_12_host_[static_cast<size_t>(i)]);
        const std::complex<Real> v0_21 = ToStd(v0_21_host_[static_cast<size_t>(i)]);
        const std::complex<Real> v0_22 = ToStd(v0_22_host_[static_cast<size_t>(i)]);

        const std::complex<Real> v11_val = ToStd(v11[static_cast<size_t>(i)]);
        const std::complex<Real> v12_val = ToStd(v12[static_cast<size_t>(i)]);
        const std::complex<Real> v21_val = ToStd(v21[static_cast<size_t>(i)]);
        const std::complex<Real> v22_val = ToStd(v22[static_cast<size_t>(i)]);

        const std::complex<Real> b11 = -(v0_11 * v11_val + v0_12 * v21_val);
        const std::complex<Real> b12 = -(v0_11 * v12_val + v0_12 * v22_val);
        const std::complex<Real> b21 = -(v0_21 * v11_val + v0_22 * v21_val);
        const std::complex<Real> b22 = -(v0_21 * v12_val + v0_22 * v22_val);

        v0iv11[static_cast<size_t>(i)] = ToCu(b11);
        v0iv12[static_cast<size_t>(i)] = ToCu(b12);
        v0iv21[static_cast<size_t>(i)] = ToCu(b21);
        v0iv22[static_cast<size_t>(i)] = ToCu(b22);
    }

    DeviceVector d_v0iv11;
    DeviceVector d_v0iv12;
    DeviceVector d_v0iv21;
    DeviceVector d_v0iv22;
    CHECK(d_v0iv11.Allocate(static_cast<size_t>(n)));
    CHECK(d_v0iv12.Allocate(static_cast<size_t>(n)));
    CHECK(d_v0iv21.Allocate(static_cast<size_t>(n)));
    CHECK(d_v0iv22.Allocate(static_cast<size_t>(n)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(d_v0iv11.get(),
                                         v0iv11.data(),
                                         v0iv11.size() * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(d_v0iv12.get(),
                                         v0iv12.data(),
                                         v0iv12.size() * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(d_v0iv21.get(),
                                         v0iv21.data(),
                                         v0iv21.size() * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(d_v0iv22.get(),
                                         v0iv22.data(),
                                         v0iv22.size() * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));

    DeviceMatrix V0iV;
    CHECK(V0iV.Allocate(n2, n2));
    CHECK(FillDiagBlock(d_v0iv11.get(), n, V0iV.get(), n2, 0, 0, stream_));
    CHECK(FillDiagBlock(d_v0iv12.get(), n, V0iV.get(), n2, 0, n, stream_));
    CHECK(FillDiagBlock(d_v0iv21.get(), n, V0iV.get(), n2, n, 0, stream_));
    CHECK(FillDiagBlock(d_v0iv22.get(), n, V0iV.get(), n2, n, n, stream_));

    std::vector<Complex> phase_host(static_cast<size_t>(n2));
    const std::complex<Real> imag_unit(0.0, 1.0);
    for (int64_t i = 0; i < n; ++i)
    {
        const std::complex<Real> phase_val =
            std::exp(imag_unit * static_cast<Real>(omega_ * thickness) *
                     kz[static_cast<size_t>(i)]);
        const Complex phase_cu = ToCu(phase_val);
        phase_host[static_cast<size_t>(i)] = phase_cu;
        phase_host[static_cast<size_t>(i + n)] = phase_cu;
    }

    DeviceVector phase_diag;
    CHECK(phase_diag.Allocate(static_cast<size_t>(n2)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(phase_diag.get(),
                                         phase_host.data(),
                                         phase_host.size() * sizeof(Complex),
                                         cudaMemcpyHostToDevice,
                                         stream_)));

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
