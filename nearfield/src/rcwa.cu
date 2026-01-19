#include <cstring>
#include <cufft.h>
#include <limits>
#include "complex_matrix.h"
#include "rcwa.h"

static Status StatusFromCuda(cudaError_t status)
{
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

static Status StatusFromCufft(cufftResult status)
{
    return status == CUFFT_SUCCESS ? Status::kSuccess : Status::kCudaError;
}

static constexpr double kExpConstant[5][5] = {
    {0.0, -0.10036558103014462001, -0.00802924648241156960, -0.00089213849804572995, 0.0},
    {0.0,
     0.39784974949964507614,
     1.36783778460411719922,
     0.49828962252538267755,
     -0.00063789819459472330},
    {-10.9676396052962062593,
     1.68015813878906197182,
     0.05717798464788655127,
     -0.00698210122488052084,
     0.00003349750170860705},
    {-0.0904316832390810561,
     -0.06764045190713819075,
     0.06759613017704596460,
     0.02955525704293155274,
     -0.00001391802575160607},
    {0.0, 0.0, -0.09233646193671185927, -0.01693649390020817171, -0.00001400867981820361}};

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
    void Reset(Complex* p = nullptr, size_t count = 0)
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
        ptr = p;
        elements = count;
    }
    Complex* Release()
    {
        Complex* out = ptr;
        ptr = nullptr;
        elements = 0;
        return out;
    }
    void Swap(DeviceMatrix& other)
    {
        Complex* tmp_ptr = ptr;
        ptr = other.ptr;
        other.ptr = tmp_ptr;
        size_t tmp_elements = elements;
        elements = other.elements;
        other.elements = tmp_elements;
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

struct DeviceRealBuffer
{
    Real* ptr = nullptr;
    size_t elements = 0;

    ~DeviceRealBuffer()
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    Real* get() const { return ptr; }

    Status Allocate(size_t count)
    {
        if (count == 0)
        {
            return Status::kInvalidValue;
        }
        if (count > (std::numeric_limits<size_t>::max() / sizeof(Real)))
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
        Real* out = nullptr;
        if (cudaMalloc(&out, count * sizeof(Real)) != cudaSuccess)
        {
            return Status::kCudaError;
        }
        ptr = out;
        elements = count;
        return Status::kSuccess;
    }
};

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
    const Complex kx2 = ComplexMul(kx, kx);
    const Complex ky2 = ComplexMul(ky, ky);
    const Complex sum = ComplexAdd(kx2, ky2);
    const Complex diff = ComplexSub(eps, sum);
    Complex kz = ComplexSqrt(diff);
    if (complex_imag(kz) < kZero)
    {
        kz = ComplexConj(kz);
    }

    const Complex kz_inv = ComplexReciprocal(kz);
    const Complex kxky = ComplexMul(kx, ky);
    const Complex v11 = ComplexMul(make_complex(kNegOne, kZero), ComplexMul(kxky, kz_inv));
    const Complex v12 = ComplexAdd(kz, ComplexMul(kx2, kz_inv));
    const Complex v21 =
        ComplexMul(make_complex(kNegOne, kZero), ComplexAdd(kz, ComplexMul(ky2, kz_inv)));
    const Complex v22 = ComplexMul(kxky, kz_inv);

    const Complex term11 = ComplexMul(v0_11[idx], v11);
    const Complex term12 = ComplexMul(v0_12[idx], v21);
    const Complex term21 = ComplexMul(v0_21[idx], v11);
    const Complex term22 = ComplexMul(v0_22[idx], v21);
    const Complex term31 = ComplexMul(v0_11[idx], v12);
    const Complex term32 = ComplexMul(v0_12[idx], v22);
    const Complex term41 = ComplexMul(v0_21[idx], v12);
    const Complex term42 = ComplexMul(v0_22[idx], v22);

    const Complex neg_one = make_complex(kNegOne, kZero);
    const Complex b11 = -(term11 + term12);
    const Complex b12 = -(term31 + term32);
    const Complex b21 = -(term21 + term22);
    const Complex b22 = -(term41 + term42);

    const int64_t row0 = idx;
    const int64_t row1 = idx + n;
    const int64_t col0 = idx;
    const int64_t col1 = idx + n;
    v0iv[col0 * n2 + row0] = b11;
    v0iv[col1 * n2 + row0] = b12;
    v0iv[col0 * n2 + row1] = b21;
    v0iv[col1 * n2 + row1] = b22;

    const Complex kzt = ComplexMul(kz, make_complex(omega * thickness, kZero));
    const Complex phase = ComplexExpI(kzt);
    phase_diag[idx] = phase;
    phase_diag[idx + n] = phase;
}

__global__ void ScaleComplexKernel(Complex* data, int64_t n, Real scale)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    const Complex value = data[idx];
    data[idx] = make_complex(complex_real(value) * scale, complex_imag(value) * scale);
}

__global__ void SetSourceKernel(Complex* vec, int64_t index, Complex ex, Complex ey, int64_t n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        vec[index] = ex;
        vec[index + n] = ey;
    }
}

__global__ void BuildKzKernel(const Complex* kx,
                              const Complex* ky,
                              Complex eps,
                              int input_layer,
                              Complex* kz,
                              int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    const Complex kxv = kx[idx];
    const Complex kyv = ky[idx];
    const Complex sum = ComplexAdd(ComplexMul(kxv, kxv), ComplexMul(kyv, kyv));
    const Complex diff = ComplexSub(eps, sum);
    Complex kzz = ComplexSqrt(diff);
    if (complex_imag(kzz) < kZero)
    {
        kzz = ComplexConj(kzz);
    }
    if (!input_layer)
    {
        kzz = ComplexMul(make_complex(kNegOne, kZero), kzz);
    }
    kz[idx] = kzz;
}

__global__ void BuildEzKernel(const Complex* kx,
                              const Complex* ky,
                              const Complex* kz,
                              const Complex* ex,
                              const Complex* ey,
                              Complex* ez,
                              int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    const Complex numerator =
        ComplexAdd(ComplexMul(kx[idx], ex[idx]), ComplexMul(ky[idx], ey[idx]));
    const Complex inv_kz = ComplexReciprocal(kz[idx]);
    ez[idx] = ComplexMul(numerator, inv_kz);
}

__global__ void FieldXYKernel(const Complex* kx,
                              const Complex* ky,
                              const Complex* ex,
                              const Complex* ey,
                              const Complex* ez,
                              const Real* x_axis,
                              const Real* y_axis,
                              int64_t x_count,
                              int64_t y_count,
                              Real omega,
                              int64_t n,
                              Complex* out_ex,
                              Complex* out_ey,
                              Complex* out_ez)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = x_count * y_count;
    if (idx >= total)
    {
        return;
    }

    const int64_t row = idx % x_count;
    const int64_t col = idx / x_count;
    const Real x = x_axis[row];
    const Real y = y_axis[col];
    const Complex omega_c = make_complex(omega, kZero);
    const Complex x_c = make_complex(x, kZero);
    const Complex y_c = make_complex(y, kZero);

    Complex sum_ex = make_complex(kZero, kZero);
    Complex sum_ey = make_complex(kZero, kZero);
    Complex sum_ez = make_complex(kZero, kZero);
    for (int64_t j = 0; j < n; ++j)
    {
        const Complex arg =
            ComplexMul(omega_c, ComplexAdd(ComplexMul(kx[j], x_c), ComplexMul(ky[j], y_c)));
        const Complex phase = ComplexExpI(arg);
        sum_ex = ComplexAdd(sum_ex, ComplexMul(ex[j], phase));
        sum_ey = ComplexAdd(sum_ey, ComplexMul(ey[j], phase));
        sum_ez = ComplexAdd(sum_ez, ComplexMul(ez[j], phase));
    }

    out_ex[col * x_count + row] = sum_ex;
    out_ey[col * x_count + row] = sum_ey;
    out_ez[col * x_count + row] = sum_ez;
}

__global__ void BuildConvMatrixKernel(const Complex* fft,
                                      int64_t nx,
                                      int64_t ny,
                                      int64_t oy_len,
                                      Complex* out,
                                      int64_t order_n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = order_n * order_n;
    if (idx >= total)
    {
        return;
    }

    const int64_t row = idx % order_n;
    const int64_t col = idx / order_n;

    const int64_t row_ix = row / oy_len;
    const int64_t row_iy = row - row_ix * oy_len;
    const int64_t col_ix = col / oy_len;
    const int64_t col_iy = col - col_ix * oy_len;

    int64_t fx = row_ix - col_ix;
    int64_t fy = row_iy - col_iy;

    if (nx > 0)
    {
        fx %= nx;
        if (fx < 0)
        {
            fx += nx;
        }
    }
    if (ny > 0)
    {
        fy %= ny;
        if (fy < 0)
        {
            fy += ny;
        }
    }

    const int64_t fft_idx = fx * ny + fy;
    out[col * order_n + row] = fft[fft_idx];
}

struct CufftPlan
{
    cufftHandle handle = 0;
    ~CufftPlan()
    {
        if (handle)
        {
            cufftDestroy(handle);
        }
    }
};

static bool IsForwardDirection(const char* direction)
{
    if (!direction)
    {
        return true;
    }
    if (std::strcmp(direction, "b") == 0 || std::strcmp(direction, "backward") == 0)
    {
        return false;
    }
    return true;
}

static bool IsInputLayerName(const char* layer)
{
    if (!layer)
    {
        return true;
    }
    if (std::strcmp(layer, "o") == 0 || std::strcmp(layer, "out") == 0 ||
        std::strcmp(layer, "output") == 0)
    {
        return false;
    }
    return true;
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
    const Complex kx2 = ComplexMul(kx, kx);
    const Complex ky2 = ComplexMul(ky, ky);
    const Complex sum = ComplexAdd(kx2, ky2);
    const Complex diff = ComplexSub(eps, sum);
    Complex kz = ComplexSqrt(diff);
    if (complex_imag(kz) < kZero)
    {
        kz = ComplexConj(kz);
    }

    const Complex kz_inv = ComplexReciprocal(kz);
    const Complex kxky = ComplexMul(kx, ky);
    v0_11[idx] = ComplexMul(make_complex(kNegOne, kZero), ComplexMul(kxky, kz_inv));
    v0_12[idx] = ComplexAdd(kz, ComplexMul(kx2, kz_inv));
    v0_21[idx] = ComplexMul(make_complex(kNegOne, kZero), ComplexAdd(kz, ComplexMul(ky2, kz_inv)));
    v0_22[idx] = ComplexMul(kxky, kz_inv);
}

__global__ void V0LeftMulKernel(const Complex* v11,
                                const Complex* v12,
                                const Complex* v21,
                                const Complex* v22,
                                const Complex* b,
                                Complex* c,
                                int64_t n,
                                int64_t n2)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n2 * n;
    if (idx >= total)
    {
        return;
    }
    int64_t row = idx % n2;
    int64_t col = idx / n2;
    int64_t top_index = col * n + row;
    int64_t bottom_index = top_index + n2;

    const Complex b_top = b[top_index];
    const Complex b_bottom = b[bottom_index];
    c[top_index] = ComplexAdd(ComplexMul(v11[row], b_top), ComplexMul(v12[row], b_bottom));
    c[bottom_index] = ComplexAdd(ComplexMul(v21[row], b_top), ComplexMul(v22[row], b_bottom));
}

__global__ void V0RightMulKernel(const Complex* v11,
                                 const Complex* v12,
                                 const Complex* v21,
                                 const Complex* v22,
                                 const Complex* b,
                                 Complex* c,
                                 int64_t n,
                                 int64_t n2)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n2 * n;
    if (idx >= total)
    {
        return;
    }
    int64_t row = idx % n;
    int64_t col = idx / n;
    int64_t left_index = col * n + row;
    int64_t right_index = left_index + n2 * n;

    const Complex b_left = b[left_index];
    const Complex b_right = b[right_index];
    c[left_index] = ComplexAdd(ComplexMul(b_left, v11[col]), ComplexMul(b_right, v21[col]));
    c[right_index] = ComplexAdd(ComplexMul(b_left, v12[col]), ComplexMul(b_right, v22[col]));
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
    kx_diag[idx] = make_complex(kx, kZero);
    ky_diag[idx] = make_complex(ky, kZero);
}

static Status
ApplyV0Left(const VMatrix& V, const Complex* B, int64_t n, DeviceMatrix* out, cudaStream_t stream)
{
    if (!B || !out || !V.V_11 || !V.V_12 || !V.V_21 || !V.V_22 || n <= 0 || (n % 2) != 0)
    {
        return Status::kInvalidValue;
    }
    CHECK(out->Allocate(n, n));
    int64_t n2 = n / 2;
    int64_t total = n2 * n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    V0LeftMulKernel<<<blocks, threads, 0, stream>>>(
        V.V_11, V.V_12, V.V_21, V.V_22, B, out->get(), n, n2);
    return StatusFromCuda(cudaGetLastError());
}

static Status
ApplyV0Right(const VMatrix& V, const Complex* B, int64_t n, DeviceMatrix* out, cudaStream_t stream)
{
    if (!B || !out || !V.V_11 || !V.V_12 || !V.V_21 || !V.V_22 || n <= 0 || (n % 2) != 0)
    {
        return Status::kInvalidValue;
    }
    CHECK(out->Allocate(n, n));
    int64_t n2 = n / 2;
    int64_t total = n2 * n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    V0RightMulKernel<<<blocks, threads, 0, stream>>>(
        V.V_11, V.V_12, V.V_21, V.V_22, B, out->get(), n, n2);
    return StatusFromCuda(cudaGetLastError());
}

RCWA::RCWA(Real freq, int order_x, int order_y, Real Lx, Real Ly, cudaStream_t stream)
    : freq_(freq), omega_(static_cast<Real>(6.28318530717958647692) * freq), Lx_(Lx), Ly_(Ly),
      order_x_(std::max(order_x, 0)), order_y_(std::max(order_y, 0)), stream_(stream)
{
    const int ox = std::max(order_x_, 0);
    const int oy = std::max(order_y_, 0);
    order_N_ = static_cast<int64_t>((2 * ox + 1) * (2 * oy + 1));
    if (freq_ != kZero && Lx_ != kZero && Ly_ != kZero)
    {
        Gx_norm_ = kOne / (Lx_ * freq_);
        Gy_norm_ = kOne / (Ly_ * freq_);
    }
}

RCWA::~RCWA()
{
    FreeLayers();
    FreeKVectors();
    FreeGlobalSMatrix();
    FreeSource();
}

void RCWA::FreeLayers()
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

void RCWA::FreeKVectors()
{
    SAFEFREE(kx_diag_);
    SAFEFREE(ky_diag_);
    SAFEFREE(V0_.V_11);
    SAFEFREE(V0_.V_12);
    SAFEFREE(V0_.V_21);
    SAFEFREE(V0_.V_22);
    kvectors_ready_ = false;
}

void RCWA::FreeGlobalSMatrix()
{
    SAFEFREE(global_smatrix_.S_11);
    SAFEFREE(global_smatrix_.S_12);
    SAFEFREE(global_smatrix_.S_21);
    SAFEFREE(global_smatrix_.S_22);
}

void RCWA::FreeSource() { SAFEFREE(source_ei_); }

Status RCWA::SetIncidentAngles(Real inc_ang, Real azi_ang, Real eps_ref)
{
    return BuildKxKy(inc_ang, azi_ang, eps_ref);
}

Status RCWA::SetIncidentAngles(Real inc_ang, Real azi_ang, const char* angle_layer)
{
    angle_layer_input_ = IsInputLayerName(angle_layer);
    const Complex eps = angle_layer_input_ ? eps_in_ : eps_out_;
    const Real sqrt_eps = complex_real(ComplexSqrt(eps));
    const Real eps_ref = sqrt_eps * sqrt_eps;
    return BuildKxKy(inc_ang, azi_ang, eps_ref);
}

Status RCWA::AddInputLayer(Real eps) { return AddInputLayer(make_complex(eps, kZero)); }

Status RCWA::AddInputLayer(Complex eps)
{
    eps_in_ = eps;
    return Status::kSuccess;
}

Status RCWA::AddOutputLayer(Real eps) { return AddOutputLayer(make_complex(eps, kZero)); }

Status RCWA::AddOutputLayer(Complex eps)
{
    eps_out_ = eps;
    return Status::kSuccess;
}

Status RCWA::SourcePlanewave(Real ex_amp, Real ey_amp, const char* direction)
{
    return SourcePlanewave(make_complex(ex_amp, kZero), make_complex(ey_amp, kZero), direction);
}

Status RCWA::SourcePlanewave(Complex ex_amp, Complex ey_amp, const char* direction)
{
    if (order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }

    source_forward_ = IsForwardDirection(direction);
    FreeSource();

    const size_t count = static_cast<size_t>(2 * order_N_);
    CHECK(StatusFromCuda(cudaMalloc(&source_ei_, count * sizeof(Complex))));
    CHECK(StatusFromCuda(cudaMemsetAsync(source_ei_, 0, count * sizeof(Complex), stream_)));

    const int64_t order_y_len = static_cast<int64_t>(2 * order_y_ + 1);
    const int64_t index = order_y_len * static_cast<int64_t>(order_x_) + order_y_;

    SetSourceKernel<<<1, 1, 0, stream_>>>(source_ei_, index, ex_amp, ey_amp, order_N_);
    return StatusFromCuda(cudaGetLastError());
}

Status RCWA::BuildKxKy(Real inc_ang, Real azi_ang, Real eps_ref)
{
    if (order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }

    FreeKVectors();

    const Real sqrt_eps = std::sqrt(static_cast<double>(eps_ref));
    const Real kx0_norm = sqrt_eps * std::sin(inc_ang) * std::cos(azi_ang);
    const Real ky0_norm = sqrt_eps * std::sin(inc_ang) * std::sin(azi_ang);

    CHECK(StatusFromCuda(cudaMalloc(&kx_diag_, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(cudaMalloc(&ky_diag_, static_cast<size_t>(order_N_) * sizeof(Complex))));

    const int threads = 256;
    const int blocks = static_cast<int>((order_N_ + threads - 1) / threads);
    BuildKxKyDiagKernel<<<blocks, threads, 0, stream_>>>(
        kx_diag_, ky_diag_, order_x_, order_y_, kx0_norm, ky0_norm, Gx_norm_, Gy_norm_, order_N_);
    CHECK(StatusFromCuda(cudaGetLastError()));

    CHECK(BuildV0(kOne));

    kvectors_ready_ = true;
    return Status::kSuccess;
}

Status RCWA::BuildV0(Real eps_ref)
{
    if (!kx_diag_ || !ky_diag_ || order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }

    SAFEFREE(V0_.V_11);
    SAFEFREE(V0_.V_12);
    SAFEFREE(V0_.V_21);
    SAFEFREE(V0_.V_22);

    CHECK(StatusFromCuda(cudaMalloc(&V0_.V_11, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(cudaMalloc(&V0_.V_12, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(cudaMalloc(&V0_.V_21, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(cudaMalloc(&V0_.V_22, static_cast<size_t>(order_N_) * sizeof(Complex))));

    const int threads = 256;
    const int blocks = static_cast<int>((order_N_ + threads - 1) / threads);
    BuildV0Kernel<<<blocks, threads, 0, stream_>>>(kx_diag_,
                                                   ky_diag_,
                                                   make_complex(eps_ref, kZero),
                                                   V0_.V_11,
                                                   V0_.V_12,
                                                   V0_.V_21,
                                                   V0_.V_22,
                                                   order_N_);
    CHECK(StatusFromCuda(cudaGetLastError()));

    return Status::kSuccess;
}

Status RCWA::ComputeTFromPQ(const Complex* P,
                            const Complex* Q,
                            int64_t n,
                            Real k0d,
                            TMatrix* T,
                            cudaStream_t stream) const
{
    if (!P || !Q || !T || n <= 0)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;

    T->T_11 = nullptr;
    T->T_12 = nullptr;
    T->T_21 = nullptr;
    T->T_22 = nullptr;

    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    // Matrix exponential via scaling/squaring + [5/5] Padé on block matrix:
    // A = [0, -P; Q, 0], then T = exp(A * k0d).
    double Pnorm = 0.0;
    double Qnorm = 0.0;
    CHECK(ComputeMatrixNorm1(P, n, &Pnorm, use_stream));
    CHECK(ComputeMatrixNorm1(Q, n, &Qnorm, use_stream));

    const double Rnorm = std::max(Pnorm, Qnorm);
    const double scaled = Rnorm * static_cast<double>(k0d);
    int m = 0;
    if (scaled > 0.0)
    {
        m = static_cast<int>(std::ceil(std::log2(scaled)));
        if (m < 0)
        {
            m = 0;
        }
    }

    const double scale_A = std::pow(2.0, -m) * static_cast<double>(k0d);
    const Complex scale_pos = make_complex(static_cast<Real>(scale_A), kZero);
    const Complex scale_neg = make_complex(static_cast<Real>(-scale_A), kZero);

    DeviceMatrix I2;
    CHECK(I2.Allocate(n, n));
    CHECK(MakeIdentity(I2.get(), n, use_stream));

    DeviceMatrix A_12;
    CHECK(A_12.Allocate(n, n));
    CHECK(ScaleMatrix(P, A_12.get(), n, scale_neg, use_stream));

    DeviceMatrix A_21;
    CHECK(A_21.Allocate(n, n));
    CHECK(ScaleMatrix(Q, A_21.get(), n, scale_pos, use_stream));

    DeviceMatrix A2_11;
    CHECK(A2_11.Allocate(n, n));
    CHECK(MatMul(ops, A_12.get(), A_21.get(), A2_11.get(), n));

    DeviceMatrix A2_22;
    CHECK(A2_22.Allocate(n, n));
    CHECK(MatMul(ops, A_21.get(), A_12.get(), A2_22.get(), n));

    DeviceMatrix A3_12;
    CHECK(A3_12.Allocate(n, n));
    CHECK(MatMul(ops, A_12.get(), A2_22.get(), A3_12.get(), n));

    DeviceMatrix A3_21;
    CHECK(A3_21.Allocate(n, n));
    CHECK(MatMul(ops, A_21.get(), A2_11.get(), A3_21.get(), n));

    DeviceMatrix A6_11;
    CHECK(A6_11.Allocate(n, n));
    CHECK(MatMul(ops, A3_12.get(), A3_21.get(), A6_11.get(), n));

    DeviceMatrix A6_22;
    CHECK(A6_22.Allocate(n, n));
    CHECK(MatMul(ops, A3_21.get(), A3_12.get(), A6_22.get(), n));

    DeviceMatrix scratch;
    DeviceMatrix B_11[5];
    DeviceMatrix B_12[5];
    DeviceMatrix B_21[5];
    DeviceMatrix B_22[5];

    for (int i = 0; i < 5; ++i)
    {
        const Complex c0 = make_complex(static_cast<Real>(kExpConstant[i][0]), kZero);
        const Complex c1 = make_complex(static_cast<Real>(kExpConstant[i][1]), kZero);
        const Complex c2 = make_complex(static_cast<Real>(kExpConstant[i][2]), kZero);
        const Complex c3 = make_complex(static_cast<Real>(kExpConstant[i][3]), kZero);
        const Complex c4 = make_complex(static_cast<Real>(kExpConstant[i][4]), kZero);

        CHECK(B_11[i].Allocate(n, n));
        CHECK(ScaleMatrix(I2.get(), B_11[i].get(), n, c0, use_stream));
        CHECK(scratch.Allocate(n, n));
        CHECK(ScaleMatrix(A2_11.get(), scratch.get(), n, c2, use_stream));
        CHECK(AddInPlace(ops, B_11[i].get(), scratch.get(), n));
        CHECK(ScaleMatrix(A6_11.get(), scratch.get(), n, c4, use_stream));
        CHECK(AddInPlace(ops, B_11[i].get(), scratch.get(), n));

        CHECK(B_12[i].Allocate(n, n));
        CHECK(ScaleMatrix(A_12.get(), B_12[i].get(), n, c1, use_stream));
        CHECK(ScaleMatrix(A3_12.get(), scratch.get(), n, c3, use_stream));
        CHECK(AddInPlace(ops, B_12[i].get(), scratch.get(), n));

        CHECK(B_21[i].Allocate(n, n));
        CHECK(ScaleMatrix(A_21.get(), B_21[i].get(), n, c1, use_stream));
        CHECK(ScaleMatrix(A3_21.get(), scratch.get(), n, c3, use_stream));
        CHECK(AddInPlace(ops, B_21[i].get(), scratch.get(), n));

        CHECK(B_22[i].Allocate(n, n));
        CHECK(ScaleMatrix(I2.get(), B_22[i].get(), n, c0, use_stream));
        CHECK(ScaleMatrix(A2_22.get(), scratch.get(), n, c2, use_stream));
        CHECK(AddInPlace(ops, B_22[i].get(), scratch.get(), n));
        CHECK(ScaleMatrix(A6_22.get(), scratch.get(), n, c4, use_stream));
        CHECK(AddInPlace(ops, B_22[i].get(), scratch.get(), n));
    }

    A2_11.Reset();
    A2_22.Reset();
    A3_12.Reset();
    A3_21.Reset();
    A6_11.Reset();
    A6_22.Reset();

    DeviceMatrix A9_11;
    CHECK(A9_11.Allocate(n, n));
    CHECK(MatMul(ops, B_11[0].get(), B_11[4].get(), A9_11.get(), n));
    CHECK(scratch.Allocate(n, n));
    CHECK(MatMul(ops, B_12[0].get(), B_21[4].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_11.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_11.get(), B_11[3].get(), n));

    DeviceMatrix A9_12;
    CHECK(A9_12.Allocate(n, n));
    CHECK(MatMul(ops, B_11[0].get(), B_12[4].get(), A9_12.get(), n));
    CHECK(scratch.Allocate(n, n));
    CHECK(MatMul(ops, B_12[0].get(), B_22[4].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_12.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_12.get(), B_12[3].get(), n));

    DeviceMatrix A9_21;
    CHECK(A9_21.Allocate(n, n));
    CHECK(MatMul(ops, B_21[0].get(), B_11[4].get(), A9_21.get(), n));
    CHECK(scratch.Allocate(n, n));
    CHECK(MatMul(ops, B_22[0].get(), B_21[4].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_21.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_21.get(), B_21[3].get(), n));

    DeviceMatrix A9_22;
    CHECK(A9_22.Allocate(n, n));
    CHECK(MatMul(ops, B_21[0].get(), B_12[4].get(), A9_22.get(), n));
    CHECK(scratch.Allocate(n, n));
    CHECK(MatMul(ops, B_22[0].get(), B_22[4].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_22.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_22.get(), B_22[3].get(), n));

    DeviceMatrix temp1;
    DeviceMatrix temp2;
    DeviceMatrix temp3;

    DeviceMatrix T_11;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_11[2].get(), A9_11.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A9_11.get(), temp2.get(), n));
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_12[2].get(), A9_12.get(), temp1.get(), n));
        CHECK(temp3.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A9_21.get(), temp3.get(), n));
        CHECK(T_11.Allocate(n, n));
        CHECK(AddMatrix(ops, B_11[1].get(), temp2.get(), T_11.get(), n));
        CHECK(AddInPlace(ops, T_11.get(), temp3.get(), n));
    }

    DeviceMatrix T_12;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_11[2].get(), A9_11.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A9_12.get(), temp2.get(), n));
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_12[2].get(), A9_12.get(), temp1.get(), n));
        CHECK(temp3.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A9_22.get(), temp3.get(), n));
        CHECK(T_12.Allocate(n, n));
        CHECK(AddMatrix(ops, B_12[1].get(), temp2.get(), T_12.get(), n));
        CHECK(AddInPlace(ops, T_12.get(), temp3.get(), n));
    }

    DeviceMatrix T_21;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_21[2].get(), A9_21.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A9_11.get(), temp2.get(), n));
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_22[2].get(), A9_22.get(), temp1.get(), n));
        CHECK(temp3.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A9_21.get(), temp3.get(), n));
        CHECK(T_21.Allocate(n, n));
        CHECK(AddMatrix(ops, B_21[1].get(), temp2.get(), T_21.get(), n));
        CHECK(AddInPlace(ops, T_21.get(), temp3.get(), n));
    }

    DeviceMatrix T_22;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_21[2].get(), A9_21.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A9_12.get(), temp2.get(), n));
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_22[2].get(), A9_22.get(), temp1.get(), n));
        CHECK(temp3.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A9_22.get(), temp3.get(), n));
        CHECK(T_22.Allocate(n, n));
        CHECK(AddMatrix(ops, B_22[1].get(), temp2.get(), T_22.get(), n));
        CHECK(AddInPlace(ops, T_22.get(), temp3.get(), n));
    }

    DeviceMatrix tmp11;
    DeviceMatrix tmp12;
    DeviceMatrix tmp21;
    DeviceMatrix tmp22;

    for (int i = 0; i < m; ++i)
    {
        CHECK(tmp11.Allocate(n, n));
        CHECK(MatMul(ops, T_11.get(), T_11.get(), tmp11.get(), n));
        CHECK(scratch.Allocate(n, n));
        CHECK(MatMul(ops, T_12.get(), T_21.get(), scratch.get(), n));
        CHECK(AddInPlace(ops, tmp11.get(), scratch.get(), n));
        CHECK(tmp12.Allocate(n, n));
        CHECK(MatMul(ops, T_11.get(), T_12.get(), tmp12.get(), n));
        CHECK(scratch.Allocate(n, n));
        CHECK(MatMul(ops, T_12.get(), T_22.get(), scratch.get(), n));
        CHECK(AddInPlace(ops, tmp12.get(), scratch.get(), n));
        CHECK(tmp21.Allocate(n, n));
        CHECK(MatMul(ops, T_21.get(), T_11.get(), tmp21.get(), n));
        CHECK(scratch.Allocate(n, n));
        CHECK(MatMul(ops, T_22.get(), T_21.get(), scratch.get(), n));
        CHECK(AddInPlace(ops, tmp21.get(), scratch.get(), n));
        CHECK(tmp22.Allocate(n, n));
        CHECK(MatMul(ops, T_21.get(), T_12.get(), tmp22.get(), n));
        CHECK(scratch.Allocate(n, n));
        CHECK(MatMul(ops, T_22.get(), T_22.get(), scratch.get(), n));
        CHECK(AddInPlace(ops, tmp22.get(), scratch.get(), n));
        T_11.Swap(tmp11);
        T_12.Swap(tmp12);
        T_21.Swap(tmp21);
        T_22.Swap(tmp22);
    }

    T->T_11 = T_11.Release();
    T->T_12 = T_12.Release();
    T->T_21 = T_21.Release();
    T->T_22 = T_22.Release();
    return Status::kSuccess;
}

Status RCWA::ComputeSFromT(const TMatrix& T,
                           const VMatrix& V,
                           int64_t n,
                           SMatrix* S,
                           cudaStream_t stream) const
{
    if (!S || !T.T_11 || !T.T_12 || !T.T_21 || !T.T_22 || !V.V_11 || !V.V_12 || !V.V_21 ||
        !V.V_22 || n <= 0 || (n % 2) != 0)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;

    S->S_11 = nullptr;
    S->S_12 = nullptr;
    S->S_21 = nullptr;
    S->S_22 = nullptr;

    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    // A = T11 + i * V0 * T21
    // B = (i * T12 - V0 * T22) * V0
    // C = (A + B) / 2, D = (A - B) / 2
    // S11 = C^{-1}, S12 = -S11 * D, S21 = S12, S22 = S11
    const Complex imag = make_complex(kZero, kOne);
    const Complex half = make_complex(kHalf, kZero);
    const Complex neg_one = make_complex(kNegOne, kZero);

    DeviceMatrix tmp_left;
    DeviceMatrix tmp_right;
    DeviceMatrix A;
    DeviceMatrix D;

    CHECK(ApplyV0Left(V, T.T_21, n, &tmp_left, use_stream));
    CHECK(ScaleInPlace(tmp_left.get(), n, imag, use_stream));

    CHECK(A.Allocate(n, n));
    CHECK(CopyMatrix(T.T_11, A.get(), n, use_stream));
    CHECK(AddInPlace(ops, A.get(), tmp_left.get(), n));

    CHECK(ApplyV0Left(V, T.T_22, n, &tmp_left, use_stream));

    CHECK(tmp_right.Allocate(n, n));
    CHECK(CopyMatrix(T.T_12, tmp_right.get(), n, use_stream));
    CHECK(ScaleInPlace(tmp_right.get(), n, imag, use_stream));
    CHECK(SubInPlace(ops, tmp_right.get(), tmp_left.get(), n));

    CHECK(ApplyV0Right(V, tmp_right.get(), n, &tmp_right, use_stream));

    CHECK(D.Allocate(n, n));
    CHECK(SubMatrix(ops, A.get(), tmp_right.get(), D.get(), n));
    CHECK(ScaleInPlace(D.get(), n, half, use_stream));

    CHECK(AddInPlace(ops, A.get(), tmp_right.get(), n));
    CHECK(ScaleInPlace(A.get(), n, half, use_stream));

    CHECK(tmp_right.Allocate(n, n));
    CHECK(MakeIdentity(tmp_right.get(), n, use_stream));

    DeviceMatrix S11;
    CHECK(S11.Allocate(n, n));
    CHECK(ops.Solve(A.get(), tmp_right.get(), S11.get(), n));

    CHECK(tmp_right.Allocate(n, n));
    CHECK(MatMul(ops, S11.get(), D.get(), tmp_right.get(), n));
    CHECK(ScaleInPlace(tmp_right.get(), n, neg_one, use_stream));

    DeviceMatrix S21;
    CHECK(S21.Allocate(n, n));
    CHECK(CopyMatrix(tmp_right.get(), S21.get(), n, use_stream));

    DeviceMatrix S22;
    CHECK(S22.Allocate(n, n));
    CHECK(CopyMatrix(S11.get(), S22.get(), n, use_stream));

    S->S_11 = S11.Release();
    S->S_12 = tmp_right.Release();
    S->S_21 = S21.Release();
    S->S_22 = S22.Release();
    return Status::kSuccess;
}

Status RCWA::AddLayer(Real thickness, Real eps)
{
    return AddLayer(thickness, make_complex(eps, kZero));
}

Status RCWA::AddLayer(Real thickness, Complex eps)
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

Status RCWA::AddLayerHomogeneous(Real thickness, Real eps)
{
    return AddLayerHomogeneous(thickness, make_complex(eps, kZero));
}

Status RCWA::AddLayerHomogeneous(Real thickness, Complex eps)
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

    // Homogeneous layer transfer via analytic exp(A*d):
    // uses V0iV blocks and phase = exp(i * kz * omega * d).
    DeviceMatrix I;
    DeviceMatrix A;
    DeviceMatrix B;
    DeviceMatrix scratch1;
    DeviceMatrix scratch2;
    DeviceMatrix C1;
    CHECK(I.Allocate(n2, n2));
    CHECK(A.Allocate(n2, n2));
    CHECK(B.Allocate(n2, n2));
    CHECK(scratch1.Allocate(n2, n2));
    CHECK(scratch2.Allocate(n2, n2));
    CHECK(C1.Allocate(n2, n2));

    CHECK(MakeIdentity(I.get(), n2, stream_));
    CHECK(AddMatrix(ops, I.get(), V0iV.get(), A.get(), n2));
    CHECK(CopyMatrix(I.get(), scratch1.get(), n2, stream_));
    CHECK(SubInPlace(ops, scratch1.get(), V0iV.get(), n2));
    CHECK(ops.DiagRightMatMul(scratch1.get(), phase_diag.get(), B.get(), n2));

    CHECK(ops.Solve(A.get(), I.get(), scratch1.get(), n2));
    CHECK(MatMul(ops, B.get(), scratch1.get(), scratch2.get(), n2));
    CHECK(MatMul(ops, scratch2.get(), B.get(), scratch1.get(), n2));
    CHECK(SubInPlace(ops, A.get(), scratch1.get(), n2));
    CHECK(ops.Solve(A.get(), I.get(), C1.get(), n2));
    CHECK(ScaleInPlace(C1.get(), n2, make_complex(kTwo, kZero), stream_));
    CHECK(MatMul(ops, C1.get(), scratch2.get(), scratch1.get(), n2));
    CHECK(ScaleInPlace(scratch1.get(), n2, make_complex(kNegOne, kZero), stream_));

    CHECK(ops.DiagLeftMatMul(phase_diag.get(), C1.get(), scratch2.get(), n2));
    CHECK(AddInPlace(ops, scratch2.get(), scratch1.get(), n2));

    CHECK(ops.DiagLeftMatMul(phase_diag.get(), scratch1.get(), A.get(), n2));
    CHECK(AddInPlace(ops, A.get(), C1.get(), n2));
    CHECK(SubInPlace(ops, A.get(), I.get(), n2));

    CHECK(CopyMatrix(A.get(), B.get(), n2, stream_));
    CHECK(CopyMatrix(scratch2.get(), C1.get(), n2, stream_));

    SMatrix S{};
    S.S_11 = scratch2.Release();
    S.S_12 = A.Release();
    S.S_21 = B.Release();
    S.S_22 = C1.Release();
    layer_smatrices_.push_back(S);
    return Status::kSuccess;
}

Status RCWA::AddLayer(Real thickness, const Complex* eps_conv)
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
        const Real thickness_thres = kTwo * std::min(dx, dy);
        if (thickness_thres > kZero && thickness > thickness_thres)
        {
            const Real ratio = thickness / thickness_thres;
            n_repeated = static_cast<int>(std::ceil(std::log2(ratio)));
            const Real denom = static_cast<Real>(1ULL << n_repeated);
            d_block = thickness / denom;
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

Status RCWA::MaterialConv(const Complex* material,
                          int64_t nx,
                          int64_t ny,
                          Complex** eps_conv,
                          cudaStream_t stream) const
{
    if (!material || !eps_conv || nx <= 0 || ny <= 0 || order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }
    if (nx > std::numeric_limits<int>::max() || ny > std::numeric_limits<int>::max())
    {
        return Status::kInvalidValue;
    }
    if (nx > (std::numeric_limits<int64_t>::max() / ny))
    {
        return Status::kInvalidValue;
    }
    if (order_N_ > (std::numeric_limits<int64_t>::max() / order_N_))
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;
    *eps_conv = nullptr;

    const int64_t fft_size = nx * ny;
    const size_t fft_bytes = static_cast<size_t>(fft_size) * sizeof(Complex);
    DeviceMatrix fft_buf;
    CHECK(fft_buf.Allocate(nx, ny));
    CHECK(StatusFromCuda(
        cudaMemcpyAsync(fft_buf.get(), material, fft_bytes, cudaMemcpyDeviceToDevice, use_stream)));

    CufftPlan plan;
#if USE_DOUBLE_PRECISION
    CHECK(StatusFromCufft(
        cufftPlan2d(&plan.handle, static_cast<int>(nx), static_cast<int>(ny), CUFFT_Z2Z)));
#else
    CHECK(StatusFromCufft(
        cufftPlan2d(&plan.handle, static_cast<int>(nx), static_cast<int>(ny), CUFFT_C2C)));
#endif
    CHECK(StatusFromCufft(cufftSetStream(plan.handle, use_stream)));

#if USE_DOUBLE_PRECISION
    CHECK(StatusFromCufft(cufftExecZ2Z(plan.handle,
                                       reinterpret_cast<cufftDoubleComplex*>(fft_buf.get()),
                                       reinterpret_cast<cufftDoubleComplex*>(fft_buf.get()),
                                       CUFFT_FORWARD)));
#else
    CHECK(StatusFromCufft(cufftExecC2C(plan.handle,
                                       reinterpret_cast<cufftComplex*>(fft_buf.get()),
                                       reinterpret_cast<cufftComplex*>(fft_buf.get()),
                                       CUFFT_FORWARD)));
#endif

    const double denom = static_cast<double>(nx) * static_cast<double>(ny);
    const Real scale = denom > 0.0 ? static_cast<Real>(1.0 / denom) : kZero;
    const int threads = 256;
    int blocks = static_cast<int>((fft_size + threads - 1) / threads);
    ScaleComplexKernel<<<blocks, threads, 0, use_stream>>>(fft_buf.get(), fft_size, scale);
    CHECK(StatusFromCuda(cudaGetLastError()));

    DeviceMatrix conv;
    CHECK(conv.Allocate(order_N_, order_N_));
    const int64_t total = order_N_ * order_N_;
    blocks = static_cast<int>((total + threads - 1) / threads);
    const int64_t oy_len = static_cast<int64_t>(2 * order_y_ + 1);
    BuildConvMatrixKernel<<<blocks, threads, 0, use_stream>>>(
        fft_buf.get(), nx, ny, oy_len, conv.get(), order_N_);
    CHECK(StatusFromCuda(cudaGetLastError()));

    *eps_conv = conv.Release();
    return Status::kSuccess;
}

Status RCWA::RSProd(const SMatrix& Sm,
                    const SMatrix& Sn,
                    int64_t n,
                    SMatrix* out,
                    cudaStream_t stream) const
{
    if (!out || !Sm.S_11 || !Sm.S_12 || !Sm.S_21 || !Sm.S_22 || !Sn.S_11 || !Sn.S_12 || !Sn.S_21 ||
        !Sn.S_22 || n <= 0)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;
    out->S_11 = nullptr;
    out->S_12 = nullptr;
    out->S_21 = nullptr;
    out->S_22 = nullptr;

    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    DeviceMatrix I;
    CHECK(I.Allocate(n, n));
    CHECK(MakeIdentity(I.get(), n, use_stream));

    DeviceMatrix Sm12Sn21;
    CHECK(Sm12Sn21.Allocate(n, n));
    CHECK(MatMul(ops, Sm.S_12, Sn.S_21, Sm12Sn21.get(), n));

    DeviceMatrix Sn21Sm12;
    CHECK(Sn21Sm12.Allocate(n, n));
    CHECK(MatMul(ops, Sn.S_21, Sm.S_12, Sn21Sm12.get(), n));

    DeviceMatrix T1;
    DeviceMatrix T2;
    CHECK(T1.Allocate(n, n));
    CHECK(SubMatrix(ops, I.get(), Sm12Sn21.get(), T1.get(), n));
    CHECK(T2.Allocate(n, n));
    CHECK(SubMatrix(ops, I.get(), Sn21Sm12.get(), T2.get(), n));

    DeviceMatrix X1;
    CHECK(X1.Allocate(n, n));
    const Status solve1 = ops.Solve(T1.get(), Sm.S_11, X1.get(), n);

    DeviceMatrix Sn21Sm11;
    CHECK(Sn21Sm11.Allocate(n, n));
    CHECK(MatMul(ops, Sn.S_21, Sm.S_11, Sn21Sm11.get(), n));
    DeviceMatrix X2;
    CHECK(X2.Allocate(n, n));
    const Status solve2 = ops.Solve(T2.get(), Sn21Sm11.get(), X2.get(), n);

    DeviceMatrix Sm12Sn22;
    CHECK(Sm12Sn22.Allocate(n, n));
    CHECK(MatMul(ops, Sm.S_12, Sn.S_22, Sm12Sn22.get(), n));
    DeviceMatrix X3;
    CHECK(X3.Allocate(n, n));
    const Status solve3 = ops.Solve(T1.get(), Sm12Sn22.get(), X3.get(), n);

    DeviceMatrix X4;
    CHECK(X4.Allocate(n, n));
    const Status solve4 = ops.Solve(T2.get(), Sn.S_22, X4.get(), n);

    if (solve1 != Status::kSuccess || solve2 != Status::kSuccess || solve3 != Status::kSuccess ||
        solve4 != Status::kSuccess)
    {
        DeviceMatrix S11;
        CHECK(S11.Allocate(n, n));
        CHECK(MatMul(ops, Sn.S_11, Sm.S_11, S11.get(), n));

        DeviceMatrix S12;
        CHECK(S12.Allocate(n, n));
        CHECK(CopyMatrix(Sn.S_12, S12.get(), n, use_stream));

        DeviceMatrix S21;
        CHECK(S21.Allocate(n, n));
        CHECK(CopyMatrix(Sm.S_21, S21.get(), n, use_stream));

        DeviceMatrix S22;
        CHECK(S22.Allocate(n, n));
        CHECK(StatusFromCuda(cudaMemsetAsync(
            S22.get(), 0, static_cast<size_t>(n) * n * sizeof(Complex), use_stream)));

        out->S_11 = S11.Release();
        out->S_12 = S12.Release();
        out->S_21 = S21.Release();
        out->S_22 = S22.Release();
        return Status::kSuccess;
    }

    DeviceMatrix S11;
    CHECK(S11.Allocate(n, n));
    CHECK(MatMul(ops, Sn.S_11, X1.get(), S11.get(), n));

    DeviceMatrix Sm22X2;
    CHECK(Sm22X2.Allocate(n, n));
    CHECK(MatMul(ops, Sm.S_22, X2.get(), Sm22X2.get(), n));
    DeviceMatrix S21;
    CHECK(S21.Allocate(n, n));
    CHECK(AddMatrix(ops, Sm.S_21, Sm22X2.get(), S21.get(), n));

    DeviceMatrix Sn11X3;
    CHECK(Sn11X3.Allocate(n, n));
    CHECK(MatMul(ops, Sn.S_11, X3.get(), Sn11X3.get(), n));
    DeviceMatrix S12;
    CHECK(S12.Allocate(n, n));
    CHECK(AddMatrix(ops, Sn.S_12, Sn11X3.get(), S12.get(), n));

    DeviceMatrix S22;
    CHECK(S22.Allocate(n, n));
    CHECK(MatMul(ops, Sm.S_22, X4.get(), S22.get(), n));

    out->S_11 = S11.Release();
    out->S_12 = S12.Release();
    out->S_21 = S21.Release();
    out->S_22 = S22.Release();
    return Status::kSuccess;
}

Status RCWA::SolveGlobalSMatrix(SMatrix* S,
                                const SMatrix* S_in,
                                const SMatrix* S_out,
                                cudaStream_t stream) const
{
    if (!S || order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }
    const int64_t n = 2 * order_N_;

    cudaStream_t use_stream = stream ? stream : stream_;
    S->S_11 = nullptr;
    S->S_12 = nullptr;
    S->S_21 = nullptr;
    S->S_22 = nullptr;

    SMatrix current{};
    if (layer_smatrices_.empty())
    {
        DeviceMatrix S11;
        DeviceMatrix S12;
        DeviceMatrix S21;
        DeviceMatrix S22;
        CHECK(S11.Allocate(n, n));
        CHECK(S12.Allocate(n, n));
        CHECK(S21.Allocate(n, n));
        CHECK(S22.Allocate(n, n));
        CHECK(MakeIdentity(S11.get(), n, use_stream));
        CHECK(MakeIdentity(S22.get(), n, use_stream));
        CHECK(StatusFromCuda(cudaMemsetAsync(
            S12.get(), 0, static_cast<size_t>(n) * n * sizeof(Complex), use_stream)));
        CHECK(StatusFromCuda(cudaMemsetAsync(
            S21.get(), 0, static_cast<size_t>(n) * n * sizeof(Complex), use_stream)));
        current.S_11 = S11.Release();
        current.S_12 = S12.Release();
        current.S_21 = S21.Release();
        current.S_22 = S22.Release();
    }
    else
    {
        const SMatrix& first = layer_smatrices_[0];
        if (!first.S_11 || !first.S_12 || !first.S_21 || !first.S_22)
        {
            return Status::kInvalidValue;
        }

        DeviceMatrix S11;
        DeviceMatrix S12;
        DeviceMatrix S21;
        DeviceMatrix S22;
        CHECK(S11.Allocate(n, n));
        CHECK(S12.Allocate(n, n));
        CHECK(S21.Allocate(n, n));
        CHECK(S22.Allocate(n, n));
        CHECK(CopyMatrix(first.S_11, S11.get(), n, use_stream));
        CHECK(CopyMatrix(first.S_12, S12.get(), n, use_stream));
        CHECK(CopyMatrix(first.S_21, S21.get(), n, use_stream));
        CHECK(CopyMatrix(first.S_22, S22.get(), n, use_stream));
        current.S_11 = S11.Release();
        current.S_12 = S12.Release();
        current.S_21 = S21.Release();
        current.S_22 = S22.Release();
    }

    for (size_t i = 1; i < layer_smatrices_.size(); ++i)
    {
        const SMatrix& next_layer = layer_smatrices_[i];
        SMatrix next{};
        CHECK(RSProd(current, next_layer, n, &next, use_stream));
        SAFEFREE(current.S_11);
        SAFEFREE(current.S_12);
        SAFEFREE(current.S_21);
        SAFEFREE(current.S_22);
        current = next;
    }

    if (S_in)
    {
        if (!S_in->S_11 || !S_in->S_12 || !S_in->S_21 || !S_in->S_22)
        {
            SAFEFREE(current.S_11);
            SAFEFREE(current.S_12);
            SAFEFREE(current.S_21);
            SAFEFREE(current.S_22);
            return Status::kInvalidValue;
        }
        SMatrix next{};
        CHECK(RSProd(*S_in, current, n, &next, use_stream));
        SAFEFREE(current.S_11);
        SAFEFREE(current.S_12);
        SAFEFREE(current.S_21);
        SAFEFREE(current.S_22);
        current = next;
    }

    if (S_out)
    {
        if (!S_out->S_11 || !S_out->S_12 || !S_out->S_21 || !S_out->S_22)
        {
            SAFEFREE(current.S_11);
            SAFEFREE(current.S_12);
            SAFEFREE(current.S_21);
            SAFEFREE(current.S_22);
            return Status::kInvalidValue;
        }
        SMatrix next{};
        CHECK(RSProd(current, *S_out, n, &next, use_stream));
        SAFEFREE(current.S_11);
        SAFEFREE(current.S_12);
        SAFEFREE(current.S_21);
        SAFEFREE(current.S_22);
        current = next;
    }

    S->S_11 = current.S_11;
    S->S_12 = current.S_12;
    S->S_21 = current.S_21;
    S->S_22 = current.S_22;
    return Status::kSuccess;
}

Status RCWA::SolveGlobalSMatrix()
{
    FreeGlobalSMatrix();
    const Status status = SolveGlobalSMatrix(&global_smatrix_, nullptr, nullptr, stream_);
    if (status != Status::kSuccess)
    {
        FreeGlobalSMatrix();
    }
    return status;
}

Status RCWA::FieldXY(const Real* x_axis,
                     int64_t x_count,
                     const Real* y_axis,
                     int64_t y_count,
                     Complex** Ex,
                     Complex** Ey,
                     Complex** Ez,
                     cudaStream_t stream) const
{
    if (!x_axis || !y_axis || !Ex || !Ey || !Ez || x_count <= 0 || y_count <= 0)
    {
        return Status::kInvalidValue;
    }
    if (!source_ei_ || !global_smatrix_.S_11 || !global_smatrix_.S_12 || !global_smatrix_.S_21 ||
        !global_smatrix_.S_22 || !kx_diag_ || !ky_diag_ || order_N_ <= 0)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;
    *Ex = nullptr;
    *Ey = nullptr;
    *Ez = nullptr;

    const int64_t n = order_N_;
    const int64_t n2 = 2 * n;
    const Complex* S_port = nullptr;
    if (angle_layer_input_)
    {
        S_port = source_forward_ ? global_smatrix_.S_21 : global_smatrix_.S_22;
    }
    else
    {
        S_port = source_forward_ ? global_smatrix_.S_11 : global_smatrix_.S_12;
    }
    if (!S_port)
    {
        return Status::kInvalidValue;
    }

    DeviceRealBuffer dx;
    DeviceRealBuffer dy;
    CHECK(dx.Allocate(static_cast<size_t>(x_count)));
    CHECK(dy.Allocate(static_cast<size_t>(y_count)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(dx.get(),
                                         x_axis,
                                         static_cast<size_t>(x_count) * sizeof(Real),
                                         cudaMemcpyHostToDevice,
                                         use_stream)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(dy.get(),
                                         y_axis,
                                         static_cast<size_t>(y_count) * sizeof(Real),
                                         cudaMemcpyHostToDevice,
                                         use_stream)));

    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    // Exy = S_port * E_i (diffraction amplitudes), Ez = (Kx*Ex + Ky*Ey)/Kz,
    // field(x,y) = sum_mn E_mn * exp(i * omega * (Kx*x + Ky*y)).
    DeviceMatrix Exy;
    CHECK(Exy.Allocate(n2, 1));
    CHECK(ops.MatMul(S_port, source_ei_, Exy.get(), n2, 1, n2));

    DeviceVector Kz;
    CHECK(Kz.Allocate(static_cast<size_t>(n)));
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const Complex eps = angle_layer_input_ ? eps_in_ : eps_out_;
    BuildKzKernel<<<blocks, threads, 0, use_stream>>>(
        kx_diag_, ky_diag_, eps, angle_layer_input_ ? 1 : 0, Kz.get(), n);
    CHECK(StatusFromCuda(cudaGetLastError()));

    DeviceVector Ezmn;
    CHECK(Ezmn.Allocate(static_cast<size_t>(n)));
    BuildEzKernel<<<blocks, threads, 0, use_stream>>>(
        kx_diag_, ky_diag_, Kz.get(), Exy.get(), Exy.get() + n, Ezmn.get(), n);
    CHECK(StatusFromCuda(cudaGetLastError()));

    DeviceMatrix ExMat;
    DeviceMatrix EyMat;
    DeviceMatrix EzMat;
    CHECK(ExMat.Allocate(x_count, y_count));
    CHECK(EyMat.Allocate(x_count, y_count));
    CHECK(EzMat.Allocate(x_count, y_count));

    const int64_t total = x_count * y_count;
    const int grid = static_cast<int>((total + threads - 1) / threads);
    FieldXYKernel<<<grid, threads, 0, use_stream>>>(kx_diag_,
                                                    ky_diag_,
                                                    Exy.get(),
                                                    Exy.get() + n,
                                                    Ezmn.get(),
                                                    dx.get(),
                                                    dy.get(),
                                                    x_count,
                                                    y_count,
                                                    omega_,
                                                    n,
                                                    ExMat.get(),
                                                    EyMat.get(),
                                                    EzMat.get());
    CHECK(StatusFromCuda(cudaGetLastError()));

    *Ex = ExMat.Release();
    *Ey = EyMat.Release();
    *Ez = EzMat.Release();
    return Status::kSuccess;
}

void RCWA::FreeDeviceTensor(Complex* ptr) const { SAFEFREE(ptr); }

Status RCWA::BuildPQ(const Complex* eps_conv, int64_t n, Complex* P, Complex* Q)
{
    if (!eps_conv || !P || !Q || n <= 0)
    {
        return Status::kInvalidValue;
    }

    ComplexMatrixOps ops(stream_);
    CHECK(ops.SetStream(stream_));

    // P/Q block structure (n x n blocks, Kx/Ky diagonal):
    // P = [ Kx*E^{-1}*Ky , Ky*E^{-1}*Ky - I;
    //       I - Kx*E^{-1}*Kx , -Ky*E^{-1}*Kx ]
    // Q = [ -Kx*Ky , E - Ky^2;
    //       Kx^2 - E, Kx*Ky ]
    DeviceVector KxKxDiag;
    DeviceVector KyKyDiag;
    DeviceVector KxKyDiag;
    CHECK(KxKxDiag.Allocate(static_cast<size_t>(n)));
    CHECK(KyKyDiag.Allocate(static_cast<size_t>(n)));
    CHECK(KxKyDiag.Allocate(static_cast<size_t>(n)));
    CHECK(MultiplyDiagonalVectors(kx_diag_, kx_diag_, KxKxDiag.get(), n, stream_));
    CHECK(MultiplyDiagonalVectors(ky_diag_, ky_diag_, KyKyDiag.get(), n, stream_));
    CHECK(MultiplyDiagonalVectors(kx_diag_, ky_diag_, KxKyDiag.get(), n, stream_));

    DeviceMatrix KxMat;
    DeviceMatrix KyMat;
    CHECK(KxMat.Allocate(n, n));
    CHECK(KyMat.Allocate(n, n));
    CHECK(FillDiagonalFromVector(kx_diag_, n, KxMat.get(), stream_));
    CHECK(FillDiagonalFromVector(ky_diag_, n, KyMat.get(), stream_));

    DeviceMatrix EiKx;
    DeviceMatrix EiKy;
    CHECK(EiKx.Allocate(n, n));
    CHECK(EiKy.Allocate(n, n));

    const Status solve_kx = ops.Solve(eps_conv, KxMat.get(), EiKx.get(), n);
    const Status solve_ky = ops.Solve(eps_conv, KyMat.get(), EiKy.get(), n);
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
        CHECK(FillDiagonalFromVector(KxKyDiag.get(), n, KxEiKy.get(), stream_));
        CHECK(FillDiagonalFromVector(KxKyDiag.get(), n, KyEiKx.get(), stream_));
        CHECK(FillDiagonalFromVector(KxKxDiag.get(), n, KxEiKx.get(), stream_));
        CHECK(FillDiagonalFromVector(KyKyDiag.get(), n, KyEiKy.get(), stream_));
    }
    else
    {
        CHECK(ops.DiagLeftMatMul(kx_diag_, EiKy.get(), KxEiKy.get(), n));
        CHECK(ops.DiagLeftMatMul(ky_diag_, EiKx.get(), KyEiKx.get(), n));
        CHECK(ops.DiagLeftMatMul(kx_diag_, EiKx.get(), KxEiKx.get(), n));
        CHECK(ops.DiagLeftMatMul(ky_diag_, EiKy.get(), KyEiKy.get(), n));
    }

    DeviceMatrix P12;
    DeviceMatrix P21;
    DeviceMatrix Q12;
    DeviceMatrix Q21;
    CHECK(P12.Allocate(n, n));
    CHECK(CopyMatrix(KyEiKy.get(), P12.get(), n, stream_));
    CHECK(AddDiagonalConstantInPlace(P12.get(), n, make_complex(kNegOne, kZero), stream_));
    CHECK(P21.Allocate(n, n));
    CHECK(CopyMatrix(KxEiKx.get(), P21.get(), n, stream_));
    CHECK(ScaleInPlace(P21.get(), n, make_complex(kNegOne, kZero), stream_));
    CHECK(AddDiagonalConstantInPlace(P21.get(), n, make_complex(kOne, kZero), stream_));
    CHECK(Q12.Allocate(n, n));
    CHECK(CopyMatrix(eps_conv, Q12.get(), n, stream_));
    CHECK(AddDiagonalInPlace(Q12.get(), KyKyDiag.get(), n, make_complex(kNegOne, kZero), stream_));
    CHECK(Q21.Allocate(n, n));
    CHECK(CopyMatrix(eps_conv, Q21.get(), n, stream_));
    CHECK(ScaleInPlace(Q21.get(), n, make_complex(kNegOne, kZero), stream_));
    CHECK(AddDiagonalInPlace(Q21.get(), KxKxDiag.get(), n, make_complex(kOne, kZero), stream_));

    const Complex one = make_complex(kOne, kZero);
    const Complex neg_one = make_complex(kNegOne, kZero);

    CHECK(StatusFromCuda(cudaMemsetAsync(
        Q, 0, static_cast<size_t>(2 * n) * static_cast<size_t>(2 * n) * sizeof(Complex), stream_)));

    CHECK(CopyBlock(KxEiKy.get(), n, P, 2 * n, 0, 0, one, stream_));
    CHECK(CopyBlock(P12.get(), n, P, 2 * n, 0, n, one, stream_));
    CHECK(CopyBlock(P21.get(), n, P, 2 * n, n, 0, one, stream_));
    CHECK(CopyBlock(KyEiKx.get(), n, P, 2 * n, n, n, neg_one, stream_));

    CHECK(CopyDiagonalBlock(KxKyDiag.get(), n, Q, 2 * n, 0, 0, neg_one, stream_));
    CHECK(CopyBlock(Q12.get(), n, Q, 2 * n, 0, n, one, stream_));
    CHECK(CopyBlock(Q21.get(), n, Q, 2 * n, n, 0, one, stream_));
    CHECK(CopyDiagonalBlock(KxKyDiag.get(), n, Q, 2 * n, n, n, one, stream_));

    return Status::kSuccess;
}

Status RCWA::RepeatSMatrix(SMatrix* S, int repeats, int64_t n)
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
    CHECK(MakeIdentity(I.get(), n, stream_));

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
        CHECK(S12S12.Allocate(n, n));
        CHECK(MatMul(ops, S->S_12, S->S_12, S12S12.get(), n));
        CHECK(R.Allocate(n, n));
        CHECK(SubMatrix(ops, I.get(), S12S12.get(), R.get(), n));

        CHECK(S12S11.Allocate(n, n));
        CHECK(MatMul(ops, S->S_12, S->S_11, S12S11.get(), n));
        CHECK(X.Allocate(n, n));
        CHECK(ops.Solve(R.get(), S12S11.get(), X.get(), n));
        CHECK(S11X.Allocate(n, n));
        CHECK(MatMul(ops, S->S_11, X.get(), S11X.get(), n));
        CHECK(newS12.Allocate(n, n));
        CHECK(AddMatrix(ops, S->S_12, S11X.get(), newS12.get(), n));

        CHECK(Y.Allocate(n, n));
        CHECK(ops.Solve(R.get(), S->S_11, Y.get(), n));
        CHECK(newS11.Allocate(n, n));
        CHECK(MatMul(ops, S->S_11, Y.get(), newS11.get(), n));

        CHECK(CopyMatrix(newS11.get(), S->S_11, n, stream_));
        CHECK(CopyMatrix(newS12.get(), S->S_12, n, stream_));
    }

    CHECK(CopyMatrix(S->S_12, S->S_21, n, stream_));
    CHECK(CopyMatrix(S->S_11, S->S_22, n, stream_));
    return Status::kSuccess;
}
