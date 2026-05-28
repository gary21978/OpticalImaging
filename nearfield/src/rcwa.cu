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

static constexpr double kExpConstantOrder18[5][5] = {
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

static constexpr double kExpConstantOrder12[4][4] = {
    {-0.01860232051462055322,
     -0.00500702322573317730,
     -0.57342012296052226390,
     -0.13339969394389205970},
    {4.60000000000000000000,
     0.99287510353848683614,
     -0.13244556105279963884,
     0.00172990000000000000},
    {0.21169311829980944294,
     0.15822438471572672537,
     0.16563516943672741501,
     0.01078627793157924250},
    {0.0, -0.13181061013830184015, -0.02027855540589259079, -0.00675951846863086359}};

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

__global__ void BuildHomogeneousSBlocksKernel(const Complex* __restrict__ kx_diag,
                                              const Complex* __restrict__ ky_diag,
                                              Complex eps,
                                              Real k0d,
                                              Complex* __restrict__ S11,
                                              Complex* __restrict__ S12,
                                              Complex* __restrict__ S21,
                                              Complex* __restrict__ S22,
                                              int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    const Complex one = make_complex(1.0, 0.0);
    const Complex neg_one = make_complex(-1.0, 0.0);
    const Complex two = make_complex(2.0, 0.0);
    const Complex zero = make_complex(0.0, 0.0);

    const Complex kx = kx_diag[idx];
    const Complex ky = ky_diag[idx];
    const Complex kx2 = ComplexMul(kx, kx);
    const Complex ky2 = ComplexMul(ky, ky);
    const Complex kxky = ComplexMul(kx, ky);
    const Complex sum = ComplexAdd(kx2, ky2);

    const Complex kz = ComplexSqrt_RCWA(ComplexSub(eps, sum));
    const Complex kz0 = ComplexSqrt_RCWA(ComplexSub(one, sum));

    const Real tiny = static_cast<Real>(1e-12);
    if (ComplexAbs(kz) <= tiny || ComplexAbs(kz0) <= tiny)
    {
        return;
    }

    const Complex kz_over_kz0 = ComplexDiv(kz, kz0);
    const Complex kz0_over_kz = ComplexDiv(kz0, kz);
    const Complex tau = ComplexSub(kz_over_kz0, kz0_over_kz);

    const Complex a = ComplexSub(ComplexMul(tau, kx2), kz_over_kz0);
    const Complex b = ComplexMul(tau, kxky);
    const Complex d = ComplexSub(ComplexMul(tau, ky2), kz_over_kz0);
    const Complex2x2 U(a, b, b, d);

    const Complex kzt = ComplexMul(kz, make_complex(k0d, 0.0));
    const Complex u = ComplexExpI(kzt);
    const Complex2x2 I(one, zero, zero, one);

    Complex2x2 A = Mat2x2Sub(I, U);
    Complex2x2 tmp = Mat2x2Add(I, U);
    Complex2x2 B = Mat2x2Scale(tmp, u);

    tmp = Mat2x2Inv(A);
    const Complex2x2 G = Mat2x2Mul(B, tmp);
    tmp = Mat2x2Mul(G, B);
    A = Mat2x2Sub(A, tmp);
    tmp = Mat2x2Inv(A);
    const Complex2x2 C = Mat2x2Scale(tmp, two);
    tmp = Mat2x2Mul(C, G);
    const Complex2x2 D = Mat2x2Scale(tmp, neg_one);
    const Complex2x2 S11_block = Mat2x2Add(Mat2x2Scale(C, u), D);
    tmp = Mat2x2Scale(D, u);
    tmp = Mat2x2Add(tmp, C);
    const Complex2x2 S12_block = Mat2x2Sub(tmp, I);

    const int64_t row0 = idx;
    const int64_t row1 = idx + n;
    const int64_t base_col0 = 2 * row0 * n;
    const int64_t base_col1 = 2 * row1 * n;
    const int64_t i00 = base_col0 + row0;
    const int64_t i10 = base_col0 + row1;
    const int64_t i01 = base_col1 + row0;
    const int64_t i11 = base_col1 + row1;

    S11[i00] = S11_block.m00;
    S11[i10] = S11_block.m10;
    S11[i01] = S11_block.m01;
    S11[i11] = S11_block.m11;

    S12[i00] = S12_block.m00;
    S12[i10] = S12_block.m10;
    S12[i01] = S12_block.m01;
    S12[i11] = S12_block.m11;

    S21[i00] = S12_block.m00;
    S21[i10] = S12_block.m10;
    S21[i01] = S12_block.m01;
    S21[i11] = S12_block.m11;

    S22[i00] = S11_block.m00;
    S22[i10] = S11_block.m10;
    S22[i01] = S11_block.m01;
    S22[i11] = S11_block.m11;
}

__global__ void BuildPortSBlocksKernel(const Complex* __restrict__ v0_11,
                                       const Complex* __restrict__ v0_12,
                                       const Complex* __restrict__ v0_21,
                                       const Complex* __restrict__ v0_22,
                                       const Complex* __restrict__ vp_11,
                                       const Complex* __restrict__ vp_12,
                                       const Complex* __restrict__ vp_21,
                                       const Complex* __restrict__ vp_22,
                                       int input_port,
                                       Complex* __restrict__ S11,
                                       Complex* __restrict__ S12,
                                       Complex* __restrict__ S21,
                                       Complex* __restrict__ S22,
                                       int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    const Complex zero = make_complex(0.0, 0.0);
    const Complex one = make_complex(1.0, 0.0);
    const Complex two = make_complex(2.0, 0.0);
    const Complex minus_one = make_complex(-1.0, 0.0);

    const Complex2x2 V0(v0_11[idx], v0_21[idx], v0_12[idx], v0_22[idx]);
    const Complex2x2 Vp(vp_11[idx], vp_21[idx], vp_12[idx], vp_22[idx]);
    const Complex2x2 I(one, zero, zero, one);

    const Complex2x2 Vsum = Mat2x2Add(Vp, V0);
    const Complex2x2 Vdiff = Mat2x2Sub(Vp, V0);

    const Complex2x2 Vsum_inv = Mat2x2Inv(Vsum);

    Complex2x2 S11_block;
    Complex2x2 S12_block;
    Complex2x2 S21_block;
    Complex2x2 S22_block;

    const Complex2x2 coupling = Mat2x2Mul(Vsum_inv, Vdiff);
    if (input_port)
    {
        S11_block = Mat2x2Scale(Mat2x2Mul(Vsum_inv, Vp), two);
        S12_block = Mat2x2Scale(coupling, minus_one);
        S21_block = coupling;
        S22_block = Mat2x2Scale(Mat2x2Mul(Vsum_inv, V0), two);
    }
    else
    {
        S11_block = Mat2x2Scale(Mat2x2Mul(Vsum_inv, V0), two);
        S12_block = coupling;
        S21_block = Mat2x2Scale(coupling, minus_one);
        S22_block = Mat2x2Scale(Mat2x2Mul(Vsum_inv, Vp), two);
    }

    const int64_t row0 = idx;
    const int64_t row1 = idx + n;
    const int64_t base_col0 = 2 * row0 * n;
    const int64_t base_col1 = 2 * row1 * n;
    const int64_t i00 = base_col0 + row0;
    const int64_t i10 = base_col0 + row1;
    const int64_t i01 = base_col1 + row0;
    const int64_t i11 = base_col1 + row1;

    S11[i00] = S11_block.m00;
    S11[i10] = S11_block.m10;
    S11[i01] = S11_block.m01;
    S11[i11] = S11_block.m11;

    S12[i00] = S12_block.m00;
    S12[i10] = S12_block.m10;
    S12[i01] = S12_block.m01;
    S12[i11] = S12_block.m11;

    S21[i00] = S21_block.m00;
    S21[i10] = S21_block.m10;
    S21[i01] = S21_block.m01;
    S21[i11] = S21_block.m11;

    S22[i00] = S22_block.m00;
    S22[i10] = S22_block.m10;
    S22[i01] = S22_block.m01;
    S22[i11] = S22_block.m11;
}

__global__ void ScaleComplexKernel(Complex* data, int64_t n, Real scale)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    const Complex value = data[idx];
    data[idx] = make_complex(ComplexReal(value) * scale, ComplexImag(value) * scale);
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
    Complex kzz = ComplexSqrt_RCWA(diff);

    if (!input_layer)
    {
        kzz = ComplexMul(make_complex(-1.0, 0.0), kzz);
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
    ez[idx] = ComplexDiv(numerator, kz[idx]);
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
    const Complex omega_c = make_complex(omega, 0.0);
    const Complex x_c = make_complex(x, 0.0);
    const Complex y_c = make_complex(y, 0.0);

    Complex sum_ex = make_complex(0.0, 0.0);
    Complex sum_ey = make_complex(0.0, 0.0);
    Complex sum_ez = make_complex(0.0, 0.0);
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

    fx %= nx;
    fx += (fx < 0) ? nx : 0;
    fy %= ny;
    fy += (fy < 0) ? ny : 0;

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
    Complex kz = ComplexSqrt_RCWA(diff);
    const Complex kz_inv = ComplexReciprocal(kz);
    const Complex kxky = ComplexMul(kx, ky);
    v0_11[idx] = ComplexMul(make_complex(-1.0, 0.0), ComplexMul(kxky, kz_inv));
    v0_12[idx] = ComplexMul(make_complex(-1.0, 0.0), ComplexAdd(kz, ComplexMul(ky2, kz_inv)));
    v0_21[idx] = ComplexAdd(kz, ComplexMul(kx2, kz_inv));
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
    kx_diag[idx] = make_complex(kx, 0.0);
    ky_diag[idx] = make_complex(ky, 0.0);
}

static Status
ApplyV0Left(const VMatrix& V, const Complex* B, int64_t n, DeviceMatrix* out, cudaStream_t stream)
{
    if (!B || !out || !V.V_11 || !V.V_12 || !V.V_21 || !V.V_22 || (n % 2) != 0)
    {
        return Status::kInvalidValue;
    }
    CHECK(out->Allocate(n, n));
    int64_t n2 = n / 2;
    int64_t total = n2 * n;
    int blocks = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    V0LeftMulKernel<<<blocks, CUDA_THREADS, 0, stream>>>(
        V.V_11, V.V_12, V.V_21, V.V_22, B, out->get(), n, n2);
    return StatusFromCuda(cudaGetLastError());
}

static Status
ApplyV0Right(const VMatrix& V, const Complex* B, int64_t n, DeviceMatrix* out, cudaStream_t stream)
{
    if (!B || !out || !V.V_11 || !V.V_12 || !V.V_21 || !V.V_22 || (n % 2) != 0)
    {
        return Status::kInvalidValue;
    }
    CHECK(out->Allocate(n, n));
    int64_t n2 = n / 2;
    int64_t total = n2 * n;
    int blocks = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    V0RightMulKernel<<<blocks, CUDA_THREADS, 0, stream>>>(
        V.V_11, V.V_12, V.V_21, V.V_22, B, out->get(), n, n2);
    return StatusFromCuda(cudaGetLastError());
}

RCWA::RCWA(Real freq, int order_x, int order_y, Real Lx, Real Ly, cudaStream_t stream)
    : freq_(freq), omega_(static_cast<Real>(6.28318530717958647692) * freq), Lx_(Lx), Ly_(Ly),
      order_x_(std::max(order_x, 0)), order_y_(std::max(order_y, 0)), stream_(stream)
{
    order_N_ = static_cast<int64_t>((2 * order_x_ + 1) * (2 * order_y_ + 1));
    Gx_norm_ = 1.0 / (Lx_ * freq_);
    Gy_norm_ = 1.0 / (Ly_ * freq_);
}

RCWA::~RCWA()
{
    FreeSMatrix(&stack_smatrix_);
    FreeKVectors();
    FreeGlobalSMatrix();
    FreeSource();
}

Status RCWA::SetStream(cudaStream_t stream)
{
    stream_ = stream;
    return Status::kSuccess;
}

void RCWA::FreeSMatrix(SMatrix* matrix) const
{
    if (!matrix)
    {
        return;
    }
    SAFEFREE(matrix->S_11);
    SAFEFREE(matrix->S_12);
    SAFEFREE(matrix->S_21);
    SAFEFREE(matrix->S_22);
}

bool RCWA::HasSMatrix(const SMatrix& matrix) const
{
    return matrix.S_11 && matrix.S_12 && matrix.S_21 && matrix.S_22;
}

Status RCWA::CopySMatrix(const SMatrix& src, int64_t n, SMatrix* dst, cudaStream_t stream) const
{
    if (!dst || !HasSMatrix(src) || n <= 0)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;
    FreeSMatrix(dst);

    DeviceMatrix S11;
    DeviceMatrix S12;
    DeviceMatrix S21;
    DeviceMatrix S22;
    CHECK(S11.Allocate(n, n));
    CHECK(S12.Allocate(n, n));
    CHECK(S21.Allocate(n, n));
    CHECK(S22.Allocate(n, n));
    CHECK(CopyMatrix(src.S_11, S11.get(), n, use_stream));
    CHECK(CopyMatrix(src.S_12, S12.get(), n, use_stream));
    CHECK(CopyMatrix(src.S_21, S21.get(), n, use_stream));
    CHECK(CopyMatrix(src.S_22, S22.get(), n, use_stream));
    dst->S_11 = S11.Release();
    dst->S_12 = S12.Release();
    dst->S_21 = S21.Release();
    dst->S_22 = S22.Release();
    return Status::kSuccess;
}

Status RCWA::BuildPortSMatrix(Complex eps, bool input_port, SMatrix* S, cudaStream_t stream) const
{
    if (!S || !kvectors_ready_ || !kx_diag_ || !ky_diag_ || !V0_.V_11 || !V0_.V_12 || !V0_.V_21 ||
        !V0_.V_22)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;
    const int64_t n = order_N_;
    const int64_t n2 = 2 * n;

    S->S_11 = nullptr;
    S->S_12 = nullptr;
    S->S_21 = nullptr;
    S->S_22 = nullptr;

    DeviceVector port11;
    DeviceVector port12;
    DeviceVector port21;
    DeviceVector port22;
    CHECK(port11.Allocate(static_cast<size_t>(n)));
    CHECK(port12.Allocate(static_cast<size_t>(n)));
    CHECK(port21.Allocate(static_cast<size_t>(n)));
    CHECK(port22.Allocate(static_cast<size_t>(n)));

    const int blocks = static_cast<int>((n + CUDA_THREADS - 1) / CUDA_THREADS);
    BuildV0Kernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(
        kx_diag_, ky_diag_, eps, port11.get(), port12.get(), port21.get(), port22.get(), n);
    CHECK(StatusFromCuda(cudaGetLastError()));

    DeviceMatrix S11;
    DeviceMatrix S12;
    DeviceMatrix S21;
    DeviceMatrix S22;
    CHECK(S11.Allocate(n2, n2));
    CHECK(S12.Allocate(n2, n2));
    CHECK(S21.Allocate(n2, n2));
    CHECK(S22.Allocate(n2, n2));

    const size_t matrix_bytes = static_cast<size_t>(n2) * static_cast<size_t>(n2) * sizeof(Complex);
    CHECK(StatusFromCuda(cudaMemsetAsync(S11.get(), 0, matrix_bytes, use_stream)));
    CHECK(StatusFromCuda(cudaMemsetAsync(S12.get(), 0, matrix_bytes, use_stream)));
    CHECK(StatusFromCuda(cudaMemsetAsync(S21.get(), 0, matrix_bytes, use_stream)));
    CHECK(StatusFromCuda(cudaMemsetAsync(S22.get(), 0, matrix_bytes, use_stream)));

    BuildPortSBlocksKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(V0_.V_11,
                                                                    V0_.V_12,
                                                                    V0_.V_21,
                                                                    V0_.V_22,
                                                                    port11.get(),
                                                                    port12.get(),
                                                                    port21.get(),
                                                                    port22.get(),
                                                                    input_port ? 1 : 0,
                                                                    S11.get(),
                                                                    S12.get(),
                                                                    S21.get(),
                                                                    S22.get(),
                                                                    n);
    CHECK(StatusFromCuda(cudaGetLastError()));

    S->S_11 = S11.Release();
    S->S_12 = S12.Release();
    S->S_21 = S21.Release();
    S->S_22 = S22.Release();
    return Status::kSuccess;
}

Status RCWA::CommitLayerSMatrix(SMatrix* layer, int64_t n)
{
    if (!layer || !HasSMatrix(*layer) || n <= 0)
    {
        return Status::kInvalidValue;
    }

    FreeGlobalSMatrix();

    if (!HasSMatrix(stack_smatrix_))
    {
        stack_smatrix_ = *layer;
        *layer = {};
        return Status::kSuccess;
    }

    cudaStream_t use_stream = stream_;
    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    SMatrix next{};
    const Status status = RSProd(stack_smatrix_, *layer, n, &next, ops, use_stream);
    if (status != Status::kSuccess)
    {
        FreeSMatrix(&next);
        FreeSMatrix(layer);
        return status;
    }

    FreeSMatrix(&stack_smatrix_);
    FreeSMatrix(layer);
    stack_smatrix_ = next;
    return Status::kSuccess;
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

void RCWA::FreeGlobalSMatrix() { FreeSMatrix(&global_smatrix_); }

void RCWA::FreeSource() { SAFEFREE(source_ei_); }

Status RCWA::SetIncidentAngles(Real inc_ang, Real azi_ang, Real eps_ref)
{
    return BuildKxKy(inc_ang, azi_ang, eps_ref);
}

Status RCWA::SetIncidentAngles(Real inc_ang, Real azi_ang, const char* angle_layer)
{
    angle_layer_input_ = IsInputLayerName(angle_layer);
    const Complex eps = angle_layer_input_ ? eps_in_ : eps_out_;
    const Real sqrt_eps = ComplexReal(ComplexSqrt(eps));
    const Real eps_ref = sqrt_eps * sqrt_eps;
    return BuildKxKy(inc_ang, azi_ang, eps_ref);
}

Status RCWA::AddInputLayer(Real eps) { return AddInputLayer(make_complex(eps, 0.0)); }

Status RCWA::AddInputLayer(Complex eps)
{
    eps_in_ = eps;
    FreeGlobalSMatrix();
    return Status::kSuccess;
}

Status RCWA::AddOutputLayer(Real eps) { return AddOutputLayer(make_complex(eps, 0.0)); }

Status RCWA::AddOutputLayer(Complex eps)
{
    eps_out_ = eps;
    FreeGlobalSMatrix();
    return Status::kSuccess;
}

Status RCWA::SourcePlanewave(Real ex_amp, Real ey_amp, const char* direction)
{
    return SourcePlanewave(make_complex(ex_amp, 0.0), make_complex(ey_amp, 0.0), direction);
}

Status RCWA::SourcePlanewave(Complex ex_amp, Complex ey_amp, const char* direction)
{
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
    FreeKVectors();

    const Real sqrt_eps = std::sqrt(static_cast<double>(eps_ref));
    const Real kx0_norm = sqrt_eps * std::sin(inc_ang) * std::cos(azi_ang);
    const Real ky0_norm = sqrt_eps * std::sin(inc_ang) * std::sin(azi_ang);

    CHECK(StatusFromCuda(cudaMalloc(&kx_diag_, static_cast<size_t>(order_N_) * sizeof(Complex))));
    CHECK(StatusFromCuda(cudaMalloc(&ky_diag_, static_cast<size_t>(order_N_) * sizeof(Complex))));

    const int blocks = static_cast<int>((order_N_ + CUDA_THREADS - 1) / CUDA_THREADS);
    BuildKxKyDiagKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(
        kx_diag_, ky_diag_, order_x_, order_y_, kx0_norm, ky0_norm, Gx_norm_, Gy_norm_, order_N_);
    CHECK(StatusFromCuda(cudaGetLastError()));

    CHECK(BuildV0(1.0));

    kvectors_ready_ = true;
    return Status::kSuccess;
}

Status RCWA::BuildV0(Real eps_ref)
{
    if (!kx_diag_ || !ky_diag_)
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

    const int blocks = static_cast<int>((order_N_ + CUDA_THREADS - 1) / CUDA_THREADS);
    BuildV0Kernel<<<blocks, CUDA_THREADS, 0, stream_>>>(kx_diag_,
                                                        ky_diag_,
                                                        make_complex(eps_ref, 0.0),
                                                        V0_.V_11,
                                                        V0_.V_12,
                                                        V0_.V_21,
                                                        V0_.V_22,
                                                        order_N_);
    CHECK(StatusFromCuda(cudaGetLastError()));

    return Status::kSuccess;
}

Status RCWA::ComputeTFromPQOrder18(const Complex* P,
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
    const Complex scale_pos = make_complex(static_cast<Real>(scale_A), 0.0);
    const Complex scale_neg = make_complex(static_cast<Real>(-scale_A), 0.0);

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
        const Complex c0 = make_complex(static_cast<Real>(kExpConstantOrder18[i][0]), 0.0);
        const Complex c1 = make_complex(static_cast<Real>(kExpConstantOrder18[i][1]), 0.0);
        const Complex c2 = make_complex(static_cast<Real>(kExpConstantOrder18[i][2]), 0.0);
        const Complex c3 = make_complex(static_cast<Real>(kExpConstantOrder18[i][3]), 0.0);
        const Complex c4 = make_complex(static_cast<Real>(kExpConstantOrder18[i][4]), 0.0);

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

    // I2, A_12, A_21 no longer needed after B coefficients are built
    I2.Reset();
    A_12.Reset();
    A_21.Reset();

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

    // Free B[0], B[3], B[4] — no longer needed after A9 blocks are built
    for (int i : {0, 3, 4})
    {
        B_11[i].Reset(); B_12[i].Reset();
        B_21[i].Reset(); B_22[i].Reset();
    }

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

    // Free B[1], B[2], A9 blocks, and temp matrices — none are used in squaring
    for (int i : {1, 2})
    {
        B_11[i].Reset(); B_12[i].Reset();
        B_21[i].Reset(); B_22[i].Reset();
    }
    A9_11.Reset(); A9_12.Reset(); A9_21.Reset(); A9_22.Reset();
    temp1.Reset(); temp2.Reset(); temp3.Reset();

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

Status RCWA::ComputeCosSinFromThetaOrder12(const Complex* Theta,
                                            int64_t n,
                                            Complex* Cos,
                                            Complex* Sin,
                                            cudaStream_t stream) const
{
    if (!Theta || !Cos || !Sin || n <= 0) { return Status::kInvalidValue; }

    cudaStream_t use_stream = stream ? stream : stream_;
    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    // Scaling factor
    double ThetaNorm = 0.0;
    CHECK(ComputeMatrixNorm1(Theta, n, &ThetaNorm, use_stream));
    int m = 0;
    if (ThetaNorm > 0.0)
    {
        m = static_cast<int>(std::ceil(std::log2(ThetaNorm)));
        if (m < 0) { m = 0; }
    }
    const Complex scale = make_complex(static_cast<Real>(std::pow(2.0, -m)), 0.0);
    const Complex neg_one = make_complex(-1.0, 0.0);

    // ── Phase 1: block powers  A¹, A², A³ ───────────────────────────────
    DeviceMatrix A1;
    CHECK(A1.Allocate(n, n));
    CHECK(ScaleMatrix(Theta, A1.get(), n, scale, use_stream));

    DeviceMatrix A2;
    CHECK(A2.Allocate(n, n));
    CHECK(MatMul(ops, A1.get(), A1.get(), A2.get(), n));

    DeviceMatrix A3;
    CHECK(A3.Allocate(n, n));
    CHECK(MatMul(ops, A1.get(), A2.get(), A3.get(), n));

    // ── Phase 2:  B₁⁽ⁱ⁾ = a₀ᵢ·I − a₂ᵢ·A²,  B₂⁽ⁱ⁾ = a₁ᵢ·A¹ − a₃ᵢ·A³  (i=0..3) ──
    DeviceMatrix I_mat;
    CHECK(I_mat.Allocate(n, n));
    CHECK(MakeIdentity(I_mat.get(), n, use_stream));

    DeviceMatrix scratch;
    CHECK(scratch.Allocate(n, n));

    DeviceMatrix B1[4], B2[4];
    for (int i = 0; i < 4; ++i)
    {
        const Complex a0 = make_complex(static_cast<Real>(kExpConstantOrder12[i][0]), 0.0);
        const Complex a1 = make_complex(static_cast<Real>(kExpConstantOrder12[i][1]), 0.0);
        const Complex a2 = make_complex(static_cast<Real>(kExpConstantOrder12[i][2]), 0.0);
        const Complex a3 = make_complex(static_cast<Real>(kExpConstantOrder12[i][3]), 0.0);

        CHECK(B1[i].Allocate(n, n));
        CHECK(ScaleMatrix(I_mat.get(), B1[i].get(), n, a0, use_stream));
        CHECK(ScaleMatrix(A2.get(), scratch.get(), n,
                          make_complex(-ComplexReal(a2), -ComplexImag(a2)), use_stream));
        CHECK(AddInPlace(ops, B1[i].get(), scratch.get(), n));

        CHECK(B2[i].Allocate(n, n));
        CHECK(ScaleMatrix(A1.get(), B2[i].get(), n, a1, use_stream));
        CHECK(ScaleMatrix(A3.get(), scratch.get(), n,
                          make_complex(-ComplexReal(a3), -ComplexImag(a3)), use_stream));
        CHECK(AddInPlace(ops, B2[i].get(), scratch.get(), n));
    }

    // A¹, A², A³, I are no longer needed after B construction —
    // reuse their buffers for A⁽⁶⁾₁, A⁽⁶⁾₂, C, S respectively.
    DeviceMatrix A6_1, A6_2;
    A6_1.Swap(A1);
    A6_2.Swap(A2);

    // ── Phase 3: A⁽⁶⁾₁₂ from B⁽³⁾, B⁽²⁾, then C, S ─────────────────────
    //
    //  A⁽⁶⁾₁ = B₁⁽³⁾·B₁⁽³⁾ − B₂⁽³⁾·B₂⁽³⁾ + B₁⁽²⁾
    //  A⁽⁶⁾₂ = B₂⁽³⁾·B₁⁽³⁾ + B₁⁽³⁾·B₂⁽³⁾ + B₂⁽²⁾
    CHECK(MatMul(ops, B1[3].get(), B1[3].get(), A6_1.get(), n));
    CHECK(MatMul(ops, B2[3].get(), B2[3].get(), scratch.get(), n));
    CHECK(ScaleMatrix(scratch.get(), scratch.get(), n, neg_one, use_stream));
    CHECK(AddInPlace(ops, A6_1.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_1.get(), B1[2].get(), n));

    CHECK(MatMul(ops, B2[3].get(), B1[3].get(), A6_2.get(), n));
    CHECK(MatMul(ops, B1[3].get(), B2[3].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_2.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_2.get(), B2[2].get(), n));

    // B⁽²⁾ and B⁽³⁾ are no longer needed — reuse as workspace for C/S.
    //
    //  C = B₁⁽⁰⁾ + (B₁⁽¹⁾ + A⁽⁶⁾₁)·A⁽⁶⁾₁ − (B₂⁽¹⁾ + A⁽⁶⁾₂)·A⁽⁶⁾₂
    //  S = B₂⁽⁰⁾ + (B₂⁽¹⁾ + A⁽⁶⁾₂)·A⁽⁶⁾₁ + (B₁⁽¹⁾ + A⁽⁶⁾₁)·A⁽⁶⁾₂

    Complex* TA  = B1[2].get();  // T_A = B₁⁽¹⁾ + A⁽⁶⁾₁
    CHECK(AddMatrix(ops, B1[1].get(), A6_1.get(), TA, n));

    Complex* TB  = B2[2].get();  // T_B = B₂⁽¹⁾ + A⁽⁶⁾₂
    CHECK(AddMatrix(ops, B2[1].get(), A6_2.get(), TB, n));

    Complex* TA1 = B1[3].get();  // T_A·A⁽⁶⁾₁
    CHECK(MatMul(ops, TA, A6_1.get(), TA1, n));

    Complex* TB2 = B2[3].get();  // T_B·A⁽⁶⁾₂
    CHECK(MatMul(ops, TB, A6_2.get(), TB2, n));

    DeviceMatrix C;
    C.Swap(A3);
    CHECK(AddMatrix(ops, B1[0].get(), TA1, C.get(), n));   // C = B₁⁽⁰⁾ + TA·A⁽⁶⁾₁
    CHECK(ScaleMatrix(TB2, scratch.get(), n, neg_one, use_stream));
    CHECK(AddInPlace(ops, C.get(), scratch.get(), n));         // C −= TB·A⁽⁶⁾₂

    CHECK(MatMul(ops, TB, A6_1.get(), TA1, n));               // now TA1 = TB·A⁽⁶⁾₁
    CHECK(MatMul(ops, TA, A6_2.get(), TB2, n));               // now TB2 = TA·A⁽⁶⁾₂

    DeviceMatrix S;
    S.Swap(I_mat);
    CHECK(AddMatrix(ops, B2[0].get(), TA1, S.get(), n));   // S = B₂⁽⁰⁾ + TB·A⁽⁶⁾₁
    CHECK(AddInPlace(ops, S.get(), TB2, n));                    // S += TA·A⁽⁶⁾₂

    // ── Phase 4: repeated squaring (m rounds of double-angle) ───────────
    const Complex two = make_complex(2.0, 0.0);
    for (int i = 0; i < m; ++i)
    {
        // S_new = 2·S·C  (compute first to avoid overwriting C)
        CHECK(MatMul(ops, S.get(), C.get(), scratch.get(), n));
        CHECK(ScaleInPlace(scratch.get(), n, two, use_stream));
        S.Swap(scratch);   // S ← S_new, scratch ← old S

        // C_new = 2·C² − I  (using C² − S² = 2C² − I)
        CHECK(MatMul(ops, C.get(), C.get(), scratch.get(), n));
        CHECK(ScaleInPlace(scratch.get(), n, two, use_stream));
        CHECK(AddDiagonalConstantInPlace(scratch.get(), n, neg_one, use_stream));
        C.Swap(scratch);   // C ← C_new, scratch ← old C
    }

    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Complex);
    CHECK(StatusFromCuda(cudaMemcpyAsync(Cos, C.get(), bytes, cudaMemcpyDeviceToDevice, use_stream)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(Sin, S.get(), bytes, cudaMemcpyDeviceToDevice, use_stream)));
    return Status::kSuccess;
}

Status RCWA::ComputeCosSinFromThetaOrder18(const Complex* Theta,
                                            int64_t n,
                                            Complex* Cos,
                                            Complex* Sin,
                                            cudaStream_t stream) const
{
    if (!Theta || !Cos || !Sin || n <= 0) { return Status::kInvalidValue; }

    cudaStream_t use_stream = stream ? stream : stream_;
    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    // Scaling factor
    double ThetaNorm = 0.0;
    CHECK(ComputeMatrixNorm1(Theta, n, &ThetaNorm, use_stream));
    int m = 0;
    if (ThetaNorm > 0.0)
    {
        m = static_cast<int>(std::ceil(std::log2(ThetaNorm)));
        if (m < 0) { m = 0; }
    }
    const Complex scale = make_complex(static_cast<Real>(std::pow(2.0, -m)), 0.0);
    const Complex neg_one = make_complex(-1.0, 0.0);

    // ── Phase 1: block powers  A¹, A², A³, A⁶ ───────────────────────────
    DeviceMatrix A1;
    CHECK(A1.Allocate(n, n));
    CHECK(ScaleMatrix(Theta, A1.get(), n, scale, use_stream));

    DeviceMatrix A2;
    CHECK(A2.Allocate(n, n));
    CHECK(MatMul(ops, A1.get(), A1.get(), A2.get(), n));

    DeviceMatrix A3;
    CHECK(A3.Allocate(n, n));
    CHECK(MatMul(ops, A1.get(), A2.get(), A3.get(), n));

    DeviceMatrix A6;
    CHECK(A6.Allocate(n, n));
    CHECK(MatMul(ops, A3.get(), A3.get(), A6.get(), n));

    // ── Phase 2:  B₁⁽ⁱ⁾ = a₀ᵢ·I − a₂ᵢ·A² − a₄ᵢ·A⁶,  B₂⁽ⁱ⁾ = a₁ᵢ·A¹ − a₃ᵢ·A³  (i=0..4) ──
    DeviceMatrix I_mat;
    CHECK(I_mat.Allocate(n, n));
    CHECK(MakeIdentity(I_mat.get(), n, use_stream));

    DeviceMatrix scratch;
    CHECK(scratch.Allocate(n, n));

    DeviceMatrix B1[5], B2[5];
    for (int i = 0; i < 5; ++i)
    {
        const Complex a0 = make_complex(static_cast<Real>(kExpConstantOrder18[i][0]), 0.0);
        const Complex a1 = make_complex(static_cast<Real>(kExpConstantOrder18[i][1]), 0.0);
        const Complex a2 = make_complex(static_cast<Real>(kExpConstantOrder18[i][2]), 0.0);
        const Complex a3 = make_complex(static_cast<Real>(kExpConstantOrder18[i][3]), 0.0);
        const Complex a4 = make_complex(static_cast<Real>(kExpConstantOrder18[i][4]), 0.0);

        CHECK(B1[i].Allocate(n, n));
        CHECK(ScaleMatrix(I_mat.get(), B1[i].get(), n, a0, use_stream));
        CHECK(ScaleMatrix(A2.get(), scratch.get(), n,
                          make_complex(-ComplexReal(a2), -ComplexImag(a2)), use_stream));
        CHECK(AddInPlace(ops, B1[i].get(), scratch.get(), n));
        CHECK(ScaleMatrix(A6.get(), scratch.get(), n,
                          make_complex(-ComplexReal(a4), -ComplexImag(a4)), use_stream));
        CHECK(AddInPlace(ops, B1[i].get(), scratch.get(), n));

        CHECK(B2[i].Allocate(n, n));
        CHECK(ScaleMatrix(A1.get(), B2[i].get(), n, a1, use_stream));
        CHECK(ScaleMatrix(A3.get(), scratch.get(), n,
                          make_complex(-ComplexReal(a3), -ComplexImag(a3)), use_stream));
        CHECK(AddInPlace(ops, B2[i].get(), scratch.get(), n));
    }

    // A¹, A², A³, A⁶ are no longer needed after B construction —
    // reuse their buffers for A⁽⁹⁾₁, A⁽⁹⁾₂, C, S respectively.
    DeviceMatrix A9_1, A9_2;
    A9_1.Swap(A1);
    A9_2.Swap(A2);
    I_mat.Reset();

    // ── Phase 3: A⁽⁹⁾₁₂ from B⁽⁰⁾, B⁽⁴⁾, B⁽³⁾, then C, S ─────────────
    //
    //  A⁽⁹⁾₁ = B₁⁽⁰⁾·B₁⁽⁴⁾ − B₂⁽⁰⁾·B₂⁽⁴⁾ + B₁⁽³⁾
    //  A⁽⁹⁾₂ = B₂⁽⁰⁾·B₁⁽⁴⁾ + B₁⁽⁰⁾·B₂⁽⁴⁾ + B₂⁽³⁾
    CHECK(MatMul(ops, B1[0].get(), B1[4].get(), A9_1.get(), n));
    CHECK(MatMul(ops, B2[0].get(), B2[4].get(), scratch.get(), n));
    CHECK(ScaleMatrix(scratch.get(), scratch.get(), n, neg_one, use_stream));
    CHECK(AddInPlace(ops, A9_1.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_1.get(), B1[3].get(), n));

    CHECK(MatMul(ops, B2[0].get(), B1[4].get(), A9_2.get(), n));
    CHECK(MatMul(ops, B1[0].get(), B2[4].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_2.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_2.get(), B2[3].get(), n));

    // B⁽⁰⁾, B⁽³⁾, B⁽⁴⁾ are no longer needed — reuse as workspace for C/S.
    //
    //  C = B₁⁽¹⁾ + (B₁⁽²⁾ + A⁽⁹⁾₁)·A⁽⁹⁾₁ − (B₂⁽²⁾ + A⁽⁹⁾₂)·A⁽⁹⁾₂
    //  S = B₂⁽¹⁾ + (B₂⁽²⁾ + A⁽⁹⁾₂)·A⁽⁹⁾₁ + (B₁⁽²⁾ + A⁽⁹⁾₁)·A⁽⁹⁾₂

    Complex* TA  = B1[0].get();  // T_A = B₁⁽²⁾ + A⁽⁹⁾₁
    CHECK(AddMatrix(ops, B1[2].get(), A9_1.get(), TA, n));

    Complex* TB  = B2[0].get();  // T_B = B₂⁽²⁾ + A⁽⁹⁾₂
    CHECK(AddMatrix(ops, B2[2].get(), A9_2.get(), TB, n));

    Complex* TA1 = B1[3].get();  // T_A·A⁽⁹⁾₁
    CHECK(MatMul(ops, TA, A9_1.get(), TA1, n));

    Complex* TB2 = B2[3].get();  // T_B·A⁽⁹⁾₂
    CHECK(MatMul(ops, TB, A9_2.get(), TB2, n));

    DeviceMatrix C;
    C.Swap(A3);
    CHECK(AddMatrix(ops, B1[1].get(), TA1, C.get(), n));   // C = B₁⁽¹⁾ + TA·A⁽⁹⁾₁
    CHECK(ScaleMatrix(TB2, scratch.get(), n, neg_one, use_stream));
    CHECK(AddInPlace(ops, C.get(), scratch.get(), n));         // C −= TB·A⁽⁹⁾₂

    CHECK(MatMul(ops, TB, A9_1.get(), TA1, n));               // now TA1 = TB·A⁽⁹⁾₁
    CHECK(MatMul(ops, TA, A9_2.get(), TB2, n));               // now TB2 = TA·A⁽⁹⁾₂

    DeviceMatrix S;
    S.Swap(A6);
    CHECK(AddMatrix(ops, B2[1].get(), TA1, S.get(), n));   // S = B₂⁽¹⁾ + TB·A⁽⁹⁾₁
    CHECK(AddInPlace(ops, S.get(), TB2, n));                    // S += TA·A⁽⁹⁾₂

    // ── Phase 4: repeated squaring (m rounds of double-angle) ───────────
    const Complex two = make_complex(2.0, 0.0);
    for (int i = 0; i < m; ++i)
    {
        // S_new = 2·S·C  (compute first to avoid overwriting C)
        CHECK(MatMul(ops, S.get(), C.get(), scratch.get(), n));
        CHECK(ScaleInPlace(scratch.get(), n, two, use_stream));
        S.Swap(scratch);   // S ← S_new, scratch ← old S

        // C_new = 2·C² − I  (using C² − S² = 2C² − I)
        CHECK(MatMul(ops, C.get(), C.get(), scratch.get(), n));
        CHECK(ScaleInPlace(scratch.get(), n, two, use_stream));
        CHECK(AddDiagonalConstantInPlace(scratch.get(), n, neg_one, use_stream));
        C.Swap(scratch);   // C ← C_new, scratch ← old C
    }

    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Complex);
    CHECK(StatusFromCuda(cudaMemcpyAsync(Cos, C.get(), bytes, cudaMemcpyDeviceToDevice, use_stream)));
    CHECK(StatusFromCuda(cudaMemcpyAsync(Sin, S.get(), bytes, cudaMemcpyDeviceToDevice, use_stream)));
    return Status::kSuccess;
}

Status RCWA::ComputeTFromPQOrder12(const Complex* P,
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

    struct ScopedCudaStream
    {
        cudaStream_t stream = nullptr;
        ~ScopedCudaStream()
        {
            if (stream)
            {
                cudaStreamDestroy(stream);
            }
        }
    };
    struct ScopedCudaEvent
    {
        cudaEvent_t event = nullptr;
        ~ScopedCudaEvent()
        {
            if (event)
            {
                cudaEventDestroy(event);
            }
        }
    };
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
    const Complex scale_pos = make_complex(static_cast<Real>(scale_A), 0.0);
    const Complex scale_neg = make_complex(static_cast<Real>(-scale_A), 0.0);

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

    DeviceMatrix scratch;
    DeviceMatrix B_11[4];
    DeviceMatrix B_12[4];
    DeviceMatrix B_21[4];
    DeviceMatrix B_22[4];

    for (int i = 0; i < 4; ++i)
    {
        const Complex c0 = make_complex(static_cast<Real>(kExpConstantOrder12[i][0]), 0.0);
        const Complex c1 = make_complex(static_cast<Real>(kExpConstantOrder12[i][1]), 0.0);
        const Complex c2 = make_complex(static_cast<Real>(kExpConstantOrder12[i][2]), 0.0);
        const Complex c3 = make_complex(static_cast<Real>(kExpConstantOrder12[i][3]), 0.0);

        CHECK(B_11[i].Allocate(n, n));
        CHECK(ScaleMatrix(I2.get(), B_11[i].get(), n, c0, use_stream));
        CHECK(scratch.Allocate(n, n));
        CHECK(ScaleMatrix(A2_11.get(), scratch.get(), n, c2, use_stream));
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
    }

    A2_11.Reset();
    A2_22.Reset();
    A3_12.Reset();
    A3_21.Reset();

    // I2, A_12, A_21 no longer needed after B coefficients are built
    I2.Reset();
    A_12.Reset();
    A_21.Reset();

    DeviceMatrix A6_11;
    CHECK(A6_11.Allocate(n, n));
    CHECK(MatMul(ops, B_11[3].get(), B_11[3].get(), A6_11.get(), n));
    CHECK(scratch.Allocate(n, n));
    CHECK(MatMul(ops, B_12[3].get(), B_21[3].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_11.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_11.get(), B_11[2].get(), n));

    DeviceMatrix A6_12;
    CHECK(A6_12.Allocate(n, n));
    CHECK(MatMul(ops, B_11[3].get(), B_12[3].get(), A6_12.get(), n));
    CHECK(scratch.Allocate(n, n));
    CHECK(MatMul(ops, B_12[3].get(), B_22[3].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_12.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_12.get(), B_12[2].get(), n));

    DeviceMatrix A6_21;
    CHECK(A6_21.Allocate(n, n));
    CHECK(MatMul(ops, B_21[3].get(), B_11[3].get(), A6_21.get(), n));
    CHECK(scratch.Allocate(n, n));
    CHECK(MatMul(ops, B_22[3].get(), B_21[3].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_21.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_21.get(), B_21[2].get(), n));

    DeviceMatrix A6_22;
    CHECK(A6_22.Allocate(n, n));
    CHECK(MatMul(ops, B_21[3].get(), B_12[3].get(), A6_22.get(), n));
    CHECK(scratch.Allocate(n, n));
    CHECK(MatMul(ops, B_22[3].get(), B_22[3].get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_22.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A6_22.get(), B_22[2].get(), n));

    // Free B[2], B[3] — no longer needed after A6 blocks are built
    for (int i : {2, 3})
    {
        B_11[i].Reset(); B_12[i].Reset();
        B_21[i].Reset(); B_22[i].Reset();
    }

    DeviceMatrix temp1;
    DeviceMatrix temp2;
    DeviceMatrix temp3;

    DeviceMatrix T_11;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_11[1].get(), A6_11.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A6_11.get(), temp2.get(), n));
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_12[1].get(), A6_12.get(), temp1.get(), n));
        CHECK(temp3.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A6_21.get(), temp3.get(), n));
        CHECK(T_11.Allocate(n, n));
        CHECK(AddMatrix(ops, B_11[0].get(), temp2.get(), T_11.get(), n));
        CHECK(AddInPlace(ops, T_11.get(), temp3.get(), n));
    }

    DeviceMatrix T_12;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_11[1].get(), A6_11.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A6_12.get(), temp2.get(), n));
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_12[1].get(), A6_12.get(), temp1.get(), n));
        CHECK(temp3.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A6_22.get(), temp3.get(), n));
        CHECK(T_12.Allocate(n, n));
        CHECK(AddMatrix(ops, B_12[0].get(), temp2.get(), T_12.get(), n));
        CHECK(AddInPlace(ops, T_12.get(), temp3.get(), n));
    }

    DeviceMatrix T_21;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_21[1].get(), A6_21.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A6_11.get(), temp2.get(), n));
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_22[1].get(), A6_22.get(), temp1.get(), n));
        CHECK(temp3.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A6_21.get(), temp3.get(), n));
        CHECK(T_21.Allocate(n, n));
        CHECK(AddMatrix(ops, B_21[0].get(), temp2.get(), T_21.get(), n));
        CHECK(AddInPlace(ops, T_21.get(), temp3.get(), n));
    }

    DeviceMatrix T_22;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_21[1].get(), A6_21.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A6_12.get(), temp2.get(), n));
        CHECK(temp1.Allocate(n, n));
        CHECK(AddMatrix(ops, B_22[1].get(), A6_22.get(), temp1.get(), n));
        CHECK(temp3.Allocate(n, n));
        CHECK(MatMul(ops, temp1.get(), A6_22.get(), temp3.get(), n));
        CHECK(T_22.Allocate(n, n));
        CHECK(AddMatrix(ops, B_22[0].get(), temp2.get(), T_22.get(), n));
        CHECK(AddInPlace(ops, T_22.get(), temp3.get(), n));
    }

    // Free B[0], B[1], A6 blocks, and temp matrices — none are used in squaring
    for (int i : {0, 1})
    {
        B_11[i].Reset(); B_12[i].Reset();
        B_21[i].Reset(); B_22[i].Reset();
    }
    A6_11.Reset(); A6_12.Reset(); A6_21.Reset(); A6_22.Reset();
    temp1.Reset(); temp2.Reset(); temp3.Reset();

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
    const Complex imag = make_complex(0.0, 1.0);
    const Complex half = make_complex(0.5, 0.0);
    const Complex neg_one = make_complex(-1.0, 0.0);

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

// ─────────────────────────────────────────────────────────────────────────────
// Matrix square root via 7th-order sign-function iteration.
// Computes Y ≈ A^{1/2} and Z ≈ A^{-1/2}.
// ─────────────────────────────────────────────────────────────────────────────
static Status MatrixSqrt(const Complex* A,
                         int64_t n,
                         Real tol,
                         int max_iter,
                         Real phi,
                         DeviceMatrix* Y,
                         DeviceMatrix* Z,
                         ComplexMatrixOps& ops,
                         cudaStream_t stream)
{
    const Complex one = make_complex(1.0, 0.0);
    const Complex phase_rot = make_complex(std::cos(phi), std::sin(phi));
    const Complex phase_Y = make_complex(std::cos(-phi / 2.0), std::sin(-phi / 2.0));
    const Complex phase_Z = make_complex(std::cos(phi / 2.0), std::sin(phi / 2.0));

    DeviceMatrix A_rot;
    CHECK(A_rot.Allocate(n, n));
    CHECK(ScaleMatrix(A, A_rot.get(), n, phase_rot, stream));

    CHECK(Y->Allocate(n, n));
    CHECK(CopyMatrix(A_rot.get(), Y->get(), n, stream));

    CHECK(Z->Allocate(n, n));
    CHECK(MakeIdentity(Z->get(), n, stream));

    DeviceMatrix X;
    CHECK(X.Allocate(n, n));
    CHECK(CopyMatrix(A_rot.get(), X.get(), n, stream));

    DeviceMatrix X2, X3, P, Q, F, scratch;
    CHECK(X2.Allocate(n, n));
    CHECK(X3.Allocate(n, n));
    CHECK(P.Allocate(n, n));
    CHECK(Q.Allocate(n, n));

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // X2 = X * X,  X3 = X2 * X
        CHECK(MatMul(ops, X.get(), X.get(), X2.get(), n));
        CHECK(MatMul(ops, X2.get(), X.get(), X3.get(), n));

        // P = 7*I + 35*X + 21*X2 + X3
        CHECK(MakeIdentity(P.get(), n, stream));
        CHECK(ScaleInPlace(P.get(), n, make_complex(7.0, 0.0), stream));
        CHECK(scratch.Allocate(n, n));
        CHECK(ScaleMatrix(X.get(), scratch.get(), n, make_complex(35.0, 0.0), stream));
        CHECK(AddInPlace(ops, P.get(), scratch.get(), n));
        CHECK(ScaleMatrix(X2.get(), scratch.get(), n, make_complex(21.0, 0.0), stream));
        CHECK(AddInPlace(ops, P.get(), scratch.get(), n));
        CHECK(AddInPlace(ops, P.get(), X3.get(), n));

        // Q = I + 21*X + 35*X2 + 7*X3
        CHECK(MakeIdentity(Q.get(), n, stream));
        CHECK(ScaleMatrix(X.get(), scratch.get(), n, make_complex(21.0, 0.0), stream));
        CHECK(AddInPlace(ops, Q.get(), scratch.get(), n));
        CHECK(ScaleMatrix(X2.get(), scratch.get(), n, make_complex(35.0, 0.0), stream));
        CHECK(AddInPlace(ops, Q.get(), scratch.get(), n));
        CHECK(ScaleMatrix(X3.get(), scratch.get(), n, make_complex(7.0, 0.0), stream));
        CHECK(AddInPlace(ops, Q.get(), scratch.get(), n));

        // F = Q \ P
        CHECK(F.Allocate(n, n));
        CHECK(ops.Solve(Q.get(), P.get(), F.get(), n));

        // Y = Y * F
        CHECK(scratch.Allocate(n, n));
        CHECK(MatMul(ops, Y->get(), F.get(), scratch.get(), n));
        Y->Swap(scratch);

        // Z = F * Z
        CHECK(scratch.Allocate(n, n));
        CHECK(MatMul(ops, F.get(), Z->get(), scratch.get(), n));
        Z->Swap(scratch);

        // X = Z * Y
        CHECK(scratch.Allocate(n, n));
        CHECK(MatMul(ops, Z->get(), Y->get(), scratch.get(), n));
        X.Swap(scratch);

        // convergence check:  ||I - X||₁ < tol
        double res = 0.0;
        CHECK(ComputeMatrixNorm1WithDiagOffset(
            X.get(), n, make_complex(1.0, 0.0), &res, stream));
        if (res < static_cast<double>(tol)) { break; }
    }

    // undo branch-cut rotation
    CHECK(scratch.Allocate(n, n));
    CHECK(ScaleMatrix(Y->get(), scratch.get(), n, phase_Y, stream));
    Y->Swap(scratch);
    CHECK(scratch.Allocate(n, n));
    CHECK(ScaleMatrix(Z->get(), scratch.get(), n, phase_Z, stream));
    Z->Swap(scratch);

    return Status::kSuccess;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute S-matrix via the SQRT path.
//
//   PQ = P·Q
//   [G, G⁻¹] = matrix_sqrt(PQ)
//   exp(i·k₀d·G)         ← general expm via [5/5] Padé + scaling/squaring
//   exp(−i·k₀d·G) = inv(exp(…))
//   Cos, Sin, K, E, F    ← matrix arithmetic
//   S-blocks              ← same conversion as GEMM
//
// expm still uses GEMM, but the matrix size is n₂×n₂ instead of 2n₂×2n₂.
// ─────────────────────────────────────────────────────────────────────────────
Status RCWA::ComputeSMatrixSQRT(const Complex* P,
                                const Complex* Q,
                                int64_t n,
                                Real k0d,
                                const VMatrix& V,
                                SMatrix* S,
                                cudaStream_t stream) const
{
    if (!P || !Q || !S || n <= 0 || (n % 2) != 0)
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

    const Complex imag = make_complex(0.0, 1.0);
    const Complex half = make_complex(0.5, 0.0);
    const Complex neg_one = make_complex(-1.0, 0.0);

    // ── 1. PQ = P * Q ──────────────────────────────────────────────────
    DeviceMatrix PQ;
    CHECK(PQ.Allocate(n, n));
    CHECK(MatMul(ops, P, Q, PQ.get(), n));

    // ── 2. [G, G_inv] = matrix_sqrt(PQ) ────────────────────────────────
    DeviceMatrix G, G_inv;
    CHECK(MatrixSqrt(
        PQ.get(), n, static_cast<Real>(1e-12), 10, static_cast<Real>(-0.7853981633974483), &G, &G_inv, ops,
        use_stream));
    PQ.Reset();

    // ── 3-4. Theta = k0d·G, then [Cos, Sin] via block-structured Padé ──
    DeviceMatrix Theta;
    CHECK(Theta.Allocate(n, n));
    {
        const Complex k0d_c = make_complex(static_cast<Real>(k0d), 0.0);
        CHECK(ScaleMatrix(G.get(), Theta.get(), n, k0d_c, use_stream));
        G.Reset();
    }

    DeviceMatrix Cos;
    CHECK(Cos.Allocate(n, n));
    DeviceMatrix Sin;
    CHECK(Sin.Allocate(n, n));
    if (high_order_)
    {
        CHECK(ComputeCosSinFromThetaOrder18(Theta.get(), n, Cos.get(), Sin.get(), use_stream));
    }
    else
    {
        CHECK(ComputeCosSinFromThetaOrder12(Theta.get(), n, Cos.get(), Sin.get(), use_stream));
    }
    Theta.Reset();

    // ── 7. K = V0 · Q · G_inv ──────────────────────────────────────────
    DeviceMatrix Q_Ginv;
    CHECK(Q_Ginv.Allocate(n, n));
    CHECK(MatMul(ops, Q, G_inv.get(), Q_Ginv.get(), n));
    DeviceMatrix K;
    CHECK(ApplyV0Left(V, Q_Ginv.get(), n, &K, use_stream));
    Q_Ginv.Reset();
    G_inv.Reset();

    // ── 8. E = Cos + i·K·Sin ──────────────────────────────────────────
    DeviceMatrix E;
    CHECK(E.Allocate(n, n));
    CHECK(CopyMatrix(Cos.get(), E.get(), n, use_stream));
    {
        DeviceMatrix KSin;
        CHECK(KSin.Allocate(n, n));
        CHECK(MatMul(ops, K.get(), Sin.get(), KSin.get(), n));
        CHECK(ScaleInPlace(KSin.get(), n, imag, use_stream));
        CHECK(AddInPlace(ops, E.get(), KSin.get(), n));
    }

    // ── 9. F = (i·Sin + K·Cos) / K ──────────────────────────────────
    //  9a. RHS = i·Sin + K·Cos
    DeviceMatrix RHS;
    CHECK(RHS.Allocate(n, n));
    CHECK(ScaleMatrix(Sin.get(), RHS.get(), n, imag, use_stream));
    {
        DeviceMatrix KCos;
        CHECK(KCos.Allocate(n, n));
        CHECK(MatMul(ops, K.get(), Cos.get(), KCos.get(), n));
        CHECK(AddInPlace(ops, RHS.get(), KCos.get(), n));
    }
    Sin.Reset();
    Cos.Reset();

    //  9b. F = RHS / K  via  Kᵀ · Fᵀ = RHSᵀ
    DeviceMatrix RHS_trans;
    CHECK(RHS_trans.Allocate(n, n));
    CHECK(TransposeMatrix(RHS.get(), RHS_trans.get(), n, use_stream));

    DeviceMatrix F_trans;
    CHECK(F_trans.Allocate(n, n));
    CHECK(ops.Solve(K.get(), RHS_trans.get(), F_trans.get(), n, n, CUBLAS_OP_T));
    K.Reset();
    RHS_trans.Reset();

    DeviceMatrix F;
    CHECK(F.Allocate(n, n));
    CHECK(TransposeMatrix(F_trans.get(), F.get(), n, use_stream));
    RHS.Reset();

    // ── 10. S-blocks: same conversion as ComputeSFromT ─────────────────
    // T22_layer = (E + F)/2,  T21_layer = (E − F)/2
    // S11 = inv(T22_layer),   S12 = −S11 · T21_layer
    DeviceMatrix T22, T21;
    CHECK(T22.Allocate(n, n));
    CHECK(AddMatrix(ops, E.get(), F.get(), T22.get(), n));
    CHECK(ScaleInPlace(T22.get(), n, half, use_stream));
    CHECK(T21.Allocate(n, n));
    CHECK(SubMatrix(ops, E.get(), F.get(), T21.get(), n));
    CHECK(ScaleInPlace(T21.get(), n, half, use_stream));

    DeviceMatrix S11, S12;
    {
        DeviceMatrix I_rhs;
        CHECK(I_rhs.Allocate(n, n));
        CHECK(MakeIdentity(I_rhs.get(), n, use_stream));
        CHECK(S11.Allocate(n, n));
        CHECK(ops.Solve(T22.get(), I_rhs.get(), S11.get(), n));
    }
    CHECK(S12.Allocate(n, n));
    CHECK(MatMul(ops, S11.get(), T21.get(), S12.get(), n));
    CHECK(ScaleInPlace(S12.get(), n, neg_one, use_stream));

    DeviceMatrix S21, S22;
    CHECK(S21.Allocate(n, n));
    CHECK(S22.Allocate(n, n));
    CHECK(CopyMatrix(S12.get(), S21.get(), n, use_stream));
    CHECK(CopyMatrix(S11.get(), S22.get(), n, use_stream));

    S->S_11 = S11.Release();
    S->S_12 = S12.Release();
    S->S_21 = S21.Release();
    S->S_22 = S22.Release();
    return Status::kSuccess;
}

Status RCWA::AddLayer(Real thickness, Real eps)
{
    return AddLayer(thickness, make_complex(eps, 0.0));
}

Status RCWA::AddLayer(Real thickness, Complex eps)
{
    DeviceMatrix E;
    CHECK(E.Allocate(order_N_, order_N_));
    CHECK(FillDiagonalConstant(order_N_, eps, E.get(), stream_));
    return AddLayer(thickness, E.get());
}

Status RCWA::AddLayerHomogeneous(Real thickness, Real eps)
{
    return AddLayerHomogeneous(thickness, make_complex(eps, 0.0));
}

Status RCWA::AddLayerHomogeneous(Real thickness, Complex eps)
{
    if (!kvectors_ready_ || !kx_diag_ || !ky_diag_)
    {
        return Status::kInvalidValue;
    }

    const int64_t n = order_N_;
    const int64_t n2 = 2 * n;
    DeviceMatrix S11;
    DeviceMatrix S12;
    DeviceMatrix S21;
    DeviceMatrix S22;
    CHECK(S11.Allocate(n2, n2));
    CHECK(S12.Allocate(n2, n2));
    CHECK(S21.Allocate(n2, n2));
    CHECK(S22.Allocate(n2, n2));

    const size_t matrix_bytes = static_cast<size_t>(n2) * static_cast<size_t>(n2) * sizeof(Complex);
    CHECK(StatusFromCuda(cudaMemsetAsync(S11.get(), 0, matrix_bytes, stream_)));
    CHECK(StatusFromCuda(cudaMemsetAsync(S12.get(), 0, matrix_bytes, stream_)));
    CHECK(StatusFromCuda(cudaMemsetAsync(S21.get(), 0, matrix_bytes, stream_)));
    CHECK(StatusFromCuda(cudaMemsetAsync(S22.get(), 0, matrix_bytes, stream_)));

    const Real k0d = omega_ * thickness;
    const int blocks = static_cast<int>((n + CUDA_THREADS - 1) / CUDA_THREADS);
    BuildHomogeneousSBlocksKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(
        kx_diag_, ky_diag_, eps, k0d, S11.get(), S12.get(), S21.get(), S22.get(), n);
    CHECK(StatusFromCuda(cudaGetLastError()));

    SMatrix S{};
    S.S_11 = S11.Release();
    S.S_12 = S12.Release();
    S.S_21 = S21.Release();
    S.S_22 = S22.Release();
    return CommitLayerSMatrix(&S, n2);
}

Status RCWA::AddLayer(Real thickness, const Complex* eps_conv)
{
    if (!eps_conv || !kvectors_ready_)
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
        const Real thickness_thres = 2.0 * std::min(dx, dy);
        if (thickness_thres > 0.0 && thickness > thickness_thres)
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

    SMatrix S{};
    if (expm_method_ == 'S')
    {
        CHECK(ComputeSMatrixSQRT(P.get(), Q.get(), n2, k0d, V0_, &S, stream_));
    }
    else
    {
        TMatrix T{};
        if (high_order_)
        {
            CHECK(ComputeTFromPQOrder18(P.get(), Q.get(), n2, k0d, &T, stream_));
        }
        else
        {
            CHECK(ComputeTFromPQOrder12(P.get(), Q.get(), n2, k0d, &T, stream_));
        }
        CHECK(ComputeSFromT(T, V0_, n2, &S, stream_));

        SAFEFREE(T.T_11);
        SAFEFREE(T.T_12);
        SAFEFREE(T.T_21);
        SAFEFREE(T.T_22);
    }

    if (n_repeated > 0)
    {
        CHECK(RepeatSMatrix(&S, n_repeated, n2));
    }

    return CommitLayerSMatrix(&S, n2);
}

Status RCWA::MaterialConv(const Complex* material,
                          int64_t nx,
                          int64_t ny,
                          Complex** eps_conv,
                          cudaStream_t stream) const
{
    if (!material || !eps_conv)
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
    const Real scale = denom > 0.0 ? static_cast<Real>(1.0 / denom) : 0.0;
    int blocks = static_cast<int>((fft_size + CUDA_THREADS - 1) / CUDA_THREADS);
    ScaleComplexKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(fft_buf.get(), fft_size, scale);
    CHECK(StatusFromCuda(cudaGetLastError()));

    DeviceMatrix conv;
    CHECK(conv.Allocate(order_N_, order_N_));
    const int64_t total = order_N_ * order_N_;
    blocks = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    const int64_t oy_len = static_cast<int64_t>(2 * order_y_ + 1);
    BuildConvMatrixKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(
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
    cudaStream_t use_stream = stream ? stream : stream_;
    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));
    return RSProd(Sm, Sn, n, out, ops, use_stream);
}

Status RCWA::RSProd(const SMatrix& Sm,
                    const SMatrix& Sn,
                    int64_t n,
                    SMatrix* out,
                    ComplexMatrixOps& ops,
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

    const Complex one = make_complex(1.0, 0.0);
    const Complex minus_one = make_complex(-1.0, 0.0);

    DeviceMatrix T;
    CHECK(T.Allocate(n, n));
    CHECK(MatMul(ops, Sm.S_12, Sn.S_21, T.get(), n));
    CHECK(ScaleInPlace(T.get(), n, minus_one, use_stream));
    CHECK(AddDiagonalConstantInPlace(T.get(), n, one, use_stream));

    const int64_t rhs_cols = 2 * n;
    DeviceMatrix rhs;
    CHECK(rhs.Allocate(n, rhs_cols));
    CHECK(CopyBlockDirect(Sm.S_11, n, rhs.get(), n, 0, 0, use_stream));
    Complex* X1 = rhs.get();
    Complex* X2 = rhs.get() + static_cast<size_t>(n) * n;
    CHECK(MatMul(ops, Sm.S_12, Sn.S_22, X2, n));

    const Status solver = ops.Solve(T.get(), rhs.get(), rhs.get(), n, rhs_cols);
    if (solver != Status::kSuccess)
    {
        DeviceMatrix S11;
        CHECK(S11.Allocate(n, n));
        CHECK(StatusFromCuda(cudaMemsetAsync(
            S11.get(), 0, static_cast<size_t>(n) * n * sizeof(Complex), use_stream)));

        DeviceMatrix S12;
        CHECK(S12.Allocate(n, n));
        CHECK(CopyMatrix(Sn.S_12, S12.get(), n, use_stream));

        DeviceMatrix S21;
        CHECK(S21.Allocate(n, n));
        CHECK(CopyMatrix(Sm.S_21, S21.get(), n, use_stream));

        DeviceMatrix S22;
        CHECK(S22.Allocate(n, n));
        CHECK(MatMul(ops, Sm.S_22, Sn.S_22, S22.get(), n));

        out->S_11 = S11.Release();
        out->S_12 = S12.Release();
        out->S_21 = S21.Release();
        out->S_22 = S22.Release();
        return Status::kSuccess;
    }

    DeviceMatrix S11;
    CHECK(S11.Allocate(n, n));
    CHECK(MatMul(ops, Sn.S_11, X1, S11.get(), n));

    DeviceMatrix S21;
    CHECK(S21.Allocate(n, n));
    CHECK(MatMul(ops, Sn.S_21, X1, T.get(), n));
    CHECK(MatMul(ops, Sm.S_22, T.get(), S21.get(), n));
    CHECK(AddInPlace(ops, S21.get(), Sm.S_21, n));

    DeviceMatrix S12;
    CHECK(S12.Allocate(n, n));
    CHECK(MatMul(ops, Sn.S_11, X2, S12.get(), n));
    CHECK(AddInPlace(ops, S12.get(), Sn.S_12, n));

    DeviceMatrix S22;
    CHECK(S22.Allocate(n, n));
    CHECK(MatMul(ops, Sm.S_22, Sn.S_22, S22.get(), n));
    CHECK(MatMul(ops, Sn.S_21, X2, T.get(), n));
    CHECK(MatMul(ops, Sm.S_22, T.get(), X1, n));
    CHECK(AddInPlace(ops, S22.get(), X1, n));

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
    if (!S)
    {
        return Status::kInvalidValue;
    }
    const int64_t n = 2 * order_N_;

    cudaStream_t use_stream = stream ? stream : stream_;

    if (!S_in && !S_out && HasSMatrix(global_smatrix_))
    {
        if (S == &global_smatrix_)
        {
            return Status::kSuccess;
        }
        S->S_11 = nullptr;
        S->S_12 = nullptr;
        S->S_21 = nullptr;
        S->S_22 = nullptr;
        return CopySMatrix(global_smatrix_, n, S, use_stream);
    }

    SMatrix auto_in{};
    SMatrix auto_out{};
    const SMatrix* use_S_in = S_in;
    const SMatrix* use_S_out = S_out;

    if (!S_in)
    {
        CHECK(BuildPortSMatrix(eps_in_, true, &auto_in, use_stream));
        use_S_in = &auto_in;
    }
    if (!S_out)
    {
        CHECK(BuildPortSMatrix(eps_out_, false, &auto_out, use_stream));
        use_S_out = &auto_out;
    }

    S->S_11 = nullptr;
    S->S_12 = nullptr;
    S->S_21 = nullptr;
    S->S_22 = nullptr;

    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    SMatrix current{};
    if (HasSMatrix(stack_smatrix_))
    {
        CHECK(CopySMatrix(stack_smatrix_, n, &current, use_stream));
    }
    else
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

    if (use_S_in)
    {
        if (!use_S_in->S_11 || !use_S_in->S_12 || !use_S_in->S_21 || !use_S_in->S_22)
        {
            FreeSMatrix(&current);
            FreeSMatrix(&auto_in);
            FreeSMatrix(&auto_out);
            return Status::kInvalidValue;
        }
        SMatrix next{};
        CHECK(RSProd(*use_S_in, current, n, &next, ops, use_stream));
        FreeSMatrix(&current);
        current = next;
    }

    if (use_S_out)
    {
        if (!use_S_out->S_11 || !use_S_out->S_12 || !use_S_out->S_21 || !use_S_out->S_22)
        {
            FreeSMatrix(&current);
            FreeSMatrix(&auto_in);
            FreeSMatrix(&auto_out);
            return Status::kInvalidValue;
        }
        SMatrix next{};
        CHECK(RSProd(current, *use_S_out, n, &next, ops, use_stream));
        FreeSMatrix(&current);
        current = next;
    }

    FreeSMatrix(&auto_in);
    FreeSMatrix(&auto_out);
    S->S_11 = current.S_11;
    S->S_12 = current.S_12;
    S->S_21 = current.S_21;
    S->S_22 = current.S_22;
    return Status::kSuccess;
}

Status RCWA::SolveGlobalSMatrix()
{
    if (HasSMatrix(global_smatrix_))
    {
        return Status::kSuccess;
    }

    FreeGlobalSMatrix();
    const Status status = SolveGlobalSMatrix(&global_smatrix_, nullptr, nullptr, stream_);
    if (status != Status::kSuccess)
    {
        FreeGlobalSMatrix();
    }
    return status;
}

Status RCWA::RepeatCurrentStackSMatrix(int repeats)
{
    if (!HasSMatrix(stack_smatrix_))
    {
        return Status::kInvalidValue;
    }

    FreeGlobalSMatrix();
    return RepeatSMatrix(&stack_smatrix_, repeats, 2 * order_N_);
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
        !global_smatrix_.S_22 || !kx_diag_ || !ky_diag_)
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
    const int blocks = static_cast<int>((n + CUDA_THREADS - 1) / CUDA_THREADS);
    const Complex eps = angle_layer_input_ ? eps_in_ : eps_out_;
    BuildKzKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(
        kx_diag_, ky_diag_, eps, angle_layer_input_ ? 1 : 0, Kz.get(), n);
    CHECK(StatusFromCuda(cudaGetLastError()));

    DeviceVector Ezmn;
    CHECK(Ezmn.Allocate(static_cast<size_t>(n)));
    BuildEzKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(
        kx_diag_, ky_diag_, Kz.get(), Exy.get(), Exy.get() + n, Ezmn.get(), n);
    CHECK(StatusFromCuda(cudaGetLastError()));

    DeviceMatrix ExMat;
    DeviceMatrix EyMat;
    DeviceMatrix EzMat;
    CHECK(ExMat.Allocate(x_count, y_count));
    CHECK(EyMat.Allocate(x_count, y_count));
    CHECK(EzMat.Allocate(x_count, y_count));

    const int64_t total = x_count * y_count;
    const int grid = static_cast<int>((total + CUDA_THREADS - 1) / CUDA_THREADS);
    FieldXYKernel<<<grid, CUDA_THREADS, 0, use_stream>>>(kx_diag_,
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

    // const Status solve_kx = ops.Solve(eps_conv, KxMat.get(), EiKx.get(), n);
    // const Status solve_ky = ops.Solve(eps_conv, KyMat.get(), EiKy.get(), n);
    //  Optimized: merge KxMat and KyMat into one 2n-column RHS so that eps_conv is
    //  LU-factorized only once.  combined = [KxMat | KyMat], solve in-place, then
    //  split the solution back into EiKx (left half) and EiKy (right half).
    const int64_t nrhs = 2 * n;
    DeviceMatrix combined;
    CHECK(combined.Allocate(n, nrhs));
    // CopyBlock's dst_n argument is the number of rows of the destination matrix
    // (the leading dimension in column-major storage).
    CHECK(CopyBlockDirect(KxMat.get(), n, combined.get(), n, 0, 0, stream_));
    CHECK(CopyBlockDirect(KyMat.get(), n, combined.get(), n, 0, n, stream_));
    const Status solve_combined = ops.Solve(eps_conv, combined.get(), combined.get(), n, nrhs);
    const bool fallback = (solve_combined != Status::kSuccess);
    if (!fallback)
    {
        // In column-major (ld=n), columns 0..n-1 occupy elements [0, n*n-1] and
        // columns n..2n-1 occupy elements [n*n, 2*n*n-1] — both regions are contiguous.
        const size_t half_bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Complex);
        CHECK(StatusFromCuda(cudaMemcpyAsync(
            EiKx.get(), combined.get(), half_bytes, cudaMemcpyDeviceToDevice, stream_)));
        CHECK(StatusFromCuda(cudaMemcpyAsync(EiKy.get(),
                                             combined.get() + static_cast<size_t>(n) * n,
                                             half_bytes,
                                             cudaMemcpyDeviceToDevice,
                                             stream_)));
    }

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
    CHECK(AddDiagonalConstantInPlace(P12.get(), n, make_complex(-1.0, 0.0), stream_));
    CHECK(P21.Allocate(n, n));
    CHECK(CopyMatrix(KxEiKx.get(), P21.get(), n, stream_));
    CHECK(ScaleInPlace(P21.get(), n, make_complex(-1.0, 0.0), stream_));
    CHECK(AddDiagonalConstantInPlace(P21.get(), n, make_complex(1.0, 0.0), stream_));
    CHECK(Q12.Allocate(n, n));
    CHECK(CopyMatrix(eps_conv, Q12.get(), n, stream_));
    CHECK(AddDiagonalInPlace(Q12.get(), KyKyDiag.get(), n, make_complex(-1.0, 0.0), stream_));
    CHECK(Q21.Allocate(n, n));
    CHECK(CopyMatrix(eps_conv, Q21.get(), n, stream_));
    CHECK(ScaleInPlace(Q21.get(), n, make_complex(-1.0, 0.0), stream_));
    CHECK(AddDiagonalInPlace(Q21.get(), KxKxDiag.get(), n, make_complex(1.0, 0.0), stream_));

    CHECK(StatusFromCuda(cudaMemsetAsync(
        Q, 0, static_cast<size_t>(2 * n) * static_cast<size_t>(2 * n) * sizeof(Complex), stream_)));

    CHECK(CopyBlockDirect(KxEiKy.get(), n, P, 2 * n, 0, 0, stream_));
    CHECK(CopyBlockDirect(P21.get(), n, P, 2 * n, 0, n, stream_));
    CHECK(CopyBlockDirect(P12.get(), n, P, 2 * n, n, 0, stream_));
    CHECK(CopyBlockNegated(KyEiKx.get(), n, P, 2 * n, n, n, stream_));

    CHECK(CopyDiagonalBlock(KxKyDiag.get(), n, Q, 2 * n, 0, 0, make_complex(-1.0, 0.0), stream_));
    CHECK(CopyBlockDirect(Q21.get(), n, Q, 2 * n, 0, n, stream_));
    CHECK(CopyBlockDirect(Q12.get(), n, Q, 2 * n, n, 0, stream_));
    CHECK(CopyDiagonalBlock(KxKyDiag.get(), n, Q, 2 * n, n, n, make_complex(1.0, 0.0), stream_));

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

    DeviceMatrix R;
    DeviceMatrix rhs;
    DeviceMatrix S11XY;

    const Complex one = make_complex(1.0, 0.0);
    const Complex minus_one = make_complex(-1.0, 0.0);
    const int64_t rhs_cols = 2 * n;
    const size_t half_bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Complex);

    for (int i = 0; i < repeats; ++i)
    {
        CHECK(R.Allocate(n, n));
        CHECK(MatMul(ops, S->S_12, S->S_12, R.get(), n));
        CHECK(ScaleInPlace(R.get(), n, minus_one, stream_));
        CHECK(AddDiagonalConstantInPlace(R.get(), n, one, stream_));

        CHECK(rhs.Allocate(n, rhs_cols));

        // rhs = [S_12*S_11 | S_11], solved in one factorization.
        CHECK(MatMul(ops, S->S_12, S->S_11, rhs.get(), n));

        // Right column (n): S_11
        CHECK(CopyBlockDirect(S->S_11, n, rhs.get(), n, 0, n, stream_));

        // Single solve: R*[X | Y] = [S_12*S_11 | S_11]
        CHECK(ops.Solve(R.get(), rhs.get(), rhs.get(), n, rhs_cols));

        // Compute S_11 * [X | Y] once, then split the two n x n blocks.
        CHECK(S11XY.Allocate(n, rhs_cols));
        CHECK(ops.MatMul(S->S_11, rhs.get(), S11XY.get(), n, rhs_cols, n));

        CHECK(StatusFromCuda(
            cudaMemcpyAsync(S->S_21, S11XY.get(), half_bytes, cudaMemcpyDeviceToDevice, stream_)));
        CHECK(AddInPlace(ops, S->S_21, S->S_12, n));

        CHECK(StatusFromCuda(cudaMemcpyAsync(S->S_22,
                                             S11XY.get() + static_cast<size_t>(n) * n,
                                             half_bytes,
                                             cudaMemcpyDeviceToDevice,
                                             stream_)));

        Complex* next_S11 = S->S_22;
        S->S_22 = S->S_11;
        S->S_11 = next_S11;

        Complex* next_S12 = S->S_21;
        S->S_21 = S->S_12;
        S->S_12 = next_S12;
    }

    CHECK(CopyMatrix(S->S_12, S->S_21, n, stream_));
    CHECK(CopyMatrix(S->S_11, S->S_22, n, stream_));
    return Status::kSuccess;
}