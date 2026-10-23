#include <cstring>
#include <cufft.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "complex_matrix.h"
#include "matrix_storage.h"
#include <stdexcept>
#include "rcwa.h"

static Status StatusFromCuda(cudaError_t status)
{
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

static Status StatusFromCufft(cufftResult status)
{
    return status == CUFFT_SUCCESS ? Status::kSuccess : Status::kCudaError;
}

static int PeakMatrixCountForPath(char expm_method, bool high_order)
{
    if (expm_method == 'F')
    {
        return high_order ? 15 : 13;
    }
    return high_order ? 30 : 24;
}

static size_t EstimatedPeakLayerBytesForOrder(int order, int peak_matrix_count)
{
    if (order < 0 || peak_matrix_count <= 0)
    {
        return std::numeric_limits<size_t>::max();
    }
    const double orders_per_axis = static_cast<double>(2 * order + 1);
    const double harmonic_count = orders_per_axis * orders_per_axis;
    const double block_dim = 2.0 * harmonic_count;
    const double bytes = static_cast<double>(peak_matrix_count) * block_dim * block_dim *
                         static_cast<double>(sizeof(Complex));
    if (bytes > static_cast<double>(std::numeric_limits<size_t>::max()))
    {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(bytes);
}

static int MemoryLimitedOrderMax(int requested_order_max,
                                 char expm_method,
                                 bool high_order,
                                 Real safety_fraction)
{
    if (requested_order_max <= 0)
    {
        return requested_order_max;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes == 0)
    {
        return requested_order_max;
    }

    const double clamped_fraction = std::min(
        0.95, std::max(0.10, static_cast<double>(safety_fraction)));
    const size_t budget_bytes = static_cast<size_t>(static_cast<double>(free_bytes) *
                                                    clamped_fraction);
    // Bound automatic order selection by the same peak-live-matrix model
    // documented above.  The safety fraction leaves room for material maps,
    // cuFFT/cuBLAS workspaces, the accumulated stack S-matrix, and allocator
    // fragmentation on the current GPU.
    const int peak_matrix_count = PeakMatrixCountForPath(expm_method, high_order);
    int limited = 0;
    for (int order = 1; order <= requested_order_max; ++order)
    {
        if (EstimatedPeakLayerBytesForOrder(order, peak_matrix_count) > budget_bytes)
        {
            break;
        }
        limited = order;
    }
    return std::max(limited, 1);
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

// ═══════════════════════════════════════════════════════════════════════════════
//  GPU Memory Estimation Guide
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Each method allocates n₂ × n₂ matrices of Complex on the GPU.
//
//      M_peak = N_peak × n₂² × sizeof(Complex)
//
//  where  n₂ = 2·(2k+1)²  (k = order_x = order_y)
//  sizeof(Complex) = 16 B (double) or 8 B (single)
//
// ── Reference GPU VRAM ──────────────────────────────────────────────────────
//
//   GPU            | VRAM
//   ───────────────┼──────
//   RTX 5060 Laptop|  8 GB
//   RTX 4090D      | 24 GB
//   H100           | 80 GB
//
// ── Peak live matrix counts per method (N_peak) ─────────────────────────────
//
//   ComputeTFromPQOrder18 (GEMM): 29; Order12: 23.
//   ComputeCosSincFromPQOrder18 (FREE): 15; Order12: 13.
//
//   End-to-end estimates used by PeakMatrixCountForPath:
//   GEMM (G): 30/24; FREE (F): 15/13, for orders 18/12 respectively.
//
// ── Estimation rules ────────────────────────────────────────────────────────
//
//  1. Allocate() with same element count is a no‑op — scratch matrices are
//     allocated once and reused throughout.
//  2. Swap() exchanges pointers without allocating — does not increase
//     the live matrix count.
//  3. The peak is almost always in the B‑coefficient loop (Phase 2).  After
//     Phase 2 cleanup the count drops and never recovers to Phase 2 levels.
// ═══════════════════════════════════════════════════════════════════════════════

struct DeviceMatrix
{
    Complex* ptr = nullptr;
    size_t elements = 0;

    ~DeviceMatrix()
    {
        if (ptr)
        {
            rcwa_memory::Free(ptr);
        }
    }

    Complex* get() const { return ptr; }
    void Reset(Complex* p = nullptr, size_t count = 0)
    {
        if (ptr)
        {
            rcwa_memory::Free(ptr);
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

    Status Allocate(int64_t rows, int64_t cols, bool device_only = false)
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
            rcwa_memory::Free(ptr);
        }
        Complex* out = nullptr;
        const cudaError_t allocation_status = device_only ? cudaMalloc(&out, count * sizeof(Complex)) :
                                                          rcwa_memory::Allocate(&out, count * sizeof(Complex));
        if (allocation_status != cudaSuccess)
        {
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            fprintf(stderr, "DeviceMatrix allocation %lld x %lld (%zu bytes): %s (%d); free=%zu total=%zu bytes\n",
                    static_cast<long long>(rows), static_cast<long long>(cols), count * sizeof(Complex),
                    cudaGetErrorString(allocation_status), static_cast<int>(allocation_status), free_bytes, total_bytes);
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
            rcwa_memory::Free(ptr);
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
            rcwa_memory::Free(ptr);
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
            rcwa_memory::Free(ptr);
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
            rcwa_memory::Free(ptr);
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
                                              int64_t n, bool packed = false)
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

    const Complex kz = RegularizeGrazingKz(ComplexSqrt_RCWA(ComplexSub(eps, sum)));
    const Complex kz0 = RegularizeGrazingKz(ComplexSqrt_RCWA(ComplexSub(one, sum)));

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
    const int64_t i00 = packed ? idx : base_col0 + row0;
    const int64_t i10 = packed ? idx+n : base_col0 + row1;
    const int64_t i01 = packed ? idx+2*n : base_col1 + row0;
    const int64_t i11 = packed ? idx+3*n : base_col1 + row1;

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
                                       int64_t n, bool packed = false)
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
    const int64_t i00 = packed ? idx : base_col0 + row0;
    const int64_t i10 = packed ? idx+n : base_col0 + row1;
    const int64_t i01 = packed ? idx+2*n : base_col1 + row0;
    const int64_t i11 = packed ? idx+3*n : base_col1 + row1;

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

__device__ inline Real PhiAt(const Complex* eps, int64_t nx, int64_t ny, int64_t x, int64_t y)
{
    x %= nx;
    y %= ny;
    const int64_t px = x < 0 ? x + nx : x;
    const int64_t py = y < 0 ? y + ny : y;
    return ComplexReal(eps[static_cast<int64_t>(px) * ny + py]);
}

__device__ inline Real BlurredPhiAt(const Complex* eps, int64_t nx, int64_t ny, int64_t x, int64_t y)
{
    constexpr Real weights[5] = {
        static_cast<Real>(1.0 / 16.0),
        static_cast<Real>(4.0 / 16.0),
        static_cast<Real>(6.0 / 16.0),
        static_cast<Real>(4.0 / 16.0),
        static_cast<Real>(1.0 / 16.0)};
    Real sum = 0.0;
    for (int dx = -2; dx <= 2; ++dx)
    {
        for (int dy = -2; dy <= 2; ++dy)
        {
            sum += weights[dx + 2] * weights[dy + 2] * PhiAt(eps, nx, ny, x + dx, y + dy);
        }
    }
    return sum;
}

__device__ inline bool BoundaryVectorAt(const Complex* eps,
                                        int64_t nx,
                                        int64_t ny,
                                        Real lx,
                                        Real ly,
                                        Real eta,
                                        int64_t x,
                                        int64_t y,
                                        Real* vx,
                                        Real* vy)
{
    const Real gx_raw = PhiAt(eps, nx, ny, x + 1, y) - PhiAt(eps, nx, ny, x - 1, y);
    const Real gy_raw = PhiAt(eps, nx, ny, x, y + 1) - PhiAt(eps, nx, ny, x, y - 1);
    const Real raw_norm2 = gx_raw * gx_raw + gy_raw * gy_raw;
    if (raw_norm2 <= eta * eta)
    {
        *vx = 0.0;
        *vy = 0.0;
        return false;
    }

    const Real inv_2dx = static_cast<Real>(nx) / (static_cast<Real>(2.0) * lx);
    const Real inv_2dy = static_cast<Real>(ny) / (static_cast<Real>(2.0) * ly);
    Real gx = (BlurredPhiAt(eps, nx, ny, x + 1, y) -
               BlurredPhiAt(eps, nx, ny, x - 1, y)) *
              inv_2dx;
    Real gy = (BlurredPhiAt(eps, nx, ny, x, y + 1) -
               BlurredPhiAt(eps, nx, ny, x, y - 1)) *
              inv_2dy;
    const Real norm = sqrt(gx * gx + gy * gy);
    if (norm <= eta)
    {
        *vx = 0.0;
        *vy = 0.0;
        return false;
    }

    gx /= norm;
    gy /= norm;
    if ((fabs(gx) >= fabs(gy) && gx < 0.0) || (fabs(gy) > fabs(gx) && gy < 0.0))
    {
        gx = -gx;
        gy = -gy;
    }
    *vx = gx;
    *vy = gy;
    return true;
}

__device__ inline void InterpolatedNormalAt(const Complex* eps,
                                            int64_t nx,
                                            int64_t ny,
                                            Real lx,
                                            Real ly,
                                            Real eta,
                                            int64_t x,
                                            int64_t y,
                                            Real* nxv,
                                            Real* nyv)
{
    *nxv = 0.0;
    *nyv = 0.0;
    if (BoundaryVectorAt(eps, nx, ny, lx, ly, eta, x, y, nxv, nyv))
    {
        return;
    }

    Real sx = 0.0;
    Real sy = 0.0;
    Real sw = 0.0;
    const Real dx_cell = lx / static_cast<Real>(nx);
    const Real dy_cell = ly / static_cast<Real>(ny);
    for (int64_t bx = 0; bx < nx; ++bx)
    {
        for (int64_t by = 0; by < ny; ++by)
        {
            Real bvx = 0.0;
            Real bvy = 0.0;
            if (!BoundaryVectorAt(eps, nx, ny, lx, ly, eta, bx, by, &bvx, &bvy))
            {
                continue;
            }
            int64_t ddx_i = llabs(x - bx);
            int64_t ddy_i = llabs(y - by);
            ddx_i = min(ddx_i, nx - ddx_i);
            ddy_i = min(ddy_i, ny - ddy_i);
            const Real ddx = static_cast<Real>(ddx_i) * dx_cell;
            const Real ddy = static_cast<Real>(ddy_i) * dy_cell;
            const Real dist2 = ddx * ddx + ddy * ddy;
            const Real w = static_cast<Real>(1.0) / (dist2 + eta * eta);
            sx += w * bvx;
            sy += w * bvy;
            sw += w;
        }
    }
    const Real norm = sqrt(sx * sx + sy * sy);
    if (sw > 0.0 && norm > eta)
    {
        *nxv = sx / norm;
        *nyv = sy / norm;
    }
}

__global__ void BuildNormalFieldKernel(const Complex* eps,
                                       int64_t nx,
                                       int64_t ny,
                                       Real lx,
                                       Real ly,
                                       Real eta,
                                       Complex* nx_field,
                                       Complex* ny_field)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = nx * ny;
    if (idx >= total)
    {
        return;
    }

    const int64_t x = idx / ny;
    const int64_t y = idx - x * ny;
    Real nxv = 0.0;
    Real nyv = 0.0;
    InterpolatedNormalAt(eps, nx, ny, lx, ly, eta, x, y, &nxv, &nyv);
    nx_field[idx] = make_complex(nxv, 0.0);
    ny_field[idx] = make_complex(nyv, 0.0);
}

__global__ void BuildNormalProductsKernel(const Complex* eps,
                                          int64_t nx,
                                          int64_t ny,
                                          Real lx,
                                          Real ly,
                                          Real eta,
                                          Complex* nxx,
                                          Complex* nxy,
                                          Complex* nyy)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = nx * ny;
    if (idx >= total)
    {
        return;
    }

    const int64_t x = idx / ny;
    const int64_t y = idx - x * ny;
    Real nxv = 0.0;
    Real nyv = 0.0;
    InterpolatedNormalAt(eps, nx, ny, lx, ly, eta, x, y, &nxv, &nyv);

    nxx[idx] = make_complex(nxv * nxv, 0.0);
    nxy[idx] = make_complex(nxv * nyv, 0.0);
    nyy[idx] = make_complex(nyv * nyv, 0.0);
}

// Cache boundary vectors once; preserve ascending pixel order in interpolation.
__global__ void CachedBoundaryKernel(const Complex* eps, int64_t nx, int64_t ny,
                                    Real lx, Real ly, Real eta, Complex* boundary)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nx * ny) return;
    Real vx = 0, vy = 0;
    BoundaryVectorAt(eps, nx, ny, lx, ly, eta, i / ny, i % ny, &vx, &vy);
    boundary[i] = make_complex(vx, vy);
}

__global__ void CachedNormalProductsKernel(const Complex* boundary, const int64_t* indices,
    int64_t count, int64_t nx, int64_t ny, Real lx, Real ly, Real eta,
    Complex* nxx, Complex* nxy, Complex* nyy)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nx * ny) return;
    Real vx = ComplexReal(boundary[i]), vy = ComplexImag(boundary[i]);
    if (vx == 0 && vy == 0) {
        Real sx = 0, sy = 0, sw = 0;
        const int64_t x = i / ny, y = i % ny;
        const Real dx_cell = lx / static_cast<Real>(nx), dy_cell = ly / static_cast<Real>(ny);
        for (int64_t j = 0; j < count; ++j) {
            const int64_t b = indices[j];
            int64_t dx = llabs(x - b / ny), dy = llabs(y - b % ny);
            dx = min(dx, nx - dx); dy = min(dy, ny - dy);
            const Real xx = static_cast<Real>(dx) * dx_cell, yy = static_cast<Real>(dy) * dy_cell;
            const Real weight = static_cast<Real>(1) / (xx * xx + yy * yy + eta * eta);
            sx += weight * ComplexReal(boundary[b]);
            sy += weight * ComplexImag(boundary[b]);
            sw += weight;
        }
        const Real norm = sqrt(sx * sx + sy * sy);
        if (sw > 0 && norm > eta) { vx = sx / norm; vy = sy / norm; }
    }
    nxx[i] = make_complex(vx * vx, 0);
    nxy[i] = make_complex(vx * vy, 0);
    nyy[i] = make_complex(vy * vy, 0);
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
    Complex kzz = RegularizeGrazingKz(ComplexSqrt_RCWA(diff));

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
    ez[idx] = ComplexDiv(numerator, RegularizeGrazingKz(kz[idx]));
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

__global__ void EnergyBalanceKernel(const Complex* kx,
                                    const Complex* ky,
                                    const Complex* S11,
                                    const Complex* S21,
                                    Complex eps_in,
                                    Complex eps_out,
                                    int64_t n,
                                    int64_t n2,
                                    int64_t source_order,
                                    int64_t source_col,
                                    Real* accum)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    const Complex kx_src = kx[source_order];
    const Complex ky_src = ky[source_order];
    const Complex src_ks = ComplexAdd(ComplexMul(kx_src, kx_src), ComplexMul(ky_src, ky_src));
    const Complex kz_src = ComplexSqrt(ComplexSub(eps_in, src_ks));
    const double kz_inc = static_cast<double>(ComplexReal(kz_src));
    if (kz_inc <= 0.0)
    {
        return;
    }

    const Complex kx_i = kx[idx];
    const Complex ky_i = ky[idx];
    const Complex ks = ComplexAdd(ComplexMul(kx_i, kx_i), ComplexMul(ky_i, ky_i));

    const Complex kz_in = ComplexSqrt(ComplexSub(eps_in, ks));
    const double wt_in = static_cast<double>(ComplexReal(kz_in));
    if (wt_in > 0.0)
    {
        const Complex re = S21[source_col * n2 + idx];
        const Complex rm = S21[source_col * n2 + idx + n];
        const double r2 = ComplexAbsSquaredDouble(re) + ComplexAbsSquaredDouble(rm);
        atomicAdd(accum, static_cast<Real>((wt_in / kz_inc) * r2));
    }

    const Complex kz_out = ComplexSqrt(ComplexSub(eps_out, ks));
    const double wt_out = static_cast<double>(ComplexReal(kz_out));
    if (wt_out > 0.0)
    {
        const Complex te = S11[source_col * n2 + idx];
        const Complex tm = S11[source_col * n2 + idx + n];
        const double t2 = ComplexAbsSquaredDouble(te) + ComplexAbsSquaredDouble(tm);
        atomicAdd(accum + 1, static_cast<Real>((wt_out / kz_inc) * t2));
    }
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
    Complex kz = RegularizeGrazingKz(ComplexSqrt_RCWA(diff));
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

static Status ApplyV0Stored(const VMatrix& V,const Complex* B,int64_t n,Complex* C,bool left,cudaStream_t stream)
{
    const int64_t h=n/2, stripe=std::min<int64_t>(64,h);
    std::vector<Complex> v11(h),v12(h),v21(h),v22(h);
    CHECK(rcwa_memory::Read(V.V_11,v11.data(),h,stream)); CHECK(rcwa_memory::Read(V.V_12,v12.data(),h,stream));
    CHECK(rcwa_memory::Read(V.V_21,v21.data(),h,stream)); CHECK(rcwa_memory::Read(V.V_22,v22.data(),h,stream));
    std::vector<Complex> a(n*stripe),b(n*stripe),c(n*stripe),d(n*stripe);
    if(left) {
        for(int64_t j=0;j<n;j+=stripe) {
            auto nc=std::min(stripe,n-j);
            CHECK(rcwa_memory::ReadRect(B,n,0,j,h,nc,a.data(),stream));
            CHECK(rcwa_memory::ReadRect(B,n,h,j,h,nc,b.data(),stream));
            for(int64_t x=0;x<h*nc;++x) { auto r=x%h;
                c[x]=ComplexAdd(ComplexMul(v11[r],a[x]),ComplexMul(v12[r],b[x]));
                d[x]=ComplexAdd(ComplexMul(v21[r],a[x]),ComplexMul(v22[r],b[x])); }
            CHECK(rcwa_memory::WriteRect(C,n,0,j,h,nc,c.data(),stream));
            CHECK(rcwa_memory::WriteRect(C,n,h,j,h,nc,d.data(),stream));
        }
    } else {
        for(int64_t j=0;j<h;j+=stripe) {
            auto nc=std::min(stripe,h-j);
            CHECK(rcwa_memory::ReadRect(B,n,0,j,n,nc,a.data(),stream));
            CHECK(rcwa_memory::ReadRect(B,n,0,h+j,n,nc,b.data(),stream));
            for(int64_t x=0;x<n*nc;++x) { auto col=j+x/n;
                c[x]=ComplexAdd(ComplexMul(a[x],v11[col]),ComplexMul(b[x],v21[col]));
                d[x]=ComplexAdd(ComplexMul(a[x],v12[col]),ComplexMul(b[x],v22[col])); }
            CHECK(rcwa_memory::WriteRect(C,n,0,j,n,nc,c.data(),stream));
            CHECK(rcwa_memory::WriteRect(C,n,0,h+j,n,nc,d.data(),stream));
        }
    }
    return Status::kSuccess;
}

static Status UnpackSBlocks(const Complex* packed,Complex* matrix,int64_t n,cudaStream_t stream)
{
    CHECK(CopyDiagonalBlock(packed,n,matrix,2*n,0,0,make_complex(1,0),stream));
    CHECK(CopyDiagonalBlock(packed+n,n,matrix,2*n,n,0,make_complex(1,0),stream));
    CHECK(CopyDiagonalBlock(packed+2*n,n,matrix,2*n,0,n,make_complex(1,0),stream));
    return CopyDiagonalBlock(packed+3*n,n,matrix,2*n,n,n,make_complex(1,0),stream);
}

static Status
ApplyV0Left(const VMatrix& V, const Complex* B, int64_t n, DeviceMatrix* out, cudaStream_t stream)
{
    if (!B || !out || !V.V_11 || !V.V_12 || !V.V_21 || !V.V_22 || (n % 2) != 0)
    {
        return Status::kInvalidValue;
    }
    CHECK(out->Allocate(n, n));
    if (rcwa_memory::IsHost(B) || rcwa_memory::IsHost(out->get())) return ApplyV0Stored(V,B,n,out->get(),true,stream);
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
    if (rcwa_memory::IsHost(B) || rcwa_memory::IsHost(out->get())) return ApplyV0Stored(V,B,n,out->get(),false,stream);
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
    rcwa_memory::Configure(order_x_,order_y_);
    auto_order_enabled_ = (order_x == 0 && order_y == 0);
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
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(S11.get(), 0, matrix_bytes, use_stream)));
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(S12.get(), 0, matrix_bytes, use_stream)));
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(S21.get(), 0, matrix_bytes, use_stream)));
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(S22.get(), 0, matrix_bytes, use_stream)));

    if (rcwa_memory::IsHost(S11.get()) || rcwa_memory::IsHost(S12.get()) ||
        rcwa_memory::IsHost(S21.get()) || rcwa_memory::IsHost(S22.get())) {
        DeviceVector p11,p12,p21,p22;
        CHECK(p11.Allocate(4*n)); CHECK(p12.Allocate(4*n)); CHECK(p21.Allocate(4*n)); CHECK(p22.Allocate(4*n));
    BuildPortSBlocksKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(V0_.V_11,
                                                                    V0_.V_12,
                                                                    V0_.V_21,
                                                                    V0_.V_22,
                                                                    port11.get(),
                                                                    port12.get(),
                                                                    port21.get(),
                                                                    port22.get(),
                                                                    input_port ? 1 : 0,
                                                                    p11.get(),
                                                                    p12.get(),
                                                                    p21.get(),
                                                                    p22.get(),
                                                                    n, true);
    CHECK(StatusFromCuda(cudaGetLastError()));
        CHECK(UnpackSBlocks(p11.get(),S11.get(),n,use_stream)); CHECK(UnpackSBlocks(p12.get(),S12.get(),n,use_stream));
        CHECK(UnpackSBlocks(p21.get(),S21.get(),n,use_stream)); CHECK(UnpackSBlocks(p22.get(),S22.get(),n,use_stream));
    } else {
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
    }

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

Status RCWA::ResetOrder(int order_x, int order_y)
{
    if (order_x < 0 || order_y < 0)
    {
        return Status::kInvalidValue;
    }

    FreeSMatrix(&stack_smatrix_);
    FreeGlobalSMatrix();
    FreeKVectors();
    FreeSource();

    rcwa_memory::Configure(order_x,order_y);
    order_x_ = order_x;
    order_y_ = order_y;
    order_N_ = static_cast<int64_t>((2 * order_x_ + 1) * (2 * order_y_ + 1));
    Gx_norm_ = 1.0 / (Lx_ * freq_);
    Gy_norm_ = 1.0 / (Ly_ * freq_);
    return Status::kSuccess;
}

Status RCWA::ReplayRecordedSetup(RCWA* target) const
{
    if (!target)
    {
        return Status::kInvalidValue;
    }

    target->SetExpmMethod(expm_method_);
    target->SetHighOrder(high_order_);
    CHECK(target->AddInputLayer(eps_in_));
    CHECK(target->AddOutputLayer(eps_out_));

    if (!incident_recorded_)
    {
        return Status::kInvalidValue;
    }

    if (incident_uses_layer_)
    {
        CHECK(target->SetIncidentAngles(
            incident_inc_ang_, incident_azi_ang_, incident_angle_layer_input_ ? "input" : "output"));
    }
    else
    {
        CHECK(target->SetIncidentAngles(incident_inc_ang_, incident_azi_ang_, incident_eps_ref_));
    }

    for (const RecordedLayer& layer : recorded_layers_)
    {
        if (layer.type == RecordedLayerType::kHomogeneous)
        {
            CHECK(target->AddLayerHomogeneous(layer.thickness, layer.eps));
        }
        else if (layer.type == RecordedLayerType::kMaterial)
        {
            Complex* d_material = nullptr;
            const size_t bytes = layer.material.size() * sizeof(Complex);
            CHECK(StatusFromCuda(cudaMalloc(&d_material, bytes)));
            const cudaError_t copy_status = rcwa_memory::MemcpyAsync(d_material,
                                                            layer.material.data(),
                                                            bytes,
                                                            cudaMemcpyHostToDevice,
                                                            target->stream_);
            if (copy_status != cudaSuccess)
            {
                cudaFree(d_material);
                return Status::kCudaError;
            }
            const Status status = target->AddLayer(layer.thickness,
                                                   d_material,
                                                   layer.nx,
                                                   layer.ny,
                                                   layer.factorization);
            const Status sync_status = StatusFromCuda(cudaStreamSynchronize(target->stream_));
            cudaFree(d_material);
            CHECK(status);
            CHECK(sync_status);
        }
    }

    if (source_recorded_)
    {
        CHECK(target->SourcePlanewave(source_ex_, source_ey_, source_forward_ ? "forward" : "backward"));
    }
    return Status::kSuccess;
}

double RCWA::CurrentReflectance() const
{
    if (!HasSMatrix(global_smatrix_))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int64_t n2 = 2 * order_N_;
    const int64_t oy_len = static_cast<int64_t>(2 * order_y_ + 1);
    const int64_t source_order = static_cast<int64_t>(order_x_) * oy_len + order_y_;
    Complex value;
    if(rcwa_memory::Read(global_smatrix_.S_11+source_order*n2+source_order,&value,1,stream_)!=Status::kSuccess)
        return std::numeric_limits<double>::quiet_NaN();
    return ComplexAbsSquaredDouble(value);
}

Status RCWA::ResolveAutoOrder()
{
    if (!auto_order_enabled_)
    {
        return Status::kSuccess;
    }
    if (auto_order_resolving_)
    {
        return Status::kInvalidValue;
    }

    const AutoOrderOptions& cfg = auto_order_options_;
    int effective_order_max = cfg.order_max;
    const char* memory_mode = std::getenv("RCWA_MEMORY_MODE");
    const int memory_order_max = cfg.cap_order_by_memory && memory_mode && std::strcmp(memory_mode,"resident")==0
                                     ? MemoryLimitedOrderMax(cfg.order_max,
                                                             expm_method_,
                                                             high_order_,
                                                             cfg.memory_safety_fraction)
                                     : cfg.order_max;
    effective_order_max = std::min(effective_order_max, memory_order_max);
    for (const RecordedLayer& layer : recorded_layers_)
    {
        if (layer.type == RecordedLayerType::kMaterial)
        {
            // The Fourier orders should stay below the sampled material bandwidth.
            // Lalanne also needs normal-vector products on the same material grid.
            effective_order_max =
                std::min(effective_order_max,
                         static_cast<int>((std::min(layer.nx, layer.ny) - 1) / 4));
        }
    }
    if (cfg.order_start > effective_order_max || cfg.order_step <= 0 ||
        cfg.order_min_step <= 0 || cfg.order_max_step < cfg.order_min_step ||
        cfg.window_size < 2 || cfg.max_evaluations <= 0 ||
        cfg.reflectance_tol <= static_cast<Real>(0.0) ||
        cfg.reflectance_floor <= static_cast<Real>(0.0))
    {
        return Status::kInvalidValue;
    }

    struct Sample
    {
        int order = 0;
        double reflectance = 0.0;
        double elapsed_ms = 0.0;
        double relative_change = std::numeric_limits<double>::quiet_NaN();
        double window_range = std::numeric_limits<double>::quiet_NaN();
        int next_step = 0;
    };

    auto relative_change = [&](double a, double b) {
        const double denom =
            std::max({std::abs(a), std::abs(b), static_cast<double>(cfg.reflectance_floor)});
        return std::abs(a - b) / denom;
    };

    auto window_range = [&](const std::vector<Sample>& samples) {
        if (samples.size() < static_cast<size_t>(cfg.window_size))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const size_t begin = samples.size() - static_cast<size_t>(cfg.window_size);
        double min_r = std::numeric_limits<double>::infinity();
        double max_r = -std::numeric_limits<double>::infinity();
        double sum_abs = 0.0;
        for (size_t i = begin; i < samples.size(); ++i)
        {
            min_r = std::min(min_r, samples[i].reflectance);
            max_r = std::max(max_r, samples[i].reflectance);
            sum_abs += std::abs(samples[i].reflectance);
        }
        const double mean_abs = sum_abs / static_cast<double>(cfg.window_size);
        return (max_r - min_r) /
               std::max(mean_abs, static_cast<double>(cfg.reflectance_floor));
    };

    auto choose_next_step = [&](double change, double range) {
        double signal = change;
        if (std::isfinite(range))
        {
            signal = std::isfinite(signal) ? std::max(signal, range) : range;
        }
        if (!cfg.adaptive_step || !std::isfinite(signal))
        {
            return std::max(cfg.order_min_step,
                            std::min(cfg.order_step, cfg.order_max_step));
        }
        const double tol = static_cast<double>(cfg.reflectance_tol);
        if (signal > tol * static_cast<double>(cfg.high_change_factor))
        {
            return cfg.order_min_step;
        }
        if (signal < tol * static_cast<double>(cfg.low_change_factor))
        {
            return cfg.order_max_step;
        }
        return std::max(cfg.order_min_step, std::min(cfg.order_step, cfg.order_max_step));
    };

    auto run_order = [&](int order, Sample* sample) -> Status {
        auto start = std::chrono::steady_clock::now();
        RCWA trial(freq_, order, order, Lx_, Ly_, stream_);
        trial.auto_order_enabled_ = false;
        CHECK(ReplayRecordedSetup(&trial));
        CHECK(trial.SolveGlobalSMatrix());
        CHECK(StatusFromCuda(cudaStreamSynchronize(stream_)));
        auto stop = std::chrono::steady_clock::now();
        sample->order = order;
        sample->reflectance = trial.CurrentReflectance();
        sample->elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        if (!std::isfinite(sample->reflectance))
        {
            return Status::kInvalidValue;
        }
        return Status::kSuccess;
    };

    auto_order_resolving_ = true;
    struct AutoOrderResolvingGuard
    {
        RCWA* rcwa = nullptr;
        ~AutoOrderResolvingGuard()
        {
            if (rcwa)
            {
                rcwa->auto_order_resolving_ = false;
            }
        }
    } resolving_guard{this};
    auto_order_converged_ = false;
    auto_selected_order_ = effective_order_max;

    double total_ms = 0.0;
    if (cfg.warmup)
    {
        Sample warmup{};
        CHECK(run_order(cfg.order_start, &warmup));
        total_ms += warmup.elapsed_ms;
    }

    std::vector<Sample> samples;
    std::cout << std::defaultfloat << std::setprecision(6);
    std::cout << "\nRCWA auto order selection\n";
    std::cout << "tol=" << static_cast<double>(cfg.reflectance_tol)
              << ", window=" << cfg.window_size
              << ", order_start=" << cfg.order_start
              << ", user_order_max=" << cfg.order_max
              << ", effective_order_max=" << effective_order_max
              << ", memory_order_max=" << memory_order_max
              << ", adaptive_step=" << (cfg.adaptive_step ? "yes" : "no")
              << ", confirm=" << (cfg.confirm_convergence ? "yes" : "no") << "\n";
    std::cout << std::setw(7) << "order"
              << std::setw(18) << "Reflectance"
              << std::setw(16) << "change %"
              << std::setw(16) << "window %"
              << std::setw(10) << "next"
              << std::setw(12) << "state"
              << std::setw(14) << "ms\n";

    int order = cfg.order_start;
    int candidate_order = -1;
    double candidate_reflectance = std::numeric_limits<double>::quiet_NaN();
    while (order <= effective_order_max)
    {
        if (static_cast<int>(samples.size()) >= cfg.max_evaluations ||
            total_ms >= static_cast<double>(cfg.max_total_ms))
        {
            break;
        }

        Sample sample{};
        CHECK(run_order(order, &sample));
        total_ms += sample.elapsed_ms;
        if (!samples.empty())
        {
            sample.relative_change = relative_change(sample.reflectance, samples.back().reflectance);
        }
        samples.push_back(sample);
        samples.back().window_range = window_range(samples);
        samples.back().next_step =
            choose_next_step(samples.back().relative_change, samples.back().window_range);
        const bool window_converged =
            samples.size() >= static_cast<size_t>(cfg.window_size) &&
            samples.back().window_range <= static_cast<double>(cfg.reflectance_tol);
        const bool candidate_confirmed =
            candidate_order >= 0 && window_converged &&
            relative_change(sample.reflectance, candidate_reflectance) <=
                static_cast<double>(cfg.reflectance_tol);
        const char* state = "-";
        if (candidate_confirmed)
        {
            state = "confirm";
        }
        else if (window_converged)
        {
            state = "candidate";
        }

        std::cout << std::setw(7) << order
                  << std::setw(18) << std::fixed << std::setprecision(10)
                  << sample.reflectance
                  << std::setw(16);
        if (std::isfinite(sample.relative_change))
        {
            std::cout << std::fixed << std::setprecision(4) << sample.relative_change * 100.0;
        }
        else
        {
            std::cout << "-";
        }
        std::cout << std::setw(16);
        if (std::isfinite(samples.back().window_range))
        {
            std::cout << std::fixed << std::setprecision(4) << samples.back().window_range * 100.0;
        }
        else
        {
            std::cout << "-";
        }
        std::cout << std::setw(10) << samples.back().next_step
                  << std::setw(12) << state
                  << std::setw(14) << std::fixed << std::setprecision(1)
                  << sample.elapsed_ms << "\n";

        if (candidate_confirmed || (window_converged && !cfg.confirm_convergence))
        {
            auto_order_converged_ = true;
            auto_selected_order_ = candidate_confirmed ? candidate_order : order;
            break;
        }
        if (window_converged)
        {
            candidate_order = order;
            candidate_reflectance = sample.reflectance;
        }
        else
        {
            candidate_order = -1;
            candidate_reflectance = std::numeric_limits<double>::quiet_NaN();
        }

        const int next_order = order + samples.back().next_step;
        if (next_order > effective_order_max)
        {
            if (order == effective_order_max)
            {
                break;
            }
            order = effective_order_max;
        }
        else
        {
            order = next_order;
        }
    }

    if (!auto_order_converged_ && !samples.empty())
    {
        auto_selected_order_ = samples.back().order;
    }

    std::cout << "RCWA auto order selected " << auto_selected_order_
              << " (converged=" << (auto_order_converged_ ? "yes" : "no")
              << ", total_ms=" << std::fixed << std::setprecision(1) << total_ms << ")\n\n";

    CHECK(ResetOrder(auto_selected_order_, auto_selected_order_));
    auto_order_enabled_ = false;
    CHECK(ReplayRecordedSetup(this));
    return Status::kSuccess;
}

Status RCWA::SetIncidentAngles(Real inc_ang, Real azi_ang, Real eps_ref)
{
    incident_recorded_ = true;
    incident_uses_layer_ = false;
    incident_inc_ang_ = inc_ang;
    incident_azi_ang_ = azi_ang;
    incident_eps_ref_ = eps_ref;
    if (IsAutoRecording())
    {
        return Status::kSuccess;
    }
    return BuildKxKy(inc_ang, azi_ang, eps_ref);
}

Status RCWA::SetIncidentAngles(Real inc_ang, Real azi_ang, const char* angle_layer)
{
    angle_layer_input_ = IsInputLayerName(angle_layer);
    incident_recorded_ = true;
    incident_uses_layer_ = true;
    incident_inc_ang_ = inc_ang;
    incident_azi_ang_ = azi_ang;
    incident_angle_layer_input_ = angle_layer_input_;
    if (IsAutoRecording())
    {
        return Status::kSuccess;
    }
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
    source_recorded_ = true;
    source_ex_ = ex_amp;
    source_ey_ = ey_amp;
    if (IsAutoRecording())
    {
        return Status::kSuccess;
    }
    FreeSource();

    const size_t count = static_cast<size_t>(2 * order_N_);
    CHECK(StatusFromCuda(cudaMalloc(&source_ei_, count * sizeof(Complex))));
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(source_ei_, 0, count * sizeof(Complex), stream_)));

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

    // ── Phase 0: scaling ──────────────────────────────────────────────────
    // T = exp(A · k₀d)  where  A = [0, −P; Q, 0]  (2n₂ × 2n₂ block matrix).
    // 18th-order optimized Taylor expansion (Bader, Blanes & Casas, 2019)
    // combined with scaling-and-squaring.
    //
    // Phases:
    //   0. Scaling:  m = ⌈log₂ max(‖P‖₁, ‖Q‖₁)·k₀d⌉,  s = 2⁻ᵐ·k₀d
    //   1. Block powers A², A³, A⁶  (n₂ × n₂ blocks)
    //   2. B⁽ⁱ⁾ = a₀ᵢ·I + a₂ᵢ·A² + a₄ᵢ·A⁶  (i = 0..4, 5 × 4 blocks)
    //   3. A⁽⁹⁾ blocks from B⁽⁰⁾·B⁽⁴⁾ + B⁽³⁾
    //   4. T blocks from (B⁽²⁾ + A⁽⁹⁾)·A⁽⁹⁾ + B⁽¹⁾
    //   5. Repeated squaring (m rounds)
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

    // ── Phase 2: B⁽ⁱ⁾ = a₀ᵢ·I + a₂ᵢ·A² + a₄ᵢ·A⁶  (i = 0..4) ────────────
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
        CHECK(ScaleAddInPlace(B_11[i].get(), A2_11.get(), n, c2, use_stream));
        CHECK(ScaleAddInPlace(B_11[i].get(), A6_11.get(), n, c4, use_stream));

        CHECK(B_12[i].Allocate(n, n));
        CHECK(ScaleMatrix(A_12.get(), B_12[i].get(), n, c1, use_stream));
        CHECK(ScaleAddInPlace(B_12[i].get(), A3_12.get(), n, c3, use_stream));

        CHECK(B_21[i].Allocate(n, n));
        CHECK(ScaleMatrix(A_21.get(), B_21[i].get(), n, c1, use_stream));
        CHECK(ScaleAddInPlace(B_21[i].get(), A3_21.get(), n, c3, use_stream));

        CHECK(B_22[i].Allocate(n, n));
        CHECK(ScaleMatrix(I2.get(), B_22[i].get(), n, c0, use_stream));
        CHECK(ScaleAddInPlace(B_22[i].get(), A2_22.get(), n, c2, use_stream));
        CHECK(ScaleAddInPlace(B_22[i].get(), A6_22.get(), n, c4, use_stream));
    }

    // Phase 2 cleanup: free consumed powers, identity, and full A-matrix blocks
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

    // ── Phase 3: A⁽⁹⁾ blocks = B⁽⁰⁾·B⁽⁴⁾ + B⁽³⁾ ──────────────────────────

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

    // Phase 3 cleanup: free B[0],B[3],B[4] — consumed by A⁽⁹⁾ blocks
    for (int i : {0, 3, 4})
    {
        B_11[i].Reset();
        B_12[i].Reset();
        B_21[i].Reset();
        B_22[i].Reset();
    }

    // ── Phase 4: T blocks = (B⁽²⁾ + A⁽⁹⁾)·A⁽⁹⁾ + B⁽¹⁾ ─────────────────
    // Precompute intermediate sums in-place on B[2] buffers (consumed after Phase 4)
    CHECK(AddInPlace(ops, B_11[2].get(), A9_11.get(), n));
    CHECK(AddInPlace(ops, B_12[2].get(), A9_12.get(), n));
    CHECK(AddInPlace(ops, B_21[2].get(), A9_21.get(), n));
    CHECK(AddInPlace(ops, B_22[2].get(), A9_22.get(), n));

    DeviceMatrix temp1;
    DeviceMatrix temp2;

    DeviceMatrix T_11;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(MatMul(ops, B_11[2].get(), A9_11.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, B_12[2].get(), A9_21.get(), temp2.get(), n));
        CHECK(T_11.Allocate(n, n));
        CHECK(AddMatrix(ops, B_11[1].get(), temp1.get(), T_11.get(), n));
        CHECK(AddInPlace(ops, T_11.get(), temp2.get(), n));
    }

    DeviceMatrix T_12;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(MatMul(ops, B_11[2].get(), A9_12.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, B_12[2].get(), A9_22.get(), temp2.get(), n));
        CHECK(T_12.Allocate(n, n));
        CHECK(AddMatrix(ops, B_12[1].get(), temp1.get(), T_12.get(), n));
        CHECK(AddInPlace(ops, T_12.get(), temp2.get(), n));
    }

    DeviceMatrix T_21;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(MatMul(ops, B_21[2].get(), A9_11.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, B_22[2].get(), A9_21.get(), temp2.get(), n));
        CHECK(T_21.Allocate(n, n));
        CHECK(AddMatrix(ops, B_21[1].get(), temp1.get(), T_21.get(), n));
        CHECK(AddInPlace(ops, T_21.get(), temp2.get(), n));
    }

    DeviceMatrix T_22;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(MatMul(ops, B_21[2].get(), A9_12.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, B_22[2].get(), A9_22.get(), temp2.get(), n));
        CHECK(T_22.Allocate(n, n));
        CHECK(AddMatrix(ops, B_22[1].get(), temp1.get(), T_22.get(), n));
        CHECK(AddInPlace(ops, T_22.get(), temp2.get(), n));
    }

    // Phase 4 cleanup: free B[1],B[2], all A⁽⁹⁾ blocks, and temps —
    // none are needed in Phase 5 squaring.  Peak memory drops to
    // 1×scratch + 4×T + 4×tmp = 9 matrices.
    for (int i : {1, 2})
    {
        B_11[i].Reset();
        B_12[i].Reset();
        B_21[i].Reset();
        B_22[i].Reset();
    }
    A9_11.Reset();
    A9_12.Reset();
    A9_21.Reset();
    A9_22.Reset();
    temp1.Reset();
    temp2.Reset();

    // ── Phase 5: repeated squaring (m rounds) ─────────────────────────────
    // T ← T²  where T is a 2×2 block matrix:  [T₁₁, T₁₂; T₂₁, T₂₂]
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

// ─────────────────────────────────────────────────────────────────────────────
// Compute Cos(Θ) and Sinc(Θ) ≡ Θ⁻¹·sin(Θ) directly from P·Q.
//
// Both are even functions of  Θ = k₀d·√(P·Q), depending only on Θ².
// A 12th-order optimized Taylor expansion (Bader, Blanes & Casas, 2019)
// followed by repeated double-angle squaring yields the result without
// ever forming the matrix square root √(P·Q).
//
// Phases (i = 0..3):
//   1.  A² = 2⁻²ᵐ·(k₀d)²·(P·Q),   m = ⌈½ log₂ max((k₀d)²‖P·Q‖₁, 1)⌉
//   2.  B⁽ⁱ⁾  = a₀ᵢ·I − a₂ᵢ·A²,   B̄⁽ⁱ⁾ = a₁ᵢ·I − a₃ᵢ·A²
//   3.  A⁽⁶⁾  = (B⁽³⁾)² − A²·(B̄⁽³⁾)² + B⁽²⁾,
//       Ā⁽⁶⁾  = B̄⁽³⁾·B⁽³⁾ + B⁽³⁾·B̄⁽³⁾ + B̄⁽²⁾
//   4.  C     = B⁽⁰⁾ + (B⁽¹⁾ + A⁽⁶⁾)·A⁽⁶⁾ − A²·(B̄⁽¹⁾ + Ā⁽⁶⁾)·Ā⁽⁶⁾
//       Sinc  = B̄⁽⁰⁾ + (B̄⁽¹⁾ + Ā⁽⁶⁾)·A⁽⁶⁾ + (B⁽¹⁾ + A⁽⁶⁾)·Ā⁽⁶⁾
//   5.  m rounds:  Sinc ← Sinc·C,   C ← 2·C² − I
// ─────────────────────────────────────────────────────────────────────────────
static Status ComputeCosSincFromPQOrder12(const Complex* P,
                                          const Complex* Q,
                                          int64_t n,
                                          Real k0d,
                                          Complex* Cos,
                                          Complex* Sinc,
                                          cudaStream_t use_stream)
{
    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    const Complex neg_one = make_complex(-1.0, 0.0);
    const Complex two = make_complex(2.0, 0.0);
    const Real k0d_sq = static_cast<Real>(k0d * k0d);

    // ── 1.  A⁽²⁾ = 2⁻²ᵐ · (k₀d)² · (P·Q) ──────────────────────────────
    DeviceMatrix PQ;
    CHECK(PQ.Allocate(n, n));
    CHECK(MatMul(ops, P, Q, PQ.get(), n));

    double PQnorm = 0.0;
    CHECK(ComputeMatrixNorm1(PQ.get(), n, &PQnorm, use_stream));
    int m = static_cast<int>(std::ceil(0.5 * std::log2(std::max(
        static_cast<double>(k0d_sq) * PQnorm, 1.0))));
    if (m < 0)
    {
        m = 0;
    }

    const Real scale_sq = static_cast<Real>(std::pow(2.0, -2 * m));
    const Complex a2_scale = make_complex(static_cast<Real>(scale_sq * k0d_sq), 0.0);
    DeviceMatrix A2;
    CHECK(A2.Allocate(n, n));
    CHECK(ScaleMatrix(PQ.get(), A2.get(), n, a2_scale, use_stream));
    PQ.Reset(); // P·Q no longer needed

    // ── 2.  B⁽ⁱ⁾  and  B̄⁽ⁱ⁾  (i = 0..3) ───────────────────────────────
    DeviceMatrix I_mat;
    CHECK(I_mat.Allocate(n, n));
    CHECK(MakeIdentity(I_mat.get(), n, use_stream));

    DeviceMatrix scratch;
    CHECK(scratch.Allocate(n, n));

    DeviceMatrix B[4], Bbar[4];
    for (int i = 0; i < 4; ++i)
    {
        const Real a0 = static_cast<Real>(kExpConstantOrder12[i][0]);
        const Real a1 = static_cast<Real>(kExpConstantOrder12[i][1]);
        const Real a2 = static_cast<Real>(kExpConstantOrder12[i][2]);
        const Real a3 = static_cast<Real>(kExpConstantOrder12[i][3]);

        // B⁽ⁱ⁾ = a₀ᵢ·I − a₂ᵢ·A²
        CHECK(B[i].Allocate(n, n));
        CHECK(ScaleMatrix(I_mat.get(), B[i].get(), n, make_complex(a0, 0.0), use_stream));
        CHECK(ScaleAddInPlace(B[i].get(), A2.get(), n, make_complex(-a2, 0.0), use_stream));

        // B̄⁽ⁱ⁾ = a₁ᵢ·I − a₃ᵢ·A²
        CHECK(Bbar[i].Allocate(n, n));
        CHECK(ScaleMatrix(I_mat.get(), Bbar[i].get(), n, make_complex(a1, 0.0), use_stream));
        CHECK(ScaleAddInPlace(Bbar[i].get(), A2.get(), n, make_complex(-a3, 0.0), use_stream));
    }
    I_mat.Reset(); // identity consumed into B, B̄ arrays

    // ── 3.  A⁽⁶⁾  and  Ā⁽⁶⁾ ────────────────────────────────────────────
    //
    //  A⁽⁶⁾ = (B⁽³⁾)² − A²·(B̄⁽³⁾)² + B⁽²⁾
    //  Ā⁽⁶⁾ = B̄⁽³⁾·B⁽³⁾ + B⁽³⁾·B̄⁽³⁾ + B̄⁽²⁾
    DeviceMatrix A6, Abar6;
    DeviceMatrix tmp;
    CHECK(tmp.Allocate(n, n));
    CHECK(scratch.Allocate(n, n)); // ensure scratch is allocated (no-op if already)

    // scratch = A² · (B̄⁽³⁾)²
    CHECK(MatMul(ops, Bbar[3].get(), Bbar[3].get(), scratch.get(), n));
    CHECK(MatMul(ops, A2.get(), scratch.get(), tmp.get(), n));

    // A6 = (B⁽³⁾)² + B⁽²⁾ − A²·(B̄⁽³⁾)²
    CHECK(A6.Allocate(n, n));
    CHECK(MatMul(ops, B[3].get(), B[3].get(), A6.get(), n));
    CHECK(AddInPlace(ops, A6.get(), B[2].get(), n));
    CHECK(ScaleAddInPlace(A6.get(), tmp.get(), n, neg_one, use_stream));

    // Abar6 = B̄⁽³⁾·B⁽³⁾ + B⁽³⁾·B̄⁽³⁾ + B̄⁽²⁾
    CHECK(Abar6.Allocate(n, n));
    CHECK(MatMul(ops, Bbar[3].get(), B[3].get(), Abar6.get(), n));
    CHECK(MatMul(ops, B[3].get(), Bbar[3].get(), tmp.get(), n));
    CHECK(AddInPlace(ops, Abar6.get(), tmp.get(), n));
    CHECK(AddInPlace(ops, Abar6.get(), Bbar[2].get(), n));
    tmp.Reset(); // temporary buffer consumed

    // Free B[2],B[3],Bbar[2],Bbar[3] — consumed by A⁽⁶⁾ / Ā⁽⁶⁾
    B[2].Reset();
    B[3].Reset();
    Bbar[2].Reset();
    Bbar[3].Reset();

    // ── 4.  C  and  Sinc ────────────────────────────────────────────────
    //
    //  T₁ = B⁽¹⁾ + A⁽⁶⁾    (into freed B[2])
    //  T₂ = B̄⁽¹⁾ + Ā⁽⁶⁾    (into freed Bbar[2])
    //  T₁A6 = T₁·A⁽⁶⁾      (into freed B[3])
    //  T₂Ā6 = T₂·Ā⁽⁶⁾      (into freed Bbar[3])
    CHECK(B[2].Allocate(n, n));
    CHECK(AddMatrix(ops, B[1].get(), A6.get(), B[2].get(), n));

    CHECK(Bbar[2].Allocate(n, n));
    CHECK(AddMatrix(ops, Bbar[1].get(), Abar6.get(), Bbar[2].get(), n));

    CHECK(B[3].Allocate(n, n));
    CHECK(MatMul(ops, B[2].get(), A6.get(), B[3].get(), n)); // T₁A6

    CHECK(Bbar[3].Allocate(n, n));
    CHECK(MatMul(ops, Bbar[2].get(), Abar6.get(), Bbar[3].get(), n)); // T₂Ā6

    // C = B⁽⁰⁾ + T₁A6 − A² · T₂Ā6
    DeviceMatrix C_mat, Sinc_mat;
    CHECK(C_mat.Allocate(n, n));
    CHECK(AddMatrix(ops, B[0].get(), B[3].get(), C_mat.get(), n));
    CHECK(MatMul(ops, A2.get(), Bbar[3].get(), scratch.get(), n));
    CHECK(ScaleAddInPlace(C_mat.get(), scratch.get(), n, neg_one, use_stream));

    // T₂A6 = T₂ · A⁽⁶⁾    (overwrite B[3] — T₁A6 already consumed into C)
    CHECK(MatMul(ops, Bbar[2].get(), A6.get(), B[3].get(), n));

    // T₁Ā6 = T₁ · Ā⁽⁶⁾    (overwrite Bbar[3] — T₂Ā6 already consumed into C)
    CHECK(MatMul(ops, B[2].get(), Abar6.get(), Bbar[3].get(), n));

    // Sinc = B̄⁽⁰⁾ + T₂A6 + T₁Ā6
    CHECK(Sinc_mat.Allocate(n, n));
    CHECK(AddMatrix(ops, Bbar[0].get(), B[3].get(), Sinc_mat.get(), n));
    CHECK(AddInPlace(ops, Sinc_mat.get(), Bbar[3].get(), n));

    // Free everything except C_mat, Sinc_mat, scratch
    A2.Reset();
    A6.Reset();
    Abar6.Reset();
    for (int i = 0; i < 4; ++i)
    {
        B[i].Reset();
        Bbar[i].Reset();
    }

    // ── 5.  Repeated squaring (m rounds of double-angle) ─────────────────
    for (int i = 0; i < m; ++i)
    {
        // Sinc ← Sinc·C      (sin(2x)/(2x) = cos(x)·sin(x)/x)
        CHECK(MatMul(ops, Sinc_mat.get(), C_mat.get(), scratch.get(), n));
        Sinc_mat.Swap(scratch);

        // C ← 2·C² − I
        CHECK(MatMul(ops, C_mat.get(), C_mat.get(), scratch.get(), n));
        CHECK(ScaleInPlace(scratch.get(), n, two, use_stream));
        CHECK(AddDiagonalConstantInPlace(scratch.get(), n, neg_one, use_stream));
        C_mat.Swap(scratch);
    }

    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Complex);
    CHECK(StatusFromCuda(
        rcwa_memory::MemcpyAsync(Cos, C_mat.get(), bytes, cudaMemcpyDeviceToDevice, use_stream)));
    CHECK(StatusFromCuda(
        rcwa_memory::MemcpyAsync(Sinc, Sinc_mat.get(), bytes, cudaMemcpyDeviceToDevice, use_stream)));
    return Status::kSuccess;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute Cos(Θ) and Sinc(Θ) ≡ Θ⁻¹·sin(Θ) — Order18 variant.
//
// 18th-order optimized Taylor expansion (Bader, Blanes & Casas, 2019).
//
// Phases (i = 0..4):
//   1.  A² = 2⁻²ᵐ·(k₀d)²·(P·Q),   m = ⌈½ log₂ max((k₀d)²‖P·Q‖₁, 1)⌉
//   2.  A⁶ = (A²)³
//   3.  B⁽ⁱ⁾  = a₀ᵢ·I − a₂ᵢ·A² − a₄ᵢ·A⁶,   B̄⁽ⁱ⁾ = a₁ᵢ·I − a₃ᵢ·A²
//   4.  A⁽⁹⁾  = B⁽⁰⁾·B⁽⁴⁾ − A²·(B̄⁽⁰⁾·B̄⁽⁴⁾) + B⁽³⁾
//       Ā⁽⁹⁾  = B̄⁽⁰⁾·B⁽⁴⁾ + B⁽⁰⁾·B̄⁽⁴⁾ + B̄⁽³⁾
//   5.  C     = B⁽¹⁾ + (B⁽²⁾ + A⁽⁹⁾)·A⁽⁹⁾ − A²·(B̄⁽²⁾ + Ā⁽⁹⁾)·Ā⁽⁹⁾
//       Sinc  = B̄⁽¹⁾ + (B̄⁽²⁾ + Ā⁽⁹⁾)·A⁽⁹⁾ + (B⁽²⁾ + A⁽⁹⁾)·Ā⁽⁹⁾
//   6.  m rounds:  Sinc ← Sinc·C,   C ← 2·C² − I
// ─────────────────────────────────────────────────────────────────────────────
static Status ComputeCosSincFromPQOrder18(const Complex* P,
                                          const Complex* Q,
                                          int64_t n,
                                          Real k0d,
                                          Complex* Cos,
                                          Complex* Sinc,
                                          cudaStream_t use_stream)
{
    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    const Complex neg_one = make_complex(-1.0, 0.0);
    const Complex two = make_complex(2.0, 0.0);
    const Real k0d_sq = static_cast<Real>(k0d * k0d);

    // ── 1.  A⁽²⁾ = 2⁻²ᵐ · (k₀d)² · (P·Q) ──────────────────────────────
    DeviceMatrix PQ;
    CHECK(PQ.Allocate(n, n));
    CHECK(MatMul(ops, P, Q, PQ.get(), n));

    double PQnorm = 0.0;
    CHECK(ComputeMatrixNorm1(PQ.get(), n, &PQnorm, use_stream));
    int m = static_cast<int>(std::ceil(0.5 * std::log2(std::max(
        static_cast<double>(k0d_sq) * PQnorm, 1.0))));
    if (m < 0)
    {
        m = 0;
    }

    const Real scale_sq = static_cast<Real>(std::pow(2.0, -2 * m));
    const Complex a2_scale = make_complex(static_cast<Real>(scale_sq * k0d_sq), 0.0);
    DeviceMatrix A2;
    CHECK(A2.Allocate(n, n));
    CHECK(ScaleMatrix(PQ.get(), A2.get(), n, a2_scale, use_stream));
    PQ.Reset();

    // ── 2.  A⁽⁶⁾ = (A²)³ ────────────────────────────────────────────────
    DeviceMatrix scratch;
    CHECK(scratch.Allocate(n, n));

    DeviceMatrix A6pow;
    CHECK(A6pow.Allocate(n, n));
    CHECK(MatMul(ops, A2.get(), A2.get(), scratch.get(), n));    // (A²)²
    CHECK(MatMul(ops, scratch.get(), A2.get(), A6pow.get(), n)); // (A²)³

    // ── 3.  B⁽ⁱ⁾  and  B̄⁽ⁱ⁾  (i = 0..4) ───────────────────────────────
    DeviceMatrix B[5], Bbar[5];
    for (int i = 0; i < 5; ++i)
    {
        const Real a0 = static_cast<Real>(kExpConstantOrder18[i][0]);
        const Real a1 = static_cast<Real>(kExpConstantOrder18[i][1]);
        const Real a2 = static_cast<Real>(kExpConstantOrder18[i][2]);
        const Real a3 = static_cast<Real>(kExpConstantOrder18[i][3]);
        const Real a4 = static_cast<Real>(kExpConstantOrder18[i][4]);

        // B⁽ⁱ⁾ = a₀ᵢ·I − a₂ᵢ·A² − a₄ᵢ·A⁶
        CHECK(B[i].Allocate(n, n));
        // Build directly in the selected GPU/RAM/file backend. Avoid storing
        // and streaming a dense identity, and skip the exactly-zero A6 term.
        CHECK(ScaleMatrix(A2.get(), B[i].get(), n, make_complex(-a2, 0.0), use_stream));
        if (a0 != 0)
            CHECK(AddDiagonalConstantInPlace(B[i].get(), n, make_complex(a0, 0.0), use_stream));
        if (a4 != 0)
            CHECK(ScaleAddInPlace(B[i].get(), A6pow.get(), n, make_complex(-a4, 0.0), use_stream));

        // B̄⁽ⁱ⁾ = a₁ᵢ·I − a₃ᵢ·A²
        CHECK(Bbar[i].Allocate(n, n));
        CHECK(ScaleMatrix(A2.get(), Bbar[i].get(), n, make_complex(-a3, 0.0), use_stream));
        if (a1 != 0)
            CHECK(AddDiagonalConstantInPlace(Bbar[i].get(), n, make_complex(a1, 0.0), use_stream));
    }
    A6pow.Reset(); // A⁶ consumed into B arrays

    // ── 4.  A⁽⁹⁾  and  Ā⁽⁹⁾ ────────────────────────────────────────────
    //
    //  A⁽⁹⁾ = B⁽⁰⁾·B⁽⁴⁾ − A²·B̄⁽⁰⁾·B̄⁽⁴⁾ + B⁽³⁾
    //  Ā⁽⁹⁾ = B̄⁽⁰⁾·B⁽⁴⁾ + B⁽⁰⁾·B̄⁽⁴⁾ + B̄⁽³⁾
    DeviceMatrix A9, Abar9;
    DeviceMatrix tmp;
    CHECK(tmp.Allocate(n, n));

    // A9 = B⁽⁰⁾·B⁽⁴⁾ + B⁽³⁾ − A² · (B̄⁽⁰⁾·B̄⁽⁴⁾)
    CHECK(A9.Allocate(n, n));
    CHECK(MatMul(ops, B[0].get(), B[4].get(), A9.get(), n));
    CHECK(MatMul(ops, Bbar[0].get(), Bbar[4].get(), tmp.get(), n));
    CHECK(MatMul(ops, A2.get(), tmp.get(), scratch.get(), n));
    CHECK(ScaleAddInPlace(A9.get(), scratch.get(), n, neg_one, use_stream));
    CHECK(AddInPlace(ops, A9.get(), B[3].get(), n));

    // Abar9 = B̄⁽⁰⁾·B⁽⁴⁾ + B⁽⁰⁾·B̄⁽⁴⁾ + B̄⁽³⁾
    CHECK(Abar9.Allocate(n, n));
    CHECK(MatMul(ops, Bbar[0].get(), B[4].get(), Abar9.get(), n));
    CHECK(MatMul(ops, B[0].get(), Bbar[4].get(), tmp.get(), n));
    CHECK(AddInPlace(ops, Abar9.get(), tmp.get(), n));
    CHECK(AddInPlace(ops, Abar9.get(), Bbar[3].get(), n));
    tmp.Reset(); // temporary buffer consumed

    // Free B[0],B[3],B[4],Bbar[0],Bbar[3],Bbar[4] — consumed by A⁽⁹⁾ / Ā⁽⁹⁾
    B[0].Reset();
    B[3].Reset();
    B[4].Reset();
    Bbar[0].Reset();
    Bbar[3].Reset();
    Bbar[4].Reset();

    // ── 5.  C  and  Sinc ────────────────────────────────────────────────
    //
    //  T₁ = B⁽²⁾ + A⁽⁹⁾    (into freed B[0])
    //  T₂ = B̄⁽²⁾ + Ā⁽⁹⁾    (into freed Bbar[0])
    //  T₁A9 = T₁·A⁽⁹⁾      (into freed B[3])
    //  T₂Ā9 = T₂·Ā⁽⁹⁾      (into freed Bbar[3])
    CHECK(B[0].Allocate(n, n));
    CHECK(AddMatrix(ops, B[2].get(), A9.get(), B[0].get(), n));

    CHECK(Bbar[0].Allocate(n, n));
    CHECK(AddMatrix(ops, Bbar[2].get(), Abar9.get(), Bbar[0].get(), n));

    CHECK(B[3].Allocate(n, n));
    CHECK(MatMul(ops, B[0].get(), A9.get(), B[3].get(), n)); // T₁A9

    CHECK(Bbar[3].Allocate(n, n));
    CHECK(MatMul(ops, Bbar[0].get(), Abar9.get(), Bbar[3].get(), n)); // T₂Ā9

    // C = B⁽¹⁾ + T₁A9 − A² · T₂Ā9
    DeviceMatrix C_mat, Sinc_mat;
    CHECK(C_mat.Allocate(n, n));
    CHECK(AddMatrix(ops, B[1].get(), B[3].get(), C_mat.get(), n));
    CHECK(MatMul(ops, A2.get(), Bbar[3].get(), scratch.get(), n));
    CHECK(ScaleAddInPlace(C_mat.get(), scratch.get(), n, neg_one, use_stream));

    // T₂A9 = T₂ · A⁽⁹⁾    (overwrite B[3] — T₁A9 already consumed into C)
    CHECK(MatMul(ops, Bbar[0].get(), A9.get(), B[3].get(), n));

    // T₁Ā9 = T₁ · Ā⁽⁹⁾    (overwrite Bbar[3] — T₂Ā9 already consumed into C)
    CHECK(MatMul(ops, B[0].get(), Abar9.get(), Bbar[3].get(), n));

    // Sinc = B̄⁽¹⁾ + T₂A9 + T₁Ā9
    CHECK(Sinc_mat.Allocate(n, n));
    CHECK(AddMatrix(ops, Bbar[1].get(), B[3].get(), Sinc_mat.get(), n));
    CHECK(AddInPlace(ops, Sinc_mat.get(), Bbar[3].get(), n));

    // Free everything except C_mat, Sinc_mat, scratch
    A2.Reset();
    A9.Reset();
    Abar9.Reset();
    for (int i = 0; i < 5; ++i)
    {
        B[i].Reset();
        Bbar[i].Reset();
    }

    // ── 6.  Repeated squaring (m rounds of double-angle) ─────────────────
    for (int i = 0; i < m; ++i)
    {
        // Sinc ← Sinc·C      (sin(2x)/(2x) = cos(x)·sin(x)/x)
        CHECK(MatMul(ops, Sinc_mat.get(), C_mat.get(), scratch.get(), n));
        Sinc_mat.Swap(scratch);

        // C ← 2·C² − I
        CHECK(MatMul(ops, C_mat.get(), C_mat.get(), scratch.get(), n));
        CHECK(ScaleInPlace(scratch.get(), n, two, use_stream));
        CHECK(AddDiagonalConstantInPlace(scratch.get(), n, neg_one, use_stream));
        C_mat.Swap(scratch);
    }

    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Complex);
    CHECK(StatusFromCuda(
        rcwa_memory::MemcpyAsync(Cos, C_mat.get(), bytes, cudaMemcpyDeviceToDevice, use_stream)));
    CHECK(StatusFromCuda(
        rcwa_memory::MemcpyAsync(Sinc, Sinc_mat.get(), bytes, cudaMemcpyDeviceToDevice, use_stream)));
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

    // ── Phase 0: scaling ──────────────────────────────────────────────────
    // T = exp(A · k₀d)  where  A = [0, −P; Q, 0]  (2n₂ × 2n₂ block matrix).
    // 12th-order optimized Taylor expansion (Bader, Blanes & Casas, 2019)
    // combined with scaling-and-squaring.
    //
    // Phases:
    //   0. Scaling:  m = ⌈log₂ max(‖P‖₁, ‖Q‖₁)·k₀d⌉,  s = 2⁻ᵐ·k₀d
    //   1. Block powers A², A³  (n₂ × n₂ blocks)
    //   2. B⁽ⁱ⁾ = a₀ᵢ·I − a₂ᵢ·A²  (i = 0..3, 4 × 4 blocks)
    //   3. A⁽⁶⁾ blocks from (B⁽³⁾)² + B⁽²⁾
    //   4. T blocks from (B⁽¹⁾ + A⁽⁶⁾)·A⁽⁶⁾ + B⁽⁰⁾
    //   5. Repeated squaring (m rounds)
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

    // ── Phase 1: block powers  A², A³ ─────────────────────────────────────
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

    // ── Phase 2: B⁽ⁱ⁾ = a₀ᵢ·I − a₂ᵢ·A²  (i = 0..3) ──────────────────────
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
        CHECK(ScaleAddInPlace(B_11[i].get(), A2_11.get(), n, c2, use_stream));

        CHECK(B_12[i].Allocate(n, n));
        CHECK(ScaleMatrix(A_12.get(), B_12[i].get(), n, c1, use_stream));
        CHECK(ScaleAddInPlace(B_12[i].get(), A3_12.get(), n, c3, use_stream));

        CHECK(B_21[i].Allocate(n, n));
        CHECK(ScaleMatrix(A_21.get(), B_21[i].get(), n, c1, use_stream));
        CHECK(ScaleAddInPlace(B_21[i].get(), A3_21.get(), n, c3, use_stream));

        CHECK(B_22[i].Allocate(n, n));
        CHECK(ScaleMatrix(I2.get(), B_22[i].get(), n, c0, use_stream));
        CHECK(ScaleAddInPlace(B_22[i].get(), A2_22.get(), n, c2, use_stream));
    }

    // Phase 2 cleanup: free consumed powers and identity
    A2_11.Reset();
    A2_22.Reset();
    A3_12.Reset();
    A3_21.Reset();

    // I2, A_12, A_21 — only needed for Phase 2 B coefficients
    I2.Reset();
    A_12.Reset();
    A_21.Reset();

    // ── Phase 3: A⁽⁶⁾ blocks = (B⁽³⁾)² + B⁽²⁾ ────────────────────────────

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

    // Phase 3 cleanup: free B[2],B[3] — consumed by A⁽⁶⁾ blocks
    for (int i : {2, 3})
    {
        B_11[i].Reset();
        B_12[i].Reset();
        B_21[i].Reset();
        B_22[i].Reset();
    }

    // ── Phase 4: T blocks = (B⁽¹⁾ + A⁽⁶⁾)·A⁽⁶⁾ + B⁽⁰⁾ ───────────────────
    // Precompute intermediate sums in-place on B[1] buffers (consumed after Phase 4)
    CHECK(AddInPlace(ops, B_11[1].get(), A6_11.get(), n));
    CHECK(AddInPlace(ops, B_12[1].get(), A6_12.get(), n));
    CHECK(AddInPlace(ops, B_21[1].get(), A6_21.get(), n));
    CHECK(AddInPlace(ops, B_22[1].get(), A6_22.get(), n));

    DeviceMatrix temp1;
    DeviceMatrix temp2;

    DeviceMatrix T_11;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(MatMul(ops, B_11[1].get(), A6_11.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, B_12[1].get(), A6_21.get(), temp2.get(), n));
        CHECK(T_11.Allocate(n, n));
        CHECK(AddMatrix(ops, B_11[0].get(), temp1.get(), T_11.get(), n));
        CHECK(AddInPlace(ops, T_11.get(), temp2.get(), n));
    }

    DeviceMatrix T_12;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(MatMul(ops, B_11[1].get(), A6_12.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, B_12[1].get(), A6_22.get(), temp2.get(), n));
        CHECK(T_12.Allocate(n, n));
        CHECK(AddMatrix(ops, B_12[0].get(), temp1.get(), T_12.get(), n));
        CHECK(AddInPlace(ops, T_12.get(), temp2.get(), n));
    }

    DeviceMatrix T_21;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(MatMul(ops, B_21[1].get(), A6_11.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, B_22[1].get(), A6_21.get(), temp2.get(), n));
        CHECK(T_21.Allocate(n, n));
        CHECK(AddMatrix(ops, B_21[0].get(), temp1.get(), T_21.get(), n));
        CHECK(AddInPlace(ops, T_21.get(), temp2.get(), n));
    }

    DeviceMatrix T_22;
    {
        CHECK(temp1.Allocate(n, n));
        CHECK(MatMul(ops, B_21[1].get(), A6_12.get(), temp1.get(), n));
        CHECK(temp2.Allocate(n, n));
        CHECK(MatMul(ops, B_22[1].get(), A6_22.get(), temp2.get(), n));
        CHECK(T_22.Allocate(n, n));
        CHECK(AddMatrix(ops, B_22[0].get(), temp1.get(), T_22.get(), n));
        CHECK(AddInPlace(ops, T_22.get(), temp2.get(), n));
    }

    // Phase 4 cleanup: free B[0],B[1], all A⁽⁶⁾ blocks, and temps —
    // none are needed in Phase 5 squaring.
    for (int i : {0, 1})
    {
        B_11[i].Reset();
        B_12[i].Reset();
        B_21[i].Reset();
        B_22[i].Reset();
    }
    A6_11.Reset();
    A6_12.Reset();
    A6_21.Reset();
    A6_22.Reset();
    temp1.Reset();
    temp2.Reset();

    // ── Phase 5: repeated squaring (m rounds) ─────────────────────────────
    // T ← T²  where T is a 2×2 block matrix:  [T₁₁, T₁₂; T₂₁, T₂₂]
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

// ─────────────────────────────────────────────────────────────────────────────
// Common tail: given E, F ∈ ℂⁿˣⁿ, compute the four S-matrix blocks.
//
//   T₂₂ = (E + F)/2,  T₂₁ = (E − F)/2
//   S₁₁ = T₂₂⁻¹,  S₁₂ = −S₁₁·T₂₁,  S₂₁ = S₁₂,  S₂₂ = S₁₁
// ─────────────────────────────────────────────────────────────────────────────
static Status EFToSMatrix(const Complex* E,
                          const Complex* F,
                          int64_t n,
                          SMatrix* S,
                          ComplexMatrixOps& ops,
                          cudaStream_t stream)
{
    const Complex half = make_complex(0.5, 0.0);
    const Complex neg_one = make_complex(-1.0, 0.0);

    DeviceMatrix T22;
    CHECK(T22.Allocate(n, n));
    CHECK(AddMatrix(ops, E, F, T22.get(), n));
    CHECK(ScaleInPlace(T22.get(), n, half, stream));

    DeviceMatrix T21;
    CHECK(T21.Allocate(n, n));
    CHECK(SubMatrix(ops, E, F, T21.get(), n));
    CHECK(ScaleInPlace(T21.get(), n, half, stream));

    DeviceMatrix S11;
    {
        DeviceMatrix I_rhs;
        CHECK(I_rhs.Allocate(n, n));
        CHECK(MakeIdentity(I_rhs.get(), n, stream));
        CHECK(S11.Allocate(n, n));
        CHECK(ops.Solve(T22.get(), I_rhs.get(), S11.get(), n));
    }

    DeviceMatrix S12;
    CHECK(S12.Allocate(n, n));
    CHECK(MatMul(ops, S11.get(), T21.get(), S12.get(), n));
    CHECK(ScaleInPlace(S12.get(), n, neg_one, stream));

    DeviceMatrix S21;
    CHECK(S21.Allocate(n, n));
    CHECK(CopyMatrix(S12.get(), S21.get(), n, stream));

    DeviceMatrix S22;
    CHECK(S22.Allocate(n, n));
    CHECK(CopyMatrix(S11.get(), S22.get(), n, stream));

    S->S_11 = S11.Release();
    S->S_12 = S12.Release();
    S->S_21 = S21.Release();
    S->S_22 = S22.Release();
    return Status::kSuccess;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute S-matrix via the GEMM path.
//
//   1. Compute T = exp(A·k₀d) via Taylor expansion (Order 12 or 18)
//   2. Convert T → S via the V₀ matrix
//
// Memory peak is higher than FREE because ComputeTFromPQ operates
// on 2n × 2n block matrices internally.
// ─────────────────────────────────────────────────────────────────────────────
Status RCWA::ComputeSMatrixGEMM(const Complex* P,
                                const Complex* Q,
                                int64_t n,
                                Real k0d,
                                const VMatrix& V,
                                SMatrix* S,
                                cudaStream_t stream) const
{
    if (!P || !Q || !S || !V.V_11 || !V.V_12 || !V.V_21 || !V.V_22 || n <= 0 || (n % 2) != 0)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;

    S->S_11 = nullptr;
    S->S_12 = nullptr;
    S->S_21 = nullptr;
    S->S_22 = nullptr;

    // ── 1. Compute T from P, Q ───────────────────────────────────────────
    TMatrix T{};
    if (high_order_)
    {
        CHECK(ComputeTFromPQOrder18(P, Q, n, k0d, &T, use_stream));
    }
    else
    {
        CHECK(ComputeTFromPQOrder12(P, Q, n, k0d, &T, use_stream));
    }

    // ── 2. T → S conversion ──────────────────────────────────────────────
    //   E = T₁₁ + i·V₀·T₂₁
    //   F = (i·T₁₂ − V₀·T₂₂)·V₀
    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    const Complex imag = make_complex(0.0, 1.0);

    DeviceMatrix tmp_left;
    DeviceMatrix E;
    DeviceMatrix F;

    CHECK(ApplyV0Left(V, T.T_21, n, &tmp_left, use_stream));
    SAFEFREE(T.T_21);
    CHECK(ScaleInPlace(tmp_left.get(), n, imag, use_stream));

    CHECK(E.Allocate(n, n));
    CHECK(CopyMatrix(T.T_11, E.get(), n, use_stream));
    SAFEFREE(T.T_11);
    CHECK(AddInPlace(ops, E.get(), tmp_left.get(), n));

    CHECK(ApplyV0Left(V, T.T_22, n, &tmp_left, use_stream));
    SAFEFREE(T.T_22);

    CHECK(F.Allocate(n, n));
    CHECK(CopyMatrix(T.T_12, F.get(), n, use_stream));
    SAFEFREE(T.T_12);
    CHECK(ScaleInPlace(F.get(), n, imag, use_stream));
    CHECK(SubInPlace(ops, F.get(), tmp_left.get(), n));
    tmp_left.Reset();

    CHECK(ApplyV0Right(V, F.get(), n, &F, use_stream));

    CHECK(EFToSMatrix(E.get(), F.get(), n, S, ops, use_stream));

    return Status::kSuccess;
}

// ─────────────────────────────────────────────────────────────────────────────
// FREE path: avoids the matrix square root entirely by computing cos(Θ) and
// Θ⁻¹·sin(Θ) directly from P·Q via Taylor expansion (see above).
//
// Both functions are even in Θ = k₀d·√(P·Q), so they depend only on
// Θ² = (k₀d)²·(P·Q).  E and F are then assembled without ever forming G
// or its inverse:
//
//   1.  [Cos, Sinc] = ComputeCosSincFromPQ(P, Q)
//   2.  V₀Q = V₀·Q
//   3.  E   = Cos + i·k₀d·V₀Q·Sinc
//   4.  F   = −(i·k₀d·Sinc + V₀·P⁻¹·Cos)·P·V₀
//   5.  S-blocks from E, F  via EFToSMatrix
// ─────────────────────────────────────────────────────────────────────────────
Status RCWA::ComputeSMatrixFREE(const Complex* P,
                                const Complex* Q,
                                const Complex* P_inv,
                                int64_t n,
                                Real k0d,
                                const VMatrix& V,
                                SMatrix* S,
                                cudaStream_t stream) const
{
    if (!P || !Q || !P_inv || !S || n <= 0 || (n % 2) != 0)
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
    const Complex k0d_imag = make_complex(0.0, static_cast<Real>(k0d));
    const Complex neg_one = make_complex(-1.0, 0.0);

    // ── 1. Cos, Sinc = cos(Θ), Θ⁻¹·sin(Θ)  (no sqrt) ───────────────────
    DeviceMatrix Cos;
    CHECK(Cos.Allocate(n, n));
    DeviceMatrix Sinc;
    CHECK(Sinc.Allocate(n, n));
    if (high_order_)
    {
        CHECK(ComputeCosSincFromPQOrder18(P, Q, n, k0d, Cos.get(), Sinc.get(), use_stream));
    }
    else
    {
        CHECK(ComputeCosSincFromPQOrder12(P, Q, n, k0d, Cos.get(), Sinc.get(), use_stream));
    }

    // ── 2. V0Q = V₀ · Q ─────────────────────────────────────────────────
    DeviceMatrix V0Q;
    CHECK(ApplyV0Left(V, Q, n, &V0Q, use_stream));

    // ── 3. E = Cos + i·k₀d·V₀Q·Sinc ────────────────────────────────────
    DeviceMatrix E;
    CHECK(E.Allocate(n, n));
    {
        DeviceMatrix V0Q_Sinc;
        CHECK(V0Q_Sinc.Allocate(n, n));
        CHECK(MatMul(ops, V0Q.get(), Sinc.get(), V0Q_Sinc.get(), n));
        CHECK(ScaleInPlace(V0Q_Sinc.get(), n, k0d_imag, use_stream));
        CHECK(AddMatrix(ops, Cos.get(), V0Q_Sinc.get(), E.get(), n));
    }

    // ── 4. F = −(i·k₀d·Sinc + V₀·P⁻¹·Cos)·P·V₀ ─────────────────────
    DeviceMatrix V0Pinv;
    CHECK(ApplyV0Left(V, P_inv, n, &V0Pinv, use_stream));

    DeviceMatrix F_left;
    CHECK(F_left.Allocate(n, n));
    CHECK(ScaleMatrix(Sinc.get(), F_left.get(), n, k0d_imag, use_stream));
    Sinc.Reset();

    DeviceMatrix V0PinvCos;
    CHECK(V0PinvCos.Allocate(n, n));
    CHECK(MatMul(ops, V0Pinv.get(), Cos.get(), V0PinvCos.get(), n));
    V0Pinv.Reset();
    Cos.Reset();
    CHECK(AddInPlace(ops, F_left.get(), V0PinvCos.get(), n));
    V0PinvCos.Reset();
    V0Q.Reset();

    DeviceMatrix F_pre;
    CHECK(F_pre.Allocate(n, n));
    CHECK(MatMul(ops, F_left.get(), P, F_pre.get(), n));
    F_left.Reset();

    DeviceMatrix F;
    CHECK(F.Allocate(n, n));
    CHECK(ApplyV0Right(V, F_pre.get(), n, &F, use_stream));
    F_pre.Reset();
    CHECK(ScaleInPlace(F.get(), n, neg_one, use_stream));

    // ── 5. S-blocks from E, F ──────────────────────────────────────────
    CHECK(EFToSMatrix(E.get(), F.get(), n, S, ops, use_stream));
    return Status::kSuccess;
}

Status RCWA::AddLayer(Real thickness, Real eps)
{
    return AddLayer(thickness, make_complex(eps, 0.0));
}

Status RCWA::AddLayer(Real thickness, Complex eps)
{
    if (IsAutoRecording())
    {
        RecordedLayer layer;
        layer.type = RecordedLayerType::kHomogeneous;
        layer.thickness = thickness;
        layer.eps = eps;
        recorded_layers_.push_back(std::move(layer));
        return Status::kSuccess;
    }

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
    rcwa_memory::Configure(order_x_,order_y_);

    if (IsAutoRecording())
    {
        RecordedLayer layer;
        layer.type = RecordedLayerType::kHomogeneous;
        layer.thickness = thickness;
        layer.eps = eps;
        recorded_layers_.push_back(std::move(layer));
        return Status::kSuccess;
    }

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
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(S11.get(), 0, matrix_bytes, stream_)));
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(S12.get(), 0, matrix_bytes, stream_)));
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(S21.get(), 0, matrix_bytes, stream_)));
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(S22.get(), 0, matrix_bytes, stream_)));

    const Real k0d = omega_ * thickness;
    const int blocks = static_cast<int>((n + CUDA_THREADS - 1) / CUDA_THREADS);
    if (rcwa_memory::IsHost(S11.get()) || rcwa_memory::IsHost(S12.get()) ||
        rcwa_memory::IsHost(S21.get()) || rcwa_memory::IsHost(S22.get())) {
        DeviceVector p11,p12,p21,p22;
        CHECK(p11.Allocate(4*n)); CHECK(p12.Allocate(4*n)); CHECK(p21.Allocate(4*n)); CHECK(p22.Allocate(4*n));
    BuildHomogeneousSBlocksKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(
        kx_diag_, ky_diag_, eps, k0d, p11.get(), p12.get(), p21.get(), p22.get(), n, true);
    CHECK(StatusFromCuda(cudaGetLastError()));
        CHECK(UnpackSBlocks(p11.get(),S11.get(),n,stream_)); CHECK(UnpackSBlocks(p12.get(),S12.get(),n,stream_));
        CHECK(UnpackSBlocks(p21.get(),S21.get(),n,stream_)); CHECK(UnpackSBlocks(p22.get(),S22.get(),n,stream_));
    } else {
    BuildHomogeneousSBlocksKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(
        kx_diag_, ky_diag_, eps, k0d, S11.get(), S12.get(), S21.get(), S22.get(), n);
    CHECK(StatusFromCuda(cudaGetLastError()));
    }

    SMatrix S{};
    S.S_11 = S11.Release();
    S.S_12 = S12.Release();
    S.S_21 = S21.Release();
    S.S_22 = S22.Release();
    return CommitLayerSMatrix(&S, n2);
}

Status RCWA::AddLayer(Real thickness, const Complex* eps_conv)
{
    if (IsAutoRecording())
    {
        return Status::kInvalidValue;
    }
    return AddLayerFromConvolution(thickness, eps_conv, LayerFactorization::kLaurent);
}

Status RCWA::AddLayerFromConvolution(Real thickness,
                                     const Complex* eps_conv,
                                     LayerFactorization factorization,
                                     const Complex* eps_inv_conv,
                                     const Complex* nxx_conv,
                                     const Complex* nxy_conv,
                                     const Complex* nyy_conv)
{
    rcwa_memory::Configure(order_x_,order_y_);

    if (!eps_conv || !kvectors_ready_)
    {
        return Status::kInvalidValue;
    }
    if (factorization != LayerFactorization::kLaurent &&
        factorization != LayerFactorization::kLalanne)
    {
        return Status::kInvalidValue;
    }
    if (factorization == LayerFactorization::kLalanne &&
        (!eps_inv_conv || !nxx_conv || !nxy_conv || !nyy_conv))
    {
        return Status::kInvalidValue;
    }

    const int64_t n = order_N_;
    const int64_t n2 = 2 * n;

    DeviceMatrix P;
    DeviceMatrix Q;
    CHECK(P.Allocate(n2, n2));
    CHECK(Q.Allocate(n2, n2));
    DeviceMatrix P_inv;
    Complex* P_inv_ptr = nullptr;
    if (expm_method_ == 'F')
    {
        CHECK(P_inv.Allocate(n2, n2));
        P_inv_ptr = P_inv.get();
    }
    CHECK(BuildPQ(eps_conv, n, P.get(), Q.get(), P_inv_ptr));
    if (factorization == LayerFactorization::kLalanne)
    {
        CHECK(AddLalanneDeltaQ(
            eps_conv, eps_inv_conv, nxx_conv, nxy_conv, nyy_conv, n, Q.get()));
    }

    return AddLayerFromPQ(thickness, P.get(), Q.get(), P_inv_ptr, n2);
}

Status RCWA::AddLayer(Real thickness,
                      const Complex* material,
                      int64_t nx,
                      int64_t ny,
                      LayerFactorization factorization)
{
    if (!material)
    {
        return Status::kInvalidValue;
    }
    if (IsAutoRecording())
    {
        if (nx <= 0 || ny <= 0)
        {
            return Status::kInvalidValue;
        }
        RecordedLayer layer;
        layer.type = RecordedLayerType::kMaterial;
        layer.thickness = thickness;
        layer.nx = nx;
        layer.ny = ny;
        layer.factorization = factorization;
        layer.material.resize(static_cast<size_t>(nx) * static_cast<size_t>(ny));
        CHECK(StatusFromCuda(cudaMemcpy(layer.material.data(),
                                        material,
                                        layer.material.size() * sizeof(Complex),
                                        cudaMemcpyDeviceToHost)));
        recorded_layers_.push_back(std::move(layer));
        return Status::kSuccess;
    }
    if (factorization == LayerFactorization::kLaurent)
    {
        Complex* raw_eps_conv = nullptr;
        CHECK(MaterialConv(material, nx, ny, &raw_eps_conv, stream_));
        DeviceMatrix eps_conv;
        eps_conv.Reset(raw_eps_conv, static_cast<size_t>(order_N_) * static_cast<size_t>(order_N_));
        return AddLayer(thickness, eps_conv.get());
    }
    else if (factorization == LayerFactorization::kLalanne)
    {
        return AddLayerLalanne(thickness, material, nx, ny);
    }
    return Status::kInvalidValue;
}

Status RCWA::AddLayerFromPQ(Real thickness,
                            const Complex* P,
                            const Complex* Q,
                            const Complex* P_inv,
                            int64_t n2)
{
    if (!P || !Q || n2 <= 0 || !kvectors_ready_)
    {
        return Status::kInvalidValue;
    }

    int n_repeated = 0;
    Real d_block = thickness;
    Real thickness_thres = 0.0;
    if (order_x_ > 0)
    {
        thickness_thres = Lx_ / static_cast<Real>(order_x_);
    }
    if (order_y_ > 0)
    {
        const Real dy = Ly_ / static_cast<Real>(order_y_);
        thickness_thres = thickness_thres > 0.0 ? std::min(thickness_thres, dy) : dy;
    }
#if USE_DOUBLE_PRECISION
    thickness_thres *= static_cast<Real>(2.0);
#endif
    if (thickness_thres > 0.0 && thickness > thickness_thres)
    {
        const Real ratio = thickness / thickness_thres;
        n_repeated = static_cast<int>(std::ceil(std::log2(ratio)));
        if (n_repeated > 0)
        {
            const Real denom = static_cast<Real>(1ULL << n_repeated);
            d_block = thickness / denom;
        }
    }

    const Real k0d = omega_ * d_block;
    SMatrix S{};
    if (expm_method_ == 'F')
    {
        CHECK(ComputeSMatrixFREE(P, Q, P_inv, n2, k0d, V0_, &S, stream_));
    }
    else
    {
        CHECK(ComputeSMatrixGEMM(P, Q, n2, k0d, V0_, &S, stream_));
    }

    if (n_repeated > 0)
    {
        CHECK(RepeatSMatrix(&S, n_repeated, n2));
    }

    return CommitLayerSMatrix(&S, n2);
}

Status RCWA::BuildLalanneNormalProducts(const Complex* material,
                                        int64_t nx,
                                        int64_t ny,
                                        Real eta,
                                        Complex* nxx,
                                        Complex* nxy,
                                        Complex* nyy) const
{
    if (!material || !nxx || !nxy || !nyy || nx <= 0 || ny <= 0 ||
        eta < static_cast<Real>(0.0))
    {
        return Status::kInvalidValue;
    }

    const int64_t map_size = nx * ny;
    const int blocks = static_cast<int>((map_size + CUDA_THREADS - 1) / CUDA_THREADS);
    DeviceMatrix boundary;
    CHECK(boundary.Allocate(nx, ny, true));
    CachedBoundaryKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(material, nx, ny, Lx_, Ly_, eta, boundary.get());
    CHECK(StatusFromCuda(cudaGetLastError()));
    std::vector<Complex> host_boundary(static_cast<size_t>(map_size));
    CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(host_boundary.data(), boundary.get(),
        host_boundary.size() * sizeof(Complex), cudaMemcpyDeviceToHost, stream_)));
    CHECK(StatusFromCuda(cudaStreamSynchronize(stream_)));
    std::vector<int64_t> indices;
    for (int64_t i = 0; i < map_size; ++i)
        if (ComplexReal(host_boundary[i]) != 0 || ComplexImag(host_boundary[i]) != 0) indices.push_back(i);
    int64_t* device_indices = nullptr;
    if (!indices.empty()) {
        CHECK(StatusFromCuda(cudaMalloc(&device_indices, indices.size() * sizeof(int64_t))));
        const cudaError_t copied = rcwa_memory::MemcpyAsync(device_indices, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream_);
        if (copied != cudaSuccess) { cudaFree(device_indices); return StatusFromCuda(copied); }
    }
    CachedNormalProductsKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(boundary.get(), device_indices,
        static_cast<int64_t>(indices.size()), nx, ny, Lx_, Ly_, eta, nxx, nxy, nyy);
    const cudaError_t launched = cudaGetLastError();
    const cudaError_t completed = cudaStreamSynchronize(stream_);
    if (device_indices) cudaFree(device_indices);
    CHECK(StatusFromCuda(launched));
    return StatusFromCuda(completed);
}

Status RCWA::GenerateNormalVectorField(const Complex* material,
                                       int64_t nx,
                                       int64_t ny,
                                       Real eta,
                                       Complex** nx_field,
                                       Complex** ny_field,
                                       cudaStream_t stream) const
{
    if (!material || !nx_field || !ny_field || nx <= 0 || ny <= 0 ||
        eta < static_cast<Real>(0.0))
    {
        return Status::kInvalidValue;
    }

    *nx_field = nullptr;
    *ny_field = nullptr;
    const cudaStream_t use_stream = stream ? stream : stream_;
    const int64_t map_size = nx * ny;
    const size_t bytes = static_cast<size_t>(map_size) * sizeof(Complex);

    Complex* raw_nx = nullptr;
    Complex* raw_ny = nullptr;
    Status status = StatusFromCuda(cudaMalloc(&raw_nx, bytes));
    if (status != Status::kSuccess)
    {
        return status;
    }
    status = StatusFromCuda(cudaMalloc(&raw_ny, bytes));
    if (status != Status::kSuccess)
    {
        SAFEFREE(raw_nx);
        return status;
    }

    const int blocks = static_cast<int>((map_size + CUDA_THREADS - 1) / CUDA_THREADS);
    BuildNormalFieldKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(
        material, nx, ny, Lx_, Ly_, eta, raw_nx, raw_ny);
    status = StatusFromCuda(cudaGetLastError());
    if (status != Status::kSuccess)
    {
        SAFEFREE(raw_nx);
        SAFEFREE(raw_ny);
        return status;
    }

    *nx_field = raw_nx;
    *ny_field = raw_ny;
    return Status::kSuccess;
}

Status RCWA::AddLayerLalanne(Real thickness,
                             const Complex* material,
                             int64_t nx,
                             int64_t ny)
{
    if (!material || !kvectors_ready_)
    {
        return Status::kInvalidValue;
    }
    if (nx < 4 * static_cast<int64_t>(order_x_) + 1 ||
        ny < 4 * static_cast<int64_t>(order_y_) + 1)
    {
        return Status::kInvalidValue;
    }

    const int64_t n = order_N_;
    const int64_t map_size = nx * ny;

    Complex* raw_eps_conv = nullptr;
    CHECK(MaterialConv(material, nx, ny, &raw_eps_conv, stream_));
    DeviceMatrix eps_conv;
    eps_conv.Reset(raw_eps_conv, static_cast<size_t>(n) * static_cast<size_t>(n));

    DeviceMatrix inv_map;
    DeviceMatrix nxx_map;
    DeviceMatrix nxy_map;
    DeviceMatrix nyy_map;
    CHECK(inv_map.Allocate(nx, ny, true));
    CHECK(nxx_map.Allocate(nx, ny, true));
    CHECK(nxy_map.Allocate(nx, ny, true));
    CHECK(nyy_map.Allocate(nx, ny, true));

    CHECK(ElementwiseInverse(material, inv_map.get(), map_size, stream_));
    CHECK(BuildLalanneNormalProducts(material,
                                     nx,
                                     ny,
                                     1e-6,
                                     nxx_map.get(),
                                     nxy_map.get(),
                                     nyy_map.get()));

    Complex* raw_inv_eps_conv = nullptr;
    Complex* raw_nxx_conv = nullptr;
    Complex* raw_nxy_conv = nullptr;
    Complex* raw_nyy_conv = nullptr;
    CHECK(MaterialConv(inv_map.get(), nx, ny, &raw_inv_eps_conv, stream_));
    CHECK(MaterialConv(nxx_map.get(), nx, ny, &raw_nxx_conv, stream_));
    CHECK(MaterialConv(nxy_map.get(), nx, ny, &raw_nxy_conv, stream_));
    CHECK(MaterialConv(nyy_map.get(), nx, ny, &raw_nyy_conv, stream_));

    DeviceMatrix inv_eps_conv;
    DeviceMatrix nxx_conv;
    DeviceMatrix nxy_conv;
    DeviceMatrix nyy_conv;
    const size_t conv_elements = static_cast<size_t>(n) * static_cast<size_t>(n);
    inv_eps_conv.Reset(raw_inv_eps_conv, conv_elements);
    nxx_conv.Reset(raw_nxx_conv, conv_elements);
    nxy_conv.Reset(raw_nxy_conv, conv_elements);
    nyy_conv.Reset(raw_nyy_conv, conv_elements);

    return AddLayerFromConvolution(thickness,
                                   eps_conv.get(),
                                   LayerFactorization::kLalanne,
                                   inv_eps_conv.get(),
                                   nxx_conv.get(),
                                   nxy_conv.get(),
                                   nyy_conv.get());
}

Status RCWA::MaterialConv(const Complex* material,
                          int64_t nx,
                          int64_t ny,
                          Complex** eps_conv,
                          cudaStream_t stream) const
{
    rcwa_memory::Configure(order_x_,order_y_);

    if (!material || !eps_conv)
    {
        return Status::kInvalidValue;
    }
    if (nx < 4 * static_cast<int64_t>(order_x_) + 1 ||
        ny < 4 * static_cast<int64_t>(order_y_) + 1)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;
    *eps_conv = nullptr;

    const int64_t fft_size = nx * ny;
    const size_t fft_bytes = static_cast<size_t>(fft_size) * sizeof(Complex);
    DeviceMatrix fft_buf;
    CHECK(fft_buf.Allocate(nx, ny, true));
    CHECK(StatusFromCuda(
        rcwa_memory::MemcpyAsync(fft_buf.get(), material, fft_bytes, cudaMemcpyDeviceToDevice, use_stream)));

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
    if(rcwa_memory::IsHost(conv.get())) {
        std::vector<Complex> fft(static_cast<size_t>(nx)*ny),block(static_cast<size_t>(order_N_)*64);
        CHECK(rcwa_memory::Read(fft_buf.get(),fft.data(),fft.size(),use_stream));
        for(int64_t j=0;j<order_N_;j+=64) {
            auto nc=std::min<int64_t>(64,order_N_-j);
            for(int64_t c=0;c<nc;++c)for(int64_t r=0;r<order_N_;++r) {
                auto fx=(r/oy_len-(j+c)/oy_len)%nx,fy=(r%oy_len-(j+c)%oy_len)%ny;
                if(fx<0)fx+=nx; if(fy<0)fy+=ny;
                block[c*order_N_+r]=fft[fx*ny+fy];
            }
            CHECK(rcwa_memory::WriteRect(conv.get(),order_N_,0,j,order_N_,nc,block.data(),use_stream));
        }
    } else {
    BuildConvMatrixKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(
        fft_buf.get(), nx, ny, oy_len, conv.get(), order_N_);
    CHECK(StatusFromCuda(cudaGetLastError()));
    }

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

    CHECK(ops.Solve(T.get(), rhs.get(), rhs.get(), n, rhs_cols));

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
    rcwa_memory::Configure(order_x_,order_y_);

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

    const bool stored = rcwa_memory::IsHost(stack_smatrix_.S_11) || rcwa_memory::IsHost(stack_smatrix_.S_12) ||
                        rcwa_memory::IsHost(stack_smatrix_.S_21) || rcwa_memory::IsHost(stack_smatrix_.S_22);
    if (!S_in && !(stored && ComplexReal(eps_in_)==1 && ComplexImag(eps_in_)==0))
    {
        CHECK(BuildPortSMatrix(eps_in_, true, &auto_in, use_stream));
        use_S_in = &auto_in;
    }
    if (!S_out && !(stored && ComplexReal(eps_out_)==1 && ComplexImag(eps_out_)==0))
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
        CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(
            S12.get(), 0, static_cast<size_t>(n) * n * sizeof(Complex), use_stream)));
        CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(
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
    if (auto_order_enabled_ && !auto_order_resolving_)
    {
        CHECK(ResolveAutoOrder());
    }
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
    if (!source_ei_ || !kx_diag_ || !ky_diag_)
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
    CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(dx.get(),
                                         x_axis,
                                         static_cast<size_t>(x_count) * sizeof(Real),
                                         cudaMemcpyHostToDevice,
                                         use_stream)));
    CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(dy.get(),
                                         y_axis,
                                         static_cast<size_t>(y_count) * sizeof(Real),
                                         cudaMemcpyHostToDevice,
                                         use_stream)));

    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    // Diffraction amplitudes:  E_xy = S·E_i (n₂ × 1).
    // Ez is recovered from the divergence condition:
    //   Ez = (Kx·Ex + Ky·Ey) / Kz.
    //
    // Field reconstruction:
    //   field(x,y) = Σ_{mn} E_{mn} · exp(i·ω·(Kx·x + Ky·y))
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
    CHECK(ExMat.Allocate(x_count, y_count, true));
    CHECK(EyMat.Allocate(x_count, y_count, true));
    CHECK(EzMat.Allocate(x_count, y_count, true));

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

Status RCWA::GetKxKyKz(std::vector<Real>* kx,
                       std::vector<Real>* ky,
                       std::vector<Real>* kz,
                       cudaStream_t stream) const
{
    if (!kx || !ky || !kz || !kx_diag_ || !ky_diag_)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;
    const int64_t n = order_N_;
    std::vector<Complex> kx_complex(static_cast<size_t>(n));
    std::vector<Complex> ky_complex(static_cast<size_t>(n));
    kx->assign(static_cast<size_t>(n), 0.0);
    ky->assign(static_cast<size_t>(n), 0.0);
    kz->assign(static_cast<size_t>(n), 0.0);

    CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(kx_complex.data(),
                                         kx_diag_,
                                         static_cast<size_t>(n) * sizeof(Complex),
                                         cudaMemcpyDeviceToHost,
                                         use_stream)));
    CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(ky_complex.data(),
                                         ky_diag_,
                                         static_cast<size_t>(n) * sizeof(Complex),
                                         cudaMemcpyDeviceToHost,
                                         use_stream)));
    CHECK(StatusFromCuda(cudaStreamSynchronize(use_stream)));

    const Complex eps = angle_layer_input_ ? eps_in_ : eps_out_;
    const Real eps_real = ComplexReal(eps);
    for (int64_t i = 0; i < n; ++i)
    {
        const Real kxv = ComplexReal(kx_complex[static_cast<size_t>(i)]);
        const Real kyv = ComplexReal(ky_complex[static_cast<size_t>(i)]);
        (*kx)[static_cast<size_t>(i)] = kxv;
        (*ky)[static_cast<size_t>(i)] = kyv;
        const Real kz_sq = std::max(static_cast<Real>(0.0), eps_real - kxv * kxv - kyv * kyv);
        (*kz)[static_cast<size_t>(i)] = std::sqrt(static_cast<double>(kz_sq));
    }
    return Status::kSuccess;
}

Status RCWA::EnergyBalance(double* R,
                           double* T,
                           double* A,
                           cudaStream_t stream) const
{
    if (!R || !T || !A || !kx_diag_ || !ky_diag_ || !HasSMatrix(global_smatrix_))
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : stream_;
    const int64_t n = order_N_;
    const int64_t n2 = 2 * n;
    const int64_t oy_len = static_cast<int64_t>(2 * order_y_ + 1);
    const int64_t source_order = static_cast<int64_t>(order_x_) * oy_len + order_y_;
    // SourcePlanewave's x component is stored in the first n entries and y
    // in the second n entries. Evaluate both source polarizations separately
    // and add their power contributions, as before for a Jones vector.
    const int64_t source_col_x = source_order;
    const int64_t source_col_y = source_order + n;

    if(rcwa_memory::IsHost(global_smatrix_.S_11) || rcwa_memory::IsHost(global_smatrix_.S_21)) {
        DeviceVector tcol,rcol; CHECK(tcol.Allocate(n2)); CHECK(rcol.Allocate(n2));
        DeviceRealBuffer sums; CHECK(sums.Allocate(2));
        std::vector<Complex> column(n2); double rt[2]={0,0};
        const int blocks=static_cast<int>((n+CUDA_THREADS-1)/CUDA_THREADS);
        for(auto col:{source_col_x,source_col_y}) {
            CHECK(rcwa_memory::Read(global_smatrix_.S_11+col*n2,column.data(),n2,use_stream));
            CHECK(rcwa_memory::Write(tcol.get(),column.data(),n2,use_stream));
            CHECK(rcwa_memory::Read(global_smatrix_.S_21+col*n2,column.data(),n2,use_stream));
            CHECK(rcwa_memory::Write(rcol.get(),column.data(),n2,use_stream));
            CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(sums.get(),0,2*sizeof(Real),use_stream)));
            EnergyBalanceKernel<<<blocks,CUDA_THREADS,0,use_stream>>>(kx_diag_,ky_diag_,tcol.get(),rcol.get(),
                eps_in_,eps_out_,n,n2,source_order,0,sums.get());
            CHECK(StatusFromCuda(cudaGetLastError()));
            Real h[2]; CHECK(StatusFromCuda(cudaMemcpyAsync(h,sums.get(),sizeof(h),cudaMemcpyDeviceToHost,use_stream)));
            CHECK(StatusFromCuda(cudaStreamSynchronize(use_stream))); rt[0]+=h[0]; rt[1]+=h[1];
        }
        *R=rt[0]*0.5; *T=rt[1]*0.5; *A=1-*R-*T; return Status::kSuccess;
    }

    DeviceRealBuffer accum;
    CHECK(accum.Allocate(2));
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(accum.get(), 0, 2 * sizeof(Real), use_stream)));

    const int blocks = static_cast<int>((n + CUDA_THREADS - 1) / CUDA_THREADS);
    // The energy kernel is launched once for each orthogonal source component.
    EnergyBalanceKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(kx_diag_,
                                                                 ky_diag_,
                                                                 global_smatrix_.S_11,
                                                                 global_smatrix_.S_21,
                                                                 eps_in_,
                                                                 eps_out_,
                                                                 n,
                                                                 n2,
                                                                 source_order,
                                                                 source_col_x,
                                                                 accum.get());
    CHECK(StatusFromCuda(cudaGetLastError()));
    // Reset accumulators and add the y-polarized source contribution; averaging
    // is applied below because the public energy API represents unpolarized
    // illumination.
    CHECK(StatusFromCuda(cudaStreamSynchronize(use_stream)));
    DeviceRealBuffer y_accum;
    CHECK(y_accum.Allocate(2));
    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(y_accum.get(), 0, 2 * sizeof(Real), use_stream)));
    EnergyBalanceKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(kx_diag_,
                                                                 ky_diag_,
                                                                 global_smatrix_.S_11,
                                                                 global_smatrix_.S_21,
                                                                 eps_in_,
                                                                 eps_out_,
                                                                 n,
                                                                 n2,
                                                                 source_order,
                                                                 source_col_y,
                                                                 y_accum.get());
    CHECK(StatusFromCuda(cudaGetLastError()));

    Real host_accum[2] = {0.0, 0.0};
    CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(
        host_accum, accum.get(), sizeof(host_accum), cudaMemcpyDeviceToHost, use_stream)));
    CHECK(StatusFromCuda(cudaStreamSynchronize(use_stream)));

    Real host_y[2] = {0.0, 0.0};
    CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(
        host_y, y_accum.get(), sizeof(host_y), cudaMemcpyDeviceToHost, use_stream)));
    CHECK(StatusFromCuda(cudaStreamSynchronize(use_stream)));
    *R = 0.5 * (static_cast<double>(host_accum[0]) + static_cast<double>(host_y[0]));
    *T = 0.5 * (static_cast<double>(host_accum[1]) + static_cast<double>(host_y[1]));
    *A = 1.0 - *R - *T;
    return Status::kSuccess;
}

void RCWA::FreeDeviceTensor(Complex* ptr) const { SAFEFREE(ptr); }

Status RCWA::BuildPQ(const Complex* eps_conv, int64_t n, Complex* P, Complex* Q, Complex* P_inv)
{
    if (!eps_conv || !P || !Q || n <= 0)
    {
        return Status::kInvalidValue;
    }

    ComplexMatrixOps ops(stream_);
    CHECK(ops.SetStream(stream_));

    // P and Q are 2×2 block matrices with diagonal Kx, Ky:
    //   P = [  Kx·E⁻¹·Ky,   Ky·E⁻¹·Ky − I;
    //          I − Kx·E⁻¹·Kx,  −Ky·E⁻¹·Kx  ]
    //   Q = [  −Kx·Ky,   E − Ky²;
    //          Kx² − E,   Kx·Ky  ]
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
        CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(
            EiKx.get(), combined.get(), half_bytes, cudaMemcpyDeviceToDevice, stream_)));
        CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(EiKy.get(),
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

    CHECK(StatusFromCuda(rcwa_memory::MemsetAsync(
        Q, 0, static_cast<size_t>(2 * n) * static_cast<size_t>(2 * n) * sizeof(Complex), stream_)));

    CHECK(CopyBlockDirect(KxEiKy.get(), n, P, 2 * n, 0, 0, stream_));
    CHECK(CopyBlockDirect(P21.get(), n, P, 2 * n, 0, n, stream_));
    CHECK(CopyBlockDirect(P12.get(), n, P, 2 * n, n, 0, stream_));
    CHECK(CopyBlockNegated(KyEiKx.get(), n, P, 2 * n, n, n, stream_));

    if (P_inv)
    {
        DeviceMatrix H;
        CHECK(H.Allocate(n, n));
        CHECK(CopyMatrix(eps_conv, H.get(), n, stream_));
        CHECK(AddDiagonalInPlace(H.get(), KxKxDiag.get(), n, make_complex(-1.0, 0.0), stream_));
        CHECK(AddDiagonalInPlace(H.get(), KyKyDiag.get(), n, make_complex(-1.0, 0.0), stream_));

        DeviceMatrix HinvK;
        CHECK(HinvK.Allocate(n, 2 * n));
        CHECK(CopyBlockDirect(KxMat.get(), n, HinvK.get(), n, 0, 0, stream_));
        CHECK(CopyBlockDirect(KyMat.get(), n, HinvK.get(), n, 0, n, stream_));
        CHECK(ops.Solve(H.get(), HinvK.get(), HinvK.get(), n, 2 * n));
        H.Reset();

        DeviceMatrix HinvKx;
        DeviceMatrix HinvKy;
        CHECK(HinvKx.Allocate(n, n));
        CHECK(HinvKy.Allocate(n, n));
        const size_t matrix_bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Complex);
        CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(
            HinvKx.get(), HinvK.get(), matrix_bytes, cudaMemcpyDeviceToDevice, stream_)));
        CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(HinvKy.get(),
                                             HinvK.get() + static_cast<size_t>(n) * n,
                                             matrix_bytes,
                                             cudaMemcpyDeviceToDevice,
                                             stream_)));

        DeviceMatrix block;
        CHECK(block.Allocate(n, n));

        CHECK(ops.DiagLeftMatMul(ky_diag_, HinvKx.get(), block.get(), n));
        CHECK(CopyBlockNegated(block.get(), n, P_inv, 2 * n, 0, 0, stream_));

        CHECK(ops.DiagLeftMatMul(ky_diag_, HinvKy.get(), block.get(), n));
        CHECK(ScaleInPlace(block.get(), n, make_complex(-1.0, 0.0), stream_));
        CHECK(AddDiagonalConstantInPlace(block.get(), n, make_complex(-1.0, 0.0), stream_));
        CHECK(CopyBlockDirect(block.get(), n, P_inv, 2 * n, 0, n, stream_));

        CHECK(ops.DiagLeftMatMul(kx_diag_, HinvKx.get(), block.get(), n));
        CHECK(AddDiagonalConstantInPlace(block.get(), n, make_complex(1.0, 0.0), stream_));
        CHECK(CopyBlockDirect(block.get(), n, P_inv, 2 * n, n, 0, stream_));

        CHECK(ops.DiagLeftMatMul(kx_diag_, HinvKy.get(), block.get(), n));
        CHECK(CopyBlockDirect(block.get(), n, P_inv, 2 * n, n, n, stream_));
    }

    CHECK(CopyDiagonalBlock(KxKyDiag.get(), n, Q, 2 * n, 0, 0, make_complex(-1.0, 0.0), stream_));
    CHECK(CopyBlockDirect(Q21.get(), n, Q, 2 * n, 0, n, stream_));
    CHECK(CopyBlockDirect(Q12.get(), n, Q, 2 * n, n, 0, stream_));
    CHECK(CopyDiagonalBlock(KxKyDiag.get(), n, Q, 2 * n, n, n, make_complex(1.0, 0.0), stream_));

    return Status::kSuccess;
}

Status RCWA::AddLalanneDeltaQ(const Complex* eps_conv,
                              const Complex* inv_eps_conv,
                              const Complex* nxx_conv,
                              const Complex* nxy_conv,
                              const Complex* nyy_conv,
                              int64_t n,
                              Complex* Q)
{
    if (!eps_conv || !inv_eps_conv || !nxx_conv || !nxy_conv || !nyy_conv || !Q || n <= 0)
    {
        return Status::kInvalidValue;
    }

    ComplexMatrixOps ops(stream_);
    CHECK(ops.SetStream(stream_));

    DeviceMatrix a_inv;
    CHECK(a_inv.Allocate(n, n));
    CHECK(MakeIdentity(a_inv.get(), n, stream_));
    CHECK(ops.Solve(inv_eps_conv, a_inv.get(), a_inv.get(), n));

    DeviceMatrix d_eps;
    CHECK(d_eps.Allocate(n, n));
    CHECK(SubMatrix(ops, eps_conv, a_inv.get(), d_eps.get(), n));

    DeviceMatrix dnxx;
    DeviceMatrix dnxy;
    DeviceMatrix dnyy;
    CHECK(dnxx.Allocate(n, n));
    CHECK(dnxy.Allocate(n, n));
    CHECK(dnyy.Allocate(n, n));
    CHECK(MatMul(ops, d_eps.get(), nxx_conv, dnxx.get(), n));
    CHECK(MatMul(ops, d_eps.get(), nxy_conv, dnxy.get(), n));
    // Nxx + Nyy = I, so D*Nyy = D - D*Nxx.
    CHECK(SubMatrix(ops, d_eps.get(), dnxx.get(), dnyy.get(), n));

    const Complex one = make_complex(1.0, 0.0);
    const Complex neg_one = make_complex(-1.0, 0.0);
    CHECK(ScaleAddBlockInPlace(Q, 2 * n, dnxy.get(), n, 0, 0, one, stream_));
    CHECK(ScaleAddBlockInPlace(Q, 2 * n, dnyy.get(), n, 0, n, one, stream_));
    CHECK(ScaleAddBlockInPlace(Q, 2 * n, dnxx.get(), n, n, 0, neg_one, stream_));
    CHECK(ScaleAddBlockInPlace(Q, 2 * n, dnxy.get(), n, n, n, neg_one, stream_));

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
        CHECK(MatMul(ops, S->S_12, S->S_11, rhs.get(), n));
        CHECK(CopyBlockDirect(S->S_11, n, rhs.get(), n, 0, n, stream_));
        CHECK(ops.Solve(R.get(), rhs.get(), rhs.get(), n, rhs_cols));

        CHECK(S11XY.Allocate(n, rhs_cols));
        CHECK(ops.MatMul(S->S_11, rhs.get(), S11XY.get(), n, rhs_cols, n));
        CHECK(StatusFromCuda(
            rcwa_memory::MemcpyAsync(S->S_21, S11XY.get(), half_bytes, cudaMemcpyDeviceToDevice, stream_)));
        CHECK(AddInPlace(ops, S->S_21, S->S_12, n));
        CHECK(StatusFromCuda(rcwa_memory::MemcpyAsync(S->S_22,
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

Status RCWA::MaterializeGlobalSMatrix() const
{
    Complex** blocks[] = {&global_smatrix_.S_11,&global_smatrix_.S_12,&global_smatrix_.S_21,&global_smatrix_.S_22};
    bool stored[4]; size_t count=0;
    for(int i=0;i<4;++i) { stored[i]=rcwa_memory::IsHost(*blocks[i]);count+=stored[i]; }
    if(!count)return Status::kSuccess;
    const size_t n=static_cast<size_t>(2*order_N_);
    if(!n || n>std::numeric_limits<size_t>::max()/n/sizeof(Complex)/count)return Status::kInvalidValue;
    const size_t bytes=n*n*sizeof(Complex),needed=count*bytes;
    size_t free=0,total=0;
    if(cudaMemGetInfo(&free,&total)!=cudaSuccess || needed>free || free-needed<64*1024*1024)
        return Status::kCudaError;
    CHECK(StatusFromCuda(cudaStreamSynchronize(stream_)));
    Complex* staged[4]={nullptr,nullptr,nullptr,nullptr};
    Status status=Status::kSuccess;
    for(int i=0;i<4;++i)if(stored[i]) {
        if(cudaMalloc(&staged[i],bytes)!=cudaSuccess ||
           rcwa_memory::MemcpyAsync(staged[i],*blocks[i],bytes,cudaMemcpyDeviceToDevice,stream_)!=cudaSuccess) {
            status=Status::kCudaError;break;
        }
    }
    if(status==Status::kSuccess) status=StatusFromCuda(cudaStreamSynchronize(stream_));
    if(status!=Status::kSuccess) {
        for(auto ptr:staged)if(ptr)cudaFree(ptr);
        return status;
    }
    // Commit only after every block has been copied. A failed export leaves
    // the original out-of-core matrices intact and available for block reads.
    for(int i=0;i<4;++i)if(stored[i]) {
        rcwa_memory::Free(*blocks[i]); *blocks[i]=staged[i];
    }
    return Status::kSuccess;
}

const SMatrix& RCWA::global_smatrix() const
{
    if(MaterializeGlobalSMatrix()!=Status::kSuccess)
        throw std::logic_error("Full S matrix does not fit device memory; use ReadSMatrixBlock or EnergyBalance");
    return global_smatrix_;
}

Status RCWA::ReadSMatrixBlock(int block,int64_t row,int64_t col,int64_t rows,int64_t cols,Complex* host) const
{
    const int64_t n=2*order_N_;
    const Complex* ptr=block==11?global_smatrix_.S_11:block==12?global_smatrix_.S_12:block==21?global_smatrix_.S_21:block==22?global_smatrix_.S_22:nullptr;
    if(!host || !ptr || row<0 || col<0 || rows<=0 || cols<=0 || row>n-rows || col>n-cols)
        return Status::kInvalidValue;
    return rcwa_memory::ReadRect(ptr,n,row,col,rows,cols,host,stream_);
}
