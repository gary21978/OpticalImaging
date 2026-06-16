#include "farfield_abbe.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <cufft.h>

namespace
{
constexpr double kPi = 3.14159265358979323846;

static Status StatusFromCudaLocal(cudaError_t status)
{
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

static Status StatusFromCufftLocal(cufftResult status)
{
    return status == CUFFT_SUCCESS ? Status::kSuccess : Status::kCudaError;
}

struct DeviceComplexBuffer
{
    Complex* ptr = nullptr;

    ~DeviceComplexBuffer() { SAFEFREE(ptr); }

    Status Allocate(size_t count)
    {
        if (count == 0)
        {
            return Status::kInvalidValue;
        }
        SAFEFREE(ptr);
        return StatusFromCudaLocal(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(Complex)));
    }
};

struct DeviceRealBuffer
{
    Real* ptr = nullptr;

    ~DeviceRealBuffer() { SAFEFREE(ptr); }

    Status Allocate(size_t count)
    {
        if (count == 0)
        {
            return Status::kInvalidValue;
        }
        SAFEFREE(ptr);
        return StatusFromCudaLocal(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(Real)));
    }
};

struct DeviceIntBuffer
{
    int* ptr = nullptr;

    ~DeviceIntBuffer() { SAFEFREE(ptr); }

    Status Allocate(size_t count)
    {
        if (count == 0)
        {
            return Status::kInvalidValue;
        }
        SAFEFREE(ptr);
        return StatusFromCudaLocal(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(int)));
    }
};

struct RcwaFarfieldData
{
    // Per-order data derived once from RCWA and reused across all source/focus/z loops.
    std::vector<Real> kx;
    std::vector<Real> ky;
    std::vector<Real> kz;
    std::vector<Complex> kz_complex;
    std::vector<Complex> g00;
    std::vector<Complex> g01;
    std::vector<Complex> g11;
    std::vector<int> source_indices;
    std::vector<Complex> source_ux;
    std::vector<Complex> source_uy;
    const Complex* s_port = nullptr;
    int src_h = 0;
    int src_w = 0;
};

__host__ __device__ inline int64_t LinearIndex(int x, int y, int width)
{
    return static_cast<int64_t>(y) * width + x;
}

Complex HostComplexSqrtRCWA(Complex z)
{
    const Real a = ComplexReal(z);
    const Real b = ComplexImag(z);
    const Real r = ComplexAbs(z);
    const Real t = std::sqrt(static_cast<Real>(0.5) * (r + std::abs(a)));
    const Real u = (t > static_cast<Real>(0.0)) ? static_cast<Real>(0.5) * b / t : static_cast<Real>(0.0);
    const Real sign_a = a < static_cast<Real>(0.0) ? static_cast<Real>(1.0) : static_cast<Real>(0.0);
    const Real abs_u = std::abs(u);
    const Real re = t + sign_a * (abs_u - t);
    const Real im = abs_u + sign_a * (t - abs_u);
    return make_complex(re, im);
}

Complex HostComplexExpI(Complex z)
{
    const Real a = ComplexReal(z);
    const Real b = ComplexImag(z);
    const Real exp_r = std::exp(-b);
    return make_complex(exp_r * std::cos(a), exp_r * std::sin(a));
}

bool ApertureValid(AbbeApertureShape shape,
                   Real sigma_out,
                   Real sigma_in,
                   Real open_angle,
                   Real rotate_angle,
                   Real rect_x,
                   Real rect_y,
                   Real kx,
                   Real ky)
{
    const Real r2 = kx * kx + ky * ky;
    const Real r = std::sqrt(std::max(r2, static_cast<Real>(0.0)));
    switch (shape)
    {
    case AbbeApertureShape::kAnnular:
        return r2 <= sigma_out * sigma_out && r2 >= sigma_in * sigma_in;
    case AbbeApertureShape::kDipoleCirc:
    {
        if (r > sigma_out || r < sigma_in)
        {
            return false;
        }
        const Real theta = std::atan2(ky, kx);
        const Real cos_diff = std::cos(theta - rotate_angle);
        return std::abs(cos_diff) >= std::cos(open_angle / 2.0);
    }
    case AbbeApertureShape::kQuasar:
    {
        if (r > sigma_out || r < sigma_in)
        {
            return false;
        }
        const Real theta = std::atan2(ky, kx);
        auto in_sector = [&](Real center) {
            const Real diff = std::remainder(theta - center, static_cast<Real>(2.0 * kPi));
            return std::abs(diff) <= open_angle / 2.0;
        };
        return in_sector(rotate_angle) || in_sector(rotate_angle + static_cast<Real>(0.5 * kPi)) ||
               in_sector(rotate_angle + static_cast<Real>(kPi)) ||
               in_sector(rotate_angle - static_cast<Real>(0.5 * kPi));
    }
    case AbbeApertureShape::kRect:
    {
        const Real c = std::cos(-rotate_angle);
        const Real s = std::sin(-rotate_angle);
        const Real xp = kx * c - ky * s;
        const Real yp = kx * s + ky * c;
        return std::abs(xp) <= rect_x && std::abs(yp) <= rect_y;
    }
    default:
        return false;
    }
}

void BuildPolarizationVector(const AbbeSourceConfig& source, Complex out[2])
{
    switch (source.polarization_mode)
    {
    case AbbePolarizationMode::kJonesVector:
        out[0] = source.polarization_x;
        out[1] = source.polarization_y;
        return;
    case AbbePolarizationMode::kX:
        out[0] = make_complex(1.0, 0.0);
        out[1] = make_complex(0.0, 0.0);
        return;
    case AbbePolarizationMode::kY:
        out[0] = make_complex(0.0, 0.0);
        out[1] = make_complex(1.0, 0.0);
        return;
    case AbbePolarizationMode::kLine:
    {
        const Real v = std::sqrt(0.5);
        out[0] = make_complex(v, 0.0);
        out[1] = make_complex(v, 0.0);
        return;
    }
    case AbbePolarizationMode::kCircular:
    {
        const Real v = std::sqrt(0.5);
        out[0] = make_complex(v, 0.0);
        out[1] = make_complex(0.0, v);
        return;
    }
    case AbbePolarizationMode::kAngleDegrees:
    {
        const Real theta = source.polarization_angle_deg * static_cast<Real>(kPi / 180.0);
        out[0] = make_complex(std::cos(theta), 0.0);
        out[1] = make_complex(std::sin(theta), 0.0);
        return;
    }
    default:
        out[0] = make_complex(1.0, 0.0);
        out[1] = make_complex(0.0, 0.0);
        return;
    }
}

void InjectionMatrix(Real kx, Real ky, Real kz, Complex mat[4])
{
    const Real denom = std::abs(1.0 + kz) > 1e-12 ? (1.0 + kz) : 1e-12;
    const Real scale = kz > 1e-12 ? std::sqrt(1.0 / kz) : 0.0;
    mat[0] = make_complex(scale * (1.0 - kx * kx / denom), 0.0);
    mat[1] = make_complex(scale * (-kx * ky / denom), 0.0);
    mat[2] = make_complex(scale * (-kx * ky / denom), 0.0);
    mat[3] = make_complex(scale * (1.0 - ky * ky / denom), 0.0);
}

Real ZernikeAt(const std::vector<Real>& c, size_t idx) { return idx < c.size() ? c[idx] : 0.0; }

Real CalculateAberrationFast(const AbbeProjectorConfig& projector, Real rho, Real theta)
{
    const Real r2 = rho * rho;
    const Real r3 = r2 * rho;
    const Real r4 = r2 * r2;
    const Real r5 = r4 * rho;
    const Real r6 = r3 * r3;
    const Real r7 = r6 * rho;
    const Real r8 = r4 * r4;
    const Real r9 = r8 * rho;
    const Real r10 = r5 * r5;
    const Real r12 = r6 * r6;

    const Real ct = std::cos(theta);
    const Real st = std::sin(theta);
    const Real c2t = std::cos(2.0 * theta);
    const Real s2t = std::sin(2.0 * theta);
    const Real c3t = std::cos(3.0 * theta);
    const Real s3t = std::sin(3.0 * theta);
    const Real c4t = std::cos(4.0 * theta);
    const Real s4t = std::sin(4.0 * theta);
    const Real c5t = std::cos(5.0 * theta);
    const Real s5t = std::sin(5.0 * theta);

    const std::vector<Real>& c = projector.aberration_zernike;
    Real a = ZernikeAt(c, 0);
    a += ZernikeAt(c, 1) * rho * ct;
    a += ZernikeAt(c, 2) * rho * st;
    a += ZernikeAt(c, 3) * (2 * r2 - 1);
    a += ZernikeAt(c, 4) * r2 * c2t;
    a += ZernikeAt(c, 5) * r2 * s2t;
    a += ZernikeAt(c, 6) * (3 * r3 - 2 * rho) * ct;
    a += ZernikeAt(c, 7) * (3 * r3 - 2 * rho) * st;
    a += ZernikeAt(c, 8) * (6 * r4 - 6 * r2 + 1);
    a += ZernikeAt(c, 9) * r3 * c3t;
    a += ZernikeAt(c, 10) * r3 * s3t;
    a += ZernikeAt(c, 11) * (4 * r4 - 3 * r2) * c2t;
    a += ZernikeAt(c, 12) * (4 * r4 - 3 * r2) * s2t;
    a += ZernikeAt(c, 13) * (10 * r5 - 12 * r3 + 3 * rho) * ct;
    a += ZernikeAt(c, 14) * (10 * r5 - 12 * r3 + 3 * rho) * st;
    a += ZernikeAt(c, 15) * (20 * r6 - 30 * r4 + 12 * r2 - 1);
    a += ZernikeAt(c, 16) * r4 * c4t;
    a += ZernikeAt(c, 17) * r4 * s4t;
    a += ZernikeAt(c, 18) * (5 * r5 - 4 * r3) * c3t;
    a += ZernikeAt(c, 19) * (5 * r5 - 4 * r3) * s3t;
    a += ZernikeAt(c, 20) * (15 * r6 - 20 * r4 + 6 * r2) * c2t;
    a += ZernikeAt(c, 21) * (15 * r6 - 20 * r4 + 6 * r2) * s2t;
    a += ZernikeAt(c, 22) * (35 * r7 - 60 * r5 + 30 * r3 - 4 * rho) * ct;
    a += ZernikeAt(c, 23) * (35 * r7 - 60 * r5 + 30 * r3 - 4 * rho) * st;
    a += ZernikeAt(c, 24) * (70 * r8 - 140 * r6 + 90 * r4 - 20 * r2 + 1);
    a += ZernikeAt(c, 25) * r5 * c5t;
    a += ZernikeAt(c, 26) * r5 * s5t;
    a += ZernikeAt(c, 27) * (6 * r6 - 5 * r4) * c4t;
    a += ZernikeAt(c, 28) * (6 * r6 - 5 * r4) * s4t;
    a += ZernikeAt(c, 29) * (21 * r7 - 30 * r5 + 10 * r3) * c3t;
    a += ZernikeAt(c, 30) * (21 * r7 - 30 * r5 + 10 * r3) * s3t;
    a += ZernikeAt(c, 31) * (56 * r8 - 105 * r6 + 60 * r4 - 10 * r2) * c2t;
    a += ZernikeAt(c, 32) * (56 * r8 - 105 * r6 + 60 * r4 - 10 * r2) * s2t;
    a += ZernikeAt(c, 33) * (126 * r9 - 280 * r7 + 210 * r5 - 60 * r3 + 5 * rho) * ct;
    a += ZernikeAt(c, 34) * (126 * r9 - 280 * r7 + 210 * r5 - 60 * r3 + 5 * rho) * st;
    a += ZernikeAt(c, 35) * (252 * r10 - 630 * r8 + 560 * r6 - 210 * r4 + 30 * r2 - 1);
    a += ZernikeAt(c, 36) * (924 * r12 - 2772 * r10 + 3150 * r8 - 1680 * r6 + 420 * r4 - 42 * r2 + 1);
    return a;
}

std::vector<Complex> BuildProjectorTransfer(const std::vector<Real>& kx,
                                            const std::vector<Real>& ky,
                                            const AbbeProjectorConfig& projector,
                                            Real focus)
{
    const size_t n = kx.size();
    std::vector<Complex> transfer(n, make_complex(0.0, 0.0));
    for (size_t i = 0; i < n; ++i)
    {
        if (!ApertureValid(projector.shape,
                           projector.sigma_out,
                           projector.sigma_in,
                           projector.open_angle,
                           projector.rotate_angle,
                           projector.rect_x,
                           projector.rect_y,
                           kx[i],
                           ky[i]))
        {
            continue;
        }
        const Real rho2 = kx[i] * kx[i] + ky[i] * ky[i];
        const Real rho = std::sqrt(std::max(rho2, static_cast<Real>(0.0)));
        const Real theta = std::atan2(ky[i], kx[i]);
        const Real obliquity = std::pow(std::max(1.0 - rho2, static_cast<Real>(1e-6)), -0.25);
        const Real focus_base = std::sqrt(std::max(1.0 - rho2, static_cast<Real>(1e-6)));
        const Real rho_norm =
            projector.sigma_out > 1e-12 ? rho / projector.sigma_out : static_cast<Real>(0.0);
        const Real aberration = CalculateAberrationFast(projector, rho_norm, theta);
        // The projector transfer combines pupil support, obliquity, aberration phase,
        // and the focus-dependent phase term into one complex factor per diffraction order.
        const Real phase =
            static_cast<Real>(2.0 * kPi) * aberration +
            static_cast<Real>(2.0 * kPi / projector.wavelength) * focus_base * focus;
        transfer[i] = make_complex(obliquity * std::cos(phase), obliquity * std::sin(phase));
    }
    return transfer;
}

__global__ void ComplexPointwiseMulKernel(const Complex* a,
                                          const Complex* b,
                                          Complex* out,
                                          int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = ComplexMul(a[idx], b[idx]);
    }
}

/*
Discrete Abbe spectrum assembly for one source tile.

For each retained source sample k and diffraction order i, this kernel evaluates

    E_i^(k) = S_port(i, k) * u_k
    U_i^(k) = H(i) * G_i * E_i^(k)

where
    - u_k is the source-side injected Jones vector after the illumination mapping,
    - S_port is the RCWA port response restricted to the selected output port,
    - G_i is the per-order projector operator built from the imaging model,
    - H(i) = projector_transfer(i) * z_phase(i).

The resulting two polarized spectrum samples Ux_i^(k), Uy_i^(k) are then written
into the padded FFT grid used by the later batched inverse FFT.
*/
__global__ void BuildSpectrumTileKernel(const Complex* S_port,
                                        int64_t order_count,
                                        int src_h,
                                        int src_w,
                                        int dst_h,
                                        int dst_w,
                                        const int* source_indices,
                                        const Complex* u_x,
                                        const Complex* u_y,
                                        int source_base,
                                        int source_count_tile,
                                        const Complex* g00,
                                        const Complex* g01,
                                        const Complex* g11,
                                        const Complex* phase_transfer,
                                        Complex* batch_padded)
{
    // Each thread handles one (diffraction order i, source sample k) pair and writes
    // the two polarized spectrum values that will later be inverse-FFT'd as one batch item.
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int k_local = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= order_count || k_local >= source_count_tile)
    {
        return;
    }

    const int64_t n = order_count;
    const int64_t n2 = 2 * n;
    const int source_idx = source_indices[source_base + k_local];
    const int64_t col_x = source_idx;
    const int64_t col_y = source_idx + n;

    // Read the two relevant S-matrix columns directly instead of forming S_port * U explicitly.
    const Complex r00 = S_port[col_x * n2 + i];
    const Complex r01 = S_port[col_y * n2 + i];
    const Complex r10 = S_port[col_x * n2 + i + n];
    const Complex r11 = S_port[col_y * n2 + i + n];

    // E_i^(k): response of diffraction order i to source sample k after source-side injection.
    const Complex ex = ComplexAdd(ComplexMul(r00, u_x[source_base + k_local]),
                                  ComplexMul(r01, u_y[source_base + k_local]));
    const Complex ey = ComplexAdd(ComplexMul(r10, u_x[source_base + k_local]),
                                  ComplexMul(r11, u_y[source_base + k_local]));

    // Apply the per-order projector operator G_i and the combined z/focus transfer.
    const Complex sx = ComplexMul(phase_transfer[i],
                                  ComplexAdd(ComplexMul(g00[i], ex), ComplexMul(g01[i], ey)));
    const Complex sy = ComplexMul(phase_transfer[i],
                                  ComplexAdd(ComplexMul(g01[i], ex), ComplexMul(g11[i], ey)));

    // Embed the small RCWA spectrum window into the larger FFT grid with wrapped indexing.
    // RCWA orders are flattened with x as the outer index and y as the inner index.
    // Map them back to the FFT grid consistently: y selects rows, x selects columns.
    const int order_x_idx = i / src_w;
    const int order_y_idx = i % src_w;
    const int dst_row = (order_y_idx - src_w / 2 + dst_h) % dst_h;
    const int dst_col = (order_x_idx - src_h / 2 + dst_w) % dst_w;
    const int64_t dst_index = LinearIndex(dst_col, dst_row, dst_w);
    const int64_t image_count = static_cast<int64_t>(dst_h) * dst_w;
    const int64_t batch_x = 2LL * k_local;
    const int64_t batch_y = batch_x + 1;
    batch_padded[batch_x * image_count + dst_index] = sx;
    batch_padded[batch_y * image_count + dst_index] = sy;
}

__global__ void ZeroComplexKernel(Complex* data, int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = make_complex(0.0, 0.0);
    }
}

__global__ void ZeroRealKernel(Real* data, int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = 0.0;
    }
}

Status CopyAxis(const Real* src, int64_t count, std::vector<Real>* dst)
{
    dst->assign(static_cast<size_t>(count), 0.0);
    if (!src)
    {
        for (int64_t i = 0; i < count; ++i)
        {
            (*dst)[static_cast<size_t>(i)] = static_cast<Real>(i);
        }
        return Status::kSuccess;
    }
    std::copy(src, src + count, dst->begin());
    return Status::kSuccess;
}

Status InitializeImageResult(int64_t x_count,
                             int64_t y_count,
                             const Real* x_axis,
                             const Real* y_axis,
                             const AbbeProjectorConfig& projector,
                             const AbbeNumericsConfig& numerics,
                             AbbeImageResult* result)
{
    const int64_t image_count = x_count * y_count;
    const int64_t object_plane_count =
        std::max<int64_t>(1, static_cast<int64_t>(numerics.object_z_positions.size()));
    const int64_t focus_count = std::max<int64_t>(1, static_cast<int64_t>(projector.focus_range.size()));
    result->width = x_count;
    result->height = y_count;
    result->focus_count = focus_count;
    result->object_plane_count = object_plane_count;
    result->intensity.assign(static_cast<size_t>(image_count * object_plane_count * focus_count), 0.0);
    CHECK(CopyAxis(x_axis, x_count, &result->image_x));
    CHECK(CopyAxis(y_axis, y_count, &result->image_y));
    return Status::kSuccess;
}

/*
Build all RCWA-derived quantities that are independent of focus and object-plane z.

This stage prepares the discrete data used later by the Abbe accumulation loop:

    - source_indices:
        retained illumination samples k after applying the source pupil.

    - u_k = [ux_k, uy_k]^T:
        source-side injected Jones vector for each retained source sample.

    - G_i:
        per-order imaging/projector operator applied after the RCWA port response.

    - kz_i:
        signed longitudinal propagation term used by the later z-dependent phase
        exp(-i * omega * z * kz_i).

    - S_port:
        the RCWA output-port block used to map each source sample into its
        diffraction-order response.
*/
Status BuildRcwaFarfieldData(const RCWA& rcwa,
                             const AbbeSourceConfig& source,
                             RcwaFarfieldData* data,
                             cudaStream_t stream)
{
    CHECK(rcwa.GetKxKyKz(&data->kx, &data->ky, &data->kz, stream));

    const int ox = rcwa.order_x();
    const int oy = rcwa.order_y();
    data->src_h = 2 * ox + 1;
    data->src_w = 2 * oy + 1;

    const int64_t order_count = rcwa.order_N();
    if (static_cast<int64_t>(data->kx.size()) != order_count ||
        static_cast<int64_t>(data->ky.size()) != order_count ||
        static_cast<int64_t>(data->kz.size()) != order_count)
    {
        return Status::kInvalidValue;
    }

    const SMatrix& smat = rcwa.global_smatrix();
    if (rcwa.source_is_forward())
    {
        data->s_port = smat.S_21;
    }
    else
    {
        data->s_port = smat.S_12;
    }
    if (!data->s_port)
    {
        return Status::kInvalidValue;
    }

    data->source_indices.clear();
    data->source_ux.clear();
    data->source_uy.clear();
    data->source_indices.reserve(static_cast<size_t>(order_count));
    data->source_ux.reserve(static_cast<size_t>(order_count));
    data->source_uy.reserve(static_cast<size_t>(order_count));

    // Precompute the source-side injection vector u_k for every source point kept by the pupil.
    Complex pol_in[2];
    BuildPolarizationVector(source, pol_in);
    for (int64_t idx = 0; idx < order_count; ++idx)
    {
        if (!ApertureValid(source.shape,
                           source.sigma_out,
                           source.sigma_in,
                           source.open_angle,
                           source.rotate_angle,
                           source.rect_x,
                           source.rect_y,
                           data->kx[static_cast<size_t>(idx)],
                           data->ky[static_cast<size_t>(idx)]))
        {
            continue;
        }

        data->source_indices.push_back(static_cast<int>(idx));
        Complex inject_mat[4];
        InjectionMatrix(data->kx[static_cast<size_t>(idx)],
                        data->ky[static_cast<size_t>(idx)],
                        data->kz[static_cast<size_t>(idx)],
                        inject_mat);
        data->source_ux.push_back(ComplexAdd(ComplexMul(inject_mat[0], pol_in[0]),
                                             ComplexMul(inject_mat[1], pol_in[1])));
        data->source_uy.push_back(ComplexAdd(ComplexMul(inject_mat[2], pol_in[0]),
                                             ComplexMul(inject_mat[3], pol_in[1])));
    }
    if (data->source_indices.empty())
    {
        return Status::kInvalidValue;
    }

    data->g00.assign(static_cast<size_t>(order_count), make_complex(0.0, 0.0));
    data->g01.assign(static_cast<size_t>(order_count), make_complex(0.0, 0.0));
    data->g11.assign(static_cast<size_t>(order_count), make_complex(0.0, 0.0));
    data->kz_complex.assign(static_cast<size_t>(order_count), make_complex(0.0, 0.0));

    // Precompute the per-order projector operator G_i and the z-propagation kz sign convention.
    const Complex eps_ref = rcwa.angle_layer_is_input() ? rcwa.eps_in() : rcwa.eps_out();
    for (int64_t i = 0; i < order_count; ++i)
    {
        const Complex kx_c = make_complex(data->kx[static_cast<size_t>(i)], 0.0);
        const Complex ky_c = make_complex(data->ky[static_cast<size_t>(i)], 0.0);
        const Complex kz_i = HostComplexSqrtRCWA(eps_ref - kx_c * kx_c - ky_c * ky_c);
        data->kz_complex[static_cast<size_t>(i)] =
            rcwa.angle_layer_is_input() ? kz_i : ComplexMul(make_complex(-1.0, 0.0), kz_i);

        const Real rho2 = data->kx[static_cast<size_t>(i)] * data->kx[static_cast<size_t>(i)] +
                          data->ky[static_cast<size_t>(i)] * data->ky[static_cast<size_t>(i)];
        if (rho2 >= static_cast<Real>(1.0))
        {
            data->g00[static_cast<size_t>(i)] = make_complex(1.0, 0.0);
            data->g01[static_cast<size_t>(i)] = make_complex(0.0, 0.0);
            data->g11[static_cast<size_t>(i)] = make_complex(1.0, 0.0);
            continue;
        }

        const Real h = std::sqrt(std::max(static_cast<Real>(0.0), static_cast<Real>(1.0) - rho2));
        const Real coeff = h > static_cast<Real>(1e-12)
                               ? static_cast<Real>(1.0) / (h * (static_cast<Real>(1.0) + h))
                               : static_cast<Real>(0.0);
        const Real kxv = data->kx[static_cast<size_t>(i)];
        const Real kyv = data->ky[static_cast<size_t>(i)];
        data->g00[static_cast<size_t>(i)] =
            make_complex(static_cast<Real>(1.0) + coeff * kxv * kxv, 0.0);
        data->g01[static_cast<size_t>(i)] = make_complex(coeff * kxv * kyv, 0.0);
        data->g11[static_cast<size_t>(i)] =
            make_complex(static_cast<Real>(1.0) + coeff * kyv * kyv, 0.0);
    }

    return Status::kSuccess;
}

Status BuildFocusTransfers(const std::vector<Real>& kx,
                           const std::vector<Real>& ky,
                           const AbbeProjectorConfig& projector,
                           int64_t focus_count,
                           std::vector<std::vector<Complex>>* transfers)
{
    transfers->resize(static_cast<size_t>(focus_count));
    for (int64_t fi = 0; fi < focus_count; ++fi)
    {
        const Real focus = projector.focus_range.empty() ? 0.0 : projector.focus_range[static_cast<size_t>(fi)];
        (*transfers)[static_cast<size_t>(fi)] = BuildProjectorTransfer(kx, ky, projector, focus);
    }
    return Status::kSuccess;
}

Status CopyFieldIntensityToResult(const Complex* ex_field,
                                  const Complex* ey_field,
                                  const Complex* ez_field,
                                  int64_t count,
                                  std::vector<Real>* intensity,
                                  cudaStream_t stream)
{
    std::vector<Complex> host_ex(static_cast<size_t>(count));
    std::vector<Complex> host_ey(static_cast<size_t>(count));
    std::vector<Complex> host_ez(static_cast<size_t>(count));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(
        host_ex.data(), ex_field, static_cast<size_t>(count) * sizeof(Complex), cudaMemcpyDeviceToHost, stream)));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(
        host_ey.data(), ey_field, static_cast<size_t>(count) * sizeof(Complex), cudaMemcpyDeviceToHost, stream)));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(
        host_ez.data(), ez_field, static_cast<size_t>(count) * sizeof(Complex), cudaMemcpyDeviceToHost, stream)));
    CHECK(StatusFromCudaLocal(cudaStreamSynchronize(stream)));

    intensity->assign(static_cast<size_t>(count), 0.0);
    for (int64_t i = 0; i < count; ++i)
    {
        (*intensity)[static_cast<size_t>(i)] =
            static_cast<Real>(ComplexAbsSquaredDouble(host_ex[static_cast<size_t>(i)]) +
                              ComplexAbsSquaredDouble(host_ey[static_cast<size_t>(i)]) +
                              ComplexAbsSquaredDouble(host_ez[static_cast<size_t>(i)]));
    }
    return Status::kSuccess;
}

} // namespace

void ReleaseAbbeKernelCache(AbbeKernelCache* kernel_cache)
{
    if (kernel_cache)
    {
        kernel_cache->Clear();
    }
}

__global__ void AccumulateTileIntensityKernel(const Complex* batch_field,
                                              Real* intensity,
                                              int64_t image_count,
                                              int batch_count,
                                              Real scale)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= image_count)
    {
        return;
    }
    Real sum = intensity[idx];
    for (int b = 0; b < batch_count; ++b)
    {
        const Complex v = batch_field[static_cast<int64_t>(b) * image_count + idx];
        const Real re = ComplexReal(v) * scale;
        const Real im = ComplexImag(v) * scale;
        sum += re * re + im * im;
    }
    intensity[idx] = sum;
}

Status CalculateAbbeImageFromFields(const Complex* ex_field,
                                    const Complex* ey_field,
                                    const Complex* ez_field,
                                    int64_t width,
                                    int64_t height,
                                    Real /*period_x*/,
                                    Real /*period_y*/,
                                    const AbbeSourceConfig& /*source*/,
                                    const AbbeProjectorConfig& projector,
                                    const AbbeNumericsConfig& numerics,
                                    AbbeImageResult* result,
                                    AbbeKernelCache* kernel_cache,
                                    cudaStream_t stream)
{
    if (!ex_field || !ey_field || !ez_field || !result || width <= 0 || height <= 0)
    {
        return Status::kInvalidValue;
    }

    // This compatibility path is intentionally minimal: it only reports direct field intensity.
    const int64_t count = width * height;
    CHECK(InitializeImageResult(width, height, nullptr, nullptr, projector, numerics, result));
    if (result->object_plane_count != 1 || result->focus_count != 1)
    {
        return Status::kInvalidValue;
    }

    const cudaStream_t use_stream = stream ? stream : nullptr;
    CHECK(CopyFieldIntensityToResult(ex_field, ey_field, ez_field, count, &result->intensity, use_stream));

    if (kernel_cache)
    {
        kernel_cache->valid = false;
    }
    return Status::kSuccess;
}

Status CalculateAbbeImageFromRCWA(const RCWA& rcwa,
                                  const Real* x_axis,
                                  int64_t x_count,
                                  const Real* y_axis,
                                  int64_t y_count,
                                  Real /*period_x*/,
                                  Real /*period_y*/,
                                  const AbbeSourceConfig& source,
                                  const AbbeProjectorConfig& projector,
                                  const AbbeNumericsConfig& numerics,
                                  AbbeImageResult* result,
                                  AbbeKernelCache* kernel_cache,
                                  cudaStream_t stream)
{
    /*
    RCWA-driven Abbe imaging flow:

        1. Precompute all RCWA- and pupil-dependent quantities that do not vary with
           focus or object-plane z: source samples, injected source vectors u_k,
           per-order projector operator G_i, and the selected S_port block.

        2. For each object-plane z, build the z-dependent phase term exp(-i * omega * z * kz_i).

        3. For each focus slice, combine projector transfer and z-phase into H(i), then
           process the retained source samples in tiles.

        4. For each (order i, source k) pair inside a tile, assemble the padded spectrum
           sample H(i) * G_i * S_port(i, k) * u_k.

        5. Run a batched inverse FFT for the two polarized channels of all source samples
           in the tile, then accumulate |Ex|^2 + |Ey|^2 into the final image.
    */
    if (!result || x_count <= 0 || y_count <= 0 || projector.wavelength <= 0.0)
    {
        return Status::kInvalidValue;
    }

    const int64_t order_count = rcwa.order_N();
    const int64_t image_count = x_count * y_count;
    CHECK(InitializeImageResult(x_count, y_count, x_axis, y_axis, projector, numerics, result));
    const int64_t object_plane_count = result->object_plane_count;
    const int64_t focus_count = result->focus_count;

    RcwaFarfieldData rcwa_data;
    CHECK(BuildRcwaFarfieldData(rcwa, source, &rcwa_data, stream));

    // One transfer vector per focus slice; z-dependent phase is applied later inside the main loop.
    std::vector<std::vector<Complex>> transfers;
    CHECK(BuildFocusTransfers(rcwa_data.kx, rcwa_data.ky, projector, focus_count, &transfers));

    cudaStream_t use_stream = stream ? stream : nullptr;
    std::vector<DeviceComplexBuffer> d_transfer(static_cast<size_t>(focus_count));
    for (int64_t fi = 0; fi < focus_count; ++fi)
    {
        CHECK(d_transfer[static_cast<size_t>(fi)].Allocate(static_cast<size_t>(order_count)));
        CHECK(StatusFromCudaLocal(cudaMemcpyAsync(d_transfer[static_cast<size_t>(fi)].ptr,
                                                  transfers[static_cast<size_t>(fi)].data(),
                                                  static_cast<size_t>(order_count) * sizeof(Complex),
                                                  cudaMemcpyHostToDevice,
                                                  use_stream)));
    }

    DeviceComplexBuffer d_g00;
    DeviceComplexBuffer d_g01;
    DeviceComplexBuffer d_g11;
    DeviceComplexBuffer d_zphase;
    DeviceComplexBuffer d_phase_transfer;
    DeviceComplexBuffer batch_padded;
    DeviceRealBuffer intensity;
    DeviceIntBuffer d_source_indices;
    DeviceComplexBuffer d_source_ux;
    DeviceComplexBuffer d_source_uy;
    CHECK(d_g00.Allocate(static_cast<size_t>(order_count)));
    CHECK(d_g01.Allocate(static_cast<size_t>(order_count)));
    CHECK(d_g11.Allocate(static_cast<size_t>(order_count)));
    CHECK(d_zphase.Allocate(static_cast<size_t>(order_count)));
    CHECK(d_phase_transfer.Allocate(static_cast<size_t>(order_count)));
    CHECK(d_source_indices.Allocate(rcwa_data.source_indices.size()));
    CHECK(d_source_ux.Allocate(rcwa_data.source_ux.size()));
    CHECK(d_source_uy.Allocate(rcwa_data.source_uy.size()));
    CHECK(intensity.Allocate(static_cast<size_t>(image_count)));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(
        d_g00.ptr, rcwa_data.g00.data(), static_cast<size_t>(order_count) * sizeof(Complex), cudaMemcpyHostToDevice, use_stream)));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(
        d_g01.ptr, rcwa_data.g01.data(), static_cast<size_t>(order_count) * sizeof(Complex), cudaMemcpyHostToDevice, use_stream)));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(
        d_g11.ptr, rcwa_data.g11.data(), static_cast<size_t>(order_count) * sizeof(Complex), cudaMemcpyHostToDevice, use_stream)));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(d_source_indices.ptr,
                                              rcwa_data.source_indices.data(),
                                              rcwa_data.source_indices.size() * sizeof(int),
                                              cudaMemcpyHostToDevice,
                                              use_stream)));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(d_source_ux.ptr,
                                              rcwa_data.source_ux.data(),
                                              rcwa_data.source_ux.size() * sizeof(Complex),
                                              cudaMemcpyHostToDevice,
                                              use_stream)));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(d_source_uy.ptr,
                                              rcwa_data.source_uy.data(),
                                              rcwa_data.source_uy.size() * sizeof(Complex),
                                              cudaMemcpyHostToDevice,
                                              use_stream)));

    cufftHandle plan;
    int64_t source_tile = static_cast<int64_t>(
        (256ull * 1024ull * 1024ull) /
        std::max<size_t>(static_cast<size_t>(image_count) * sizeof(Complex) * 2ull, 1ull));
    source_tile =
        std::max<int64_t>(1, std::min<int64_t>(source_tile, static_cast<int64_t>(rcwa_data.source_indices.size())));
    const int batch_count_max = static_cast<int>(2 * source_tile);
    CHECK(batch_padded.Allocate(static_cast<size_t>(image_count * batch_count_max)));
#if USE_DOUBLE_PRECISION
    int n_fft[2] = {static_cast<int>(y_count), static_cast<int>(x_count)};
    CHECK(StatusFromCufftLocal(
        cufftPlanMany(&plan, 2, n_fft, nullptr, 1, static_cast<int>(image_count), nullptr, 1, static_cast<int>(image_count), CUFFT_Z2Z, batch_count_max)));
#else
    int n_fft[2] = {static_cast<int>(y_count), static_cast<int>(x_count)};
    CHECK(StatusFromCufftLocal(
        cufftPlanMany(&plan, 2, n_fft, nullptr, 1, static_cast<int>(image_count), nullptr, 1, static_cast<int>(image_count), CUFFT_C2C, batch_count_max)));
#endif
    CHECK(StatusFromCufftLocal(cufftSetStream(plan, use_stream)));

    const int image_blocks = static_cast<int>((image_count + CUDA_THREADS - 1) / CUDA_THREADS);
    // cuFFT inverse is unnormalized, so no extra Nx*Ny scale is needed here.
    const Real ifft_scale = static_cast<Real>(1.0);
    const Real normalization = source.sigma_out * source.sigma_out /
                               static_cast<Real>(rcwa_data.source_indices.size());

    for (int64_t zi = 0; zi < object_plane_count; ++zi)
    {
        const Real z_posi = numerics.object_z_positions.empty()
                                ? 0.0
                                : numerics.object_z_positions[static_cast<size_t>(zi)];
        std::vector<Complex> z_phase(static_cast<size_t>(order_count));
        for (int64_t i = 0; i < order_count; ++i)
        {
            const Complex kz_i = rcwa_data.kz_complex[static_cast<size_t>(i)];
            const Complex arg = make_complex(-rcwa.omega() * z_posi * ComplexReal(kz_i),
                                             -rcwa.omega() * z_posi * ComplexImag(kz_i));
            z_phase[static_cast<size_t>(i)] = HostComplexExpI(arg);
        }
        CHECK(StatusFromCudaLocal(cudaMemcpyAsync(d_zphase.ptr,
                                                  z_phase.data(),
                                                  static_cast<size_t>(order_count) * sizeof(Complex),
                                                  cudaMemcpyHostToDevice,
                                                  use_stream)));
        for (int64_t fi = 0; fi < focus_count; ++fi)
        {
            // phase_transfer(i) = projector_transfer_f(i) * z_phase(i)
            ZeroRealKernel<<<image_blocks, CUDA_THREADS, 0, use_stream>>>(intensity.ptr, image_count);
            CHECK(StatusFromCudaLocal(cudaGetLastError()));
            ComplexPointwiseMulKernel<<<static_cast<int>((order_count + CUDA_THREADS - 1) / CUDA_THREADS),
                                        CUDA_THREADS,
                                        0,
                                        use_stream>>>(d_transfer[static_cast<size_t>(fi)].ptr,
                                                      d_zphase.ptr,
                                                      d_phase_transfer.ptr,
                                                      order_count);
            CHECK(StatusFromCudaLocal(cudaGetLastError()));

            for (int64_t source_base = 0; source_base < static_cast<int64_t>(rcwa_data.source_indices.size());
                 source_base += source_tile)
            {
                const int current_sources =
                    static_cast<int>(std::min<int64_t>(source_tile,
                                                       static_cast<int64_t>(rcwa_data.source_indices.size()) - source_base));
                const int current_batches = 2 * current_sources;
                // The tile is zero-filled first because only the RCWA support region is written explicitly.
                ZeroComplexKernel<<<static_cast<int>((image_count * current_batches + CUDA_THREADS - 1) / CUDA_THREADS),
                                    CUDA_THREADS,
                                    0,
                                    use_stream>>>(batch_padded.ptr, image_count * current_batches);
                CHECK(StatusFromCudaLocal(cudaGetLastError()));

                const dim3 block(32, 4, 1);
                const dim3 grid(static_cast<unsigned int>((order_count + block.x - 1) / block.x),
                                static_cast<unsigned int>((current_sources + block.y - 1) / block.y),
                                1);
                BuildSpectrumTileKernel<<<grid, block, 0, use_stream>>>(rcwa_data.s_port,
                                                                        order_count,
                                                                        rcwa_data.src_h,
                                                                        rcwa_data.src_w,
                                                                        static_cast<int>(y_count),
                                                                        static_cast<int>(x_count),
                                                                        d_source_indices.ptr,
                                                                        d_source_ux.ptr,
                                                                        d_source_uy.ptr,
                                                                        static_cast<int>(source_base),
                                                                        current_sources,
                                                                        d_g00.ptr,
                                                                        d_g01.ptr,
                                                                        d_g11.ptr,
                                                                        d_phase_transfer.ptr,
                                                                        batch_padded.ptr);
                CHECK(StatusFromCudaLocal(cudaGetLastError()));

#if USE_DOUBLE_PRECISION
                CHECK(StatusFromCufftLocal(cufftExecZ2Z(plan,
                                                        reinterpret_cast<cufftDoubleComplex*>(batch_padded.ptr),
                                                        reinterpret_cast<cufftDoubleComplex*>(batch_padded.ptr),
                                                        CUFFT_INVERSE)));
#else
                CHECK(StatusFromCufftLocal(cufftExecC2C(plan,
                                                        reinterpret_cast<cufftComplex*>(batch_padded.ptr),
                                                        reinterpret_cast<cufftComplex*>(batch_padded.ptr),
                                                        CUFFT_INVERSE)));
#endif
                // Accumulate |E_x|^2 + |E_y|^2 over all source samples in this tile.
                AccumulateTileIntensityKernel<<<image_blocks, CUDA_THREADS, 0, use_stream>>>(
                    batch_padded.ptr, intensity.ptr, image_count, current_batches, ifft_scale);
                CHECK(StatusFromCudaLocal(cudaGetLastError()));
            }

            std::vector<Real> host_intensity(static_cast<size_t>(image_count));
            CHECK(StatusFromCudaLocal(cudaMemcpyAsync(host_intensity.data(),
                                                      intensity.ptr,
                                                      static_cast<size_t>(image_count) * sizeof(Real),
                                                      cudaMemcpyDeviceToHost,
                                                      use_stream)));
            CHECK(StatusFromCudaLocal(cudaStreamSynchronize(use_stream)));
            const size_t offset = static_cast<size_t>((zi * focus_count + fi) * image_count);
            for (size_t i = 0; i < host_intensity.size(); ++i)
            {
                result->intensity[offset + i] = host_intensity[i] * normalization;
            }
        }
    }

    CHECK(StatusFromCufftLocal(cufftDestroy(plan)));
    if (kernel_cache)
    {
        kernel_cache->valid = false;
    }
    return Status::kSuccess;
}
