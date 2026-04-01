#include "farfield_hopkins.h"

#include <cufft.h>
#include <cusolverDn.h>

#include "complex_matrix.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;

struct SourceSample
{
    Real x = 0.0;
    Real y = 0.0;
    Real weight = 0.0;
};

static Status StatusFromCudaLocal(cudaError_t status)
{
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

static Status StatusFromCufftLocal(cufftResult status)
{
    return status == CUFFT_SUCCESS ? Status::kSuccess : Status::kCudaError;
}

static Status StatusFromCusolverLocal(cusolverStatus_t status)
{
    return status == CUSOLVER_STATUS_SUCCESS ? Status::kSuccess : Status::kCusolverError;
}

struct DeviceComplexBuffer
{
    Complex* ptr = nullptr;

    ~DeviceComplexBuffer() { SAFEFREE(ptr); }

    Complex* get() const { return ptr; }

    Status Allocate(size_t count)
    {
        if (count == 0)
        {
            return Status::kInvalidValue;
        }
        if (ptr)
        {
            SAFEFREE(ptr);
        }
        return StatusFromCudaLocal(
            cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(Complex)));
    }
};

struct DeviceRealBufferLocal
{
    Real* ptr = nullptr;

    ~DeviceRealBufferLocal() { SAFEFREE(ptr); }

    Real* get() const { return ptr; }

    Status Allocate(size_t count)
    {
        if (count == 0)
        {
            return Status::kInvalidValue;
        }
        if (ptr)
        {
            SAFEFREE(ptr);
        }
        return StatusFromCudaLocal(
            cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(Real)));
    }
};

struct DeviceIntBuffer
{
    int* ptr = nullptr;

    ~DeviceIntBuffer() { SAFEFREE(ptr); }

    int* get() const { return ptr; }

    Status Allocate()
    {
        if (ptr)
        {
            SAFEFREE(ptr);
        }
        return StatusFromCudaLocal(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(int)));
    }
};

struct DeviceKernelBank
{
    int64_t elements_per_kernel = 0;
    std::vector<Complex*> focus_data;
    std::vector<int> kernel_counts;

    ~DeviceKernelBank() { Clear(); }

    void Clear()
    {
        for (Complex*& ptr : focus_data)
        {
            SAFEFREE(ptr);
        }
        focus_data.clear();
        kernel_counts.clear();
        elements_per_kernel = 0;
    }
};

std::unordered_map<uintptr_t, DeviceKernelBank> g_device_kernel_banks;

__global__ void Shift2DComplexKernel(const Complex* input,
                                     Complex* output,
                                     int width,
                                     int height,
                                     int shift_x,
                                     int shift_y)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(width) * height;
    if (idx >= total)
    {
        return;
    }

    const int x = static_cast<int>(idx % width);
    const int y = static_cast<int>(idx / width);
    const int src_x = (x + shift_x) % width;
    const int src_y = (y + shift_y) % height;
    output[idx] = input[static_cast<int64_t>(src_y) * width + src_x];
}

__global__ void ScaleComplexKernelHopkins(Complex* data, int64_t count, Real scale)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
    {
        return;
    }
    const Complex value = data[idx];
    data[idx] = make_complex(ComplexReal(value) * scale, ComplexImag(value) * scale);
}

__global__ void PrepareBatchedProductsKernel(const Complex* kernel_base,
                                             const Complex* spectrum_x,
                                             const Complex* spectrum_y,
                                             const Complex* spectrum_z,
                                             Complex* out,
                                             int64_t count,
                                             int kernel_count)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(kernel_count) * 3 * count;
    if (idx >= total)
    {
        return;
    }

    const int64_t element_idx = idx % count;
    const int batch_idx = static_cast<int>(idx / count);
    const int spectrum_idx = batch_idx % 3;
    const int kernel_idx = batch_idx / 3;

    const Complex* spectrum =
        spectrum_idx == 0 ? spectrum_x : (spectrum_idx == 1 ? spectrum_y : spectrum_z);
    const Complex kernel_value =
        kernel_base[static_cast<int64_t>(kernel_idx) * count + element_idx];
    out[idx] = ComplexMul(spectrum[element_idx], kernel_value);
}

__global__ void AccumulateBatchedIntensityKernel(const Complex* fields,
                                                 Real* intensity,
                                                 int64_t count,
                                                 int batch_count,
                                                 Real field_scale)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
    {
        return;
    }

    Real sum = 0.0;
    for (int batch_idx = 0; batch_idx < batch_count; ++batch_idx)
    {
        const Complex value = fields[static_cast<int64_t>(batch_idx) * count + idx];
        const Real re = ComplexReal(value) * field_scale;
        const Real im = ComplexImag(value) * field_scale;
        sum += re * re + im * im;
    }
    intensity[idx] += sum;
}

inline int64_t LinearIndex(int x, int y, int width) { return static_cast<int64_t>(y) * width + x; }

void FFTShiftHost(const std::vector<Complex>& input,
                  std::vector<Complex>* output,
                  int width,
                  int height)
{
    output->assign(input.size(), make_complex(0.0, 0.0));
    const int shift_x = width / 2;
    const int shift_y = height / 2;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int src_x = (x + shift_x) % width;
            const int src_y = (y + shift_y) % height;
            (*output)[LinearIndex(x, y, width)] = input[LinearIndex(src_x, src_y, width)];
        }
    }
}

void FFTShiftHost(const std::vector<Real>& input, std::vector<Real>* output, int width, int height)
{
    output->assign(input.size(), 0.0);
    const int shift_x = width / 2;
    const int shift_y = height / 2;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int src_x = (x + shift_x) % width;
            const int src_y = (y + shift_y) % height;
            (*output)[LinearIndex(x, y, width)] = input[LinearIndex(src_x, src_y, width)];
        }
    }
}

std::vector<Real> MakeLinearAxis(Real min_value, Real max_value, int64_t count)
{
    std::vector<Real> axis(static_cast<size_t>(count), 0.0);
    if (count <= 0)
    {
        return axis;
    }
    if (count == 1)
    {
        axis[0] = 0.5 * (min_value + max_value);
        return axis;
    }

    const Real step = (max_value - min_value) / static_cast<Real>(count - 1);
    for (int64_t i = 0; i < count; ++i)
    {
        axis[static_cast<size_t>(i)] = min_value + step * static_cast<Real>(i);
    }
    return axis;
}

std::vector<SourceSample> BuildAnnularSourceSamples(const HopkinsSourceConfig& config)
{
    const int point_count = std::max(config.point_count, 3);
    const Real denom = std::max(
        static_cast<Real>(
            (config.sigma_out * config.sigma_out - config.sigma_in * config.sigma_in) * kPi),
        static_cast<Real>(1e-12));
    const Real weight = 1.0 / denom;

    std::vector<SourceSample> samples;
    samples.reserve(static_cast<size_t>(point_count) * point_count);

    for (int iy = 0; iy < point_count; ++iy)
    {
        const Real y = -1.0 + 2.0 * static_cast<Real>(iy) / static_cast<Real>(point_count - 1);
        for (int ix = 0; ix < point_count; ++ix)
        {
            const Real x = -1.0 + 2.0 * static_cast<Real>(ix) / static_cast<Real>(point_count - 1);
            const Real r2 = x * x + y * y;
            const Real r = std::sqrt(std::max(r2, static_cast<Real>(0.0)));
            if (r <= 1.0 && r <= config.sigma_out && r >= config.sigma_in)
            {
                samples.push_back(SourceSample{x, y, weight});
            }
        }
    }
    return samples;
}

Real ZernikeAt(const std::vector<Real>& c, size_t idx) { return idx < c.size() ? c[idx] : 0.0; }

Real CalculateAberrationFast(const HopkinsProjectorConfig& projector, Real rho, Real theta)
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

    Real aberration = ZernikeAt(projector.aberration_zernike, 0);
    aberration += ZernikeAt(projector.aberration_zernike, 1) * (rho * ct);
    aberration += ZernikeAt(projector.aberration_zernike, 2) * (rho * st);
    aberration += ZernikeAt(projector.aberration_zernike, 3) * (2.0 * r2 - 1.0);
    aberration += ZernikeAt(projector.aberration_zernike, 4) * (r2 * c2t);
    aberration += ZernikeAt(projector.aberration_zernike, 5) * (r2 * s2t);
    aberration += ZernikeAt(projector.aberration_zernike, 6) * ((3.0 * r3 - 2.0 * rho) * ct);
    aberration += ZernikeAt(projector.aberration_zernike, 7) * ((3.0 * r3 - 2.0 * rho) * st);
    aberration += ZernikeAt(projector.aberration_zernike, 8) * (6.0 * r4 - 6.0 * r2 + 1.0);
    aberration += ZernikeAt(projector.aberration_zernike, 9) * (r3 * c3t);
    aberration += ZernikeAt(projector.aberration_zernike, 10) * (r3 * s3t);
    aberration += ZernikeAt(projector.aberration_zernike, 11) * ((4.0 * r4 - 3.0 * r2) * c2t);
    aberration += ZernikeAt(projector.aberration_zernike, 12) * ((4.0 * r4 - 3.0 * r2) * s2t);
    aberration +=
        ZernikeAt(projector.aberration_zernike, 13) * ((10.0 * r5 - 12.0 * r3 + 3.0 * rho) * ct);
    aberration +=
        ZernikeAt(projector.aberration_zernike, 14) * ((10.0 * r5 - 12.0 * r3 + 3.0 * rho) * st);
    aberration +=
        ZernikeAt(projector.aberration_zernike, 15) * (20.0 * r6 - 30.0 * r4 + 12.0 * r2 - 1.0);
    aberration += ZernikeAt(projector.aberration_zernike, 16) * (r4 * c4t);
    aberration += ZernikeAt(projector.aberration_zernike, 17) * (r4 * s4t);
    aberration += ZernikeAt(projector.aberration_zernike, 18) * ((5.0 * r5 - 4.0 * r3) * c3t);
    aberration += ZernikeAt(projector.aberration_zernike, 19) * ((5.0 * r5 - 4.0 * r3) * s3t);
    aberration +=
        ZernikeAt(projector.aberration_zernike, 20) * ((15.0 * r6 - 20.0 * r4 + 6.0 * r2) * c2t);
    aberration +=
        ZernikeAt(projector.aberration_zernike, 21) * ((15.0 * r6 - 20.0 * r4 + 6.0 * r2) * s2t);
    aberration += ZernikeAt(projector.aberration_zernike, 22) *
                  ((35.0 * r7 - 60.0 * r5 + 30.0 * r3 - 4.0 * rho) * ct);
    aberration += ZernikeAt(projector.aberration_zernike, 23) *
                  ((35.0 * r7 - 60.0 * r5 + 30.0 * r3 - 4.0 * rho) * st);
    aberration += ZernikeAt(projector.aberration_zernike, 24) *
                  (70.0 * r8 - 140.0 * r6 + 90.0 * r4 - 20.0 * r2 + 1.0);
    aberration += ZernikeAt(projector.aberration_zernike, 25) * (r5 * c5t);
    aberration += ZernikeAt(projector.aberration_zernike, 26) * (r5 * s5t);
    aberration += ZernikeAt(projector.aberration_zernike, 27) * ((6.0 * r6 - 5.0 * r4) * c4t);
    aberration += ZernikeAt(projector.aberration_zernike, 28) * ((6.0 * r6 - 5.0 * r4) * s4t);
    aberration +=
        ZernikeAt(projector.aberration_zernike, 29) * ((21.0 * r7 - 30.0 * r5 + 10.0 * r3) * c3t);
    aberration +=
        ZernikeAt(projector.aberration_zernike, 30) * ((21.0 * r7 - 30.0 * r5 + 10.0 * r3) * s3t);
    aberration += ZernikeAt(projector.aberration_zernike, 31) *
                  ((56.0 * r8 - 105.0 * r6 + 60.0 * r4 - 10.0 * r2) * c2t);
    aberration += ZernikeAt(projector.aberration_zernike, 32) *
                  ((56.0 * r8 - 105.0 * r6 + 60.0 * r4 - 10.0 * r2) * s2t);
    aberration += ZernikeAt(projector.aberration_zernike, 33) *
                  ((126.0 * r9 - 280.0 * r7 + 210.0 * r5 - 60.0 * r3 + 5.0 * rho) * ct);
    aberration += ZernikeAt(projector.aberration_zernike, 34) *
                  ((126.0 * r9 - 280.0 * r7 + 210.0 * r5 - 60.0 * r3 + 5.0 * rho) * st);
    aberration += ZernikeAt(projector.aberration_zernike, 35) *
                  (252.0 * r10 - 630.0 * r8 + 560.0 * r6 - 210.0 * r4 + 30.0 * r2 - 1.0);
    aberration +=
        ZernikeAt(projector.aberration_zernike, 36) *
        (924.0 * r12 - 2772.0 * r10 + 3150.0 * r8 - 1680.0 * r6 + 420.0 * r4 - 42.0 * r2 + 1.0);
    return aberration;
}

bool NearlyEqualReal(Real a, Real b)
{
    const Real diff = RealAbs(a - b);
    const Real scale = std::max(static_cast<Real>(1.0), std::max(RealAbs(a), RealAbs(b)));
    return diff <= static_cast<Real>(1e-9) * scale;
}

bool NearlyEqualComplex(Complex a, Complex b)
{
    return NearlyEqualReal(ComplexReal(a), ComplexReal(b)) &&
           NearlyEqualReal(ComplexImag(a), ComplexImag(b));
}

bool NearlyEqualVector(const std::vector<Real>& lhs, const std::vector<Real>& rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        if (!NearlyEqualReal(lhs[i], rhs[i]))
        {
            return false;
        }
    }
    return true;
}

std::vector<Real> NormalizedFocusRange(const HopkinsProjectorConfig& projector)
{
    return projector.focus_range.empty() ? std::vector<Real>{0.0} : projector.focus_range;
}

bool CacheMatches(const HopkinsKernelCache& cache,
                  int64_t width,
                  int64_t height,
                  Real period_x,
                  Real period_y,
                  const HopkinsSourceConfig& source,
                  const HopkinsProjectorConfig& projector,
                  const HopkinsNumericsConfig& numerics,
                  const std::vector<Real>& focus_range)
{
    return cache.valid && cache.width == width && cache.height == height &&
           NearlyEqualReal(cache.period_x, period_x) && NearlyEqualReal(cache.period_y, period_y) &&
           cache.source.point_count == source.point_count &&
           NearlyEqualReal(cache.source.sigma_out, source.sigma_out) &&
           NearlyEqualReal(cache.source.sigma_in, source.sigma_in) &&
           NearlyEqualComplex(cache.source.polarization_x, source.polarization_x) &&
           NearlyEqualComplex(cache.source.polarization_y, source.polarization_y) &&
           NearlyEqualReal(cache.projector.wavelength, projector.wavelength) &&
           NearlyEqualReal(cache.projector.magnification, projector.magnification) &&
           NearlyEqualReal(cache.projector.NA, projector.NA) &&
           NearlyEqualReal(cache.projector.index_image, projector.index_image) &&
           NearlyEqualVector(cache.projector.focus_range, focus_range) &&
           NearlyEqualVector(cache.projector.aberration_zernike, projector.aberration_zernike) &&
           cache.numerics.imaging_mode == numerics.imaging_mode &&
           cache.numerics.max_kernels == numerics.max_kernels &&
           cache.numerics.use_threshold == numerics.use_threshold &&
           NearlyEqualReal(cache.numerics.threshold, numerics.threshold) &&
           cache.kernels_by_focus.size() == focus_range.size();
}

void ResetKernelCache(HopkinsKernelCache* cache,
                      int64_t width,
                      int64_t height,
                      Real period_x,
                      Real period_y,
                      const HopkinsSourceConfig& source,
                      const HopkinsProjectorConfig& projector,
                      const HopkinsNumericsConfig& numerics,
                      const std::vector<Real>& focus_range)
{
    if (!cache)
    {
        return;
    }

    cache->Clear();
    cache->width = width;
    cache->height = height;
    cache->period_x = period_x;
    cache->period_y = period_y;
    cache->source = source;
    cache->projector = projector;
    cache->projector.focus_range = focus_range;
    cache->numerics = numerics;
    cache->kernels_by_focus.assign(focus_range.size(), {});
}

Status EnsureDeviceKernelsForFocus(const HopkinsKernelCache* cache,
                                   size_t focus_idx,
                                   int64_t elements_per_kernel,
                                   cudaStream_t stream,
                                   const Complex** device_kernel_base,
                                   int* kernel_count)
{
    if (!cache || !device_kernel_base || !kernel_count ||
        focus_idx >= cache->kernels_by_focus.size())
    {
        return Status::kInvalidValue;
    }

    auto& bank = g_device_kernel_banks[reinterpret_cast<uintptr_t>(cache)];
    if (bank.elements_per_kernel != elements_per_kernel ||
        bank.focus_data.size() != cache->kernels_by_focus.size())
    {
        bank.Clear();
        bank.elements_per_kernel = elements_per_kernel;
        bank.focus_data.resize(cache->kernels_by_focus.size(), nullptr);
        bank.kernel_counts.resize(cache->kernels_by_focus.size(), 0);
    }

    const auto& kernels = cache->kernels_by_focus[focus_idx];
    const int required_kernel_count = static_cast<int>(kernels.size());
    if (required_kernel_count <= 0)
    {
        return Status::kInvalidValue;
    }

    if (!bank.focus_data[focus_idx] || bank.kernel_counts[focus_idx] != required_kernel_count)
    {
        SAFEFREE(bank.focus_data[focus_idx]);
        bank.kernel_counts[focus_idx] = required_kernel_count;

        std::vector<Complex> flattened(static_cast<size_t>(required_kernel_count) *
                                       elements_per_kernel);
        for (int kernel_idx = 0; kernel_idx < required_kernel_count; ++kernel_idx)
        {
            std::copy(kernels[static_cast<size_t>(kernel_idx)].begin(),
                      kernels[static_cast<size_t>(kernel_idx)].end(),
                      flattened.begin() + static_cast<size_t>(kernel_idx) * elements_per_kernel);
        }

        CHECK(StatusFromCudaLocal(cudaMalloc(reinterpret_cast<void**>(&bank.focus_data[focus_idx]),
                                             flattened.size() * sizeof(Complex))));
        CHECK(StatusFromCudaLocal(cudaMemcpy(bank.focus_data[focus_idx],
                                             flattened.data(),
                                             flattened.size() * sizeof(Complex),
                                             cudaMemcpyHostToDevice)));
    }

    *device_kernel_base = bank.focus_data[focus_idx];
    *kernel_count = bank.kernel_counts[focus_idx];
    return Status::kSuccess;
}

Status HermitianEigenDecompositionDevice(const Complex* device_matrix,
                                         int n,
                                         std::vector<Real>* eigenvalues,
                                         std::vector<Complex>* eigenvectors,
                                         cudaStream_t stream)
{
    if (!device_matrix || !eigenvalues || !eigenvectors || n <= 0)
    {
        return Status::kInvalidValue;
    }

    cudaStream_t use_stream = stream ? stream : nullptr;
    cusolverDnHandle_t handle = nullptr;
    CHECK(StatusFromCusolverLocal(cusolverDnCreate(&handle)));
    CHECK(StatusFromCusolverLocal(cusolverDnSetStream(handle, use_stream)));

    DeviceComplexBuffer d_matrix;
    DeviceRealBufferLocal d_evals;
    DeviceIntBuffer d_info;
    CHECK(d_matrix.Allocate(static_cast<size_t>(n) * n));
    CHECK(d_evals.Allocate(static_cast<size_t>(n)));
    CHECK(d_info.Allocate());

    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(d_matrix.get(),
                                              device_matrix,
                                              static_cast<size_t>(n) * n * sizeof(Complex),
                                              cudaMemcpyDeviceToDevice,
                                              use_stream)));

    int workspace_elems = 0;
#if USE_DOUBLE_PRECISION
    CHECK(StatusFromCusolverLocal(cusolverDnZheevd_bufferSize(handle,
                                                              CUSOLVER_EIG_MODE_VECTOR,
                                                              CUBLAS_FILL_MODE_LOWER,
                                                              n,
                                                              d_matrix.get(),
                                                              n,
                                                              d_evals.get(),
                                                              &workspace_elems)));
#else
    CHECK(StatusFromCusolverLocal(cusolverDnCheevd_bufferSize(handle,
                                                              CUSOLVER_EIG_MODE_VECTOR,
                                                              CUBLAS_FILL_MODE_LOWER,
                                                              n,
                                                              d_matrix.get(),
                                                              n,
                                                              d_evals.get(),
                                                              &workspace_elems)));
#endif

    DeviceComplexBuffer d_work;
    CHECK(d_work.Allocate(static_cast<size_t>(std::max(workspace_elems, 1))));

#if USE_DOUBLE_PRECISION
    CHECK(StatusFromCusolverLocal(cusolverDnZheevd(handle,
                                                   CUSOLVER_EIG_MODE_VECTOR,
                                                   CUBLAS_FILL_MODE_LOWER,
                                                   n,
                                                   d_matrix.get(),
                                                   n,
                                                   d_evals.get(),
                                                   d_work.get(),
                                                   workspace_elems,
                                                   d_info.get())));
#else
    CHECK(StatusFromCusolverLocal(cusolverDnCheevd(handle,
                                                   CUSOLVER_EIG_MODE_VECTOR,
                                                   CUBLAS_FILL_MODE_LOWER,
                                                   n,
                                                   d_matrix.get(),
                                                   n,
                                                   d_evals.get(),
                                                   d_work.get(),
                                                   workspace_elems,
                                                   d_info.get())));
#endif

    int info = 0;
    CHECK(StatusFromCudaLocal(
        cudaMemcpyAsync(&info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost, use_stream)));
    CHECK(StatusFromCudaLocal(cudaStreamSynchronize(use_stream)));
    if (info != 0)
    {
        cusolverDnDestroy(handle);
        return Status::kCusolverError;
    }

    eigenvalues->resize(static_cast<size_t>(n));
    eigenvectors->resize(static_cast<size_t>(n) * n);

    CHECK(StatusFromCudaLocal(cudaMemcpy(eigenvalues->data(),
                                         d_evals.get(),
                                         static_cast<size_t>(n) * sizeof(Real),
                                         cudaMemcpyDeviceToHost)));
    CHECK(StatusFromCudaLocal(cudaMemcpy(eigenvectors->data(),
                                         d_matrix.get(),
                                         static_cast<size_t>(n) * n * sizeof(Complex),
                                         cudaMemcpyDeviceToHost)));

    CHECK(StatusFromCusolverLocal(cusolverDnDestroy(handle)));
    return Status::kSuccess;
}

Status ComputeSpectrumFromField(const Complex* field,
                                int width,
                                int height,
                                Real scale,
                                cufftHandle plan,
                                cudaStream_t stream,
                                Complex* spectrum_out,
                                Complex* shifted_workspace = nullptr)
{
    if (!field || !spectrum_out || width <= 0 || height <= 0)
    {
        return Status::kInvalidValue;
    }

    const int64_t count = static_cast<int64_t>(width) * height;
    const int blocks = static_cast<int>((count + CUDA_THREADS - 1) / CUDA_THREADS);

    DeviceComplexBuffer shifted_local;
    Complex* shifted_ptr = shifted_workspace;
    if (!shifted_ptr)
    {
        CHECK(shifted_local.Allocate(static_cast<size_t>(count)));
        shifted_ptr = shifted_local.get();
    }

    const int shift_x = (width + 1) / 2;
    const int shift_y = (height + 1) / 2;
    Shift2DComplexKernel<<<blocks, CUDA_THREADS, 0, stream>>>(
        field, shifted_ptr, width, height, shift_x, shift_y);
    CHECK(StatusFromCudaLocal(cudaGetLastError()));

    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(spectrum_out,
                                              shifted_ptr,
                                              static_cast<size_t>(count) * sizeof(Complex),
                                              cudaMemcpyDeviceToDevice,
                                              stream)));

#if USE_DOUBLE_PRECISION
    CHECK(StatusFromCufftLocal(cufftExecZ2Z(plan,
                                            reinterpret_cast<cufftDoubleComplex*>(spectrum_out),
                                            reinterpret_cast<cufftDoubleComplex*>(spectrum_out),
                                            CUFFT_INVERSE)));
#else
    CHECK(StatusFromCufftLocal(cufftExecC2C(plan,
                                            reinterpret_cast<cufftComplex*>(spectrum_out),
                                            reinterpret_cast<cufftComplex*>(spectrum_out),
                                            CUFFT_INVERSE)));
#endif

    ScaleComplexKernelHopkins<<<blocks, CUDA_THREADS, 0, stream>>>(spectrum_out, count, scale);
    CHECK(StatusFromCudaLocal(cudaGetLastError()));
    return Status::kSuccess;
}

Status BuildHopkinsKernels(int width,
                           int height,
                           Real period_x,
                           Real period_y,
                           Real focus,
                           const HopkinsSourceConfig& source,
                           const HopkinsProjectorConfig& projector,
                           const HopkinsNumericsConfig& numerics,
                           std::vector<std::vector<Complex>>* kernels,
                           cudaStream_t stream)
{
    if (!kernels || width <= 0 || height <= 0 || projector.wavelength <= 0.0 || projector.NA <= 0.0)
    {
        return Status::kInvalidValue;
    }

    const std::vector<SourceSample> samples = BuildAnnularSourceSamples(source);
    if (samples.empty())
    {
        return Status::kInvalidValue;
    }

    const Real weight_sum =
        std::accumulate(samples.begin(),
                        samples.end(),
                        static_cast<Real>(0.0),
                        [](Real acc, const SourceSample& sample) { return acc + sample.weight; });
    if (weight_sum <= 0.0)
    {
        return Status::kInvalidValue;
    }

    const bool vector_mode = numerics.imaging_mode == HopkinsImagingMode::kVector;
    const Complex zero = make_complex(0.0, 0.0);
    const Complex one = make_complex(1.0, 0.0);
    const Complex minus_one = make_complex(-1.0, 0.0);

    const Real normalized_pitch_x = period_x / (projector.wavelength / projector.NA);
    const Real normalized_pitch_y = period_y / (projector.wavelength / projector.NA);
    const int valid_fx_order = std::max(1, static_cast<int>(std::ceil(2.0 * normalized_pitch_x)));
    const int valid_fy_order = std::max(1, static_cast<int>(std::ceil(2.0 * normalized_pitch_y)));
    const int valid_width = 2 * valid_fx_order + 1;
    const int valid_height = 2 * valid_fy_order + 1;
    const int mode_count = valid_width * valid_height;
    const int sample_count = static_cast<int>(samples.size());

    std::vector<Real> fx(static_cast<size_t>(valid_width), 0.0);
    std::vector<Real> fy(static_cast<size_t>(valid_height), 0.0);
    for (int ix = -valid_fx_order; ix <= valid_fx_order; ++ix)
    {
        fx[static_cast<size_t>(ix + valid_fx_order)] = static_cast<Real>(ix) / normalized_pitch_x;
    }
    for (int iy = -valid_fy_order; iy <= valid_fy_order; ++iy)
    {
        fy[static_cast<size_t>(iy + valid_fy_order)] = static_cast<Real>(iy) / normalized_pitch_y;
    }

    const Real na_sq = projector.NA * projector.NA;
    const Real na_over_m_sq =
        (projector.NA / projector.magnification) * (projector.NA / projector.magnification);
    const Real focus_phase_scale = 2.0 * static_cast<Real>(kPi) / projector.wavelength;
    const Real alpha_scale = projector.NA / projector.index_image;

    std::vector<Complex> pupil_x_matrix(static_cast<size_t>(mode_count) * sample_count, zero);
    std::vector<Complex> pupil_y_matrix;
    std::vector<Complex> pupil_z_matrix;
    if (vector_mode)
    {
        pupil_y_matrix.assign(static_cast<size_t>(mode_count) * sample_count, zero);
        pupil_z_matrix.assign(static_cast<size_t>(mode_count) * sample_count, zero);
    }

    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx)
    {
        const SourceSample& sample = samples[static_cast<size_t>(sample_idx)];
        const Complex weight_scale =
            make_complex(std::sqrt(projector.index_image * sample.weight / weight_sum), 0.0);
        const size_t col_offset = static_cast<size_t>(sample_idx) * mode_count;

        int mode_index = 0;
        for (int iy = 0; iy < valid_height; ++iy)
        {
            for (int ix = 0; ix < valid_width; ++ix, ++mode_index)
            {
                const Real fx_shift = fx[static_cast<size_t>(ix)] + sample.x;
                const Real fy_shift = fy[static_cast<size_t>(iy)] + sample.y;
                const Real rho2 = fx_shift * fx_shift + fy_shift * fy_shift;
                if (rho2 > 1.0)
                {
                    continue;
                }

                const Real ob_num = 1.0 - na_over_m_sq * rho2;
                const Real ob_den = 1.0 - na_sq * rho2;
                if (ob_num <= 0.0 || ob_den <= 0.0)
                {
                    continue;
                }

                const Real rho = std::sqrt(std::max(rho2, static_cast<Real>(0.0)));
                const Real theta = std::atan2(fy_shift, fx_shift);
                const Real obliquity = std::pow(ob_num / ob_den, 0.25);
                const Real aberration = CalculateAberrationFast(projector, rho, theta);
                const Real focus_term = std::sqrt(
                    std::max(projector.index_image * projector.index_image - na_over_m_sq * rho2,
                             static_cast<Real>(0.0)));
                const Real phase = 2.0 * static_cast<Real>(kPi) * aberration +
                                   focus_phase_scale * focus_term * focus;
                const Complex shifted_pupil = ComplexMul(
                    make_complex(obliquity * std::cos(phase), obliquity * std::sin(phase)),
                    weight_scale);

                const size_t dst = col_offset + static_cast<size_t>(mode_index);
                if (!vector_mode)
                {
                    pupil_x_matrix[dst] = shifted_pupil;
                    continue;
                }

                const Complex alpha = make_complex(alpha_scale * fx_shift, 0.0);
                const Complex beta = make_complex(alpha_scale * fy_shift, 0.0);
                const Complex gamma = ComplexSqrt(
                    ComplexSub(one, ComplexAdd(ComplexMul(alpha, alpha), ComplexMul(beta, beta))));
                const Complex denom = ComplexAdd(one, gamma);
                if (ComplexAbs(denom) <= static_cast<Real>(1e-12))
                {
                    continue;
                }

                const Complex inv_denom = ComplexDiv(one, denom);
                const Complex mxx =
                    ComplexSub(one, ComplexMul(ComplexMul(alpha, alpha), inv_denom));
                const Complex myx =
                    ComplexMul(minus_one, ComplexMul(ComplexMul(alpha, beta), inv_denom));
                const Complex mxy = myx;
                const Complex myy = ComplexSub(one, ComplexMul(ComplexMul(beta, beta), inv_denom));
                const Complex mxz = ComplexMul(minus_one, alpha);
                const Complex myz = ComplexMul(minus_one, beta);

                pupil_x_matrix[dst] = ComplexMul(ComplexAdd(ComplexMul(source.polarization_x, mxx),
                                                            ComplexMul(source.polarization_y, myx)),
                                                 shifted_pupil);
                pupil_y_matrix[dst] = ComplexMul(ComplexAdd(ComplexMul(source.polarization_x, mxy),
                                                            ComplexMul(source.polarization_y, myy)),
                                                 shifted_pupil);
                pupil_z_matrix[dst] = ComplexMul(ComplexAdd(ComplexMul(source.polarization_x, mxz),
                                                            ComplexMul(source.polarization_y, myz)),
                                                 shifted_pupil);
            }
        }
    }

    cudaStream_t use_stream = stream ? stream : nullptr;
    ComplexMatrixOps ops(use_stream);
    CHECK(ops.SetStream(use_stream));

    DeviceComplexBuffer d_px;
    DeviceComplexBuffer d_py;
    DeviceComplexBuffer d_pz;
    DeviceComplexBuffer d_tcc;
    DeviceComplexBuffer d_temp;
    CHECK(d_px.Allocate(pupil_x_matrix.size()));
    CHECK(StatusFromCudaLocal(cudaMemcpyAsync(d_px.get(),
                                              pupil_x_matrix.data(),
                                              pupil_x_matrix.size() * sizeof(Complex),
                                              cudaMemcpyHostToDevice,
                                              use_stream)));
    CHECK(d_tcc.Allocate(static_cast<size_t>(mode_count) * mode_count));
    CHECK(ops.MatMul(d_px.get(),
                     d_px.get(),
                     d_tcc.get(),
                     mode_count,
                     mode_count,
                     sample_count,
                     CUBLAS_OP_N,
                     CUBLAS_OP_C));

    if (vector_mode)
    {
        CHECK(d_py.Allocate(pupil_y_matrix.size()));
        CHECK(d_pz.Allocate(pupil_z_matrix.size()));
        CHECK(d_temp.Allocate(static_cast<size_t>(mode_count) * mode_count));

        CHECK(StatusFromCudaLocal(cudaMemcpyAsync(d_py.get(),
                                                  pupil_y_matrix.data(),
                                                  pupil_y_matrix.size() * sizeof(Complex),
                                                  cudaMemcpyHostToDevice,
                                                  use_stream)));
        CHECK(StatusFromCudaLocal(cudaMemcpyAsync(d_pz.get(),
                                                  pupil_z_matrix.data(),
                                                  pupil_z_matrix.size() * sizeof(Complex),
                                                  cudaMemcpyHostToDevice,
                                                  use_stream)));

        CHECK(ops.MatMul(d_py.get(),
                         d_py.get(),
                         d_temp.get(),
                         mode_count,
                         mode_count,
                         sample_count,
                         CUBLAS_OP_N,
                         CUBLAS_OP_C));
        CHECK(AddInPlace(ops, d_tcc.get(), d_temp.get(), mode_count));

        CHECK(ops.MatMul(d_pz.get(),
                         d_pz.get(),
                         d_temp.get(),
                         mode_count,
                         mode_count,
                         sample_count,
                         CUBLAS_OP_N,
                         CUBLAS_OP_C));
        CHECK(AddInPlace(ops, d_tcc.get(), d_temp.get(), mode_count));
    }

    std::vector<Real> eigenvalues;
    std::vector<Complex> eigenvectors;
    CHECK(HermitianEigenDecompositionDevice(
        d_tcc.get(), mode_count, &eigenvalues, &eigenvectors, use_stream));

    int kernel_count = 0;
    if (numerics.use_threshold)
    {
        const Real total_energy = std::accumulate(
            eigenvalues.begin(),
            eigenvalues.end(),
            static_cast<Real>(0.0),
            [](Real acc, Real value) { return acc + std::max(value, static_cast<Real>(0.0)); });
        Real running_energy = 0.0;
        for (int idx = mode_count - 1; idx >= 0; --idx)
        {
            const Real value =
                std::max(eigenvalues[static_cast<size_t>(idx)], static_cast<Real>(0.0));
            if (value <= static_cast<Real>(1e-12))
            {
                continue;
            }
            running_energy += value;
            ++kernel_count;
            if (total_energy > 0.0 && running_energy >= numerics.threshold * total_energy)
            {
                break;
            }
        }
    }
    else
    {
        for (int idx = mode_count - 1; idx >= 0 && kernel_count < numerics.max_kernels; --idx)
        {
            if (eigenvalues[static_cast<size_t>(idx)] > static_cast<Real>(1e-12))
            {
                ++kernel_count;
            }
        }
    }

    if (kernel_count <= 0)
    {
        return Status::kInvalidValue;
    }

    kernels->clear();
    kernels->reserve(static_cast<size_t>(kernel_count));

    const int start_x = static_cast<int>((width - valid_width + 1) / 2);
    const int start_y = static_cast<int>((height - valid_height + 1) / 2);

    for (int keep = 0; keep < kernel_count; ++keep)
    {
        const int eig_index = mode_count - 1 - keep;
        const Real lambda =
            std::max(eigenvalues[static_cast<size_t>(eig_index)], static_cast<Real>(0.0));
        if (lambda <= static_cast<Real>(1e-12))
        {
            continue;
        }

        std::vector<Complex> centered(static_cast<size_t>(width) * height, zero);
        const Real root_lambda = std::sqrt(lambda);
        const Complex scale = make_complex(root_lambda, 0.0);

        for (int iy = 0; iy < valid_height; ++iy)
        {
            for (int ix = 0; ix < valid_width; ++ix)
            {
                const int mode_idx = iy * valid_width + ix;
                const Complex eigvec =
                    eigenvectors[static_cast<size_t>(eig_index) * mode_count + mode_idx];
                centered[LinearIndex(start_x + ix, start_y + iy, width)] =
                    ComplexMul(eigvec, scale);
            }
        }

        std::vector<Complex> shifted;
        FFTShiftHost(centered, &shifted, width, height);
        kernels->push_back(std::move(shifted));
    }

    return kernels->empty() ? Status::kInvalidValue : Status::kSuccess;
}

} // namespace

void ReleaseHopkinsKernelCache(HopkinsKernelCache* kernel_cache)
{
    if (!kernel_cache)
    {
        return;
    }

    auto it = g_device_kernel_banks.find(reinterpret_cast<uintptr_t>(kernel_cache));
    if (it != g_device_kernel_banks.end())
    {
        it->second.Clear();
        g_device_kernel_banks.erase(it);
    }
}

Status CalculateHopkinsImageFromFields(const Complex* ex_field,
                                       const Complex* ey_field,
                                       const Complex* ez_field,
                                       int64_t width,
                                       int64_t height,
                                       Real period_x,
                                       Real period_y,
                                       const HopkinsSourceConfig& source,
                                       const HopkinsProjectorConfig& projector,
                                       const HopkinsNumericsConfig& numerics,
                                       HopkinsImageResult* result,
                                       HopkinsKernelCache* kernel_cache,
                                       cudaStream_t stream)
{
    if (!ex_field || !ey_field || !ez_field || !result || width <= 0 || height <= 0)
    {
        return Status::kInvalidValue;
    }
    if (projector.wavelength <= 0.0 || projector.NA <= 0.0 || projector.magnification <= 0.0)
    {
        return Status::kInvalidValue;
    }

    const std::vector<Real> focus_range = NormalizedFocusRange(projector);
    const int64_t count = width * height;
    const int blocks = static_cast<int>((count + CUDA_THREADS - 1) / CUDA_THREADS);
    cudaStream_t use_stream = stream ? stream : nullptr;

    const bool cache_hit = kernel_cache && CacheMatches(*kernel_cache,
                                                        width,
                                                        height,
                                                        period_x,
                                                        period_y,
                                                        source,
                                                        projector,
                                                        numerics,
                                                        focus_range);
    if (kernel_cache && !cache_hit)
    {
        ReleaseHopkinsKernelCache(kernel_cache);
        ResetKernelCache(kernel_cache,
                         width,
                         height,
                         period_x,
                         period_y,
                         source,
                         projector,
                         numerics,
                         focus_range);
    }

    const Real normalized_pitch_x = period_x / (projector.wavelength / projector.NA);
    const Real normalized_pitch_y = period_y / (projector.wavelength / projector.NA);
    const Real inverse_fft_scale =
        (normalized_pitch_x * normalized_pitch_y) / static_cast<Real>(count);
    const Real image_field_scale = (1.0 / normalized_pitch_x) * (1.0 / normalized_pitch_y);

    cufftHandle plan = 0;
#if USE_DOUBLE_PRECISION
    CHECK(StatusFromCufftLocal(
        cufftPlan2d(&plan, static_cast<int>(height), static_cast<int>(width), CUFFT_Z2Z)));
#else
    CHECK(StatusFromCufftLocal(
        cufftPlan2d(&plan, static_cast<int>(height), static_cast<int>(width), CUFFT_C2C)));
#endif
    CHECK(StatusFromCufftLocal(cufftSetStream(plan, use_stream)));

    DeviceComplexBuffer spectrum_x;
    DeviceComplexBuffer spectrum_y;
    DeviceComplexBuffer spectrum_z;
    DeviceComplexBuffer kernel_device;
    DeviceComplexBuffer work;
    DeviceComplexBuffer fft_shifted;
    DeviceRealBufferLocal intensity_device;

    CHECK(spectrum_x.Allocate(static_cast<size_t>(count)));
    CHECK(spectrum_y.Allocate(static_cast<size_t>(count)));
    CHECK(spectrum_z.Allocate(static_cast<size_t>(count)));
    CHECK(fft_shifted.Allocate(static_cast<size_t>(count)));
    CHECK(intensity_device.Allocate(static_cast<size_t>(count)));

    CHECK(ComputeSpectrumFromField(ex_field,
                                   static_cast<int>(width),
                                   static_cast<int>(height),
                                   inverse_fft_scale,
                                   plan,
                                   use_stream,
                                   spectrum_x.get(),
                                   fft_shifted.get()));
    CHECK(ComputeSpectrumFromField(ey_field,
                                   static_cast<int>(width),
                                   static_cast<int>(height),
                                   inverse_fft_scale,
                                   plan,
                                   use_stream,
                                   spectrum_y.get(),
                                   fft_shifted.get()));
    CHECK(ComputeSpectrumFromField(ez_field,
                                   static_cast<int>(width),
                                   static_cast<int>(height),
                                   inverse_fft_scale,
                                   plan,
                                   use_stream,
                                   spectrum_z.get(),
                                   fft_shifted.get()));

    result->width = width;
    result->height = height;
    result->focus_count = static_cast<int64_t>(focus_range.size());
    result->intensity.assign(static_cast<size_t>(result->focus_count * count), 0.0);
    result->image_x = MakeLinearAxis(-period_x / 2.0, period_x / 2.0, width);
    result->image_y = MakeLinearAxis(-period_y / 2.0, period_y / 2.0, height);
    result->image_z = focus_range;

    std::vector<Real> host_intensity(static_cast<size_t>(count), 0.0);
    std::vector<Real> shifted_intensity;

    for (size_t focus_idx = 0; focus_idx < focus_range.size(); ++focus_idx)
    {
        std::vector<std::vector<Complex>> kernels_local;
        const std::vector<std::vector<Complex>>* kernels = nullptr;
        if (cache_hit && kernel_cache)
        {
            kernels = &kernel_cache->kernels_by_focus[focus_idx];
        }
        else
        {
            CHECK(BuildHopkinsKernels(static_cast<int>(width),
                                      static_cast<int>(height),
                                      period_x,
                                      period_y,
                                      focus_range[focus_idx],
                                      source,
                                      projector,
                                      numerics,
                                      &kernels_local,
                                      use_stream));
            if (kernel_cache)
            {
                kernel_cache->kernels_by_focus[focus_idx] = std::move(kernels_local);
                kernels = &kernel_cache->kernels_by_focus[focus_idx];
            }
            else
            {
                kernels = &kernels_local;
            }
        }

        CHECK(StatusFromCudaLocal(cudaMemsetAsync(
            intensity_device.get(), 0, static_cast<size_t>(count) * sizeof(Real), use_stream)));

        constexpr int spectrum_count = 3;
        const Complex* device_kernel_base = nullptr;
        int device_kernel_count = 0;
        if (kernel_cache)
        {
            CHECK(EnsureDeviceKernelsForFocus(kernel_cache,
                                              focus_idx,
                                              count,
                                              use_stream,
                                              &device_kernel_base,
                                              &device_kernel_count));
        }

        if (!device_kernel_base)
        {
            device_kernel_count = static_cast<int>(kernels->size());
            if (device_kernel_count <= 0)
            {
                CHECK(Status::kInvalidValue);
            }

            std::vector<Complex> flattened(static_cast<size_t>(device_kernel_count) * count);
            for (int kernel_idx = 0; kernel_idx < device_kernel_count; ++kernel_idx)
            {
                std::copy((*kernels)[static_cast<size_t>(kernel_idx)].begin(),
                          (*kernels)[static_cast<size_t>(kernel_idx)].end(),
                          flattened.begin() + static_cast<size_t>(kernel_idx) * count);
            }
            CHECK(kernel_device.Allocate(flattened.size()));
            CHECK(StatusFromCudaLocal(cudaMemcpyAsync(kernel_device.get(),
                                                      flattened.data(),
                                                      flattened.size() * sizeof(Complex),
                                                      cudaMemcpyHostToDevice,
                                                      use_stream)));
            device_kernel_base = kernel_device.get();
        }

        const int batch_count = device_kernel_count * spectrum_count;
        const size_t batched_elements =
            static_cast<size_t>(batch_count) * static_cast<size_t>(count);
        CHECK(work.Allocate(batched_elements));

        cufftHandle batched_plan = 0;
        int dims[2] = {static_cast<int>(height), static_cast<int>(width)};
#if USE_DOUBLE_PRECISION
        CHECK(StatusFromCufftLocal(cufftPlanMany(&batched_plan,
                                                 2,
                                                 dims,
                                                 dims,
                                                 1,
                                                 static_cast<int>(count),
                                                 dims,
                                                 1,
                                                 static_cast<int>(count),
                                                 CUFFT_Z2Z,
                                                 batch_count)));
#else
        CHECK(StatusFromCufftLocal(cufftPlanMany(&batched_plan,
                                                 2,
                                                 dims,
                                                 dims,
                                                 1,
                                                 static_cast<int>(count),
                                                 dims,
                                                 1,
                                                 static_cast<int>(count),
                                                 CUFFT_C2C,
                                                 batch_count)));
#endif
        CHECK(StatusFromCufftLocal(cufftSetStream(batched_plan, use_stream)));

        const int64_t total_batched = static_cast<int64_t>(batch_count) * count;
        const int batched_blocks =
            static_cast<int>((total_batched + CUDA_THREADS - 1) / CUDA_THREADS);
        PrepareBatchedProductsKernel<<<batched_blocks, CUDA_THREADS, 0, use_stream>>>(
            device_kernel_base,
            spectrum_x.get(),
            spectrum_y.get(),
            spectrum_z.get(),
            work.get(),
            count,
            device_kernel_count);
        CHECK(StatusFromCudaLocal(cudaGetLastError()));

#if USE_DOUBLE_PRECISION
        CHECK(StatusFromCufftLocal(cufftExecZ2Z(batched_plan,
                                                reinterpret_cast<cufftDoubleComplex*>(work.get()),
                                                reinterpret_cast<cufftDoubleComplex*>(work.get()),
                                                CUFFT_FORWARD)));
#else
        CHECK(StatusFromCufftLocal(cufftExecC2C(batched_plan,
                                                reinterpret_cast<cufftComplex*>(work.get()),
                                                reinterpret_cast<cufftComplex*>(work.get()),
                                                CUFFT_FORWARD)));
#endif

        AccumulateBatchedIntensityKernel<<<blocks, CUDA_THREADS, 0, use_stream>>>(
            work.get(), intensity_device.get(), count, batch_count, image_field_scale);
        CHECK(StatusFromCudaLocal(cudaGetLastError()));
        CHECK(StatusFromCufftLocal(cufftDestroy(batched_plan)));

        CHECK(StatusFromCudaLocal(cudaMemcpyAsync(host_intensity.data(),
                                                  intensity_device.get(),
                                                  static_cast<size_t>(count) * sizeof(Real),
                                                  cudaMemcpyDeviceToHost,
                                                  use_stream)));
        CHECK(StatusFromCudaLocal(cudaStreamSynchronize(use_stream)));

        FFTShiftHost(
            host_intensity, &shifted_intensity, static_cast<int>(width), static_cast<int>(height));
        std::copy(shifted_intensity.begin(),
                  shifted_intensity.end(),
                  result->intensity.begin() + static_cast<int64_t>(focus_idx) * count);
    }

    CHECK(StatusFromCufftLocal(cufftDestroy(plan)));
    if (kernel_cache)
    {
        kernel_cache->valid = true;
    }
    return Status::kSuccess;
}

Status CalculateHopkinsImageFromRCWA(const RCWA& rcwa,
                                     const Real* x_axis,
                                     int64_t x_count,
                                     const Real* y_axis,
                                     int64_t y_count,
                                     Real period_x,
                                     Real period_y,
                                     const HopkinsSourceConfig& source,
                                     const HopkinsProjectorConfig& projector,
                                     const HopkinsNumericsConfig& numerics,
                                     HopkinsImageResult* result,
                                     HopkinsKernelCache* kernel_cache,
                                     cudaStream_t stream)
{
    Complex* ex = nullptr;
    Complex* ey = nullptr;
    Complex* ez = nullptr;

    Status status = rcwa.FieldXY(x_axis, x_count, y_axis, y_count, &ex, &ey, &ez, stream);
    if (status != Status::kSuccess)
    {
        return status;
    }

    status = CalculateHopkinsImageFromFields(ex,
                                             ey,
                                             ez,
                                             x_count,
                                             y_count,
                                             period_x,
                                             period_y,
                                             source,
                                             projector,
                                             numerics,
                                             result,
                                             kernel_cache,
                                             stream);

    rcwa.FreeDeviceTensor(ex);
    rcwa.FreeDeviceTensor(ey);
    rcwa.FreeDeviceTensor(ez);
    return status;
}
