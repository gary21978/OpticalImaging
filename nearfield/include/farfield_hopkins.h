#pragma once

#include <cstdint>
#include <vector>

#include "common_utils.h"
#include "rcwa.h"

struct HopkinsSourceConfig
{
    int point_count = 81;
    Real sigma_out = 0.7;
    Real sigma_in = 0.1;
    Complex polarization_x = make_complex(1.0, 0.0);
    Complex polarization_y = make_complex(0.0, 0.0);
};

struct HopkinsProjectorConfig
{
    Real wavelength = 365.0;
    Real magnification = 100.0;
    Real NA = 0.9;
    Real index_image = 1.0;
    std::vector<Real> focus_range{0.0};
    std::vector<Real> aberration_zernike = std::vector<Real>(37, 0.0);
};

enum class HopkinsImagingMode
{
    kScalar = 0,
    kVector = 1,
};

struct HopkinsNumericsConfig
{
    HopkinsImagingMode imaging_mode = HopkinsImagingMode::kScalar;
    // Mirrors the Python implementation's Hopkins order truncation.
    int max_kernels = 100;
    bool use_threshold = false;
    Real threshold = 0.95;
};

struct HopkinsImageResult
{
    int64_t width = 0;
    int64_t height = 0;
    int64_t focus_count = 0;
    std::vector<Real> intensity; // focus-major, each slice uses row-major [y * width + x]
    std::vector<Real> image_x;
    std::vector<Real> image_y;
    std::vector<Real> image_z;
};

struct HopkinsKernelCache
{
    bool valid = false;
    int64_t width = 0;
    int64_t height = 0;
    Real period_x = 0.0;
    Real period_y = 0.0;
    HopkinsSourceConfig source{};
    HopkinsProjectorConfig projector{};
    HopkinsNumericsConfig numerics{};
    std::vector<std::vector<std::vector<Complex>>> kernels_by_focus;

    void Clear()
    {
        valid = false;
        width = 0;
        height = 0;
        period_x = 0.0;
        period_y = 0.0;
        kernels_by_focus.clear();
    }
};

void ReleaseHopkinsKernelCache(HopkinsKernelCache* kernel_cache);

// Hopkins imaging ported from the Python farfield pipeline, supporting scalar and vector modes.
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
                                       cudaStream_t stream = nullptr);

inline Status CalculateHopkinsImageFromFields(const Complex* ex_field,
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
                                              cudaStream_t stream = nullptr)
{
    return CalculateHopkinsImageFromFields(ex_field,
                                           ey_field,
                                           ez_field,
                                           width,
                                           height,
                                           period_x,
                                           period_y,
                                           source,
                                           projector,
                                           numerics,
                                           result,
                                           nullptr,
                                           stream);
}

// Convenience overload that samples the RCWA field first and then runs Hopkins imaging.
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
                                     cudaStream_t stream = nullptr);

inline Status CalculateHopkinsImageFromRCWA(const RCWA& rcwa,
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
                                            cudaStream_t stream = nullptr)
{
    return CalculateHopkinsImageFromRCWA(rcwa,
                                         x_axis,
                                         x_count,
                                         y_axis,
                                         y_count,
                                         period_x,
                                         period_y,
                                         source,
                                         projector,
                                         numerics,
                                         result,
                                         nullptr,
                                         stream);
}
