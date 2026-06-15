#pragma once

#include <cstdint>
#include <vector>

#include "common_utils.h"
#include "rcwa.h"

enum class AbbeApertureShape
{
    kAnnular = 0,
    kDipoleCirc = 1,
    kQuasar = 2,
    kRect = 3,
};

enum class AbbePolarizationMode
{
    kJonesVector = 0,
    kX = 1,
    kY = 2,
    kLine = 3,
    kCircular = 4,
    kAngleDegrees = 5,
};

struct AbbeSourceConfig
{
    AbbeApertureShape shape = AbbeApertureShape::kAnnular;
    Real sigma_out = 0.9;
    Real sigma_in = 0.15;
    Real open_angle = 0.0;
    Real rotate_angle = 0.0;
    Real rect_x = 0.0;
    Real rect_y = 0.0;
    AbbePolarizationMode polarization_mode = AbbePolarizationMode::kX;
    Complex polarization_x = make_complex(1.0, 0.0);
    Complex polarization_y = make_complex(0.0, 0.0);
    Real polarization_angle_deg = 0.0;
};

struct AbbeProjectorConfig
{
    Real wavelength = 365.0;
    Real magnification = 1.0;
    Real index_image = 1.0;
    AbbeApertureShape shape = AbbeApertureShape::kAnnular;
    Real sigma_out = 0.9;
    Real sigma_in = 0.0;
    Real open_angle = 0.0;
    Real rotate_angle = 0.0;
    Real rect_x = 0.0;
    Real rect_y = 0.0;
    std::vector<Real> focus_range{0.0};
    std::vector<Real> aberration_zernike = std::vector<Real>(37, 0.0);
};

enum class AbbeImagingMode
{
    kScalar = 0,
    kVector = 1,
};

struct AbbeNumericsConfig
{
    AbbeImagingMode imaging_mode = AbbeImagingMode::kVector;
    std::vector<Real> object_z_positions{0.0};
};

struct AbbeImageResult
{
    int64_t width = 0;
    int64_t height = 0;
    int64_t focus_count = 0;
    int64_t object_plane_count = 0;
    std::vector<Real> intensity; // [z][focus][y][x]
    std::vector<Real> image_x;
    std::vector<Real> image_y;
};

struct AbbeKernelCache
{
    bool valid = false;

    void Clear() { valid = false; }
};

void ReleaseAbbeKernelCache(AbbeKernelCache* kernel_cache);

// Compatibility entrypoint. The RCWA-aligned implementation is centered on
// `CalculateAbbeImageFromRCWA`; this overload only performs direct field
// intensity evaluation and currently supports a single object plane and
// single focus slice.
Status CalculateAbbeImageFromFields(const Complex* ex_field,
                                    const Complex* ey_field,
                                    const Complex* ez_field,
                                    int64_t width,
                                    int64_t height,
                                    Real period_x,
                                    Real period_y,
                                    const AbbeSourceConfig& source,
                                    const AbbeProjectorConfig& projector,
                                    const AbbeNumericsConfig& numerics,
                                    AbbeImageResult* result,
                                    AbbeKernelCache* kernel_cache,
                                    cudaStream_t stream = nullptr);

inline Status CalculateAbbeImageFromFields(const Complex* ex_field,
                                           const Complex* ey_field,
                                           const Complex* ez_field,
                                           int64_t width,
                                           int64_t height,
                                           Real period_x,
                                           Real period_y,
                                           const AbbeSourceConfig& source,
                                           const AbbeProjectorConfig& projector,
                                           const AbbeNumericsConfig& numerics,
                                           AbbeImageResult* result,
                                           cudaStream_t stream = nullptr)
{
    return CalculateAbbeImageFromFields(ex_field,
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

Status CalculateAbbeImageFromRCWA(const RCWA& rcwa,
                                  const Real* x_axis,
                                  int64_t x_count,
                                  const Real* y_axis,
                                  int64_t y_count,
                                  Real period_x,
                                  Real period_y,
                                  const AbbeSourceConfig& source,
                                  const AbbeProjectorConfig& projector,
                                  const AbbeNumericsConfig& numerics,
                                  AbbeImageResult* result,
                                  AbbeKernelCache* kernel_cache,
                                  cudaStream_t stream = nullptr);

inline Status CalculateAbbeImageFromRCWA(const RCWA& rcwa,
                                         const Real* x_axis,
                                         int64_t x_count,
                                         const Real* y_axis,
                                         int64_t y_count,
                                         Real period_x,
                                         Real period_y,
                                         const AbbeSourceConfig& source,
                                         const AbbeProjectorConfig& projector,
                                         const AbbeNumericsConfig& numerics,
                                         AbbeImageResult* result,
                                         cudaStream_t stream = nullptr)
{
    return CalculateAbbeImageFromRCWA(rcwa,
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
