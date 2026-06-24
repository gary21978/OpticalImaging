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

struct AbbeNumericsConfig
{
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

Status CalculateAbbeImageFromRCWA(const RCWA& rcwa,
                                  const Real* x_axis,
                                  int64_t x_count,
                                  const Real* y_axis,
                                  int64_t y_count,
                                  const AbbeSourceConfig& source,
                                  const AbbeProjectorConfig& projector,
                                  const AbbeNumericsConfig& numerics,
                                  AbbeImageResult* result,
                                  cudaStream_t stream = nullptr);
