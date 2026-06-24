#include "simulation_backend.h"

#include "abbe_imager.h"
#include "material_database.h"
#include "png_mask.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

struct DeviceBuffer
{
    Complex* ptr = nullptr;
    ~DeviceBuffer()
    {
        if (ptr)
            cudaFree(ptr);
    }
};

bool Check(Status status, const char* step, SimulationResult* result)
{
    if (status == Status::kSuccess)
        return true;
    result->error = std::string(step) + ": " + StatusToString(status);
    return false;
}

bool CheckCuda(cudaError_t status, const char* step, SimulationResult* result)
{
    if (status == cudaSuccess)
        return true;
    result->error = std::string(step) + ": " + cudaGetErrorString(status);
    return false;
}

std::vector<Real> BuildAxis(int count, Real length)
{
    std::vector<Real> axis(static_cast<size_t>(count));
    const Real step = length / static_cast<Real>(count);
    for (int i = 0; i < count; ++i)
        axis[static_cast<size_t>(i)] = step * static_cast<Real>(i + 0.5);
    return axis;
}

std::vector<Real> BuildLayerGeometry(const SimulationConfig& c, const LayerConfig& layer)
{
    if (layer.geometry_mode == "uniform")
        return std::vector<Real>(static_cast<size_t>(c.geometry_x) * c.geometry_y, 1.0);
    if (layer.geometry_mode == "png")
    {
        std::vector<double> png_mask;
        std::string error;
        if (LoadBinaryPngMask(layer.png_path, c.geometry_x, c.geometry_y,
                              &png_mask, &error))
            return std::vector<Real>(png_mask.begin(), png_mask.end());
    }
    return std::vector<Real>(static_cast<size_t>(c.geometry_x) * c.geometry_y, 1.0);
}

std::string NearfieldCacheKey(const SimulationConfig& c)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << c.wavelength_nm << '|'
        << c.period_x_nm << '|' << c.period_y_nm << '|'
        << c.order_x << '|' << c.order_y << '|'
        << c.geometry_x << '|' << c.geometry_y << '|'
        << c.image_width << '|' << c.image_height << '|'
        << c.input_n_real << '|' << c.input_n_imag << '|'
        << c.substrate_n_real << '|' << c.substrate_n_imag << '|'
        << c.substrate_material << '|'
        << c.precision << '|'
        << c.approximation_order;
    for (const LayerConfig& layer : c.layers)
    {
        out << '|' << layer.name << '|' << layer.material << '|' << layer.thickness_nm
            << '|' << layer.n_real << '|' << layer.n_imag << '|'
            << layer.geometry_mode << '|' << layer.png_path;
        for (const ShapeConfig& shape : layer.shapes)
            out << '|' << shape.type << '|' << shape.center_x << '|' << shape.center_y
                << '|' << shape.size_x << '|' << shape.size_y;
    }
    const std::string text = out.str();
    const size_t hash = std::hash<std::string>{}(text);
    std::ostringstream name;
    name << "simulok_nearfield_" << std::hex << hash << ".bin";
    return (std::filesystem::temp_directory_path() / name.str()).string();
}

bool LoadNearfieldCache(const std::string& path, int width, int height, SimulationResult* result)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    int cached_w = 0;
    int cached_h = 0;
    double peak = 0.0;
    uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&cached_w), sizeof(cached_w));
    in.read(reinterpret_cast<char*>(&cached_h), sizeof(cached_h));
    in.read(reinterpret_cast<char*>(&peak), sizeof(peak));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in || cached_w != width || cached_h != height)
        return false;
    result->nearfield_peak_intensity = peak;
    result->nearfield_rgba.resize(count);
    in.read(reinterpret_cast<char*>(result->nearfield_rgba.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(in);
}

void SaveNearfieldCache(const std::string& path, const SimulationResult& result)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return;
    const uint64_t count = result.nearfield_rgba.size();
    out.write(reinterpret_cast<const char*>(&result.width), sizeof(result.width));
    out.write(reinterpret_cast<const char*>(&result.height), sizeof(result.height));
    out.write(reinterpret_cast<const char*>(&result.nearfield_peak_intensity), sizeof(result.nearfield_peak_intensity));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(result.nearfield_rgba.data()), static_cast<std::streamsize>(count * sizeof(float)));
}

void GrayColor(double x, float* rgba)
{
    x = std::clamp(x, 0.0, 1.0);
    const float g = static_cast<float>(x);
    rgba[0] = g;
    rgba[1] = g;
    rgba[2] = g;
    rgba[3] = 1.0f;
}

void BuildEpsMap(const std::vector<Real>& mask, Complex n, std::vector<Complex>* map)
{
    const Complex one = make_complex(1.0, 0.0);
    const Complex delta = ComplexSub(ComplexMul(n, n), one);
    map->resize(mask.size());
    for (size_t i = 0; i < mask.size(); ++i)
        (*map)[i] = ComplexAdd(one, ComplexMul(make_complex(mask[i], 0.0), delta));
}

} // namespace

SimulationResult RunSimulation(const SimulationConfig& c)
{
    SimulationResult result;
    const auto start = Clock::now();
    std::string validation;
    if (!ValidateSimulationConfig(c, &validation))
    {
        result.error = validation;
        return result;
    }

    int device_count = 0;
    if (!CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", &result) || device_count == 0)
    {
        if (result.error.empty())
            result.error = "No CUDA device found.";
        return result;
    }

    std::string material_error;
    LoadSharedMaterialDatabase("resources/tztek_materials.csv", &material_error);
    RCWA rcwa(1.0 / c.wavelength_nm, c.order_x, c.order_y, c.period_x_nm, c.period_y_nm);
    rcwa.SetHighOrder(c.approximation_order == 18);
    const Complex input_n = make_complex(c.input_n_real, c.input_n_imag);
    double substrate_real = c.substrate_n_real;
    double substrate_imag = c.substrate_n_imag;
    if (c.substrate_material != "custom")
        SharedMaterialDatabase().Lookup(c.substrate_material, c.wavelength_nm, &substrate_real, &substrate_imag);
    const Complex substrate_n = make_complex(substrate_real, substrate_imag);
    if (!Check(rcwa.AddInputLayer(ComplexMul(input_n, input_n)), "AddInputLayer", &result) ||
        !Check(rcwa.AddOutputLayer(ComplexMul(substrate_n, substrate_n)), "AddOutputLayer", &result) ||
        !Check(rcwa.SetIncidentAngles(0.0,
                                      0.0,
                                      "input"),
               "SetIncidentAngles",
               &result))
        return result;

    DeviceBuffer d_material;
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_material.ptr),
                              static_cast<size_t>(c.geometry_x) * c.geometry_y * sizeof(Complex)),
                   "cudaMalloc(material)",
                   &result))
        return result;

    std::vector<Complex> eps_map;
    for (const LayerConfig& layer : c.layers)
    {
        double layer_real = layer.n_real;
        double layer_imag = layer.n_imag;
        if (layer.material != "custom" &&
            !SharedMaterialDatabase().Lookup(layer.material, c.wavelength_nm, &layer_real, &layer_imag))
        {
            result.error = "Material " + layer.material + " has no data at " +
                           std::to_string(c.wavelength_nm) + " nm.";
            return result;
        }
        const Complex n = make_complex(layer_real, layer_imag);
        const bool homogeneous = layer.geometry_mode == "uniform" || !layer.patterned;
        if (layer.geometry_mode == "png" && !std::filesystem::exists(layer.png_path))
        {
            result.error = "PNG mask not found: " + layer.png_path;
            return result;
        }
        if (homogeneous)
        {
            if (!Check(rcwa.AddLayerHomogeneous(layer.thickness_nm, ComplexMul(n, n)),
                       "AddLayerHomogeneous",
                       &result))
                return result;
            continue;
        }
        const std::vector<Real> geometry = BuildLayerGeometry(c, layer);
        BuildEpsMap(geometry, n, &eps_map);
        if (!CheckCuda(cudaMemcpy(d_material.ptr,
                                  eps_map.data(),
                                  eps_map.size() * sizeof(Complex),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(material)",
                       &result))
            return result;
        Complex* eps_conv = nullptr;
        if (!Check(rcwa.MaterialConv(d_material.ptr, c.geometry_x, c.geometry_y, &eps_conv),
                   "MaterialConv",
                   &result))
            return result;
        const Status add_status = rcwa.AddLayer(layer.thickness_nm, eps_conv);
        rcwa.FreeDeviceTensor(eps_conv);
        if (!Check(add_status, "AddLayer", &result))
            return result;
    }

    if (!Check(rcwa.SolveGlobalSMatrix(), "SolveGlobalSMatrix", &result))
        return result;

    Complex source_ex = make_complex(1.0, 0.0);
    Complex source_ey = make_complex(0.0, 0.0);
    if (c.polarization == "y")
    {
        source_ex = make_complex(0.0, 0.0);
        source_ey = make_complex(1.0, 0.0);
    }
    else if (c.polarization == "circular")
    {
        source_ex = make_complex(0.70710678, 0.0);
        source_ey = make_complex(0.0, 0.70710678);
    }
    else if (c.polarization == "angle")
    {
        const double a = c.polarization_angle_deg * M_PI / 180.0;
        source_ex = make_complex(std::cos(a), 0.0);
        source_ey = make_complex(std::sin(a), 0.0);
    }
    if (!Check(rcwa.SourcePlanewave(source_ex, source_ey), "SourcePlanewave", &result))
        return result;

    const std::vector<Real> xs = BuildAxis(c.image_width, c.period_x_nm);
    const std::vector<Real> ys = BuildAxis(c.image_height, c.period_y_nm);
    AbbeSourceConfig source;
    source.shape = c.source_shape == "dipole" ? AbbeApertureShape::kDipoleCirc :
                   c.source_shape == "quasar" ? AbbeApertureShape::kQuasar :
                   c.source_shape == "rect" ? AbbeApertureShape::kRect :
                   AbbeApertureShape::kAnnular;
    source.sigma_out = c.sigma_outer;
    source.sigma_in = c.sigma_inner;
    source.open_angle = c.source_open_angle_deg * M_PI / 180.0;
    source.rotate_angle = c.source_rotate_angle_deg * M_PI / 180.0;
    source.rect_x = c.source_rect_x;
    source.rect_y = c.source_rect_y;
    source.polarization_mode = c.polarization == "y" ? AbbePolarizationMode::kY :
                               c.polarization == "circular" ? AbbePolarizationMode::kCircular :
                               c.polarization == "angle" ? AbbePolarizationMode::kAngleDegrees :
                               AbbePolarizationMode::kX;
    source.polarization_angle_deg = c.polarization_angle_deg;
    AbbeProjectorConfig projector;
    projector.wavelength = c.wavelength_nm;
    projector.shape = source.shape;
    projector.sigma_out = c.sigma_outer;
    projector.sigma_in = c.sigma_inner;
    projector.open_angle = source.open_angle;
    projector.rotate_angle = source.rotate_angle;
    projector.rect_x = source.rect_x;
    projector.rect_y = source.rect_y;
    projector.aberration_zernike.assign(c.zernike.begin(), c.zernike.end());
    projector.focus_range.assign(c.focus_nm.begin(), c.focus_nm.end());
    AbbeImager imager(source, projector, {});
    if (!Check(imager.ComputeFromRCWA(rcwa,
                                      xs.data(),
                                      c.image_width,
                                      ys.data(),
                                      c.image_height),
               "ComputeFromRCWA",
               &result) ||
        !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", &result))
        return result;

    result.width = c.image_width;
    result.height = c.image_height;
    result.focus_count = static_cast<int>(c.focus_nm.size());
    result.focus_nm = c.focus_nm;
    const size_t slice_pixels = static_cast<size_t>(result.width) * result.height;
    result.farfield_rgba.resize(slice_pixels * 4 * result.focus_count);
    Real peak = 0.0;
    for (int focus_index = 0; focus_index < result.focus_count; ++focus_index)
    {
        const std::vector<Real> intensity = imager.GetFocusSlice(focus_index);
        for (Real value : intensity)
            peak = std::max(peak, value);
    }
    result.peak_intensity = peak;
    for (int focus_index = 0; focus_index < result.focus_count; ++focus_index)
    {
        const std::vector<Real> intensity = imager.GetFocusSlice(focus_index);
        Real focus_peak = 0.0;
        for (Real value : intensity)
            focus_peak = std::max(focus_peak, value);
        const double scale = focus_peak > 0.0 ? 1.0 / focus_peak : 1.0;
        float* destination = result.farfield_rgba.data() +
            static_cast<size_t>(focus_index) * slice_pixels * 4;
        for (size_t i = 0; i < intensity.size(); ++i)
            GrayColor(std::sqrt(std::max(0.0, static_cast<double>(intensity[i]) * scale)),
                      destination + i * 4);
    }

    const std::string nearfield_cache_path = NearfieldCacheKey(c);
    if (!LoadNearfieldCache(nearfield_cache_path, result.width, result.height, &result))
    {
        Complex* ex = nullptr;
        Complex* ey = nullptr;
        Complex* ez = nullptr;
        if (!Check(rcwa.FieldXY(xs.data(), c.image_width, ys.data(), c.image_height, &ex, &ey, &ez),
                   "FieldXY",
                   &result))
            return result;
        std::vector<Complex> host_ex(static_cast<size_t>(c.image_width) * c.image_height);
        std::vector<Complex> host_ey(static_cast<size_t>(c.image_width) * c.image_height);
        std::vector<Complex> host_ez(static_cast<size_t>(c.image_width) * c.image_height);
        const size_t field_bytes = host_ex.size() * sizeof(Complex);
        if (!CheckCuda(cudaMemcpy(host_ex.data(), ex, field_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(Ex)", &result) ||
            !CheckCuda(cudaMemcpy(host_ey.data(), ey, field_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(Ey)", &result) ||
            !CheckCuda(cudaMemcpy(host_ez.data(), ez, field_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(Ez)", &result))
        {
            rcwa.FreeDeviceTensor(ex);
            rcwa.FreeDeviceTensor(ey);
            rcwa.FreeDeviceTensor(ez);
            return result;
        }
        rcwa.FreeDeviceTensor(ex);
        rcwa.FreeDeviceTensor(ey);
        rcwa.FreeDeviceTensor(ez);
        result.nearfield_rgba.resize(host_ex.size() * 4);
        double near_peak = 0.0;
        std::vector<double> near_intensity(host_ex.size(), 0.0);
        for (size_t i = 0; i < host_ex.size(); ++i)
        {
            near_intensity[i] = ComplexAbsSquaredDouble(host_ex[i]) +
                                ComplexAbsSquaredDouble(host_ey[i]) +
                                ComplexAbsSquaredDouble(host_ez[i]);
            near_peak = std::max(near_peak, near_intensity[i]);
        }
        result.nearfield_peak_intensity = near_peak;
        const double near_scale = near_peak > 0.0 ? 1.0 / near_peak : 1.0;
        for (size_t i = 0; i < near_intensity.size(); ++i)
            GrayColor(std::sqrt(std::max(0.0, near_intensity[i] * near_scale)),
                      result.nearfield_rgba.data() + i * 4);
        SaveNearfieldCache(nearfield_cache_path, result);
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return result;
}
