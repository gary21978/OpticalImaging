#include "simulation_backend.h"

#include <cstdint>
#include <fstream>

bool WriteSimulationResult(const std::string& path, const SimulationResult& result)
{
    std::ofstream out(path, std::ios::binary);
    const uint32_t magic = 0x53494D52u;
    const uint64_t far_count = result.farfield_rgba.size();
    const uint64_t near_count = result.nearfield_rgba.size();
    const uint64_t focus_value_count = result.focus_nm.size();
    const uint64_t error_size = result.error.size();
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&result.width), sizeof(result.width));
    out.write(reinterpret_cast<const char*>(&result.height), sizeof(result.height));
    out.write(reinterpret_cast<const char*>(&result.focus_count), sizeof(result.focus_count));
    out.write(reinterpret_cast<const char*>(&result.elapsed_ms), sizeof(result.elapsed_ms));
    out.write(reinterpret_cast<const char*>(&result.peak_intensity), sizeof(result.peak_intensity));
    out.write(reinterpret_cast<const char*>(&result.nearfield_peak_intensity), sizeof(result.nearfield_peak_intensity));
    out.write(reinterpret_cast<const char*>(&far_count), sizeof(far_count));
    out.write(reinterpret_cast<const char*>(&near_count), sizeof(near_count));
    out.write(reinterpret_cast<const char*>(&focus_value_count), sizeof(focus_value_count));
    out.write(reinterpret_cast<const char*>(&error_size), sizeof(error_size));
    out.write(result.error.data(), static_cast<std::streamsize>(error_size));
    out.write(reinterpret_cast<const char*>(result.focus_nm.data()),
              static_cast<std::streamsize>(focus_value_count * sizeof(double)));
    out.write(reinterpret_cast<const char*>(result.farfield_rgba.data()), static_cast<std::streamsize>(far_count * sizeof(float)));
    out.write(reinterpret_cast<const char*>(result.nearfield_rgba.data()), static_cast<std::streamsize>(near_count * sizeof(float)));
    return static_cast<bool>(out);
}

bool ReadSimulationResult(const std::string& path, SimulationResult* result)
{
    if (!result)
        return false;
    std::ifstream in(path, std::ios::binary);
    uint32_t magic = 0;
    uint64_t far_count = 0, near_count = 0, focus_value_count = 0, error_size = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!in || magic != 0x53494D52u)
        return false;
    in.read(reinterpret_cast<char*>(&result->width), sizeof(result->width));
    in.read(reinterpret_cast<char*>(&result->height), sizeof(result->height));
    in.read(reinterpret_cast<char*>(&result->focus_count), sizeof(result->focus_count));
    in.read(reinterpret_cast<char*>(&result->elapsed_ms), sizeof(result->elapsed_ms));
    in.read(reinterpret_cast<char*>(&result->peak_intensity), sizeof(result->peak_intensity));
    in.read(reinterpret_cast<char*>(&result->nearfield_peak_intensity), sizeof(result->nearfield_peak_intensity));
    in.read(reinterpret_cast<char*>(&far_count), sizeof(far_count));
    in.read(reinterpret_cast<char*>(&near_count), sizeof(near_count));
    in.read(reinterpret_cast<char*>(&focus_value_count), sizeof(focus_value_count));
    in.read(reinterpret_cast<char*>(&error_size), sizeof(error_size));
    result->error.resize(error_size);
    result->focus_nm.resize(focus_value_count);
    result->farfield_rgba.resize(far_count);
    result->nearfield_rgba.resize(near_count);
    in.read(result->error.data(), static_cast<std::streamsize>(error_size));
    in.read(reinterpret_cast<char*>(result->focus_nm.data()),
            static_cast<std::streamsize>(focus_value_count * sizeof(double)));
    in.read(reinterpret_cast<char*>(result->farfield_rgba.data()), static_cast<std::streamsize>(far_count * sizeof(float)));
    in.read(reinterpret_cast<char*>(result->nearfield_rgba.data()), static_cast<std::streamsize>(near_count * sizeof(float)));
    return static_cast<bool>(in);
}
