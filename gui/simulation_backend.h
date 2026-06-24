#pragma once

#include "simulation_config.h"

#include <string>
#include <vector>

struct SimulationResult
{
    int width = 0;
    int height = 0;
    int focus_count = 0;
    std::vector<double> focus_nm;
    std::vector<float> farfield_rgba;
    std::vector<float> nearfield_rgba;
    double elapsed_ms = 0.0;
    double peak_intensity = 0.0;
    double nearfield_peak_intensity = 0.0;
    std::string error;
};

SimulationResult RunSimulation(const SimulationConfig& config);
bool WriteSimulationResult(const std::string& path, const SimulationResult& result);
bool ReadSimulationResult(const std::string& path, SimulationResult* result);
