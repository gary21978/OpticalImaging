#pragma once

#include <array>
#include <string>
#include <vector>

struct ShapeConfig
{
    std::string type = "rect";
    double center_x = 0.5;
    double center_y = 0.5;
    double size_x = 0.4;
    double size_y = 0.4;
};

struct LayerConfig
{
    std::string name = "Layer";
    std::string material = "custom";
    double thickness_nm = 100.0;
    double n_real = 1.5;
    double n_imag = 0.0;
    bool patterned = true;
    std::string geometry_mode = "uniform";
    std::string png_path;
    std::array<float, 3> color{0.15f, 0.75f, 0.95f};
    std::vector<ShapeConfig> shapes;
};

struct SimulationConfig
{
    double period_x_nm = 3000.0;
    double period_y_nm = 3000.0;
    double wavelength_nm = 365.0;
    int order_x = 12;
    int order_y = 12;
    int geometry_x = 200;
    int geometry_y = 100;
    int image_width = 1024;
    int image_height = 512;
    double sigma_outer = 0.9;
    double sigma_inner = 0.15;
    std::string source_shape = "annular";
    double source_open_angle_deg = 30.0;
    double source_rotate_angle_deg = 0.0;
    double source_rect_x = 0.8;
    double source_rect_y = 0.3;
    std::string polarization = "x";
    double polarization_angle_deg = 0.0;
    int approximation_order = 12;
    std::string precision = "double";
    std::vector<double> zernike = std::vector<double>(37, 0.0);
    double focus_nm = 0.0;
    double incident_theta_deg = 0.0;
    double incident_phi_deg = 0.0;
    double input_n_real = 1.0;
    double input_n_imag = 0.0;
    double substrate_n_real = 6.5271;
    double substrate_n_imag = 2.6672;
    std::string substrate_material = "Si";
    std::vector<LayerConfig> layers;
};

SimulationConfig DefaultSimulationConfig();
void UpdateNyquistGrid(SimulationConfig* config);
bool LoadSimulationConfig(const std::string& path, SimulationConfig* config, std::string* error);
bool SaveSimulationConfig(const std::string& path, const SimulationConfig& config, std::string* error);
bool ValidateSimulationConfig(const SimulationConfig& config, std::string* error);
