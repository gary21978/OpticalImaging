#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rcwa.h"

namespace
{
Status CudaStatus(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::cerr << "CUDA runtime error: " << cudaGetErrorString(status) << "\n";
    }
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

void Require(Status status, const char* step)
{
    if (status != Status::kSuccess)
    {
        std::cerr << "ERROR: " << step << " failed: " << StatusToString(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

struct DeviceBuffer
{
    Complex* ptr = nullptr;
    ~DeviceBuffer() { SAFEFREE(ptr); }
};

enum class GratingAxis
{
    kX,
    kY,
};

void BuildConductiveGrating(std::vector<Complex>& eps, int nx, int ny, GratingAxis axis)
{
    eps.resize(static_cast<size_t>(nx) * static_cast<size_t>(ny));
    const Complex air = make_complex(1.0, 0.0);
    const Complex conductor = make_complex(-80.0, 35.0);
    for (int x = 0; x < nx; ++x)
    {
        for (int y = 0; y < ny; ++y)
        {
            const bool in_bar = axis == GratingAxis::kX
                                    ? (x >= nx / 4 && x < 3 * nx / 4)
                                    : (y >= ny / 4 && y < 3 * ny / 4);
            eps[static_cast<size_t>(x) * ny + y] = in_bar ? conductor : air;
        }
    }
}

double RunTmReflectance(int order, int cross_order, GratingAxis axis, bool lalanne)
{
    constexpr Real wavelength = 1.0;
    constexpr Real period_x = 0.72;
    constexpr Real period_y = 1.0;
    constexpr Real thickness = 0.18;
    const int geom_x = axis == GratingAxis::kX ? 256 : std::max(8, 4 * cross_order + 1);
    const int geom_y = axis == GratingAxis::kX ? std::max(8, 4 * cross_order + 1) : 256;
    const int order_x = axis == GratingAxis::kX ? order : cross_order;
    const int order_y = axis == GratingAxis::kX ? cross_order : order;

    std::vector<Complex> eps_map;
    BuildConductiveGrating(eps_map, geom_x, geom_y, axis);

    DeviceBuffer d_eps;
    Require(CudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_eps.ptr),
                                  eps_map.size() * sizeof(Complex))),
            "cudaMalloc material");
    Require(CudaStatus(cudaMemcpy(d_eps.ptr,
                                  eps_map.data(),
                                  eps_map.size() * sizeof(Complex),
                                  cudaMemcpyHostToDevice)),
            "cudaMemcpy material");

    RCWA rcwa(1.0 / wavelength, order_x, order_y, period_x, period_y);
    rcwa.SetExpmMethod('F');
    rcwa.SetHighOrder(false);
    Require(rcwa.AddInputLayer(1.0), "AddInputLayer");
    Require(rcwa.AddOutputLayer(1.0), "AddOutputLayer");
    constexpr double pi = 3.14159265358979323846;
    const Real azi = axis == GratingAxis::kX ? static_cast<Real>(0.0)
                                             : static_cast<Real>(0.5 * pi);
    Require(rcwa.SetIncidentAngles(static_cast<Real>(30.0 * pi / 180.0), azi, "input"),
            "SetIncidentAngles");

    if (lalanne)
    {
        Require(rcwa.AddLayer(thickness,
                              d_eps.ptr,
                              geom_x,
                              geom_y,
                              LayerFactorization::kLalanne),
                "AddLayer Lalanne");
    }
    else
    {
        Require(rcwa.AddLayer(thickness,
                              d_eps.ptr,
                              geom_x,
                              geom_y,
                              LayerFactorization::kLaurent),
                "AddLayer Laurent");
    }

    Require(rcwa.SolveGlobalSMatrix(), "SolveGlobalSMatrix");
    const int64_t n2 = 2 * rcwa.order_N();
    std::vector<Complex> s11(static_cast<size_t>(n2) * static_cast<size_t>(n2));
    Require(CudaStatus(cudaMemcpy(s11.data(),
                                  rcwa.global_smatrix().S_11,
                                  s11.size() * sizeof(Complex),
                                  cudaMemcpyDeviceToHost)),
            "copy S11");

    const int64_t source_order = rcwa.order_y() + static_cast<int64_t>(rcwa.order_x()) *
                                                      (2 * static_cast<int64_t>(rcwa.order_y()) + 1);
    const int64_t source_col =
        axis == GratingAxis::kX ? source_order : source_order + rcwa.order_N();
    const Complex value = s11[static_cast<size_t>(source_col) * n2 + source_col];
    return ComplexAbsSquaredDouble(value);
}

struct ConvergenceData
{
    std::vector<int> orders;
    std::vector<double> laurent;
    std::vector<double> lalanne;
};

ConvergenceData RunConvergenceSweep(GratingAxis axis, int cross_order)
{
    ConvergenceData data;
    const int orders[] = {9, 11, 13, 15, 17, 19, 21, 23};
    for (int order : orders)
    {
        const double r_laurent = RunTmReflectance(order, cross_order, axis, false);
        const double r_lalanne = RunTmReflectance(order, cross_order, axis, true);
        data.orders.push_back(order);
        data.laurent.push_back(r_laurent);
        data.lalanne.push_back(r_lalanne);
    }
    return data;
}

void WriteDataFile(const char* filename, const ConvergenceData& data)
{
    std::ofstream out(filename);
    for (size_t i = 0; i < data.orders.size(); ++i)
    {
        out << data.orders[i] << " "
            << std::fixed << std::setprecision(10)
            << data.laurent[i] * 100.0 << " "
            << data.lalanne[i] * 100.0 << "\n";
    }
}

void PrintTable(const char* label, const ConvergenceData& data)
{
    std::cout << label << "\n";
    std::cout << std::setw(7) << "order"
              << std::setw(22) << "Laurent R (%)"
              << std::setw(22) << "Lalanne R (%)" << "\n";
    for (size_t i = 0; i < data.orders.size(); ++i)
    {
        std::cout << std::setw(7) << data.orders[i]
                  << std::setw(22) << std::fixed << std::setprecision(8)
                  << data.laurent[i] * 100.0
                  << std::setw(22) << data.lalanne[i] * 100.0 << "\n";
    }
}
} // namespace

int main()
{
    const auto x_data = RunConvergenceSweep(GratingAxis::kX, 0);
    PrintTable("X-axis TM high-conductivity grating reflectance", x_data);
    WriteDataFile("convergence_x.txt", x_data);

    const auto y_data = RunConvergenceSweep(GratingAxis::kY, 0);
    PrintTable("Y-axis TM high-conductivity grating reflectance", y_data);
    WriteDataFile("convergence_y.txt", y_data);

    const auto x_cross_data = RunConvergenceSweep(GratingAxis::kX, 2);
    PrintTable("X-axis lamellar grating reflectance (cross order 2)", x_cross_data);
    WriteDataFile("convergence_x_cross.txt", x_cross_data);

    const auto y_cross_data = RunConvergenceSweep(GratingAxis::kY, 2);
    PrintTable("Y-axis lamellar grating reflectance (cross order 2)", y_cross_data);
    WriteDataFile("convergence_y_cross.txt", y_cross_data);

    std::cout << "\nData files written: convergence_x.txt, convergence_y.txt, "
                 "convergence_x_cross.txt, convergence_y_cross.txt\n";

    return EXIT_SUCCESS;
}
