#include <algorithm>
#include <cmath>
#include <cstdlib>
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

bool RunAxisConvergenceCheck(GratingAxis axis)
{
    const double reference = RunTmReflectance(25, 0, axis, true);

    struct Sample
    {
        int order;
        double laurent;
        double lalanne;
    };

    const int orders[] = {9, 11, 13, 15, 17, 19, 21, 23};
    std::vector<Sample> samples;
    for (int order : orders)
    {
        const double r_laurent = RunTmReflectance(order, 0, axis, false);
        const double r_lalanne = RunTmReflectance(order, 0, axis, true);
        samples.push_back({order, r_laurent, r_lalanne});
    }

    std::cout << (axis == GratingAxis::kX ? "X" : "Y")
              << "-axis TM high-conductivity grating reflectance\n";
    std::cout << std::setw(7) << "order"
              << std::setw(22) << "Laurent R (%)"
              << std::setw(22) << "Lalanne R (%)" << "\n";
    for (const Sample& sample : samples)
    {
        std::cout << std::setw(7) << sample.order
                  << std::setw(22) << std::fixed << std::setprecision(8)
                  << sample.laurent * 100.0
                  << std::setw(22) << sample.lalanne * 100.0 << "\n";
    }

    const Sample& high = samples.back();
    const double high_laurent_error = std::abs(high.laurent - reference);
    const double high_lalanne_error = std::abs(high.lalanne - reference);
    if (!(high_lalanne_error < high_laurent_error))
    {
        std::cerr << "Lalanne correction was not more accurate than Laurent at order "
                  << high.order << ".\n";
        return false;
    }
    return true;
}

bool RunLamellarWithCrossOrdersCheck(GratingAxis axis)
{
    constexpr int cross_order = 2;
    const double reference = RunTmReflectance(25, cross_order, axis, true);

    std::cout << (axis == GratingAxis::kX ? "X" : "Y")
              << "-axis lamellar grating reflectance with cross order " << cross_order << "\n";
    std::cout << std::setw(7) << "order"
              << std::setw(22) << "Laurent R (%)"
              << std::setw(22) << "Lalanne R (%)" << "\n";

    const int orders[] = {9, 11, 13, 15, 17, 19, 21, 23};
    double high_laurent = 0.0;
    double high_lalanne = 0.0;
    for (int order : orders)
    {
        const double r_laurent = RunTmReflectance(order, cross_order, axis, false);
        const double r_lalanne = RunTmReflectance(order, cross_order, axis, true);
        if (order == orders[sizeof(orders) / sizeof(orders[0]) - 1])
        {
            high_laurent = r_laurent;
            high_lalanne = r_lalanne;
        }
        std::cout << std::setw(7) << order
                  << std::setw(22) << std::fixed << std::setprecision(8)
                  << r_laurent * 100.0
                  << std::setw(22) << r_lalanne * 100.0 << "\n";
    }

    const double laurent_error = std::abs(high_laurent - reference);
    const double lalanne_error = std::abs(high_lalanne - reference);

    if (!(lalanne_error < laurent_error))
    {
        std::cerr << "Lalanne correction was not more accurate with retained cross orders.\n";
        return false;
    }
    return true;
}
} // namespace

int main()
{
    const bool x_ok = RunAxisConvergenceCheck(GratingAxis::kX);
    const bool y_ok = RunAxisConvergenceCheck(GratingAxis::kY);
    const bool x_cross_ok = RunLamellarWithCrossOrdersCheck(GratingAxis::kX);
    const bool y_cross_ok = RunLamellarWithCrossOrdersCheck(GratingAxis::kY);
    return x_ok && y_ok && x_cross_ok && y_cross_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
