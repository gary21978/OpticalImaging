#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
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

struct Pattern
{
    std::string name;
    std::vector<Complex> eps;
};

bool InCircle(double x, double y, double cx, double cy, double radius)
{
    const double dx = x - cx;
    const double dy = y - cy;
    return dx * dx + dy * dy <= radius * radius;
}

Pattern BuildConcavePolygon(int nx, int ny)
{
    Pattern pattern{"concave_l_polygon", {}};
    pattern.eps.assign(static_cast<size_t>(nx) * ny, make_complex(1.0, 0.0));
    const Complex material = make_complex(6.0, 0.0);
    for (int x = 0; x < nx; ++x)
    {
        for (int y = 0; y < ny; ++y)
        {
            const bool outer = x >= 12 && x < 52 && y >= 12 && y < 52;
            const bool notch = x >= 30 && y >= 30;
            if (outer && !notch)
            {
                pattern.eps[static_cast<size_t>(x) * ny + y] = material;
            }
        }
    }
    return pattern;
}

Pattern BuildCorner(int nx, int ny)
{
    Pattern pattern{"right_angle_corner", {}};
    pattern.eps.assign(static_cast<size_t>(nx) * ny, make_complex(1.0, 0.0));
    const Complex material = make_complex(8.0, 0.0);
    for (int x = 0; x < nx; ++x)
    {
        for (int y = 0; y < ny; ++y)
        {
            if ((x >= 10 && x < 54 && y >= 28 && y < 42) ||
                (x >= 28 && x < 42 && y >= 10 && y < 54))
            {
                pattern.eps[static_cast<size_t>(x) * ny + y] = material;
            }
        }
    }
    return pattern;
}

Pattern BuildMultipleHoles(int nx, int ny)
{
    Pattern pattern{"multiple_holes", {}};
    pattern.eps.assign(static_cast<size_t>(nx) * ny, make_complex(9.0, 0.0));
    for (int x = 0; x < nx; ++x)
    {
        for (int y = 0; y < ny; ++y)
        {
            const bool hole = InCircle(x, y, 20.0, 20.0, 7.0) ||
                              InCircle(x, y, 43.0, 23.0, 8.0) ||
                              InCircle(x, y, 32.0, 44.0, 9.0);
            if (hole)
            {
                pattern.eps[static_cast<size_t>(x) * ny + y] = make_complex(1.0, 0.0);
            }
        }
    }
    return pattern;
}

Pattern BuildTiltedGrating(int nx, int ny)
{
    Pattern pattern{"tilted_grating", {}};
    pattern.eps.assign(static_cast<size_t>(nx) * ny, make_complex(1.0, 0.0));
    const Complex material = make_complex(-80.0, 35.0);
    constexpr double pi = 3.14159265358979323846;
    const double angle = 28.0 * pi / 180.0;
    const double ca = std::cos(angle);
    const double sa = std::sin(angle);
    constexpr double period = 18.0;
    for (int x = 0; x < nx; ++x)
    {
        for (int y = 0; y < ny; ++y)
        {
            double phase = std::fmod(ca * x + sa * y + 10.0 * period, period);
            if (phase < 0.48 * period)
            {
                pattern.eps[static_cast<size_t>(x) * ny + y] = material;
            }
        }
    }
    return pattern;
}

double RunReflectance(const Pattern& pattern,
                      int nx,
                      int ny,
                      int order,
                      LayerFactorization factorization)
{
    constexpr Real wavelength = 1.0;
    constexpr Real Lx = 1.0;
    constexpr Real Ly = 1.0;
    constexpr Real thickness = 0.1;

    DeviceBuffer d_eps;
    const size_t bytes = pattern.eps.size() * sizeof(Complex);
    Require(CudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_eps.ptr), bytes)), "cudaMalloc eps");
    Require(CudaStatus(cudaMemcpy(d_eps.ptr, pattern.eps.data(), bytes, cudaMemcpyHostToDevice)),
            "copy eps");

    RCWA rcwa(1.0 / wavelength, order, order, Lx, Ly);
    rcwa.SetHighOrder(true);
    Require(rcwa.AddInputLayer(1.0), "AddInputLayer");
    Require(rcwa.AddOutputLayer(1.0), "AddOutputLayer");
    Require(rcwa.SetIncidentAngles(static_cast<Real>(0.0), static_cast<Real>(0.0), "input"),
            "SetIncidentAngles");

    Require(rcwa.AddLayer(thickness, d_eps.ptr, static_cast<int64_t>(nx),
                          static_cast<int64_t>(ny), factorization),
            "AddLayer");

    Require(rcwa.SolveGlobalSMatrix(), "SolveGlobalSMatrix");

    const int64_t n = rcwa.order_N();
    const int64_t n2 = 2 * n;
    std::vector<Complex> s11(static_cast<size_t>(n2) * static_cast<size_t>(n2));
    Require(CudaStatus(cudaMemcpy(s11.data(),
                                  rcwa.global_smatrix().S_11,
                                  s11.size() * sizeof(Complex),
                                  cudaMemcpyDeviceToHost)),
            "copy S11");

    const int order_y = rcwa.order_y();
    const int64_t source_order =
        static_cast<int64_t>(order_y) +
        static_cast<int64_t>(order) * (2 * static_cast<int64_t>(order_y) + 1);
    const Complex value = s11[static_cast<size_t>(source_order) * n2 + source_order];
    return ComplexAbsSquaredDouble(value);
}

struct ConvergenceData
{
    std::string pattern_name;
    std::vector<int> orders;
    std::vector<double> laurent;
    std::vector<double> lalanne;
};

ConvergenceData RunConvergenceSweep(const Pattern& pattern,
                                     int nx,
                                     int ny,
                                     int order_min,
                                     int order_max)
{
    ConvergenceData data;
    data.pattern_name = pattern.name;

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Pattern: " << pattern.name << "\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::setw(7) << "order"
              << std::setw(18) << "Laurent R (%)"
              << std::setw(18) << "Lalanne R (%)"
              << std::setw(18) << "|diff| (%)\n";
    std::cout << std::string(70, '-') << "\n";

    for (int order = order_min; order <= order_max; ++order)
    {
        const double r_laurent = RunReflectance(pattern, nx, ny, order, LayerFactorization::kLaurent);
        const double r_lalanne =
            RunReflectance(pattern, nx, ny, order, LayerFactorization::kLalanne);

        data.orders.push_back(order);
        data.laurent.push_back(r_laurent);
        data.lalanne.push_back(r_lalanne);

        std::cout << std::setw(7) << order
                  << std::setw(18) << std::fixed << std::setprecision(8)
                  << r_laurent * 100.0
                  << std::setw(18) << std::fixed << std::setprecision(8)
                  << r_lalanne * 100.0
                  << std::setw(18) << std::scientific << std::setprecision(4)
                  << std::abs(r_laurent - r_lalanne) * 100.0 << "\n";
    }
    return data;
}

void WriteConvergenceFile(const std::string& filename, const ConvergenceData& data)
{
    std::ofstream out(filename);
    out << "# order  Laurent_R(%)  Lalanne_R(%)\n";
    out << "# Pattern: " << data.pattern_name << "\n";
    for (size_t i = 0; i < data.orders.size(); ++i)
    {
        out << data.orders[i] << " "
            << std::fixed << std::setprecision(10)
            << data.laurent[i] * 100.0 << " "
            << data.lalanne[i] * 100.0 << "\n";
    }
}

void WriteErrorFile(const std::string& filename, const ConvergenceData& data)
{
    std::ofstream out(filename);
    out << "# order  Laurent_rel_err  Lalanne_rel_err\n";
    out << "# Relative error w.r.t. highest-order Lalanne result\n";
    out << "# Pattern: " << data.pattern_name << "\n";

    const double ref = data.lalanne.back();
    for (size_t i = 0; i < data.orders.size(); ++i)
    {
        const double laurent_err = std::abs(data.laurent[i] - ref) / std::abs(ref);
        const double lalanne_err = std::abs(data.lalanne[i] - ref) / std::abs(ref);
        out << data.orders[i] << " "
            << std::scientific << std::setprecision(8)
            << laurent_err << " "
            << lalanne_err << "\n";
    }
}

void PrintSummary(const std::vector<ConvergenceData>& all_data)
{
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Convergence Summary\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::setw(25) << "Pattern"
              << std::setw(22) << "Laurent final R(%)"
              << std::setw(22) << "Lalanne final R(%)\n";
    std::cout << std::string(70, '-') << "\n";

    for (const auto& data : all_data)
    {
        std::cout << std::setw(25) << data.pattern_name
                  << std::setw(22) << std::fixed << std::setprecision(8)
                  << data.laurent.back() * 100.0
                  << std::setw(22) << std::fixed << std::setprecision(8)
                  << data.lalanne.back() * 100.0 << "\n";
    }
}
} // namespace

int main()
{
    constexpr int nx = 64;
    constexpr int ny = 64;
    constexpr int order_min = 3;
    constexpr int order_max = 12;

    const std::vector<Pattern> patterns = {
        BuildConcavePolygon(nx, ny),
        BuildCorner(nx, ny),
        BuildMultipleHoles(nx, ny),
        BuildTiltedGrating(nx, ny),
    };

    std::vector<ConvergenceData> all_data;
    for (const auto& pattern : patterns)
    {
        auto data = RunConvergenceSweep(pattern, nx, ny, order_min, order_max);
        WriteConvergenceFile(data.pattern_name + "_convergence.txt", data);
        WriteErrorFile(data.pattern_name + "_error.txt", data);
        all_data.push_back(std::move(data));
    }

    PrintSummary(all_data);

    std::cout << "\nData files written:\n";
    for (const auto& data : all_data)
    {
        std::cout << "  " << data.pattern_name << "_convergence.txt\n";
        std::cout << "  " << data.pattern_name << "_error.txt\n";
    }

    return EXIT_SUCCESS;
}
