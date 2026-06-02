#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rcwa.h"

namespace
{
using Clock = std::chrono::steady_clock;

double ElapsedMillis(const Clock::time_point& s, const Clock::time_point& e)
{
    return std::chrono::duration<double, std::milli>(e - s).count();
}

void Check(Status st, const char* step)
{
    if (st != Status::kSuccess)
    {
        std::cerr << step << " failed: " << StatusToString(st) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

double RelDiff(const std::vector<Complex>& A, const std::vector<Complex>& B, int64_t n)
{
    double dn = 0.0;
    double rn = 0.0;
    for (int64_t i = 0; i < n * n; ++i)
    {
        const double dr = ComplexReal(A[i]) - ComplexReal(B[i]);
        const double di = ComplexImag(A[i]) - ComplexImag(B[i]);
        const double ar = ComplexReal(A[i]);
        const double ai = ComplexImag(A[i]);
        dn += dr * dr + di * di;
        rn += ar * ar + ai * ai;
    }
    return std::sqrt(dn) / std::sqrt(rn);
}

void CopySMatrixToHost(const SMatrix& S,
                       int64_t n,
                       std::vector<Complex>& S11,
                       std::vector<Complex>& S12,
                       std::vector<Complex>& S21,
                       std::vector<Complex>& S22)
{
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(Complex);
    S11.resize(n * n);
    S12.resize(n * n);
    S21.resize(n * n);
    S22.resize(n * n);
    cudaMemcpy(S11.data(), S.S_11, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(S12.data(), S.S_12, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(S21.data(), S.S_21, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(S22.data(), S.S_22, bytes, cudaMemcpyDeviceToHost);
}

void FillRect(std::vector<Real>& g, int nx, int ny, int x0, int y0, int x1, int y1)
{
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::min(nx, x1);
    y1 = std::min(ny, y1);
    for (int x = x0; x < x1; ++x)
    {
        for (int y = y0; y < y1; ++y)
        {
            g[static_cast<size_t>(x) * ny + y] = 1.0;
        }
    }
}

void PatternSerpentine(std::vector<Real>& g, int nx, int ny)
{
    g.assign(static_cast<size_t>(nx) * ny, 0.0);

    const int trace_w = std::max(2, nx / 33);
    const int pad = std::max(6, nx / 20);
    const int x_left = std::max(trace_w, nx / 9);
    const int x_right = nx - x_left;
    const int y_levels[] = {ny / 5, ny / 3, ny / 2, 2 * ny / 3, 4 * ny / 5};
    constexpr int level_count = sizeof(y_levels) / sizeof(y_levels[0]);

    for (int i = 0; i < level_count; ++i)
    {
        const int y = y_levels[i];
        FillRect(g, nx, ny, x_left, y - trace_w / 2, x_right, y + trace_w / 2);
        if (i + 1 < level_count)
        {
            const int x_turn = (i % 2 == 0) ? x_right : x_left;
            FillRect(g,
                     nx,
                     ny,
                     x_turn - trace_w / 2,
                     y - trace_w / 2,
                     x_turn + trace_w / 2,
                     y_levels[i + 1] + trace_w / 2);
        }
    }

    FillRect(g,
             nx,
             ny,
             x_left - pad / 2,
             y_levels[0] - pad / 2,
             x_left + pad / 2,
             y_levels[0] + pad / 2);
    FillRect(g,
             nx,
             ny,
             x_right - pad / 2,
             y_levels[level_count - 1] - pad / 2,
             x_right + pad / 2,
             y_levels[level_count - 1] + pad / 2);

    const int contact = std::max(4, nx / 50);
    const int contacts[][2] = {
        {nx / 3, y_levels[1]},
        {3 * nx / 5, y_levels[2]},
        {2 * nx / 5, y_levels[3]},
        {3 * nx / 4, y_levels[0]},
    };
    for (const auto& c : contacts)
    {
        FillRect(g, nx, ny, c[0] - contact, c[1] - contact, c[0] + contact, c[1] + contact);
    }
}

void BuildEps(const std::vector<Real>& g, Complex n, std::vector<Complex>& eps)
{
    eps.resize(g.size());
    const Complex one = make_complex(1.0, 0.0);
    const Complex n2 = ComplexMul(n, n);
    const Complex delta = ComplexSub(n2, one);
    for (size_t i = 0; i < g.size(); ++i)
    {
        eps[i] = ComplexAdd(one, ComplexMul(delta, make_complex(g[i], 0.0)));
    }
}

int MaterialGridSize(int base, int order)
{
    return std::max(base, 4 * order + 1);
}

struct DeviceBuffer
{
    Complex* p = nullptr;
    ~DeviceBuffer() { SAFEFREE(p); }
};

struct Result
{
    double t_layer = 0.0;
    double t_global = 0.0;
    double R = 0.0;
    double T = 0.0;
    double A = 0.0;
    std::vector<Complex> S11;
    std::vector<Complex> S12;
    std::vector<Complex> S21;
    std::vector<Complex> S22;
};

Result Run(Real freq,
           int order,
           Real Lx,
           Real Ly,
           const std::vector<Real>& geom,
           int gx,
           int gy,
           Complex material_n,
           Real thickness,
           Complex eps_out,
           char method)
{
    RCWA rcwa(freq, order, order, Lx, Ly);
    Check(rcwa.AddInputLayer(1.0), "input");
    Check(rcwa.AddOutputLayer(eps_out), "output");
    Check(rcwa.SetIncidentAngles(0.0, 0.0, "input"), "angles");
    rcwa.SetExpmMethod(method);
    rcwa.SetHighOrder(false);

    std::vector<Complex> eps_map;
    BuildEps(geom, material_n, eps_map);

    DeviceBuffer d_material;
    cudaMalloc(reinterpret_cast<void**>(&d_material.p),
               static_cast<size_t>(gx) * gy * sizeof(Complex));
    cudaMemcpy(d_material.p, eps_map.data(), eps_map.size() * sizeof(Complex), cudaMemcpyHostToDevice);

    Complex* eps_conv = nullptr;
    Check(rcwa.MaterialConv(d_material.p, gx, gy, &eps_conv), "conv");

    const auto t_layer0 = Clock::now();
    Check(rcwa.AddLayer(thickness, eps_conv), "add");
    const double t_layer = ElapsedMillis(t_layer0, Clock::now());
    SAFEFREE(eps_conv);

    Check(rcwa.AddLayerHomogeneous(20.0, eps_out), "output spacer");

    const auto t_global0 = Clock::now();
    Check(rcwa.SolveGlobalSMatrix(), "solve");
    const double t_global = ElapsedMillis(t_global0, Clock::now());

    Result r;
    r.t_layer = t_layer;
    r.t_global = t_global;
    Check(rcwa.EnergyBalance(&r.R, &r.T, &r.A), "energy");

    const int64_t n2 = 2 * rcwa.order_N();
    CopySMatrixToHost(rcwa.global_smatrix(), n2, r.S11, r.S12, r.S21, r.S22);
    return r;
}

void PrintHeader()
{
    std::cout << std::setw(7) << "order" << std::setw(10) << "n2"
              << std::setw(10) << "grid"
              << std::setw(14) << "t_G12(ms)" << std::setw(14) << "t_F12(ms)"
              << std::setw(11) << "speedup"
              << std::setw(12) << "rel_diff"
              << std::setw(8) << "R(%)" << std::setw(8) << "T(%)"
              << std::setw(8) << "A(%)" << std::setw(11) << "R+T+A"
              << "\n" << std::string(113, '-') << "\n";
}

void PrintRow(int order, int gx, int gy, const Result& g, const Result& f)
{
    const int64_t side = 2 * static_cast<int64_t>(order) + 1;
    const int64_t n2 = 2 * side * side;
    const double speedup = g.t_layer / std::max(f.t_layer, 1e-9);
    const double rel_diff = std::max({RelDiff(g.S11, f.S11, n2),
                                      RelDiff(g.S12, f.S12, n2),
                                      RelDiff(g.S21, f.S21, n2),
                                      RelDiff(g.S22, f.S22, n2)});

    std::cout << std::setw(7) << order << std::setw(10) << n2
              << std::setw(5) << gx << "x" << std::left << std::setw(4) << gy << std::right
              << std::setw(14) << std::fixed << std::setprecision(0) << g.t_layer
              << std::setw(14) << std::fixed << std::setprecision(0) << f.t_layer
              << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x"
              << std::setw(12) << std::scientific << std::setprecision(2) << rel_diff
              << std::setw(8) << std::fixed << std::setprecision(2) << g.R * 100
              << std::setw(8) << std::fixed << std::setprecision(2) << g.T * 100
              << std::setw(8) << std::fixed << std::setprecision(2) << g.A * 100
              << std::setw(11) << std::fixed << std::setprecision(4) << g.R + g.T + g.A
              << "\n";
}

} // namespace

int main()
{
    std::cout << "Serpentine Pattern Verification\n"
              << "===============================\n"
#if USE_DOUBLE_PRECISION
              << "Precision: double\n\n";
#else
              << "Precision: single\n\n";
#endif

    const Real Lx = 3000.0;
    const Real Ly = 1500.0;
    const Real freq = 1.0 / 365.0;
    const int base_gx = 200;
    const int base_gy = 100;
    const Complex n_TEOS = make_complex(1.380, 0.0);
    const Complex eps_out = make_complex(1.0, 0.0);
    const Real thickness = 21.0;

    const int orders[] = {8, 16};
    constexpr int order_count = sizeof(orders) / sizeof(orders[0]);

    std::vector<Real> geom;
    const int warmup_order = orders[0];
    const int warmup_gx = MaterialGridSize(base_gx, warmup_order);
    const int warmup_gy = MaterialGridSize(base_gy, warmup_order);
    PatternSerpentine(geom, warmup_gx, warmup_gy);
    Run(freq, warmup_order, Lx, Ly, geom, warmup_gx, warmup_gy, n_TEOS, thickness, eps_out, 'G');

    PrintHeader();
    for (int i = 0; i < order_count; ++i)
    {
        const int order = orders[i];
        const int gx = MaterialGridSize(base_gx, order);
        const int gy = MaterialGridSize(base_gy, order);
        PatternSerpentine(geom, gx, gy);

        std::cout << "  order=" << order << " grid=" << gx << "x" << gy
                  << " (G12)..." << std::flush;
        const Result g = Run(freq, order, Lx, Ly, geom, gx, gy, n_TEOS, thickness, eps_out, 'G');
        std::cout << " (F12)..." << std::flush;
        const Result f = Run(freq, order, Lx, Ly, geom, gx, gy, n_TEOS, thickness, eps_out, 'F');
        std::cout << " done\n";

        PrintRow(order, gx, gy, g, f);
    }

    std::cout << "\nDone.\n";
    return 0;
}
