#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "rcwa.h"

namespace
{
using Clock = std::chrono::steady_clock;

double ElapsedMillis(const Clock::time_point& start, const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void RequireStatus(Status status, const char* step)
{
    if (status != Status::kSuccess)
    {
        std::cerr << "ERROR: " << step << " failed: " << StatusToString(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// Frobenius norm of (A − B) relative to norm(A)
double RelativeDiff(const std::vector<Complex>& A,
                    const std::vector<Complex>& B,
                    int64_t n)
{
    double diff_norm2 = 0.0;
    double ref_norm2 = 0.0;
    for (int64_t i = 0; i < n * n; ++i)
    {
        const double dr = static_cast<double>(ComplexReal(A[i]) - ComplexReal(B[i]));
        const double di = static_cast<double>(ComplexImag(A[i]) - ComplexImag(B[i]));
        diff_norm2 += dr * dr + di * di;

        const double ar = static_cast<double>(ComplexReal(A[i]));
        const double ai = static_cast<double>(ComplexImag(A[i]));
        ref_norm2 += ar * ar + ai * ai;
    }
    return std::sqrt(diff_norm2) / std::sqrt(ref_norm2);
}

void CopySMatrixToHost(const SMatrix& S,
                       int64_t n,
                       std::vector<Complex>& S11,
                       std::vector<Complex>& S12,
                       std::vector<Complex>& S21,
                       std::vector<Complex>& S22)
{
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(Complex);
    S11.resize(static_cast<size_t>(n * n));
    S12.resize(static_cast<size_t>(n * n));
    S21.resize(static_cast<size_t>(n * n));
    S22.resize(static_cast<size_t>(n * n));
    cudaMemcpy(S11.data(), S.S_11, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(S12.data(), S.S_12, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(S21.data(), S.S_21, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(S22.data(), S.S_22, bytes, cudaMemcpyDeviceToHost);
}

// Simple geometry: a small cross pattern
void LoadTestGeometry(std::vector<Real>& geom, int nx, int ny)
{
    geom.assign(static_cast<size_t>(nx) * ny, 0.0);
    for (int y = 0; y < ny; ++y)
    {
        for (int x = nx / 4; x < 3 * nx / 4; ++x)
        {
            geom[static_cast<size_t>(x) * ny + y] = 1.0;
        }
    }
    for (int x = 0; x < nx; ++x)
    {
        for (int y = ny / 4; y < 3 * ny / 4; ++y)
        {
            geom[static_cast<size_t>(x) * ny + y] = 1.0;
        }
    }
}

void BuildEpsMap(const std::vector<Real>& geom, Complex n, std::vector<Complex>& eps_map)
{
    const size_t count = geom.size();
    eps_map.resize(count);
    const Complex one = make_complex(1.0, 0.0);
    const Complex n2 = ComplexMul(n, n);
    const Complex delta = ComplexSub(n2, one);
    for (size_t i = 0; i < count; ++i)
    {
        eps_map[i] = ComplexAdd(one, ComplexMul(delta, make_complex(geom[i], 0.0)));
    }
}

struct DeviceBuffer
{
    Complex* ptr = nullptr;
    ~DeviceBuffer() { SAFEFREE(ptr); }
    Complex* get() const { return ptr; }
};

struct RunResult
{
    double t_addlayer = 0.0;
    double t_global = 0.0;
    std::vector<Complex> S11;
    std::vector<Complex> S12;
    std::vector<Complex> S21;
    std::vector<Complex> S22;
};

RunResult RunPipeline(const char* method_name,
                      Real freq,
                      int order_x,
                      int order_y,
                      Real Lx,
                      Real Ly,
                      const std::vector<Real>& geometry,
                      int geom_x,
                      int geom_y,
                      const Complex* materials,
                      const Real* thicknesses,
                      int num_layers,
                      Complex eps_sub)
{
    RCWA rcwa(freq, order_x, order_y, Lx, Ly);
    rcwa.AddInputLayer(1.0);
    RequireStatus(rcwa.SetIncidentAngles(0.0, 0.0, "input"), "SetIncidentAngles");

    if (method_name[0] == 'S')
    {
        rcwa.SetExpmMethod('S');
    }

    DeviceBuffer d_material;
    const size_t mat_bytes = static_cast<size_t>(geom_x) * geom_y * sizeof(Complex);
    cudaMalloc(reinterpret_cast<void**>(&d_material.ptr), mat_bytes);

    std::vector<Complex> eps_map;
    double t_addlayer = 0.0;

    for (int i = 0; i < num_layers; ++i)
    {
        BuildEpsMap(geometry, materials[i], eps_map);
        cudaMemcpy(d_material.get(), eps_map.data(), mat_bytes, cudaMemcpyHostToDevice);

        Complex* eps_conv = nullptr;
        RequireStatus(
            rcwa.MaterialConv(d_material.get(), geom_x, geom_y, &eps_conv), "MaterialConv");

        auto t0 = Clock::now();
        RequireStatus(rcwa.AddLayer(thicknesses[i], eps_conv), "AddLayer");
        t_addlayer += ElapsedMillis(t0, Clock::now());
        SAFEFREE(eps_conv);
    }

    RequireStatus(rcwa.AddLayerHomogeneous(20.0, eps_sub), "AddLayerHomogeneous");

    auto t0 = Clock::now();
    RequireStatus(rcwa.SolveGlobalSMatrix(), "SolveGlobalSMatrix");
    const double t_global = ElapsedMillis(t0, Clock::now());

    RunResult result;
    result.t_addlayer = t_addlayer;
    result.t_global = t_global;

    const int64_t n_smatrix = 2 * rcwa.order_N();
    CopySMatrixToHost(rcwa.global_smatrix(),
                      n_smatrix,
                      result.S11,
                      result.S12,
                      result.S21,
                      result.S22);
    return result;
}
} // namespace

int main()
{
    std::cout << "Precision: "
#if USE_DOUBLE_PRECISION
              << "double\n";
#else
              << "single\n";
#endif

    // ── Test parameters ─────────────────────────────────────────────────
    const Real period_x = 3000.0;
    const Real period_y = 1500.0;
    const Real wavelength = 365.0;
    const Real freq = 1.0 / wavelength;
    const int geom_x = 100;
    const int geom_y = 50;

    std::vector<Real> geometry;
    LoadTestGeometry(geometry, geom_x, geom_y);

    const Complex n_OX = make_complex(1.4745, 0.0);
    const Complex n_TIN = make_complex(2.0669, 1.0563);
    const Complex n_TEOS = make_complex(1.380, 0.0);
    const Complex n_sub = make_complex(6.5271, 2.6672);
    const Complex eps_sub = ComplexMul(n_sub, n_sub);

    const Complex materials[] = {n_OX, n_TIN, n_TEOS};
    const Real thicknesses[] = {100.0, 21.0, 100.0};
    constexpr int num_layers = 3;

    // ── Test with multiple orders ───────────────────────────────────────
    const int test_orders[] = {3, 5, 7, 9, 11, 13};
    constexpr int num_orders = sizeof(test_orders) / sizeof(test_orders[0]);

    std::cout << "\n"
              << std::setw(7) << "order"
              << std::setw(10) << "n2"
              << std::setw(14) << "rel_diff_S"
              << std::setw(14) << "t_GEMM(ms)"
              << std::setw(14) << "t_SQRT(ms)"
              << std::setw(12) << "speedup" << "\n";
    std::cout << std::string(71, '-') << "\n";

    for (int oi = 0; oi < num_orders; ++oi)
    {
        const int order = test_orders[oi];

        const RunResult r_gemm = RunPipeline("GEMM",
                                             freq,
                                             order,
                                             order,
                                             period_x,
                                             period_y,
                                             geometry,
                                             geom_x,
                                             geom_y,
                                             materials,
                                             thicknesses,
                                             num_layers,
                                             eps_sub);

        const RunResult r_sqrt = RunPipeline("SQRT",
                                             freq,
                                             order,
                                             order,
                                             period_x,
                                             period_y,
                                             geometry,
                                             geom_x,
                                             geom_y,
                                             materials,
                                             thicknesses,
                                             num_layers,
                                             eps_sub);

        const int rcwa_n = (2 * order + 1) * (2 * order + 1);
        const int64_t n2 = 2 * rcwa_n;

        const double rel_S11 = RelativeDiff(r_gemm.S11, r_sqrt.S11, n2);
        const double rel_S12 = RelativeDiff(r_gemm.S12, r_sqrt.S12, n2);
        const double rel_S21 = RelativeDiff(r_gemm.S21, r_sqrt.S21, n2);
        const double rel_S22 = RelativeDiff(r_gemm.S22, r_sqrt.S22, n2);
        const double rel_diff = std::max({rel_S11, rel_S12, rel_S21, rel_S22});

        const double speedup = r_gemm.t_addlayer / std::max(r_sqrt.t_addlayer, 1e-9);

        std::cout << std::setw(7) << order
                  << std::setw(10) << n2
                  << std::setw(14) << std::scientific << std::setprecision(2) << rel_diff
                  << std::setw(14) << std::fixed << std::setprecision(1) << r_gemm.t_addlayer
                  << std::setw(14) << std::fixed << std::setprecision(1) << r_sqrt.t_addlayer
                  << std::setw(11) << std::fixed << std::setprecision(2) << speedup << "x"
                  << "\n";
    }

    std::cout << "\n";
    return 0;
}
