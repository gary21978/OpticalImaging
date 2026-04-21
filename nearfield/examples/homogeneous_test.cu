#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "rcwa.h"

static double ComplexDiff(const Complex& a, const Complex& b)
{
    const double dr = static_cast<double>(ComplexReal(a) - ComplexReal(b));
    const double di = static_cast<double>(ComplexImag(a) - ComplexImag(b));
    return std::sqrt(dr * dr + di * di);
}

using SteadyClock = std::chrono::steady_clock;

static void PrintCaseRuntime(const char* case_name, SteadyClock::time_point start_time)
{
    const double total_ms =
        std::chrono::duration<double, std::milli>(SteadyClock::now() - start_time).count();
    std::cout << case_name << " runtime (ms): " << total_ms << "\n";
}

static bool CheckStatus(Status status, const char* step)
{
    if (status == Status::kSuccess)
    {
        return true;
    }
    std::cout << step << " failed with status " << static_cast<int>(status) << "\n";
    return false;
}

static bool MeasureCudaCallMillis(const std::function<Status()>& fn, float* elapsed_ms)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
    {
        if (start)
        {
            cudaEventDestroy(start);
        }
        if (stop)
        {
            cudaEventDestroy(stop);
        }
        return false;
    }

    cudaEventRecord(start);
    const Status status = fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return status == Status::kSuccess;
}

static double MaxMatrixDiff(const Complex* a, const Complex* b, int64_t n)
{
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
    std::vector<Complex> ha(count);
    std::vector<Complex> hb(count);
    if (cudaMemcpy(ha.data(), a, count * sizeof(Complex), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return std::numeric_limits<double>::infinity();
    }
    if (cudaMemcpy(hb.data(), b, count * sizeof(Complex), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        max_err = std::max(max_err, ComplexDiff(ha[i], hb[i]));
    }
    return max_err;
}

static int TestHomogeneousConsistency()
{
    const char* case_name = "homogeneous test";
    const auto case_start_time = SteadyClock::now();

    const Real freq = 1.0;
    const Real period_x = 1.0;
    const Real period_y = 1.0;
    const int order_x = 15;
    const int order_y = 15;
    const Real inc_ang = 0.1;
    const Real azi_ang = 0.2;
    const Real thickness = 0.7;
    const Complex eps = make_complex(2.3, 0.1);

    RCWA rcwa_ref(freq, order_x, order_y, period_x, period_y);
    RCWA rcwa_hom(freq, order_x, order_y, period_x, period_y);

    rcwa_ref.AddInputLayer(1.0);
    rcwa_hom.AddInputLayer(1.0);
    rcwa_ref.SetIncidentAngles(inc_ang, azi_ang);
    rcwa_hom.SetIncidentAngles(inc_ang, azi_ang);

    float addlayer_ms = 0.0f;
    float homogeneous_ms = 0.0f;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    rcwa_ref.AddLayer(thickness, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&addlayer_ms, start, stop);

    cudaEventRecord(start);
    rcwa_hom.AddLayerHomogeneous(thickness, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&homogeneous_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    rcwa_ref.SolveGlobalSMatrix();
    rcwa_hom.SolveGlobalSMatrix();

    const int64_t n2 = 2 * rcwa_ref.order_N();
    const auto& s_ref = rcwa_ref.global_smatrix();
    const auto& s_hom = rcwa_hom.global_smatrix();

    double max_layer_err = 0.0;
    max_layer_err = std::max(max_layer_err, MaxMatrixDiff(s_ref.S_11, s_hom.S_11, n2));
    max_layer_err = std::max(max_layer_err, MaxMatrixDiff(s_ref.S_12, s_hom.S_12, n2));
    max_layer_err = std::max(max_layer_err, MaxMatrixDiff(s_ref.S_21, s_hom.S_21, n2));
    max_layer_err = std::max(max_layer_err, MaxMatrixDiff(s_ref.S_22, s_hom.S_22, n2));

    const double tol = static_cast<double>(std::numeric_limits<Real>::epsilon()) * 1e5;
    std::cout << "Single-layer homogeneous consistency:\n";
    std::cout << "  Max layer diff: " << max_layer_err << "\n";
    std::cout << "  Timing (ms): AddLayer=" << addlayer_ms
              << ", AddLayerHomogeneous=" << homogeneous_ms << "\n";
    std::cout << "  Tolerance: " << tol << "\n";

    if (max_layer_err > tol)
    {
        std::cout << "homogeneous test failed.\n";
        PrintCaseRuntime(case_name, case_start_time);
        return 1;
    }

    std::cout << "homogeneous test passed.\n";
    PrintCaseRuntime(case_name, case_start_time);
    return 0;
}

static int TestFreeSpaceOutputConsistency()
{
    const char* case_name = "free space test";

    const Real freq = 1.0;
    const Real period_x = 1.0;
    const Real period_y = 1.0;
    const int order_x = 15;
    const int order_y = 15;
    const Real inc_ang = 0.1;
    const Real azi_ang = 0.2;
    const Real thickness = 0.7;
    const Complex eps = make_complex(2.3, 0.1);

    RCWA rcwa_default_out(freq, order_x, order_y, period_x, period_y);
    RCWA rcwa_explicit_out(freq, order_x, order_y, period_x, period_y);

    rcwa_explicit_out.AddInputLayer(1.0);
    rcwa_explicit_out.AddOutputLayer(1.0);
    rcwa_default_out.SetIncidentAngles(inc_ang, azi_ang);
    rcwa_explicit_out.SetIncidentAngles(inc_ang, azi_ang);
    rcwa_default_out.AddLayerHomogeneous(thickness, eps);
    rcwa_explicit_out.AddLayerHomogeneous(thickness, eps);
    
    const auto case_start_time = SteadyClock::now();
    rcwa_default_out.SolveGlobalSMatrix();
    rcwa_explicit_out.SolveGlobalSMatrix();
    PrintCaseRuntime(case_name, case_start_time);

    const int64_t n2 = 2 * rcwa_default_out.order_N();
    const auto& s_default_out = rcwa_default_out.global_smatrix();
    const auto& s_explicit_out = rcwa_explicit_out.global_smatrix();

    double err = 0.0;
    err = std::max(err, MaxMatrixDiff(s_default_out.S_11, s_explicit_out.S_11, n2));
    err = std::max(err, MaxMatrixDiff(s_default_out.S_12, s_explicit_out.S_12, n2));
    err = std::max(err, MaxMatrixDiff(s_default_out.S_21, s_explicit_out.S_21, n2));
    err = std::max(err, MaxMatrixDiff(s_default_out.S_22, s_explicit_out.S_22, n2));

    const double tol = static_cast<double>(std::numeric_limits<Real>::epsilon()) * 1e5;
    std::cout << "Explicit free-space I/O-layer consistency:\n";
    std::cout << "  Max I/O-layer diff: " << err << "\n";

    if (err > tol)
    {
        std::cout << "free space test failed.\n";
        return 1;
    }
    else 
    {
        std::cout << "free space test passed.\n";
        return 0;
    }
}

static int TestHomogeneousRepeatConsistency()
{
    const char* case_name = "homogeneous repeat test";
    const auto case_start_time = SteadyClock::now();

    const Real freq = 1.0;
    const Real period_x = 1.0;
    const Real period_y = 1.0;
    const int order_x = 15;
    const int order_y = 15;
    const Real inc_ang = 0.1;
    const Real azi_ang = 0.2;
    const Real thickness = 0.7;
    const Complex eps = make_complex(2.3, 0.1);

    RCWA rcwa_repeat(freq, order_x, order_y, period_x, period_y);
    RCWA rcwa_thick(freq, order_x, order_y, period_x, period_y);

    if (!CheckStatus(rcwa_repeat.AddInputLayer(1.0), "AddInputLayer repeat") ||
        !CheckStatus(rcwa_thick.AddInputLayer(1.0), "AddInputLayer thick") ||
        !CheckStatus(rcwa_repeat.SetIncidentAngles(inc_ang, azi_ang), "SetIncidentAngles repeat") ||
        !CheckStatus(rcwa_thick.SetIncidentAngles(inc_ang, azi_ang), "SetIncidentAngles thick"))
    {
        PrintCaseRuntime(case_name, case_start_time);
        return 1;
    }

    float base_homogeneous_ms = 0.0f;
    float repeat_ms = 0.0f;
    float thick_homogeneous_ms = 0.0f;
    if (!MeasureCudaCallMillis([&]() { return rcwa_repeat.AddLayerHomogeneous(thickness, eps); },
                               &base_homogeneous_ms) ||
        !MeasureCudaCallMillis([&]() { return rcwa_repeat.RepeatCurrentStackSMatrix(3); },
                               &repeat_ms) ||
        !MeasureCudaCallMillis([&]() { return rcwa_thick.AddLayerHomogeneous(thickness * 8.0, eps); },
                               &thick_homogeneous_ms))
    {
        std::cout << "repeat/thick homogeneous setup failed.\n";
        PrintCaseRuntime(case_name, case_start_time);
        return 1;
    }

    if (!CheckStatus(rcwa_repeat.SolveGlobalSMatrix(), "SolveGlobalSMatrix repeat") ||
        !CheckStatus(rcwa_thick.SolveGlobalSMatrix(), "SolveGlobalSMatrix thick"))
    {
        PrintCaseRuntime(case_name, case_start_time);
        return 1;
    }

    const int64_t n2 = 2 * rcwa_repeat.order_N();
    const auto& s_repeat = rcwa_repeat.global_smatrix();
    const auto& s_thick = rcwa_thick.global_smatrix();

    double max_err = 0.0;
    max_err = std::max(max_err, MaxMatrixDiff(s_repeat.S_11, s_thick.S_11, n2));
    max_err = std::max(max_err, MaxMatrixDiff(s_repeat.S_12, s_thick.S_12, n2));
    max_err = std::max(max_err, MaxMatrixDiff(s_repeat.S_21, s_thick.S_21, n2));
    max_err = std::max(max_err, MaxMatrixDiff(s_repeat.S_22, s_thick.S_22, n2));

    const double tol = static_cast<double>(std::numeric_limits<Real>::epsilon()) * 1e5;
    std::cout << "Homogeneous RepeatSMatrix consistency:\n";
    std::cout << "  Max repeated-vs-thick diff: " << max_err << "\n";
    std::cout << "  Timing (ms): base AddLayerHomogeneous=" << base_homogeneous_ms
              << ", RepeatCurrentStackSMatrix(x3)=" << repeat_ms
              << ", thick AddLayerHomogeneous(x8)=" << thick_homogeneous_ms << "\n";
    std::cout << "  Tolerance: " << tol << "\n";

    if (max_err > tol)
    {
        std::cout << "homogeneous repeat test failed.\n";
        PrintCaseRuntime(case_name, case_start_time);
        return 1;
    }

    std::cout << "homogeneous repeat test passed.\n";
    PrintCaseRuntime(case_name, case_start_time);
    return 0;
}

int main()
{
    int status = 0;
    status |= TestHomogeneousConsistency();
    status |= TestFreeSpaceOutputConsistency();
    status |= TestHomogeneousRepeatConsistency();
    return status;
}
