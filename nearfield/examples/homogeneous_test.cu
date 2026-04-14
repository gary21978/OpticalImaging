#include <cmath>
#include <cstdint>
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

int main()
{
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

    if (rcwa_ref.AddInputLayer(1.0) != Status::kSuccess ||
        rcwa_hom.AddInputLayer(1.0) != Status::kSuccess)
    {
        std::cout << "Failed to add input layer.\n";
        return 1;
    }
    
    if (rcwa_ref.SetIncidentAngles(inc_ang, azi_ang) != Status::kSuccess ||
        rcwa_hom.SetIncidentAngles(inc_ang, azi_ang) != Status::kSuccess)
    {
        std::cout << "Failed to set incident angles.\n";
        return 1;
    }

    float addlayer_ms = 0.0f;
    float homogeneous_ms = 0.0f;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (rcwa_ref.AddLayer(thickness, eps) != Status::kSuccess)
    {
        std::cout << "Reference AddLayer failed.\n";
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&addlayer_ms, start, stop);

    cudaEventRecord(start);
    if (rcwa_hom.AddLayerHomogeneous(thickness, eps) != Status::kSuccess)
    {
        std::cout << "AddLayerHomogeneous failed.\n";
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&homogeneous_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (rcwa_ref.SolveGlobalSMatrix() != Status::kSuccess ||
        rcwa_hom.SolveGlobalSMatrix() != Status::kSuccess)
    {
        std::cout << "SolveGlobalSMatrix failed.\n";
        return 1;
    }

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
        std::cout << "homogeneous_real_test failed.\n";
        return 1;
    }

    std::cout << "homogeneous_real_test passed.\n";
    return 0;
}
