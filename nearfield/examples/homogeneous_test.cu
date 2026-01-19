#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "rcwa.h"

double ComplexDiff(const Complex& a, const Complex& b)
{
    const double dr = static_cast<double>(complex_real(a) - complex_real(b));
    const double di = static_cast<double>(complex_imag(a) - complex_imag(b));
    return std::sqrt(dr * dr + di * di);
}

double MaxMatrixDiff(const Complex* a, const Complex* b, int64_t n)
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
    const int order_x = 9;
    const int order_y = 2;
    const Real Lx = 1.0;
    const Real Ly = 1.0;
    const Real inc_ang = 0.1;
    const Real azi_ang = 0.2;
    const Real thickness = 0.7;
    const Complex eps = make_complex(2.3, 0.1);

    RCWA rcwa_exp(freq, order_x, order_y, Lx, Ly);
    RCWA rcwa_hom(freq, order_x, order_y, Lx, Ly);

    if (rcwa_exp.SetIncidentAngles(inc_ang, azi_ang) != Status::kSuccess ||
        rcwa_hom.SetIncidentAngles(inc_ang, azi_ang) != Status::kSuccess)
    {
        std::cout << "Failed to set incident angles.\n";
        return 1;
    }

    float exp_ms = 0.0f;
    float hom_ms = 0.0f;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (rcwa_exp.AddLayer(thickness, eps) != Status::kSuccess)
    {
        std::cout << "AddLayer (exp) failed.\n";
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exp_ms, start, stop);

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
    cudaEventElapsedTime(&hom_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const auto& s_exp = rcwa_exp.layer_smatrices().front();
    const auto& s_hom = rcwa_hom.layer_smatrices().front();
    const int64_t n = 2 * rcwa_exp.order_N();

    const double tol = static_cast<double>(std::numeric_limits<Real>::epsilon()) * 1e4;
    const double err11 = MaxMatrixDiff(s_exp.S_11, s_hom.S_11, n);
    const double err12 = MaxMatrixDiff(s_exp.S_12, s_hom.S_12, n);
    const double err21 = MaxMatrixDiff(s_exp.S_21, s_hom.S_21, n);
    const double err22 = MaxMatrixDiff(s_exp.S_22, s_hom.S_22, n);

    Complex s_exp_00;
    Complex s_hom_00;
    cudaMemcpy(&s_exp_00, s_exp.S_11, sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s_hom_00, s_hom.S_11, sizeof(Complex), cudaMemcpyDeviceToHost);

    std::cout << "S_11(0,0): exp=(" << complex_real(s_exp_00) << ", " << complex_imag(s_exp_00)
              << "), hom=(" << complex_real(s_hom_00) << ", " << complex_imag(s_hom_00) << ")\n";
    std::cout << "Max diffs: S11=" << err11 << ", S12=" << err12 << ", S21=" << err21
              << ", S22=" << err22 << "\n";
    std::cout << "Timing (ms): AddLayer=" << exp_ms << ", AddLayerHomogeneous=" << hom_ms << "\n";

    if (err11 > tol || err12 > tol || err21 > tol || err22 > tol)
    {
        std::cout << "RCWA homogeneous vs exp test failed.\n";
        return 1;
    }

    std::cout << "RCWA homogeneous vs exp test passed.\n";
    return 0;
}
