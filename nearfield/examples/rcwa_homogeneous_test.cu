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
    const Real freq = kOne;
    const int order_x = 5;
    const int order_y = 7;
    const Real Lx = kOne;
    const Real Ly = kOne;
    const Real inc_ang = static_cast<Real>(0.1);
    const Real azi_ang = static_cast<Real>(0.2);
    const Real thickness = static_cast<Real>(0.7);
    const Complex eps = make_complex(static_cast<Real>(2.3), static_cast<Real>(0.1));

    RCWA rcwa_exp(freq, order_x, order_y, Lx, Ly);
    RCWA rcwa_hom(freq, order_x, order_y, Lx, Ly);

    if (rcwa_exp.SetIncidentAngles(inc_ang, azi_ang) != Status::kSuccess ||
        rcwa_hom.SetIncidentAngles(inc_ang, azi_ang) != Status::kSuccess)
    {
        std::cout << "Failed to set incident angles.\n";
        return 1;
    }

    if (rcwa_exp.AddLayer(thickness, eps) != Status::kSuccess)
    {
        std::cout << "AddLayer (exp) failed.\n";
        return 1;
    }

    if (rcwa_hom.AddLayerHomogeneous(thickness, eps) != Status::kSuccess)
    {
        std::cout << "AddLayerHomogeneous failed.\n";
        return 1;
    }

    const auto& s_exp = rcwa_exp.layer_smatrices().front();
    const auto& s_hom = rcwa_hom.layer_smatrices().front();
    const int64_t n = 2 * rcwa_exp.order_N();

    const double tol = static_cast<double>(std::numeric_limits<Real>::epsilon()) * 1e4;
    const double err11 = MaxMatrixDiff(s_exp.S_11, s_hom.S_11, n);
    const double err12 = MaxMatrixDiff(s_exp.S_12, s_hom.S_12, n);
    const double err21 = MaxMatrixDiff(s_exp.S_21, s_hom.S_21, n);
    const double err22 = MaxMatrixDiff(s_exp.S_22, s_hom.S_22, n);

    std::cout << "Max diffs: S11=" << err11 << ", S12=" << err12 << ", S21=" << err21
              << ", S22=" << err22 << "\n";

    if (err11 > tol || err12 > tol || err21 > tol || err22 > tol)
    {
        std::cout << "RCWA homogeneous vs exp test failed.\n";
        return 1;
    }

    std::cout << "RCWA homogeneous vs exp test passed.\n";
    return 0;
}
