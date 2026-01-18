#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "layer_matrix.h"
#include "matrix_print_utils.h"

namespace
{

struct TMatrixHolder
{
    TMatrix value{};

    ~TMatrixHolder()
    {
        SAFEFREE(value.T_11);
        SAFEFREE(value.T_12);
        SAFEFREE(value.T_21);
        SAFEFREE(value.T_22);
    }
};

struct SMatrixHolder
{
    SMatrix value{};

    ~SMatrixHolder()
    {
        SAFEFREE(value.S_11);
        SAFEFREE(value.S_12);
        SAFEFREE(value.S_21);
        SAFEFREE(value.S_22);
    }
};

struct VMatrixHolder
{
    VMatrix value{};

    ~VMatrixHolder()
    {
        SAFEFREE(value.V_11);
        SAFEFREE(value.V_12);
        SAFEFREE(value.V_21);
        SAFEFREE(value.V_22);
    }
};

double ComplexError(const Complex& a, const Complex& b)
{
    const double dr = static_cast<double>(complex_real(a) - complex_real(b));
    const double di = static_cast<double>(complex_imag(a) - complex_imag(b));
    return std::sqrt(dr * dr + di * di);
}

Status CheckBlockMatrix(const std::vector<Complex>& matrix,
                        int n,
                        const Complex& diag_value,
                        const Complex& off_value,
                        double tol,
                        const char* name)
{
    const int n2 = n / 2;
    double max_err = 0.0;
    int max_row = 0;
    int max_col = 0;
    Complex max_expected = make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0));

    for (int row = 0; row < n; ++row)
    {
        const int row_mode = row < n2 ? row : row - n2;
        const bool row_top = row < n2;
        for (int col = 0; col < n; ++col)
        {
            const int col_mode = col < n2 ? col : col - n2;
            const bool col_top = col < n2;
            Complex expected = make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0));
            if (row_mode == col_mode)
            {
                expected = (row_top == col_top) ? diag_value : off_value;
            }
            const Complex value = matrix[static_cast<size_t>(col) * n + row];
            const double err = ComplexError(value, expected);
            if (err > max_err)
            {
                max_err = err;
                max_row = row;
                max_col = col;
                max_expected = expected;
            }
        }
    }

    if (max_err > tol)
    {
        const Complex actual = matrix[static_cast<size_t>(max_col) * n + max_row];
        std::cout << name << " check failed at (" << max_row << "," << max_col
                  << "), max_err=" << max_err << "\n";
        std::cout << "Expected=" << FormatComplex(max_expected, tol)
                  << ", got=" << FormatComplex(actual, tol) << "\n";
        if (n <= 8)
        {
            PrintMatrix(matrix, n, n, name);
        }
        return Status::kInvalidValue;
    }

    return Status::kSuccess;
}

Status RunIdentitySMatrixTest(int n)
{
    if (n <= 0 || (n % 2) != 0)
    {
        return Status::kInvalidValue;
    }

    const int64_t elements = static_cast<int64_t>(n) * n;
    const int64_t n2 = n / 2;

    std::vector<Complex> h_identity(elements, make_complex(static_cast<Real>(0.0),
                                                           static_cast<Real>(0.0)));
    for (int i = 0; i < n; ++i)
    {
        const int64_t idx = static_cast<int64_t>(i) * n + i;
        h_identity[static_cast<size_t>(idx)] = make_complex(static_cast<Real>(1.0),
                                                            static_cast<Real>(0.0));
    }

    std::vector<Complex> h_diag(static_cast<size_t>(n2),
                                make_complex(static_cast<Real>(1.0), static_cast<Real>(0.0)));

    TMatrixHolder T;
    VMatrixHolder V;
    SMatrixHolder S;

    CHECK(cudaMalloc(reinterpret_cast<void**>(&T.value.T_11), elements * sizeof(Complex)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&T.value.T_12), elements * sizeof(Complex)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&T.value.T_21), elements * sizeof(Complex)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&T.value.T_22), elements * sizeof(Complex)));

    CHECK(cudaMemcpy(
        T.value.T_11, h_identity.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        T.value.T_12, h_identity.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        T.value.T_21, h_identity.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        T.value.T_22, h_identity.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(reinterpret_cast<void**>(&V.value.V_11), n2 * sizeof(Complex)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&V.value.V_12), n2 * sizeof(Complex)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&V.value.V_21), n2 * sizeof(Complex)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&V.value.V_22), n2 * sizeof(Complex)));

    CHECK(cudaMemcpy(
        V.value.V_11, h_diag.data(), n2 * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        V.value.V_12, h_diag.data(), n2 * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        V.value.V_21, h_diag.data(), n2 * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        V.value.V_22, h_diag.data(), n2 * sizeof(Complex), cudaMemcpyHostToDevice));

    CHECK(ComputeSFromT(T.value, V.value, n, &S.value));
    CHECK(cudaDeviceSynchronize());

    std::vector<Complex> hS11(static_cast<size_t>(elements));
    std::vector<Complex> hS12(static_cast<size_t>(elements));
    std::vector<Complex> hS21(static_cast<size_t>(elements));
    std::vector<Complex> hS22(static_cast<size_t>(elements));

    CHECK(cudaMemcpy(
        hS11.data(), S.value.S_11, elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        hS12.data(), S.value.S_12, elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        hS21.data(), S.value.S_21, elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        hS22.data(), S.value.S_22, elements * sizeof(Complex), cudaMemcpyDeviceToHost));

    const double tol = static_cast<double>(std::numeric_limits<Real>::epsilon()) * 1000.0;
    const Complex s11_diag =
        make_complex(static_cast<Real>(22.0 / 25.0), static_cast<Real>(-4.0 / 25.0));
    const Complex s11_off =
        make_complex(static_cast<Real>(-28.0 / 25.0), static_cast<Real>(-4.0 / 25.0));
    const Complex s12_diag =
        make_complex(static_cast<Real>(-1.0 / 5.0), static_cast<Real>(2.0 / 5.0));
    const Complex s12_off =
        make_complex(static_cast<Real>(4.0 / 5.0), static_cast<Real>(2.0 / 5.0));

    CHECK(CheckBlockMatrix(hS11, n, s11_diag, s11_off, tol, "S_11"));
    CHECK(CheckBlockMatrix(hS12, n, s12_diag, s12_off, tol, "S_12"));
    CHECK(CheckBlockMatrix(hS21, n, s12_diag, s12_off, tol, "S_21"));
    CHECK(CheckBlockMatrix(hS22, n, s11_diag, s11_off, tol, "S_22"));

    std::cout << "ComputeSFromT identity-block test passed (n=" << n << ").\n";

    return Status::kSuccess;
}

} // namespace

int main(int argc, char** argv)
{
    int n = 4;
    if (argc > 1)
    {
        char* end = nullptr;
        const long parsed = std::strtol(argv[1], &end, 10);
        if (end != argv[1] && parsed > 0)
        {
            n = static_cast<int>(parsed);
        }
    }

    Status status = RunIdentitySMatrixTest(n);
    if (status != Status::kSuccess)
    {
        std::cout << "S-matrix test failed: " << StatusToString(status) << "\n";
        return 1;
    }
    return 0;
}
