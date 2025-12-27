#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "complex_matrix_ops.cuh"

namespace
{

bool CheckCuda(cudaError_t error)
{
    if (error == cudaSuccess)
    {
        return true;
    }
    std::cerr << "cuda error: " << cudaGetErrorString(error) << "\n";
    return false;
}

bool CheckStatus(Status status)
{
    if (status == Status::kSuccess)
    {
        return true;
    }
    std::cerr << "status: " << StatusToString(status) << "\n";
    return false;
}

std::string FormatComplex(const Complex& value, double eps)
{
    double real = static_cast<double>(complex_real(value));
    double imag = static_cast<double>(complex_imag(value));

    if (std::abs(real) <= eps)
    {
        real = 0.0;
    }
    if (std::abs(imag) <= eps)
    {
        imag = 0.0;
    }

    std::ostringstream oss;
    oss << PRINT_PRECISION;

    if (imag == 0.0)
    {
        oss << real;
        return oss.str();
    }

    const double abs_imag = std::abs(imag);
    const bool imag_is_one = std::abs(abs_imag - 1.0) <= eps;

    if (real == 0.0)
    {
        if (imag < 0.0)
        {
            oss << "-";
        }
        if (imag_is_one)
        {
            oss << "i";
        }
        else
        {
            oss << abs_imag << "i";
        }
        return oss.str();
    }

    oss << real;
    if (imag > 0.0)
    {
        oss << "+";
    }
    else
    {
        oss << "-";
    }

    if (imag_is_one)
    {
        oss << "i";
    }
    else
    {
        oss << abs_imag << "i";
    }

    return oss.str();
}

void PrintMatrix(const std::vector<Complex>& matrix, int n, int k, const char* name)
{
    std::cout << name << " =\n";
    const int limit = k > 0 ? std::min(n, k) : n;
    const double eps = static_cast<double>(std::numeric_limits<Real>::epsilon()) * 10.0;
    std::vector<std::string> formatted(static_cast<size_t>(limit) * limit);
    std::vector<size_t> widths(static_cast<size_t>(limit), 0);

    for (int row = 0; row < limit; ++row)
    {
        for (int col = 0; col < limit; ++col)
        {
            const Complex value = matrix[col * n + row];
            std::string text = FormatComplex(value, eps);
            const size_t idx = static_cast<size_t>(col) * limit + row;
            formatted[idx] = std::move(text);
            widths[static_cast<size_t>(col)] =
                std::max(widths[static_cast<size_t>(col)], formatted[idx].size());
        }
    }

    std::cout << std::right;
    for (int row = 0; row < limit; ++row)
    {
        for (int col = 0; col < limit; ++col)
        {
            const size_t idx = static_cast<size_t>(col) * limit + row;
            std::cout << std::setw(static_cast<int>(widths[static_cast<size_t>(col)]))
                      << formatted[idx];
            if (col + 1 < limit)
            {
                std::cout << "  ";
            }
        }
        std::cout << "\n";
    }
}

void RunDemo(int k)
{
    const int n = 2;
    const int64_t elements = static_cast<int64_t>(n) * n;

    std::vector<Complex> hA(elements);
    std::vector<Complex> hB(elements);
    std::vector<Complex> hC(elements);
    std::vector<Complex> hAdd(elements);
    std::vector<Complex> hSub(elements);
    std::vector<Complex> hX(elements);

    const Real one = static_cast<Real>(1.0);
    const Real zero = static_cast<Real>(0.0);

    hA[0] = make_complex(one, zero);
    hA[1] = make_complex(static_cast<Real>(3.0), static_cast<Real>(-1.0));
    hA[2] = make_complex(static_cast<Real>(2.0), static_cast<Real>(1.0));
    hA[3] = make_complex(static_cast<Real>(4.0), zero);

    hB[0] = make_complex(one, zero);
    hB[1] = make_complex(zero, zero);
    hB[2] = make_complex(zero, zero);
    hB[3] = make_complex(one, zero);

    Complex* dA = nullptr;
    Complex* dB = nullptr;
    Complex* dC = nullptr;
    Complex* dAdd = nullptr;
    Complex* dSub = nullptr;
    Complex* dX = nullptr;

    auto cleanup = [&]()
    {
        if (dX)
        {
            cudaFree(dX);
        }
        if (dSub)
        {
            cudaFree(dSub);
        }
        if (dAdd)
        {
            cudaFree(dAdd);
        }
        if (dC)
        {
            cudaFree(dC);
        }
        if (dB)
        {
            cudaFree(dB);
        }
        if (dA)
        {
            cudaFree(dA);
        }
    };

    if (!CheckCuda(cudaMalloc(&dA, elements * sizeof(Complex))) ||
        !CheckCuda(cudaMalloc(&dB, elements * sizeof(Complex))) ||
        !CheckCuda(cudaMalloc(&dC, elements * sizeof(Complex))) ||
        !CheckCuda(cudaMalloc(&dAdd, elements * sizeof(Complex))) ||
        !CheckCuda(cudaMalloc(&dSub, elements * sizeof(Complex))) ||
        !CheckCuda(cudaMalloc(&dX, elements * sizeof(Complex))))
    {
        cleanup();
        return;
    }

    if (!CheckCuda(cudaMemcpy(
            dA, hA.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice)) ||
        !CheckCuda(cudaMemcpy(
            dB, hB.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice)))
    {
        cleanup();
        return;
    }

    ComplexMatrixOps ops;
    if (!CheckStatus(ops.MatMul(dA, dB, dC, n)) ||
        !CheckStatus(ops.MatAdd(dA, dB, dAdd, n)) ||
        !CheckStatus(ops.MatSub(dA, dB, dSub, n)) ||
        !CheckStatus(ops.Solve(dA, dB, dX, n)))
    {
        cleanup();
        return;
    }

    if (!CheckCuda(cudaMemcpy(
            hC.data(), dC, elements * sizeof(Complex), cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(
            hAdd.data(), dAdd, elements * sizeof(Complex), cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(
            hSub.data(), dSub, elements * sizeof(Complex), cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(
            hX.data(), dX, elements * sizeof(Complex), cudaMemcpyDeviceToHost)))
    {
        cleanup();
        return;
    }

    std::cout << "\n==== demo ====" << "\n";
    std::cout << PRINT_PRECISION;
    PrintMatrix(hA, n, k, "A");
    PrintMatrix(hB, n, k, "B");
    PrintMatrix(hC, n, k, "A * B");
    PrintMatrix(hAdd, n, k, "A + B");
    PrintMatrix(hSub, n, k, "A - B");
    PrintMatrix(hX, n, k, "Solve(A, B) -> A^{-1}");

    cleanup();
}

} // namespace

int main(int argc, char** argv)
{
    int k = 0;
    if (argc > 1)
    {
        char* end = nullptr;
        const long parsed = std::strtol(argv[1], &end, 10);
        if (end != argv[1] && parsed > 0)
        {
            k = static_cast<int>(parsed);
        }
    }

    RunDemo(k);
    return 0;
}
