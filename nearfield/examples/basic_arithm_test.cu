#include <cstdlib>
#include <iostream>
#include <vector>

#include "common_utils.h"
#include "complex_matrix_ops.h"
#include "matrix_print_utils.h"

namespace
{

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
        SAFEFREE(dX);
        SAFEFREE(dSub);
        SAFEFREE(dAdd);
        SAFEFREE(dC);
        SAFEFREE(dB);
        SAFEFREE(dA);
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

int main()
{
    RunDemo(0);
    return 0;
}
