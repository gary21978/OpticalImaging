#include <iostream>
#include <vector>

#include "common_utils.h"
#include "matrix_print_utils.h"
#include "layer_matrix.h"

int main()
{
    const int n = 2;
    const uint64_t elements = static_cast<int64_t>(n * n);
    const Real k0d = static_cast<Real>(0.25);

    std::vector<Complex> hP(elements);
    std::vector<Complex> hQ(elements);

    auto set = [&](std::vector<Complex>& mat, int row, int col, double real, double imag)
    {
        mat[col * n + row] = make_complex(static_cast<Real>(real), static_cast<Real>(imag));
    };

    set(hP, 0, 0, 1.0, 0.0);
    set(hP, 1, 0, 0.3, -0.2);
    set(hP, 0, 1, 0.2, 0.1);
    set(hP, 1, 1, 0.8, 0.0);

    set(hQ, 0, 0, 0.9, 0.0);
    set(hQ, 1, 0, 0.4, 0.3);
    set(hQ, 0, 1, -0.1, 0.05);
    set(hQ, 1, 1, 0.7, 0.0);

    Complex* dP = nullptr;
    Complex* dQ = nullptr;

    TMatrix T{};

    auto cleanup = [&]()
    {
        SAFEFREE(T.T_11);
        SAFEFREE(T.T_12);
        SAFEFREE(T.T_21);
        SAFEFREE(T.T_22);
        SAFEFREE(dQ);
        SAFEFREE(dP);
    };

    if (!CheckCuda(cudaMalloc(&dP, elements * sizeof(Complex))) ||
        !CheckCuda(cudaMalloc(&dQ, elements * sizeof(Complex))))
    {
        cleanup();
        return 1;
    }

    if (!CheckCuda(cudaMemcpy(dP, hP.data(), elements * sizeof(Complex),
                              cudaMemcpyHostToDevice)) ||
        !CheckCuda(cudaMemcpy(dQ, hQ.data(), elements * sizeof(Complex),
                              cudaMemcpyHostToDevice)))
    {
        cleanup();
        return 1;
    }

    if (!CheckStatus(ComputeTFromPQ(dP, dQ, n, k0d, &T)))
    {
        cleanup();
        return 1;
    }

    std::vector<Complex> hT11(elements);
    std::vector<Complex> hT12(elements);
    std::vector<Complex> hT21(elements);
    std::vector<Complex> hT22(elements);

    if (!CheckCuda(cudaMemcpy(hT11.data(), T.T_11,
                              elements * sizeof(Complex), cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(hT12.data(), T.T_12,
                              elements * sizeof(Complex), cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(hT21.data(), T.T_21,
                              elements * sizeof(Complex), cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(hT22.data(), T.T_22,
                              elements * sizeof(Complex), cudaMemcpyDeviceToHost)))
    {
        cleanup();
        return 1;
    }

    std::cout << "\n==== ComputeTFromPQ demo ====" << "\n";
    std::cout << PRINT_PRECISION;
    PrintMatrix(hP, n, "P");
    PrintMatrix(hQ, n, "Q");
    PrintMatrix(hT11, n, "T_11");
    PrintMatrix(hT12, n, "T_12");
    PrintMatrix(hT21, n, "T_21");
    PrintMatrix(hT22, n, "T_22");

    cleanup();
    return 0;
}
