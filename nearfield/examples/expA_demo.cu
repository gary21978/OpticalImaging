#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "complex_matrix_ops.cuh"
#include "common_utils.h"
#include "matrix_print_utils.h"
#include "smatrix_from_pq.cuh"

int main()
{
    const int n = 2;
    const int64_t elements = static_cast<int64_t>(n) * n;
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

    ExpAResult result;

    auto cleanup = [&]()
    {
        if (result.expA_11)
        {
            cudaFree(result.expA_11);
        }
        if (result.expA_12)
        {
            cudaFree(result.expA_12);
        }
        if (result.expA_21)
        {
            cudaFree(result.expA_21);
        }
        if (result.expA_22)
        {
            cudaFree(result.expA_22);
        }
        if (dQ)
        {
            cudaFree(dQ);
        }
        if (dP)
        {
            cudaFree(dP);
        }
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

    if (!CheckStatus(ComputeExpAFromPQ(dP, dQ, n, k0d, &result)))
    {
        cleanup();
        return 1;
    }

    std::vector<Complex> hExpA11(elements);
    std::vector<Complex> hExpA12(elements);
    std::vector<Complex> hExpA21(elements);
    std::vector<Complex> hExpA22(elements);

    if (!CheckCuda(cudaMemcpy(hExpA11.data(), result.expA_11,
                              elements * sizeof(Complex), cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(hExpA12.data(), result.expA_12,
                              elements * sizeof(Complex), cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(hExpA21.data(), result.expA_21,
                              elements * sizeof(Complex), cudaMemcpyDeviceToHost)) ||
        !CheckCuda(cudaMemcpy(hExpA22.data(), result.expA_22,
                              elements * sizeof(Complex), cudaMemcpyDeviceToHost)))
    {
        cleanup();
        return 1;
    }

    std::cout << "\n==== ComputeExpAFromPQ demo ====" << "\n";
    std::cout << PRINT_PRECISION;
    PrintMatrix(hP, n, "P");
    PrintMatrix(hQ, n, "Q");
    PrintMatrix(hExpA11, n, "expA_11");
    PrintMatrix(hExpA12, n, "expA_12");
    PrintMatrix(hExpA21, n, "expA_21");
    PrintMatrix(hExpA22, n, "expA_22");

    cleanup();
    return 0;
}
