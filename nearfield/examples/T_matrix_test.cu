#include <cstdint>
#include <iostream>
#include <vector>

#include "layer_matrix.h"
#include "matrix_print_utils.h"

namespace
{

struct DeviceBuffer
{
    Complex* ptr = nullptr;

    ~DeviceBuffer() { SAFEFREE(ptr); }

    Complex** address() { return &ptr; }
    Complex* get() const { return ptr; }
};

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

Status RunTest()
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

    DeviceBuffer dP;
    DeviceBuffer dQ;
    TMatrixHolder T;

    CHECK(cudaMalloc(dP.address(), elements * sizeof(Complex)));
    CHECK(cudaMalloc(dQ.address(), elements * sizeof(Complex)));

    CHECK(cudaMemcpy(dP.get(), hP.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dQ.get(), hQ.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));

    CHECK(ComputeTFromPQ(dP.get(), dQ.get(), n, k0d, &T.value));

    std::vector<Complex> hT11(elements);
    std::vector<Complex> hT12(elements);
    std::vector<Complex> hT21(elements);
    std::vector<Complex> hT22(elements);

    CHECK(cudaMemcpy(
        hT11.data(), T.value.T_11, elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        hT12.data(), T.value.T_12, elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        hT21.data(), T.value.T_21, elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        hT22.data(), T.value.T_22, elements * sizeof(Complex), cudaMemcpyDeviceToHost));

    std::cout << "\n==== ComputeTFromPQ demo ====" << "\n";
    std::cout << PRINT_PRECISION;
    PrintMatrix(hP, n, "P");
    PrintMatrix(hQ, n, "Q");
    PrintMatrix(hT11, n, "T_11");
    PrintMatrix(hT12, n, "T_12");
    PrintMatrix(hT21, n, "T_21");
    PrintMatrix(hT22, n, "T_22");

    return Status::kSuccess;
}

} // namespace

int main()
{
    return RunTest() == Status::kSuccess ? 0 : 1;
}
