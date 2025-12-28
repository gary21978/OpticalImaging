#include <iostream>
#include <vector>

#include "complex_matrix_ops.h"
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

Status RunDemo(int k)
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

    DeviceBuffer dA;
    DeviceBuffer dB;
    DeviceBuffer dC;
    DeviceBuffer dAdd;
    DeviceBuffer dSub;
    DeviceBuffer dX;

    CHECK(cudaMalloc(dA.address(), elements * sizeof(Complex)));
    CHECK(cudaMalloc(dB.address(), elements * sizeof(Complex)));
    CHECK(cudaMalloc(dC.address(), elements * sizeof(Complex)));
    CHECK(cudaMalloc(dAdd.address(), elements * sizeof(Complex)));
    CHECK(cudaMalloc(dSub.address(), elements * sizeof(Complex)));
    CHECK(cudaMalloc(dX.address(), elements * sizeof(Complex)));

    CHECK(cudaMemcpy(
        dA.get(), hA.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        dB.get(), hB.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));

    ComplexMatrixOps ops;
    CHECK(ops.MatMul(dA.get(), dB.get(), dC.get(), n));
    CHECK(ops.MatAdd(dA.get(), dB.get(), dAdd.get(), n));
    CHECK(ops.MatSub(dA.get(), dB.get(), dSub.get(), n));
    CHECK(ops.Solve(dA.get(), dB.get(), dX.get(), n));

    CHECK(cudaMemcpy(
        hC.data(), dC.get(), elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        hAdd.data(), dAdd.get(), elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        hSub.data(), dSub.get(), elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        hX.data(), dX.get(), elements * sizeof(Complex), cudaMemcpyDeviceToHost));

    std::cout << "\n==== demo ====" << "\n";
    std::cout << PRINT_PRECISION;
    PrintMatrix(hA, n, k, "A");
    PrintMatrix(hB, n, k, "B");
    PrintMatrix(hC, n, k, "A * B");
    PrintMatrix(hAdd, n, k, "A + B");
    PrintMatrix(hSub, n, k, "A - B");
    PrintMatrix(hX, n, k, "Solve(A, B) -> A^{-1}");

    return Status::kSuccess;
}

} // namespace

int main()
{
    return RunDemo(0) == Status::kSuccess ? 0 : 1;
}
