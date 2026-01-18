#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

Status RunIdentityTest(int n)
{
    if (n <= 0)
    {
        return Status::kInvalidValue;
    }

    const int64_t elements = static_cast<int64_t>(n) * n;
    const Real k0d = static_cast<Real>(1.0);

    std::vector<Complex> hP(elements);
    std::vector<Complex> hQ(elements);
    for (int i = 0; i < n; ++i)
    {
        const int64_t idx = static_cast<int64_t>(i) * n + i;
        hP[static_cast<size_t>(idx)] = make_complex(static_cast<Real>(1.0), static_cast<Real>(0.0));
        hQ[static_cast<size_t>(idx)] = make_complex(static_cast<Real>(1.0), static_cast<Real>(0.0));
    }

    DeviceBuffer dP;
    DeviceBuffer dQ;
    TMatrixHolder T;

    CHECK(cudaMalloc(dP.address(), elements * sizeof(Complex)));
    CHECK(cudaMalloc(dQ.address(), elements * sizeof(Complex)));

    CHECK(cudaMemcpy(dP.get(), hP.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dQ.get(), hQ.data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));

    CHECK(ComputeTFromPQ(dP.get(), dQ.get(), n, k0d, &T.value));
    CHECK(cudaDeviceSynchronize());
    SAFEFREE(T.value.T_11);
    SAFEFREE(T.value.T_12);
    SAFEFREE(T.value.T_21);
    SAFEFREE(T.value.T_22);

    const int repeats = 3;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    for (int iter = 0; iter < repeats; ++iter)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        CHECK(ComputeTFromPQ(dP.get(), dQ.get(), n, k0d, &T.value));
        CHECK(cudaDeviceSynchronize());
        const auto end = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);

        if (iter + 1 < repeats)
        {
            SAFEFREE(T.value.T_11);
            SAFEFREE(T.value.T_12);
            SAFEFREE(T.value.T_21);
            SAFEFREE(T.value.T_22);
        }
    }

    std::cout << "\n==== Identity test ====" << "\n";
    std::cout << "N=" << n << ", avg=" << std::fixed << std::setprecision(3)
              << (total_ms / repeats) << " ms, min=" << min_ms << " ms, max=" << max_ms
              << " ms, repeats=" << repeats << "\n";

    if (n >= 2)
    {
        const size_t stride = static_cast<size_t>(n);
        std::vector<Complex> t11(stride * 2);
        std::vector<Complex> t12(stride * 2);
        std::vector<Complex> t21(stride * 2);
        std::vector<Complex> t22(stride * 2);

        CHECK(cudaMemcpy(t11.data(), T.value.T_11, 2 * sizeof(Complex), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(
            t11.data() + stride, T.value.T_11 + n, 2 * sizeof(Complex), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(t12.data(), T.value.T_12, 2 * sizeof(Complex), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(
            t12.data() + stride, T.value.T_12 + n, 2 * sizeof(Complex), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(t21.data(), T.value.T_21, 2 * sizeof(Complex), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(
            t21.data() + stride, T.value.T_21 + n, 2 * sizeof(Complex), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(t22.data(), T.value.T_22, 2 * sizeof(Complex), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(
            t22.data() + stride, T.value.T_22 + n, 2 * sizeof(Complex), cudaMemcpyDeviceToHost));

        PrintMatrix(t11, n, 2, "T_11(2x2)");
        PrintMatrix(t12, n, 2, "T_12(2x2)");
        PrintMatrix(t21, n, 2, "T_21(2x2)");
        PrintMatrix(t22, n, 2, "T_22(2x2)");
    }

    return Status::kSuccess;
}
} // namespace

int main(int argc, char** argv)
{
    int orders = 10;
    if (argc > 1)
    {
        char* end = nullptr;
        const long parsed = std::strtol(argv[1], &end, 10);
        if (end != argv[1] && parsed > 0)
        {
            orders = static_cast<int>(parsed);
        }
    }
    int PxQ = (2 * orders + 1) * (2 * orders + 1);

    //RunSmallTest();
    RunIdentityTest(2*PxQ);
    return 0;
}
