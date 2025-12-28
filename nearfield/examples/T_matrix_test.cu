#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
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

Status RunSmallTest()
{
    const int n = 2;
    const int64_t elements = static_cast<int64_t>(n) * n;
    const Real k0d = static_cast<Real>(0.25);

    std::vector<Complex> hP(elements);
    std::vector<Complex> hQ(elements);

    auto set = [&](std::vector<Complex>& mat, int row, int col, double real, double imag)
    { mat[col * n + row] = make_complex(static_cast<Real>(real), static_cast<Real>(imag)); };

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

    CHECK(
        cudaMemcpy(hT11.data(), T.value.T_11, elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(
        cudaMemcpy(hT12.data(), T.value.T_12, elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(
        cudaMemcpy(hT21.data(), T.value.T_21, elements * sizeof(Complex), cudaMemcpyDeviceToHost));
    CHECK(
        cudaMemcpy(hT22.data(), T.value.T_22, elements * sizeof(Complex), cudaMemcpyDeviceToHost));

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

    const auto start = std::chrono::high_resolution_clock::now();
    CHECK(ComputeTFromPQ(dP.get(), dQ.get(), n, k0d, &T.value));
    CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n==== Identity test ====" << "\n";
    std::cout << "N=" << n << ", elapsed=" << std::fixed << std::setprecision(3) << ms
              << " ms\n";

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

Status RunMultiStreamTest(int n)
{
    if (n <= 0)
    {
        return Status::kInvalidValue;
    }

    const int64_t elements = static_cast<int64_t>(n) * n;
    const Real k0d = static_cast<Real>(1.0);
    const Real scales[4] = {static_cast<Real>(1.0),
                            static_cast<Real>(1.1),
                            static_cast<Real>(1.2),
                            static_cast<Real>(1.3)};

    struct StreamHolder
    {
        cudaStream_t stream = nullptr;
        ~StreamHolder()
        {
            if (stream)
            {
                cudaStreamDestroy(stream);
            }
        }
    };

    StreamHolder streams[4];
    for (int i = 0; i < 4; ++i)
    {
        CHECK(cudaStreamCreate(&streams[i].stream));
    }

    std::cout << "\n==== Multi-stream test ====" << "\n";
    CHECK(cudaDeviceSynchronize());

    std::vector<std::vector<Complex>> hP(4);
    std::vector<std::vector<Complex>> hQ(4);
    DeviceBuffer dP[4];
    DeviceBuffer dQ[4];
    TMatrixHolder T[4];

    for (int i = 0; i < 4; ++i)
    {
        hP[i].assign(static_cast<size_t>(elements),
                     make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0)));
        hQ[i].assign(static_cast<size_t>(elements),
                     make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0)));
        for (int j = 0; j < n; ++j)
        {
            const int64_t idx = static_cast<int64_t>(j) * n + j;
            hP[i][static_cast<size_t>(idx)] = make_complex(scales[i], static_cast<Real>(0.0));
            hQ[i][static_cast<size_t>(idx)] = make_complex(scales[i], static_cast<Real>(0.0));
        }

        CHECK(cudaMalloc(dP[i].address(), elements * sizeof(Complex)));
        CHECK(cudaMalloc(dQ[i].address(), elements * sizeof(Complex)));

        CHECK(cudaMemcpy(
            dP[i].get(), hP[i].data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(
            dQ[i].get(), hQ[i].data(), elements * sizeof(Complex), cudaMemcpyHostToDevice));
    }

    const auto start_multi = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; ++i)
    {
        CHECK(ComputeTFromPQ(dP[i].get(), dQ[i].get(), n, k0d, &T[i].value, streams[i].stream));
    }
    CHECK(cudaDeviceSynchronize());
    const auto end_multi = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 4; ++i)
    {
        SAFEFREE(T[i].value.T_11);
        SAFEFREE(T[i].value.T_12);
        SAFEFREE(T[i].value.T_21);
        SAFEFREE(T[i].value.T_22);
    }

    const auto start_single = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; ++i)
    {
        CHECK(ComputeTFromPQ(dP[i].get(), dQ[i].get(), n, k0d, &T[i].value, streams[0].stream));
    }
    CHECK(cudaStreamSynchronize(streams[0].stream));
    const auto end_single = std::chrono::high_resolution_clock::now();

    const double ms_multi =
        std::chrono::duration<double, std::milli>(end_multi - start_multi).count();
    const double ms_single =
        std::chrono::duration<double, std::milli>(end_single - start_single).count();

    std::cout << "4 streams total elapsed=" << std::fixed << std::setprecision(3) << ms_multi
              << " ms\n";
    std::cout << "1 stream total elapsed=" << std::fixed << std::setprecision(3) << ms_single
              << " ms\n";

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
    RunMultiStreamTest(2*PxQ);
    return 0;
}
