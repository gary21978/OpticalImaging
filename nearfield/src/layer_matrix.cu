#include "complex_matrix_ops.h"
#include "layer_matrix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{

constexpr double kExpConstant[5][5] = {
    {0.0, -0.10036558103014462001, -0.00802924648241156960, -0.00089213849804572995, 0.0},
    {0.0,
     0.39784974949964507614,
     1.36783778460411719922,
     0.49828962252538267755,
     -0.00063789819459472330},
    {-10.9676396052962062593,
     1.68015813878906197182,
     0.05717798464788655127,
     -0.00698210122488052084,
     0.00003349750170860705},
    {-0.0904316832390810561,
     -0.06764045190713819075,
     0.06759613017704596460,
     0.02955525704293155274,
     -0.00001391802575160607},
    {0.0, 0.0, -0.09233646193671185927, -0.01693649390020817171, -0.00001400867981820361}};

struct DeviceMatrix
{
    Complex* ptr = nullptr;
    size_t elements = 0;

    DeviceMatrix() = default;
    ~DeviceMatrix()
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;

    Complex* get() const { return ptr; }

    void Reset(Complex* p = nullptr, size_t count = 0)
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
        ptr = p;
        elements = count;
    }

    Complex* Release()
    {
        Complex* out = ptr;
        ptr = nullptr;
        elements = 0;
        return out;
    }

    void Swap(DeviceMatrix& other)
    {
        Complex* tmp_ptr = ptr;
        ptr = other.ptr;
        other.ptr = tmp_ptr;
        size_t tmp_elements = elements;
        elements = other.elements;
        other.elements = tmp_elements;
    }
};

__device__ inline Complex Multiply(Complex a, Complex b)
{
    const Real ar = complex_real(a);
    const Real ai = complex_imag(a);
    const Real br = complex_real(b);
    const Real bi = complex_imag(b);
    return make_complex(ar * br - ai * bi, ar * bi + ai * br);
}

__global__ void ScaleKernel(const Complex* A, Complex* C, int64_t count, Complex scalar)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        C[idx] = Multiply(A[idx], scalar);
    }
}

__global__ void IdentityKernel(Complex* A, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n * n;
    if (idx < total)
    {
        int64_t row = idx % n;
        int64_t col = idx / n;
        if (row == col)
        {
            A[idx] = make_complex(static_cast<Real>(1.0), static_cast<Real>(0.0));
        }
        else
        {
            A[idx] = make_complex(static_cast<Real>(0.0), static_cast<Real>(0.0));
        }
    }
}

Status StatusFromCuda(cudaError_t status)
{
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

Status AllocateMatrix(int64_t rows, int64_t cols, DeviceMatrix* out)
{
    if (!out || rows <= 0 || cols <= 0)
    {
        return Status::kInvalidValue;
    }
    const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (count > (std::numeric_limits<size_t>::max() / sizeof(Complex)))
    {
        return Status::kInvalidValue;
    }
    if (out->ptr && out->elements == count)
    {
        return Status::kSuccess;
    }
    Complex* ptr = nullptr;
    if (cudaMalloc(&ptr, count * sizeof(Complex)) != cudaSuccess)
    {
        return Status::kCudaError;
    }
    out->Reset(ptr, count);
    return Status::kSuccess;
}

Status MakeIdentity(int64_t n, DeviceMatrix* out, cudaStream_t stream)
{
    CHECK(AllocateMatrix(n, n, out));

    int64_t total = n * n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    IdentityKernel<<<blocks, threads, 0, stream>>>(out->get(), n);
    return StatusFromCuda(cudaGetLastError());
}

Status
ScaleMatrix(const Complex* A, int64_t n, Complex scalar, DeviceMatrix* out, cudaStream_t stream)
{
    CHECK(AllocateMatrix(n, n, out));

    int64_t count = n * n;
    int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    ScaleKernel<<<blocks, threads, 0, stream>>>(A, out->get(), count, scalar);
    return StatusFromCuda(cudaGetLastError());
}

Status
AddMatrix(ComplexMatrixOps& ops, const Complex* A, const Complex* B, int64_t n, DeviceMatrix* out)
{
    CHECK(AllocateMatrix(n, n, out));
    return ops.MatAdd(A, B, out->get(), n, n);
}

Status AddInPlace(ComplexMatrixOps& ops, Complex* A, const Complex* B, int64_t n)
{
    return ops.MatAdd(A, B, A, n, n);
}

Status
MatMul(ComplexMatrixOps& ops, const Complex* A, const Complex* B, int64_t n, DeviceMatrix* out)
{
    CHECK(AllocateMatrix(n, n, out));
    return ops.MatMul(A, B, out->get(), n, n, n);
}

Status ComputeMatrixNorm1(const Complex* A, int64_t n, double* out_norm, cudaStream_t stream)
{
    if (!A || !out_norm || n <= 0)
    {
        return Status::kInvalidValue;
    }

    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
    std::vector<Complex> host(count);

    CHECK(StatusFromCuda(
        cudaMemcpyAsync(host.data(), A, count * sizeof(Complex), cudaMemcpyDeviceToHost, stream)));

    CHECK(StatusFromCuda(cudaStreamSynchronize(stream)));

    double max_norm = 0.0;
    for (int64_t col = 0; col < n; ++col)
    {
        double sum = 0.0;
        const size_t base = static_cast<size_t>(col) * static_cast<size_t>(n);
        for (int64_t row = 0; row < n; ++row)
        {
            const Complex value = host[base + static_cast<size_t>(row)];
            const double real = static_cast<double>(complex_real(value));
            const double imag = static_cast<double>(complex_imag(value));
            sum += std::hypot(real, imag);
        }
        if (sum > max_norm)
        {
            max_norm = sum;
        }
    }

    *out_norm = max_norm;
    return Status::kSuccess;
}

} // namespace

Status ComputeTFromPQ(const Complex* P,
                      const Complex* Q,
                      int64_t n,
                      Real k0d,
                      TMatrix* out,
                      cudaStream_t stream)
{
    if (!P || !Q || !out || n <= 0)
    {
        return Status::kInvalidValue;
    }

    out->T_11 = nullptr;
    out->T_12 = nullptr;
    out->T_21 = nullptr;
    out->T_22 = nullptr;

    ComplexMatrixOps ops(stream);
    CHECK(ops.SetStream(stream));

    double Pnorm = 0.0;
    double Qnorm = 0.0;
    CHECK(ComputeMatrixNorm1(P, n, &Pnorm, stream));
    CHECK(ComputeMatrixNorm1(Q, n, &Qnorm, stream));

    const double Rnorm = std::max(Pnorm, Qnorm);
    const double scaled = Rnorm * static_cast<double>(k0d);
    int m = 0;
    if (scaled > 0.0)
    {
        m = static_cast<int>(std::ceil(std::log2(scaled)));
        if (m < 0)
        {
            m = 0;
        }
    }

    const double scale_A = std::pow(2.0, -m) * static_cast<double>(k0d);
    const Complex scale_pos = make_complex(static_cast<Real>(scale_A), static_cast<Real>(0.0));
    const Complex scale_neg = make_complex(static_cast<Real>(-scale_A), static_cast<Real>(0.0));

    DeviceMatrix I2;
    CHECK(MakeIdentity(n, &I2, stream));

    DeviceMatrix A_12;
    CHECK(ScaleMatrix(P, n, scale_neg, &A_12, stream));

    DeviceMatrix A_21;
    CHECK(ScaleMatrix(Q, n, scale_pos, &A_21, stream));

    DeviceMatrix A2_11;
    CHECK(MatMul(ops, A_12.get(), A_21.get(), n, &A2_11));

    DeviceMatrix A2_22;
    CHECK(MatMul(ops, A_21.get(), A_12.get(), n, &A2_22));

    DeviceMatrix A3_12;
    CHECK(MatMul(ops, A_12.get(), A2_22.get(), n, &A3_12));

    DeviceMatrix A3_21;
    CHECK(MatMul(ops, A_21.get(), A2_11.get(), n, &A3_21));

    DeviceMatrix A6_11;
    CHECK(MatMul(ops, A3_12.get(), A3_21.get(), n, &A6_11));

    DeviceMatrix A6_22;
    CHECK(MatMul(ops, A3_21.get(), A3_12.get(), n, &A6_22));

    DeviceMatrix scratch;

    DeviceMatrix B_11[5];
    DeviceMatrix B_12[5];
    DeviceMatrix B_21[5];
    DeviceMatrix B_22[5];

    for (int i = 0; i < 5; ++i)
    {
        const Complex c0 =
            make_complex(static_cast<Real>(kExpConstant[i][0]), static_cast<Real>(0.0));
        const Complex c1 =
            make_complex(static_cast<Real>(kExpConstant[i][1]), static_cast<Real>(0.0));
        const Complex c2 =
            make_complex(static_cast<Real>(kExpConstant[i][2]), static_cast<Real>(0.0));
        const Complex c3 =
            make_complex(static_cast<Real>(kExpConstant[i][3]), static_cast<Real>(0.0));
        const Complex c4 =
            make_complex(static_cast<Real>(kExpConstant[i][4]), static_cast<Real>(0.0));

        CHECK(ScaleMatrix(I2.get(), n, c0, &B_11[i], stream));
        CHECK(ScaleMatrix(A2_11.get(), n, c2, &scratch, stream));
        CHECK(AddInPlace(ops, B_11[i].get(), scratch.get(), n));
        CHECK(ScaleMatrix(A6_11.get(), n, c4, &scratch, stream));
        CHECK(AddInPlace(ops, B_11[i].get(), scratch.get(), n));

        CHECK(ScaleMatrix(A_12.get(), n, c1, &B_12[i], stream));
        CHECK(ScaleMatrix(A3_12.get(), n, c3, &scratch, stream));
        CHECK(AddInPlace(ops, B_12[i].get(), scratch.get(), n));

        CHECK(ScaleMatrix(A_21.get(), n, c1, &B_21[i], stream));
        CHECK(ScaleMatrix(A3_21.get(), n, c3, &scratch, stream));
        CHECK(AddInPlace(ops, B_21[i].get(), scratch.get(), n));

        CHECK(ScaleMatrix(I2.get(), n, c0, &B_22[i], stream));
        CHECK(ScaleMatrix(A2_22.get(), n, c2, &scratch, stream));
        CHECK(AddInPlace(ops, B_22[i].get(), scratch.get(), n));
        CHECK(ScaleMatrix(A6_22.get(), n, c4, &scratch, stream));
        CHECK(AddInPlace(ops, B_22[i].get(), scratch.get(), n));
    }

    A2_11.Reset();
    A2_22.Reset();
    A3_12.Reset();
    A3_21.Reset();
    A6_11.Reset();
    A6_22.Reset();

    DeviceMatrix A9_11;
    CHECK(MatMul(ops, B_11[0].get(), B_11[4].get(), n, &A9_11));
    CHECK(MatMul(ops, B_12[0].get(), B_21[4].get(), n, &scratch));
    CHECK(AddInPlace(ops, A9_11.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_11.get(), B_11[3].get(), n));

    DeviceMatrix A9_12;
    CHECK(MatMul(ops, B_11[0].get(), B_12[4].get(), n, &A9_12));
    CHECK(MatMul(ops, B_12[0].get(), B_22[4].get(), n, &scratch));
    CHECK(AddInPlace(ops, A9_12.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_12.get(), B_12[3].get(), n));

    DeviceMatrix A9_21;
    CHECK(MatMul(ops, B_21[0].get(), B_11[4].get(), n, &A9_21));
    CHECK(MatMul(ops, B_22[0].get(), B_21[4].get(), n, &scratch));
    CHECK(AddInPlace(ops, A9_21.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_21.get(), B_21[3].get(), n));

    DeviceMatrix A9_22;
    CHECK(MatMul(ops, B_21[0].get(), B_12[4].get(), n, &A9_22));
    CHECK(MatMul(ops, B_22[0].get(), B_22[4].get(), n, &scratch));
    CHECK(AddInPlace(ops, A9_22.get(), scratch.get(), n));
    CHECK(AddInPlace(ops, A9_22.get(), B_22[3].get(), n));

    DeviceMatrix temp1;
    DeviceMatrix temp2;
    DeviceMatrix temp3;

    DeviceMatrix T_11;
    {
        CHECK(AddMatrix(ops, B_11[2].get(), A9_11.get(), n, &temp1));
        CHECK(MatMul(ops, temp1.get(), A9_11.get(), n, &temp2));
        CHECK(AddMatrix(ops, B_12[2].get(), A9_12.get(), n, &temp1));
        CHECK(MatMul(ops, temp1.get(), A9_21.get(), n, &temp3));
        CHECK(AddMatrix(ops, B_11[1].get(), temp2.get(), n, &T_11));
        CHECK(AddInPlace(ops, T_11.get(), temp3.get(), n));
    }

    DeviceMatrix T_12;
    {
        CHECK(AddMatrix(ops, B_11[2].get(), A9_11.get(), n, &temp1));
        CHECK(MatMul(ops, temp1.get(), A9_12.get(), n, &temp2));
        CHECK(AddMatrix(ops, B_12[2].get(), A9_12.get(), n, &temp1));
        CHECK(MatMul(ops, temp1.get(), A9_22.get(), n, &temp3));
        CHECK(AddMatrix(ops, B_12[1].get(), temp2.get(), n, &T_12));
        CHECK(AddInPlace(ops, T_12.get(), temp3.get(), n));
    }

    DeviceMatrix T_21;
    {
        CHECK(AddMatrix(ops, B_21[2].get(), A9_21.get(), n, &temp1));
        CHECK(MatMul(ops, temp1.get(), A9_11.get(), n, &temp2));
        CHECK(AddMatrix(ops, B_22[2].get(), A9_22.get(), n, &temp1));
        CHECK(MatMul(ops, temp1.get(), A9_21.get(), n, &temp3));
        CHECK(AddMatrix(ops, B_21[1].get(), temp2.get(), n, &T_21));
        CHECK(AddInPlace(ops, T_21.get(), temp3.get(), n));
    }

    DeviceMatrix T_22;
    {
        CHECK(AddMatrix(ops, B_21[2].get(), A9_21.get(), n, &temp1));
        CHECK(MatMul(ops, temp1.get(), A9_12.get(), n, &temp2));
        CHECK(AddMatrix(ops, B_22[2].get(), A9_22.get(), n, &temp1));
        CHECK(MatMul(ops, temp1.get(), A9_22.get(), n, &temp3));
        CHECK(AddMatrix(ops, B_22[1].get(), temp2.get(), n, &T_22));
        CHECK(AddInPlace(ops, T_22.get(), temp3.get(), n));
    }

    DeviceMatrix tmp11;
    DeviceMatrix tmp12;
    DeviceMatrix tmp21;
    DeviceMatrix tmp22;
    for (int i = 0; i < m; ++i)
    {
        CHECK(MatMul(ops, T_11.get(), T_11.get(), n, &tmp11));
        CHECK(MatMul(ops, T_12.get(), T_21.get(), n, &scratch));
        CHECK(AddInPlace(ops, tmp11.get(), scratch.get(), n));
        CHECK(MatMul(ops, T_11.get(), T_12.get(), n, &tmp12));
        CHECK(MatMul(ops, T_12.get(), T_22.get(), n, &scratch));
        CHECK(AddInPlace(ops, tmp12.get(), scratch.get(), n));
        CHECK(MatMul(ops, T_21.get(), T_11.get(), n, &tmp21));
        CHECK(MatMul(ops, T_22.get(), T_21.get(), n, &scratch));
        CHECK(AddInPlace(ops, tmp21.get(), scratch.get(), n));
        CHECK(MatMul(ops, T_21.get(), T_12.get(), n, &tmp22));
        CHECK(MatMul(ops, T_22.get(), T_22.get(), n, &scratch));
        CHECK(AddInPlace(ops, tmp22.get(), scratch.get(), n));
        T_11.Swap(tmp11);
        T_12.Swap(tmp12);
        T_21.Swap(tmp21);
        T_22.Swap(tmp22);
    }

    out->T_11 = T_11.Release();
    out->T_12 = T_12.Release();
    out->T_21 = T_21.Release();
    out->T_22 = T_22.Release();
    return Status::kSuccess;
}
