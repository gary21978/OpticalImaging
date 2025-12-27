#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "complex_matrix_ops.cuh"

struct ExpAResult
{
    Complex* expA_11 = nullptr;
    Complex* expA_12 = nullptr;
    Complex* expA_21 = nullptr;
    Complex* expA_22 = nullptr;
};

Status ComputeExpAFromPQ(const Complex* P,
                         const Complex* Q,
                         int64_t n,
                         Real k0d,
                         ExpAResult* out,
                         cudaStream_t stream = nullptr);

