#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_

#include <iostream>

#include <cuda_runtime.h>

#include "complex_matrix_ops.cuh"

inline bool CheckCuda(cudaError_t error)
{
    if (error == cudaSuccess)
    {
        return true;
    }
    std::cerr << "cuda error: " << cudaGetErrorString(error) << "\n";
    return false;
}

inline bool CheckStatus(Status status)
{
    if (status == Status::kSuccess)
    {
        return true;
    }
    std::cerr << "status: " << StatusToString(status) << "\n";
    return false;
}

#endif  // COMMON_UTILS_H_
