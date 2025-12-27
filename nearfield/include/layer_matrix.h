#pragma once

#include <cstdint>
#include "common_utils.h"

struct TMatrix
{
    Complex* T_11 = nullptr;
    Complex* T_12 = nullptr;
    Complex* T_21 = nullptr;
    Complex* T_22 = nullptr;
};

Status ComputeTFromPQ(const Complex* P,
                      const Complex* Q,
                      int64_t n,
                      Real k0d,
                      TMatrix* out,
                      cudaStream_t stream = nullptr);
