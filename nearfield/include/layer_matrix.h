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

struct SMatrix
{
    Complex* S_11 = nullptr;
    Complex* S_12 = nullptr;
    Complex* S_21 = nullptr;
    Complex* S_22 = nullptr;
};

struct VMatrix
{
    // diagonal blocks
    Complex* V_11 = nullptr;
    Complex* V_12 = nullptr;
    Complex* V_21 = nullptr;
    Complex* V_22 = nullptr;
};

Status ComputeTFromPQ(const Complex* P,
                      const Complex* Q,
                      int64_t n,
                      Real k0d,
                      TMatrix* T,
                      cudaStream_t stream = nullptr);

Status ComputeSFromT(const TMatrix& T,
                     const VMatrix& V,
                     int64_t n,
                     SMatrix* S,
                     cudaStream_t stream = nullptr);