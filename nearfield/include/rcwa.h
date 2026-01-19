#pragma once

#include <cstdint>
#include <vector>

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

class RCWA
{
  public:
    RCWA(Real freq, int order_x, int order_y, Real Lx, Real Ly, cudaStream_t stream = nullptr);
    ~RCWA();

    RCWA(const RCWA&) = delete;
    RCWA& operator=(const RCWA&) = delete;

    Status SetIncidentAngles(Real inc_ang, Real azi_ang, Real eps_ref = kOne);

    Status AddLayer(Real thickness, Real eps);
    Status AddLayer(Real thickness, Complex eps);
    Status AddLayerHomogeneous(Real thickness, Real eps);
    Status AddLayerHomogeneous(Real thickness, Complex eps);
    Status AddLayer(Real thickness, const Complex* eps_conv);

    Status MaterialConv(const Complex* material,
                        int64_t nx,
                        int64_t ny,
                        Complex** eps_conv,
                        cudaStream_t stream = nullptr) const;
    Status SolveGlobalSMatrix(SMatrix* S,
                              const SMatrix* S_in = nullptr,
                              const SMatrix* S_out = nullptr,
                              cudaStream_t stream = nullptr) const;

    Status ComputeTFromPQ(const Complex* P,
                          const Complex* Q,
                          int64_t n,
                          Real k0d,
                          TMatrix* T,
                          cudaStream_t stream = nullptr) const;
    Status ComputeSFromT(const TMatrix& T,
                         const VMatrix& V,
                         int64_t n,
                         SMatrix* S,
                         cudaStream_t stream = nullptr) const;

    int64_t order_N() const { return order_N_; }
    const std::vector<SMatrix>& layer_smatrices() const { return layer_smatrices_; }

  private:
    Status BuildKxKy(Real inc_ang, Real azi_ang, Real eps_ref);
    Status BuildV0(Real eps_ref);
    Status BuildPQ(const Complex* eps_conv, int64_t n, Complex* P, Complex* Q);
    Status RepeatSMatrix(SMatrix* S, int repeats, int64_t n);
    Status RSProd(const SMatrix& Sm,
                  const SMatrix& Sn,
                  int64_t n,
                  SMatrix* out,
                  cudaStream_t stream = nullptr) const;

    void FreeLayers();
    void FreeKVectors();

    Real freq_ = kZero;
    Real omega_ = kZero;
    Real Lx_ = kZero;
    Real Ly_ = kZero;
    Real Gx_norm_ = kZero;
    Real Gy_norm_ = kZero;
    int order_x_ = 0;
    int order_y_ = 0;
    int64_t order_N_ = 0;
    bool kvectors_ready_ = false;
    cudaStream_t stream_ = nullptr;

    Complex* kx_diag_ = nullptr;
    Complex* ky_diag_ = nullptr;
    Complex* kx_mat_ = nullptr;
    Complex* ky_mat_ = nullptr;
    VMatrix V0_{};

    std::vector<SMatrix> layer_smatrices_;
};
