#pragma once

#include <cstdint>
#include <vector>

#include "common_utils.h"
#include "layer_matrix.h"

class Rcwa
{
  public:
    Rcwa(Real freq, int order_x, int order_y, Real Lx, Real Ly, cudaStream_t stream = nullptr);
    ~Rcwa();

    Rcwa(const Rcwa&) = delete;
    Rcwa& operator=(const Rcwa&) = delete;

    Status SetIncidentAngles(Real inc_ang, Real azi_ang, Real eps_ref = static_cast<Real>(1.0));

    Status AddLayer(Real thickness, Real eps);
    Status AddLayer(Real thickness, Complex eps);
    Status AddLayerHomogeneous(Real thickness, Real eps);
    Status AddLayerHomogeneous(Real thickness, Complex eps);
    Status AddLayer(Real thickness, const Complex* eps_conv);

    int64_t order_N() const { return order_N_; }
    const std::vector<SMatrix>& layer_smatrices() const { return layer_smatrices_; }

  private:
    Status BuildKxKy(Real inc_ang, Real azi_ang, Real eps_ref);
    Status BuildV0(Real eps_ref);
    Status BuildPQ(const Complex* eps_conv, int64_t n, Complex* P, Complex* Q);
    Status RepeatSMatrix(SMatrix* S, int repeats, int64_t n);

    void FreeLayers();
    void FreeKVectors();

    Real freq_ = static_cast<Real>(0.0);
    Real omega_ = static_cast<Real>(0.0);
    Real Lx_ = static_cast<Real>(0.0);
    Real Ly_ = static_cast<Real>(0.0);
    Real Gx_norm_ = static_cast<Real>(0.0);
    Real Gy_norm_ = static_cast<Real>(0.0);
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
