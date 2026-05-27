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

class ComplexMatrixOps;

class RCWA
{
  public:
    RCWA(Real freq, int order_x, int order_y, Real Lx, Real Ly, cudaStream_t stream = nullptr);
    ~RCWA();

    RCWA(const RCWA&) = delete;
    RCWA& operator=(const RCWA&) = delete;

    Status SetStream(cudaStream_t stream);
    Status SetIncidentAngles(Real inc_ang, Real azi_ang, Real eps_ref = 1.0);
    Status SetIncidentAngles(Real inc_ang, Real azi_ang, const char* angle_layer);

    Status AddLayer(Real thickness, Real eps);
    Status AddLayer(Real thickness, Complex eps);
    Status AddLayerHomogeneous(Real thickness, Real eps);
    Status AddLayerHomogeneous(Real thickness, Complex eps);
    Status AddLayer(Real thickness, const Complex* eps_conv);
    Status AddInputLayer(Real eps);
    Status AddInputLayer(Complex eps);
    Status AddOutputLayer(Real eps);
    Status AddOutputLayer(Complex eps);

    Status SourcePlanewave(Real ex_amp, Real ey_amp, const char* direction = "forward");
    Status SourcePlanewave(Complex ex_amp, Complex ey_amp, const char* direction = "forward");

    Status MaterialConv(const Complex* material,
                        int64_t nx,
                        int64_t ny,
                        Complex** eps_conv,
                        cudaStream_t stream = nullptr) const;
    Status SolveGlobalSMatrix(SMatrix* S,
                              const SMatrix* S_in = nullptr,
                              const SMatrix* S_out = nullptr,
                              cudaStream_t stream = nullptr) const;
    Status SolveGlobalSMatrix();
    Status FieldXY(const Real* x_axis,
                   int64_t x_count,
                   const Real* y_axis,
                   int64_t y_count,
                   Complex** Ex,
                   Complex** Ey,
                   Complex** Ez,
                   cudaStream_t stream = nullptr) const;
    void FreeDeviceTensor(Complex* ptr) const;

    Status ComputeTFromPQOrder18(const Complex* P,
                                 const Complex* Q,
                                 int64_t n,
                                 Real k0d,
                                 TMatrix* T,
                                 cudaStream_t stream = nullptr) const;
    Status ComputeTFromPQOrder12(const Complex* P,
                                 const Complex* Q,
                                 int64_t n,
                                 Real k0d,
                                 TMatrix* T,
                                 cudaStream_t stream = nullptr) const;
    Status ComputeCosSinFromThetaOrder12(const Complex* Theta,
                                          int64_t n,
                                          Complex* Cos,
                                          Complex* Sin,
                                          cudaStream_t stream = nullptr) const;
    Status ComputeCosSinFromThetaOrder18(const Complex* Theta,
                                          int64_t n,
                                          Complex* Cos,
                                          Complex* Sin,
                                          cudaStream_t stream = nullptr) const;
    Status ComputeSFromT(const TMatrix& T,
                         const VMatrix& V,
                         int64_t n,
                         SMatrix* S,
                         cudaStream_t stream = nullptr) const;
    Status ComputeSMatrixSQRT(const Complex* P,
                               const Complex* Q,
                               int64_t n,
                               Real k0d,
                               const VMatrix& V,
                               SMatrix* S,
                               cudaStream_t stream = nullptr) const;
    Status RepeatCurrentStackSMatrix(int repeats);

    int64_t order_N() const { return order_N_; }
    const SMatrix& global_smatrix() const { return global_smatrix_; }
    cudaStream_t stream() const { return stream_; }
    void SetExpmMethod(char method) { expm_method_ = method; }
    char expm_method() const { return expm_method_; }

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
    Status RSProd(const SMatrix& Sm,
                  const SMatrix& Sn,
                  int64_t n,
                  SMatrix* out,
                  ComplexMatrixOps& ops,
                  cudaStream_t stream) const;

    void FreeSMatrix(SMatrix* matrix) const;
    bool HasSMatrix(const SMatrix& matrix) const;
    Status CopySMatrix(const SMatrix& src,
                       int64_t n,
                       SMatrix* dst,
                       cudaStream_t stream = nullptr) const;
    Status BuildPortSMatrix(Complex eps,
                            bool input_port,
                            SMatrix* S,
                            cudaStream_t stream = nullptr) const;
    Status CommitLayerSMatrix(SMatrix* layer, int64_t n);

    void FreeKVectors();
    void FreeGlobalSMatrix();
    void FreeSource();

    Real freq_ = 0.0;
    Real omega_ = 0.0;
    Real Lx_ = 0.0;
    Real Ly_ = 0.0;
    Real Gx_norm_ = 0.0;
    Real Gy_norm_ = 0.0;
    int order_x_ = 0;
    int order_y_ = 0;
    int64_t order_N_ = 0;
    bool kvectors_ready_ = false;
    char expm_method_ = 'G';
    cudaStream_t stream_ = nullptr;

    Complex* kx_diag_ = nullptr;
    Complex* ky_diag_ = nullptr;
    VMatrix V0_{};

    SMatrix stack_smatrix_{};
    SMatrix global_smatrix_{};
    Complex* source_ei_ = nullptr;
    Complex eps_in_ = make_complex(1.0, 0.0);
    Complex eps_out_ = make_complex(1.0, 0.0);
    bool angle_layer_input_ = true;
    bool source_forward_ = true;
};
