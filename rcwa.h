#ifndef RCWA_CUH
#define RCWA_CUH

#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <stdexcept>

// CUDA Headers
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cufft.h>

// Error Checking Macros
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at line " + std::to_string(__LINE__)); \
    } \
} while (0)

#define CHECK_CUSOLVER(call) do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuSOLVER error: ") + std::to_string(status) + " at line " + std::to_string(__LINE__)); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(status) + " at line " + std::to_string(__LINE__)); \
    } \
} while (0)

#define CHECK_CUFFT(call) do { \
    cufftResult status = call; \
    if (status != CUFFT_SUCCESS) { \
        throw std::runtime_error(std::string("cuFFT error: ") + std::to_string(status) + " at line " + std::to_string(__LINE__)); \
    } \
} while (0)

// Type Aliases (aligned with PyTorch complex types)
using complexf = std::complex<float>;   // Corresponding to torch.complex64
using complexd = std::complex<double>;  // Corresponding to torch.complex128

/**
 * @brief RCWA (Rigorous Coupled-Wave Analysis) Class (CUDA Accelerated Version)
 * @details Implements GPU-accelerated RCWA simulation using cuSOLVER/cuBLAS/cuFFT, 
 *          with interfaces aligned to the Python RCWA class
 */
class RCWA {
public:
    /**
     * @brief Data precision enumeration
     */
    enum class DType {
        COMPLEX64,  // Single-precision complex (float complex)
        COMPLEX128  // Double-precision complex (double complex)
    };

    /**
     * @brief Constructor
     * @param freq Frequency (unit: 1/length)
     * @param order Fourier orders [order_x, order_y]
     * @param L Lattice vectors [Lx, Ly] (unit: length)
     * @param dtype Data precision (default: COMPLEX64)
     * @param fast Enable fast mode (not implemented yet)
     */
    RCWA(double freq, const std::vector<int>& order, const std::vector<double>& L,
         DType dtype = DType::COMPLEX64, bool fast = false);

    /**
     * @brief Destructor (releases all CUDA resources)
     */
    ~RCWA();

    // -------------------------- Layer Management Interfaces --------------------------
    /**
     * @brief Add input layer
     * @param eps Relative permittivity of input layer (default: vacuum = 1.0)
     */
    void addInputLayer(double eps = 1.0);

    /**
     * @brief Add output layer
     * @param eps Relative permittivity of output layer (default: vacuum = 1.0)
     */
    void addOutputLayer(double eps = 1.0);

    /**
     * @brief Add internal layer
     * @param thickness Layer thickness (unit: length)
     * @param eps Relative permittivity (nullptr = homogeneous medium 1.0, non-nullptr = inhomogeneous medium tensor)
     */
    void addLayer(double thickness, const void* eps = nullptr);

    // -------------------------- Incident Condition Interfaces --------------------------
    /**
     * @brief Set incident angle
     * @param inc_ang Incident angle (unit: radian)
     * @param azi_ang Azimuthal angle (unit: radian)
     * @param angle_layer Reference layer ("input"/"in"/"i" or "output"/"out"/"o")
     */
    void setIncidentAngle(double inc_ang, double azi_ang, const std::string& angle_layer = "input");

    // -------------------------- Source Configuration Interfaces --------------------------
    /**
     * @brief Set plane wave source
     * @param amplitude Amplitude [Ex_amp, Ey_amp] (default: [1.0, 0.0])
     * @param direction Incident direction ("forward"/"f" or "backward"/"b")
     */
    void sourcePlanewave(const std::vector<complexf>& amplitude = {1.0, 0.0}, 
                         const std::string& direction = "forward");

    /**
     * @brief Set Fourier source
     * @param amplitude Amplitudes at each order [[Ex_amp, Ey_amp], ...]
     * @param orders Diffraction orders [[mx, my], ...]
     * @param direction Incident direction ("forward"/"f" or "backward"/"b")
     */
    void sourceFourier(const std::vector<complexf>& amplitude, 
                       const std::vector<std::vector<int>>& orders,
                       const std::string& direction = "forward");

    // -------------------------- Solver Interfaces --------------------------
    /**
     * @brief Solve global S-matrix
     */
    void solveGlobalSmatrix();

    // -------------------------- Field Calculation Interfaces --------------------------
    /**
     * @brief Calculate XY-plane field distribution
     * @param x_axis X-axis sampling coordinates (CPU array)
     * @param y_axis Y-axis sampling coordinates (CPU array)
     * @param x_N Number of sampling points on X-axis
     * @param y_N Number of sampling points on Y-axis
     * @return Field components [Ex, Ey, Ez] (GPU pointers, need manual release)
     */
    std::tuple<complexf*, complexf*, complexf*> fieldXY(const float* x_axis, const float* y_axis, 
                                                        int x_N, int y_N);

    /**
     * @brief Calculate Floquet mode field distribution
     * @return Field components [Ex_mn, Ey_mn, Ez_mn] (GPU pointers, need manual release)
     */
    std::tuple<complexf*, complexf*, complexf*> floquetMode();

    // -------------------------- Utility Interfaces --------------------------
    /**
     * @brief Get total number of Fourier orders
     * @return Total number of orders
     */
    int getOrderN() const { return order_N_; }

    /**
     * @brief Get number of orders in X/Y directions
     * @return [order_x_N, order_y_N]
     */
    std::tuple<int, int> getOrderXYN() const { return {order_x_N_, order_y_N_}; }

    /**
     * @brief Release GPU tensor memory
     * @param ptr GPU tensor pointer
     */
    void freeDeviceTensor(void* ptr) {
        if (ptr) CHECK_CUDA(cudaFree(ptr));
    }

private:
    // -------------------------- Core Member Variables --------------------------
    // Basic simulation parameters
    double freq_;                  // Frequency
    std::vector<int> order_;       // Fourier orders [order_x, order_y]
    std::vector<double> L_;        // Lattice vectors [Lx, Ly]
    DType dtype_;                  // Data precision
    bool fast_;                    // Fast mode flag
    double omega_;                 // Angular frequency (2πf)
    int order_x_N_;                // Number of orders in X direction (2*order_x+1)
    int order_y_N_;                // Number of orders in Y direction (2*order_y+1)
    int order_N_;                  // Total number of orders (order_x_N * order_y_N)
    double Gx_norm_;               // Normalized X component of reciprocal lattice vector
    double Gy_norm_;               // Normalized Y component of reciprocal lattice vector

    // Incident parameters
    std::string angle_layer_;      // Reference layer for angle calculation (input/output)
    double inc_ang_;               // Incident angle (radian)
    double azi_ang_;               // Azimuthal angle (radian)
    std::string source_direction_; // Source direction (forward/backward)

    // CUDA handles
    cusolverDnHandle_t cusolver_handle_ = nullptr;  // cuSOLVER handle
    cublasHandle_t cublas_handle_ = nullptr;        // cuBLAS handle
    cufftHandle cufft_plan_ = nullptr;              // cuFFT handle

    // GPU device tensors (raw pointers, manual management required)
    void* eps_in_ = nullptr;       // Permittivity of input layer (scalar)
    void* eps_out_ = nullptr;      // Permittivity of output layer (scalar)
    void* order_x_ = nullptr;      // X-direction order array
    void* order_y_ = nullptr;      // Y-direction order array
    void* Kx_norm_dn_ = nullptr;   // Normalized Kx vector (flattened)
    void* Ky_norm_dn_ = nullptr;   // Normalized Ky vector (flattened)
    void* Kx_norm_ = nullptr;      // Kx diagonal matrix
    void* Ky_norm_ = nullptr;      // Ky diagonal matrix
    void* Vf_ = nullptr;           // Free-space V matrix
    void* exp_constant_ = nullptr; // Exponential approximation constant matrix

    // Layer-related data
    int layer_N_ = 0;                          // Number of internal layers
    std::vector<void*> eps_conv_;              // Layer permittivity convolution matrix
    std::vector<void*> thickness_;             // Layer thickness (scalar)
    std::vector<void*> layer_S11_, layer_S12_; // Layer S-matrices
    std::vector<void*> layer_S21_, layer_S22_; // Layer S-matrices

    // Global S-matrix
    std::vector<void*> Sin_;       // Input layer S-matrix
    std::vector<void*> Sout_;      // Output layer S-matrix
    std::vector<void*> S_;         // Global S-matrix
    bool need_update_Sin_ = false; // Input layer S-matrix update flag
    bool need_update_Sout_ = false;// Output layer S-matrix update flag

    // Source-related
    void* E_i_ = nullptr;          // Incident field vector

    // -------------------------- Initialization Functions --------------------------
    /**
     * @brief Initialize CUDA handles (cusolver/cublas/cufft)
     */
    void initCUDABindles();

    /**
     * @brief Initialize Fourier order grid (CPU → GPU)
     */
    void initOrderGrid();

    /**
     * @brief Initialize exponential approximation constant matrix
     */
    void initExpConstant();

    // -------------------------- Core Calculation Functions --------------------------
    /**
     * @brief Calculate normalized K vectors
     */
    void computeKvectors();

    /**
     * @brief Material permittivity convolution (FFT method)
     * @param material Spatial distribution of permittivity (CPU pointer)
     * @return Convolution matrix (GPU pointer)
     */
    void* materialConv(const complexf* material);

    /**
     * @brief RS matrix product (S-matrix cascading)
     * @param Sm Previous stage S-matrix [S11, S21, S12, S22]
     * @param Sn Next stage S-matrix [S11, S21, S12, S22]
     * @return Cascaded S-matrix [S11, S21, S12, S22]
     */
    std::tuple<void*, void*, void*, void*> rsProd(const std::vector<void*>& Sm, 
                                                  const std::vector<void*>& Sn);

    /**
     * @brief Calculate exponential matrix for inhomogeneous layer
     */
    void computeLayerExp();

    /**
     * @brief Calculate exponential matrix for homogeneous layer
     * @param eps Layer permittivity (GPU scalar pointer)
     * @param thickness Layer thickness
     */
    void computeLayerExpHomogenous(void* eps, double thickness);

    /**
     * @brief Get V matrix (E→H transformation matrix)
     * @param eps Permittivity
     * @return [V matrix, kz_norm vector] (GPU pointers)
     */
    std::tuple<void*, void*> getV(double eps);

    /**
     * @brief Calculate input layer S-matrix
     */
    void computeInputLayerSmatrix();

    /**
     * @brief Calculate output layer S-matrix
     */
    void computeOutputLayerSmatrix();

    // -------------------------- CUDA Utility Functions --------------------------
    /**
     * @brief Create device tensor
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dtype Data precision
     * @return GPU tensor pointer
     */
    void* createDeviceTensor(int rows, int cols, DType dtype);

    /**
     * @brief Create scalar (GPU-side)
     * @param val Scalar value
     * @return GPU scalar pointer
     */
    void* createDeviceScalar(double val);

    /**
     * @brief Fill matrix with identity matrix
     * @param ptr GPU tensor pointer
     * @param N Matrix dimension
     * @param dtype Data precision
     */
    void fillEye(void* ptr, int N, DType dtype);

    /**
     * @brief Fill matrix with zeros
     * @param ptr GPU tensor pointer
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dtype Data precision
     */
    void fillZero(void* ptr, int rows, int cols, DType dtype);

    /**
     * @brief Matrix inversion (cuSOLVER implementation)
     * @param A Input matrix (GPU pointer)
     * @param N Matrix dimension
     * @param dtype Data precision
     * @return Inverted matrix (GPU pointer)
     */
    void* matrixInverse(void* A, int N, DType dtype);

    /**
     * @brief Matrix solve (A*X=B, cuSOLVER implementation)
     * @param A Coefficient matrix (GPU pointer)
     * @param B Right-hand side vector/matrix (GPU pointer)
     * @param N Matrix dimension
     * @param dtype Data precision
     * @return Solution X (GPU pointer)
     */
    void* matrixSolve(void* A, void* B, int N, DType dtype);

    /**
     * @brief Matrix multiplication (cuBLAS implementation)
     * @param A Matrix A (GPU pointer)
     * @param B Matrix B (GPU pointer)
     * @param m Rows of A
     * @param k Columns of A / Rows of B
     * @param n Columns of B
     * @param dtype Data precision
     * @return Product C=A*B (GPU pointer)
     */
    void* matMul(void* A, void* B, int m, int k, int n, DType dtype);

    /**
     * @brief Match diffraction order indices
     * @param orders Diffraction orders [[mx, my], ...]
     * @return Index array
     */
    std::vector<int> matchingIndices(const std::vector<std::vector<int>>& orders);

    /**
     * @brief Calculate Kz_norm vector
     * @param eps Permittivity (GPU scalar pointer)
     * @return Kz_norm vector (GPU pointer)
     */
    void* computeKzNorm(void* eps);

    /**
     * @brief Calculate Kz_norm vector (output to specified tensor)
     * @param eps Permittivity (GPU scalar pointer)
     * @param kz_norm Output tensor (GPU pointer)
     */
    void computeKzNorm(void* eps, void* kz_norm);

    // -------------------------- Basic Matrix Operations --------------------------
    /**
     * @brief Matrix addition
     * @param A Matrix A (GPU pointer)
     * @param B Matrix B (GPU pointer)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dtype Data precision
     * @return A+B (GPU pointer)
     */
    void* addTensor(void* A, void* B, int rows, int cols, DType dtype);

    /**
     * @brief Matrix subtraction
     * @param A Matrix A (GPU pointer)
     * @param B Matrix B (GPU pointer)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dtype Data precision
     * @return A-B (GPU pointer)
     */
    void* subTensor(void* A, void* B, int rows, int cols, DType dtype);

    /**
     * @brief Matrix scaling
     * @param A Matrix A (GPU pointer)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param val Scaling factor
     * @param dtype Data precision
     * @return val*A (GPU pointer)
     */
    void* scaleTensor(void* A, int rows, int cols, complexf val, DType dtype);

    /**
     * @brief Matrix copy
     * @param A Input matrix (GPU pointer)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dtype Data precision
     * @return Copied matrix (GPU pointer)
     */
    void* copyTensor(void* A, int rows, int cols, DType dtype);

    /**
     * @brief Matrix slicing
     * @param A Input matrix (GPU pointer)
     * @param start_row Start row
     * @param end_row End row
     * @param start_col Start column
     * @param end_col End column
     * @param dtype Data precision
     * @return Sliced matrix (GPU pointer)
     */
    void* sliceTensor(void* A, int start_row, int end_row, int start_col, int end_col, DType dtype);

    /**
     * @brief Matrix reshaping
     * @param A Input matrix (GPU pointer)
     * @param in_rows Input rows
     * @param in_cols Input columns
     * @param out_rows Output rows
     * @param out_cols Output columns
     * @param dtype Data precision
     * @return Reshaped matrix (GPU pointer)
     */
    void* reshapeTensor(void* A, int in_rows, int in_cols, int out_rows, int out_cols, DType dtype);

    /**
     * @brief Helper: Compute max norm of a matrix (column-wise L1 norm)
     * @param A Input matrix (GPU pointer)
     * @param rows Rows of A
     * @param cols Columns of A
     * @param dtype Data precision
     * @return Max column-wise L1 norm
     */
    double computeMatrixNorm(void* A, int rows, int cols, DType dtype);

    /**
     * @brief Helper: Matrix horizontal stack (hstack)
     * @param A First matrix (GPU pointer)
     * @param B Second matrix (GPU pointer)
     * @param rows Rows of A/B (must match)
     * @param colsA Columns of A
     * @param colsB Columns of B
     * @param dtype Data precision
     * @return Hstacked matrix (GPU pointer)
     */
    void* hstackTensor(void* A, void* B, int rows, int colsA, int colsB, DType dtype);

    /**
     * @brief Helper: Matrix vertical stack (vstack)
     * @param A First matrix (GPU pointer)
     * @param B Second matrix (GPU pointer)
     * @param rowsA Rows of A
     * @param rowsB Rows of B
     * @param cols Columns of A/B (must match)
     * @param dtype Data precision
     * @return Vstacked matrix (GPU pointer)
     */
    void* vstackTensor(void* A, void* B, int rowsA, int rowsB, int cols, DType dtype);
};

#endif // RCWA_CUH