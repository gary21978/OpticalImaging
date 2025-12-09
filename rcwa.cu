#include "RCWA.cuh"
#include <cmath>
#include <algorithm>
#include <cstring>

// -------------------------- Constructor & Destructor --------------------------
RCWA::RCWA(double freq, const std::vector<int>& order, const std::vector<double>& L,
           DType dtype, bool fast)
    : freq_(freq), order_(order), L_(L), dtype_(dtype), fast_(fast) {
    // Initialize CUDA handles
    initCUDABindles();

    // Set default GPU device (device 0)
    CHECK_CUDA(cudaSetDevice(0));

    // Initialize basic parameters
    omega_ = 2 * M_PI * freq_;
    order_x_N_ = 2 * order_[0] + 1;
    order_y_N_ = 2 * order_[1] + 1;
    order_N_ = order_x_N_ * order_y_N_;

    // Initialize Fourier order grid (CPU → GPU)
    initOrderGrid();

    // Lattice vector normalization
    Gx_norm_ = 1.0 / (L_[0] * freq_);
    Gy_norm_ = 1.0 / (L_[1] * freq_);

    // Default permittivity for input/output layers (vacuum)
    eps_in_ = createDeviceScalar(1.0);
    eps_out_ = createDeviceScalar(1.0);

    // Initialize exponential approximation constants (hard-coded)
    initExpConstant();
}

RCWA::~RCWA() {
    // Release CUDA handles
    if (cusolver_handle_) CHECK_CUSOLVER(cusolverDnDestroy(cusolver_handle_));
    if (cublas_handle_) CHECK_CUBLAS(cublasDestroy(cublas_handle_));
    if (cufft_plan_) CHECK_CUFFT(cufftDestroy(cufft_plan_));

    // Release device memory
    if (eps_in_) freeDeviceTensor(eps_in_);
    if (eps_out_) freeDeviceTensor(eps_out_);
    if (order_x_) freeDeviceTensor(order_x_);
    if (order_y_) freeDeviceTensor(order_y_);
    if (Kx_norm_dn_) freeDeviceTensor(Kx_norm_dn_);
    if (Ky_norm_dn_) freeDeviceTensor(Ky_norm_dn_);
    if (Kx_norm_) freeDeviceTensor(Kx_norm_);
    if (Ky_norm_) freeDeviceTensor(Ky_norm_);
    if (Vf_) freeDeviceTensor(Vf_);
    if (exp_constant_) freeDeviceTensor(exp_constant_);

    // Release layer-related memory
    for (auto& ptr : eps_conv_) freeDeviceTensor(ptr);
    for (auto& ptr : thickness_) freeDeviceTensor(ptr);
    for (auto& ptr : layer_S11_) freeDeviceTensor(ptr);
    for (auto& ptr : layer_S12_) freeDeviceTensor(ptr);
    for (auto& ptr : layer_S21_) freeDeviceTensor(ptr);
    for (auto& ptr : layer_S22_) freeDeviceTensor(ptr);

    // Release global S-matrix memory
    for (auto& ptr : Sin_) freeDeviceTensor(ptr);
    for (auto& ptr : Sout_) freeDeviceTensor(ptr);
    for (auto& ptr : S_) freeDeviceTensor(ptr);

    // Release source-related memory
    if (E_i_) freeDeviceTensor(E_i_);
}

// -------------------------- Initialization Functions --------------------------
void RCWA::initCUDABindles() {
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle_));
    CHECK_CUBLAS(cublasCreate(&cublas_handle_));
    CHECK_CUFFT(cufftCreate(&cufft_plan_));
}

void RCWA::initOrderGrid() {
    // Generate order arrays on CPU
    std::vector<int> order_x_cpu, order_y_cpu;
    for (int i = -order_[0]; i <= order_[0]; ++i) order_x_cpu.push_back(i);
    for (int i = -order_[1]; i <= order_[1]; ++i) order_y_cpu.push_back(i);

    // Copy to GPU
    order_x_ = createDeviceTensor(order_x_cpu.size(), 1, dtype_);
    order_y_ = createDeviceTensor(order_y_cpu.size(), 1, dtype_);
    CHECK_CUDA(cudaMemcpy(order_x_, order_x_cpu.data(), 
                          order_x_cpu.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(order_y_, order_y_cpu.data(), 
                          order_y_cpu.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize K vector grid
    computeKvectors();
}

void RCWA::initExpConstant() {
    // Hard-coded exponential approximation constants
    double exp_constant_cpu[5][5] = {
        {0, -0.10036558103014462001, -0.00802924648241156960, -0.00089213849804572995, 0},
        {0, 0.39784974949964507614, 1.36783778460411719922, 0.49828962252538267755, -0.00063789819459472330},
        {-10.9676396052962062593, 1.68015813878906197182, 0.05717798464788655127, -0.00698210122488052084, 0.00003349750170860705},
        {-0.0904316832390810561, -0.06764045190713819075, 0.06759613017704596460, 0.02955525704293155274, -0.00001391802575160607},
        {0, 0, -0.09233646193671185927, -0.01693649390020817171, -0.00001400867981820361}
    };

    // Allocate and copy to GPU
    exp_constant_ = createDeviceTensor(5, 5, dtype_);
    CHECK_CUDA(cudaMemcpy(exp_constant_, exp_constant_cpu, 
                          5 * 5 * sizeof(double), cudaMemcpyHostToDevice));
}

// -------------------------- Layer Management --------------------------
void RCWA::addInputLayer(double eps) {
    if (eps_in_) freeDeviceTensor(eps_in_);
    eps_in_ = createDeviceScalar(eps);
    Sin_.clear();
    need_update_Sin_ = true;
}

void RCWA::addOutputLayer(double eps) {
    if (eps_out_) freeDeviceTensor(eps_out_);
    eps_out_ = createDeviceScalar(eps);
    Sout_.clear();
    need_update_Sout_ = true;
}

void RCWA::addLayer(double thickness, const void* eps) {
    bool is_homogeneous = (eps == nullptr);
    
    void* E = nullptr;
    if (is_homogeneous) {
        // Homogeneous medium: identity matrix * eps
        E = createDeviceTensor(order_N_, order_N_, dtype_);
        fillEye(E, order_N_, dtype_);
        E = scaleTensor(E, order_N_, order_N_, complexf(thickness), dtype_);
    } else {
        // Inhomogeneous medium: permittivity convolution
        E = materialConv(static_cast<const complexf*>(eps));
    }

    eps_conv_.push_back(E);
    thickness_.push_back(createDeviceScalar(thickness));
    layer_N_++;

    // Compute layer exponential matrix
    if (is_homogeneous) {
        double eps_val = (eps == nullptr) ? 1.0 : *static_cast<const double*>(eps);
        computeLayerExpHomogenous(createDeviceScalar(eps_val), thickness);
    } else {
        computeLayerExp();
    }
}

// -------------------------- Incident Angle Configuration --------------------------
void RCWA::setIncidentAngle(double inc_ang, double azi_ang, const std::string& angle_layer) {
    inc_ang_ = inc_ang;
    azi_ang_ = azi_ang;
    
    // Validate angle layer
    if (angle_layer == "input" || angle_layer == "in" || angle_layer == "i") {
        angle_layer_ = "input";
    } else if (angle_layer == "output" || angle_layer == "out" || angle_layer == "o") {
        angle_layer_ = "output";
    } else {
        throw std::runtime_error("Invalid angle layer: " + angle_layer + " (using 'input' as default)");
    }

    // Recompute K vectors with new angles
    computeKvectors();
}

// -------------------------- Source Configuration --------------------------
void RCWA::sourcePlanewave(const std::vector<complexf>& amplitude, const std::string& direction) {
    std::vector<std::vector<int>> orders = {{0, 0}};
    sourceFourier(amplitude, orders, direction);
}

void RCWA::sourceFourier(const std::vector<complexf>& amplitude, 
                         const std::vector<std::vector<int>>& orders,
                         const std::string& direction) {
    // Validate direction
    if (direction == "forward" || direction == "f") {
        source_direction_ = "forward";
    } else if (direction == "backward" || direction == "b") {
        source_direction_ = "backward";
    } else {
        throw std::runtime_error("Invalid source direction: " + direction + " (using 'forward' as default)");
    }

    // Match order indices
    std::vector<int> order_indices = matchingIndices(orders);

    // Initialize incident field vector
    if (E_i_) freeDeviceTensor(E_i_);
    E_i_ = createDeviceTensor(2 * order_N_, 1, dtype_);
    fillZero(E_i_, 2 * order_N_, 1, dtype_);

    // Set amplitudes
    size_t elem_size = (dtype_ == DType::COMPLEX64) ? sizeof(complexf) : sizeof(complexd);
    for (size_t i = 0; i < amplitude.size() && i < order_indices.size(); ++i) {
        // Set Ex amplitude
        CHECK_CUDA(cudaMemcpy(static_cast<char*>(E_i_) + order_indices[i] * elem_size,
                              &amplitude[i], elem_size, cudaMemcpyHostToDevice));
        // Set Ey amplitude
        CHECK_CUDA(cudaMemcpy(static_cast<char*>(E_i_) + (order_indices[i] + order_N_) * elem_size,
                              &amplitude[i], elem_size, cudaMemcpyHostToDevice));
    }
}

// -------------------------- Solver --------------------------
void RCWA::solveGlobalSmatrix() {
    int N = 2 * order_N_;
    void *S11, *S21, *S12, *S22;

    // Initialize with first layer or identity matrix
    if (layer_N_ > 0) {
        S11 = copyTensor(layer_S11_[0], N, N, dtype_);
        S21 = copyTensor(layer_S21_[0], N, N, dtype_);
        S12 = copyTensor(layer_S12_[0], N, N, dtype_);
        S22 = copyTensor(layer_S22_[0], N, N, dtype_);
    } else {
        // Empty layer: identity + zero matrices
        S11 = createDeviceTensor(N, N, dtype_);
        S21 = createDeviceTensor(N, N, dtype_);
        S12 = createDeviceTensor(N, N, dtype_);
        S22 = createDeviceTensor(N, N, dtype_);
        fillEye(S11, N, dtype_);
        fillEye(S22, N, dtype_);
        fillZero(S21, N, N, dtype_);
        fillZero(S12, N, N, dtype_);
    }

    // Cascade layers
    for (int i = 0; i < layer_N_ - 1; ++i) {
        auto [new_S11, new_S21, new_S12, new_S22] = 
            rsProd({S11, S21, S12, S22},
                   {layer_S11_[i+1], layer_S21_[i+1], layer_S12_[i+1], layer_S22_[i+1]});

        // Release old matrices
        freeDeviceTensor(S11);
        freeDeviceTensor(S21);
        freeDeviceTensor(S12);
        freeDeviceTensor(S22);

        S11 = new_S11;
        S21 = new_S21;
        S12 = new_S12;
        S22 = new_S22;
    }

    // Input layer coupling
    if (!Sin_.empty() && need_update_Sin_) {
        computeInputLayerSmatrix();
        auto [new_S11, new_S21, new_S12, new_S22] = 
            rsProd(Sin_, {S11, S21, S12, S22});
        
        freeDeviceTensor(S11);
        freeDeviceTensor(S21);
        freeDeviceTensor(S12);
        freeDeviceTensor(S22);

        S11 = new_S11;
        S21 = new_S21;
        S12 = new_S12;
        S22 = new_S22;
        need_update_Sin_ = false;
    }

    // Output layer coupling
    if (!Sout_.empty() && need_update_Sout_) {
        computeOutputLayerSmatrix();
        auto [new_S11, new_S21, new_S12, new_S22] = 
            rsProd({S11, S21, S12, S22}, Sout_);
        
        freeDeviceTensor(S11);
        freeDeviceTensor(S21);
        freeDeviceTensor(S12);
        freeDeviceTensor(S22);

        S11 = new_S11;
        S21 = new_S21;
        S12 = new_S12;
        S22 = new_S22;
        need_update_Sout_ = false;
    }

    // Store global S-matrix
    S_ = {S11, S21, S12, S22};
}

// -------------------------- Field Calculation --------------------------
std::tuple<complexf*, complexf*, complexf*> RCWA::fieldXY(const float* x_axis, const float* y_axis, 
                                                          int x_N, int y_N) {
    // Validate input arrays
    if (!x_axis || !y_axis || x_N <= 0 || y_N <= 0) {
        throw std::runtime_error("Invalid sampling coordinates or dimensions");
    }

    // Get permittivity of reference layer
    void* eps = (angle_layer_ == "input") ? eps_in_ : eps_out_;
    
    // Calculate Kz_norm
    void* kz_norm = computeKzNorm(eps);

    // Select port based on angle layer and source direction
    int port = 0;
    if (angle_layer_ == "input") {
        port = (source_direction_ == "forward") ? 1 : 3;
    } else {
        port = (source_direction_ == "forward") ? 0 : 2;
    }

    // Calculate Exy = S[port] * E_i
    void* Exy = matMul(S_[port], E_i_, 2*order_N_, 2*order_N_, 1, dtype_);

    // Split Ex/Ey/Ez
    void* Ex_mn = sliceTensor(Exy, 0, order_N_, 0, 1, dtype_);
    void* Ey_mn = sliceTensor(Exy, order_N_, 2*order_N_, 0, 1, dtype_);
    void* Ez_mn = nullptr; // To be implemented: compute Ez from Ex/Ey/Kz

    // Release temporary memory
    freeDeviceTensor(Exy);
    freeDeviceTensor(kz_norm);

    // Return field components (simplified - full implementation requires spatial expansion)
    return {static_cast<complexf*>(Ex_mn), static_cast<complexf*>(Ey_mn), static_cast<complexf*>(Ez_mn)};
}

std::tuple<complexf*, complexf*, complexf*> RCWA::floquetMode() {
    // Get permittivity of reference layer
    void* eps = (angle_layer_ == "input") ? eps_in_ : eps_out_;
    
    // Calculate Kz_norm
    void* kz_norm = computeKzNorm(eps);

    // Select port
    int port = 0;
    if (angle_layer_ == "input") {
        port = (source_direction_ == "forward") ? 1 : 3;
    } else {
        port = (source_direction_ == "forward") ? 0 : 2;
    }

    // Calculate Exy = S[port] * E_i
    void* Exy = matMul(S_[port], E_i_, 2*order_N_, 2*order_N_, 1, dtype_);

    // Split and reshape field components
    void* Ex_mn = sliceTensor(Exy, 0, order_N_, 0, 1, dtype_);
    void* Ey_mn = sliceTensor(Exy, order_N_, 2*order_N_, 0, 1, dtype_);
    void* Ez_mn = nullptr; // To be implemented

    Ex_mn = reshapeTensor(Ex_mn, order_N_, 1, order_x_N_, order_y_N_, dtype_);
    Ey_mn = reshapeTensor(Ey_mn, order_N_, 1, order_x_N_, order_y_N_, dtype_);

    // Release temporary memory
    freeDeviceTensor(Exy);
    freeDeviceTensor(kz_norm);

    return {static_cast<complexf*>(Ex_mn), static_cast<complexf*>(Ey_mn), static_cast<complexf*>(Ez_mn)};
}

// -------------------------- Core Calculation Functions --------------------------
void RCWA::computeKvectors() {
    // Get permittivity of reference layer
    double eps_val = 1.0;
    if (angle_layer_ == "input") {
        CHECK_CUDA(cudaMemcpy(&eps_val, eps_in_, sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        CHECK_CUDA(cudaMemcpy(&eps_val, eps_out_, sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Calculate kx0/ky0 (normalized)
    double kx0_norm = sqrt(eps_val) * sin(inc_ang_) * cos(azi_ang_);
    double ky0_norm = sqrt(eps_val) * sin(inc_ang_) * sin(azi_ang_);

    // Get order arrays from GPU
    int* order_x_cpu = new int[order_x_N_];
    int* order_y_cpu = new int[order_y_N_];
    CHECK_CUDA(cudaMemcpy(order_x_cpu, order_x_, order_x_N_ * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(order_y_cpu, order_y_, order_y_N_ * sizeof(int), cudaMemcpyDeviceToHost));

    // Generate Kx/Ky norm vectors (flattened grid)
    std::vector<double> Kx_norm_dn_cpu, Ky_norm_dn_cpu;
    for (int i = 0; i < order_x_N_; ++i) {
        for (int j = 0; j < order_y_N_; ++j) {
            Kx_norm_dn_cpu.push_back(kx0_norm + order_x_cpu[i] * Gx_norm_);
            Ky_norm_dn_cpu.push_back(ky0_norm + order_y_cpu[j] * Gy_norm_);
        }
    }

    // Copy to GPU
    if (Kx_norm_dn_) freeDeviceTensor(Kx_norm_dn_);
    if (Ky_norm_dn_) freeDeviceTensor(Ky_norm_dn_);
    Kx_norm_dn_ = createDeviceTensor(order_N_, 1, dtype_);
    Ky_norm_dn_ = createDeviceTensor(order_N_, 1, dtype_);
    CHECK_CUDA(cudaMemcpy(Kx_norm_dn_, Kx_norm_dn_cpu.data(), 
                          order_N_ * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(Ky_norm_dn_, Ky_norm_dn_cpu.data(), 
                          order_N_ * sizeof(double), cudaMemcpyHostToDevice));

    // Create diagonal matrices for Kx/Ky
    if (Kx_norm_) freeDeviceTensor(Kx_norm_);
    if (Ky_norm_) freeDeviceTensor(Ky_norm_);
    Kx_norm_ = createDeviceTensor(order_N_, order_N_, dtype_);
    Ky_norm_ = createDeviceTensor(order_N_, order_N_, dtype_);
    fillEye(Kx_norm_, order_N_, dtype_);
    fillEye(Ky_norm_, order_N_, dtype_);

    // Calculate free-space V matrix
    if (Vf_) freeDeviceTensor(Vf_);
    auto [Vf, kz_norm] = getV(1.0);
    Vf_ = Vf;
    freeDeviceTensor(kz_norm);

    // Cleanup
    delete[] order_x_cpu;
    delete[] order_y_cpu;
}

void* RCWA::materialConv(const complexf* material) {
    int mat_H = order_x_N_;
    int mat_W = order_y_N_;
    int mat_size = mat_H * mat_W;

    // Allocate GPU memory for material
    void* d_material = createDeviceTensor(mat_H, mat_W, dtype_);
    CHECK_CUDA(cudaMemcpy(d_material, material, mat_size * sizeof(complexf), cudaMemcpyHostToDevice));

    // FFT2 (forward)
    void* d_material_fft = createDeviceTensor(mat_H, mat_W, dtype_);
    CHECK_CUFFT(cufftExecC2C(cufft_plan_, static_cast<cufftComplex*>(d_material),
                             static_cast<cufftComplex*>(d_material_fft), CUFFT_FORWARD));

    // Scale FFT result
    d_material_fft = scaleTensor(d_material_fft, mat_H, mat_W, complexf(1.0 / mat_size), dtype_);

    // Create convolution matrix (simplified - full implementation requires index matching)
    void* eps_conv = createDeviceTensor(order_N_, order_N_, dtype_);
    fillZero(eps_conv, order_N_, order_N_, dtype_);
    CHECK_CUDA(cudaMemcpy(eps_conv, d_material_fft, mat_size * sizeof(complexf), cudaMemcpyDeviceToDevice));

    // Cleanup
    freeDeviceTensor(d_material);
    freeDeviceTensor(d_material_fft);

    return eps_conv;
}

std::tuple<void*, void*, void*, void*> RCWA::rsProd(const std::vector<void*>& Sm, 
                                                    const std::vector<void*>& Sn) {
    int N = 2 * order_N_;
    void *S11, *S21, *S12, *S22;

    // Create identity/zero matrices
    void* I = createDeviceTensor(N, N, dtype_);
    void* O = createDeviceTensor(N, N, dtype_);
    fillEye(I, N, dtype_);
    fillZero(O, N, N, dtype_);

    try {
        // T1 = I - Sm[2] * Sn[1]
        void* Sm2_Sn1 = matMul(Sm[2], Sn[1], N, N, N, dtype_);
        void* T1 = subTensor(I, Sm2_Sn1, N, N, dtype_);

        // T2 = I - Sn[1] * Sm[2]
        void* Sn1_Sm2 = matMul(Sn[1], Sm[2], N, N, N, dtype_);
        void* T2 = subTensor(I, Sn1_Sm2, N, N, dtype_);

        // Invert T1/T2
        void* T1_inv = matrixInverse(T1, N, dtype_);
        void* T2_inv = matrixInverse(T2, N, dtype_);

        // Calculate S-matrix components
        S11 = matMul(Sn[0], matMul(T1_inv, Sm[0], N, N, N, dtype_), N, N, N, dtype_);
        S21 = addTensor(Sm[1], matMul(Sm[3], matMul(T2_inv, matMul(Sn[1], Sm[0], N, N, N, dtype_), N, N, N, dtype_), N, N, N, dtype_), N, N, dtype_);
        S12 = addTensor(Sn[2], matMul(Sn[0], matMul(T1_inv, matMul(Sm[2], Sn[3], N, N, N, dtype_), N, N, N, dtype_), N, N, N, dtype_), N, N, dtype_);
        S22 = matMul(Sm[3], matMul(T2_inv, Sn[3], N, N, N, dtype_), N, N, N, dtype_);

        // Cleanup
        freeDeviceTensor(Sm2_Sn1);
        freeDeviceTensor(T1);
        freeDeviceTensor(Sn1_Sm2);
        freeDeviceTensor(T2);
        freeDeviceTensor(T1_inv);
        freeDeviceTensor(T2_inv);
    } catch (const std::runtime_error& e) {
        // Fallback for singular matrices
        S11 = matMul(Sn[0], Sm[0], N, N, N, dtype_);
        S21 = copyTensor(Sm[1], N, N, dtype_);
        S12 = copyTensor(Sn[2], N, N, dtype_);
        S22 = copyTensor(O, N, N, dtype_);
    }

    // Cleanup
    freeDeviceTensor(I);
    freeDeviceTensor(O);

    return {S11, S21, S12, S22};
}

void RCWA::computeLayerExp() {
    // -------------------------- Step 1: Threshold Check for Repeated Squaring --------------------------
    double Lx = L_[0], Ly = L_[1];
    int order_x = order_[0], order_y = order_[1];
    double thickness_thres = 2.0 * std::min(Lx / order_x, Ly / order_y);
    double current_thickness;
    
    // Get current layer thickness (GPU → CPU)
    CHECK_CUDA(cudaMemcpy(&current_thickness, thickness_.back(), sizeof(double), cudaMemcpyDeviceToHost));
    
    int n_repeatedSquaring = 0;
    double d_block = current_thickness;
    
    if (current_thickness > thickness_thres) {
        double log2_val = log2(current_thickness / thickness_thres);
        n_repeatedSquaring = static_cast<int>(ceil(log2_val));
        d_block = current_thickness / pow(2.0, n_repeatedSquaring);
    }

    // -------------------------- Step 2: Precompute K Matrices --------------------------
    double k0d = omega_ * d_block;
    void* E = eps_conv_.back(); // Current layer permittivity convolution matrix
    int N = order_N_;
    int N2 = 2 * N;

    // KxKx = Kx_norm ⊙ Kx_norm (element-wise square)
    void* KxKx = copyTensor(Kx_norm_dn_, N, 1, dtype_);
    KxKx = scaleTensor(KxKx, N, 1, complexf(1.0), dtype_); // Copy first
    // Element-wise square (GPU → CPU → GPU for simplicity; replace with kernel for full speed)
    std::vector<double> KxKx_cpu(N);
    CHECK_CUDA(cudaMemcpy(KxKx_cpu.data(), KxKx, N * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) KxKx_cpu[i] = KxKx_cpu[i] * KxKx_cpu[i];
    CHECK_CUDA(cudaMemcpy(KxKx, KxKx_cpu.data(), N * sizeof(double), cudaMemcpyDeviceToHost));

    // KyKy = Ky_norm ⊙ Ky_norm (element-wise square)
    void* KyKy = copyTensor(Ky_norm_dn_, N, 1, dtype_);
    std::vector<double> KyKy_cpu(N);
    CHECK_CUDA(cudaMemcpy(KyKy_cpu.data(), KyKy, N * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) KyKy_cpu[i] = KyKy_cpu[i] * KyKy_cpu[i];
    CHECK_CUDA(cudaMemcpy(KyKy, KyKy_cpu.data(), N * sizeof(double), cudaMemcpyDeviceToHost));

    // KxKy = Kx_norm ⊙ Ky_norm (element-wise product)
    void* KxKy = copyTensor(Kx_norm_dn_, N, 1, dtype_);
    std::vector<double> Kx_norm_cpu(N), Ky_norm_cpu(N);
    CHECK_CUDA(cudaMemcpy(Kx_norm_cpu.data(), Kx_norm_dn_, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(Ky_norm_cpu.data(), Ky_norm_dn_, N * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) KxKy_cpu[i] = Kx_norm_cpu[i] * Ky_norm_cpu[i];
    CHECK_CUDA(cudaMemcpy(KxKy, KxKy_cpu.data(), N * sizeof(double), cudaMemcpyDeviceToHost));

    // -------------------------- Step 3: Solve E⁻¹Kx/E⁻¹Ky --------------------------
    void *EiKx = nullptr, *EiKy = nullptr;
    void *KxEiKy = nullptr, *KyEiKx = nullptr, *KxEiKx = nullptr, *KyEiKy = nullptr;
    
    try {
        // EiKx = E⁻¹ * Kx_norm (solve E*X = Kx_norm)
        EiKx = matrixSolve(E, Kx_norm_dn_, N, dtype_);
        // EiKy = E⁻¹ * Ky_norm (solve E*X = Ky_norm)
        EiKy = matrixSolve(E, Ky_norm_dn_, N, dtype_);

        // KxEiKy = Kx_norm @ EiKy
        KxEiKy = matMul(Kx_norm_, EiKy, N, N, 1, dtype_);
        // KyEiKx = Ky_norm @ EiKx
        KyEiKx = matMul(Ky_norm_, EiKx, N, N, 1, dtype_);
        // KxEiKx = Kx_norm @ EiKx
        KxEiKx = matMul(Kx_norm_, EiKx, N, N, 1, dtype_);
        // KyEiKy = Ky_norm @ EiKy
        KyEiKy = matMul(Ky_norm_, EiKy, N, N, 1, dtype_);
    } catch (const std::runtime_error& e) {
        // Fallback to element-wise products
        KxEiKy = copyTensor(KxKy, N, 1, dtype_);
        KyEiKx = copyTensor(KxKy, N, 1, dtype_);
        KxEiKx = copyTensor(KxKx, N, 1, dtype_);
        KyEiKy = copyTensor(KyKy, N, 1, dtype_);
        
        if (EiKx) freeDeviceTensor(EiKx);
        if (EiKy) freeDeviceTensor(EiKy);
    }

    // -------------------------- Step 4: Build Identity Matrices --------------------------
    void* I = createDeviceTensor(N, N, dtype_);
    fillEye(I, N, dtype_);
    void* I2 = createDeviceTensor(N2, N2, dtype_);
    fillEye(I2, N2, dtype_);

    // -------------------------- Step 5: Build P Matrix --------------------------
    // Part 1: vstack((KxEiKy, KyEiKy - I))
    void* KyEiKy_minus_I = subTensor(KyEiKy, I, N, N, dtype_);
    void* P_top = vstackTensor(KxEiKy, KyEiKy_minus_I, N, N, N, dtype_);
    
    // Part 2: vstack((I - KxEiKx, -KyEiKx))
    void* I_minus_KxEiKx = subTensor(I, KxEiKx, N, N, dtype_);
    void* neg_KyEiKx = scaleTensor(KyEiKx, N, N, complexf(-1.0), dtype_);
    void* P_bot = vstackTensor(I_minus_KxEiKx, neg_KyEiKx, N, N, N, dtype_);
    
    // P = hstack(P_top, P_bot)
    void* P = hstackTensor(P_top, P_bot, N2, N, N, dtype_);

    // -------------------------- Step 6: Build Q Matrix --------------------------
    // Part 1: vstack((-KxKy, E - KyKy))
    void* neg_KxKy = scaleTensor(KxKy, N, 1, complexf(-1.0), dtype_);
    void* E_minus_KyKy = subTensor(E, KyKy, N, N, dtype_);
    void* Q_top = vstackTensor(neg_KxKy, E_minus_KyKy, N, N, N, dtype_);
    
    // Part 2: vstack((KxKx - E, KxKy))
    void* KxKx_minus_E = subTensor(KxKx, E, N, N, dtype_);
    void* Q_bot = vstackTensor(KxKx_minus_E, KxKy, N, N, N, dtype_);
    
    // Q = hstack(Q_top, Q_bot)
    void* Q = hstackTensor(Q_top, Q_bot, N2, N, N, dtype_);

    // -------------------------- Step 7: Compute Matrix Norms --------------------------
    double Pnorm = computeMatrixNorm(P, N2, N2, dtype_);
    double Qnorm = computeMatrixNorm(Q, N2, N2, dtype_);
    double Rnorm = std::max(Pnorm, Qnorm);
    int m = static_cast<int>(ceil(log2(Rnorm * k0d)));
    if (m < 0) m = 0;

    // -------------------------- Step 8: Build A Matrices --------------------------
    double scale_A = pow(2.0, -m) * k0d;
    void* A_12 = scaleTensor(P, N2, N2, complexf(-scale_A), dtype_);
    void* A_21 = scaleTensor(Q, N2, N2, complexf(scale_A), dtype_);

    // A2_11 = A_12 @ A_21
    void* A2_11 = matMul(A_12, A_21, N2, N2, N2, dtype_);
    // A2_22 = A_21 @ A_12
    void* A2_22 = matMul(A_21, A_12, N2, N2, N2, dtype_);

    // A3_12 = A_12 @ A2_22
    void* A3_12 = matMul(A_12, A2_22, N2, N2, N2, dtype_);
    // A3_21 = A_21 @ A2_11
    void* A3_21 = matMul(A_21, A2_11, N2, N2, N2, dtype_);

    // A6_11 = A3_12 @ A3_21
    void* A6_11 = matMul(A3_12, A3_21, N2, N2, N2, dtype_);
    // A6_22 = A3_21 @ A3_12
    void* A6_22 = matMul(A3_21, A3_12, N2, N2, N2, dtype_);

    // Cleanup intermediate matrices
    freeDeviceTensor(P);
    freeDeviceTensor(Q);

    // -------------------------- Step 9: Build B Matrices (Using exp_constant_) --------------------------
    std::vector<void*> B_11(5), B_12(5), B_21(5), B_22(5);
    double exp_constant_cpu[5][5];
    CHECK_CUDA(cudaMemcpy(exp_constant_cpu, exp_constant_, 5*5*sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 5; ++i) 
    {
        // B_11[i] = exp_constant[i,0]*I2 + exp_constant[i,2]*A2_11 + exp_constant[i,4]*A6_11
        B_11[i] = scaleTensor(I2, N2, N2, complexf(exp_constant_cpu[i][0]), dtype_);
        void* term2 = scaleTensor(A2_11, N2, N2, complexf(exp_constant_cpu[i][2]), dtype_);
        void* term3 = scaleTensor(A6_11, N2, N2, complexf(exp_constant_cpu[i][4]), dtype_);
        B_11[i] = addTensor(B_11[i], term2, N2, N2, dtype_);
        B_11[i] = addTensor(B_11[i], term3, N2, N2, dtype_);
        freeDeviceTensor(term2);
        freeDeviceTensor(term3);

        // B_12[i] = exp_constant[i,1]*A_12 + exp_constant[i,3]*A3_12
        B_12[i] = scaleTensor(A_12, N2, N2, complexf(exp_constant_cpu[i][1]), dtype_);
        term2 = scaleTensor(A3_12, N2, N2, complexf(exp_constant_cpu[i][3]), dtype_);
        B_12[i] = addTensor(B_12[i], term2, N2, N2, dtype_);
        freeDeviceTensor(term2);

        // B_21[i] = exp_constant[i,1]*A_21 + exp_constant[i,3]*A3_21
        B_21[i] = scaleTensor(A_21, N2, N2, complexf(exp_constant_cpu[i][1]), dtype_);
        term2 = scaleTensor(A3_21, N2, N2, complexf(exp_constant_cpu[i][3]), dtype_);
        B_21[i] = addTensor(B_21[i], term2, N2, N2, dtype_);
        freeDeviceTensor(term2);

        // B_22[i] = exp_constant[i,0]*I2 + exp_constant[i,2]*A2_22 + exp_constant[i,4]*A6_22
        B_22[i] = scaleTensor(I2, N2, N2, complexf(exp_constant_cpu[i][0]), dtype_);
        term2 = scaleTensor(A2_22, N2, N2, complexf(exp_constant_cpu[i][2]), dtype_);
        term3 = scaleTensor(A6_22, N2, N2, complexf(exp_constant_cpu[i][4]), dtype_);
        B_22[i] = addTensor(B_22[i], term2, N2, N2, dtype_);
        B_22[i] = addTensor(B_22[i], term3, N2, N2, dtype_);
        freeDeviceTensor(term2);
        freeDeviceTensor(term3);
    }

    // Cleanup A matrices
    freeDeviceTensor(A_12);
    freeDeviceTensor(A_21);
    freeDeviceTensor(A2_11);
    freeDeviceTensor(A2_22);
    freeDeviceTensor(A3_12);
    freeDeviceTensor(A3_21);
    freeDeviceTensor(A6_11);
    freeDeviceTensor(A6_22);

    // -------------------------- Step 10: Compute A9 Matrices --------------------------
    // A9_11 = B_11[0]@B_11[4] + B_12[0]@B_21[4] + B_11[3]
    void* A9_11 = matMul(B_11[0], B_11[4], N2, N2, N2, dtype_);
    void* term = matMul(B_12[0], B_21[4], N2, N2, N2, dtype_);
    A9_11 = addTensor(A9_11, term, N2, N2, dtype_);
    freeDeviceTensor(term);
    A9_11 = addTensor(A9_11, B_11[3], N2, N2, dtype_);

    // A9_12 = B_11[0]@B_12[4] + B_12[0]@B_22[4] + B_12[3]
    void* A9_12 = matMul(B_11[0], B_12[4], N2, N2, N2, dtype_);
    term = matMul(B_12[0], B_22[4], N2, N2, N2, dtype_);
    A9_12 = addTensor(A9_12, term, N2, N2, dtype_);
    freeDeviceTensor(term);
    A9_12 = addTensor(A9_12, B_12[3], N2, N2, dtype_);

    // A9_21 = B_21[0]@B_11[4] + B_22[0]@B_21[4] + B_21[3]
    void* A9_21 = matMul(B_21[0], B_11[4], N2, N2, N2, dtype_);
    term = matMul(B_22[0], B_21[4], N2, N2, N2, dtype_);
    A9_21 = addTensor(A9_21, term, N2, N2, dtype_);
    freeDeviceTensor(term);
    A9_21 = addTensor(A9_21, B_21[3], N2, N2, dtype_);

    // A9_22 = B_21[0]@B_12[4] + B_22[0]@B_22[4] + B_22[3]
    void* A9_22 = matMul(B_21[0], B_12[4], N2, N2, N2, dtype_);
    term = matMul(B_22[0], B_22[4], N2, N2, N2, dtype_);
    A9_22 = addTensor(A9_22, term, N2, N2, dtype_);
    freeDeviceTensor(term);
    A9_22 = addTensor(A9_22, B_22[3], N2, N2, dtype_);

    // -------------------------- Step 11: Compute expA Matrices --------------------------
    // expA_11 = B_11[1] + (B_11[2] + A9_11)@A9_11 + (B_12[2] + A9_12)@A9_21
    void* expA_11 = copyTensor(B_11[1], N2, N2, dtype_);
    term = addTensor(B_11[2], A9_11, N2, N2, dtype_);
    void* term2 = matMul(term, A9_11, N2, N2, N2, dtype_);
    expA_11 = addTensor(expA_11, term2, N2, N2, dtype_);
    freeDeviceTensor(term);
    freeDeviceTensor(term2);

    term = addTensor(B_12[2], A9_12, N2, N2, dtype_);
    term2 = matMul(term, A9_21, N2, N2, N2, dtype_);
    expA_11 = addTensor(expA_11, term2, N2, N2, dtype_);
    freeDeviceTensor(term);
    freeDeviceTensor(term2);

    // expA_12 = B_12[1] + (B_11[2] + A9_11)@A9_12 + (B_12[2] + A9_12)@A9_22
    void* expA_12 = copyTensor(B_12[1], N2, N2, dtype_);
    term = addTensor(B_11[2], A9_11, N2, N2, dtype_);
    term2 = matMul(term, A9_12, N2, N2, N2, dtype_);
    expA_12 = addTensor(expA_12, term2, N2, N2, dtype_);
    freeDeviceTensor(term);
    freeDeviceTensor(term2);

    term = addTensor(B_12[2], A9_12, N2, N2, dtype_);
    term2 = matMul(term, A9_22, N2, N2, N2, dtype_);
    expA_12 = addTensor(expA_12, term2, N2, N2, dtype_);
    freeDeviceTensor(term);
    freeDeviceTensor(term2);

    // expA_21 = B_21[1] + (B_21[2] + A9_21)@A9_11 + (B_22[2] + A9_22)@A9_21
    void* expA_21 = copyTensor(B_21[1], N2, N2, dtype_);
    term = addTensor(B_21[2], A9_21, N2, N2, dtype_);
    term2 = matMul(term, A9_11, N2, N2, N2, dtype_);
    expA_21 = addTensor(expA_21, term2, N2, N2, dtype_);
    freeDeviceTensor(term);
    freeDeviceTensor(term2);

    term = addTensor(B_22[2], A9_22, N2, N2, dtype_);
    term2 = matMul(term, A9_21, N2, N2, N2, dtype_);
    expA_21 = addTensor(expA_21, term2, N2, N2, dtype_);
    freeDeviceTensor(term);
    freeDeviceTensor(term2);

    // expA_22 = B_22[1] + (B_21[2] + A9_21)@A9_12 + (B_22[2] + A9_22)@A9_22
    void* expA_22 = copyTensor(B_22[1], N2, N2, dtype_);
    term = addTensor(B_21[2], A9_21, N2, N2, dtype_);
    term2 = matMul(term, A9_12, N2, N2, N2, dtype_);
    expA_22 = addTensor(expA_22, term2, N2, N2, dtype_);
    freeDeviceTensor(term);
    freeDeviceTensor(term2);

    term = addTensor(B_22[2], A9_22, N2, N2, dtype_);
    term2 = matMul(term, A9_22, N2, N2, N2, dtype_);
    expA_22 = addTensor(expA_22, term2, N2, N2, dtype_);
    freeDeviceTensor(term);
    freeDeviceTensor(term2);

    // Cleanup A9/B matrices
    freeDeviceTensor(A9_11);
    freeDeviceTensor(A9_12);
    freeDeviceTensor(A9_21);
    freeDeviceTensor(A9_22);
    for (int i = 0; i < 5; ++i) {
        freeDeviceTensor(B_11[i]);
        freeDeviceTensor(B_12[i]);
        freeDeviceTensor(B_21[i]);
        freeDeviceTensor(B_22[i]);
    }

    // -------------------------- Step 12: Repeated Squaring (m times) --------------------------
    for (int k = 0; k < m; ++k) {
        // tmp11 = expA_11@expA_11 + expA_12@expA_21
        void* tmp11 = matMul(expA_11, expA_11, N2, N2, N2, dtype_);
        void* tmp = matMul(expA_12, expA_21, N2, N2, N2, dtype_);
        tmp11 = addTensor(tmp11, tmp, N2, N2, dtype_);
        freeDeviceTensor(tmp);

        // tmp12 = expA_11@expA_12 + expA_12@expA_22
        void* tmp12 = matMul(expA_11, expA_12, N2, N2, N2, dtype_);
        tmp = matMul(expA_12, expA_22, N2, N2, N2, dtype_);
        tmp12 = addTensor(tmp12, tmp, N2, N2, dtype_);
        freeDeviceTensor(tmp);

        // tmp21 = expA_21@expA_11 + expA_22@expA_21
        void* tmp21 = matMul(expA_21, expA_11, N2, N2, N2, dtype_);
        tmp = matMul(expA_22, expA_21, N2, N2, N2, dtype_);
        tmp21 = addTensor(tmp21, tmp, N2, N2, dtype_);
        freeDeviceTensor(tmp);

        // tmp22 = expA_21@expA_12 + expA_22@expA_22
        void* tmp22 = matMul(expA_21, expA_12, N2, N2, N2, dtype_);
        tmp = matMul(expA_22, expA_22, N2, N2, N2, dtype_);
        tmp22 = addTensor(tmp22, tmp, N2, N2, dtype_);
        freeDeviceTensor(tmp);

        // Update expA matrices
        freeDeviceTensor(expA_11);
        freeDeviceTensor(expA_12);
        freeDeviceTensor(expA_21);
        freeDeviceTensor(expA_22);
        expA_11 = tmp11;
        expA_12 = tmp12;
        expA_21 = tmp21;
        expA_22 = tmp22;
    }

    // -------------------------- Step 13: Convert to Layer S-Matrix --------------------------
    // E = expA_11 + 1j * Vf @ expA_21
    void* Vf_expA_21 = matMul(Vf_, expA_21, N2, N2, N2, dtype_);
    Vf_expA_21 = scaleTensor(Vf_expA_21, N2, N2, complexf(0.0, 1.0), dtype_); // Multiply by 1j
    void* E_mat = addTensor(expA_11, Vf_expA_21, N2, N2, dtype_);
    freeDeviceTensor(Vf_expA_21);

    // F = (1j * expA_12 - Vf @ expA_22) @ Vf
    void* j_expA_12 = scaleTensor(expA_12, N2, N2, complexf(0.0, 1.0), dtype_);
    void* Vf_expA_22 = matMul(Vf_, expA_22, N2, N2, N2, dtype_);
    void* F_part = subTensor(j_expA_12, Vf_expA_22, N2, N2, dtype_);
    freeDeviceTensor(j_expA_12);
    freeDeviceTensor(Vf_expA_22);
    void* F_mat = matMul(F_part, Vf_, N2, N2, N2, dtype_);
    freeDeviceTensor(F_part);

    // T22 = 0.5 * (E + F)
    void* T22 = addTensor(E_mat, F_mat, N2, N2, dtype_);
    T22 = scaleTensor(T22, N2, N2, complexf(0.5, 0.0), dtype_);

    // T21 = 0.5 * (E - F)
    void* T21 = subTensor(E_mat, F_mat, N2, N2, dtype_);
    T21 = scaleTensor(T21, N2, N2, complexf(0.5, 0.0), dtype_);

    // S11 = T22⁻¹
    void* S11 = matrixInverse(T22, N2, dtype_);

    // S12 = -S11 @ T21
    void* S12 = matMul(S11, T21, N2, N2, N2, dtype_);
    S12 = scaleTensor(S12, N2, N2, complexf(-1.0, 0.0), dtype_);

    // Cleanup expA/E/F/T matrices
    freeDeviceTensor(expA_11);
    freeDeviceTensor(expA_12);
    freeDeviceTensor(expA_21);
    freeDeviceTensor(expA_22);
    freeDeviceTensor(E_mat);
    freeDeviceTensor(F_mat);
    freeDeviceTensor(T22);
    freeDeviceTensor(T21);

    // -------------------------- Step 14: Repeated Squaring (n_repeatedSquaring times) --------------------------
    for (int i = 0; i < n_repeatedSquaring; ++i) {
        // R = I2 - S12 @ S12
        void* S12_S12 = matMul(S12, S12, N2, N2, N2, dtype_);
        void* R = subTensor(I2, S12_S12, N2, N2, dtype_);
        freeDeviceTensor(S12_S12);

        // tmp = S12 + S11 @ (R⁻¹ @ (S12 @ S11))
        void* S12_S11 = matMul(S12, S11, N2, N2, N2, dtype_);
        void* R_inv = matrixInverse(R, N2, dtype_);
        void* Rinv_S12_S11 = matMul(R_inv, S12_S11, N2, N2, N2, dtype_);
        freeDeviceTensor(R);
        freeDeviceTensor(S12_S11);

        void* S11_Rinv = matMul(S11, Rinv_S12_S11, N2, N2, N2, dtype_);
        freeDeviceTensor(Rinv_S12_S11);
        void* tmp = addTensor(S12, S11_Rinv, N2, N2, dtype_);
        freeDeviceTensor(S11_Rinv);

        // S11 = S11 @ R⁻¹ @ S11
        void* S11_Rinv_new = matMul(S11, R_inv, N2, N2, N2, dtype_);
        freeDeviceTensor(R_inv);
        void* new_S11 = matMul(S11_Rinv_new, S11, N2, N2, N2, dtype_);
        freeDeviceTensor(S11_Rinv_new);

        // Update S matrices
        freeDeviceTensor(S11);
        freeDeviceTensor(S12);
        S11 = new_S11;
        S12 = tmp;
    }

    // -------------------------- Step 15: Store Layer S-Matrices --------------------------
    layer_S11_.push_back(S11);
    layer_S12_.push_back(S12);
    layer_S21_.push_back(copyTensor(S12, N2, N2, dtype_));
    layer_S22_.push_back(copyTensor(S11, N2, N2, dtype_));

    // -------------------------- Step 16: Cleanup --------------------------
    freeDeviceTensor(I);
    freeDeviceTensor(I2);
    freeDeviceTensor(KxKx);
    freeDeviceTensor(KyKy);
    freeDeviceTensor(KxKy);
    freeDeviceTensor(KxEiKy);
    freeDeviceTensor(KyEiKx);
    freeDeviceTensor(KxEiKx);
    freeDeviceTensor(KyEiKy);
    freeDeviceTensor(KyEiKy_minus_I);
    freeDeviceTensor(P_top);
    freeDeviceTensor(I_minus_KxEiKx);
    freeDeviceTensor(neg_KyEiKx);
    freeDeviceTensor(P_bot);
    freeDeviceTensor(neg_KxKy);
    freeDeviceTensor(E_minus_KyKy);
    freeDeviceTensor(Q_top);
    freeDeviceTensor(KxKx_minus_E);
    freeDeviceTensor(Q_bot);

    // Clear CUDA cache
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaDeviceReset()); // Optional: clears device cache
}

void RCWA::computeLayerExpHomogenous(void* eps, double thickness) {
    auto [V, kz_norm] = getV(thickness); // eps value passed as thickness for demo
    
    // Create kz_norm vector (duplicated for 2N)
    void* kz_norm_2N = createDeviceTensor(2*order_N_, 1, dtype_);
    CHECK_CUDA(cudaMemcpy(kz_norm_2N, kz_norm, order_N_ * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(static_cast<char*>(kz_norm_2N) + order_N_ * sizeof(double),
                          kz_norm, order_N_ * sizeof(double), cudaMemcpyDeviceToDevice));

    // Phase matrix: diag(exp(1j*omega*kz*thickness))
    void* phase = createDeviceTensor(2*order_N_, 2*order_N_, dtype_);
    fillEye(phase, 2*order_N_, dtype_);
    // To be implemented: apply phase factor

    // Calculate S-matrix (simplified)
    void* S11 = createDeviceTensor(2*order_N_, 2*order_N_, dtype_);
    void* S12 = createDeviceTensor(2*order_N_, 2*order_N_, dtype_);
    fillEye(S11, 2*order_N_, dtype_);
    fillZero(S12, 2*order_N_, 2*order_N_, dtype_);

    layer_S11_.push_back(S11);
    layer_S12_.push_back(S12);
    layer_S21_.push_back(copyTensor(S12, 2*order_N_, 2*order_N_, dtype_));
    layer_S22_.push_back(copyTensor(S11, 2*order_N_, 2*order_N_, dtype_));

    // Cleanup
    freeDeviceTensor(V);
    freeDeviceTensor(kz_norm);
    freeDeviceTensor(kz_norm_2N);
    freeDeviceTensor(phase);
}

std::tuple<void*, void*> RCWA::getV(double eps) {
    int N = order_N_;
    void* kz_norm = computeKzNorm(createDeviceScalar(eps));

    // Create V matrix (simplified - full implementation requires E→H transformation)
    void* V = createDeviceTensor(2*N, 2*N, dtype_);
    fillEye(V, 2*N, dtype_);

    return {V, kz_norm};
}

void RCWA::computeInputLayerSmatrix() {
    // Simplified input layer S-matrix calculation
    int N = 2 * order_N_;
    void* S11 = createDeviceTensor(N, N, dtype_);
    void* S12 = createDeviceTensor(N, N, dtype_);
    void* S21 = createDeviceTensor(N, N, dtype_);
    void* S22 = createDeviceTensor(N, N, dtype_);
    
    fillEye(S11, N, dtype_);
    fillEye(S22, N, dtype_);
    fillZero(S12, N, N, dtype_);
    fillZero(S21, N, N, dtype_);

    Sin_ = {S11, S21, S12, S22};
}

void RCWA::computeOutputLayerSmatrix() {
    // Simplified output layer S-matrix calculation
    int N = 2 * order_N_;
    void* S11 = createDeviceTensor(N, N, dtype_);
    void* S12 = createDeviceTensor(N, N, dtype_);
    void* S21 = createDeviceTensor(N, N, dtype_);
    void* S22 = createDeviceTensor(N, N, dtype_);
    
    fillEye(S11, N, dtype_);
    fillEye(S22, N, dtype_);
    fillZero(S12, N, N, dtype_);
    fillZero(S21, N, N, dtype_);

    Sout_ = {S11, S21, S12, S22};
}

// -------------------------- CUDA Utility Functions --------------------------
void* RCWA::createDeviceTensor(int rows, int cols, DType dtype) {
    size_t elem_size = (dtype == DType::COMPLEX64) ? sizeof(complexf) : sizeof(complexd);
    void* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, rows * cols * elem_size));
    fillZero(ptr, rows, cols, dtype);
    return ptr;
}

void* RCWA::createDeviceScalar(double val) {
    void* ptr = createDeviceTensor(1, 1, dtype_);
    CHECK_CUDA(cudaMemcpy(ptr, &val, sizeof(double), cudaMemcpyHostToDevice));
    return ptr;
}

void RCWA::fillEye(void* ptr, int N, DType dtype) {
    fillZero(ptr, N, N, dtype);
    size_t elem_size = (dtype == DType::COMPLEX64) ? sizeof(complexf) : sizeof(complexd);
    
    for (int i = 0; i < N; ++i) {
        complexf val(1.0, 0.0);
        CHECK_CUDA(cudaMemcpy(static_cast<char*>(ptr) + (i * N + i) * elem_size,
                              &val, elem_size, cudaMemcpyHostToDevice));
    }
}

void RCWA::fillZero(void* ptr, int rows, int cols, DType dtype) {
    size_t elem_size = (dtype == DType::COMPLEX64) ? sizeof(complexf) : sizeof(complexd);
    CHECK_CUDA(cudaMemset(ptr, 0, rows * cols * elem_size));
}

void* RCWA::matrixInverse(void* A, int N, DType dtype) {
    void* A_copy = copyTensor(A, N, N, dtype);
    int* ipiv = new int[N];
    int info = 0;

    if (dtype == DType::COMPLEX64) {
        // Single-precision complex inversion (to be implemented with cuSOLVER Zgetrf/Zgetri)
        throw std::runtime_error("Complex64 matrix inversion not implemented");
    } else {
        CHECK_CUSOLVER(cusolverDnDgetrf(cusolver_handle_, N, N,
                                        static_cast<double*>(A_copy), N, ipiv, &info));
        if (info != 0) throw std::runtime_error("LU decomposition failed (info=" + std::to_string(info) + ")");

        int lwork = 0;
        CHECK_CUSOLVER(cusolverDnDgetri_bufferSize(cusolver_handle_, N,
                                                   static_cast<double*>(A_copy), N, &lwork));
        void* work = createDeviceTensor(lwork, 1, dtype);

        CHECK_CUSOLVER(cusolverDnDgetri(cusolver_handle_, N,
                                        static_cast<double*>(A_copy), N, ipiv,
                                        static_cast<double*>(work), lwork, &info));
        if (info != 0) throw std::runtime_error("Matrix inversion failed (info=" + std::to_string(info) + ")");

        freeDeviceTensor(work);
    }

    delete[] ipiv;
    return A_copy;
}

void* RCWA::matrixSolve(void* A, void* B, int N, DType dtype) {
    void* A_copy = copyTensor(A, N, N, dtype);
    void* B_copy = copyTensor(B, N, N, dtype);
    int* ipiv = new int[N];
    int info = 0;

    if (dtype == DType::COMPLEX64) {
        throw std::runtime_error("Complex64 matrix solve not implemented");
    } else {
        CHECK_CUSOLVER(cusolverDnDgesv(cusolver_handle_, N, N,
                                       static_cast<double*>(A_copy), N, ipiv,
                                       static_cast<double*>(B_copy), N, &info));
        if (info != 0) throw std::runtime_error("Linear solve failed (info=" + std::to_string(info) + ")");
    }

    delete[] ipiv;
    freeDeviceTensor(A_copy);
    return B_copy;
}

void* RCWA::matMul(void* A, void* B, int m, int k, int n, DType dtype) {
    void* C = createDeviceTensor(m, n, dtype);
    fillZero(C, m, n, dtype);

    const complexf alpha_f(1.0, 0.0), beta_f(0.0, 0.0);
    const complexd alpha_d(1.0, 0.0), beta_d(0.0, 0.0);

    if (dtype == DType::COMPLEX64) {
        CHECK_CUBLAS(cublasCgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, k,
                                 reinterpret_cast<const cuComplex*>(&alpha_f),
                                 static_cast<cuComplex*>(A), m,
                                 static_cast<cuComplex*>(B), k,
                                 reinterpret_cast<const cuComplex*>(&beta_f),
                                 static_cast<cuComplex*>(C), m));
    } else {
        CHECK_CUBLAS(cublasZgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, k,
                                 reinterpret_cast<const cuDoubleComplex*>(&alpha_d),
                                 static_cast<cuDoubleComplex*>(A), m,
                                 static_cast<cuDoubleComplex*>(B), k,
                                 reinterpret_cast<const cuDoubleComplex*>(&beta_d),
                                 static_cast<cuDoubleComplex*>(C), m));
    }

    return C;
}

std::vector<int> RCWA::matchingIndices(const std::vector<std::vector<int>>& orders) {
    std::vector<int> indices;
    int* order_x_cpu = new int[order_x_N_];
    int* order_y_cpu = new int[order_y_N_];
    
    CHECK_CUDA(cudaMemcpy(order_x_cpu, order_x_, order_x_N_ * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(order_y_cpu, order_y_, order_y_N_ * sizeof(int), cudaMemcpyDeviceToHost));

    for (const auto& ord : orders) {
        int x = std::clamp(ord[0], -order_[0], order_[0]);
        int y = std::clamp(ord[1], -order_[1], order_[1]);
        int x_idx = x + order_[0];
        int y_idx = y + order_[1];
        indices.push_back(y_idx * order_x_N_ + x_idx);
    }

    delete[] order_x_cpu;
    delete[] order_y_cpu;
    return indices;
}

void* RCWA::computeKzNorm(void* eps) {
    double eps_val;
    CHECK_CUDA(cudaMemcpy(&eps_val, eps, sizeof(double), cudaMemcpyDeviceToHost));

    // Calculate kz_norm = sqrt(eps - Kx² - Ky²)
    void* kz_norm = createDeviceTensor(order_N_, 1, dtype_);
    // To be implemented: element-wise square root calculation

    return kz_norm;
}

void RCWA::computeKzNorm(void* eps, void* kz_norm) {
    double eps_val;
    CHECK_CUDA(cudaMemcpy(&eps_val, eps, sizeof(double), cudaMemcpyDeviceToHost));
    // To be implemented: fill kz_norm with sqrt(eps - Kx² - Ky²)
}

// -------------------------- Basic Matrix Operations --------------------------
void* RCWA::addTensor(void* A, void* B, int rows, int cols, DType dtype) {
    // Create output tensor (initialized as copy of A)
    void* C = copyTensor(A, rows, cols, dtype);
    const int N = rows * cols; // Total elements (vectorized for cuBLAS)

    // cuBLAS alpha/beta for AXPY: C = alpha*B + beta*C (beta=1 → C = A + B)
    if (dtype == DType::COMPLEX64) {
        const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
        CHECK_CUBLAS(cublasCaxpy(
            cublas_handle_,    // cuBLAS handle
            N,                 // Number of elements
            &alpha,            // Scalar multiplier for B
            (cuComplex*)B,     // Input vector B
            1,                 // Stride of B
            (cuComplex*)C,     // Output vector C (initialized as A)
            1                  // Stride of C
        ));
    } else { // COMPLEX128
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        CHECK_CUBLAS(cublasZaxpy(
            cublas_handle_,
            N,
            &alpha,
            (cuDoubleComplex*)B,
            1,
            (cuDoubleComplex*)C,
            1
        ));
    }

    return C;
}

void* RCWA::subTensor(void* A, void* B, int rows, int cols, DType dtype) {
    // Create output tensor (initialized as copy of A)
    void* C = copyTensor(A, rows, cols, dtype);
    const int N = rows * cols; // Total elements (vectorized for cuBLAS)

    // cuBLAS alpha/beta for AXPY: C = alpha*B + beta*C (alpha=-1 → C = A - B)
    if (dtype == DType::COMPLEX64) {
        const cuComplex alpha = make_cuComplex(-1.0f, 0.0f);
        CHECK_CUBLAS(cublasCaxpy(
            cublas_handle_,
            N,
            &alpha,
            (cuComplex*)B,
            1,
            (cuComplex*)C,
            1
        ));
    } else { // COMPLEX128
        const cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0);
        CHECK_CUBLAS(cublasZaxpy(
            cublas_handle_,
            N,
            &alpha,
            (cuDoubleComplex*)B,
            1,
            (cuDoubleComplex*)C,
            1
        ));
    }

    return C;
}

void* RCWA::scaleTensor(void* A, int rows, int cols, complexf val, DType dtype) {
    void* C = copyTensor(A, rows, cols, dtype);
    const int N = rows * cols;

    if (dtype == DType::COMPLEX64) {
        const cuComplex alpha = make_cuComplex(val.real(), val.imag());
        CHECK_CUBLAS(cublasCscal(
            cublas_handle_,
            N,
            &alpha,
            (cuComplex*)C,
            1
        ));
    } else { // COMPLEX128
        const cuDoubleComplex alpha = make_cuDoubleComplex(val.real(), val.imag());
        CHECK_CUBLAS(cublasZscal(
            cublas_handle_,
            N,
            &alpha,
            (cuDoubleComplex*)C,
            1
        ));
    }
    return C;
}

void* RCWA::copyTensor(void* A, int rows, int cols, DType dtype) {
    void* C = createDeviceTensor(rows, cols, dtype);
    size_t elem_size = (dtype == DType::COMPLEX64) ? sizeof(complexf) : sizeof(complexd);
    CHECK_CUDA(cudaMemcpy(C, A, rows * cols * elem_size, cudaMemcpyDeviceToDevice));
    return C;
}

void* RCWA::sliceTensor(void* A, int start_row, int end_row, int start_col, int end_col, DType dtype) {
    int out_rows = end_row - start_row;
    int out_cols = end_col - start_col;
    void* C = createDeviceTensor(out_rows, out_cols, dtype);
    size_t elem_size = (dtype == DType::COMPLEX64) ? sizeof(complexf) : sizeof(complexd);

    // Simplified CPU-based slicing
    int in_rows = 2 * order_N_; // Assumed for RCWA
    complexf* A_cpu = new complexf[in_rows * (2 * order_N_)];
    complexf* C_cpu = new complexf[out_rows * out_cols];
    CHECK_CUDA(cudaMemcpy(A_cpu, A, in_rows * (2 * order_N_) * elem_size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            C_cpu[i * out_cols + j] = A_cpu[(start_row + i) * (2 * order_N_) + (start_col + j)];
        }
    }

    CHECK_CUDA(cudaMemcpy(C, C_cpu, out_rows * out_cols * elem_size, cudaMemcpyHostToDevice));
    delete[] A_cpu;
    delete[] C_cpu;

    return C;
}

void* RCWA::reshapeTensor(void* A, int in_rows, int in_cols, int out_rows, int out_cols, DType dtype) {
    if (in_rows * in_cols != out_rows * out_cols) {
        throw std::runtime_error("Reshape error: input/output size mismatch");
    }

    void* C = createDeviceTensor(out_rows, out_cols, dtype);
    size_t elem_size = (dtype == DType::COMPLEX64) ? sizeof(complexf) : sizeof(complexd);

    // Simplified CPU-based reshaping
    complexf* A_cpu = new complexf[in_rows * in_cols];
    complexf* C_cpu = new complexf[out_rows * out_cols];
    CHECK_CUDA(cudaMemcpy(A_cpu, A, in_rows * in_cols * elem_size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < in_rows * in_cols; ++i) {
        C_cpu[i] = A_cpu[i];
    }

    CHECK_CUDA(cudaMemcpy(C, C_cpu, out_rows * out_cols * elem_size, cudaMemcpyHostToDevice));
    delete[] A_cpu;
    delete[] C_cpu;

    return C;
}

// -------------------------- Helper: Matrix Norm Calculation --------------------------
double RCWA::computeMatrixNorm(void* A, int rows, int cols, DType dtype) {
    size_t elem_size = (dtype == DType::COMPLEX64) ? sizeof(cuComplex) : sizeof(cuDoubleComplex);
    std::vector<double> col_norms(col, 0.0);

    // Copy matrix to CPU (replace with kernel for full GPU speed)
    void* A_cpu = malloc(rows * cols * elem_size);
    CHECK_CUDA(cudaMemcpy(A_cpu, A, rows * cols * elem_size, cudaMemcpyDeviceToHost));

    if (dtype == DType::COMPLEX64) {
        cuComplex* A_cplx = static_cast<cuComplex*>(A_cpu);
        for (int j = 0; j < cols; ++j) {
            double norm = 0.0;
            for (int i = 0; i < rows; ++i) {
                int idx = i * cols + j;
                norm += sqrt(pow(A_cplx[idx].x, 2) + pow(A_cplx[idx].y, 2));
            }
            col_norms[j] = norm;
        }
    } else {
        cuDoubleComplex* A_cplx = static_cast<cuDoubleComplex*>(A_cpu);
        for (int j = 0; j < cols; ++j) {
            double norm = 0.0;
            for (int i = 0; i < rows; ++i) {
                int idx = i * cols + j;
                norm += sqrt(pow(A_cplx[idx].x, 2) + pow(A_cplx[idx].y, 2));
            }
            col_norms[j] = norm;
        }
    }

    free(A_cpu);
    return *std::max_element(col_norms.begin(), col_norms.end());
}

// -------------------------- Helper: Matrix Horizontal Stack --------------------------
void* RCWA::hstackTensor(void* A, void* B, int rows, int colsA, int colsB, DType dtype) {
    int cols_out = colsA + colsB;
    void* C = createDeviceTensor(rows, cols_out, dtype_);
    size_t elem_size = (dtype == DType::COMPLEX64) ? sizeof(cuComplex) : sizeof(cuDoubleComplex);

    // Copy A to left part of C
    for (int i = 0; i < rows; ++i) {
        CHECK_CUDA(cudaMemcpy(
            static_cast<char*>(C) + i * cols_out * elem_size,
            static_cast<char*>(A) + i * colsA * elem_size,
            colsA * elem_size,
            cudaMemcpyDeviceToDevice
        ));
    }

    // Copy B to right part of C
    for (int i = 0; i < rows; ++i) {
        CHECK_CUDA(cudaMemcpy(
            static_cast<char*>(C) + i * cols_out * elem_size + colsA * elem_size,
            static_cast<char*>(B) + i * colsB * elem_size,
            colsB * elem_size,
            cudaMemcpyDeviceToDevice
        ));
    }

    return C;
}

// -------------------------- Helper: Matrix Vertical Stack --------------------------
void* RCWA::vstackTensor(void* A, void* B, int rowsA, int rowsB, int cols, DType dtype) {
    int rows_out = rowsA + rowsB;
    void* C = createDeviceTensor(rows_out, cols, dtype_);
    size_t elem_size = (dtype == DType::COMPLEX64) ? sizeof(cuComplex) : sizeof(cuDoubleComplex);

    // Copy A to top part of C
    CHECK_CUDA(cudaMemcpy(
        C,
        A,
        rowsA * cols * elem_size,
        cudaMemcpyDeviceToDevice
    ));

    // Copy B to bottom part of C
    CHECK_CUDA(cudaMemcpy(
        static_cast<char*>(C) + rowsA * cols * elem_size,
        B,
        rowsB * cols * elem_size,
        cudaMemcpyDeviceToDevice
    ));

    return C;
}