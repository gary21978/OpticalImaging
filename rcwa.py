import warnings
import torch
import torch.fft
from typing import Optional
import numpy as np



class rcwa:
    # Simulation setting
    def __init__(self, freq, order, L, *,
                 dtype=torch.complex64,
                 device=torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu'),
                 fast=False,
                 method='free'
                 ):
        '''
            Rigorous Coupled Wave Analysis
            - Lorentz-Heaviside units
            - Speed of light: 1
            - Time harmonics notation: exp(-jωt)

            Parameters
            - freq: simulation frequency (unit: length^-1)
            - order: Fourier order [x_order (int), y_order (int)]
            - L: Lattice constant [Lx, Ly] (unit: length)

            Keyword Parameters
            - dtype: simulation data type (only torch.complex64 and torch.complex128 are allowed.)
            - device: simulation device (only torch.device('cpu') and torch.device('cuda') are allowed.)
            - fast: use fast exponential method (GEMM or FREE)
            - method: 'free' (default) or 'gemm' — layer S-matrix computation method
                 'free': Cos·Sinc direct from P·Q, lower memory, Order18 Taylor
                 'gemm': matrix exponential exp(A) via block 2n×2n Taylor, higher memory
        '''

        if method not in ('free', 'gemm'):
            raise ValueError(f"method must be 'free' or 'gemm', got '{method}'")
        self.method = method

        # Hardware
        if dtype != torch.complex64 and dtype != torch.complex128:
            warnings.warn(
                'Invalid simulation data type. Set as torch.complex64.', UserWarning)
            self._dtype = torch.complex64
        else:
            self._dtype = dtype
        self._device = device
        torch.set_grad_enabled(False)  # disable gradient calc

        # Simulation parameters
        self.freq = torch.as_tensor(
            freq, dtype=self._dtype, device=self._device)  # unit^-1
        self.omega = 2 * torch.pi * freq
        self.L = torch.as_tensor(L, dtype=self._dtype, device=self._device)
        self.fast_exp = fast

        # Fourier order
        self.order = order
        self.order_x = torch.linspace(-self.order[0], self.order[0], 2 * self.order[0] + 1, dtype=torch.int64,
                                      device=self._device)
        self.order_y = torch.linspace(-self.order[1], self.order[1], 2 * self.order[1] + 1, dtype=torch.int64,
                                      device=self._device)
        self.order_N = len(self.order_x) * len(self.order_y)

        # Lattice vector
        self.L = L  # unit
        # 使用1/𝜆进行归一化
        self.Gx_norm, self.Gy_norm = 1 / \
            (L[0] * self.freq), 1 / (L[1] * self.freq)

        # Input and output layer (Default: free space)
        self.eps_in = torch.tensor(1., dtype=self._dtype, device=self._device)
        self.eps_out = torch.tensor(1., dtype=self._dtype, device=self._device)

        # Internal layers
        self.layer_N = 0  # total number of layers
        self.thickness = []
        self.eps_conv = []
        self.P = []
        self.Q = []
        self.PQ = []

        # Internal layer eigenmodes
        self.kz_norm, self.E_eigvec, self.H_eigvec = [], [], []

        # Single layer scattering matrices
        self.layer_S11, self.layer_S21, self.layer_S12, self.layer_S22 = [], [], [], []

        self.exp_constant = torch.tensor(
            [[0, -0.10036558103014462001, -0.00802924648241156960, -0.00089213849804572995, 0],
             [0, 0.39784974949964507614, 1.36783778460411719922,
                 0.49828962252538267755, -0.00063789819459472330],
             [-10.9676396052962062593, 1.68015813878906197182, 0.05717798464788655127, -0.00698210122488052084,
              0.00003349750170860705],
             [-0.0904316832390810561, -0.06764045190713819075, 0.06759613017704596460, 0.02955525704293155274,
              -0.00001391802575160607],
             [0, 0, -0.09233646193671185927, -0.01693649390020817171, -0.00001400867981820361]])
        self.layer_exp = []

    def add_input_layer(self, eps=1.):
        '''
            Add input layer
            - If this function is not used, simulation will be performed under free space input layer.

            Parameters
            - eps: relative permittivity
        '''

        self.eps_in = torch.as_tensor(
            eps, dtype=self._dtype, device=self._device)
        self.Sin = []

    def add_output_layer(self, eps=1.):
        '''
            Add output layer
            - If this function is not used, simulation will be performed under free space output layer.

            Parameters
            - eps: relative permittivity
        '''

        self.eps_out = torch.as_tensor(
            eps, dtype=self._dtype, device=self._device)
        self.Sout = []

    def set_incident_angle(self, inc_ang, azi_ang, angle_layer='input'):
        '''
            Set incident angle

            Parameters
            - inc_ang: incident angle (unit: radian)
            - azi_ang: azimuthal angle (unit: radian)
            - angle_layer: reference layer to calculate angle ('i', 'in', 'input' / 'o', 'out', 'output')
        '''

        self.inc_ang = torch.as_tensor(
            inc_ang, dtype=self._dtype, device=self._device)
        self.azi_ang = torch.as_tensor(
            azi_ang, dtype=self._dtype, device=self._device)

        if angle_layer in ['i', 'in', 'input']:
            self.angle_layer = 'input'
        elif angle_layer in ['o', 'out', 'output']:
            self.angle_layer = 'output'
        else:
            warnings.warn(
                'Invalid angle layer. Set as input layer.', UserWarning)
            self.angle_layer = 'input'

        self._kvectors()

    def add_layer(self, thickness, eps=1.):
        '''
            Add internal layer

            Parameters
            - thickness: layer thickness (unit: length)
            - eps: relative permittivity
        '''
        is_homogenous = ((type(eps) == float) or (type(eps) == complex) or (eps.dim() == 0) or
                         ((eps.dim() == 1) and eps.shape[0] == 1)) or torch.all(eps == eps.flatten()[0])
        if is_homogenous:
            E = torch.full((self.order_N, self.order_N), eps[0][0], dtype=self._dtype, device=self._device) * torch.eye(
                self.order_N, dtype=self._dtype, device=self._device)
        else:
            E = self._material_conv(eps)

        self.eps_conv.append(E)
        self.layer_N += 1
        self.thickness.append(thickness)

        if self.fast_exp:
            if is_homogenous:
                self._compute_layer_exp_homogenous(eps, thickness)
            elif self.method == 'free':
                self._compute_layer_free()
            else:
                self._compute_layer_exp()
        else:
            # if is_homogenous:
            #     self._eigen_decomposition_homogenous(eps)
            # else:
            self._eigen_decomposition()
            self._solve_layer_smatrix()

    def add_defect(self, layer_num=0, deps=None):
        deps_conv = self._material_conv(deps)
        self._update_layer_smatrix(layer_num, deps_conv)

    # Solve simulation
    def solve_global_smatrix(self):
        '''
            Solve global S-matrix
        '''

        # Initialization
        if self.layer_N > 0:
            S11 = self.layer_S11[0]
            S21 = self.layer_S21[0]
            S12 = self.layer_S12[0]
            S22 = self.layer_S22[0]
        else:
            N = 2 * self.order_N
            S11 = torch.eye(N, dtype=self._dtype, device=self._device)
            S21 = torch.zeros(N, N, dtype=self._dtype, device=self._device)  # ← 修复：2D zero matrix
            S12 = torch.zeros(N, N, dtype=self._dtype, device=self._device)  # ← 修复
            S22 = torch.eye(N, dtype=self._dtype, device=self._device)
    
        # ... rest of the code ...

        # Connection
        for i in range(self.layer_N - 1):
            S11, S21, S12, S22 = self._RS_prod(Sm=[S11, S21, S12, S22],
                                               Sn=[self.layer_S11[i + 1], self.layer_S21[i + 1], self.layer_S12[i + 1],
                                                   self.layer_S22[i + 1]])

        if hasattr(self, 'Sin'):
            # input layer coupling
            S11, S21, S12, S22 = self._RS_prod(Sm=[self.Sin[0], self.Sin[1], self.Sin[2], self.Sin[3]],
                                               Sn=[S11, S21, S12, S22])

        if hasattr(self, 'Sout'):
            # output layer coupling
            S11, S21, S12, S22 = self._RS_prod(Sm=[S11, S21, S12, S22],
                                               Sn=[self.Sout[0], self.Sout[1], self.Sout[2], self.Sout[3]])

        self.S = [S11, S21, S12, S22]

    def source_planewave(self, *, amplitude=[1., 0.], orders=[0, 0], direction='forward'):
        '''
            Generate planewave

            Paramters
            - amplitude: amplitudes at the matched diffraction orders ([Ex_amp, Ey_amp])
              (list / np.ndarray / torch.Tensor) (Recommended shape: 1x2)
            - direction: incident direction ('f', 'forward' / 'b', 'backward')
        '''

        self.source_fourier(amplitude=amplitude,
                            orders=orders, direction=direction)

    def source_fourier(self, *, amplitude, orders, direction='forward'):
        '''
            Generate Fourier source

            Paramters
            - amplitude: amplitudes at the matched diffraction orders [([Ex_amp, Ey_amp] at orders[0]), ..., ...]
                (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
            - orders: diffraction orders (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
            - direction: incident direction ('f', 'forward' / 'b', 'backward')
        '''
        amplitude = torch.as_tensor(
            amplitude, dtype=self._dtype, device=self._device).reshape([-1, 2])
        orders = torch.as_tensor(
            orders, dtype=torch.int64, device=self._device).reshape([-1, 2])

        if direction in ['f', 'forward']:
            direction = 'forward'
        elif direction in ['b', 'backward']:
            direction = 'backward'
        else:
            warnings.warn(
                'Invalid source direction. Set as forward.', UserWarning)
            direction = 'forward'

        # Matching indices
        order_indices = self._matching_indices(orders)
        self.source_direction = direction
        E_i = torch.zeros([2 * self.order_N, 1],
                          dtype=self._dtype, device=self._device)
        E_i[order_indices, 0] = amplitude[:, 0]
        E_i[order_indices + self.order_N, 0] = amplitude[:, 1]

        self.E_i = E_i

    def field_xy(self, x_axis, y_axis, z_posi=0, port='reflection'):
        '''
            XY-plane field distribution at the selected layer.

            Parameters
            - x_axis: x-direction sampling coordinates (torch.Tensor)
            - y_axis: y-direction sampling coordinates (torch.Tensor)

            Return
            - [Ex, Ey, Ez] (list[torch.Tensor])
        '''

        if type(x_axis) != torch.Tensor or type(y_axis) != torch.Tensor:
            warnings.warn(
                'x and y axis must be torch.Tensor type. Return None.', UserWarning)
            return None

        # [x, y, diffraction order]
        x_axis = x_axis.reshape([-1, 1, 1])
        y_axis = y_axis.reshape([1, -1, 1])

        if self.angle_layer == 'input':
            eps = self.eps_in if hasattr(self, 'eps_in') else 1.
        else:
            eps = self.eps_out if hasattr(self, 'eps_out') else 1.

        if self.source_direction == 'forward' and port == 'transmission':
            S = self.S[0]  # ← 前向透射
        elif self.source_direction == 'forward' and port == 'reflection':
            S = self.S[1]  # ← 前向反射
        elif self.source_direction == 'backward' and port == 'reflection':
            S = self.S[2]  # ← 后向反射
        elif self.source_direction == 'backward' and port == 'transmission':
            S = self.S[3]  # ← 后向透射

        Kz_norm_dn = torch.sqrt(eps - self.Kx_norm_dn **
                                2 - self.Ky_norm_dn ** 2).reshape([-1, 1])

        # Create mask for propagating waves (real Kz, imag == 0)
        prop_mask = (torch.abs(torch.imag(Kz_norm_dn))
                     <= 1e-5).type_as(Kz_norm_dn)
        prop_mask_2 = torch.vstack((prop_mask, prop_mask))

        Kz_norm_dn_2 = torch.vstack((Kz_norm_dn, Kz_norm_dn))
        z_phase = torch.exp(-1.j * self.omega * Kz_norm_dn_2 * z_posi)

        # Fourier domain fields
        # [diffraction order, diffraction order]
        Exy = torch.matmul(S, self.E_i) * z_phase

        # Apply mask to filter out evanescent waves
        Exy = Exy * prop_mask_2

        Ex_mn = Exy[:self.order_N]
        Ey_mn = Exy[self.order_N:]
        Ez_mn = (self.Kx_norm_dn.reshape(
            [-1, 1]) * Ex_mn + self.Ky_norm_dn.reshape([-1, 1]) * Ey_mn) / Kz_norm_dn

        # return Ex_mn, Ey_mn
        # Spatial domain fields
        xy_phase = torch.exp(
            1.j * self.omega * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y_axis))
        Ex = torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Ey = torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2)
        Ez = torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2)

        return Ex, Ey, Ez

    def get_kx_ky_kz_nrom(self):
        if self.angle_layer == 'input':
            eps = self.eps_in if hasattr(self, 'eps_in') else 1.
        else:
            eps = self.eps_out if hasattr(self, 'eps_out') else 1.
        Kz_norm_dn = torch.sqrt(eps - self.Kx_norm_dn **
                                2 - self.Ky_norm_dn ** 2)
        return self.Kx_norm_dn.real, self.Ky_norm_dn.real, Kz_norm_dn.real

    def Floquet_mode(self, z_posi, port='reflection'):
        if self.angle_layer == 'input':
            eps = self.eps_in if hasattr(self, 'eps_in') else 1.
        else:
            eps = self.eps_out if hasattr(self, 'eps_out') else 1.

        if self.source_direction == 'forward' and port == 'transmission':
            S = self.S[0]  # ← 前向透射
        elif self.source_direction == 'forward' and port == 'reflection':
            S = self.S[1]  # ← 前向反射
        elif self.source_direction == 'backward' and port == 'reflection':
            S = self.S[2]  # ← 后向反射
        elif self.source_direction == 'backward' and port == 'transmission':
            S = self.S[3]  # ← 后向透射

        Kz_norm_dn = torch.sqrt(eps - self.Kx_norm_dn **
                                2 - self.Ky_norm_dn ** 2).reshape([-1, 1])

        # Create mask for propagating waves (real Kz, imag == 0)
        prop_mask = (torch.abs(torch.imag(Kz_norm_dn))
                     <= 1e-5).type_as(Kz_norm_dn)
        prop_mask_2 = torch.vstack((prop_mask, prop_mask))

        Kz_norm_dn_2 = torch.vstack((Kz_norm_dn, Kz_norm_dn))
        z_phase = torch.exp(-1.j * self.omega * Kz_norm_dn_2 * z_posi)

        # Fourier domain fields
        Exy = torch.matmul(S, self.E_i) * z_phase

        # Apply mask to filter out evanescent waves
        Exy = Exy * prop_mask_2

        Ex_mn = Exy[:self.order_N]
        Ey_mn = Exy[self.order_N:]
        Ez_mn = -(self.Kx_norm_dn.reshape([-1, 1]) * Ex_mn +
                  self.Ky_norm_dn.reshape([-1, 1]) * Ey_mn) / Kz_norm_dn
        
        return Ex_mn, Ey_mn, Ez_mn
        # --- 可选：验证横波条件（调试时可用，正式运行可注释掉）---
        # 验证电场振动方向垂直于波矢方向
        # k_dot_E = (self.Kx_norm_dn.reshape([-1, 1]) * Ex_mn
        #            + self.Ky_norm_dn.reshape([-1, 1]) * Ey_mn
        #            + Kz_norm_dn * Ez_mn)
        # print("Max |k·E| for modes:", torch.max(torch.abs(k_dot_E)).item())

        # 构造 M（转换矩阵: xyz to sp）
        # M = self.construct_M(Kz_norm_dn.squeeze())

        # # xyz to sp to xy
        # E_xyz_object = torch.stack([Ex_mn, Ey_mn, Ez_mn], -1).transpose(-2, -1)
        # E_xy_pupil = M.transpose(-2, -1).to(torch.complex128) @ E_xyz_object

        # return E_xy_pupil[:, 0], E_xy_pupil[:, 1]

    # Internal functions
    def _matching_indices(self, orders):
        orders[orders[:, 0] < -self.order[0], 0] = int(-self.order[0])
        orders[orders[:, 0] > self.order[0], 0] = int(self.order[0])
        orders[orders[:, 1] < -self.order[1], 1] = int(-self.order[1])
        orders[orders[:, 1] > self.order[1], 1] = int(self.order[1])
        order_indices = len(
            self.order_y) * (orders[:, 0] + int(self.order[0])) + orders[:, 1] + int(self.order[1])

        return order_indices

    def _kvectors(self):
        if self.angle_layer == 'input':
            # 此处norm是使用k0（即2π/𝜆）进行归一化，取值范围（0-1）
            self.kx0_norm = torch.real(torch.sqrt(
                self.eps_in)) * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
            self.ky0_norm = torch.real(torch.sqrt(
                self.eps_in)) * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)
        else:
            self.kx0_norm = torch.real(torch.sqrt(
                self.eps_out)) * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
            self.ky0_norm = torch.real(torch.sqrt(
                self.eps_out)) * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)

        # Free space k-vectors and E to H transformation matrix
        self.kx_norm = self.kx0_norm + self.order_x * self.Gx_norm
        self.ky_norm = self.ky0_norm + self.order_y * self.Gy_norm

        kx_norm_grid, ky_norm_grid = torch.meshgrid(
            self.kx_norm, self.ky_norm, indexing='ij')

        self.Kx_norm_dn = torch.reshape(kx_norm_grid, (-1,))
        self.Ky_norm_dn = torch.reshape(ky_norm_grid, (-1,))
        self.Kx_norm = torch.diag(self.Kx_norm_dn)
        self.Ky_norm = torch.diag(self.Ky_norm_dn)

        Kz_norm_dn = torch.sqrt(1. - self.Kx_norm_dn **
                                2 - self.Ky_norm_dn ** 2)
        Kz_norm_dn = torch.where(torch.imag(
            Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn)
        tmp1 = torch.vstack((torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                             torch.diag(Kz_norm_dn + self.Kx_norm_dn ** 2 / Kz_norm_dn)))
        tmp2 = torch.vstack((torch.diag(-Kz_norm_dn - self.Ky_norm_dn ** 2 / Kz_norm_dn),
                             torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn)))
        self.Vf = torch.hstack((tmp1, tmp2))

        if hasattr(self, 'Sin'):
            # Input layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(
                self.eps_in - self.Kx_norm_dn ** 2 - self.Ky_norm_dn ** 2)
            Kz_norm_dn = torch.where(torch.imag(
                Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn)
            tmp1 = torch.vstack((torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                                 torch.diag(Kz_norm_dn + self.Kx_norm_dn ** 2 / Kz_norm_dn)))
            tmp2 = torch.vstack((torch.diag(-Kz_norm_dn - self.Ky_norm_dn ** 2 / Kz_norm_dn),
                                 torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn)))
            self.Vi = torch.hstack((tmp1, tmp2))

            Vtmp1 = torch.linalg.inv(self.Vf + self.Vi)
            Vtmp2 = self.Vf - self.Vi

            # Input layer S-matrix
            self.Sin.append(2 * torch.matmul(Vtmp1, self.Vi))  # Tf S11
            self.Sin.append(-torch.matmul(Vtmp1, Vtmp2))  # Rf S21
            self.Sin.append(torch.matmul(Vtmp1, Vtmp2))  # Rb S12
            self.Sin.append(2 * torch.matmul(Vtmp1, self.Vf))  # Tb S22

        if hasattr(self, 'Sout'):
            # Output layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(
                self.eps_out - self.Kx_norm_dn ** 2 - self.Ky_norm_dn ** 2)
            Kz_norm_dn = torch.where(torch.imag(
                Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn)
            tmp1 = torch.vstack((torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                                 torch.diag(Kz_norm_dn + self.Kx_norm_dn ** 2 / Kz_norm_dn)))
            tmp2 = torch.vstack((torch.diag(-Kz_norm_dn - self.Ky_norm_dn ** 2 / Kz_norm_dn),
                                 torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn)))
            self.Vo = torch.hstack((tmp1, tmp2))

            Vtmp1 = torch.linalg.inv(self.Vf + self.Vo)
            Vtmp2 = self.Vf - self.Vo

            # Output layer S-matrix
            self.Sout.append(2 * torch.matmul(Vtmp1, self.Vf))  # Tf S11
            self.Sout.append(torch.matmul(Vtmp1, Vtmp2))  # Rf S21
            self.Sout.append(-torch.matmul(Vtmp1, Vtmp2))  # Rb S12
            self.Sout.append(2 * torch.matmul(Vtmp1, self.Vo))  # Tb S22

    def _material_conv(self, material):
        material = material.to(self._dtype)
        material_N = material.shape[0] * material.shape[1]

        # Matching indices
        order_x_grid, order_y_grid = torch.meshgrid(
            self.order_x, self.order_y, indexing='ij')
        ox = order_x_grid.to(torch.int64).reshape([-1])
        oy = order_y_grid.to(torch.int64).reshape([-1])

        ind = torch.arange(len(self.order_x) *
                           len(self.order_y), device=self._device)
        indx, indy = torch.meshgrid(
            ind.to(torch.int64), ind.to(torch.int64), indexing='ij')
        material_fft = torch.fft.fft2(material) / material_N
        material_convmat = material_fft[ox[indx] -
                                        ox[indy], oy[indx] - oy[indy]]
        return material_convmat

    def _eigen_decomposition_homogenous(self, eps):
        kz_norm = torch.sqrt(eps - self.Kx_norm_dn ** 2 - self.Ky_norm_dn ** 2)
        kz_norm = torch.where(torch.imag(kz_norm) < 0, torch.conj(
            kz_norm), kz_norm)  # Normalized kz for positive mode
        KxKyKzi = self.Kx_norm * self.Ky_norm * torch.diag(1. / kz_norm)
        KzplusKxKxKzi = torch.diag(
            kz_norm) + self.Kx_norm ** 2 * torch.diag(1. / kz_norm)
        KzplusKyKyKzi = torch.diag(
            kz_norm) + self.Ky_norm ** 2 * torch.diag(1. / kz_norm)
        V = torch.hstack((torch.vstack((-KxKyKzi, KzplusKxKxKzi)),
                         torch.vstack((-KzplusKyKyKzi, KxKyKzi))))

        self.kz_norm.append(torch.cat((kz_norm, kz_norm)))
        self.E_eigvec.append(
            torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device))
        self.H_eigvec.append(V)

    def _eigen_decomposition(self):
        E = self.eps_conv[-1]
        Ei = torch.linalg.inv(E)
        KxEiKy = torch.matmul(torch.matmul(self.Kx_norm, Ei), self.Ky_norm)
        KyEiKx = torch.matmul(torch.matmul(self.Ky_norm, Ei), self.Kx_norm)
        KxEiKx = torch.matmul(torch.matmul(self.Kx_norm, Ei), self.Kx_norm)
        KyEiKy = torch.matmul(torch.matmul(self.Ky_norm, Ei), self.Ky_norm)

        KxKx = self.Kx_norm ** 2
        KxKy = self.Kx_norm * self.Ky_norm
        KyKy = self.Ky_norm ** 2
        I = torch.eye(self.order_N, dtype=self._dtype, device=self._device)
        P = torch.hstack((torch.vstack((KxEiKy, KyEiKy - I)),
                         torch.vstack((I - KxEiKx, -KyEiKx))))
        Q = torch.hstack((torch.vstack((-KxKy, E - KyKy)),
                         torch.vstack((KxKx - E, KxKy))))
        PQ = torch.matmul(P, Q)
        # Eigen-decomposition
        kz_norm, E_eigvec = torch.linalg.eig(PQ)
        kz_norm = torch.sqrt(kz_norm)
        kz_norm = torch.where(torch.imag(kz_norm) < 0, -kz_norm, kz_norm)
        self.kz_norm.append(kz_norm)
        self.E_eigvec.append(E_eigvec)
        self.H_eigvec.append(torch.matmul(Q, torch.matmul(
            E_eigvec, torch.linalg.inv(torch.diag(kz_norm)))))

    def _solve_layer_smatrix(self):
        phase = torch.diag(torch.exp(1.j * self.omega *
                           self.kz_norm[-1] * self.thickness[-1]))
        W = self.E_eigvec[-1]
        V0iV = torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1])

        # Mode coupling coefficients
        A = W + V0iV
        B = torch.matmul(W - V0iV, phase)
        BAi = torch.matmul(B, torch.linalg.inv(A))
        C1 = 2 * torch.linalg.inv(A - torch.matmul(BAi, B))
        C2 = -torch.matmul(C1, BAi)
        S11 = torch.matmul(W, torch.matmul(phase, C1) + C2)
        S12 = torch.matmul(W, torch.matmul(phase, C2) + C1) - torch.eye(2 * self.order_N, dtype=self._dtype,
                                                                        device=self._device)
        self.layer_S11.append(S11)
        self.layer_S12.append(S12)
        self.layer_S21.append(S12)
        self.layer_S22.append(S11)

    def _RS_prod(self, Sm, Sn):
        # S11 = S[0] / S21 = S[1] / S12 = S[2] / S22 = S[3]
        # Cf = C[0] / Cb = C[1]

        tmp1 = torch.linalg.inv(
            torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device) - torch.matmul(Sm[2], Sn[1]))
        tmp2 = torch.linalg.inv(
            torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device) - torch.matmul(Sn[1], Sm[2]))

        # Layer S-matrix
        S11 = torch.matmul(Sn[0], torch.matmul(tmp1, Sm[0]))
        S21 = Sm[1] + torch.matmul(Sm[3],
                                   torch.matmul(tmp2, torch.matmul(Sn[1], Sm[0])))
        S12 = Sn[2] + torch.matmul(Sn[0],
                                   torch.matmul(tmp1, torch.matmul(Sm[2], Sn[3])))
        S22 = torch.matmul(Sm[3], torch.matmul(tmp2, Sn[3]))

        return S11, S21, S12, S22

    ########################################################################################
    #######################  FREE method: Cos·Sinc from P·Q   ################################
    ########################################################################################
    def _compute_layer_free(self):
        """
        Compute layer S-matrix via the FREE path (Cos·Sinc direct from P·Q).

        Algorithm (Order18, matching Bader-Blanes-Casas 2019):
          1. A² = 2⁻²ᵐ·(k₀d)²·(P·Q),  m = ⌈½ log₂ max(‖P·Q‖₁, 1)⌉
          2. A⁶ = (A²)³
          3. B⁽ⁱ⁾ = a₀ᵢ·I − a₂ᵢ·A² − a₄ᵢ·A⁶,   B̄⁽ⁱ⁾ = a₁ᵢ·I − a₃ᵢ·A²  (i=0..4)
          4. A⁹ = B⁽⁰⁾·B⁽⁴⁾ − A²·(B̄⁽⁰⁾·B̄⁽⁴⁾) + B⁽³⁾
             Ā⁹ = B̄⁽⁰⁾·B⁽⁴⁾ + B⁽⁰⁾·B̄⁽⁴⁾ + B̄⁽³⁾
          5. Cos  = B⁽¹⁾ + (B⁽²⁾ + A⁹)·A⁹ − A²·(B̄⁽²⁾ + Ā⁹)·Ā⁹
             Sinc = B̄⁽¹⁾ + (B̄⁽²⁾ + Ā⁹)·A⁹ + (B⁽²⁾ + A⁹)·Ā⁹
          6. m rounds: Sinc ← Sinc·Cos,  Cos ← 2·Cos² − I
          7. E = Cos + i·k₀d·Vf·Q·Sinc
             F = −(i·k₀d·Sinc·P + Vf·Q·Cos·Q⁻¹)·Vf
          8. T₂₂ = ½(E+F), T₂₁ = ½(E−F),  S₁₁ = T₂₂⁻¹, S₁₂ = −S₁₁·T₂₁

        Memory peak: ~15 × n²  (n = 2·order_N), compared to ~29 × n² for GEMM Order18.
        """
        dtype = self._dtype
        device = self._device
        n = 2 * self.order_N

        k0d = self.omega * self.thickness[-1]
        k0d_val = k0d
        E_eps = self.eps_conv[-1]
        KxKx = self.Kx_norm ** 2
        KxKy = self.Kx_norm * self.Ky_norm
        KyKy = self.Ky_norm ** 2

        try:
            EiKx = torch.linalg.solve(E_eps, self.Kx_norm)
            EiKy = torch.linalg.solve(E_eps, self.Ky_norm)
            KxEiKy = self.Kx_norm @ EiKy
            KyEiKx = self.Ky_norm @ EiKx
            KxEiKx = self.Kx_norm @ EiKx
            KyEiKy = self.Ky_norm @ EiKy
        except RuntimeError:
            KxEiKy = KxKy
            KyEiKx = KxKy
            KxEiKx = KxKx
            KyEiKy = KyKy

        I_n = torch.eye(self.order_N, dtype=dtype, device=device)
        P = torch.hstack((torch.vstack((KxEiKy, KyEiKy - I_n)),
                         torch.vstack((I_n - KxEiKx, -KyEiKx))))  # n×n
        Q = torch.hstack((torch.vstack((-KxKy, E_eps - KyKy)),
                         torch.vstack((KxKx - E_eps, KxKy))))     # n×n

        # ── 1. A² = 2⁻²ᵐ·(k₀d)²·(P·Q) ─────────────────────────────────────
        PQ = P @ Q
        PQnorm = torch.max(torch.sum(torch.abs(PQ), dim=0))
        m = int(torch.ceil(0.5 * torch.log2(torch.clamp(PQnorm, min=1.0))).item())
        if m < 0:
            m = 0
        scale_sq = 2.0 ** (-2 * m)
        A2 = (scale_sq * k0d_val ** 2) * PQ
        del PQ

        # ── 2. A⁶ = (A²)³ ──────────────────────────────────────────────────
        A2_sq = A2 @ A2
        A6 = A2_sq @ A2
        del A2_sq

        # ── 3. B⁽ⁱ⁾ and B̄⁽ⁱ⁾ (i = 0..4) ───────────────────────────────────
        I_mat = torch.eye(n, dtype=dtype, device=device)
        ec = self.exp_constant.to(device=device, dtype=torch.float64)
        B, Bbar = [], []
        for i in range(5):
            a0, a1, a2, a3, a4 = ec[i, 0], ec[i, 1], ec[i, 2], ec[i, 3], ec[i, 4]
            Bi = a0 * I_mat - a2 * A2 - a4 * A6
            Bbari = a1 * I_mat - a3 * A2
            B.append(Bi)
            Bbar.append(Bbari)
        del A6

        # ── 4. A⁹ and Ā⁹ ──────────────────────────────────────────────────
        # A9 = B⁽⁰⁾·B⁽⁴⁾ + B⁽³⁾ − A²·(B̄⁽⁰⁾·B̄⁽⁴⁾)
        tmp = Bbar[0] @ Bbar[4]
        A9 = B[0] @ B[4] - A2 @ tmp + B[3]
        # Ā⁹ = B̄⁽⁰⁾·B⁽⁴⁾ + B⁽⁰⁾·B̄⁽⁴⁾ + B̄⁽³⁾
        Abar9 = Bbar[0] @ B[4] + B[0] @ Bbar[4] + Bbar[3]
        del tmp

        # ── 5. Cos and Sinc ────────────────────────────────────────────────
        # Cos = B⁽¹⁾ + (B⁽²⁾ + A⁹)·A⁹ − A²·(B̄⁽²⁾ + Ā⁹)·Ā⁹
        T1 = B[2] + A9
        T2 = Bbar[2] + Abar9
        Cos = B[1] + T1 @ A9 - A2 @ (T2 @ Abar9)
        # Sinc = B̄⁽¹⁾ + (B̄⁽²⁾ + Ā⁹)·A⁹ + (B⁽²⁾ + A⁹)·Ā⁹
        Sinc = Bbar[1] + T2 @ A9 + T1 @ Abar9
        del A9, Abar9, T1, T2, B, Bbar

        # ── 6. m rounds of double-angle ────────────────────────────────────
        two = torch.tensor(2.0, dtype=dtype, device=device)
        for _ in range(m):
            # Sinc ← Sinc·Cos
            Sinc_new = Sinc @ Cos
            # Cos ← 2·Cos² − I
            Cos_new = two * (Cos @ Cos) - I_mat
            Cos, Sinc = Cos_new, Sinc_new

        # ── 7. E and F ─────────────────────────────────────────────────────
        # E = Cos + i·k₀d·Vf·Q·Sinc
        VfQ = self.Vf @ Q
        E_mat = Cos + (1j * k0d_val) * (VfQ @ Sinc)

        # F = −(i·k₀d·Sinc·P + VfQ·Cos·Q⁻¹)·Vf
        part1 = (1j * k0d_val) * (Sinc @ P)
        # CosQ = Cos·Q⁻¹ (solve Q^T·X = Cos^T, then transpose)
        CosT = Cos.T.contiguous()
        X = torch.linalg.solve(Q.T.contiguous(), CosT)
        CosQ = X.T.contiguous()
        tmpF = part1 + VfQ @ CosQ
        F_mat = -(tmpF @ self.Vf)
        del Cos, Sinc, P, Q, VfQ, CosT, X, CosQ, part1, tmpF, E_eps

        # ── 8. Convert to layer S-matrix ───────────────────────────────────
        half = torch.tensor(0.5, dtype=dtype, device=device)
        T22 = half * (E_mat + F_mat)
        T21 = half * (E_mat - F_mat)
        S11 = torch.linalg.inv(T22)
        S12 = -S11 @ T21
        del E_mat, F_mat, T22, T21

        self.layer_S11.append(S11)
        self.layer_S12.append(S12)
        self.layer_S21.append(S12)
        self.layer_S22.append(S11)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ########################################################################################
    #######################  GEMM method: exp(A) via block Taylor ##########################
    ########################################################################################
    def _compute_layer_exp(self):
        thickness_thres = 2.0 * \
            min(self.L[0] / self.order[0], self.L[1] / self.order[1])
        if (self.thickness[-1] > thickness_thres):
            n_repeatedSquaring = int(torch.ceil(torch.log2(
                torch.tensor(self.thickness[-1] / thickness_thres))))
            d_block = self.thickness[-1] / 2 ** (n_repeatedSquaring)
        else:
            n_repeatedSquaring = 0
            d_block = self.thickness[-1]

        k0d = self.omega * d_block
        E = self.eps_conv[-1]
        KxKx = self.Kx_norm ** 2
        KxKy = self.Kx_norm * self.Ky_norm
        KyKy = self.Ky_norm ** 2

        try:
            EiKx = torch.linalg.solve(E, self.Kx_norm)
            EiKy = torch.linalg.solve(E, self.Ky_norm)
            KxEiKy = self.Kx_norm @ EiKy
            KyEiKx = self.Ky_norm @ EiKx
            KxEiKx = self.Kx_norm @ EiKx
            KyEiKy = self.Ky_norm @ EiKy
        except RuntimeError:
            KxEiKy = KxKy
            KyEiKx = KxKy
            KxEiKx = KxKx
            KyEiKy = KyKy

        I = torch.eye(self.order_N, dtype=self._dtype, device=self._device)
        I2 = torch.eye(2 * self.order_N, dtype=self._dtype,
                       device=self._device)
        P = torch.hstack((torch.vstack((KxEiKy, KyEiKy - I)),
                         torch.vstack((I - KxEiKx, -KyEiKx))))
        Q = torch.hstack((torch.vstack((-KxKy, E - KyKy)),
                         torch.vstack((KxKx - E, KxKy))))
        Pnorm = torch.max(torch.sum(torch.abs(P), dim=0))
        Qnorm = torch.max(torch.sum(torch.abs(Q), dim=0))
        Rnorm = torch.max(Pnorm, Qnorm)
        m = int(torch.ceil(torch.log2(Rnorm * k0d)))
        if m < 0:
            m = 0

        A_12 = -2 ** (-m) * k0d * P
        A_21 = 2 ** (-m) * k0d * Q
        A2_11 = A_12 @ A_21
        A2_22 = A_21 @ A_12
        A3_12 = A_12 @ A2_22
        A3_21 = A_21 @ A2_11
        A6_11 = A3_12 @ A3_21
        A6_22 = A3_21 @ A3_12

        del P, Q

        B_11 = [[], [], [], [], []]
        B_12 = [[], [], [], [], []]
        B_21 = [[], [], [], [], []]
        B_22 = [[], [], [], [], []]
        for i in range(5):
            B_11[i] = self.exp_constant[i, 0] * I2 + self.exp_constant[i,
                                                                       2] * A2_11 + self.exp_constant[i, 4] * A6_11
            B_12[i] = self.exp_constant[i, 1] * \
                A_12 + self.exp_constant[i, 3] * A3_12
            B_21[i] = self.exp_constant[i, 1] * \
                A_21 + self.exp_constant[i, 3] * A3_21
            B_22[i] = self.exp_constant[i, 0] * I2 + self.exp_constant[i,
                                                                       2] * A2_22 + self.exp_constant[i, 4] * A6_22

        del A_12, A_21, A2_11, A2_22, A3_12, A3_21, A6_11, A6_22

        A9_11 = B_11[0] @ B_11[4] + B_12[0] @ B_21[4] + B_11[3]
        A9_12 = B_11[0] @ B_12[4] + B_12[0] @ B_22[4] + B_12[3]
        A9_21 = B_21[0] @ B_11[4] + B_22[0] @ B_21[4] + B_21[3]
        A9_22 = B_21[0] @ B_12[4] + B_22[0] @ B_22[4] + B_22[3]

        expA_11 = B_11[1] + (B_11[2] + A9_11) @ A9_11 + \
            (B_12[2] + A9_12) @ A9_21
        expA_12 = B_12[1] + (B_11[2] + A9_11) @ A9_12 + \
            (B_12[2] + A9_12) @ A9_22
        expA_21 = B_21[1] + (B_21[2] + A9_21) @ A9_11 + \
            (B_22[2] + A9_22) @ A9_21
        expA_22 = B_22[1] + (B_21[2] + A9_21) @ A9_12 + \
            (B_22[2] + A9_22) @ A9_22
        del A9_11, A9_12, A9_21, A9_22, B_11, B_12, B_21, B_22

        for k in range(m):
            tmp11 = expA_11 @ expA_11 + expA_12 @ expA_21
            tmp12 = expA_11 @ expA_12 + expA_12 @ expA_22
            tmp21 = expA_21 @ expA_11 + expA_22 @ expA_21
            tmp22 = expA_21 @ expA_12 + expA_22 @ expA_22
            expA_11 = tmp11
            expA_12 = tmp12
            expA_21 = tmp21
            expA_22 = tmp22

        # Convert to layer S-matrix
        E = expA_11 + 1j * self.Vf @ expA_21
        F = (1j * expA_12 - self.Vf @ expA_22) @ self.Vf
        T22 = 0.5 * (E + F)
        T21 = 0.5 * (E - F)
        S11 = torch.linalg.inv(T22)
        S12 = -S11 @ T21

        del expA_11, expA_12, expA_21, expA_22, E, F, T22, T21

        for i in range(n_repeatedSquaring):
            R = I2 - S12 @ S12
            tmp = S12 + S11 @ torch.linalg.solve(R, S12 @ S11)
            S11 = S11 @ torch.linalg.solve(R, S11)
            S12 = tmp

        self.layer_S11.append(S11)
        self.layer_S12.append(S12)
        self.layer_S21.append(S12)
        self.layer_S22.append(S11)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # clear cache

    def _compute_layer_exp_homogenous(self, eps, thickness):
        V, kz_norm = self.get_V(eps)
        kz_norm = torch.cat((kz_norm, kz_norm))
        phase = torch.diag(torch.exp(1.j * self.omega * kz_norm * thickness))
        I = torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
        V0iV = -self.Vf @ V
        A = I + V0iV
        B = (I - V0iV) @ phase
        BAi = B @ torch.linalg.inv(A)
        C1 = 2 * torch.linalg.inv(A - (BAi @ B))
        C2 = -C1 @ BAi
        S11 = phase @ C1 + C2
        S12 = phase @ C2 + C1 - I
        self.layer_S11.append(S11)
        self.layer_S12.append(S12)
        self.layer_S21.append(S12)
        self.layer_S22.append(S11)

    def get_V(self, eps):
        kz_norm = torch.sqrt(torch.diag(torch.full((self.order_N, self.order_N), eps[0][0], dtype=self._dtype, device=self._device))
                             - self.Kx_norm_dn ** 2 - self.Ky_norm_dn ** 2)
        kz_norm = torch.where(torch.imag(kz_norm) < 0,
                              torch.conj(kz_norm), kz_norm)
        KxKyKzi = self.Kx_norm * self.Ky_norm * torch.diag(1. / kz_norm)
        KzplusKxKxKzi = torch.diag(
            kz_norm) + self.Kx_norm ** 2 * torch.diag(1. / kz_norm)
        KzplusKyKyKzi = torch.diag(
            kz_norm) + self.Ky_norm ** 2 * torch.diag(1. / kz_norm)
        V = torch.hstack((torch.vstack((-KxKyKzi, KzplusKxKxKzi)),
                         torch.vstack((-KzplusKyKyKzi, KxKyKzi))))
        return V, kz_norm

class Layer:
    """
    Layer class for TORCWA.

    Args:
        geometry (Optional[torch.Tensor]): Geometry of the layer
        thickness (float): Thickness of the layer
        eps (float): Material Epsilon
    """

    def __init__(self, geometry: Optional[torch.Tensor], thickness: float, eps: float):
        # ? Geometry
        self.geometry = geometry
        # ? Thickness
        self.thickness = thickness
        # ? Material Eps
        self.eps = eps
