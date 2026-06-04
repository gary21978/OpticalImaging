#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:06:50 2025

@author: siok003
"""
import math
from Source import Source
from Scatter import Scatter
from Projection import Projection
from CalculateAerialImage import CalculateAbbeImage
import matplotlib.pyplot as plt
import torch
import numpy as np
from rcwa import rcwa, Layer
import time
from PIL import Image
import extra_func
import os
import tifffile

class ImagingModel:
    def __init__(self):
        self.Scatter = Scatter()
        self.Projector = Projection()

    def CalculateAerialImage(self):
        Scatter = self.Scatter
        Projector = self.Projector
        ali = CalculateAbbeImage(Scatter, Projector)
        return ali


class Simulok:
    def __init__(self, args=None):
       # 处理参数并映射到对象属性
       self.device = str(args.device).lower()
       self.save_name = ""
       self.png_path = args.pngpath

       # 定义问题，为了直观理解，在设置参数中，【ox，oy】分别代表x方向和y方向的阶数；
       self.order = [args.order[1], args.order[0]]
       self.Wavelength = args.lamda  # nm

       # Lx/Ly：若 period 提供一个值，两个方向同值；若提供两值，按顺序赋值
       # 目前不支持非正方形的仿真
       period_vals = list(args.period)
       if len(period_vals) == 1:
           self.Lx = period_vals[0]
           self.Ly = period_vals[0]
       else:
           self.Lx = period_vals[0]
           self.Ly = period_vals[1]

       # 分辨率
       self.resolution_x = int(args.resolution[0])
       self.resolution_y = int(args.resolution[1])

       # 其他参数
       # self.source.PolarizationType = args.polarization
       self.Geometry = torch.tensor([])
       self.Intensity = torch.tensor([])
       self.enable_fast_exp = True
       self.pixelsize = int(args.pixelsize)
       self.focusrange_far = torch.tensor([0])
       self.focusrange_near = [0]
       self.Aberration_Zernike = torch.zeros(37)
        # self.Projector.Aberration_Zernike[4] = 0.5

       # 光源初始化，确保参数存在
       self.source = Source()
       self.source.Shape = args.shape
       self.source.PolarizationType = args.polarization
       self.source.SigmaOut = getattr(args, 'sigmaOut', 0.9)
       self.source.SigmaIn = getattr(args, 'sigmaIn', 0.15)
       self.source.OpenAngle = getattr(args, 'openAngle', 0.0)
       self.source.RotateAngle = getattr(args, 'rotateAngle', 0.0)
       self.source.RectX = getattr(args, 'RectX', 0.0)
       self.source.RectY = getattr(args, 'RectY', 0.0)       
       self.source.PntNum = (2 * self.order[0] + 1) * (2 * self.order[1] + 1)
       self.invalid = 0
       
       # pro初始化，确保参数存在
       self.projection = Projection()
       self.projection.Shape = args.imaging_pupil_shape
       self.projection.PolarizationType = args.imaging_pupil_polarization
       self.projection.SigmaOut = getattr(args, 'imaging_pupil_sigmaOut', 0.9)
       self.projection.SigmaIn = getattr(args, 'imaging_pupil_sigmaIn', 0.15)
       self.projection.OpenAngle = getattr(args, 'imaging_pupil_openAngle', 0.0)
       self.projection.RotateAngle = getattr(args, 'imaging_pupil_rotateAngle', 0.0)
       self.projection.RectX = getattr(args, 'imaging_pupil_RectX', 0.0)
       self.projection.RectY = getattr(args, 'imaging_pupil_RectY', 0.0)
       
       self.projection.Magnification = getattr(args, 'magnification', 1.0)


       # RCWA计算方法
       self.rcwa_method = getattr(args, 'rcwa_method', 'free')

       # 材料/折射率
       self.Si = extra_func.get_eps('Si', self.Wavelength)
       self.SiO2 = extra_func.get_eps('SiO2', self.Wavelength)
       self.Si3N4 = extra_func.get_eps('Si3N4', self.Wavelength)
       self.TiN = extra_func.get_eps('TiN', self.Wavelength)
       
    def LoadGeometry(self):
        """
        物理结构堆叠及光传播方向（简化示意，仅展示关键两层）：
        
        若 layers = [Layer(geometry=self.Geometry,thickness=100.,eps=self.SiO2),
                    Layer(geometry=torch.oneslike(self.Geometry),thickness=300.,eps=self.Si3N4)]，则：
        
          Substrate: Crystalline Si     |  ← 由 add_output_layer(eps=Si) 定义
        ================================|  
        |                               |  
        | Si3N4_layer                   |  ← 如：300 nm Si₃N₄，geometry = 1（全覆盖）
        |                               |
        |-------------------------------|
        | SiO2_layer                    |  ← 如：100 nm SiO₂，geometry = mask（掩模图形）
        |-------------------------------|
        | Air (input)                   |  ← 入射介质（空气，由 add_input_layer() 定义）
          -------------------------------
        
        当执行 forward 时（光从底部空气向上入射 ↑）：
        当执行 backward 时（光从顶部 Si 向下入射 ↓）：
        
        此说明有助于理解多层堆叠顺序与光传播方向的关系。
        """
        image = np.array(Image.open(self.png_path).convert('L'), dtype=np.uint8)
        self.raster_img = image
        image = np.where(image != 0, 1, 0).astype(np.uint8)
        self.Geometry = torch.from_numpy(image).to(device=self.device)
        self.substrate_eps = self.Si

        # layers
        self.layers = [
            Layer(
                # TIN，氮化钛
                geometry=self.Geometry,
                thickness=300.,
                eps=self.TiN
            ),
        ]

    def Scattering(self):
        start_time_scattering = time.time()
        sim_dtype = torch.complex128

        inc_ang = 0. * (torch.pi / 180)  # 0-180
        azi_ang = 0. * (torch.pi / 180)  # 0-360

        self.kxy0 = np.sin(inc_ang)
        self.kx0 = self.kxy0 * np.cos(azi_ang)
        self.ky0 = self.kxy0 * np.sin(azi_ang)

        self.xs = (self.Lx / self.resolution_x) * (torch.arange(self.resolution_x + 1 , dtype=torch.float32, device=self.device))
        self.ys = (self.Ly / self.resolution_y) * (torch.arange(self.resolution_y + 1, dtype=torch.float32, device=self.device))

        self.sim = rcwa(freq=1 / self.Wavelength, order=self.order, L=[self.Lx, self.Ly], dtype=sim_dtype, device=self.device, method=self.rcwa_method)
        self.sim.fast_exp = self.enable_fast_exp
        self.sim.add_input_layer()
        self.sim.add_output_layer(eps=self.substrate_eps)
        self.sim.set_incident_angle(inc_ang=inc_ang, azi_ang=azi_ang, angle_layer='input')

        for layer in self.layers:
            # 计算xy平面的eps分布，式子中的第二项是为了确保空气的位置eps=1
            layer.eps = layer.geometry * layer.eps + (1. - layer.geometry)
        for index, layer in enumerate(self.layers):
            self.sim.add_layer(thickness=layer.thickness, eps=layer.eps)
            print(f"[Near Field] adding layer{index}...")
        self.sim.solve_global_smatrix()
        
        end_time_scattering = time.time()
        elapsed_time_scattering = end_time_scattering - start_time_scattering
        print(f"S 矩阵计算耗时: {elapsed_time_scattering:.4f} 秒")

    def get_M_matrix(self):
        """
        构造批量矩阵 M 及其分解形式 R_0、R_1，满足（忽略偏振调制时）：
            M ≈ R_1^T @ R_0，
        其中 M: (N, 3, 2)，R_0: (N, 2, 2)，R_1: (N, 2, 3)。
    
        特殊情形处理：
          - 隐失波（|kxy| ≥ 1）：M、R_0、R_1 替换为单位形式以避免数值不稳定；
          - 垂直入射（kxy = 0）：R_0 和 R_1 退化为标准基，避免除零。
    
        返回:
            M:   (N, 3, 2)
            R_0: (N, 2, 2)
            R_1: (N, 2, 3)
        """
        Kx_norm_dn, Ky_norm_dn, Kz_norm_dn = self.sim.get_kx_ky_kz_nrom()
        Kx, Ky, kz = Kx_norm_dn, Ky_norm_dn, Kz_norm_dn
    
        # 计算横向波矢模长
        kxy2 = Kx**2 + Ky**2
        kxy = torch.sqrt(kxy2)
    
        # 构造 M 矩阵（直接形式）
        m11 = 1 - Kx ** 2 / (1 + kz)
        m12 = -Kx * Ky / (1 + kz)
        m21 = -Kx * Ky / (1 + kz)
        m22 = 1 - Ky ** 2 / (1 + kz)
        m31 = -Kx
        m32 = -Ky
    
        M = torch.stack([
            torch.stack([m11, m12], dim=-1),
            torch.stack([m21, m22], dim=-1),
            torch.stack([m31, m32], dim=-1),
        ], dim=-2)  # (N, 3, 2)
    
        # 构造 R_0（横向偏振基，2×2）
        r0_11 = -Kx / kxy
        r0_12 = -Ky / kxy
        r0_21 = Ky / kxy
        r0_22 = -Kx / kxy
        R_0 = torch.stack([
            torch.stack([r0_11, r0_12], dim=-1),
            torch.stack([r0_21, r0_22], dim=-1),
        ], dim=-2)  # (N, 2, 2)
    
        # 构造 R_1（三维偏振基，2×3）
        r1_11 = -Kx * kz / kxy
        r1_12 = -Ky * kz / kxy
        r1_13 = kxy
        r1_21 = Ky / kxy
        r1_22 = -Kx / kxy
        R_1 = torch.stack([
            torch.stack([r1_11, r1_12, r1_13], dim=-1),
            torch.stack([r1_21, r1_22, torch.zeros_like(r0_11)], dim=-1),
        ], dim=-2)  # (N, 2, 3)
    
        # 处理隐失波：|kxy| ≥ 1
        idx = (torch.abs(kxy2) >= 1)
        if idx.any():
            M[idx] = torch.tensor([[1,0],[0,1],[0,0]], dtype=kxy2.dtype, device=kxy2.device)
            R_0[idx] = torch.tensor([[1,0],[0,1]], dtype=kxy2.dtype, device=kxy2.device)
            R_1[idx] = torch.tensor([[1,0,0],[0,1,0]], dtype=kxy2.dtype, device=kxy2.device)
    
        # 处理垂直入射：kxy = 0
        idx = (torch.abs(kxy2) == 0)
        if idx.any():
            R_0[idx] = torch.tensor([[1,0],[0,1]], dtype=kxy2.dtype, device=kxy2.device)
            R_1[idx] = torch.tensor([[1,0,0],[0,1,0]], dtype=kxy2.dtype, device=kxy2.device)
    
        return M.real, R_0.real, R_1.real
    
    def set_jones_vector(self, M, R0, R1, kz, r=0.0, sigma=0.1):
        """
        Compute 3D Jones vector from 2D input polarization using rotation matrices.
    
        Args:
            M: (unused, kept for interface compatibility)
            R0: (2, 2) tensor — rotation from global (x,y) to local (s,p) for incidence
            R1: (2, 3) tensor — mapping from local (s,p) to global (x,y,z) for output
            kz: scalar tensor — z-component of wavevector (>0)
            r: float or tensor — radial distance for apodization
            sigma: float — apodization width
    
        Sets:
            self.jones_vector: (3,) complex tensor [Ex, Ey, Ez]
        """
        # --- Device and dtype setup ---
        dtype_real = torch.float64
        dtype_complex = torch.complex128
        device = self.device
    
        # Convert scalars to tensors on device
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=dtype_real, device=device)
        else:
            r = r.to(device, dtype=dtype_real)
    
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=dtype_real, device=device)
        else:
            sigma = sigma.to(device, dtype=dtype_real)
    
        # --- Manual Apodization ---
        A = torch.exp(-r**2 / (2.0 * sigma**2))  # scalar
    
        # --- Polarization modulation matrix F (2x2 identity) ---
        F = torch.eye(2, dtype=dtype_complex, device=device)
    
        # --- Build total transformation matrix: M_total = R1.T @ F @ R0 ---
        # R0: (2,2), F: (2,2), R1: (2,3) → R1.T: (3,2)
        R0_c = R0.to(dtype_complex)
        R1_c = R1.to(dtype_complex)
        M_total = R1_c.T @ (F @ R0_c)  # (3,2) @ (2,2) = (3,2)
    
        # --- Apply apodization and kz scaling ---
        if kz <= 0:
            raise ValueError("kz must be positive.")
        scale_factor = A * torch.sqrt(1.0 / kz.to(dtype_real))  # scalar
        M_A = scale_factor * M_total  # (3,2)
    
        # --- Input polarization vector (2,) ---
        pol_type = self.source.PolarizationType
        if pol_type == 'x_pol':
            e_in = torch.tensor([1.0, 0.0], dtype=dtype_complex, device=device)
        elif pol_type == 'y_pol':
            e_in = torch.tensor([0.0, 1.0], dtype=dtype_complex, device=device)
        elif pol_type == 'line_pol':  # 45°
            v = 1.0 / math.sqrt(2)
            e_in = torch.tensor([v, v], dtype=dtype_complex, device=device)
        elif pol_type == 'c_pol':  # right-hand circular
            v = 1.0 / math.sqrt(2)
            e_in = torch.tensor([v, 1j * v], dtype=dtype_complex, device=device)
        elif isinstance(pol_type, (int, float, np.integer, np.floating)):
            angle = torch.tensor(math.radians(float(pol_type)), dtype=dtype_real, device=device)
            e_in = torch.stack([torch.cos(angle), torch.sin(angle)]).to(dtype_complex)
        else:
            raise ValueError(f'Unsupported polarization type: {pol_type}')
    
        # --- Compute 3D Jones vector: (3,2) @ (2,) -> (3,) ---
        jones_3d = M_A @ e_in  # shape (3,)
    
        # Store result
        self.jones_vector = jones_3d  # [Ex, Ey, Ez]
    
        return True
          
    def scatter_to_imaging(self):
        """
        遍历所有光源位置，对每个有效照明位置计算 Floquet 模式(即角谱)，
        并将其从物体表面传播到成像光瞳平面。
        """
        start_time_scattering = time.time()
        Kx_norm_dn, Ky_norm_dn, Kz_norm_dn = self.sim.get_kx_ky_kz_nrom()
        M_matrix, R0, R1 = self.get_M_matrix()
        
        self.scatter_result = {}
        self.scatter_result['z_posi'] = {}
        for z_posi in self.focusrange_near:
            self.floquet = []
            self.incident_k = []
            self.invalid = 0
            for idx in range(Kx_norm_dn.numel()):
                kx = Kx_norm_dn[idx]
                ky = Ky_norm_dn[idx]
                kz = Kz_norm_dn[idx]
                if self.source.Cal_valid(kx, ky):
                    oder_y = (idx % (2 * self.order[0] + 1)) - self.order[0]
                    oder_x = (idx // (2 * self.order[0] + 1)) - self.order[1]
                    # self.get_jones_vector(M_matrix[idx], kz, r = np.asrt((kx**2 + ky**2).cpu().numpy()))
                    self.set_jones_vector(M_matrix[idx], R0[idx], R1[idx], kz)
                    # 设置入射平面波和其偏振
                    self.sim.source_planewave(amplitude=self.jones_vector[:2], orders=[oder_y, oder_x], direction='forward')
                    # print(self.jones_vector[:2], oder_x, oder_y)
                    # 获取角谱
                    Ex_mn, Ey_mn, Ez_mn = self.sim.Floquet_mode(z_posi=z_posi)
                    
                    # object plane to pupil plane
                    E_xyz_object = torch.stack([Ex_mn, Ey_mn, Ez_mn], -1).transpose(-2, -1)
                    E_xy_pupil = M_matrix.transpose(-2, -1).to(torch.complex128) @ E_xyz_object
                    
                    self.floquet.append(torch.stack((E_xy_pupil[:, 0].squeeze(), E_xy_pupil[:, 1].squeeze()), dim=-1))
                    self.incident_k.append(torch.tensor((kx, ky), device=self.device))
                    # Ex, Ey, Ez = self.sim.field_xy(self.xs, self.ys, z_posi=z_posi)
                    # self.Visualizing_near_filed(kx, ky, Ex, Ey, Ez)            
                    # 获取S矩阵中心点入射平面波的反射电场空域结果，用于可视化
                    if kx.item() == self.kx0 and ky.item() == self.ky0:
                        Ex, Ey, Ez = self.sim.field_xy(self.xs, self.ys, z_posi=z_posi)
                        # print(E_xy_pupil[:, 0], E_xy_pupil[:, 1])
                        # print(self.jones_vector, oder_x, oder_y)
                else:
                    self.invalid += 1
            
            try:
                Ex  
            except NameError:
                Ex, Ey, Ez = self.sim.field_xy(self.xs, self.ys, z_posi=z_posi)
                
            self.scatter_result['z_posi'][z_posi] = {'k_inc': torch.stack(self.incident_k, dim=-1),
                                                     'Floquet_mode': torch.stack(self.floquet, dim=-1),
                                                     'field': [Ex, Ey, Ez]}


        self.scatter_result['Kx_norm_dn'] = Kx_norm_dn
        self.scatter_result['Ky_norm_dn'] = Ky_norm_dn
        end_time_scattering = time.time()
        elapsed_time_scattering = end_time_scattering - start_time_scattering
        print(f"获取Floquet_mode耗时: {elapsed_time_scattering:.4f} 秒，无效光源数：{self.invalid}, 阶数：{self.order}, 物距计算点数：{len(self.focusrange_near)}")
                 
    def Imaging(self):
        start_time_imaging = time.time()
        im = ImagingModel()
        im.Scatter.Period_X = self.Lx
        im.Scatter.Period_Y = self.Ly
        im.Scatter.resolution_x = self.resolution_x
        im.Scatter.resolution_y = self.resolution_y
        im.Scatter.ScatterResult = self.scatter_result
        im.Scatter.Order = self.order
        
        im.Projector = self.projection
        im.Projector.Wavelength = self.Wavelength
        im.Projector.Aberration_Zernike = self.Aberration_Zernike
        im.Projector.IndexImage = 1.0
        im.Projector.FocusRange = im.Projector.Magnification ** 2 * self.focusrange_far
        
        source_all = (2 * self.order[0] + 1) * (2 * self.order[1] + 1)
        # 入射平面波数量和光源面积归一化（假设不同光源形状的面密度强度分布一致）
        self.Intensity = im.CalculateAerialImage().Intensity/(source_all - self.invalid) * self.source.SigmaOut**2
        # self.Intensity = im.CalculateAerialImage().Intensity
        # print(self.invalid)
        # self.Illuminant = im.Source.source_data

        end_time_imaging = time.time()
        elapsed_time_imaging = end_time_imaging - start_time_imaging
        print(f"远场耗时: {elapsed_time_imaging:.4f} 秒")
    
    def Visualizing_near_filed(self,kx, ky, E_x, E_y, E_z):
        struct_name = self.png_path.split('/')[-1][:-4]
        save_dir = f'./image_result/{self.save_name}/{struct_name}/near_field'
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(torch.abs(E_x.squeeze()).cpu(), cmap='jet',
                   extent=(0, 0.001 * self.Lx, 0, 0.001 * self.Ly))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('|E_x|')
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(torch.abs(E_y.squeeze()).cpu(), cmap='jet',
                   extent=(0, 0.001 * self.Lx, 0, 0.001 * self.Ly))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('|E_y|')
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(torch.abs(E_z.squeeze()).cpu(), cmap='jet',
                   extent=(0, 0.001 * self.Lx, 0, 0.001 * self.Ly))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('|E_z|')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"pol_{self.source.PolarizationType}_kx_{kx:.4f}_ky_{ky:.4f}.png"))
        plt.close()
        
    def Visualizing(self):
        struct_name = self.png_path.split('/')[-1][:-4]
        save_dir = f'./image_result/{self.save_name}/{struct_name}'
        os.makedirs(os.path.join(save_dir, 'Intensity'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'Normalized'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'Resized'), exist_ok=True)
        for i in range(len(self.focusrange_near)):
            z_posi = self.focusrange_near[i]
            for j in range(len(self.focusrange_far)):
                focus = self.focusrange_far[j]
                scatter_field = self.scatter_result['z_posi'][z_posi]['field']
                Intensity = torch.abs(self.Intensity[i, j, :, :].cpu().squeeze()).numpy()
                plt.figure(figsize=(12, 8))
                plt.subplot(231)
                plt.imshow(self.Geometry.cpu(), cmap='gray',
                           extent=(0, 0.001 * self.Lx, 0, 0.001 * self.Ly))
                plt.xlabel('μm')
                plt.ylabel('μm')
                plt.title('Pattern')
        
                plt.subplot(232)
                plt.imshow(Intensity, cmap='gray', extent=(0, 0.001 * self.projection.Magnification * self.Lx, 0, 0.001 * self.projection.Magnification * self.Ly))
                plt.xlabel('μm')
                plt.ylabel('μm')
                plt.title('Intensity')
                plt.colorbar()

                plt.subplot(233)
                # 生成有效的kx,ky点对用于显示
                valid_kx = []
                valid_ky = []
                # 直接使用 Imaging() 中已配置的 Source 对象
                temp_source = self.source

                # 遍历所有可能的kx,ky组合，找出有效的点
                for order_i in range(-self.order[0], self.order[0] + 1):
                    kx = self.kx0 + order_i * self.Wavelength / self.Lx
                    for order_j in range(-self.order[1], self.order[1] + 1):
                        ky = self.ky0 + order_j * self.Wavelength / self.Ly
                        if temp_source.Cal_valid(kx, ky):
                            valid_kx.append(kx)
                            valid_ky.append(ky)

                if valid_kx:  # 如果有有效点才绘制
                    plt.scatter(valid_kx, valid_ky, c='r', alpha=0.7, s=1)

                # 用橙色点单独标记 kx0, ky0
                plt.scatter(self.kx0, self.ky0, c='orange', s=30, marker='o', edgecolors='black', linewidth=1, label='(kx0, ky0)')

                # 画半径为1的圆
                circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', linewidth=1)
                plt.gca().add_patch(circle)

                plt.axis('square')
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.xlabel('kx')
                plt.ylabel('ky')
                plt.title('Valid Source')
                plt.legend()
        
                plt.subplot(234)
                plt.imshow(torch.abs(scatter_field[0].squeeze()).cpu(), cmap='jet',
                           extent=(0, 0.001 * self.Lx, 0, 0.001 * self.Ly))
                plt.xlabel('μm')
                plt.ylabel('μm')
                plt.title('|E_x|')
                plt.colorbar()
        
                plt.subplot(235)
                plt.imshow(torch.abs(scatter_field[1].squeeze()).cpu(), cmap='jet',
                           extent=(0, 0.001 * self.Lx, 0, 0.001 * self.Ly))
                plt.xlabel('μm')
                plt.ylabel('μm')
                plt.title('|E_y|')
                plt.colorbar()
        
                plt.subplot(236)
                plt.imshow(torch.abs(scatter_field[2].squeeze()).cpu(), cmap='jet',
                           extent=(0, 0.001 * self.Lx, 0, 0.001 * self.Ly))
                plt.xlabel('μm')
                plt.ylabel('μm')
                plt.title('|E_z|')
                plt.colorbar()
                plt.tight_layout()

                # plt.show()
                plt.savefig(os.path.join(save_dir, f"Show_o_{self.order}_l_{self.Wavelength}_sigma_{self.source.SigmaOut:.2f}"
                                                   f"_pol_{self.source.PolarizationType}_u_{z_posi:.2f}_v_{focus:.2f}.png"))
                plt.close()
                tifffile.imwrite(os.path.join(save_dir, f"Intensity/o_{self.order}_l_{self.Wavelength}_sigma_{self.source.SigmaOut:.2f}"
                    f"_pol_{self.source.PolarizationType}_u_{z_posi:.2f}_v_{focus:.2f}.tif"),Intensity.astype(np.float32))
                extra_func.saveNumpyAsImage(self.raster_img, os.path.join(save_dir, "raster_img.png"))
                extra_func.saveNumpyAsImage(extra_func.normalize_image(Intensity), os.path.join(save_dir, f"o_{self.order}_l_{self.Wavelength}"
                    f"_sigma_{self.source.SigmaOut:.2f}_pol_{self.source.PolarizationType}_u_{z_posi:.2f}_v_{focus:.2f}.png"))
                extra_func.saveNumpyAsImage(extra_func.normalize_image(Intensity, 110, 210), os.path.join(
                    save_dir, f"Normalized/o_{self.order}_l_{self.Wavelength}_sigma_{self.source.SigmaOut:.2f}"
                    f"_pol_{self.source.PolarizationType}_u_{z_posi:.2f}_v_{focus:.2f}_normalized.png"))
                extra_func.saveNumpyAsImage(extra_func.normalize_image(Intensity, 110, 210,[int(self.Lx / self.pixelsize),
                    int(self.Ly / self.pixelsize)]), os.path.join(save_dir, f"Resized/o_{self.order}_l_{self.Wavelength}_sigma_{self.source.SigmaOut:.2f}"
                    f"_pol_{self.source.PolarizationType}_u_{z_posi:.2f}_v_{focus:.2f}_norm_resized.png"))

    def RunBroadband(self):
        step = 10
        if self.band == 'ILine':
            Wavelength_range = [361, 365]
        elif self.band == 'MUV':
            Wavelength_range = [285, 310]
        elif self.band == 'DUV':
            Wavelength_range = [305, 306]
        elif self.band == 'GLine':
            Wavelength_range = [390, 400]
            
        cal_num = int(1 + (Wavelength_range[1] - Wavelength_range[0]) / step)
        if self.band == 'DUV':
            decay = 0.3 / cal_num  # 考虑到DUV波段光信号转换成电信号的效率
        elif self.band == 'ILine':
            # decay = -0.2 / cal_num # 考虑到ILine波段光信号转换成电信号的效率
            decay = 0
        else:
            decay = 0
        band_Intensity = 0
        
        for i in range(cal_num):
            self.Wavelength = Wavelength_range[0] + step * i
            print("\n" + " - " * 30)
            print(f"当前波长为:{self.Wavelength}，偏振为：{self.source.PolarizationType}")
            # 波长改变影响折射率和吸收系数，需要重新读取Geo信息
            self.LoadGeometry()
            
            self.Run_Single_Wavelength()
            
            band_Intensity += (1 - decay * i) * self.Intensity

        self.Intensity = band_Intensity
        self.Wavelength = str(Wavelength_range)  # 用于区分多波段与单波段结果；
        self.Visualizing()

    def RunIterative(self):
        start_time = time.time()
        Patterns = ['./SimPattern/ZQ/ZQ.png', './SimPattern/ZQ/ZQ-Open.png', './SimPattern/ZQ/ZQ-Bridge.png']
        WaveLengthes = [265, 365]
        # PolarAngles = np.concatenate([np.arange(0, 20, 2), np.arange(90, 110, 2)])
        PolarAngles = [0, 90]
        # NAs = np.arange(0.4, 0.7 +0.05/2, 0.05)
        NAs = [0.9, ]
        # SigmaOuts = np.arange(0.4, 0.7 +0.05/2, 0.05)
        SigmaOuts = [0.9, ]
        for Pattern in Patterns:
            for WaveLength in WaveLengthes:
                self.Wavelength = WaveLength
                self.png_path = Pattern
                self.LoadGeometry()
                self.Scattering()
                for PolarAngle in PolarAngles:
                    self.source.PolarizationType = PolarAngle
                    self.scatter_to_imaging()
                    for SigmaOut in SigmaOuts:
                        for NA in NAs:
                            self.source.SigmaOut = SigmaOut
                            self.ImagingAperture = NA
                            self.Imaging()
                            self.Visualizing()
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"代码总耗时: {elapsed_time:.4f} 秒")

    def Run(self):
        start_time = time.time()
        self.LoadGeometry()
        self.Scattering()
        if self.source.PolarizationType == 'None':
            self.source.PolarizationType = 'x_pol'
            self.scatter_to_imaging()
            self.Imaging()
            self.Visualizing()
            Intensity_s_pol = self.Intensity

            self.source.PolarizationType = 'y_pol'
            self.scatter_to_imaging()
            self.Imaging()
            self.Visualizing()
            self.Intensity += Intensity_s_pol

            # 宽光谱成像需要重置偏振状态
            self.source.PolarizationType = 'None'
        else:
            self.scatter_to_imaging()
            self.Imaging()
        self.Visualizing()
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"代码总耗时: {elapsed_time:.4f} 秒")