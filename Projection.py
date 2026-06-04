import torch
import sys
import math

class Projection:
    def __init__(self):
        self.Wavelength = 500
        self.Aberration_Zernike = torch.zeros(37)
        self.Magnification = 100
        self.IndexImage = 1.0
        self.FocusRange = torch.tensor([0])
        
        self.Shape = "annular"
        # Source shape :
        # 'annular''dipolecirc''quasar''rect'

        self.SigmaOut = 0.9
        self.SigmaIn = 0.0
        self.OpenAngle = math.pi/3
        self.RotateAngle = 0
        self.RectX = 0.5
        self.RectY = 0.5
        
    def Cal_valid(self, kx, ky):
        """
        判断 (kx, ky) 是否落在投影光瞳（imaging pupil）的有效区域内。
        
        参数:
            kx, ky: torch.Tensor，形状相同，表示归一化频谱坐标（通常来自衍射级次）
            
        返回:
            valid_mask: torch.BoolTensor，与 kx/ky 同形状，True 表示在光瞳内
        """
        # 确保输入为张量
        if not isinstance(kx, torch.Tensor):
            kx = torch.tensor(kx, dtype=torch.float64)
        if not isinstance(ky, torch.Tensor):
            ky = torch.tensor(ky, dtype=torch.float64)
    
        shape = self.Shape.lower()
    
        if shape == "pixel":
            raise ValueError("Unsupported imaging pupil: 'pixel' is under development.")
    
        elif shape == "annular":
            r2 = kx**2 + ky**2
            return (r2 <= self.SigmaOut**2) & (r2 >= self.SigmaIn**2)
    
        elif shape == "dipolecirc":
            open_angle = self.OpenAngle
            rotate_angle = self.RotateAngle
    
            if not (-math.pi / 2 <= rotate_angle <= math.pi / 2):
                raise ValueError("rotateAngle must be in [-pi/2, pi/2]")
            if not (0 <= open_angle <= math.pi / 2):
                raise ValueError("openAngle must be in [0, pi/2]")
    
            radius = torch.sqrt(kx**2 + ky**2)
            theta = torch.atan2(ky, kx)
    
            # 方向约束：主瓣在 rotate_angle 方向，张角为 open_angle
            cos_diff = torch.cos(theta - rotate_angle)
            angle_ok = torch.abs(cos_diff) >= math.cos(open_angle / 2)
    
            return (radius <= self.SigmaOut) & (radius >= self.SigmaIn) & angle_ok
    
        elif shape == "rect":
            rect_x = self.RectX
            rect_y = self.RectY
            rotate_angle = self.RotateAngle
    
            if rect_x**2 + rect_y**2 > 1.0:
                raise ValueError("RectX^2 + RectY^2 must be <= 1 for normalized pupil")
    
            # 逆旋转坐标系
            cos_t = math.cos(-rotate_angle)
            sin_t = math.sin(-rotate_angle)
            x_prime = kx * cos_t - ky * sin_t
            y_prime = kx * sin_t + ky * cos_t
    
            return (torch.abs(x_prime) <= rect_x) & (torch.abs(y_prime) <= rect_y)
    
        elif shape == "quasar":
            open_angle = self.OpenAngle
            rotate_angle = self.RotateAngle
    
            if not (-math.pi / 2 <= rotate_angle <= math.pi / 2):
                raise ValueError("rotateAngle must be in [-pi/2, pi/2]")
            if not (0 <= open_angle <= math.pi / 2):
                raise ValueError("openAngle must be in [0, pi/2]")
    
            radius = torch.sqrt(kx**2 + ky**2)
            theta = torch.atan2(ky, kx)
    
            # 四极：0°, 90°, 180°, 270° 附近四个瓣
            def in_sector(center):
                diff = torch.remainder(theta - center + math.pi, 2 * math.pi) - math.pi  # wrap to [-pi, pi]
                return torch.abs(diff) <= open_angle / 2
    
            sector0 = in_sector(rotate_angle)
            sector90 = in_sector(rotate_angle + math.pi / 2)
            sector180 = in_sector(rotate_angle + math.pi)
            sector270 = in_sector(rotate_angle - math.pi / 2)
    
            angle_ok = sector0 | sector90 | sector180 | sector270
    
            return (radius <= self.SigmaOut) & (radius >= self.SigmaIn) & angle_ok
    
        elif shape == "multipole":
            raise ValueError("Unsupported imaging pupil: 'multipole' is under development.")
    
        else:
            raise ValueError(f"Unsupported pupil shape: {self.Shape}")
            
    """Fringe set"""
    def CalculateAberrationFast(self, rho, theta, Orientation):
        r2 = rho.pow(2)
        r3 = rho.pow(3)
        r4 = rho.pow(4)
        r5 = rho.pow(5)
        r6 = rho.pow(6)
        r7 = rho.pow(7)
        r8 = rho.pow(8)
        r9 = rho.pow(9)
        r10 = rho.pow(10)
        r12 = rho.pow(12)

        # azimuthal value
        ct = torch.cos(theta)
        st = torch.sin(theta)

        c2t = torch.cos(2*theta)
        s2t = torch.sin(2*theta)

        c3t = torch.cos(3*theta)
        s3t = torch.sin(3*theta)

        c4t = torch.cos(4*theta)
        s4t = torch.sin(4*theta)

        c5t = torch.cos(5*theta)
        s5t = torch.sin(5*theta)

        if (abs(Orientation) > sys.float_info.epsilon):
            Coefficients = Projection.RotateAngle(
                self.Aberration_Zernike,
                Orientation)
        else:
            Coefficients = self.Aberration_Zernike

        # 计算像差分布，逐步累加每个Zernike项的贡献
        # Z0: (Piston) - 常数项，常用于整体相位偏移
        aberration = Coefficients[0]
        # Z1: X方向倾斜 (Tilt X) - 对应瞳孔坐标的x分量倾斜
        aberration = aberration + (Coefficients[1] *
                                   torch.mul(rho, ct))
        # Z2: Y方向倾斜 (Tilt Y) - 对应瞳孔坐标的y分量倾斜
        aberration = aberration + (Coefficients[2] *
                                   torch.mul(rho, st))
        # Z3: 离焦 (Defocus) - 焦点位置的偏移
        aberration = aberration + (Coefficients[3] *
                                   (2*r2-1))
        # Z4: 0°像散 (Astigmatism 0°) - 45°方向的像散
        aberration = aberration + (Coefficients[4] *
                                   torch.mul(r2, c2t))
        # Z5: 45°像散 (Astigmatism 45°) - 0°方向的像散
        aberration = aberration + (Coefficients[5] *
                                   torch.mul(r2, s2t))
        # Z6: X方向彗差 (Coma X) - x方向的彗形像差
        aberration = aberration + (Coefficients[6] *
                                   torch.mul(3*r3-2*rho, ct))
        # Z7: Y方向彗差 (Coma Y) - y方向的彗形像差
        aberration = aberration + (Coefficients[7] *
                                   torch.mul(3*r3-2*rho, st))
        # Z8: 球差 (Spherical Aberration) - 径向对称的球面像差
        aberration = aberration + (Coefficients[8] *
                                   (6*r4-6*r2+1))
        # Z9: 三叶形 (Trifoil) cos - 三叶形像差的cos分量
        aberration = aberration + (Coefficients[9] *
                                   torch.mul(r3, c3t))
        # Z10: 三叶形 (Trifoil) sin - 三叶形像差的sin分量
        aberration = aberration + (Coefficients[10] *
                                   torch.mul(r3, s3t))
        # Z11: 0°像散二阶 (Secondary Astigmatism 0°)
        aberration = aberration + (Coefficients[11] *
                                   torch.mul(4*r4-3*r2, c2t))
        # Z12: 45°像散二阶 (Secondary Astigmatism 45°)
        aberration = aberration + (Coefficients[12] *
                                   torch.mul(4*r4-3*r2, s2t))
        # Z13: X方向二阶彗差 (Secondary Coma X)
        aberration = aberration + (Coefficients[13] *
                                   torch.mul(10*r5-12*r3+3*rho, ct))
        # Z14: Y方向二阶彗差 (Secondary Coma Y)
        aberration = aberration + (Coefficients[14] *
                                   torch.mul(10*r5-12*r3+3*rho, st))
        # Z15: 二阶球差 (Secondary Spherical Aberration)
        aberration = aberration + (Coefficients[15] *
                                   (20*r6-30*r4+12*r2-1))
        # Z16: 四叶形 (Quadrafoil) cos
        aberration = aberration + (Coefficients[16] *
                                   torch.mul(r4, c4t))
        # Z17: 四叶形 (Quadrafoil) sin
        aberration = aberration + (Coefficients[17] *
                                   torch.mul(r4, s4t))
        # Z18: 三叶形二阶 (Tertiary Trifoil) cos
        aberration = aberration + (Coefficients[18] *
                                   torch.mul(5*r5-4*r3, c3t))
        # Z19: 三叶形二阶 (Tertiary Trifoil) sin
        aberration = aberration + (Coefficients[19] *
                                   torch.mul(5*r5-4*r3, s3t))
        # Z20: 0°像散三阶 (Tertiary Astigmatism 0°)
        aberration = aberration + (Coefficients[20] *
                                   torch.mul(15*r6-20*r4+6*r2, c2t))
        # Z21: 45°像散三阶 (Tertiary Astigmatism 45°)
        aberration = aberration + (Coefficients[21] *
                                   torch.mul(15*r6-20*r4+6*r2, s2t))
        # Z22: X方向三阶彗差 (Tertiary Coma X)
        aberration = aberration + (Coefficients[22] *
                                   torch.mul(35*r7-60*r5+30*r3-4*rho, ct))
        # Z23: Y方向三阶彗差 (Tertiary Coma Y)
        aberration = aberration + (Coefficients[23] *
                                   torch.mul(35*r7-60*r5+30*r3-4*rho, st))
        # Z24: 三阶球差 (Tertiary Spherical Aberration)
        aberration = aberration + (Coefficients[24] *
                                   (70*r8-140*r6+90*r4-20*r2+1))
        # Z25: 五叶形 (Pentafoil) cos
        aberration = aberration + (Coefficients[25] *
                                   torch.mul(r5, c5t))
        # Z26: 五叶形 (Pentafoil) sin
        aberration = aberration + (Coefficients[26] *
                                   torch.mul(r5, s5t))
        # Z27: 四叶形二阶 (Secondary Quadrafoil) cos
        aberration = aberration + (Coefficients[27] *
                                   torch.mul(6*r6-5*r4, c4t))
        # Z28: 四叶形二阶 (Secondary Quadrafoil) sin
        aberration = aberration + (Coefficients[28] *
                                   torch.mul(6*r6-5*r4, s4t))
        # Z29: 三叶形三阶 (Tertiary Trifoil) cos
        aberration = aberration + (Coefficients[29] *
                                   torch.mul(21*r7-30*r5+10*r3, c3t))
        # Z30: 三叶形三阶 (Tertiary Trifoil) sin
        aberration = aberration + (Coefficients[30] *
                                   torch.mul(21*r7-30*r5+10*r3, s3t))
        # Z31: 0°像散四阶 (Quaternary Astigmatism 0°)
        aberration = aberration + (Coefficients[31] *
                                   torch.mul(56*r8-105*r6+60*r4-10*r2, c2t))
        # Z32: 45°像散四阶 (Quaternary Astigmatism 45°)
        aberration = aberration + (Coefficients[32] *
                                   torch.mul(56*r8-105*r6+60*r4-10*r2, s2t))
        # Z33: X方向四阶彗差 (Quaternary Coma X)
        aberration = aberration + (Coefficients[33] *
                                   torch.mul(126*r9-280*r7+210*r5
                                             - 60*r3+5*rho,
                                             ct))
        # Z34: Y方向四阶彗差 (Quaternary Coma Y)
        aberration = aberration + (Coefficients[34] *
                                   torch.mul(126*r9-280*r7+210*r5
                                             - 60*r3+5*rho,
                                             st))
        # Z35: 四阶球差 (Quaternary Spherical Aberration)
        aberration = aberration + (Coefficients[35] *
                                   (252*r10-630*r8+560*r6
                                    - 210*r4+30*r2-1))
        # Z36: 高阶径向多项式 (High-order radial term)
        aberration = aberration + (Coefficients[36] *
                                   (924*r12-2772*r10+3150*r8
                                    - 1680*r6+420*r4-42*r2+1))
        return aberration

    @staticmethod
    def RotateAngle(c0, theta):  # Rotate zernike aberration
        # 1. There is no change in COS and SIN items
        # 2. The cos term is equal to itself multiplied by cos
        #    plus the corresponding sin-containing term multiplied by sin
        # 3. The sin term is equal to itself multiplied by sin
        #    plus the corresponding cos-containing term multiplied by cos
        mm = torch.tensor([0, 1, 1, 0, 2, 2,
                           1, 1, 0, 3, 3, 2,
                           2, 1, 1, 0, 4, 4,
                           3, 3, 2, 2, 1, 1,
                           0, 5, 5, 4, 4, 3,
                           3, 2, 2, 1, 1, 0, 0])  # m
        tt = torch.tensor([0, 1, -1, 0, 1, -1,
                           1, -1, 0, 1, -1, 1,
                          -1, 1, -1, 0, 1, -1,
                           1, -1, 1, -1, 1, -1,
                           0, 1, -1, 1, -1, 1,
                          -1, 1, -1, 1, -1, 0, 0])
        pp = torch.tensor([0, 2, 1, 3, 5, 4, 7, 6,
                           8, 10, 9, 12, 11, 14,
                           13, 15, 17, 16, 19, 18,
                           21, 20, 23, 22, 24, 26,
                           25, 28, 27, 30, 29, 32,
                           31, 34, 33, 35, 36])

        c1 = torch.zeros(37)
        for ii in range(37):
            c1[ii] = c1[ii] + c0[ii] * torch.cos(mm[ii] * theta)
            c1[pp[ii]] = c1[pp[ii]] - tt[ii] * c0[ii]\
                * torch.sin(mm[ii]*theta)
        return c1
