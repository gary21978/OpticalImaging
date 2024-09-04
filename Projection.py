import torch
import sys

class Projection:
    def __init__(self):
        self.Aberration_Zernike = torch.zeros(37)
        self.Magnification = 1.0
        self.NA = 0.9
        self.IndexImage = 1.0
        self.FocusRange = torch.tensor([0])

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

        # calcualte aberration distribution
        aberration = Coefficients[0]
        aberration = aberration + (Coefficients[1] *
                                   torch.mul(rho, ct))
        aberration = aberration + (Coefficients[2] *
                                   torch.mul(rho, st))
        aberration = aberration + (Coefficients[3] *
                                   (2*r2-1))
        aberration = aberration + (Coefficients[4] *
                                   torch.mul(r2, c2t))
        aberration = aberration + (Coefficients[5] *
                                   torch.mul(r2, s2t))
        aberration = aberration + (Coefficients[6] *
                                   torch.mul(3*r3-2*rho, ct))
        aberration = aberration + (Coefficients[7] *
                                   torch.mul(3*r3-2*rho, st))
        aberration = aberration + (Coefficients[8] *
                                   (6*r4-6*r2+1))
        aberration = aberration + (Coefficients[9] *
                                   torch.mul(r3, c3t))
        aberration = aberration + (Coefficients[10] *
                                   torch.mul(r3, s3t))
        aberration = aberration + (Coefficients[11] *
                                   torch.mul(4*r4-3*r2, c2t))
        aberration = aberration + (Coefficients[12] *
                                   torch.mul(4*r4-3*r2, s2t))
        aberration = aberration + (Coefficients[13] *
                                   torch.mul(10*r5-12*r3+3*rho, ct))
        aberration = aberration + (Coefficients[14] *
                                   torch.mul(10*r5-12*r3+3*rho, st))
        aberration = aberration + (Coefficients[15] *
                                   (20*r6-30*r4+12*r2-1))
        aberration = aberration + (Coefficients[16] *
                                   torch.mul(r4, c4t))
        aberration = aberration + (Coefficients[17] *
                                   torch.mul(r4, s4t))
        aberration = aberration + (Coefficients[18] *
                                   torch.mul(5*r5-4*r3, c3t))
        aberration = aberration + (Coefficients[19] *
                                   torch.mul(5*r5-4*r3, s3t))
        aberration = aberration + (Coefficients[20] *
                                   torch.mul(15*r6-20*r4+6*r2, c2t))
        aberration = aberration + (Coefficients[21] *
                                   torch.mul(15*r6-20*r4+6*r2, s2t))
        aberration = aberration + (Coefficients[22] *
                                   torch.mul(35*r7-60*r5+30*r3-4*rho, ct))
        aberration = aberration + (Coefficients[23] *
                                   torch.mul(35*r7-60*r5+30*r3-4*rho, st))
        aberration = aberration + (Coefficients[24] *
                                   (70*r8-140*r6+90*r4-20*r2+1))
        aberration = aberration + (Coefficients[25] *
                                   torch.mul(r5, c5t))
        aberration = aberration + (Coefficients[26] *
                                   torch.mul(r5, s5t))
        aberration = aberration + (Coefficients[27] *
                                   torch.mul(6*r6-5*r4, c4t))
        aberration = aberration + (Coefficients[28] *
                                   torch.mul(6*r6-5*r4, s4t))
        aberration = aberration + (Coefficients[29] *
                                   torch.mul(21*r7-30*r5+10*r3, c3t))
        aberration = aberration + (Coefficients[30] *
                                   torch.mul(21*r7-30*r5+10*r3, s3t))
        aberration = aberration + (Coefficients[31] *
                                   torch.mul(56*r8-105*r6+60*r4-10*r2, c2t))
        aberration = aberration + (Coefficients[32] *
                                   torch.mul(56*r8-105*r6+60*r4-10*r2, s2t))
        aberration = aberration + (Coefficients[33] *
                                   torch.mul(126*r9-280*r7+210*r5
                                             - 60*r3+5*rho,
                                             ct))
        aberration = aberration + (Coefficients[34] *
                                   torch.mul(126*r9-280*r7+210*r5
                                             - 60*r3+5*rho,
                                             st))
        aberration = aberration + (Coefficients[35] *
                                   (252*r10-630*r8+560*r6
                                    - 210*r4+30*r2-1))
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
