import math
import cmath
import torch
import torch.special
from scipy.spatial import Delaunay
import numpy as np
from matplotlib import path
from Source import Source
from ProjectionObjective import ProjectionObjective as PO

IMAGE_WH = 2048
class Mask:
    Spectrum = 1
    Fx = 0
    Gy = 0

    def __init__(self):
        self.MaskType = '2D'
        self.Period_X = 500
        self.Period_Y = 500
        self.Nf = 81
        self.Ng = 81
        self.Orientation = 0
        self.Background_Transmissivity = 0
        self.Background_Phase = 0
        self.MaskDefineType = '2D'
        self.Feature = []
        self.Cutline = []

    def CalculateMaskSpectrum(self, po, sr):
        NA = po.NA
        Wavelength = sr.Wavelength
        Hk, F, G = self.__Mask2D2Spectrum(NA, Wavelength)
        return Hk, F, G, None
    
    def __PixelMask2D2Spectrum(self, NA, Wavelength):
        h = self.Feature
        h = h.to(torch.complex64)
        PntNum_Y, PntNum_X = h.size()  # PntNum is odd

        if (self.Nf != PntNum_X):
            raise ValueError(
                'error: mask data size 2 must be equal to Mask.Nf ')
        if (self.Ng != PntNum_Y):
            raise ValueError(
                'error: mask data size 1 must be equal to Mask.Ng ')

        period_X = self.Period_X
        period_Y = self.Period_Y
        Start_X = -self.Period_X / 2
        Start_Y = -self.Period_Y / 2

        normPitchX = period_X / (Wavelength / NA)
        normPitchY = period_Y / (Wavelength / NA)
        normSX = Start_X / (Wavelength / NA)
        normSY = Start_Y / (Wavelength / NA)

        f = (1/normPitchX) * torch.arange(start=-(PntNum_X - 1) / 2,
                                          end=(PntNum_X - 1) / 2 + 1,
                                          step=1)
        g = (1/normPitchY) * torch.arange(start=-(PntNum_Y - 1) / 2,
                                          end=(PntNum_Y - 1) / 2 + 1,
                                          step=1)

        normdx = normPitchX / (PntNum_X - 1)
        normdy = normPitchY / (PntNum_Y - 1)

        normDfx = 1 / normPitchX
        normDfy = 1 / normPitchY
        normfx = normDfx * torch.arange(start=-(PntNum_X - 1) / 2,
                                        end=(PntNum_X - 1) / 2 + 1,
                                        step=1)
        normfy = normDfy * torch.arange(start=-(PntNum_Y - 1) / 2,
                                        end=(PntNum_Y - 1) / 2 + 1,
                                        step=1)
        nDFTVectorX = torch.linspace(0, PntNum_X - 1, PntNum_X)
        nDFTVectorY = torch.linspace(0, PntNum_Y - 1, PntNum_Y)

        # AFactor: PntNum_X rows PntNum_Y columns
        AFactor1 = torch.mm(
            torch.exp(-1j * 2 * math.pi * normSY * normfy).unsqueeze(1),
            torch.exp(-1j * 2 * math.pi * normSX * normfx).unsqueeze(0)
        )
        AFactor = (normdx * normdy) * AFactor1

        hFactor = torch.mm(
            torch.exp(1j * math.pi * nDFTVectorY).unsqueeze(1),
            torch.exp(1j * math.pi * nDFTVectorX).unsqueeze(0)
        )
        hh = torch.mul(h, hFactor)

        h2D = hh[0:-1, 0:-1]
        HYEnd = hh[-1, 0:-1]
        HXEnd = hh[0:-1, -1]
        HXYEnd = hh[-1, -1]

        DFTH_2D = torch.fft.fft2(h2D)
        DFTH_2D = torch.cat((DFTH_2D, DFTH_2D[:, 0].unsqueeze(1)), 1)
        DFTH_2D = torch.cat((DFTH_2D, DFTH_2D[0].unsqueeze(0)), 0)

        DFTHYEnd = torch.fft.fft(HYEnd)
        DFTHYEnd = torch.cat((DFTHYEnd, DFTHYEnd[0].unsqueeze(0)), 0)
        DFTHM_12D = torch.mm(
            torch.ones((PntNum_Y, 1), dtype=torch.complex64),
            DFTHYEnd.unsqueeze(0)
        )

        DFTHXEnd = torch.fft.fft(HXEnd)
        DFTHXEnd = torch.cat((DFTHXEnd, DFTHXEnd[0].unsqueeze(0)), 0)
        DFTHN_12D = torch.mm(
            DFTHXEnd.unsqueeze(1),
            torch.ones((1, PntNum_X), dtype=torch.complex64)
        )

        Hk = torch.mul(
            AFactor,
            DFTH_2D + DFTHM_12D + DFTHN_12D + HXYEnd
        )

        normalized_Frequency = NA / Wavelength
        tempx = torch.sinc(self.Period_X * normalized_Frequency * f)
        tempy = torch.sinc(self.Period_Y * normalized_Frequency * g)
        HkBlank = normalized_Frequency**2 * self.Period_X * \
            self.Period_Y * torch.mm(tempy.unsqueeze(1), tempx.unsqueeze(0))
        return Hk, f, g, HkBlank

    def __Mask2D2Spectrum(self, NA, Wavelength):
        bgComplexAmplitude = self.Background_Transmissivity *\
            cmath.exp(1j * self.Background_Phase)
        # Normalize the sampling spacing
        normalized_Period_X = self.Period_X / (Wavelength/NA)
        # The square refractive index has been taken into account in NA
        normalized_Period_Y = self.Period_Y / (Wavelength/NA)
        f = (1/normalized_Period_X) * torch.arange(start=-(self.Nf-1)/2,
                                                   end=(self.Nf-1)/2 + 1,
                                                   step=1)
        g = (1/normalized_Period_Y) * torch.arange(start=-(self.Ng-1)/2,
                                                   end=(self.Ng-1)/2 + 1,
                                                   step=1)
        normalized_Frequency = NA/Wavelength
        Hk = torch.zeros(len(g), len(f))
        SubRectNum = len(self.Feature)
        
        for ii in range(SubRectNum):
            # Simplify design by only defining diagonal points
            xWidth = abs(self.Feature[ii].BoundaryVertexX[1]
                             - self.Feature[ii].BoundaryVertexX[0])
            yWidth = abs(self.Feature[ii].BoundaryVertexY[1]
                             - self.Feature[ii].BoundaryVertexY[0])
            a = self.Feature[ii].ComplexAm[0] *\
                    cmath.exp(1j * self.Feature[ii].ComplexAm[1])
            xCenter = (self.Feature[ii].BoundaryVertexX[0] +
                           self.Feature[ii].BoundaryVertexX[1]) / 2
            yCenter = (self.Feature[ii].BoundaryVertexY[0] +
                           self.Feature[ii].BoundaryVertexY[1]) / 2
            tempx = (torch.mul(
                    torch.exp(-1j*2*math.pi*xCenter*normalized_Frequency*f),
                    torch.sinc(xWidth*normalized_Frequency*f))).unsqueeze(0)
            tempy = (torch.mul(
                    torch.exp(-1j*2*math.pi*yCenter*normalized_Frequency*g),
                    torch.sinc(yWidth*normalized_Frequency*g))).unsqueeze(1)
            temp = (a-bgComplexAmplitude) * normalized_Frequency**2\
                    * xWidth * yWidth * (torch.mm(tempy, tempx))
            Hk = Hk + temp
        bool_tensor = torch.mm((torch.abs(g) < 1e-9).unsqueeze(1).int(),
                               (torch.abs(f) < 1e-9).unsqueeze(0).int())
        value = (normalized_Period_X*normalized_Period_Y) * bgComplexAmplitude
        BkSpectrum = torch.where(bool_tensor == 1, value, 0)
        Hk = Hk + BkSpectrum
        Hk = torch.t(Hk)
        return Hk, f, g

    @staticmethod
    def CreateMask(maskType, varargin=[]):
        mask = Mask()
        length = len(varargin)
        if True:
            mask.bgTransmission = 0.0
            mask.bgPhase = 0
            mask.Background_Transmissivity = 0
            mask.Background_Phase = 0
            if (maskType.lower() == 'line_space'):
                mask.Period_X = 720  # 1000 at the beginning
                mask.Period_Y = 720  # 1000 at the beginning
                boundaryVertexX = [[-22.5, 22.5],
                                   [67.5, 112.5],
                                   [157.5, 202.5],
                                   [-112.5, -67.5],
                                   [-202.5, -157.5]]
                boundaryVertexY = [[-300, 300],
                                   [-300, 300],
                                   [-300, 300],
                                   [-300, 300],
                                   [-300, 300]]
                for i in range(len(boundaryVertexX)):
                    mask.Feature.append(
                        Feature(shapetype='r',
                                boundaryvertexX=boundaryVertexX[i],
                                boundaryvertexY=boundaryVertexY[i],
                                complexam=[1, 0])
                        )
            else:
                raise ValueError(
                    'error:This type of mask has not been included')

        return mask
    
    

class Feature:
    def __init__(self,
                 shapetype,
                 boundaryvertexX,
                 boundaryvertexY,
                 complexam):
        self.ShapeType = shapetype
        self.BoundaryVertexX = boundaryvertexX
        self.BoundaryVertexY = boundaryvertexY
        self.ComplexAm = complexam
