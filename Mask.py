import math
import cmath
import torch
import torch.special

class Mask:
    def __init__(self):
        self.MaskType = '2D'
        self.Period_X = 500
        self.Period_Y = 500
        self.Nf = 81
        self.Ng = 81
        self.Background_Transmissivity = 0
        self.Background_Phase = 0
        self.Orientation = 0
        self.Feature = []
        self.Cutline = []

    def CalculateMaskSpectrum(self, po, sr):
        NA = po.NA
        Wavelength = sr.Wavelength
        HkBlank = None
        if (self.MaskType.lower() == '2d'):
            Hk, F, G =\
                self.__Mask2D2Spectrum(NA, Wavelength)
        elif (self.MaskType.lower() == '2dpixel'):
            Hk, F, G =\
                self.__PixelMask2D2Spectrum(NA, Wavelength)
        else:
            raise ValueError(
                'Error: There is not this type of mask!')
        return Hk, F, G

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
            temp = (a - bgComplexAmplitude) * normalized_Frequency**2\
                    * xWidth * yWidth * (torch.mm(tempy, tempx))
            Hk = Hk + temp
            
        bool_tensor = torch.mm((torch.abs(g) < 1e-9).unsqueeze(1).int(),
                               (torch.abs(f) < 1e-9).unsqueeze(0).int())
        value = (normalized_Period_X*normalized_Period_Y) * bgComplexAmplitude
        BkSpectrum = torch.where(bool_tensor == 1, value, 0)
        Hk = Hk + BkSpectrum
        Hk = torch.t(Hk)
        return Hk, f, g

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

        # AFactor: PntNum_X rows PntNum_Y columns
        AFactor = normPitchX / (PntNum_X - 1) * normPitchY / (PntNum_Y - 1) *\
        torch.mm(
            torch.exp(-1j * 2 * math.pi * normSY * g).unsqueeze(1),
            torch.exp(-1j * 2 * math.pi * normSX * f).unsqueeze(0)
        )

        nDFTVectorX = torch.arange(PntNum_X)
        nDFTVectorY = torch.arange(PntNum_Y)
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
        return Hk, f, g

    def CreateLineMask(self, lineCD, linePitch):
        self.MaskType = '1D'
        self.bgTransmission = 0.0
        self.bgPhase = 0
        self.Background_Transmissivity = 1
        self.Background_Phase = 0
        self.Period_X = linePitch  # 1000 at the beginning
        self.Period_Y = linePitch  # 1000 at the beginning

        self.Feature = [Feature(boundaryvertexX=[-lineCD/2, lineCD/2],
                        boundaryvertexY=[-linePitch/2, linePitch/2],
                        complexam=[0, 0])]

        # Set cutline
        self.Cutline = [Cutline(x=[-linePitch/2, linePitch/2],
                                y=[0, 0])]
        return self

    @staticmethod
    def CreateMask(maskType, varargin=[]):
        mask = Mask()
        
        if (1):
            mask.bgTransmission = 0.0
            mask.bgPhase = 0
            mask.Background_Transmissivity = 0
            mask.Background_Phase = 0
            if (maskType.lower() == 'crossgate'):
                mask.Period_X = 420  # 1000 at the beginning
                mask.Period_Y = 420  # 1000 at the beginning
                boundaryVertexX = [[-180, -135],
                                   [-75, -30],
                                   [-75, -30],
                                   [30, 75],
                                   [-30, 75],
                                   [30, 75],
                                   [135, 180]]
                boundaryVertexY = [[-210, 210],
                                   [-25, 210],
                                   [-210, -70],
                                   [-210, -25],
                                   [-25, 25],
                                   [70, 210],
                                   [-210, 210]]
                cutlineX = [[0, 105],
                            [0, 0],
                            [-105, 0],
                            [105, 210]]
                cutlineY = [[100, 100],
                            [-100, 100],
                            [100, 100],
                            [0, 0]]
                # rectangle
                for i in range(len(boundaryVertexX)):
                    mask.Feature.append(
                        Feature(        boundaryvertexX=boundaryVertexX[i],
                                boundaryvertexY=boundaryVertexY[i],
                                complexam=[1, 0])
                        )
                # Set cutline
                for i in range(len(cutlineX)):
                    mask.Cutline.append(
                        Cutline(x=cutlineX[i],
                                y=cutlineY[i])
                        )
            elif (maskType.lower() == 'line_space'):
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
                        Feature(        boundaryvertexX=boundaryVertexX[i],
                                boundaryvertexY=boundaryVertexY[i],
                                complexam=[1, 0])
                        )
            elif (maskType.lower() == 'complex'):
                mask.bgTransmission = 0.0
                mask.bgPhase = 0
                mask.Period_X = 1200  # 1000 at the beginning
                mask.Period_Y = 1200  # 1000 at the beginning
                boundaryVertexX = [[-510, 510],
                                   [240, 465],
                                   [240, 245],
                                   [120, 245],
                                   [120, 245],
                                   [240, 245],
                                   [240, 465],
                                   [-345, -120],
                                   [-345, -240],
                                   [-465, -240],
                                   [-465, -240],
                                   [-345, -240],
                                   [-345, -120],
                                   [-510, 510],
                                   [-345, -120],
                                   [-345, -240],
                                   [-465, -240],
                                   [-465, -240],
                                   [-345, -240],
                                   [-345, -120],
                                   [240, 465],
                                   [240, 245],
                                   [120, 245],
                                   [120, 245],
                                   [240, 245],
                                   [240, 465],
                                   [-510, 510]]
                boundaryVertexY = [[427.5, 472.5],
                                   [337.5, 382.5],
                                   [292.5, 337.5],
                                   [247.5, 292.5],
                                   [157.5, 202.5],
                                   [112.5, 157.5],
                                   [67.5, 112.5],
                                   [247.5, 292.5],
                                   [292.5, 337.5],
                                   [337.5, 382.5],
                                   [67.5, 112.5],
                                   [112.5, 157.5],
                                   [157.5, 202.5],
                                   [-22.5, 22.5],
                                   [-202.5, -157.5],
                                   [-157.5, -112.5],
                                   [-112.5, -67.5],
                                   [-382.5, -337.5],
                                   [-337.5, -292.5],
                                   [-292.5, -247.5],
                                   [-112.5, -67.5],
                                   [-157.5, -112.5],
                                   [-202.5, -157.5],
                                   [-292.5, -247.5],
                                   [-337.5, -292.5],
                                   [-382.5, -337.5],
                                   [-472.5, -427.5]]
                for i in range(len(boundaryVertexX)):
                    mask.Feature.append(
                        Feature(        boundaryvertexX=boundaryVertexX[i],
                                boundaryvertexY=boundaryVertexY[i],
                                complexam=[1, 0])
                        )
            elif (maskType.lower() == 'sram'):
                maskSize = 400
                mask.bgTransmission = 0.0
                mask.bgPhase = 0
                mask.Period_X = maskSize  # 1000 at the beginning
                mask.Period_Y = maskSize  # 1000 at the beginning
                mask.Feature.append(
                        Feature(        boundaryvertexX=[-510, -500],
                                boundaryvertexY=[-472.5, -300.5],
                                complexam=[1, 0])
                        )
            else:
                raise ValueError(
                    'error:This type of mask has not been included')

        return mask

    @staticmethod
    def CreateparameterizedMask(maskType, varargin=[]):
        mask = Mask()
        length = len(varargin)
        if (maskType.lower() == 'line'):
            if (length == 0):
                lineCD = 45
                linePitch = 90
            elif (length == 1):
                lineCD = varargin[0]
                linePitch = lineCD * 2
            elif (length == 2):
                lineCD = varargin[0]
                linePitch = varargin[1]
            mask = mask.CreateLineMask(lineCD, linePitch)
        elif (maskType.lower() == 'space'):
            if (length == 0):
                spaceCD = 45
                spacePitch = 90
            elif (length == 1):
                spaceCD = varargin[0]
                spacePitch = spaceCD * 2
            elif (length == 2):
                spaceCD = varargin[0]
                spacePitch = varargin[1]
            mask = mask.CreateLineMask(spaceCD, spacePitch)
        elif (maskType.lower() == 'space_end_dense'):
            if (length == 0):
                gapCD = 45
                spaceCD = 45
                spacePitch = 1000
            elif (length == 1):
                gapCD = varargin[0]
                spaceCD = 45
                spacePitch = spaceCD * 2
            elif (length == 2):
                gapCD = varargin[0]
                spaceCD = varargin[1]
                spacePitch = spaceCD * 2
            elif (length == 3):
                gapCD = varargin[0]
                spaceCD = varargin[1]
                spacePitch = varargin[2]

            mask.bgTransmission = 0.0
            mask.bgPhase = 0
            mask.Background_Transmissivity = 0
            mask.Background_Phase = 0
            mask.Period_X = spacePitch  # default: 90
            mask.Period_Y = spacePitch  # default: 90

            mask.Feature.append(
                Feature(boundaryvertexX=[-mask.Period_X/2, -gapCD/2],
                        boundaryvertexY=[-spaceCD/2, spaceCD/2],
                        complexam=[1, 0])
            )
            mask.Feature.append(
                Feature(boundaryvertexX=[gapCD/2, mask.Period_X/2],
                        boundaryvertexY=[-spaceCD/2, spaceCD/2],
                        complexam=[1, 0])
            )
        return mask

class Feature:
    def __init__(self,
                 boundaryvertexX,
                 boundaryvertexY,
                 complexam):
        self.BoundaryVertexX = boundaryvertexX
        self.BoundaryVertexY = boundaryvertexY
        self.ComplexAm = complexam

class Cutline:
    def __init__(self, x, y):
        self.X = x
        self.Y = y