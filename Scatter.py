import torch

class Scatter:
    def __init__(self):
        self.Period_X = 1000  # nanometer
        self.Period_Y = 1000  # nanometer
        self.ScatterField = []

    def CalculateSpectrum(self, po, sr):
        NA = po.NA
        Wavelength = sr.Wavelength
        h = self.ScatterField
        h = h.to(torch.complex64)
        PntNum_Y, PntNum_X, _ = h.size()
        normPitchX = self.Period_X / (Wavelength / NA)
        normPitchY = self.Period_Y / (Wavelength / NA)
        f = (1/normPitchX)*torch.arange(-(PntNum_X//2), (PntNum_X + 1)//2)
        g = (1/normPitchY)*torch.arange(-(PntNum_Y//2), (PntNum_Y + 1)//2)
        hh = torch.fft.fftshift(h, dim=(0, 1))
        DFThh = torch.fft.fft2(hh, dim=(0, 1))
        Hk = torch.fft.fftshift(DFThh, dim=(0, 1))
        Hk = normPitchX*normPitchY/PntNum_X/PntNum_Y * Hk
        
        PupilImage = Hk.clone()
        fm, gm = torch.meshgrid(g, f, indexing='ij')
        fm = torch.stack((fm, fm, fm),dim=2)
        gm = torch.stack((gm, gm, gm),dim=2)
        PupilImage[fm**2 + gm**2 > 1] = float('nan')

        offsetX = int(normPitchX) + 3
        offsetY = int(normPitchY) + 3

        PupilImage = PupilImage[(PntNum_Y // 2) - offsetY:(PntNum_Y // 2) + offsetY,\
                                (PntNum_X // 2) - offsetX:(PntNum_X // 2) + offsetX]
        return Hk, f, g, PupilImage