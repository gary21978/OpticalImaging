import torch

class Mask:
    def __init__(self):
        self.Period_X = 1000  # nanometer
        self.Period_Y = 1000  # nanometer
        self.Feature = []

    def CalculateSpectrum(self, po, sr):
        NA = po.NA
        Wavelength = sr.Wavelength
        h = self.Feature
        h = h.to(torch.complex64)
        PntNum_Y, PntNum_X = h.size()  # PntNum is odd

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
        """
        # AFactor: PntNum_X rows PntNum_Y columns
        AFactor = normPitchX / (PntNum_X - 1) * normPitchY / (PntNum_Y - 1) *\
        torch.mm(
            torch.exp(-1j * 2 * torch.pi * normSY * g).unsqueeze(1),
            torch.exp(-1j * 2 * torch.pi * normSX * f).unsqueeze(0)
        )

        nDFTVectorX = torch.arange(PntNum_X)
        nDFTVectorY = torch.arange(PntNum_Y)
        hFactor = torch.mm(
            torch.exp(1j * torch.pi * nDFTVectorY).unsqueeze(1),
            torch.exp(1j * torch.pi * nDFTVectorX).unsqueeze(0)
        )
        hh = torch.mul(h, hFactor)
        DFTH_2D = torch.fft.fft2(hh)
        Hk = torch.mul(
            AFactor,
            DFTH_2D
        )
        """
        hh = torch.fft.fftshift(h)
        DFTH_2D = torch.fft.fft2(hh)
        Hk = torch.fft.fftshift(DFTH_2D)
        Hk = self.period_X*self.period_Y/PntNum_X/PntNum_Y /(Wavelength / NA)**2 * Hk
        return Hk, f, g