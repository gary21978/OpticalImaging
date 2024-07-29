import torch
import torch.special

class Mask:
    def __init__(self):
        self.Period_X = 500
        self.Period_Y = 500
        self.Feature = []

    def CalculateMaskSpectrum(self, po, sr):
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