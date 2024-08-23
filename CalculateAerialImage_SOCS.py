import torch

def CalculateAerialImage_SOCS(mask, TCCMatrix_SOCS, source, projector):
    spectrum, f, g = mask.CalculateMaskSpectrum(projector, source)
    spectrumEx = spectrum[:-1, :-1]
    temp = TCCMatrix_SOCS * torch.fft.fftshift(spectrumEx).unsqueeze(2)
    temp = temp.permute(2, 1, 0)
    Etemp = (f[1] - f[0]) * (g[1] - g[0]) * torch.fft.fft2(temp)
    Esquare = torch.abs(Etemp)**2
    intensity = torch.sum(Esquare, dim=0)
    intensity = torch.fft.fftshift(intensity)
    intensity = torch.cat((intensity, intensity[:, 0].unsqueeze(1)), dim = 1)
    intensity = torch.cat((intensity, intensity[0, :].unsqueeze(0)), dim = 0)
    intensity = torch.rot90(intensity, 2)
    return intensity