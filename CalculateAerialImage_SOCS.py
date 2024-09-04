import torch

def CalculateAerialImage_SOCS(scatter, TCCMatrix_SOCS, source, projector):
    spectrum, f, g = scatter.CalculateSpectrum(projector, source)
    temp = TCCMatrix_SOCS * torch.fft.fftshift(spectrum).unsqueeze(2)
    temp = temp.permute(2, 1, 0)
    Etemp = (f[1] - f[0]) * (g[1] - g[0]) * torch.fft.fft2(temp)
    Esquare = torch.abs(Etemp)**2
    intensity = torch.sum(Esquare, dim=0)
    intensity = torch.fft.fftshift(intensity)
    intensity = torch.rot90(intensity, 2)
    return intensity