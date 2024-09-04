import torch

def CalculateAerialImage_SOCS(scatter, TCCMatrix_SOCS, source, projector):
    spectrum, f, g = scatter.CalculateSpectrum(projector, source)
    intensity = 0
    for channel in range(3):
        spectrum_channel = spectrum[:, :, channel].squeeze()
        temp = TCCMatrix_SOCS * torch.fft.fftshift(spectrum_channel).unsqueeze(2)
        temp = temp.permute(2, 1, 0)
        Etemp = (f[1] - f[0]) * (g[1] - g[0]) * torch.fft.fft2(temp)
        Esquare = torch.abs(Etemp)**2
        intensity_channel = torch.sum(Esquare, dim=0)
        intensity_channel = torch.fft.fftshift(intensity_channel)
        intensity_channel = torch.rot90(intensity_channel, 2)
        intensity = intensity + intensity_channel
    return intensity