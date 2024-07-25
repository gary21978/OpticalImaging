import torch

def CalculateAerialImage_SOCS(mask, TCCMatrix_SOCS, source, projector, numerics):
    # Get Image
    maskNf = numerics.SampleNumber_Mask_X
    maskNg = numerics.SampleNumber_Mask_Y

    waferNf = numerics.SampleNumber_Wafer_X
    waferNg = numerics.SampleNumber_Wafer_Y

    spectrum, f, g, _ = mask.CalculateMaskSpectrum(projector, source)

    if waferNf == maskNf and waferNg == maskNg:
        spectrumEx = spectrum[:-1, :-1]
    else:
        if waferNf > maskNf:
            rangeWaferNf = range((waferNf - maskNf + 2) // 2, (waferNf + maskNf - 2) // 2)
            rangeMaskNf = range(0, maskNf - 1)
        else:
            rangeWaferNf = range(0, waferNf - 1)
            rangeMaskNf = range((maskNf - waferNf + 2) // 2, (waferNf + maskNf - 2) // 2)

        if waferNg > maskNg:
            rangeWaferNg = range((waferNg - maskNg + 2) // 2, (waferNg + maskNg - 2) // 2)
            rangeMaskNg = range(0, maskNg - 1)
        else:
            rangeWaferNg = range(0, waferNg - 1)
            rangeMaskNg = range((maskNg - waferNg + 2) // 2, (waferNg + maskNg - 2) // 2)

        spectrumEx = torch.zeros(waferNf - 1, waferNg - 1)
        spectrumEx[rangeWaferNf, rangeWaferNg] = spectrum[rangeMaskNf, rangeMaskNg]

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