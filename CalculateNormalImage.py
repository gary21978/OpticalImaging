import torch

def cartesian_to_polar(x, y):
    
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)

    return rho, theta

def CalculateNormalImage(source, mask, projector, numerics):

    source.PntNum = numerics.SampleNumber_Source
    sourceData = source.Calc_SourceSimple()
    weight = sourceData.Value.sum()
    wavelength = source.Wavelength  # 193
    NA = projector.NA
    M = projector.Reduction
    if projector.LensType == 'Immersion':
        indexImage = projector.Index_ImmersionLiquid
    elif projector.LensType == 'Dry':
        indexImage = 1
        if NA >= 1:
            raise ValueError('Wrong NA!')
    else:
        raise ValueError('Unsupported Lens Type')
   
    if mask.Orientation == 0:
        pass
    elif abs(mask.Orientation - torch.pi / 2) < 1e-6:
        tenpY = sourceData.Y.clone()
        sourceData.Y = -sourceData.X
        sourceData.X = tenpY
    else:
        raise ValueError('Not supported orientation angle')

    normalized_Frequency = projector.NA / source.Wavelength  
    mask_type = mask.MaskType.lower()
    if mask_type == '1d':
        SpectrumCalc = normalized_Frequency * mask.Period_X
        normalized_Period_X = mask.Period_X * normalized_Frequency  
        dfmdg = 1 / normalized_Period_X
    elif mask_type == '1dpixel':
        raise NotImplementedError('To be implemented')
    elif mask_type == '2d':
        SpectrumCalc = normalized_Frequency**2 * mask.Period_X * mask.Period_Y
        normalized_Period_X = mask.Period_X * normalized_Frequency  
        normalized_Period_Y = mask.Period_Y * normalized_Frequency  
        dfmdg = 1 / normalized_Period_X / normalized_Period_Y
    elif mask_type == '2dpixel':
        SpectrumCalc = normalized_Frequency**2 * mask.Period_X * mask.Period_Y
        normalized_Period_X = mask.Period_X * normalized_Frequency  
        normalized_Period_Y = mask.Period_Y * normalized_Frequency  
        dfmdg = 1 / normalized_Period_X / normalized_Period_Y
    elif mask_type == '3d':
        raise NotImplementedError('3D mask algorithm is not implemented yet')

    ExyzCalculateNumber = 1

    f0_s = sourceData.X
    g0_s = sourceData.Y

    fgSquare = f0_s**2 + g0_s**2

    rho_calc, theta_calc = cartesian_to_polar(f0_s, g0_s)

    obliquityFactor = torch.sqrt(torch.sqrt((1 - (M**2 * NA**2) * fgSquare) / (1 - (NA / indexImage)**2 * fgSquare)))

    aberration = projector.CalculateAberrationFast(rho_calc, theta_calc, 0)
    TempH0Aber = SpectrumCalc * obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration)

    obliqueRaysMatrix = torch.ones(len(rho_calc), ExyzCalculateNumber)

    if numerics.ImageCalculationMode.lower() == 'vector':
        ExyzCalculateNumber = 3
        nrs, nts = cartesian_to_polar(f0_s, g0_s)
        PolarizedX, PolarizedY = source.Calc_PolarizationMap(nts, nrs)
        alpha = (NA / indexImage) * f0_s
        beta = (NA / indexImage) * g0_s
        gamma = torch.sqrt(1 - (NA / indexImage)**2 * fgSquare)
        vectorRho2 = 1 - gamma**2
        Pxsx = (beta**2) / vectorRho2
        Pysx = (-alpha * beta) / vectorRho2
        Pxsy = Pysx
        Pysy = (alpha**2) / vectorRho2
        Pxpx = gamma * Pysy
        Pypx = -gamma * Pysx
        Pxpy = Pypx
        Pypy = gamma * Pxsx
        Pxpz = -alpha
        Pypz = -beta
        centerpoint = fgSquare < 1e-6
        if centerpoint.sum() > 1e-6:
            Pxsx[centerpoint] = 1
            Pysx[centerpoint] = 0
            Pxsy[centerpoint] = 0
            Pysy[centerpoint] = 0
            Pxpx[centerpoint] = 0
            Pypx[centerpoint] = 0
            Pxpy[centerpoint] = 0
            Pypy[centerpoint] = 1
            Pxpz[centerpoint] = 0
            Pypz[centerpoint] = 0
        M0xx = Pxsx + Pxpx
        M0yx = Pysx + Pypx
        M0xy = Pxsy + Pxpy
        M0yy = Pysy + Pypy
        M0xz = Pxpz
        M0yz = Pypz
        obliqueRaysMatrix = torch.ones(len(rho_calc), ExyzCalculateNumber)
        obliqueRaysMatrix[:, 0] = PolarizedX * M0xx + PolarizedY * M0yx
        obliqueRaysMatrix[:, 1] = PolarizedX * M0xy + PolarizedY * M0yy
        obliqueRaysMatrix[:, 2] = PolarizedX * M0xz + PolarizedY * M0yz
        
    TempFocus = -1j * 2 * torch.pi / wavelength * (indexImage - torch.sqrt(indexImage**2 - NA**2 * fgSquare))
    tempF = torch.exp(TempFocus * projector.Focus)
    intensityBlank = 0
    for iEM in range(ExyzCalculateNumber):
        ExyzFrequency = obliqueRaysMatrix[:, iEM] * TempH0Aber * tempF
        Exyz = ExyzFrequency
        IntensityCon = torch.abs(Exyz)**2
        IntensityTemp = indexImage * dfmdg**2 * torch.fft.fftshift(sourceData.Value.unsqueeze(0) @ IntensityCon.unsqueeze(1))
        intensityBlank = intensityBlank + IntensityTemp
        
    normalIntensity = intensityBlank / weight
    return normalIntensity