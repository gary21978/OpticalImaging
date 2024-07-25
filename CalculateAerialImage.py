import torch
from CalculateCharacteristicMatrix import CalculateCharacteristicMatrix
from CalculateTCCMatrix import CalculateTCCMatrix
from DecomposeTCC_SOCS import DecomposeTCC_SOCS
from CalculateAerialImage_SOCS import CalculateAerialImage_SOCS
from ImageData import ImageData

def cartesian_to_polar(x, y):
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return rho, theta

def CalculateAbbeImage(source, mask, projector, recipe, numerics):
    mask_nf = numerics.SampleNumber_Mask_X
    mask_ng = numerics.SampleNumber_Mask_Y
    wafer_nf = numerics.SampleNumber_Wafer_X
    wafer_ng = numerics.SampleNumber_Wafer_Y

    source.PntNum = numerics.SampleNumber_Source
    sourceData = source.Calc_SourceSimple()
    weight = torch.sum(sourceData.Value)
    wavelength = source.Wavelength
    NA = projector.NA
    M = projector.Reduction
    indexImage = projector.IndexImage

    mask.Nf = mask_nf
    mask.Ng = mask_ng
    spectrum, mask_fs, mask_gs, _ = mask.CalculateMaskSpectrum(projector, source)

    SimulationRange = recipe.FocusRange - recipe.Focus
    Intensity = torch.zeros(len(SimulationRange), wafer_nf, wafer_ng)
    Orientation = mask.Orientation

    for iFocus in range(len(SimulationRange)):
        mask_fm, mask_gm = torch.meshgrid(mask_fs[:-1], mask_gs[:-1], indexing='ij')
        intensity2D = torch.zeros(wafer_nf - 1, wafer_ng - 1, len(sourceData.Value))
        sourceX = sourceData.X
        sourceY = sourceData.Y
        sourceV = sourceData.Value

        focus = SimulationRange[iFocus]
        dfmdg = (mask_fs[1] - mask_fs[0]) * (mask_gs[1] - mask_gs[0])

        source_rho, source_theta = cartesian_to_polar(sourceX, sourceY)
        PolarizedX, PolarizedY = source.Calc_PolarizationMap(source_theta, source_rho)
        new_spectrum = spectrum[:-1, :-1] # Discard last row and column??
        mask_fg2m = mask_fm ** 2 + mask_gm ** 2
        sourceXY2 = sourceX ** 2 + sourceY ** 2

        for j in range(len(sourceData.Value)):
            obliqueRaysMatrix = torch.ones(1, 1, dtype=torch.complex64)
            ExyzCalculateNumber_2D = 1

            if numerics.ImageCalculationMode.lower() == 'vector':
                ExyzCalculateNumber_2D = 3
            elif numerics.ImageCalculationMode.lower() == 'scalar':
                ExyzCalculateNumber_2D = 1

            rho2 = (mask_fg2m + 2 * (sourceX[j] * mask_fm + sourceY[j] * mask_gm) + sourceXY2[j]).to(torch.complex64)
            validPupil = torch.where(torch.real(rho2) <= 1)

            f_calc = mask_fm[validPupil] + sourceX[j]
            g_calc = mask_gm[validPupil] + sourceY[j]
            rho_calc, theta_calc = cartesian_to_polar(f_calc, g_calc)
            fgSquare = rho_calc ** 2

            obliquityFactor = torch.sqrt(torch.sqrt(
                (1 - (M ** 2 * NA ** 2) * fgSquare) / (1 - ((NA / indexImage) ** 2) * fgSquare)))
            
            aberration = projector.CalculateAberrationFast(rho_calc, theta_calc, Orientation)
            pupilFilter = projector.CalculatePupilFilter(rho_calc, theta_calc)
            focusFactor = torch.exp(-1j * 2 * torch.pi / wavelength * (indexImage - torch.sqrt(indexImage ** 2 - NA * NA * fgSquare)) * focus)
            SpectrumCalc = new_spectrum[validPupil]

            TempHAber = SpectrumCalc * obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration) * pupilFilter * focusFactor
            
            if numerics.ImageCalculationMode == 'vector':
                obliqueRaysMatrix = torch.zeros(len(fgSquare), ExyzCalculateNumber_2D, dtype=torch.complex64)
                m0xx, m0yx, m0xy, m0yy, m0xz, m0yz = CalculateCharacteristicMatrix(f_calc, g_calc, NA, indexImage)
                
                obliqueRaysMatrix[:, 0] = PolarizedX[j] * m0xx + PolarizedY[j] * m0yx
                obliqueRaysMatrix[:, 1] = PolarizedX[j] * m0xy + PolarizedY[j] * m0yy
                obliqueRaysMatrix[:, 2] = PolarizedX[j] * m0xz + PolarizedY[j] * m0yz
                
            rho2[:] = 0
            intensityTemp = torch.zeros(wafer_nf - 1, wafer_ng - 1)
            
            for iEM in range(ExyzCalculateNumber_2D):
                rho2[validPupil] = TempHAber * obliqueRaysMatrix[:, iEM]

                if wafer_nf == mask_nf and wafer_ng == mask_ng:
                    ExyzFrequency = rho2
                else:
                    ExyzFrequency = torch.zeros(wafer_nf - 1, wafer_ng - 1)
                    if wafer_nf > mask_nf:
                        rangeWaferNf = torch.arange((wafer_nf - mask_nf + 2) // 2, (wafer_nf + mask_nf - 2) // 2)
                        rangeMaskNf = torch.arange(0, mask_nf - 1)
                    else:
                        rangeWaferNf = torch.arange(0, wafer_nf - 1)
                        rangeMaskNf = torch.arange((mask_nf - wafer_nf + 2) // 2, (wafer_nf + mask_nf - 2) // 2)

                    if wafer_ng > mask_ng:
                        rangeWaferNg = torch.arange((wafer_ng - mask_ng + 2) // 2, (wafer_ng + mask_ng - 2) // 2)
                        rangeMaskNg = torch.arange(0, mask_ng - 1)
                    else:
                        rangeWaferNg = torch.arange(0, wafer_ng - 1)
                        rangeMaskNg = torch.arange((mask_ng - wafer_ng + 2) // 2, (wafer_ng + mask_ng - 2) // 2)

                    ExyzFrequency[rangeWaferNf, rangeWaferNg] = rho2[rangeMaskNf, rangeMaskNg]

                Exyz_Partial = torch.fft.fft2(ExyzFrequency)
                intensityTemp = intensityTemp + torch.abs(Exyz_Partial) ** 2

            intensity2D[:, :, j] = intensityTemp
        
        intensity2D = torch.reshape(sourceV, (1, 1, -1)) * intensity2D
        intensity2D = dfmdg ** 2 * torch.fft.fftshift(torch.sum(intensity2D, dim=2))
        intensity2D = torch.cat((intensity2D, intensity2D[:, 0].unsqueeze(1)), 1)
        intensity2D = torch.cat((intensity2D, intensity2D[0, :].unsqueeze(0)), 0)
        intensity2D = torch.real(torch.rot90(intensity2D, 2))
        Intensity[iFocus, :, :] = indexImage / weight * torch.transpose(intensity2D, 0, 1)
    ImageX = torch.linspace(-mask.Period_X/2, mask.Period_X/2, wafer_nf)
    ImageY = torch.linspace(-mask.Period_Y/2, mask.Period_Y/2, wafer_ng)
    ImageZ = recipe.FocusRange

    farfieldImage = ImageData()
    farfieldImage.Intensity = projector.IndexImage*Intensity
    farfieldImage.ImageX = ImageX
    farfieldImage.ImageY = ImageY
    farfieldImage.ImageZ = ImageZ

    return farfieldImage

def CalculateHopkinsImage(source, mask, projector, recipe, numerics):
    pitchxy = [mask.Period_X, mask.Period_Y]
    SimulationRange = recipe.FocusRange - recipe.Focus
    farfieldImage = ImageData()
    Intensity = torch.zeros(len(SimulationRange), numerics.SampleNumber_Wafer_X, numerics.SampleNumber_Wafer_Y)
    for iFocus, focus in enumerate(SimulationRange):
        TCCMatrix_Stacked, FG_ValidSize = \
                        CalculateTCCMatrix(source, pitchxy, projector, focus, numerics)
        
        TCCMatrix_Kernel = \
                        DecomposeTCC_SOCS(TCCMatrix_Stacked, FG_ValidSize, numerics)
        
        
        Intensity[iFocus, :, :] = CalculateAerialImage_SOCS(mask, TCCMatrix_Kernel, \
                                                source, projector, numerics)
        
    farfieldImage.Intensity = Intensity
    farfieldImage.ImageX = torch.linspace(-mask.Period_X / 2,
                                            mask.Period_X / 2,
                                            numerics.SampleNumber_Wafer_X)
    farfieldImage.ImageY = torch.linspace(-mask.Period_Y / 2,
                                            mask.Period_Y / 2,
                                            numerics.SampleNumber_Wafer_Y)
    farfieldImage.ImageZ = recipe.FocusRange

    return farfieldImage