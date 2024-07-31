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

def CalculateAbbeImage(source, mask, projector, numerics):
    target_nf = mask.Feature.shape[1]
    target_ng = mask.Feature.shape[0]

    source.PntNum = numerics.SampleNumber_Source
    sourceData = source.Calc_SourceSimple()
    weight = torch.sum(sourceData.Value)
    wavelength = source.Wavelength
    NA = projector.NA
    M = projector.Magnification
    indexImage = projector.IndexImage
    spectrum, mask_fs, mask_gs = mask.CalculateMaskSpectrum(projector, source)

    SimulationRange = projector.FocusRange
    Intensity = torch.zeros(len(SimulationRange), target_nf, target_ng)

    for iFocus in range(len(SimulationRange)):
        mask_fm, mask_gm = torch.meshgrid(mask_gs[:-1], mask_fs[:-1], indexing='ij')
        intensity2D = torch.zeros(target_ng - 1, target_nf - 1, len(sourceData.Value))
        sourceX = sourceData.X
        sourceY = sourceData.Y
        sourceV = sourceData.Value

        focus = SimulationRange[iFocus]
        dfmdg = (mask_fs[1] - mask_fs[0]) * (mask_gs[1] - mask_gs[0])

        source_rho, source_theta = cartesian_to_polar(sourceX, sourceY)
        PolarizedX, PolarizedY = source.Calc_PolarizationMap(source_theta, source_rho)
        new_spectrum = spectrum[:-1, :-1]
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
            fgSquare = rho_calc**2

            obliquityFactor = torch.pow((1 - (M ** 2 * NA ** 2) * fgSquare) / (1 - ((NA / indexImage) ** 2) * fgSquare), 0.25)

            aberration = projector.CalculateAberrationFast(rho_calc, theta_calc, 0)
            focusFactor = torch.exp(-1j * 2 * torch.pi / wavelength * (indexImage - torch.sqrt(indexImage ** 2 - NA * NA * fgSquare)) * focus)
            SpectrumCalc = new_spectrum[validPupil]

            TempHAber = SpectrumCalc * obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration) * focusFactor
            
            if numerics.ImageCalculationMode == 'vector':
                obliqueRaysMatrix = torch.zeros(len(fgSquare), ExyzCalculateNumber_2D, dtype=torch.complex64)
                m0xx, m0yx, m0xy, m0yy, m0xz, m0yz = CalculateCharacteristicMatrix(f_calc, g_calc, NA, indexImage)
                
                obliqueRaysMatrix[:, 0] = PolarizedX[j] * m0xx + PolarizedY[j] * m0yx
                obliqueRaysMatrix[:, 1] = PolarizedX[j] * m0xy + PolarizedY[j] * m0yy
                obliqueRaysMatrix[:, 2] = PolarizedX[j] * m0xz + PolarizedY[j] * m0yz
                
            rho2[:] = 0
            intensityTemp = torch.zeros(target_ng - 1, target_nf - 1)
            
            for iEM in range(ExyzCalculateNumber_2D):
                rho2[validPupil] = TempHAber * obliqueRaysMatrix[:, iEM]
                ExyzFrequency = rho2
                Exyz_Partial = torch.fft.fft2(ExyzFrequency)
                intensityTemp = intensityTemp + torch.abs(Exyz_Partial) ** 2

            intensity2D[:, :, j] = intensityTemp
        
        intensity2D = torch.reshape(sourceV, (1, 1, -1)) * intensity2D
        intensity2D = dfmdg ** 2 * torch.fft.fftshift(torch.sum(intensity2D, dim=2))
        intensity2D = torch.cat((intensity2D, intensity2D[:, 0].unsqueeze(1)), 1)
        intensity2D = torch.cat((intensity2D, intensity2D[0, :].unsqueeze(0)), 0)
        intensity2D = torch.real(torch.rot90(intensity2D, 2))
        Intensity[iFocus, :, :] = indexImage / weight * torch.transpose(intensity2D, 0, 1)
    ImageX = torch.linspace(-mask.Period_X/2, mask.Period_X/2, target_nf)
    ImageY = torch.linspace(-mask.Period_Y/2, mask.Period_Y/2, target_ng)
    ImageZ = projector.FocusRange

    farfieldImage = ImageData()
    farfieldImage.Intensity = Intensity
    farfieldImage.ImageX = ImageX
    farfieldImage.ImageY = ImageY
    farfieldImage.ImageZ = ImageZ

    return farfieldImage

def CalculateHopkinsImage(source, mask, projector, numerics):
    pitchxy = [mask.Period_Y, mask.Period_X]
    Nfg = [mask.Feature.shape[1], mask.Feature.shape[0]]

    SimulationRange = projector.FocusRange
    farfieldImage = ImageData()
    Intensity = torch.zeros(len(SimulationRange), mask.Feature.shape[1], mask.Feature.shape[0])
    for iFocus, focus in enumerate(SimulationRange):
        TCCMatrix_Stacked, FG_ValidSize = \
                        CalculateTCCMatrix(source, pitchxy, projector, focus, numerics)
        
        TCCMatrix_Kernel = \
                        DecomposeTCC_SOCS(TCCMatrix_Stacked, FG_ValidSize, Nfg, numerics)
        
        """
        # Visualize kernels
        kernels = TCCMatrix_Kernel.permute(2, 1, 0)
        spatial = torch.fft.fft2(kernels)
        spatial = torch.abs(torch.fft.fftshift(spatial, dim=(1,2)))
        import matplotlib.pyplot as plt
        for i in range(4):
            for j in range(5):
                plt.subplot(5, 4, i * 5 + j + 1)
                plt.imshow(spatial[i * 5 + j, :, :].detach().numpy(), cmap='jet')
                plt.colorbar()
        plt.show()
        """
        
        Intensity[iFocus, :, :] = CalculateAerialImage_SOCS(mask, TCCMatrix_Kernel, \
                                                            source, projector)
        
    farfieldImage.Intensity = Intensity
    farfieldImage.ImageX = torch.linspace(-mask.Period_X / 2,
                                            mask.Period_X / 2,
                                            mask.Feature.shape[1])
    farfieldImage.ImageY = torch.linspace(-mask.Period_Y / 2,
                                           mask.Period_Y / 2,
                                           mask.Feature.shape[0])
    farfieldImage.ImageZ = projector.FocusRange

    return farfieldImage