import torch
from CalculateCharacteristicMatrix import CalculateCharacteristicMatrix
from CalculateTCCMatrix import CalculateTCCMatrix
from DecomposeTCC_SOCS import DecomposeTCC_SOCS
from CalculateAerialImage_SOCS import CalculateAerialImage_SOCS, CalculateSOCS

class ImageData:
    Intensity: torch.tensor = None
    ImageX: torch.tensor = None
    ImageY: torch.tensor = None
    ImageZ: torch.tensor = None

def cartesian_to_polar(x, y):
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return rho, theta

def CalculateAbbeImage(source, scatter, projector, numerics):
    target_nf = scatter.ScatterField.shape[1]
    target_ng = scatter.ScatterField.shape[0]
    sourceData = source.Calc_SourceSimple()
    weight = torch.sum(sourceData.Value)
    wavelength = source.Wavelength
    NA = projector.NA
    M = projector.Magnification
    indexImage = projector.IndexImage
    spectrum, scatter_fs, scatter_gs, pupilImage = scatter.CalculateSpectrum(projector, source)
    SimulationRange = projector.FocusRange
    Intensity = torch.zeros(len(SimulationRange), target_ng, target_nf)
    for iFocus in range(len(SimulationRange)):
        scatter_fm, scatter_gm = torch.meshgrid(scatter_gs, scatter_fs, indexing='ij')
        intensity2D = torch.zeros(target_ng, target_nf, len(sourceData.Value))
        sourceX = sourceData.X
        sourceY = sourceData.Y
        sourceV = sourceData.Value

        focus = SimulationRange[iFocus]
        dfmdg = (scatter_fs[1] - scatter_fs[0]) * (scatter_gs[1] - scatter_gs[0])

        source_rho, source_theta = cartesian_to_polar(sourceX, sourceY)
        PolarizedX, PolarizedY = source.Calc_PolarizationMap(source_theta, source_rho)

        scatter_fg2m = scatter_fm ** 2 + scatter_gm ** 2
        sourceXY2 = sourceX ** 2 + sourceY ** 2

        for j in range(len(sourceData.Value)):
            obliqueRaysMatrix = torch.ones(1, 1, dtype=torch.complex64)
            ExyzCalculateNumber_2D = 1

            if numerics.ImageCalculationMode.lower() == 'vector':
                ExyzCalculateNumber_2D = 3
            elif numerics.ImageCalculationMode.lower() == 'scalar':
                ExyzCalculateNumber_2D = 1

            rho2 = (scatter_fg2m + 2 * (sourceX[j] * scatter_fm + sourceY[j] * scatter_gm) + sourceXY2[j]).to(torch.complex64)
            
            validPupil = torch.where(torch.real(rho2) <= 1)

            f_calc = scatter_fm[validPupil] + sourceX[j]
            g_calc = scatter_gm[validPupil] + sourceY[j]
            rho_calc, theta_calc = cartesian_to_polar(f_calc, g_calc)
            fgSquare = rho_calc**2

            obliquityFactor = torch.pow((1 - (NA/M)**2 * fgSquare) / (1 - (NA ** 2) * fgSquare), 0.25)

            aberration = projector.CalculateAberrationFast(rho_calc, theta_calc, 0)
            focusFactor = torch.exp(1j * 2 * torch.pi / wavelength * torch.sqrt(indexImage**2 - (NA/M)**2*fgSquare) * focus)
            if numerics.ImageCalculationMode == 'vector':
                obliqueRaysMatrix = torch.zeros(len(fgSquare), ExyzCalculateNumber_2D, dtype=torch.complex64)
                m0xx, m0yx, m0xy, m0yy, m0xz, m0yz = CalculateCharacteristicMatrix(f_calc, g_calc, NA, indexImage)
                    
                obliqueRaysMatrix[:, 0] = PolarizedX[j] * m0xx + PolarizedY[j] * m0yx
                obliqueRaysMatrix[:, 1] = PolarizedX[j] * m0xy + PolarizedY[j] * m0yy
                obliqueRaysMatrix[:, 2] = PolarizedX[j] * m0xz + PolarizedY[j] * m0yz
                    
            for channel in range(3):
                spectrum_channel = spectrum[:, :, channel].squeeze()
                SpectrumCalc = spectrum_channel[validPupil]
                TempHAber = SpectrumCalc * obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration) * focusFactor
                rho2[:] = 0
                intensityTemp = torch.zeros(target_ng, target_nf)
                for iEM in range(ExyzCalculateNumber_2D):
                    rho2[validPupil] = TempHAber * obliqueRaysMatrix[:, iEM]
                    ExyzFrequency = rho2
                    Exyz_Partial = torch.fft.fft2(ExyzFrequency)
                    intensityTemp = intensityTemp + torch.abs(Exyz_Partial) ** 2
                intensity2D[:, :, j] = intensity2D[:, :, j] + intensityTemp
        
        intensity2D = torch.reshape(sourceV, (1, 1, -1)) * intensity2D
        intensity2D = dfmdg ** 2 * torch.fft.fftshift(torch.sum(intensity2D, dim=2))
        intensity2D = torch.real(intensity2D)
        Intensity[iFocus, :, :] = indexImage / weight * intensity2D
    ImageX = torch.linspace(-scatter.Period_X/2, scatter.Period_X/2, target_nf)
    ImageY = torch.linspace(-scatter.Period_Y/2, scatter.Period_Y/2, target_ng)
    ImageZ = projector.FocusRange

    farfieldImage = ImageData()
    farfieldImage.Intensity = Intensity
    farfieldImage.ImageX = ImageX
    farfieldImage.ImageY = ImageY
    farfieldImage.ImageZ = ImageZ

    return farfieldImage, pupilImage

def CalculateHopkinsImage(source, scatter, projector, numerics):
    pitchxy = [scatter.Period_Y, scatter.Period_X]
    Nfg = [scatter.ScatterField.shape[1], scatter.ScatterField.shape[0]]

    SimulationRange = projector.FocusRange
    farfieldImage = ImageData()
    Intensity = torch.zeros(len(SimulationRange), scatter.ScatterField.shape[0], scatter.ScatterField.shape[1])
    for iFocus, focus in enumerate(SimulationRange):
        TCCMatrix_Stacked, FG_ValidSize = \
                        CalculateTCCMatrix(source, pitchxy, projector, focus, numerics)
        
        TCCMatrix_Kernel = \
                        DecomposeTCC_SOCS(TCCMatrix_Stacked, FG_ValidSize, Nfg, numerics)
        Intensity[iFocus, :, :], pupilImage = CalculateAerialImage_SOCS(scatter, TCCMatrix_Kernel, \
                                                                      source, projector)
        
    farfieldImage.Intensity = Intensity
    farfieldImage.ImageX = torch.linspace(-scatter.Period_X / 2,
                                            scatter.Period_X / 2,
                                            scatter.ScatterField.shape[1])
    farfieldImage.ImageY = torch.linspace(-scatter.Period_Y / 2,
                                           scatter.Period_Y / 2,
                                           scatter.ScatterField.shape[0])
    farfieldImage.ImageZ = projector.FocusRange

    return farfieldImage, pupilImage
def CalculateOptimized(source, scatter, projector, numerics):
    pitchxy = [scatter.Period_Y, scatter.Period_X]
    FloquetMode = scatter.FloquetMode
    Nfg = [FloquetMode.shape[0], FloquetMode.shape[1]]
    SimulationRange = projector.FocusRange
    farfieldImage = ImageData()
    Intensity = torch.zeros(len(SimulationRange), numerics.ScatterGrid_X, numerics.ScatterGrid_Y)
    for iFocus, focus in enumerate(SimulationRange):
        TCCMatrix_Stacked, FG_ValidSize = \
                        CalculateTCCMatrix(source, pitchxy, projector, focus, numerics)
        TCCMatrix_Kernel = \
                        DecomposeTCC_SOCS(TCCMatrix_Stacked, FG_ValidSize, Nfg, numerics)
        Intensity[iFocus, :, :] = CalculateSOCS(FloquetMode, TCCMatrix_Kernel, numerics)
    farfieldImage.Intensity = Intensity
    farfieldImage.ImageX = torch.linspace(-scatter.Period_X / 2,
                                            scatter.Period_X / 2,
                                            numerics.ScatterGrid_X)
    farfieldImage.ImageY = torch.linspace(-scatter.Period_Y / 2,
                                           scatter.Period_Y / 2,
                                           numerics.ScatterGrid_Y)
    farfieldImage.ImageZ = projector.FocusRange
    return farfieldImage, torch.tensor([])