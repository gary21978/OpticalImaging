import torch
from CalculateCharacteristicMatrix import CalculateCharacteristicMatrix

def cartesian_to_polar(x, y):
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return rho, theta

def CalculateTCCMatrix(source, pitchxy, projector, recipe, numerics):
    indexImage = projector.IndexImage
    wavelength = source.Wavelength
    xPitch = pitchxy[0]
    yPitch = pitchxy[1]

    # Calculate shifted pupil function
    if numerics.ImageCalculationMode == 'scalar':
        SP, F_Valid, G_Valid, sourceData = CalculateShiftedPupilS(wavelength, projector, source, xPitch, yPitch, indexImage, recipe.Focus)
        TCCMatrix_Stacked = GetTCCMatrix(sourceData, SP)
    elif numerics.ImageCalculationMode == 'vector':
        Gx, Gy, Gz, F_Valid, G_Valid, sourceData = CalculateShiftedPupilV(wavelength, projector, source, xPitch, yPitch, indexImage, recipe.Focus)

        TCCMatrixX = GetTCCMatrix(sourceData, Gx)
        TCCMatrixY = GetTCCMatrix(sourceData, Gy)
        TCCMatrixZ = GetTCCMatrix(sourceData, Gz)
        TCCMatrix_Stacked = TCCMatrixX + TCCMatrixY + TCCMatrixZ
    TCCMatrix_Stacked = projector.IndexImage * TCCMatrix_Stacked
    FG_ValidSize = [len(G_Valid), len(F_Valid)]
    
    return TCCMatrix_Stacked, FG_ValidSize

def CalculateShiftedPupilS(wavelength, projector, source, xPitch, yPitch, indexImage, focus):
    sourceData = source.Calc_SourceSimple()  # You need to implement this function
    M = projector.Reduction
    NA = projector.NA
    normalized_xPitch = torch.tensor(xPitch / (wavelength / NA))
    normalized_yPitch = torch.tensor(yPitch / (wavelength / NA))
    Nf = torch.ceil(2 * normalized_xPitch).int()
    Ng = torch.ceil(2 * normalized_yPitch).int()
    f = (1 / normalized_xPitch) * torch.arange(-Nf, Nf + 1)
    g = (1 / normalized_yPitch) * torch.arange(-Ng, Ng + 1)
    ff, gg = torch.meshgrid(f, g, indexing='ij')

    new_f = ff.reshape(-1, 1) + sourceData.X.reshape(1, -1)
    new_g = gg.reshape(-1, 1) + sourceData.Y.reshape(1, -1)
    rho, theta = cartesian_to_polar(new_f, new_g)
    rhoSquare = rho.pow(2)
    validPupil = (rho <= 1)
    validRho = rho[validPupil]
    validTheta = theta[validPupil]
    validRhoSquare = rhoSquare[validPupil]
    obliquityFactor = torch.sqrt(torch.sqrt((1 - (M ** 2 * projector.NA ** 2) * validRhoSquare) / (1 - ((projector.NA / indexImage) ** 2) * validRhoSquare)))
    Orientation = 0
    aberration = projector.CalculateAberrationFast(validRho, validTheta, Orientation)
    shiftedPupil = torch.zeros(validPupil.size()).to(torch.complex64)
    TempFocus = 1j * 2 * torch.pi / wavelength * (indexImage - torch.sqrt(indexImage ** 2 - NA ** 2 * validRhoSquare))
    shiftedPupil[validPupil] = obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration) * torch.exp(TempFocus * focus)
    shiftedPupil = shiftedPupil.to(torch.complex64)
    return shiftedPupil, f, g, sourceData

def CalculateShiftedPupilV(wavelength, projector, source, xPitch, yPitch, indexImage, focus):
    sourceData = source.Calc_SourceSimple()  # You need to implement this function
    M = projector.Reduction
    NA = projector.NA
    normalized_xPitch = torch.tensor(xPitch / (wavelength / NA))
    normalized_yPitch = torch.tensor(yPitch / (wavelength / NA))
    Nf = torch.ceil(2 * normalized_xPitch).int()
    Ng = torch.ceil(2 * normalized_yPitch).int()
    f = (1 / normalized_xPitch) * torch.arange(-Nf, Nf + 1)
    g = (1 / normalized_yPitch) * torch.arange(-Ng, Ng + 1)
    ff, gg = torch.meshgrid(f, g, indexing='ij')

    new_f = ff.reshape(-1, 1) + sourceData.X.reshape(1, -1)
    new_g = gg.reshape(-1, 1) + sourceData.Y.reshape(1, -1)
    rho, theta = cartesian_to_polar(new_f, new_g)
    rhoSquare = rho.pow(2)
    validPupil = (rho <= 1)
    validRho = rho[validPupil]
    validTheta = theta[validPupil]
    validRhoSquare = rhoSquare[validPupil]
    obliquityFactor = torch.sqrt(torch.sqrt((1 - (M ** 2 * projector.NA ** 2) * validRhoSquare) / (1 - ((projector.NA / indexImage) ** 2) * validRhoSquare)))
    Orientation = 0
    aberration = projector.CalculateAberrationFast(validRho, validTheta, Orientation)
    shiftedPupil = torch.zeros(validPupil.size()).to(torch.complex64)
    TempFocus = 1j * 2 * torch.pi / wavelength * (indexImage - torch.sqrt(indexImage ** 2 - NA ** 2 * validRhoSquare))
    shiftedPupil[validPupil] = obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration) * torch.exp(TempFocus * focus)

    M0xx, M0yx, M0xy, M0yy, M0xz, M0yz = CalculateCharacteristicMatrix(new_f, new_g, rhoSquare, NA, indexImage)
    rho_s, theta_s = cartesian_to_polar(sourceData.X, sourceData.Y)
    PolarizedX, PolarizedY = source.Calc_PolarizationMap(theta_s, rho_s)

    Gx = (PolarizedX * M0xx + PolarizedY * M0yx) * shiftedPupil
    Gy = (PolarizedX * M0xy + PolarizedY * M0yy) * shiftedPupil
    Gz = (PolarizedX * M0xz + PolarizedY * M0yz) * shiftedPupil

    return Gx, Gy, Gz, f, g, sourceData

def GetTCCMatrix(sourceData, shiftedPupil):
    n = sourceData.Value.size(0)  # Get the number of elements
    i = torch.arange(n)  # Create a tensor for row indices
    j = i.clone()  # Create a tensor for column indices (assuming it's a square matrix)
    # Convert i and j to tensors with dtype=torch.long
    
    # Create a sparse COO tensor
    S = torch.sparse_coo_tensor(
        torch.stack((i, j)),  # indices
        torch.complex(sourceData.Value, torch.zeros_like(sourceData.Value)),             # values
        size=(n, n),  # size of the sparse tensor
    )
    # Perform matrix operations
    TCCMatrix = torch.matmul(torch.matmul(shiftedPupil, S), shiftedPupil.t())  # Utilizing matrix conjugate transpose to get HSH*
    # Normalize the entire matrix by dividing it by the sum of all elements
    sum_value = torch.sum(sourceData.Value)
    TCCMatrix = TCCMatrix / sum_value
    return TCCMatrix