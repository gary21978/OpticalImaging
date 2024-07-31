import torch

def CalculateCharacteristicMatrix(f_calc, g_calc, NA, indexImage):
    # Calculate the directional cosines based on numerical aperture (NA) and index of the medium (indexImage)
    alpha = torch.complex((NA / indexImage) * f_calc, torch.zeros_like(f_calc))
    beta = torch.complex((NA / indexImage) * g_calc, torch.zeros_like(g_calc))
    gamma = torch.sqrt(1 - alpha**2 - beta**2)
    
    Mxx = 1 - alpha**2 / (1 + gamma)
    Myx = -alpha * beta / (1 + gamma)
    Mxy = Myx
    Myy = 1 - beta**2 / (1 + gamma)
    Mxz = -alpha
    Myz = -beta

    return Mxx, Myx, Mxy, Myy, Mxz, Myz

