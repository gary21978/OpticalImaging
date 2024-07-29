import torch

def CalculateCharacteristicMatrix(f_calc, g_calc, NA, indexImage):
    # Calculate the directional cosines based on numerical aperture (NA) and index of the medium (indexImage)
    alpha = (NA / indexImage) * f_calc
    beta = (NA / indexImage) * g_calc
    gamma = torch.sqrt(torch.complex(1 - alpha**2 - beta**2, torch.tensor(0.0)))
    
    Mxx = 1 - alpha**2 / (1 + gamma)
    Myx = -alpha * beta / (1 + gamma)
    Mxy = Myx
    Myy = 1 - beta**2 / (1 + gamma)
    Mxz = torch.complex(-alpha, torch.zeros_like(alpha))
    Myz = torch.complex(-beta, torch.zeros_like(beta))

    return Mxx, Myx, Mxy, Myy, Mxz, Myz

