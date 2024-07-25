import torch

def CalculateCharacteristicMatrix(f_calc, g_calc, fgSquare, NA, indexImage):
    # Calculate the scaling factors based on numerical aperture (NA) and index of the medium (indexImage)
    alpha = (NA / indexImage) * f_calc
    beta = (NA / indexImage) * g_calc
    
    # Calculate the gamma factor using the numerical aperture and index of the medium
    gamma = torch.sqrt(torch.complex(1 - (NA / indexImage)**2 * fgSquare, torch.tensor(0.0)))
    
    # Calculate vectorRho2, used for some matrix calculations
    vectorRho2 = 1 - gamma**2

    # Calculate the elements of the characteristic matrix
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

    # Find the center point where fgSquare is close to zero
    centerpoint = (fgSquare < torch.finfo(float).eps).nonzero(as_tuple=True)
    
    if len(centerpoint[0]) > torch.finfo(float).eps:
        # Set matrix elements to zero at the center point
        print(centerpoint)
        Pxsx[centerpoint] = 0
        Pysx[centerpoint] = 0
        Pxsy[centerpoint] = 0
        Pysy[centerpoint] = 0
        Pxpx[centerpoint] = 0
        Pypx[centerpoint] = 0
        Pxpy[centerpoint] = 0
        Pypy[centerpoint] = 0
        Pxpz[centerpoint] = 0
        Pypz[centerpoint] = 0
    # Calculate the components of the characteristic matrix
    Mxx = Pxsx + Pxpx
    Myx = Pysx + Pypx
    Mxy = Pxsy + Pxpy
    Myy = Pysy + Pypy
    Mxz = torch.complex(Pxpz, torch.zeros_like(Pxpz))
    Myz = torch.complex(Pypz, torch.zeros_like(Pypz))

    # Coefficient match ??? #TODO Eli
    #Mxx = 1 - alpha**2 / (1 + gamma)
    #Myx = -alpha * beta / (1 + gamma)
    #Mxy = Myx
    #Myy = 1 - beta**2 / (1 + gamma)
    #Mxz = torch.complex(-alpha, torch.zeros_like(alpha))
    #Myz = torch.complex(-beta, torch.zeros_like(beta))

    # Return the calculated characteristic matrix components
    return Mxx, Myx, Mxy, Myy, Mxz, Myz

