from Numerics import Numerics
from Source import Source
from Scatter import Scatter
from Projection import Projection
from CalculateAerialImage import CalculateAbbeImage, CalculateHopkinsImage
import matplotlib.pyplot as plt
import torch

def createGeometry():
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    image = np.zeros((51, 51), dtype=np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 32)
    text = "T"
    draw.text((15, 10), text, fill=1, font=font)
    P = np.transpose(np.array(image, dtype=np.int8))
    return torch.from_numpy(P)

class ImagingModel:
    def __init__(self):
        self.Numerics = Numerics()
        self.Source = Source()
        self.Scatter = Scatter()
        self.Projector = Projection()

    def CalculateAerialImage(self):
        sr = self.Source
        sc = self.Scatter
        po = self.Projector
        nm = self.Numerics
        if (nm.ImageCalculationMethod == 'abbe'):
            ali = CalculateAbbeImage(sr, sc, po, nm)
        elif (nm.ImageCalculationMethod == 'hopkins'):
            ali = CalculateHopkinsImage(sr, sc, po, nm)
        else:
            raise ValueError('Unsupported Calculation Method')
        return ali

def compareAbbeHopkins():
    im = ImagingModel()
    geometry = createGeometry().to(torch.complex64)
    scatter_field = torch.stack((geometry, 1j*geometry, 0.3*geometry), 2)
    im.Scatter.ScatterField = scatter_field
    ############################################
    im.Scatter.Period_X = 2000
    im.Scatter.Period_Y = 2000
    im.Source.Wavelength = 365
    im.Source.PntNum = 41
    im.Source.Shape = "annular"
    im.Source.PolarizationType = "c_pol"
    im.Source.SigmaOut = 0.9
    im.Source.SigmaIn = 0.0
    im.Projector.Aberration_Zernike = torch.zeros(37)
    im.Projector.Magnification = 100.0
    im.Projector.NA = 0.9
    im.Projector.IndexImage = 1.0
    im.Projector.FocusRange = torch.tensor([-5e+6, 0, 5e+6])
    im.Numerics.ImageCalculationMode = "scalar"
    im.Numerics.ImageCalculationMethod = "abbe"
    im.Numerics.Hopkins_SettingType = "order"
    im.Numerics.Hopkins_Order = 100
    im.Numerics.Hopkins_Threshold = 0.95
    ############################################

    Lx = im.Scatter.Period_X
    Ly = im.Scatter.Period_Y
    M = im.Projector.Magnification
    scatter_field = im.Scatter.ScatterField.detach().numpy()
    Ex = scatter_field[:,:,0].squeeze().transpose().real
    im.Numerics.ImageCalculationMethod = "abbe"
    intensity_Abbe = im.CalculateAerialImage().Intensity.detach().numpy()
    im.Numerics.ImageCalculationMethod = "hopkins"
    intensity_Hopkins = im.CalculateAerialImage().Intensity.detach().numpy()

    NFocus = intensity_Abbe.shape[0]
    for i in range(NFocus):
        plt.subplot(NFocus, 4, 4*i+1)
        plt.imshow(Ex, cmap='gray', interpolation='none', \
                   extent = (0, 0.001*Lx, 0, 0.001*Ly))
        plt.title("Real(Ex)")
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.colorbar()

        plt.subplot(NFocus, 4, 4*i+2)
        plt.imshow(intensity_Abbe[i, :, :].squeeze(), cmap='jet', \
                   extent = (0, 0.001*Lx*M, 0, 0.001*Ly*M))
        plt.title("Abbe")
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.colorbar()

        plt.subplot(NFocus, 4, 4*i+3)
        plt.imshow(intensity_Hopkins[i, :, :].squeeze(), cmap='jet', \
                   extent = (0, 0.001*Lx*M, 0, 0.001*Ly*M))
        plt.title("Hopkins")
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.colorbar()

        plt.subplot(NFocus, 4, 4*i+4)
        plt.imshow(intensity_Hopkins[i, :, :].squeeze() - intensity_Abbe[i, :, :].squeeze(), cmap='jet', \
                   extent = (0, 0.001*Lx*M, 0, 0.001*Ly*M))
        plt.title("Diff")
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.colorbar()
    
    plt.show()
    
if __name__ == '__main__':
    compareAbbeHopkins()
