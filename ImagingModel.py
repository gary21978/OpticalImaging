from Numerics import Numerics
from Source import Source
from Mask import Mask
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
    text = "S"
    draw.text((15, 10), text, fill=1, font=font)
    P = np.transpose(np.array(image, dtype=np.int8))
    return torch.from_numpy(P)

class ImagingModel:
    def __init__(self):
        self.Numerics = Numerics()
        self.Source = Source()
        self.Mask = Mask()
        self.Projector = Projection()

    def CalculateAerialImage(self):
        sr = self.Source
        mk = self.Mask
        po = self.Projector
        nm = self.Numerics
        if (nm.ImageCalculationMethod == 'abbe'):
            ali = CalculateAbbeImage(sr, mk, po, nm)
        elif (nm.ImageCalculationMethod == 'hopkins'):
            ali = CalculateHopkinsImage(sr, mk, po, nm)
        else:
            raise ValueError('Unsupported Calculation Method')
        return ali

def compareAbbeHopkins():
    im = ImagingModel()
    im.Mask.Feature = createGeometry()

    ############################################
    im.Mask.Period_X = 2000
    im.Mask.Period_Y = 2000
    im.Source.Wavelength = 365
    im.Source.PntNum = 41
    im.Source.Shape = "annular"
    im.Source.PolarizationType = 'c_pol'
    im.Source.SigmaOut = 0.9
    im.Source.SigmaIn = 0.0
    im.Projector.Aberration_Zernike = torch.zeros(37)
    im.Projector.Magnification = 100.0
    im.Projector.NA = 0.9
    im.Projector.IndexImage = 1.0
    im.Projector.FocusRange = torch.tensor([0])
    im.Numerics.ImageCalculationMode = "scalar"
    im.Numerics.ImageCalculationMethod = "abbe"
    im.Numerics.Hopkins_SettingType = 'order'
    im.Numerics.Hopkins_Order = 50
    im.Numerics.Hopkins_Threshold = 0.95
    ############################################

    Lx = im.Mask.Period_X
    Ly = im.Mask.Period_Y
    M = im.Projector.Magnification
    mask = im.Mask.Feature.detach().numpy().squeeze().transpose()
    im.Numerics.ImageCalculationMethod = "abbe"
    intensity_Abbe = im.CalculateAerialImage().Intensity.detach().numpy()
    im.Numerics.ImageCalculationMethod = "hopkins"
    intensity_Hopkins = im.CalculateAerialImage().Intensity.detach().numpy()

    NFocus = intensity_Abbe.shape[0]
    for i in range(NFocus):
        plt.subplot(NFocus, 4, 4*i+1)
        plt.imshow(mask, cmap='gray', interpolation='none', \
                   extent = (0, 0.001*Lx, 0, 0.001*Ly))
        plt.title("Mask")
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
