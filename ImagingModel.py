from Numerics import Numerics
from Source import Source
from Mask import Mask
from Projection import Projection
from CalculateAerialImage import CalculateAbbeImage, CalculateHopkinsImage
from CalculateNormalImage import CalculateNormalImage
import matplotlib.pyplot as plt
import torch

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
        
        #if nm.Normalization_Intensity:
        #    ni = CalculateNormalImage(sr, mk, po, nm)
        #    ali.Intensity = ali.Intensity / ni
        return ali

def compareAbbeHopkins():
    im = ImagingModel()
    im.Mask.Feature = torch.zeros((81, 81))
    im.Mask.Feature[20:40, 10:15] = 1
    im.Mask.Feature[20:40, 70:75] = 1

    im.Numerics.ImageCalculationMethod = "abbe"
    intensity_Abbe = im.CalculateAerialImage().Intensity.detach().numpy()
    im.Numerics.ImageCalculationMethod = "hopkins"
    intensity_Hopkins = im.CalculateAerialImage().Intensity.detach().numpy()

    NFocus = intensity_Abbe.shape[0]
    for i in range(NFocus):
        plt.subplot(NFocus, 3, 3*i+1)
        plt.imshow(intensity_Abbe[i, :, :].squeeze(), cmap='jet')
        plt.title("Abbe")
        plt.colorbar()

        plt.subplot(NFocus, 3, 3*i+2)
        plt.imshow(intensity_Hopkins[i, :, :].squeeze(), cmap='jet')
        plt.title("Hopkins")
        plt.colorbar()

        plt.subplot(NFocus, 3, 3*i+3)
        plt.imshow(intensity_Hopkins[i, :, :].squeeze() - intensity_Abbe[i, :, :].squeeze(), cmap='jet')
        plt.title("Diff")
        plt.colorbar()
    
    plt.show()
    
if __name__ == '__main__':
    compareAbbeHopkins()
