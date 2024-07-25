from Numerics import Numerics
from Source import Source
from Recipe import Recipe
from Mask import Mask
from ProjectionObjective import ProjectionObjective
from CalculateAerialImage import CalculateAbbeImage, CalculateHopkinsImage
from CalculateNormalImage import CalculateNormalImage
import matplotlib.pyplot as plt

class ImagingModel:
    def __init__(self):
        self.Numerics = Numerics()
        self.Source = Source()
        self.Mask = Mask.CreateMask('crossgate')
        self.Projector = ProjectionObjective()
        self.Receipe = Recipe()

    def CalculateAerialImage(self):
        sr = self.Source
        mk = self.Mask
        po = self.Projector
        rp = self.Receipe
        nm = self.Numerics
        if (nm.ImageCalculationMethod == 'abbe'):
            ali = CalculateAbbeImage(sr, mk, po, rp, nm)
        elif (nm.ImageCalculationMethod == 'hopkins'):
            ali = CalculateHopkinsImage(sr, mk, po, rp, nm)
        else:
            raise ValueError('Unsupported Calculation Method')
        
        if nm.Normalization_Intensity:
            ni = CalculateNormalImage(sr, mk, po, rp, nm)
            ali.Intensity = ali.Intensity / ni
        return ali

def check():
    im = ImagingModel()
    im.Mask = Mask.CreateMask('crossgate')

    im.Numerics.ImageCalculationMethod = "abbe"
    intensity_Abbe = im.CalculateAerialImage().Intensity.detach().numpy().squeeze()

    im.Numerics.ImageCalculationMethod = "hopkins"
    intensity_Hopkins = im.CalculateAerialImage().Intensity.detach().numpy().squeeze()

    plt.subplot(1, 3, 1)
    plt.imshow(intensity_Abbe, cmap='gray')
    plt.title("Abbe")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(intensity_Hopkins, cmap='gray')
    plt.title("Hopkins")
    plt.colorbar()
    
    plt.show()
    
if __name__ == '__main__':
    # Call the check function to test Calculate1DAerialImage
    check()
