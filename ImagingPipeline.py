from Numerics import Numerics
from Source import Source
from Scatter import Scatter
from Projection import Projection
from CalculateAerialImage import CalculateAbbeImage, CalculateHopkinsImage
from rcwa import rcwa
import matplotlib.pyplot as plt
import torch

class ImagingModel:
    def __init__(self):
        self.Numerics = Numerics()
        self.Source = Source()
        self.Scatter = Scatter()
        self.Projector = Projection()

        self.Scatter.Period_X = 2000 # nm
        self.Scatter.Period_Y = 2000 # nm
        self.Source.Wavelength = 365 # nm
        self.Projector.Magnification = 100
        self.Projector.NA = 0.9
        self.Numerics.ScatterGrid_X = 300
        self.Numerics.ScatterGrid_Y = 300
        self.Numerics.ScatterOrder_X = 10
        self.Numerics.ScatterOrder_Y = 10
        self.Source.PolarizationVector = [0.0, 1.0]
        self.Projector.FocusRange = torch.tensor([0])
        self.Projector.Aberration_Zernike = torch.zeros(37)
        
        self.Geometry = torch.tensor([])
        self.Intensity = torch.tensor([])

    def CalculateImage(self):
        sr = self.Source
        sc = self.Scatter
        po = self.Projector
        nm = self.Numerics
        if (nm.ImageCalculationMethod == 'abbe'):
            ali, sp = CalculateAbbeImage(sr, sc, po, nm)
        elif (nm.ImageCalculationMethod == 'hopkins'):
            ali, sp = CalculateHopkinsImage(sr, sc, po, nm)
        else:
            raise ValueError('Unsupported Calculation Method')
        return ali, sp

    def LoadGeometry(self):
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        image = np.zeros((400, 400), dtype=np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 150)
        text = "SIOK"
        draw.text((10, 5), text, fill=1, font=font)
        P = np.array(image, dtype=np.int8)
        self.Geometry = torch.from_numpy(P)

    def Scattering(self):
        sim_dtype = torch.complex64
        # material
        substrate_eps = 1.1**2
        silicon_eps = 1.45**2

        n_scatter_x = self.Numerics.ScatterGrid_X
        n_scatter_y = self.Numerics.ScatterGrid_Y

        xs = (self.Scatter.Period_X/n_scatter_x)*(torch.arange(n_scatter_x)+0.5)
        ys = (self.Scatter.Period_Y/n_scatter_y)*(torch.arange(n_scatter_y)+0.5)
        
        # layers
        layer0_geometry = self.Geometry
        layer0_thickness = 30.

        order = [self.Numerics.ScatterOrder_X, self.Numerics.ScatterOrder_Y]
        sim = rcwa(freq=1/self.Source.Wavelength,order=order,
                   L=[self.Scatter.Period_X, self.Scatter.Period_Y],dtype=sim_dtype)
        sim.add_input_layer(eps=substrate_eps)
        sim.set_incident_angle(inc_ang=0,azi_ang=0)

        layer0_eps = layer0_geometry*silicon_eps + (1.-layer0_geometry)
        sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=self.Source.PolarizationVector, direction='forward', notation='xy')
        [Ex, Ey, Ez], [_, _, _] = sim.field_xy(xs, ys)
        self.Scatter.ScatterField = torch.stack((Ex, Ey, Ez), 2)

    def Imaging(self):
        img, pupilimg = self.CalculateImage()
        self.Intensity = img.Intensity
        self.Illuminant = self.Source.source_data
        self.PupilImage = pupilimg
        # For cross check purpose
        #self.Numerics.ImageCalculationMethod = 'abbe'
        #img, pupilimg = self.CalculateImage()
        #self.Intensity_abbe = img.Intensity

    def Visualizing(self):
        plt.subplot(331)
        plt.imshow(self.Geometry.cpu(), cmap='gray',
                   extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Pattern')

        plt.subplot(332)
        plt.imshow(self.Intensity[0,:,:].squeeze().cpu(),cmap='gray',
                   extent = (0, 1e-3*self.Projector.Magnification*self.Scatter.Period_X, \
                             0, 1e-3*self.Projector.Magnification*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Intensity')
        plt.colorbar()
        
        plt.subplot(333)
        plt.scatter(self.Illuminant.X, self.Illuminant.Y, self.Illuminant.Value, c='r', alpha=0.5)
        plt.axis('square')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.title('Source')
        """
        plt.subplot(333)
        plt.imshow((self.Intensity_abbe[0,:,:] - self.Intensity[0,:,:]).squeeze().cpu(),cmap='jet',
                   extent = (0, 1e-3*self.Projector.Magnification*self.Scatter.Period_X, \
                             0, 1e-3*self.Projector.Magnification*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Intensity (diff)')
        plt.colorbar()
        """
        
        ExEyEz = ['Scatter |Ex|','Scatter |Ey|','Scatter |Ez|']
        for k in range(3):
            plt.subplot(3, 3, k + 4)
            plt.imshow(torch.abs(self.Scatter.ScatterField[:,:,k].squeeze()).cpu(),cmap='jet',
                       extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
            plt.xlabel('μm')
            plt.ylabel('μm')
            plt.title(ExEyEz[k])
            plt.colorbar()

        ExEyEz = ['Pupil |Ex|','Pupil |Ey|','Pupil |Ez|']
        for k in range(3):
            plt.subplot(3, 3, k + 7)
            plt.imshow(torch.abs(self.PupilImage[:,:,k].squeeze()).cpu(),cmap='jet')
            plt.axis('off')
            plt.title(ExEyEz[k])
            plt.colorbar()
        plt.show()

    def Run(self):
        self.LoadGeometry()
        self.Scattering()
        self.Imaging()
        self.Visualizing()
    
if __name__ == '__main__':
    app = ImagingModel()
    app.Run()
