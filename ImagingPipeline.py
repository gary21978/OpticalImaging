from Numerics import Numerics
from Source import Source
from Scatter import Scatter
from Projection import Projection
from CalculateAerialImage import CalculateAbbeImage, CalculateHopkinsImage
import matplotlib.pyplot as plt
import torch
from rcwa import rcwa
    
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

class Pipeline:
    def __init__(self):
        self.Lx = 2000 # nm
        self.Ly = 2000 # nm
        self.Wavelength = 365 # nm
        self.Magnification = 100
        self.NumericalAperture = 0.9
        self.n_scatter_x = 300
        self.n_scatter_y = 300

        self.Geometry = torch.tensor([])
        self.Intensity = torch.tensor([])

    def LoadGeometry(self):
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        image = np.zeros((50, 50), dtype=np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 8)
        text = "I"
        draw.text((15, 10), text, fill=1, font=font)
        P = np.transpose(np.array(image, dtype=np.int8))
        self.Geometry = torch.from_numpy(P)

    def Scattering(self):
        sim_dtype = torch.complex64
        # material
        substrate_eps = 1.1**2
        silicon_eps = 1.45**2

        xs = (self.Lx/self.n_scatter_x)*(torch.arange(self.n_scatter_x)+0.5)
        ys = (self.Ly/self.n_scatter_y)*(torch.arange(self.n_scatter_y)+0.5)
        
        # layers
        layer0_geometry = self.Geometry
        layer0_thickness = 300.

        order_N = 5
        order = [order_N, order_N]
        sim = rcwa(freq=1/self.Wavelength,order=order,L=[self.Lx, self.Ly],dtype=sim_dtype)
        sim.add_input_layer(eps=substrate_eps)
        sim.set_incident_angle(inc_ang=0,azi_ang=0)

        layer0_eps = layer0_geometry*silicon_eps + (1.-layer0_geometry)
        sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=[1.,0.],direction='forward')
        [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xy(-1, xs, ys)
        self.scatter_field = torch.stack((Ex, Ey, Ez), 2)

    def Imaging(self):
        im = ImagingModel()
        im.Scatter.Period_X = self.Lx
        im.Scatter.Period_Y = self.Ly
        im.Source.Wavelength = self.Wavelength
        im.Source.PntNum = 81
        im.Source.Shape = "quasar"
        im.Source.PolarizationType = "c_pol"
        im.Source.SigmaOut = 0.7
        im.Source.SigmaIn = 0.1
        im.Projector.Aberration_Zernike = torch.zeros(37)
        im.Projector.Magnification = self.Magnification
        im.Projector.NA = self.NumericalAperture
        im.Projector.IndexImage = 1.0
        im.Projector.FocusRange = torch.tensor([0])
        im.Numerics.ImageCalculationMethod = "hopkins"
        im.Numerics.ImageCalculationMode = "scalar"
        im.Numerics.Hopkins_SettingType = "order"
        im.Numerics.Hopkins_Order = 100
        im.Numerics.Hopkins_Threshold = 0.95
        im.Scatter.ScatterField = self.scatter_field

        self.Intensity = im.CalculateAerialImage().Intensity
        self.Illuminant = im.Source.source_data

    def Visualizing(self):
        plt.subplot(231)
        plt.imshow(self.Geometry.cpu(), cmap='gray',
                   extent = (0, 0.001*self.Lx, 0, 0.001*self.Ly))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Pattern')

        plt.subplot(232)
        plt.imshow(torch.transpose(torch.abs(self.Intensity[0,:,:].squeeze()),-2, -1).cpu(),cmap='jet',
                   extent = (0, 0.001*self.Magnification*self.Lx, 0, 0.001*self.Magnification*self.Ly))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Intensity')
        plt.colorbar()

        plt.subplot(233)
        plt.scatter(self.Illuminant.X, self.Illuminant.Y, self.Illuminant.Value, c='r', alpha=0.5)
        plt.axis('square')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.title('Source')
        
        plt.subplot(234)
        plt.imshow(torch.abs(self.scatter_field[:,:,0].squeeze()).cpu(),cmap='jet',
                   extent = (0, 0.001*self.Lx, 0, 0.001*self.Ly))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('|E_x|')
        plt.colorbar()

        plt.subplot(235)
        plt.imshow(torch.abs(self.scatter_field[:,:,1].squeeze()).cpu(),cmap='jet',
                   extent = (0, 0.001*self.Lx, 0, 0.001*self.Ly))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('|E_y|')
        plt.colorbar()

        plt.subplot(236)
        plt.imshow(torch.abs(self.scatter_field[:,:,2].squeeze()).cpu(),cmap='jet',
                   extent = (0, 0.001*self.Lx, 0, 0.001*self.Ly))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('|E_z|')
        plt.colorbar()

        plt.show()

    def Run(self):
        self.LoadGeometry()
        self.Scattering()
        self.Imaging()
        self.Visualizing()
    
if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.Run()
