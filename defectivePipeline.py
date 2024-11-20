from Numerics import Numerics
from Source import Source
from Scatter import Scatter
from Projection import Projection
from CalculateAerialImage import CalculateAbbeImage, CalculateHopkinsImage, CalculateOptimized
from rcwa import rcwa
import matplotlib.pyplot as plt
import torch

class ImagingModel:
    def __init__(self):
        self.Numerics = Numerics()
        self.Source = Source()
        self.Scatter = Scatter()
        self.Projector = Projection()

        self.Scatter.Period_X = 3000 # nm
        self.Scatter.Period_Y = 1500 # nm
        self.Source.Wavelength = 365 # nm
        self.Projector.Magnification = 100
        self.Projector.NA = 0.9
        self.Numerics.ScatterGrid_X = 256
        self.Numerics.ScatterGrid_Y = 128
        self.Numerics.ScatterOrder_X = 5
        self.Numerics.ScatterOrder_Y = 5
        self.Source.PolarizationVector = [0.0, 1.0]
        self.Projector.FocusRange = torch.tensor([0])
        self.Projector.Aberration_Zernike = torch.zeros(37)
        
        self.Geometry = torch.tensor([])
        self.Defect = torch.tensor([])

    def CalculateImage(self):
        sr = self.Source
        sc = self.Scatter
        po = self.Projector
        nm = self.Numerics
        if (nm.ImageCalculationMethod == 'abbe'):
            ali, _ = CalculateAbbeImage(sr, sc, po, nm)
        elif (nm.ImageCalculationMethod == 'hopkins'):
            ali, _ = CalculateHopkinsImage(sr, sc, po, nm)
        else:
            raise ValueError('Unsupported Calculation Method')
        return ali

    def LoadGeometry(self):
        image = torch.zeros((200, 400))
        image[:, 100:200] = 1
        image[:, 300:400] = 1
        self.Geometry = image

    def LoadDefect(self):
        image = torch.zeros((200, 400))
        #image[100:120, 200:300] = 1
        #image[100:110, 200:210] = 1 # large defect A
        #image[100:110, 250:260] = 1 # large defect B
        image[100:109, 250:259] = 1 # medium defect A
        #image[100:101, 200:201] = 1 # small defect A
        #image[100:102, 250:252] = 1 # small defect B
        self.Defect = image

    def Scattering(self):
        sim_dtype = torch.complex64

        sim = rcwa(freq=1/self.Source.Wavelength, order=[self.Numerics.ScatterOrder_X, self.Numerics.ScatterOrder_Y],
                   L=[self.Scatter.Period_X, self.Scatter.Period_Y],dtype=sim_dtype)
        sim.add_input_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0, azi_ang=0)

        # reflective indices
        n_OX = 1.4745
        n_TIN = 2.0669 + 1.0563j
        n_TEOS = 1.380
        n_SIN = 2.1222
        n_substrate = 6.5271 + 2.6672j
        ##################################################
        layer_eps = self.Geometry*(n_OX**2 - 1) + 1.0
        sim.add_layer(thickness=5, eps=layer_eps)
        sim.add_layer(thickness=20, eps=n_SIN**2)
        sim.add_layer(thickness=5, eps=n_substrate**2)
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=self.Source.PolarizationVector, direction='forward')
        n_scatter_x = self.Numerics.ScatterGrid_X
        n_scatter_y = self.Numerics.ScatterGrid_Y

        xs = (self.Scatter.Period_X/n_scatter_x)*(torch.arange(n_scatter_x)+0.5)
        ys = (self.Scatter.Period_Y/n_scatter_y)*(torch.arange(n_scatter_y)+0.5)
        Ex, Ey, Ez = sim.field_xy(xs, ys)
        self.Scatter.ScatterField = torch.stack((Ex, Ey, Ez), 2)
        self.NormalScatterField = self.Scatter.ScatterField

    def ScatteringDefective(self):
        sim_dtype = torch.complex64

        sim = rcwa(freq=1/self.Source.Wavelength, order=[self.Numerics.ScatterOrder_X, self.Numerics.ScatterOrder_Y],
                   L=[self.Scatter.Period_X, self.Scatter.Period_Y],dtype=sim_dtype)
        sim.add_input_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0, azi_ang=0)

        # reflective indices
        n_OX = 1.4745
        n_TIN = 2.0669 + 1.0563j
        n_TEOS = 1.380
        n_SIN = 2.1222
        n_substrate = 6.5271 + 2.6672j
        ##################################################
        layer_eps = (self.Geometry + self.Defect)*(n_OX**2 - 1) + 1.0
        sim.add_layer(thickness=5, eps=layer_eps)
        sim.add_layer(thickness=20, eps=n_SIN**2)
        sim.add_layer(thickness=5, eps=n_substrate**2)
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=self.Source.PolarizationVector, direction='forward')
        n_scatter_x = self.Numerics.ScatterGrid_X
        n_scatter_y = self.Numerics.ScatterGrid_Y

        xs = (self.Scatter.Period_X/n_scatter_x)*(torch.arange(n_scatter_x)+0.5)
        ys = (self.Scatter.Period_Y/n_scatter_y)*(torch.arange(n_scatter_y)+0.5)
        Ex, Ey, Ez = sim.field_xy(xs, ys)

        self.Scatter.ScatterField = torch.stack((Ex, Ey, Ez), 2)
        self.DefectiveScatterField = self.Scatter.ScatterField

    def ScatteringPerturbation(self):
        sim_dtype = torch.complex64
        sim = rcwa(freq=1/self.Source.Wavelength, order=[self.Numerics.ScatterOrder_X, self.Numerics.ScatterOrder_Y],
                   L=[self.Scatter.Period_X, self.Scatter.Period_Y],dtype=sim_dtype)
        sim.add_input_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0, azi_ang=0)

        # reflective indices
        n_OX = 1.4745
        n_TIN = 2.0669 + 1.0563j
        n_TEOS = 1.380
        n_SIN = 2.1222
        n_substrate = 6.5271 + 2.6672j
        ##################################################
        layer_eps = self.Geometry*(n_OX**2 - 1) + 1.0
        sim.add_layer(thickness=5, eps=layer_eps)
        sim.add_layer(thickness=20, eps=n_SIN**2)
        sim.add_layer(thickness=5, eps=n_substrate**2)
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=self.Source.PolarizationVector, direction='forward')
        n_scatter_x = self.Numerics.ScatterGrid_X
        n_scatter_y = self.Numerics.ScatterGrid_Y

        xs = (self.Scatter.Period_X/n_scatter_x)*(torch.arange(n_scatter_x)+0.5)
        ys = (self.Scatter.Period_Y/n_scatter_y)*(torch.arange(n_scatter_y)+0.5)
        Ex, Ey, Ez = sim.field_xy(xs, ys)

        #### Add defect
        sim.add_defect(layer_num=0, deps=self.Defect*(n_OX**2 - 1))
        sim.solve_global_smatrix()
        Ex, Ey, Ez = sim.field_xy(xs, ys)
        self.Scatter.ScatterField = torch.stack((Ex, Ey, Ez), 2)
        self.PerturbationScatterField = self.Scatter.ScatterField

    def Imaging(self):
        img = self.CalculateImage()
        self.Intensity = img.Intensity

    def ImagingDefective(self):
        img = self.CalculateImage()
        self.IntensityDefective = img.Intensity

    def ImagingPerturbation(self):
        img = self.CalculateImage()
        self.IntensityPerturbation = img.Intensity   

    def Visualizing(self):
        plt.subplot(361)
        plt.imshow(1.0 - self.Geometry.cpu(), cmap='gray',
                   extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Pattern')
        plt.colorbar()

        plt.subplot(362)
        plt.imshow(self.Intensity[0,:,:].squeeze().cpu(),cmap='gray',
                   extent = (0, 1e-3*self.Projector.Magnification*self.Scatter.Period_X, \
                             0, 1e-3*self.Projector.Magnification*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Intensity')
        plt.colorbar()
        
        ExEyEz = ['Scatter |Ex|','Scatter |Ey|','Scatter |Ez|']
        for k in range(3):
            plt.subplot(3, 6, k + 4)
            plt.imshow(torch.abs(self.NormalScatterField[:,:,k].squeeze()).cpu(),cmap='jet',
                       extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
            plt.xlabel('μm')
            plt.ylabel('μm')
            plt.title(ExEyEz[k])
            plt.colorbar()
        ###############################
        plt.subplot(367)
        plt.imshow(1.0 - self.Geometry.cpu() - self.Defect.cpu(), cmap='gray',
                   extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Pattern')
        plt.colorbar()

        plt.subplot(368)
        plt.imshow(self.IntensityDefective[0,:,:].squeeze().cpu(),cmap='gray',
                   extent = (0, 1e-3*self.Projector.Magnification*self.Scatter.Period_X, \
                             0, 1e-3*self.Projector.Magnification*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Intensity')
        plt.colorbar()

        plt.subplot(369)
        plt.imshow(self.IntensityDefective[0,:,:].squeeze().cpu() - self.Intensity[0,:,:].squeeze().cpu(),cmap='gray',
                   extent = (0, 1e-3*self.Projector.Magnification*self.Scatter.Period_X, \
                             0, 1e-3*self.Projector.Magnification*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Intensity')
        plt.colorbar()
        
        ExEyEz = ['Scatter |Ex|','Scatter |Ey|','Scatter |Ez|']
        for k in range(3):
            plt.subplot(3, 6, k + 10)
            plt.imshow(torch.abs(self.DefectiveScatterField[:,:,k].squeeze()).cpu(),cmap='jet',
                       extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
            plt.xlabel('μm')
            plt.ylabel('μm')
            plt.title(ExEyEz[k])
            plt.colorbar()
        ###############################
        plt.subplot(3,6,13)
        plt.imshow(1.0 - self.Geometry.cpu() - self.Defect.cpu(), cmap='gray',
                   extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Pattern')
        plt.colorbar()

        plt.subplot(3,6,14)
        plt.imshow(self.IntensityPerturbation[0,:,:].squeeze().cpu(),cmap='gray',
                   extent = (0, 1e-3*self.Projector.Magnification*self.Scatter.Period_X, \
                             0, 1e-3*self.Projector.Magnification*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Intensity')
        plt.colorbar()

        plt.subplot(3,6,15)
        plt.imshow(self.IntensityPerturbation[0,:,:].squeeze().cpu() - self.Intensity[0,:,:].squeeze().cpu(),cmap='gray',
                   extent = (0, 1e-3*self.Projector.Magnification*self.Scatter.Period_X, \
                             0, 1e-3*self.Projector.Magnification*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Intensity')
        plt.colorbar()
        
        ExEyEz = ['Scatter |Ex|','Scatter |Ey|','Scatter |Ez|']
        for k in range(3):
            plt.subplot(3, 6, k + 16)
            plt.imshow(torch.abs(self.PerturbationScatterField[:,:,k].squeeze()).cpu(),cmap='jet',
                       extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
            plt.xlabel('μm')
            plt.ylabel('μm')
            plt.title(ExEyEz[k])
            plt.colorbar()

        plt.show()

    def Run(self):
        self.LoadGeometry()
        self.LoadDefect()
        self.Scattering()
        self.Imaging()

        self.ScatteringDefective()
        self.ImagingDefective()

        self.ScatteringPerturbation()
        self.ImagingPerturbation()

        self.Visualizing()
    
if __name__ == '__main__':
    app = ImagingModel()
    app.Run()
