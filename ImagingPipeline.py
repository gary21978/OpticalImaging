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
        elif (nm.ImageCalculationMethod == 'optimized'):
            ali, sp = CalculateOptimized(sr, sc, po, nm)
        else:
            raise ValueError('Unsupported Calculation Method')
        return ali

    def LoadGeometry(self):
        T = torch.zeros([200, 100], dtype=torch.float64)
        T[:, 25:50] = 1
        T[50:100, :] = 1
        self.Geometry = T

    def Scattering(self):
        sim_dtype = torch.complex128
        normalized_xPitch = torch.tensor(self.Scatter.Period_X / (self.Source.Wavelength / self.Projector.NA))
        normalized_yPitch = torch.tensor(self.Scatter.Period_Y / (self.Source.Wavelength / self.Projector.NA))
        # TODO use recommended Fourier orders
        #Nf = torch.ceil(2 * normalized_xPitch).int()
        #Ng = torch.ceil(2 * normalized_yPitch).int()
        #print("Recommended Fourier orders: [%d,%d]" % (Nf, Ng))
        #self.Numerics.ScatterOrder_X = Nf
        #self.Numerics.ScatterOrder_Y = Ng

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
        thickness = [10, 21, 100]
        ns = [n_OX, n_TIN, n_TEOS]
        for t, n in zip(thickness, ns):
            layer_eps = self.Geometry*(n**2 - 1) + 1.0
            sim.add_layer(thickness=t, eps=layer_eps)
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

        # Compute Floquet mode ONLY
        #Ex_mn, Ey_mn, Ez_mn = sim.Floquet_mode()
        #self.Scatter.FloquetMode = torch.stack((Ex_mn, Ey_mn, Ez_mn), 2)

    def Imaging(self):
        img = self.CalculateImage()
        self.Intensity = img.Intensity
        self.Illuminant = self.Source.source_data

    def Visualizing(self):
        plt.subplot(231)
        plt.imshow(1.0 - self.Geometry.cpu(), cmap='gray',
                   extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
        plt.xlabel('μm')
        plt.ylabel('μm')
        plt.title('Pattern')
        plt.colorbar()

        plt.subplot(232)
        plt.imshow(self.Intensity[0,:,:].squeeze().cpu(),cmap='gray',
                   extent = (0, 1e-3*self.Projector.Magnification*self.Scatter.Period_X, \
                             0, 1e-3*self.Projector.Magnification*self.Scatter.Period_Y))
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
        
        ExEyEz = ['Scatter |Ex|','Scatter |Ey|','Scatter |Ez|']
        for k in range(3):
            plt.subplot(2, 3, k + 4)
            plt.imshow(torch.abs(self.Scatter.ScatterField[:,:,k].squeeze()).cpu(),cmap='jet',
                       extent = (0, 1e-3*self.Scatter.Period_X, 0, 1e-3*self.Scatter.Period_Y))
            plt.xlabel('μm')
            plt.ylabel('μm')
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
