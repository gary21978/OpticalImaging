import warnings
import torch

class Eig(torch.autograd.Function):
    broadening_parameter = 1e-10

    @staticmethod
    def forward(ctx,x):
        ctx.input = x
        eigval, eigvec = torch.linalg.eig(x)
        ctx.eigval = eigval.cpu()
        ctx.eigvec = eigvec.cpu()
        return eigval, eigvec

    @staticmethod
    def backward(ctx,grad_eigval,grad_eigvec):
        eigval = ctx.eigval.to(grad_eigval)
        eigvec = ctx.eigvec.to(grad_eigvec)

        grad_eigval = torch.diag(grad_eigval)
        s = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)

        # Lorentzian broadening: get small error but stabilizing the gradient calculation
        if Eig.broadening_parameter is not None:
            F = torch.conj(s)/(torch.abs(s)**2 + Eig.broadening_parameter)
        elif s.dtype == torch.complex64:
            F = torch.conj(s)/(torch.abs(s)**2 + 1.4e-45)
        elif s.dtype == torch.complex128:
            F = torch.conj(s)/(torch.abs(s)**2 + 4.9e-324)

        diag_indices = torch.linspace(0,F.shape[-1]-1,F.shape[-1],dtype=torch.int64)
        F[diag_indices,diag_indices] = 0.
        XH = torch.transpose(torch.conj(eigvec),-2,-1)
        tmp = torch.conj(F) * torch.matmul(XH, grad_eigvec)

        grad = torch.matmul(torch.matmul(torch.inverse(XH), grad_eigval + tmp), XH)
        if not torch.is_complex(ctx.input):
            grad = torch.real(grad)

        return grad

class rcwa:
    # Simulation setting
    def __init__(self,freq,order,L,*,
            dtype=torch.complex64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            stable_eig_grad=True,
        ):

        '''
            Rigorous Coupled Wave Analysis
            - Lorentz-Heaviside units
            - Speed of light: 1
            - Time harmonics notation: exp(-jωt)

            Parameters
            - freq: simulation frequency (unit: length^-1)
            - order: Fourier order [x_order (int), y_order (int)]
            - L: Lattice constant [Lx, Ly] (unit: length)

            Keyword Parameters
            - dtype: simulation data type (only torch.complex64 and torch.complex128 are allowed.)
            - device: simulation device (only torch.device('cpu') and torch.device('cuda') are allowed.)
            - stable_eig_grad: stabilize gradient calculation of eigendecompsition (default as True)
        '''

        # Hardware
        if dtype != torch.complex64 and dtype != torch.complex128:
            warnings.warn('Invalid simulation data type. Set as torch.complex64.',UserWarning)
            self._dtype = torch.complex64
        else:
            self._dtype = dtype
        self._device = device

        # Stabilize the gradient of eigendecomposition
        self.stable_eig_grad = True if stable_eig_grad else False

        # Simulation parameters
        self.freq = torch.as_tensor(freq,dtype=self._dtype,device=self._device) # unit^-1
        self.omega = 2*torch.pi*freq # same as k0a
        self.L = torch.as_tensor(L,dtype=self._dtype,device=self._device)

        # Fourier order
        self.order = order
        self.order_x = torch.linspace(-self.order[0],self.order[0],2*self.order[0]+1,dtype=torch.int64,device=self._device)
        self.order_y = torch.linspace(-self.order[1],self.order[1],2*self.order[1]+1,dtype=torch.int64,device=self._device)
        self.order_N = len(self.order_x)*len(self.order_y)

        # Lattice vector
        self.L = L  # unit
        self.Gx_norm, self.Gy_norm = 1/(L[0]*self.freq), 1/(L[1]*self.freq)

        # Input and output layer (Default: free space)
        self.eps_in = torch.tensor(1.,dtype=self._dtype,device=self._device)
        self.mu_in = torch.tensor(1.,dtype=self._dtype,device=self._device)
        self.eps_out = torch.tensor(1.,dtype=self._dtype,device=self._device)
        self.mu_out = torch.tensor(1.,dtype=self._dtype,device=self._device)

        # Internal layers
        self.layer_N = 0  # total number of layers
        self.thickness = []
        self.eps_conv, self.mu_conv = [], []

        # Internal layer eigenmodes
        self.P, self.Q = [], []
        self.kz_norm, self.E_eigvec, self.H_eigvec = [], [], []

        # Single layer scattering matrices
        self.layer_S11, self.layer_S21, self.layer_S12, self.layer_S22 = [], [], [], []

    def add_input_layer(self,eps=1.,mu=1.):
        '''
            Add input layer
            - If this function is not used, simulation will be performed under free space input layer.

            Parameters
            - eps: relative permittivity
            - mu: relative permeability
        '''

        self.eps_in = torch.as_tensor(eps,dtype=self._dtype,device=self._device)
        self.mu_in = torch.as_tensor(mu,dtype=self._dtype,device=self._device)
        self.Sin = []

    def add_output_layer(self,eps=1.,mu=1.):
        '''
            Add output layer
            - If this function is not used, simulation will be performed under free space output layer.

            Parameters
            - eps: relative permittivity
            - mu: relative permeability
        '''

        self.eps_out = torch.as_tensor(eps,dtype=self._dtype,device=self._device)
        self.mu_out = torch.as_tensor(mu,dtype=self._dtype,device=self._device)
        self.Sout = []

    def set_incident_angle(self,inc_ang,azi_ang):
        '''
            Set incident angle

            Parameters
            - inc_ang: incident angle (unit: radian)
            - azi_ang: azimuthal angle (unit: radian)
        '''

        self.inc_ang = torch.as_tensor(inc_ang,dtype=self._dtype,device=self._device)
        self.azi_ang = torch.as_tensor(azi_ang,dtype=self._dtype,device=self._device)

        self._kvectors()

    def add_layer(self,thickness,eps=1.,mu=1.):
        '''
            Add internal layer

            Parameters
            - thickness: layer thickness (unit: length)
            - eps: relative permittivity
            - mu: relative permeability
        '''

        is_eps_homogenous = (type(eps) == float) or (type(eps) == complex) or (eps.dim() == 0) or ((eps.dim() == 1) and eps.shape[0] == 1)
        is_mu_homogenous = (type(mu) == float) or (type(mu) == complex) or (mu.dim() == 0) or ((mu.dim() == 1) and mu.shape[0] == 1)
        
        self.eps_conv.append(eps*torch.eye(self.order_N,dtype=self._dtype,device=self._device) if is_eps_homogenous else self._material_conv(eps))
        self.mu_conv.append(mu*torch.eye(self.order_N,dtype=self._dtype,device=self._device) if is_mu_homogenous else self._material_conv(mu))

        self.layer_N += 1
        self.thickness.append(thickness)

        if is_eps_homogenous and is_mu_homogenous:
            self._eigen_decomposition_homogenous(eps,mu)
        else:
            self._eigen_decomposition()

        self._solve_layer_smatrix()

    # Solve simulation
    def solve_global_smatrix(self):
        '''
            Solve global S-matrix
        '''

        # Initialization
        if self.layer_N > 0:
            S11 = self.layer_S11[0]
            S21 = self.layer_S21[0]
            S12 = self.layer_S12[0]
            S22 = self.layer_S22[0]
        else:
            S11 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
            S21 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
            S12 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
            S22 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)

        # Connection
        for i in range(self.layer_N-1):
            S11, S21, S12, S22 = self._RS_prod(Sm=[S11, S21, S12, S22],
                Sn=[self.layer_S11[i+1], self.layer_S21[i+1], self.layer_S12[i+1], self.layer_S22[i+1]])

        if hasattr(self,'Sin'):
            # input layer coupling
            S11, S21, S12, S22= self._RS_prod(Sm=[self.Sin[0], self.Sin[1], self.Sin[2], self.Sin[3]],
                Sn=[S11, S21, S12, S22])

        if hasattr(self,'Sout'):
            # output layer coupling
            S11, S21, S12, S22 = self._RS_prod(Sm=[S11, S21, S12, S22],
                Sn=[self.Sout[0], self.Sout[1], self.Sout[2], self.Sout[3]])

        self.S = [S11, S21, S12, S22]
        
    def source_planewave(self,*,amplitude=[1.,0.],direction='forward'):
        '''
            Generate planewave

            Paramters
            - amplitude: amplitudes at the matched diffraction orders ([Ex_amp, Ey_amp])
              (list / np.ndarray / torch.Tensor) (Recommended shape: 1x2)
            - direction: incident direction ('f', 'forward' / 'b', 'backward')
        '''

        self.source_fourier(amplitude=amplitude,orders=[0,0],direction=direction)

    def source_fourier(self,*,amplitude,orders,direction='forward'):
        '''
            Generate Fourier source

            Paramters
            - amplitude: amplitudes at the matched diffraction orders [([Ex_amp, Ey_amp] at orders[0]), ..., ...]
                (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
            - orders: diffraction orders (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
            - direction: incident direction ('f', 'forward' / 'b', 'backward')
        '''
        amplitude = torch.as_tensor(amplitude,dtype=self._dtype,device=self._device).reshape([-1,2])
        orders = torch.as_tensor(orders,dtype=torch.int64,device=self._device).reshape([-1,2])

        if direction in ['f', 'forward']:
            direction = 'forward'
        elif direction in ['b', 'backward']:
            direction = 'backward'
        else:
            warnings.warn('Invalid source direction. Set as forward.',UserWarning)
            direction = 'forward'

        # Matching indices
        order_indices = self._matching_indices(orders)

        self.source_direction = direction

        E_i = torch.zeros([2*self.order_N,1],dtype=self._dtype,device=self._device)
        E_i[order_indices,0] = amplitude[:,0]
        E_i[order_indices+self.order_N,0] = amplitude[:,1]

        self.E_i = E_i

    def field_xy(self,x_axis,y_axis,z_prop=0.):
        '''
            XY-plane field distribution at the selected layer.
            Returns the field at z_prop away from the lower boundary of the layer.

            Parameters
            - x_axis: x-direction sampling coordinates (torch.Tensor)
            - y_axis: y-direction sampling coordinates (torch.Tensor)
            - z_prop: z-direction distance from the upper boundary of the layer and should be negative.

            Return
            - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
        '''

        if type(x_axis) != torch.Tensor or type(y_axis) != torch.Tensor:
            warnings.warn('x and y axis must be torch.Tensor type. Return None.',UserWarning)
            return None
        
        # [x, y, diffraction order]
        x_axis = x_axis.reshape([-1,1,1])
        y_axis = y_axis.reshape([1,-1,1])

        Kx_norm, Ky_norm = self.Kx_norm, self.Ky_norm

        # Input and output layers
        Kx_norm_dn, Ky_norm_dn = self.Kx_norm_dn, self.Ky_norm_dn

        
        z_prop = z_prop if z_prop <= 0. else 0.
        eps = self.eps_in if hasattr(self,'eps_in') else 1.
        mu = self.mu_in if hasattr(self,'mu_in') else 1.
        Vi = self.Vi if hasattr(self,'Vi') else self.Vf
        Kz_norm_dn = torch.sqrt(eps*mu - Kx_norm_dn**2 - Ky_norm_dn**2)
        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)>0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])

        # Phase
        Kz_norm_dn = torch.vstack((Kz_norm_dn,Kz_norm_dn))
        z_phase = torch.exp(1.j*self.omega*Kz_norm_dn*z_prop)
            
        # Fourier domain fields
        # [diffraction order, diffraction order]
        if self.source_direction == 'forward':
            Exy_p = self.E_i*z_phase
            Hxy_p = torch.matmul(Vi,Exy_p)
            Exy_m = torch.matmul(self.S[1],self.E_i)*torch.conj(z_phase)
            Hxy_m = torch.matmul(-Vi,Exy_m)
        elif self.source_direction == 'backward':
            Exy_p = torch.zeros_like(self.E_i)
            Hxy_p = torch.zeros_like(self.E_i)
            Exy_m = torch.matmul(self.S[3],self.E_i)*torch.conj(z_phase)
            Hxy_m = torch.matmul(-Vi,Exy_m)

        Ex_mn = Exy_p[:self.order_N] + Exy_m[:self.order_N]
        Ey_mn = Exy_p[self.order_N:] + Exy_m[self.order_N:]
        Hz_mn = torch.matmul(Kx_norm,Ey_mn)/mu - torch.matmul(Ky_norm,Ex_mn)/mu
        Hx_mn = Hxy_p[:self.order_N] + Hxy_m[:self.order_N]
        Hy_mn = Hxy_p[self.order_N:] + Hxy_m[self.order_N:]
        Ez_mn = torch.matmul(Ky_norm,Hx_mn)/eps - torch.matmul(Kx_norm,Hy_mn)/eps

        # Spatial domain fields
        xy_phase = torch.exp(1.j * self.omega * (self.Kx_norm_dn*x_axis + self.Ky_norm_dn*y_axis))
        Ex = torch.sum(Ex_mn.reshape(1,1,-1)*xy_phase,dim=2)
        Ey = torch.sum(Ey_mn.reshape(1,1,-1)*xy_phase,dim=2)
        Ez = torch.sum(Ez_mn.reshape(1,1,-1)*xy_phase,dim=2)
        Hx = torch.sum(Hx_mn.reshape(1,1,-1)*xy_phase,dim=2)
        Hy = torch.sum(Hy_mn.reshape(1,1,-1)*xy_phase,dim=2)
        Hz = torch.sum(Hz_mn.reshape(1,1,-1)*xy_phase,dim=2)

        return [Ex, Ey, Ez], [Hx, Hy, Hz]
        
    def Floquet_mode(self):
        Kz_norm_dn = torch.sqrt(1.0 - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])
        
        Exy = self.E_i + torch.matmul(self.S[1],self.E_i)
        
        Ex_mn = Exy[:self.order_N]
        Ey_mn = Exy[self.order_N:]
        Ez_mn = (self.Kx_norm_dn.reshape((-1,1))*Ex_mn + self.Ky_norm_dn.reshape((-1,1))*Ey_mn)/Kz_norm_dn
        
        mnshape = (self.order_x.shape[0], self.order_y.shape[0])
        Ex_mn = torch.reshape(Ex_mn, mnshape)
        Ey_mn = torch.reshape(Ey_mn, mnshape)
        Ez_mn = torch.reshape(Ez_mn, mnshape)
        return Ex_mn, Ey_mn, Ez_mn

    # Internal functions
    def _matching_indices(self,orders):
        orders[orders[:,0]<-self.order[0],0] = int(-self.order[0])
        orders[orders[:,0]>self.order[0],0] = int(self.order[0])
        orders[orders[:,1]<-self.order[1],1] = int(-self.order[1])
        orders[orders[:,1]>self.order[1],1] = int(self.order[1])
        order_indices = len(self.order_y)*(orders[:,0]+int(self.order[0])) + orders[:,1]+int(self.order[1])

        return order_indices

    def _kvectors(self):
        self.kx0_norm = torch.real(torch.sqrt(self.eps_in*self.mu_in)) * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
        self.ky0_norm = torch.real(torch.sqrt(self.eps_in*self.mu_in)) * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)

        # Free space k-vectors and E to H transformation matrix
        self.kx_norm = self.kx0_norm + self.order_x * self.Gx_norm
        self.ky_norm = self.ky0_norm + self.order_y * self.Gy_norm

        kx_norm_grid, ky_norm_grid = torch.meshgrid(self.kx_norm,self.ky_norm,indexing='ij')

        self.Kx_norm_dn = torch.reshape(kx_norm_grid,(-1,))
        self.Ky_norm_dn = torch.reshape(ky_norm_grid,(-1,))
        self.Kx_norm = torch.diag(self.Kx_norm_dn)
        self.Ky_norm = torch.diag(self.Ky_norm_dn)

        Kz_norm_dn = torch.sqrt(1. - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn)
        tmp1 = torch.vstack((torch.diag(-self.Ky_norm_dn*self.Kx_norm_dn/Kz_norm_dn), torch.diag(Kz_norm_dn + self.Kx_norm_dn**2/Kz_norm_dn)))
        tmp2 = torch.vstack((torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2/Kz_norm_dn), torch.diag(self.Kx_norm_dn*self.Ky_norm_dn/Kz_norm_dn)))
        self.Vf = torch.hstack((tmp1, tmp2))

        if hasattr(self,'Sin'):
            # Input layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(self.eps_in*self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
            Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn)
            tmp1 = torch.vstack((torch.diag(-self.Ky_norm_dn*self.Kx_norm_dn/Kz_norm_dn), torch.diag(Kz_norm_dn + self.Kx_norm_dn**2/Kz_norm_dn)))
            tmp2 = torch.vstack((torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2/Kz_norm_dn), torch.diag(self.Kx_norm_dn*self.Ky_norm_dn/Kz_norm_dn)))
            self.Vi = torch.hstack((tmp1, tmp2))

            Vtmp1 = torch.linalg.inv(self.Vf+self.Vi)
            Vtmp2 = self.Vf-self.Vi

            # Input layer S-matrix
            self.Sin.append(2*torch.matmul(Vtmp1,self.Vi))  # Tf S11
            self.Sin.append(-torch.matmul(Vtmp1,Vtmp2))     # Rf S21
            self.Sin.append(torch.matmul(Vtmp1,Vtmp2))      # Rb S12
            self.Sin.append(2*torch.matmul(Vtmp1,self.Vf))  # Tb S22

        if hasattr(self,'Sout'):
            # Output layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(self.eps_out*self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
            Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn)
            tmp1 = torch.vstack((torch.diag(-self.Ky_norm_dn*self.Kx_norm_dn/Kz_norm_dn), torch.diag(Kz_norm_dn + self.Kx_norm_dn**2/Kz_norm_dn)))
            tmp2 = torch.vstack((torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2/Kz_norm_dn), torch.diag(self.Kx_norm_dn*self.Ky_norm_dn/Kz_norm_dn)))
            self.Vo = torch.hstack((tmp1, tmp2))

            Vtmp1 = torch.linalg.inv(self.Vf+self.Vo)
            Vtmp2 = self.Vf-self.Vo

            # Output layer S-matrix
            self.Sout.append(2*torch.matmul(Vtmp1,self.Vf))  # Tf S11
            self.Sout.append(torch.matmul(Vtmp1,Vtmp2))      # Rf S21
            self.Sout.append(-torch.matmul(Vtmp1,Vtmp2))     # Rb S12
            self.Sout.append(2*torch.matmul(Vtmp1,self.Vo))  # Tb S22

    def _material_conv(self,material):
        material = material.to(self._dtype)
        material_N = material.shape[0]*material.shape[1]

        # Matching indices
        order_x_grid, order_y_grid = torch.meshgrid(self.order_x,self.order_y,indexing='ij')
        ox = order_x_grid.to(torch.int64).reshape([-1])
        oy = order_y_grid.to(torch.int64).reshape([-1])

        ind = torch.arange(len(self.order_x)*len(self.order_y),device=self._device)
        indx, indy = torch.meshgrid(ind.to(torch.int64),ind.to(torch.int64),indexing='ij')

        material_fft = torch.fft.fft2(material)/material_N
        material_fft_real = torch.real(material_fft)
        material_fft_imag = torch.imag(material_fft)
        
        material_convmat_real = material_fft_real[ox[indx]-ox[indy],oy[indx]-oy[indy]]
        material_convmat_imag = material_fft_imag[ox[indx]-ox[indy],oy[indx]-oy[indy]]
        
        material_convmat = torch.complex(material_convmat_real,material_convmat_imag)
        return material_convmat
    
    def _eigen_decomposition_homogenous(self,eps,mu):
        # H to E transformation matirx
        self.P.append(torch.hstack((torch.vstack((torch.zeros_like(self.mu_conv[-1]),-self.mu_conv[-1])),
            torch.vstack((self.mu_conv[-1],torch.zeros_like(self.mu_conv[-1]))))) +
            1/eps * torch.matmul(torch.vstack((self.Kx_norm,self.Ky_norm)), torch.hstack((self.Ky_norm,-self.Kx_norm))))
        # E to H transformation matrix
        self.Q.append(torch.hstack((torch.vstack((torch.zeros_like(self.eps_conv[-1]),self.eps_conv[-1])),
            torch.vstack((-self.eps_conv[-1],torch.zeros_like(self.eps_conv[-1]))))) +
            1/mu * torch.matmul(torch.vstack((self.Kx_norm,self.Ky_norm)), torch.hstack((-self.Ky_norm,self.Kx_norm))))
        
        E_eigvec = torch.eye(self.P[-1].shape[-1],dtype=self._dtype,device=self._device)
        kz_norm = torch.sqrt(eps*mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        kz_norm = torch.where(torch.imag(kz_norm)<0,torch.conj(kz_norm),kz_norm) # Normalized kz for positive mode
        kz_norm = torch.cat((kz_norm,kz_norm))

        self.kz_norm.append(kz_norm) 
        self.E_eigvec.append(E_eigvec)

    def _eigen_decomposition(self):
        # H to E transformation matirx
        P_tmp = torch.matmul(torch.vstack((self.Kx_norm,self.Ky_norm)), torch.linalg.inv(self.eps_conv[-1]))
        self.P.append(torch.hstack((torch.vstack((torch.zeros_like(self.mu_conv[-1]),-self.mu_conv[-1])),
            torch.vstack((self.mu_conv[-1],torch.zeros_like(self.mu_conv[-1]))))) + torch.matmul(P_tmp, torch.hstack((self.Ky_norm,-self.Kx_norm))))
        # E to H transformation matrix
        Q_tmp = torch.matmul(torch.vstack((self.Kx_norm,self.Ky_norm)), torch.linalg.inv(self.mu_conv[-1]))
        self.Q.append(torch.hstack((torch.vstack((torch.zeros_like(self.eps_conv[-1]),self.eps_conv[-1])),
            torch.vstack((-self.eps_conv[-1],torch.zeros_like(self.eps_conv[-1]))))) + torch.matmul(Q_tmp, torch.hstack((-self.Ky_norm,self.Kx_norm))))
        
        # Eigen-decomposition
        if self.stable_eig_grad is True:
            kz_norm, E_eigvec = Eig.apply(torch.matmul(self.P[-1],self.Q[-1]))
        else:
            kz_norm, E_eigvec = torch.linalg.eig(torch.matmul(self.P[-1],self.Q[-1]))
        
        kz_norm = torch.sqrt(kz_norm)
        self.kz_norm.append(torch.where(torch.imag(kz_norm)<0,-kz_norm,kz_norm)) # Normalized kz for positive mode
        self.E_eigvec.append(E_eigvec)

    def _solve_layer_smatrix(self):
        Kz_norm = torch.diag(self.kz_norm[-1])
        phase = torch.diag(torch.exp(1.j*self.omega*self.kz_norm[-1]*self.thickness[-1]))

        Pinv_tmp = torch.linalg.inv(self.P[-1])
        self.H_eigvec.append(torch.matmul(Pinv_tmp,torch.matmul(self.E_eigvec[-1],Kz_norm)))
        #self.H_eigvec.append(torch.matmul(self.Q[-1],torch.matmul(self.E_eigvec[-1],torch.linalg.inv(Kz_norm)))) # another form

        W = self.E_eigvec[-1]
        V0iV = torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1])
        A = W + V0iV
        B = torch.matmul(W - V0iV, phase)
        BAi = torch.matmul(B, torch.linalg.inv(A))
        C1 = 2*torch.linalg.inv(A - torch.matmul(BAi, B))
        C2 = -torch.matmul(C1, BAi)
        S11 = torch.matmul(W, torch.matmul(phase, C1) + C2)
        S12 = torch.matmul(W, torch.matmul(phase, C2) + C1) - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        self.layer_S11.append(S11)
        self.layer_S12.append(S12)
        self.layer_S21.append(S12)
        self.layer_S22.append(S11)

    def _RS_prod(self,Sm,Sn):
        # S11 = S[0] / S21 = S[1] / S12 = S[2] / S22 = S[3]

        tmp1 = torch.linalg.inv(torch.eye(2*self.order_N,dtype=self._dtype,device=self._device) - torch.matmul(Sm[2],Sn[1]))
        tmp2 = torch.linalg.inv(torch.eye(2*self.order_N,dtype=self._dtype,device=self._device) - torch.matmul(Sn[1],Sm[2]))

        # Layer S-matrix
        S11 = torch.matmul(Sn[0],torch.matmul(tmp1,Sm[0]))
        S21 = Sm[1] + torch.matmul(Sm[3],torch.matmul(tmp2,torch.matmul(Sn[1],Sm[0])))
        S12 = Sn[2] + torch.matmul(Sn[0],torch.matmul(tmp1,torch.matmul(Sm[2],Sn[3])))
        S22 = torch.matmul(Sm[3],torch.matmul(tmp2,Sn[3]))

        return S11, S21, S12, S22