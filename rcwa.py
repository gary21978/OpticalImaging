import warnings
import torch

class rcwa:
    # Simulation setting
    def __init__(self,freq,order,L,*,
            dtype=torch.complex64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
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
        '''

        # Hardware
        if dtype != torch.complex64 and dtype != torch.complex128:
            warnings.warn('Invalid simulation data type. Set as torch.complex64.',UserWarning)
            self._dtype = torch.complex64
        else:
            self._dtype = dtype
        self._device = device

        # Simulation parameters
        self.freq = torch.as_tensor(freq,dtype=self._dtype,device=self._device) # unit^-1
        self.omega = 2*torch.pi*freq
        self.L = torch.as_tensor(L,dtype=self._dtype,device=self._device)
        self.stable_inverse = True
        self.fast_exp = False

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
        self.eps_out = torch.tensor(1.,dtype=self._dtype,device=self._device)

        # Internal layers
        self.layer_N = 0  # total number of layers
        self.thickness = []
        self.eps_conv = []
        self.P = []
        self.Q = []
        self.PQ = []

        # Internal layer eigenmodes
        self.kz_norm, self.E_eigvec, self.H_eigvec = [], [], []

        # Single layer scattering matrices
        self.layer_S11, self.layer_S21, self.layer_S12, self.layer_S22 = [], [], [], []

        self.exp_constant = torch.tensor([[                      0,-0.10036558103014462001, -0.00802924648241156960, -0.00089213849804572995,                       0],
                                          [                      0, 0.39784974949964507614,  1.36783778460411719922,  0.49828962252538267755, -0.00063789819459472330],
                                          [-10.9676396052962062593, 1.68015813878906197182,  0.05717798464788655127, -0.00698210122488052084,  0.00003349750170860705],
                                          [ -0.0904316832390810561,-0.06764045190713819075,  0.06759613017704596460,  0.02955525704293155274, -0.00001391802575160607],
                                          [                      0,                      0, -0.09233646193671185927, -0.01693649390020817171, -0.00001400867981820361]])
        #self.layer_exp = []

    def add_input_layer(self,eps=1.):
        '''
            Add input layer
            - If this function is not used, simulation will be performed under free space input layer.

            Parameters
            - eps: relative permittivity
        '''

        self.eps_in = torch.as_tensor(eps,dtype=self._dtype,device=self._device)
        self.Sin = []

    def add_output_layer(self,eps=1.):
        '''
            Add output layer
            - If this function is not used, simulation will be performed under free space output layer.

            Parameters
            - eps: relative permittivity
        '''

        self.eps_out = torch.as_tensor(eps,dtype=self._dtype,device=self._device)
        self.Sout = []

    def set_incident_angle(self,inc_ang,azi_ang,angle_layer='input'):
        '''
            Set incident angle

            Parameters
            - inc_ang: incident angle (unit: radian)
            - azi_ang: azimuthal angle (unit: radian)
            - angle_layer: reference layer to calculate angle ('i', 'in', 'input' / 'o', 'out', 'output')
        '''

        self.inc_ang = torch.as_tensor(inc_ang,dtype=self._dtype,device=self._device)
        self.azi_ang = torch.as_tensor(azi_ang,dtype=self._dtype,device=self._device)
        if angle_layer in ['i', 'in', 'input']:
            self.angle_layer = 'input'
        elif angle_layer in ['o', 'out', 'output']:
            self.angle_layer = 'output'
        else:
            warnings.warn('Invalid angle layer. Set as input layer.',UserWarning)
            self.angle_layer = 'input'
        self._kvectors()

    def add_layer(self,thickness,eps=1.):
        '''
            Add internal layer

            Parameters
            - thickness: layer thickness (unit: length)
            - eps: relative permittivity
        '''
        is_homogenous = (type(eps) == float) or (type(eps) == complex) or (eps.dim() == 0) or ((eps.dim() == 1) and eps.shape[0] == 1)
        if is_homogenous:
            E = eps*torch.eye(self.order_N,dtype=self._dtype,device=self._device)
        else:
            E = self._material_conv(eps)
            
        self.eps_conv.append(E)
        self.layer_N += 1
        self.thickness.append(thickness)
        
        if self.fast_exp:
            self._compute_layer_exp()
        else:
            if is_homogenous:
                self._eigen_decomposition_homogenous(eps)
            else:
                self._eigen_decomposition()
            self._solve_layer_smatrix()

    def add_defect(self, layer_num=0, deps=None):
        deps_conv = self._material_conv(deps)
        self._update_layer_smatrix(layer_num, deps_conv)

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

    def field_xy(self,x_axis,y_axis):
        '''
            XY-plane field distribution at the selected layer.

            Parameters
            - x_axis: x-direction sampling coordinates (torch.Tensor)
            - y_axis: y-direction sampling coordinates (torch.Tensor)
            - z_prop: z-direction distance from the upper boundary of the layer and should be negative.

            Return
            - [Ex, Ey, Ez] (list[torch.Tensor])
        '''

        if type(x_axis) != torch.Tensor or type(y_axis) != torch.Tensor:
            warnings.warn('x and y axis must be torch.Tensor type. Return None.',UserWarning)
            return None
        
        # [x, y, diffraction order]
        x_axis = x_axis.reshape([-1,1,1])
        y_axis = y_axis.reshape([1,-1,1])
        
        if self.angle_layer == 'input':
            eps = self.eps_in if hasattr(self,'eps_in') else 1.
        else:
            eps = self.eps_out if hasattr(self,'eps_out') else 1.

        if self.angle_layer == 'input':
            port = 1 if self.source_direction == 'forward' else 3
        else:   
            port = 0 if self.source_direction == 'forward' else 2

        Kz_norm_dn = torch.sqrt(eps - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        if self.angle_layer == 'input':
            Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])
        else:
            Kz_norm_dn = -torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])

        Exy = self.S[port]@self.E_i
        Ex_mn = Exy[:self.order_N]
        Ey_mn = Exy[self.order_N:]
        Ez_mn = (self.Kx_norm_dn.reshape([-1,1])*Ex_mn + self.Ky_norm_dn.reshape([-1,1])*Ey_mn)/Kz_norm_dn    

        # Spatial domain fields
        xy_phase = torch.exp(1.j * self.omega * (self.Kx_norm_dn*x_axis + self.Ky_norm_dn*y_axis))
        Ex = torch.sum(Ex_mn.reshape(1,1,-1)*xy_phase,dim=2)
        Ey = torch.sum(Ey_mn.reshape(1,1,-1)*xy_phase,dim=2)
        Ez = torch.sum(Ez_mn.reshape(1,1,-1)*xy_phase,dim=2)

        return Ex, Ey, Ez
        
    def Floquet_mode(self):
        Kz_norm_dn = torch.sqrt(1.0 - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])
        
        if self.angle_layer == 'input':
            port = 1 if self.source_direction == 'forward' else 3
        else:
            port = 0 if self.source_direction == 'forward' else 2

        Exy = self.S[port]@self.E_i
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
        if self.angle_layer == 'input':
            self.kx0_norm = torch.real(torch.sqrt(self.eps_in)) * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
            self.ky0_norm = torch.real(torch.sqrt(self.eps_in)) * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)
        else:
            self.kx0_norm = torch.real(torch.sqrt(self.eps_out)) * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
            self.ky0_norm = torch.real(torch.sqrt(self.eps_out)) * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)

        # Free space k-vectors and E to H transformation matrix
        self.kx_norm = self.kx0_norm + self.order_x * self.Gx_norm
        self.ky_norm = self.ky0_norm + self.order_y * self.Gy_norm

        kx_norm_grid, ky_norm_grid = torch.meshgrid(self.kx_norm,self.ky_norm,indexing='ij')

        self.Kx_norm_dn = torch.reshape(kx_norm_grid,(-1,))
        self.Ky_norm_dn = torch.reshape(ky_norm_grid,(-1,))
        self.Kx_norm = torch.diag(self.Kx_norm_dn)
        self.Ky_norm = torch.diag(self.Ky_norm_dn)
        self.Vf, _ = self.get_V(1.0)

        if hasattr(self,'Sin'):
            # Input layer k-vectors and E to H transformation matrix
            self.Vi, _ = self.get_V(self.eps_in)

            if self.stable_inverse:
                VfVi1 = self.Vf+self.Vi
                VfVi2 = self.Vf-self.Vi
                # Input layer S-matrix
                try:
                    S11 = 2*torch.linalg.solve(VfVi1, self.Vi)
                    S12 = torch.linalg.solve(VfVi1, VfVi2)
                    S21 = -S12
                    S22 = 2*torch.linalg.solve(VfVi1, self.Vf)
                except RuntimeError:
                    S11 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
                    S12 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
                    S21 = S12
                    S22 = S11

                self.Sin.append(S11)
                self.Sin.append(S21)
                self.Sin.append(S12)
                self.Sin.append(S22)
            else:
                Vtmp1 = torch.linalg.inv(self.Vf+self.Vi)
                Vtmp2 = self.Vf-self.Vi

                # Input layer S-matrix
                self.Sin.append(2*Vtmp1@self.Vi)  # S11
                self.Sin.append(-Vtmp1@Vtmp2)     # S21
                self.Sin.append(Vtmp1@Vtmp2)      # S12
                self.Sin.append(2*Vtmp1@self.Vf)  # S22

        if hasattr(self,'Sout'):
            # Output layer k-vectors and E to H transformation matrix
            self.Vi, _ = self.get_V(self.eps_out)

            if self.stable_inverse:
                VfVo1 = self.Vf+self.Vo
                VfVo2 = self.Vf-self.Vo
                # Output layer S-matrix
                try:
                    S11 = 2*torch.linalg.solve(VfVo1, self.Vf)
                    S12 = -torch.linalg.solve(VfVo1, VfVo2)
                    S21 = S12
                    S22 = 2*torch.linalg.solve(VfVo1, self.Vo)
                except RuntimeError:
                    S11 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
                    S12 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
                    S21 = S12
                    S22 = S11

                self.Sout.append(S11)
                self.Sout.append(S21)
                self.Sout.append(S12)
                self.Sout.append(S22)
            else:
                Vtmp1 = torch.linalg.inv(self.Vf+self.Vo)
                Vtmp2 = self.Vf-self.Vo

                # Output layer S-matrix
                self.Sout.append(2*Vtmp1@self.Vf)  # S11
                self.Sout.append(Vtmp1@Vtmp2)      # S21
                self.Sout.append(-Vtmp1@Vtmp2)     # S12
                self.Sout.append(2*Vtmp1@self.Vo)  # S22

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
        material_convmat = material_fft[ox[indx]-ox[indy],oy[indx]-oy[indy]]

        return material_convmat
    
    def _eigen_decomposition_homogenous(self,eps):
        V, kz_norm = self.get_V(eps)
        KxKx = self.Kx_norm**2
        KxKy = self.Kx_norm*self.Ky_norm
        KyKy = self.Ky_norm**2
        I = torch.eye(self.order_N,dtype=self._dtype,device=self._device)
        P = torch.hstack((torch.vstack((KxKy/eps, I - KxKx/eps)), torch.vstack((KyKy/eps - I, -KxKy/eps))))
        Q = -eps*P
        PQ = torch.kron(torch.eye(2), eps - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)

        self.P.append(P)
        self.Q.append(Q)
        self.PQ.append(PQ)
        self.kz_norm.append(torch.cat((kz_norm,kz_norm)))
        self.E_eigvec.append(torch.eye(2*self.order_N,dtype=self._dtype,device=self._device))
        self.H_eigvec.append(V)

    def _eigen_decomposition(self):
        E = self.eps_conv[-1]
        KxKx = self.Kx_norm**2
        KxKy = self.Kx_norm*self.Ky_norm
        KyKy = self.Ky_norm**2

        if self.stable_inverse:
            try:
                EiKx = torch.linalg.solve(E, self.Kx_norm)
                EiKy = torch.linalg.solve(E, self.Ky_norm)
                KxEiKy = self.Kx_norm@EiKy
                KyEiKx = self.Ky_norm@EiKx
                KxEiKx = self.Kx_norm@EiKx
                KyEiKy = self.Ky_norm@EiKy
            except RuntimeError:
                KxEiKy = KxKy
                KyEiKx = KxKy
                KxEiKx = KxKx
                KyEiKy = KyKy
        else:
            Ei = torch.linalg.inv(E)
            KxEiKy = self.Kx_norm@Ei@self.Ky_norm
            KyEiKx = self.Ky_norm@Ei@self.Kx_norm
            KxEiKx = self.Kx_norm@Ei@self.Kx_norm
            KyEiKy = self.Ky_norm@Ei@self.Ky_norm

        I = torch.eye(self.order_N,dtype=self._dtype,device=self._device)
        P = torch.hstack((torch.vstack((KxEiKy, KyEiKy - I)), torch.vstack((I - KxEiKx, -KyEiKx))))
        Q = torch.hstack((torch.vstack((-KxKy, E - KyKy)), torch.vstack((KxKx - E, KxKy))))
        PQ = P@Q
        # Eigen-decomposition
        kz_norm, E_eigvec = torch.linalg.eig(PQ)
        kz_norm = torch.sqrt(kz_norm)
        kz_norm = torch.where(torch.imag(kz_norm)<0,-kz_norm,kz_norm)

        self.P.append(P)
        self.Q.append(Q)
        self.PQ.append(PQ)
        self.kz_norm.append(kz_norm)
        self.E_eigvec.append(E_eigvec)
        kz_normi = torch.where(kz_norm==0, 0, 1.0/kz_norm)
        H_eigvec = Q@E_eigvec@torch.diag(kz_normi)
        self.H_eigvec.append(H_eigvec)

    def _solve_layer_smatrix(self):
        phase = torch.diag(torch.exp(1.j*self.omega*self.kz_norm[-1]*self.thickness[-1]))
        W = self.E_eigvec[-1]
        V = self.H_eigvec[-1]

        S11 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        S12 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)

        if self.stable_inverse:
            alternative_form = False

            V0iV = -self.Vf@V # inv(Vf)=-Vf
            A = W + V0iV
            B = (W - V0iV)@phase
            try: # invert matrix A
                BAi = torch.linalg.solve(A.t(), B.t()).t()
            except RuntimeError:
                try: # invert matrix B
                    ABi = torch.linalg.solve(B.t(), A.t()).t()
                except RuntimeError: # Neither A nor B are invertible
                    alternative_form = True
                else:
                    T = (B - ABi@A).t()
                    WX = W@phase
                    try:
                        WXC2 = 2*torch.linalg.solve(T, WX.t()).t()
                        WC2 = 2*torch.linalg.solve(T, W.t()).t()
                    except RuntimeError: # singular T
                        alternative_form = True
                    else:
                        WXC1 = -WXC2@ABi
                        WC1 = -WC2@ABi
                        S11 = WXC1 + WC2
                        S12 = WXC2 + WC1 - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
            else:
                T = (A - BAi@B).t()
                WX = W@phase
                try:
                    WXC1 = 2*torch.linalg.solve(T, WX.t()).t()
                    WC1 = 2*torch.linalg.solve(T, W.t()).t()
                except RuntimeError: # singular T
                    alternative_form = True
                else:
                    WXC2 = -WXC1@BAi
                    WC2 = -WC1@BAi
                    S11 = WXC1 + WC2
                    S12 = WXC2 + WC1 - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)

            if alternative_form:
                try:
                    Wi = torch.linalg.inv(W)
                    ViV0 = torch.linalg.solve(V, self.Vf)
                except RuntimeError:
                    pass
                else:
                    C = Wi + ViV0
                    D = Wi - ViV0
                    try:
                        E = torch.linalg.solve(C.t(), D.t()).t()
                        G = C - phase@E@phase@D
                        S11 = torch.linalg.solve(G, phase@(C - E@D))
                        S12 = torch.linalg.solve(G, phase@E@phase@C - D)
                    except RuntimeError:
                        pass
        else:
            V0iV = torch.linalg.inv(self.Vf)@V
            A = W + V0iV
            B = (W - V0iV)@phase
            BAi = B@torch.linalg.inv(A)
            C1 = 2*torch.linalg.inv(A - (BAi@B))
            C2 = -C1@BAi
            S11 = W@(phase@C1 + C2)
            S12 = W@(phase@C2 + C1) - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
           
        self.layer_S11.append(S11)
        self.layer_S12.append(S12)
        self.layer_S21.append(S12)
        self.layer_S22.append(S11)

    def _update_layer_smatrix(self, layer_num=0, deps_conv=None, use_2ndorder=True):
        # baseline states
        E0 = self.eps_conv[layer_num]
        PQ0 = self.PQ[layer_num]
        Q0 = self.Q[layer_num]
        W0 = self.E_eigvec[layer_num]
        Kz0 = self.kz_norm[layer_num]
        Kx = self.Kx_norm
        Ky = self.Ky_norm
        O = torch.zeros_like(E0)

        # PQ perturbation
        Ed = deps_conv
        E0iEd = torch.linalg.solve(E0, Ed)
        E0iKx = torch.linalg.solve(E0, self.Kx_norm)
        E0iKy = torch.linalg.solve(E0, self.Ky_norm)
        PQd11 = Ed + Kx@E0iEd@E0iKx@E0 - Kx@E0iKx@Ed
        PQd12 = Kx@E0iEd@E0iKy@E0 - Kx@E0iKy@Ed
        PQd21 = Ky@E0iEd@E0iKx@E0 - Ky@E0iKx@Ed
        PQd22 = Ed + Ky@E0iEd@E0iKy@E0 - Ky@E0iKy@Ed
        PQd = torch.hstack((torch.vstack((PQd11, PQd21)), torch.vstack((PQd12, PQd22))))
        Qd = torch.hstack((torch.vstack((O, Ed)), torch.vstack((-Ed, O))))
        Q = Q0 + Qd

        # Check validity
        Fnorm = lambda X: torch.sqrt(torch.sum(torch.abs(X)**2)).item()
        Anorm = Fnorm(PQ0)
        Adnorm = Fnorm(PQd)
        cond_num = torch.linalg.cond(W0).item()
        if (Adnorm > Anorm/cond_num):
        #if (Adnorm > 0.005*Anorm): # Practical constraint
            warnings.warn('The perturbation method is invalid here!',UserWarning)

        eig_tol = 1.0/Anorm
        # First-order perturbation
        Kz0sq = Kz0**2
        phi = Kz0sq.unsqueeze(1) - Kz0sq.unsqueeze(0)
        phi = torch.where(torch.abs(phi) < eig_tol, 0, 1.0/phi) # Avoid numerical overflow
        WdW = torch.linalg.solve(W0, PQd@W0)
        ld = torch.diagonal(WdW)
        Kzd = ld*torch.where(Kz0==0, 0, 0.5/Kz0)
        Wd = -W0@(phi*WdW)

        kz_norm = Kz0 + Kzd
        E_eigvec = W0 + Wd

        # Second-order perturbation
        if use_2ndorder:
            WdWd = torch.linalg.solve(W0, PQd@Wd)
            ldd = torch.diagonal(WdWd)
            Kzdd = ldd*torch.where(Kz0==0, 0, 0.5/Kz0)
            secondorder = torch.linalg.solve(W0, Wd@ld-PQd@Wd)
            Wdd = W0@(phi*secondorder)
            
            kz_norm += Kzdd
            E_eigvec += Wdd
        
        kz_normi = torch.where(kz_norm==0, 0, 1.0/kz_norm)
        H_eigvec = Q@E_eigvec@torch.diag(kz_normi)
        self.kz_norm[layer_num] = kz_norm
        self.E_eigvec[layer_num] = E_eigvec
        self.H_eigvec[layer_num] = H_eigvec

        W = E_eigvec
        phase = torch.diag(torch.exp(1.j*self.omega*kz_norm*self.thickness[layer_num]))

        V0iV = torch.linalg.solve(self.Vf, H_eigvec)
        A = W + V0iV
        B = (W - V0iV)@phase

        try: # invert matrix A
            BAi = torch.linalg.solve(A.t(), B.t()).t()
        except RuntimeError:
            try: # invert matrix B
                ABi = torch.linalg.solve(B.t(), A.t()).t()
            except RuntimeError:
                warnings.warn('Fail to compute the scattering matrix.',UserWarning)
                S11 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
                S12 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
            else:
                T = (B - ABi@A).t()
                WX = W@phase
                WXC2 = 2*torch.linalg.solve(T, WX.t()).t()
                WXC1 = -WXC2@ABi
                WC2 = 2*torch.linalg.solve(T, W.t()).t()
                WC1 = -WC2@ABi
                S11 = WXC1 + WC2
                S12 = WXC2 + WC1 - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        else:
            T = (A - BAi@B).t()
            WX = W@phase
            WXC1 = 2*torch.linalg.solve(T, WX.t()).t()
            WXC2 = -WXC1@BAi
            WC1 = 2*torch.linalg.solve(T, W.t()).t()
            WC2 = -WC1@BAi
            S11 = WXC1 + WC2
            S12 = WXC2 + WC1 - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)

        self.layer_S11[layer_num] = S11
        self.layer_S12[layer_num] = S12
        self.layer_S21[layer_num] = S12
        self.layer_S22[layer_num] = S11

    def _RS_prod(self,Sm,Sn):
        # S11 = S[0] / S21 = S[1] / S12 = S[2] / S22 = S[3]
        I = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        O = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
        if self.stable_inverse:
            T1 = I - Sm[2]@Sn[1]
            T2 = I - Sn[1]@Sm[2]
            # Layer S-matrix
            try:
                S11 = Sn[0]@torch.linalg.solve(T1, Sm[0])
                S21 = Sm[1] + Sm[3]@torch.linalg.solve(T2, Sn[1]@Sm[0])
                S12 = Sn[2] + Sn[0]@torch.linalg.solve(T1, Sm[2]@Sn[3])
                S22 = Sm[3]@torch.linalg.solve(T2, Sn[3])
            except RuntimeError: # assuming zero u-
                S11 = Sn[0]@Sm[0]
                S21 = Sm[1]
                S12 = Sn[2]
                S22 = O
        else:
            tmp1 = torch.linalg.inv(I - (Sm[2]@Sn[1]))
            tmp2 = torch.linalg.inv(I - (Sn[1]@Sm[2]))
            # Layer S-matrix
            S11 = Sn[0]@tmp1@Sm[0]
            S21 = Sm[1] + (Sm[3]@tmp2@Sn[1]@Sm[0])
            S12 = Sn[2] + (Sn[0]@tmp1@Sm[2]@Sn[3])
            S22 = Sm[3]@tmp2@Sn[3]

        return S11, S21, S12, S22
    
########################################################################################
#######################  Matrix exponential method   ###################################
########################################################################################
    def _compute_layer_exp(self):
        thickness_thres = 2.0 * min(self.L[0] / self.order[0], self.L[1] / self.order[1])
        if (self.thickness[-1] > thickness_thres):
            n_repeatedSquaring = int(torch.ceil(torch.log2(torch.tensor(self.thickness[-1] / thickness_thres))))
            d_block = self.thickness[-1] / 2**(n_repeatedSquaring)
        else:
            n_repeatedSquaring = 0
            d_block = self.thickness[-1]
        
        k0d = self.omega * d_block
        E = self.eps_conv[-1]
        KxKx = self.Kx_norm**2
        KxKy = self.Kx_norm*self.Ky_norm
        KyKy = self.Ky_norm**2
        
        try:
            EiKx = torch.linalg.solve(E, self.Kx_norm)
            EiKy = torch.linalg.solve(E, self.Ky_norm)
            KxEiKy = self.Kx_norm@EiKy
            KyEiKx = self.Ky_norm@EiKx
            KxEiKx = self.Kx_norm@EiKx
            KyEiKy = self.Ky_norm@EiKy
        except RuntimeError:
            KxEiKy = KxKy
            KyEiKx = KxKy
            KxEiKx = KxKx
            KyEiKy = KyKy

        I = torch.eye(self.order_N,dtype=self._dtype,device=self._device)
        I2 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        P = torch.hstack((torch.vstack((KxEiKy, KyEiKy - I)), torch.vstack((I - KxEiKx, -KyEiKx))))
        Q = torch.hstack((torch.vstack((-KxKy, E - KyKy)), torch.vstack((KxKx - E, KxKy))))

        Pnorm = torch.max(torch.sum(torch.abs(P), dim=0))
        Qnorm = torch.max(torch.sum(torch.abs(Q), dim=0))
        Rnorm = torch.max(Pnorm, Qnorm)
        m = int(torch.ceil(torch.log2(Rnorm*k0d)))
        if m < 0:
            m = 0

        A_12 = -2**(-m)*k0d*P
        A_21 = 2**(-m)*k0d*Q
        A2_11 = A_12 @ A_21
        A2_22 = A_21 @ A_12
        A3_12 = A_12 @ A2_22
        A3_21 = A_21 @ A2_11
        A6_11 = A3_12 @ A3_21
        A6_22 = A3_21 @ A3_12

        B_11 = [[],[],[],[],[]]
        B_12 = [[],[],[],[],[]]
        B_21 = [[],[],[],[],[]]
        B_22 = [[],[],[],[],[]]
        for i in range(5):
            B_11[i] = self.exp_constant[i, 0]*I2 + self.exp_constant[i, 2]*A2_11 + self.exp_constant[i, 4]*A6_11
            B_12[i] = self.exp_constant[i, 1]*A_12 + self.exp_constant[i, 3]*A3_12
            B_21[i] = self.exp_constant[i, 1]*A_21 + self.exp_constant[i, 3]*A3_21
            B_22[i] = self.exp_constant[i, 0]*I2 + self.exp_constant[i, 2]*A2_22 + self.exp_constant[i, 4]*A6_22

        A9_11 = B_11[0] @ B_11[4] + B_12[0] @ B_21[4] + B_11[3]
        A9_12 = B_11[0] @ B_12[4] + B_12[0] @ B_22[4] + B_12[3]
        A9_21 = B_21[0] @ B_11[4] + B_22[0] @ B_21[4] + B_21[3]
        A9_22 = B_21[0] @ B_12[4] + B_22[0] @ B_22[4] + B_22[3]
        
        expA_11 = B_11[1] + (B_11[2] + A9_11) @ A9_11 + (B_12[2] + A9_12) @ A9_21
        expA_12 = B_12[1] + (B_11[2] + A9_11) @ A9_12 + (B_12[2] + A9_12) @ A9_22
        expA_21 = B_21[1] + (B_21[2] + A9_21) @ A9_11 + (B_22[2] + A9_22) @ A9_21
        expA_22 = B_22[1] + (B_21[2] + A9_21) @ A9_12 + (B_22[2] + A9_22) @ A9_22

        for k in range(m):
            expA_11, expA_12, expA_21, expA_22 = expA_11 @ expA_11 + expA_12 @ expA_21,\
                                                 expA_11 @ expA_12 + expA_12 @ expA_22,\
                                                 expA_21 @ expA_11 + expA_22 @ expA_21,\
                                                 expA_21 @ expA_12 + expA_22 @ expA_22
        #appro_exp = [expA_11, expA_12, expA_21, expA_22]
        #self.layer_exp.append(appro_exp) # append T matrix

        # Convert to layer S-matrix
        E = expA_11 + 1j * self.Vf @ expA_21
        F = (1j * expA_12 - self.Vf @ expA_22) @ self.Vf
        T22 = 0.5 * (E + F)
        T21 = 0.5 * (E - F)
        S11 = torch.linalg.inv(T22)
        S12 = -S11 @ T21

        for i in range(n_repeatedSquaring):
            R = I2 - S12 @ S12
            tmp = S12 + S11 @ torch.linalg.solve(R, S12 @ S11)
            S11 = S11 @ torch.linalg.solve(R, S11)
            S12 = tmp

        self.layer_S11.append(S11)
        self.layer_S12.append(S12)
        self.layer_S21.append(S12)
        self.layer_S22.append(S11)

    def _compute_prod(self):
        G11 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        G12 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
        G21 = torch.zeros(2*self.order_N,dtype=self._dtype,device=self._device)
        G22 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        # Layer connection
        for i in range(self.layer_N):
            expA = self.layer_exp[i]
            G11, G12, G21, G22 = G11 @ expA[0] + G12 @ expA[2],\
                                 G11 @ expA[1] + G12 @ expA[3],\
                                 G21 @ expA[0] + G22 @ expA[2],\
                                 G21 @ expA[1] + G22 @ expA[3]
        self.global_exp = [G11, G12, G21, G22]

    def solve_global_tmatrix(self):
        self._compute_prod()
        V_trn, _ = self.get_V(self.eps_out)
        V_ref, _ = self.get_V(self.eps_in)
        A1 = self.global_exp[0] + 1j*self.global_exp[1] @ V_trn
        B1 = self.global_exp[0] - 1j*self.global_exp[1] @ V_trn
        C1 = self.global_exp[2] + 1j*self.global_exp[3] @ V_trn
        D1 = self.global_exp[2] - 1j*self.global_exp[3] @ V_trn

        S11 = 2.0 * torch.linalg.solve(C1 + 1j*V_ref @ A1, 1j*V_ref)
        S12 = -torch.linalg.solve(C1 + 1j*V_ref @ A1, D1 + 1j*V_ref @ B1)
        S21 = A1 @ S11 - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        S22 = A1 @ S12 + B1
        # Save S-matrix
        self.S = [S11, S21, S12, S22]

    def get_V(self, eps):
        kz_norm = torch.sqrt(eps - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        kz_norm = torch.where(torch.imag(kz_norm)<0,torch.conj(kz_norm),kz_norm)
        KxKyKzi = self.Kx_norm*self.Ky_norm*torch.diag(1./kz_norm)
        KzplusKxKxKzi = torch.diag(kz_norm) + self.Kx_norm**2*torch.diag(1./kz_norm)
        KzplusKyKyKzi = torch.diag(kz_norm) + self.Ky_norm**2*torch.diag(1./kz_norm)
        V = torch.hstack((torch.vstack((-KxKyKzi, KzplusKxKxKzi)), torch.vstack((-KzplusKyKyKzi, KxKyKzi))))
        return V, kz_norm