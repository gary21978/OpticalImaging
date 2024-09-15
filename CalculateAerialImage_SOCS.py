import torch

def CalculateAerialImage_SOCS(scatter, TCCMatrix_SOCS, source, projector):
    spectrum, f, g, pupilImage = scatter.CalculateSpectrum(projector, source)
    sp = spectrum.unsqueeze(3).permute(0, 1, 3, 2)
    Es = TCCMatrix_SOCS.unsqueeze(3) * torch.fft.ifftshift(sp, dim=(0, 1))
    E = (f[1] - f[0]) * (g[1] - g[0]) * torch.fft.fft2(Es, dim=(0, 1))
    Esquare = torch.abs(E)**2
    intensity = torch.sum(Esquare, dim=(2, 3)) # sum over xyz and coherent systems
    intensity = torch.fft.fftshift(intensity, dim=(0, 1))
    return intensity, pupilImage
    
def CalculateSOCS(FloquetMode, TCCMatrix_SOCS, numerics):
    n_scatter_x = numerics.ScatterGrid_X
    n_scatter_y = numerics.ScatterGrid_Y
    x_axis = (torch.arange(n_scatter_x)+0.5)/n_scatter_x
    y_axis = (torch.arange(n_scatter_y)+0.5)/n_scatter_y
    x_axis = x_axis.reshape([-1,1,1])
    y_axis = y_axis.reshape([1,-1,1])
    order_x = torch.arange(-numerics.ScatterOrder_X, numerics.ScatterOrder_X + 1)
    order_y = torch.arange(-numerics.ScatterOrder_Y, numerics.ScatterOrder_Y + 1)
    order_x_grid, order_y_grid = torch.meshgrid(order_x,order_y,indexing='ij')
    order_x_dn = torch.reshape(order_x_grid,(-1,))
    order_y_dn = torch.reshape(order_y_grid,(-1,))
    xy_phase = torch.exp(2*torch.pi*1.j*(order_x_dn*x_axis + order_y_dn*y_axis)).unsqueeze(3).unsqueeze(4)
    FM = FloquetMode.reshape([1,1,-1,3,1])
    transposed_kernel = torch.transpose(TCCMatrix_SOCS, 0, 1)
    transposed_kernel = torch.fft.ifftshift(transposed_kernel, dim=(0, 1))
    kernel = transposed_kernel.reshape([1,1,-1,1,TCCMatrix_SOCS.shape[2]])
    S = torch.sum(xy_phase*FM*kernel, dim=2)          # Sum over orders
    intensity = torch.sum(torch.abs(S)**2, dim=(2,3)) # Sum over xyz and coherent systems
    return intensity
