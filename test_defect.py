from rcwa import rcwa
import torch
import matplotlib.pyplot as plt

#func = lambda x: torch.sum(torch.abs(x)**2, axis=-1).squeeze().cpu()
func = lambda x: torch.real(x[:,:,0]).squeeze().cpu()

def Scattering():
    normal_geometry = torch.zeros((256, 256))
    normal_geometry[:, 64:128] = 1
    normal_geometry[:, 192:256] = 1

    defect_geometry = torch.zeros((256, 256))
    #defect_geometry[20:95, 144:150] = 1
    defect_geometry[70:72, 135:137] = 1

    sim_dtype = torch.complex128
    order = [1, 5]
    wavelength = 365
    period = 2000
    n_scatter = 256
    # reflective indices
    n_OX = 1.4745
    n_SIN = 2.1222
    xs = (period/n_scatter)*(torch.arange(n_scatter)+0.5)
    ys = (period/n_scatter)*(torch.arange(n_scatter)+0.5)

    # Direct method
    print('Direct method')
    ss = rcwa(freq=1/wavelength, order=order,
                   L=[period, period],dtype=sim_dtype)
    ss.set_incident_angle(inc_ang=0, azi_ang=0)
    ss.add_layer(thickness=5, eps=(normal_geometry+defect_geometry)*(n_OX**2 - 1) + 1.0)
    ss.add_layer(thickness=20, eps=n_SIN**2)
    ss.solve_global_smatrix()
    ss.source_planewave(amplitude=[1,0], direction='forward')
    [Ex, Ey, Ez], [_, _, _] = ss.field_xy(xs, ys)
    normal_field = torch.stack((Ex, Ey, Ez), 2)
    direct_intensity = func(normal_field)

    # Perturbation method
    print('Baseline result')
    sim = rcwa(freq=1/wavelength, order=order,
                   L=[period, period],dtype=sim_dtype)
    sim.set_incident_angle(inc_ang=0, azi_ang=0)
    sim.add_layer(thickness=5, eps=normal_geometry*(n_OX**2 - 1) + 1.0)
    sim.add_layer(thickness=20, eps=n_SIN**2)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[1,0], direction='forward')
    [Ex, Ey, Ez], [_, _, _] = sim.field_xy(xs, ys)
    normal_field = torch.stack((Ex, Ey, Ez), 2)
    normal_intensity = func(normal_field)
    ####################################
    print('Perturbed result')
    sim.add_defect(layer_num=0, deps=defect_geometry*(n_OX**2 - 1))
    sim.solve_global_smatrix()
    [Ex, Ey, Ez], [_, _, _] = sim.field_xy(xs, ys)
    defective_field = torch.stack((Ex, Ey, Ez), 2)
    pert_intensity = func(defective_field)

    plt.subplot(231)
    #plt.imshow(normal_geometry,cmap='gray')
    plt.imshow(normal_intensity,cmap='jet')
    plt.colorbar()
    plt.title('Normal')
    plt.subplot(232)
    plt.imshow(direct_intensity,cmap='jet')
    plt.colorbar()
    plt.title('Defective')
    plt.subplot(233)
    plt.imshow(direct_intensity - normal_intensity,cmap='jet')
    plt.colorbar()
    plt.title('Difference')
    plt.subplot(234)
    plt.imshow(normal_intensity,cmap='jet')
    plt.colorbar()
    plt.subplot(235)
    plt.imshow(pert_intensity,cmap='jet')
    plt.colorbar()
    plt.subplot(236)
    plt.imshow(pert_intensity - normal_intensity,cmap='jet')
    plt.colorbar()
    plt.show()

Scattering()
