import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.colors import LinearSegmentedColormap

class ImageData:
    Intensity: torch.tensor = None
    ImageX: torch.tensor = None
    ImageY: torch.tensor = None
    ImageZ: torch.tensor = None
    VPF: torch.tensor = None

def cartesian_to_polar(x, y):
    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return rho, theta


def CalculateAbbeImage(scatter, projector):
    # 获取基本参数
    wavelength = projector.Wavelength
    M = projector.Magnification
    # 像距
    SimulationRange = projector.FocusRange       
    target_nf = scatter.resolution_x
    target_ng = scatter.resolution_y 
    # 获取物面频谱的坐标
    scatter_fm = (scatter.ScatterResult['Kx_norm_dn'])
    scatter_gm = (scatter.ScatterResult['Ky_norm_dn'])  
    device = scatter_fm.device
    
    # 成像结果初始化
    Intensity = torch.zeros(len(scatter.ScatterResult['z_posi'].keys()), len(SimulationRange), target_nf, target_ng, device=device)
    # VPF初始化
    VPF = torch.zeros((len(scatter.ScatterResult['z_posi'].keys()), len(SimulationRange), 2, 2*scatter.Order[0] + 1,2*scatter.Order[1] + 1)
                      , dtype=torch.complex128, device=device) 
     
    # 计算validPupil
    f_valid = torch.zeros_like(scatter_fm, dtype=scatter_fm.dtype, device=device)
    g_valid = torch.zeros_like(scatter_gm, dtype=scatter_gm.dtype, device=device)
    validPupil =  torch.where(projector.Cal_valid(scatter_fm, scatter_gm))
    
    # 根据validPupil筛选出有效的频率坐标
    f_valid[validPupil] = scatter_fm[validPupil]
    g_valid[validPupil] = scatter_gm[validPupil]
    # 计算极坐标（用于像差计算）
    rho_valid, theta_valid = cartesian_to_polar(f_valid, g_valid)
    fgSquare = rho_valid**2
                    
    # 倾斜因子
    obliquityFactor = torch.pow((1 - fgSquare) / (1 - ((1 / M) ** 2) * fgSquare), 0.25)
    
    # 默认为圆形的zernike基函数，需要对NA进行归一化
    rho_valid_norm = rho_valid/projector.SigmaOut
    aberration = projector.CalculateAberrationFast(rho_valid_norm, theta_valid, 0)
    
    # 高斯型Apodization  
    r = torch.zeros_like(fgSquare) # 此时“高斯型Apodization ”不生效
    # r = rho_calc
    sigma = torch.tensor(0.1)
    apodization = torch.exp(-r**2 / (2.0 * sigma**2))
                
    for value in scatter.ScatterResult['z_posi'].values():
        valid_source_num = value['k_inc'].shape[-1]
        break

    i_z_posi = 0
    # 离焦（物距）
    for z_posi in scatter.ScatterResult['z_posi'].keys():
        # 获取当前物距下对应的spectrum；
        spectrum = scatter.GetSpectrum(z_posi)
        
        # 离焦（像距）
        for iFocus in range(len(SimulationRange)):
            intensity2D = torch.zeros(target_nf, target_ng, spectrum.shape[-1]).to(device)
            focus = SimulationRange[iFocus]
            
            # 离焦相差
            focusFactor = torch.exp(1j * 2 * torch.pi / wavelength * torch.sqrt(1 - (1/M)**2*fgSquare) * focus)

            for k in range(valid_source_num):
                spectrum_now = spectrum[:,:,k].squeeze()
                for channel in range(spectrum_now.size()[-1]): 
                    # channel = 1 # 0表示成像光瞳只允许x方向偏振通过，1表示只允许y
                    # 获取光瞳允许通过的Floquet_mode强度（spectrum）
                    spectrum_channel = torch.zeros_like(spectrum_now[:, channel].squeeze(), dtype=spectrum_now[:, channel].dtype, device=spectrum_now[:, channel].device)
                    spectrum_channel[validPupil] = spectrum_now[:, channel].squeeze()[validPupil]
                    
                    # spectrum * H，H是传递函数
                    Spectrum_H = (spectrum_channel * apodization * obliquityFactor * torch.exp(1j * 2 * torch.pi * aberration) * focusFactor).reshape(2*scatter.Order[0] + 1,2*scatter.Order[1] + 1)
                    VPF[i_z_posi, iFocus, channel, :, :] += Spectrum_H
                    intensityTemp = torch.zeros(target_nf, target_ng)
                    
                    # 频域补零，提高分辨率到设定值；
                    # 假设 Spectrum_H 是 (2*Ox+1, 2*Oy+1)，DC 在中心
                    target_size_Spectrum = pad_spectrum_centered(Spectrum_H, (target_nf, target_ng))
                    
                    # 经过4f系统中的最后一个透镜，正确做法是使用fft，但是由于ifft与fft的结果只差一个坐标对称，为了可视化效果（fft是倒立翻转的像），这里使用ifft；
                    intensityTemp = (torch.abs(target_nf * target_ng * torch.fft.ifft2(torch.fft.ifftshift(target_size_Spectrum)))).to(device)
                    # Ex + Ey
                    intensity2D[:, :, k] = intensity2D[:, :, k] + intensityTemp ** 2
            
            # ax = plot_polarization_analysis(VPF[i_z_posi, iFocus, :, :, :], f_calc=scatter_fm, g_calc=scatter_gm)
            # 多个角度入射平面波成像结果像加
            intensity2D_sum = torch.sum(intensity2D, dim=2)            

            # 将结果写入Intensity
            Intensity[i_z_posi, iFocus, :, :] = intensity2D_sum
        i_z_posi += 1

    farfieldImage = ImageData()
    farfieldImage.Intensity = Intensity
    farfieldImage.VPF = VPF

    return farfieldImage

def pad_spectrum_centered(spectrum, target_shape):
    """
    将中心对齐的频谱（DC at center）零填充到 target_shape，
    保持 DC 严格位于新网格的中心。
    
    Args:
        spectrum: (H, W), 奇数 or 偶数, DC at [H//2, W//2]
        target_shape: (H_new, W_new)
    
    Returns:
        padded: (H_new, W_new), zero-padded, DC at [H_new//2, W_new//2]
    """
    H, W = spectrum.shape
    H_new, W_new = target_shape

    # 原 DC 位置（整数除法，对奇/偶都适用）
    dc_old = (H // 2, W // 2)
    dc_new = (H_new // 2, W_new // 2)

    # 创建目标张量
    padded = torch.zeros((H_new, W_new), 
                         dtype=spectrum.dtype, 
                         device=spectrum.device)

    # 计算复制区域的偏移（以 DC 为锚点）
    # 左上角偏移 = 新DC - 旧DC
    offset_h = dc_new[0] - dc_old[0]
    offset_w = dc_new[1] - dc_old[1]

    # 目标区域（可能被裁剪，但通常不会，因为 target >= original）
    dst_h_start = offset_h
    dst_h_end = offset_h + H
    dst_w_start = offset_w
    dst_w_end = offset_w + W

    # 安全复制（防止越界，虽然一般不会）
    h_valid = min(dst_h_end, H_new) - max(dst_h_start, 0)
    w_valid = min(dst_w_end, W_new) - max(dst_w_start, 0)

    if h_valid > 0 and w_valid > 0:
        padded[
            max(dst_h_start, 0):min(dst_h_end, H_new),
            max(dst_w_start, 0):min(dst_w_end, W_new)
        ] = spectrum[
            max(-dst_h_start, 0):max(-dst_h_start, 0) + h_valid,
            max(-dst_w_start, 0):max(-dst_w_start, 0) + w_valid
        ]

    return padded

def compute_weighted_pupil_phase(Epx, Epy):
    """
    基于强度加权的向量平均方法，计算每个像素点的“公共相位”pupil_phs
    """
    I_total = np.abs(Epx)**2 + np.abs(Epy)**2
    mask = I_total > 0
    w_x = np.zeros_like(I_total)
    w_y = np.zeros_like(I_total)
    
    w_x[mask] = np.abs(Epx[mask])**2 / I_total[mask]
    w_y[mask] = np.abs(Epy[mask])**2 / I_total[mask]

    phase_x = np.angle(Epx)
    phase_y = np.angle(Epy)

    avg_cos = w_x * np.cos(phase_x) + w_y * np.cos(phase_y)
    avg_sin = w_x * np.sin(phase_x) + w_y * np.sin(phase_y)

    return np.arctan2(avg_sin, avg_cos)

def create_circular_colormap():
    """创建首尾一致的圆形色图"""
    color_points = [
        (0.0, 0.0, 0.0),      # start: 黑色
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),      # middle: 白
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),      # end/start: 黑色
    ]
    return LinearSegmentedColormap.from_list("circular_custom", color_points, N=256)

def cart2pol(x, y):
    """笛卡尔坐标转极坐标"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pupilshowplr(
    plr,
    num: int,
    color='k',
    ax=None,
    cmap_style='custom',
    f_calc=None,
    g_calc=None
):
    """
    可视化瞳孔平面上的偏振分布。
    """
    px = plr[:, :, 0]
    py = plr[:, :, 1]
    N_full = px.shape[0]

    # 下采样
    scale = max(1, N_full // num)
    px = px[::scale, ::scale]
    py = py[::scale, ::scale]
    N = px.shape[0]

    # 处理坐标
    if f_calc is None or g_calc is None:
        coord_vals = np.linspace(-1, 1, N)
        xx, yy = np.meshgrid(coord_vals, coord_vals)
    else:
        x_full = f_calc.reshape(N_full, N_full)
        y_full = g_calc.reshape(N_full, N_full)
        xx = x_full[::scale, ::scale]
        yy = y_full[::scale, ::scale]

    # 计算网格步长
    if N > 1:
        step_x = np.abs(xx[1, 0] - xx[0, 0])
        step_y = np.abs(yy[0, 1] - yy[0, 0])
        step = min(step_x, step_y)
    else:
        step_x = 1.0
        step_y = 1.0
        step = 1.0
    r = step * 0.45

    # 归一化幅度
    Ax1 = np.abs(px)
    Ay1 = np.abs(py)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Ax = Ax1 / np.sqrt(Ax1**2 + Ay1**2)
        Ay = Ay1 / np.sqrt(Ax1**2 + Ay1**2)
    Ax[np.isnan(Ax)] = 1.0
    Ay[np.isnan(Ay)] = 0.0

    phs_x = np.angle(px)
    phs_y = np.angle(py)
    delta_phs = phs_y - phs_x
    delta_phs = delta_phs % (2 * np.pi)  # 处理相位范围

    # 圆外掩码
    rho, _ = cart2pol(xx, yy)
    outside = rho > 1
    Ax[outside] = np.nan
    Ay[outside] = np.nan

    # 箭头形状
    h, w = 0.2, 0.2
    ar = np.array([[-w/2, h], [0, 0], [w/2, h]])

    # 色图
    if cmap_style == 'custom':
        colors = [(0.0, 0.0, 0.0), 'green', 'white', 'magenta', (0.0, 0.0, 0.0)]
        cmap = LinearSegmentedColormap.from_list('custom_polar', colors, N=256)
    else:
        cmap = plt.get_cmap(cmap_style)

    # 展平有效点
    mask = ~np.isnan(Ax)
    xx_flat = xx[mask]
    yy_flat = yy[mask]
    Ax_flat = Ax[mask]
    Ay_flat = Ay[mask]
    delta_phs_flat = delta_phs[mask]
    phs_x_flat = phs_x[mask]
    phs_y_flat = phs_y[mask]

    # 分类偏振态
    case1 = np.abs(np.sin(delta_phs_flat)) < 1e-2  # 线性
    case2 = np.abs(np.cos(delta_phs_flat)) < 1e-2  # 圆形
    case3 = np.mod(delta_phs_flat, 2*np.pi) < np.pi  # y滞后x

    k1 = np.where(case1)[0]  # 线性
    k2 = np.where(case2 & case3)[0]   # 右圆
    k3 = np.where(case2 & ~case3)[0]  # 左圆
    k4 = np.where(~case1 & ~case2 & case3)[0]   # 右椭圆
    k5 = np.where(~case1 & ~case2 & ~case3)[0]  # 左椭圆

    # 设置轴
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')

    from matplotlib.patches import Rectangle

    # 1. 线性偏振: 细长矩形
    if len(k1) > 0:
        x1 = xx_flat[k1]
        y1 = yy_flat[k1]
        delta1 = delta_phs_flat[k1]
        colors = cmap(delta1 / (2 * np.pi))  # 归一化到 [0,1]

        u1 = np.sign(np.cos(phs_x_flat[k1])) * Ax_flat[k1]
        v1 = np.sign(np.cos(phs_y_flat[k1])) * Ay_flat[k1]

        in_circle = (x1**2 + y1**2) < 1.0
        x1, y1, u1, v1, colors = x1[in_circle], y1[in_circle], u1[in_circle], v1[in_circle], colors[in_circle]

        if len(x1) > 0:
            for xi, yi, ui, vi, fc in zip(x1, y1, u1, v1, colors):
                length = step * 0.6   # 矩形长度
                width = step * 0.1    # 矩形宽度
                angle_rad = np.arctan2(vi, ui)
                angle_deg = np.degrees(angle_rad)

                # 单位方向向量（作为 numpy 数组）
                norm_vec = np.hypot(ui, vi)
                if norm_vec == 0:
                    continue
                dx = ui / norm_vec
                dy = vi / norm_vec
                direction = np.array([dx, dy])
                perpendicular = np.array([-dy, dx])  # 垂直方向

                # 计算左下角：从中心后退 length/2 沿反方向，再左移 width/2
                corner = np.array([xi, yi]) \
                          + (-length/2) * direction \
                          + (-width/2) * perpendicular
                # corner = np.array([xi-length/2, yi-width/2])

                rect = Rectangle(
                    (corner[0], corner[1]),
                    width=length,
                    height=width,
                    angle=angle_deg,
                    linewidth=0.8,
                    edgecolor='k',
                    facecolor=fc,
                    capstyle='projecting'
                )
                ax.add_patch(rect)
                
    # 1. 线性偏振: 箭头
    # if len(k1) > 0:
    #     x1 = xx_flat[k1]
    #     y1 = yy_flat[k1]
    #     # --- 或者，方法2：基于网格间距动态缩放 ---
    #     # 这样箭头大小会随图像分辨率自适应
    #     arrow_scale_factor = 0.6  # 例如，箭头长度是网格间距的 50%
    #     u1 = np.sign(np.cos(phs_x_flat[k1])) * Ax_flat[k1] * (step * arrow_scale_factor)
    #     v1 = np.sign(np.cos(phs_y_flat[k1])) * Ay_flat[k1] * (step * arrow_scale_factor)
        
    #     in_circle = (x1**2 + y1**2) < 1.0
    #     x1, y1, u1, v1 = x1[in_circle], y1[in_circle], u1[in_circle], v1[in_circle]
    #     if len(x1) > 0:
    #         # 注意：因为 u1, v1 已经是物理长度，scale 参数需要相应调整
    #         # scale=1 意味着 1 单位数据长度 = 1 单位绘图长度
    #         ax.quiver(x1, y1, u1, v1, 
    #                   color='k', 
    #                   scale=0.5,        # 关键：改为 1.0
    #                   scale_units='xy', # 非常重要：让 quiver 知道 u1, v1 是数据坐标单位
    #                   width=0.003, 
    #                   headwidth=3,
    #                   headlength=5,     # 可选：调整箭头头部长度
    #                   headaxislength=4.5 # 可选：调整箭头头部轴线长度
    #                   )
    
    # 椭圆绘制函数
    phi = np.linspace(0, 2*np.pi, 361)

    def plot_ellipses(k_idx, alpha, sign_arrow=1, color=color):
        M = len(k_idx)
        if M == 0:
            return

        ax_k = Ax_flat[k_idx]
        ay_k = Ay_flat[k_idx]
        x0_k = xx_flat[k_idx]
        y0_k = yy_flat[k_idx]
        delta_k = delta_phs_flat[k_idx]

        # a_major = r * np.maximum(ax_k, ay_k)
        # a_minor = r * np.minimum(ax_k, ay_k)
        
        # xo = a_major[:, None] * np.cos(phi)
        # yo = a_minor[:, None] * np.sin(phi)
        
        
        # 计算椭圆的实际长短轴
        # 参考标准偏振椭圆参数
        cos_delta = np.cos(delta_k)
        sin_delta = np.sin(delta_k)
        
        # 椭圆的半长轴 a 和半短轴 b
        # 公式来自：椭圆的特征值
        A = ax_k**2 + ay_k**2
        B = np.sqrt((ax_k**2 - ay_k**2)**2 + 4 * ax_k**2 * ay_k**2 * cos_delta**2)
        a = np.sqrt(0.5 * (A + B))  # 半长轴
        b = np.sqrt(0.5 * (A - B))  # 半短轴
        
        # 现在用 a 和 b 作为椭圆的轴
        xo = r * a[:, None] * np.cos(phi)
        yo = r * b[:, None] * np.sin(phi)
        
        cos_alpha = np.cos(alpha)[:, None]
        sin_alpha = np.sin(alpha)[:, None]
        x2 = x0_k[:, None] + xo * cos_alpha - yo * sin_alpha
        y2 = y0_k[:, None] + xo * sin_alpha + yo * cos_alpha
        

        delta_norm = (delta_k) / (2 * np.pi)
        face_color = cmap(delta_norm)

        for i in range(M):
            if (x0_k[i]**2 + y0_k[i]**2) > 1.0:
                continue
            ax.plot(x2[i], y2[i], color=color, linewidth=1.2)
            ax.fill(x2[i], y2[i], color=face_color[i], alpha=0.7, edgecolor='k', linewidth=0.8)

        centers_in_circle = (x0_k**2 + y0_k**2) <= 1.0
        if not np.any(centers_in_circle):
            return

        x0_k = x0_k[centers_in_circle]
        y0_k = y0_k[centers_in_circle]
        ax_k = ax_k[centers_in_circle]
        ay_k = ay_k[centers_in_circle]
        alpha = alpha[centers_in_circle]
        cos_alpha = np.cos(alpha)[None, :]
        sin_alpha = np.sin(alpha)[None, :]
        
        arx = np.tile(ar[:, 0], (len(x0_k), 1)).T * r
        ary = sign_arrow * np.tile(ar[:, 1], (len(x0_k), 1)).T * r
        ax_arrow = arx + r * ax_k
        ay_arrow = ary
        x_arrow = x0_k[:, None] + (ax_arrow * cos_alpha - ay_arrow * sin_alpha).T
        y_arrow = y0_k[:, None] + (ax_arrow * sin_alpha + ay_arrow * cos_alpha).T
        ax.plot(x_arrow.T, y_arrow.T, 'r', linewidth=1.4)

    # 绘制不同类型的椭圆
    alpha_k2 = np.zeros(len(k2))
    plot_ellipses(k2, alpha_k2, sign_arrow=-1, color=color)

    alpha_k3 = np.zeros(len(k3))
    plot_ellipses(k3, alpha_k3, sign_arrow=1, color=color)

    if len(k4) > 0:
        Ax4 = Ax_flat[k4]
        Ay4 = Ay_flat[k4]
        delta4 = delta_phs_flat[k4]
        denom = (Ax4**2 - Ay4**2)
        denom[denom == 0] = 1e-8
        alpha_k4 = 0.5 * np.arctan2(2 * Ax4 * Ay4 * np.cos(delta4), denom)
        plot_ellipses(k4, alpha_k4, sign_arrow=-1, color=color)

    if len(k5) > 0:
        Ax5 = Ax_flat[k5]
        Ay5 = Ay_flat[k5]
        delta5 = delta_phs_flat[k5]
        denom = (Ax5**2 - Ay5**2)
        denom[denom == 0] = 1e-8
        alpha_k5 = 0.5 * np.arctan2(2 * Ax5 * Ay5 * np.cos(delta5), denom)
        plot_ellipses(k5, alpha_k5, sign_arrow=1, color=color)

    # 设置范围
    lim = 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    sm.set_clim(vmin=0.0, vmax=2*np.pi)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    
    ticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    def to_pi_label(v):
        if np.isclose(v, 0.0): return r"$0$"
        if np.isclose(v, np.pi/2): return r"$\pi/2$"
        if np.isclose(v, np.pi): return r"$\pi$"
        if np.isclose(v, 3*np.pi/2): return r"$3\pi/2$"
        if np.isclose(v, 2*np.pi): return r"$2\pi$"
        return f"{v:.2f}"
    
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([to_pi_label(t) for t in ticks])
    cbar.set_label('angle (rad)')
    
    # Debug print
    print(f"Valid points: {mask.sum()}")
    print(f"Linear: {len(k1)}, Right Circ: {len(k2)}, Left Circ: {len(k3)}")
    print(f"Right Ellip: {len(k4)}, Left Ellip: {len(k5)}")
  
    return ax

def plot_polarization_analysis(plr, num=20, cmap_style='plasma', f_calc=None, g_calc=None):
    """
    绘制包含电场强度、相位差以及偏振态的分析图。
    """
    # 处理torch张量
    try:
        import torch
        if isinstance(plr, torch.Tensor):
            plr = plr.cpu().numpy()
    except ImportError:
        pass
    if plr.shape[0] == 2 and len(plr.shape) == 3:
        plr = np.transpose(plr, (1, 2, 0))
    plr = plr.astype(np.complex128)
    N_full = plr.shape[0]

    # 处理f_calc和g_calc
    if f_calc is not None:
        try:
            import torch
            if isinstance(f_calc, torch.Tensor):
                f_calc = f_calc.cpu().numpy()
        except ImportError:
            pass
    if g_calc is not None:
        try:
            import torch
            if isinstance(g_calc, torch.Tensor):
                g_calc = g_calc.cpu().numpy()
        except ImportError:
            pass

    border = 1
    # 计算坐标
    if f_calc is not None and g_calc is not None:
        x_full = (f_calc.reshape(N_full, N_full))[:-border,:-border]
        y_full = (g_calc.reshape(N_full, N_full))[:-border,:-border]
        intensity = np.abs(plr[:-border, :-border, 0])**2 + np.abs(plr[:-border, :-border, 1])**2
    else:
        coord_vals_full = np.linspace(-1, 1, N_full)
        x_full, y_full = np.meshgrid(coord_vals_full, coord_vals_full)

    # 圆外掩码
    rho_full = np.sqrt(x_full**2 + y_full**2)
    outside_full = rho_full > 1

    # 强度
    intensity[outside_full] = np.nan

    # 相位
    phase = compute_weighted_pupil_phase(plr[:-border, :-border, 0], plr[:-border, :-border, 1])
    phase[outside_full] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    extent = [x_full.min(), x_full.max(), y_full.min(), y_full.max()]

    # 第一个子图 - 强度
    im0 = axes[0].imshow(intensity, cmap='hot', origin='lower', extent=extent)
    axes[0].set_title('Intensity')
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
    axes[0].add_artist(circle)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # 第二个子图 - 相位
    cmap_phase = create_circular_colormap()
    im1 = axes[1].imshow(phase, origin='lower', cmap=cmap_phase, extent=extent, vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Phase')
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].add_artist(plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
    cbar = fig.colorbar(im1, ax=axes[1], shrink=0.8)
    
    ticks = np.array([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
    def to_pi_label(v):
        if np.isclose(v, -np.pi):   return r"$-\pi$"
        if np.isclose(v, -np.pi/2): return r"$-\pi/2$"
        if np.isclose(v, 0.0):       return r"$0$"
        if np.isclose(v, np.pi/2):   return r"$\pi/2$"
        if np.isclose(v, np.pi):     return r"$\pi$"
        return f"{v:.2f}"
    
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([to_pi_label(t) for t in ticks])
    cbar.set_label('angle (rad)')

    # 第三个子图 - 偏振态
    ax = pupilshowplr(plr, num=num, ax=axes[2], f_calc=f_calc, g_calc=g_calc)
    ax.set_title('Polarization States')
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
    ax.add_artist(circle)
    
    plt.tight_layout()
    plt.show()





