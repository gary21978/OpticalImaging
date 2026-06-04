"""
Inverse-design demo: recover layer thickness from a target field-intensity
pattern using the RCWA FREE method + Adam optimizer.

Target:   a complex dielectric pattern (posts + ring),  thickness = 0.5
Initial:  same pattern,  thickness = 0.35
"""

import torch
import sys
sys.path.insert(0, '.')
from python.rcwa import rcwa


def build_pattern(order, dtype, device):
    """Build a complex non-periodic dielectric pattern.

    Four squared posts of different sizes, a central ring, and a small
    offset rectangle — enough structure to break all symmetries and
    degeneracies without being a simple grating.
    """
    nx = 4 * order[0] + 1
    ny = 4 * order[0] + 1
    x = torch.arange(nx, device=device, dtype=torch.float64)
    y = torch.arange(ny, device=device, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    cx, cy = nx / 2.0, ny / 2.0

    eps = torch.ones(nx, ny, dtype=dtype, device=device)
    eps_val = torch.tensor(2.25, dtype=dtype, device=device)

    # annular ring around centre
    r = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
    ring_inner = nx / 5.0
    ring_outer = nx / 3.5
    eps[(r > ring_inner) & (r < ring_outer)] = eps_val

    # four posts at different positions with different sizes
    posts = [
        (cx - nx / 4.0,  cy - ny / 4.0,  nx / 12.0),
        (cx + nx / 4.0,  cy + ny / 4.0,  nx / 15.0),
        (cx + nx / 3.5,  cy - ny / 3.5,  nx / 18.0),
        (cx - nx / 3.0,  cy + ny / 3.0,  nx / 10.0),
    ]
    for px, py, pr in posts:
        rp = torch.sqrt((xx - px)**2 + (yy - py)**2)
        eps[rp < pr] = eps_val

    # offset rectangular patch
    rx0 = int(nx * 0.15)
    rx1 = int(nx * 0.25)
    ry0 = int(ny * 0.65)
    ry1 = int(ny * 0.80)
    eps[rx0:rx1, ry0:ry1] = eps_val

    return eps


def field_intensity(sim, x_axis, y_axis):
    """Return |Ex|² + |Ey|² + |Ez|²."""
    Ex, Ey, Ez = sim.field_xy(x_axis, y_axis)
    return Ex.abs()**2 + Ey.abs()**2 + Ez.abs()**2


def main():
    dtype = torch.complex128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    order = (8, 8)
    L = (1.0, 1.0)
    freq = 1.0

    n_pts = 64
    x_axis = torch.linspace(0, L[0], n_pts, device=device)
    y_axis = torch.linspace(0, L[1], n_pts, device=device)

    pattern = build_pattern(order, dtype, device)

    # ── ground truth ───────────────────────────────────────────────────────
    thickness_true = 0.5
    sim_true = rcwa(freq, order, L, dtype=dtype, device=device)
    sim_true.set_incident_angle(0.2, 0.0, 'input')
    sim_true.add_input_layer(1.0)
    sim_true.add_output_layer(1.0)
    sim_true.add_layer(thickness_true, pattern)
    sim_true.source_planewave(amplitude=(1.0, 0.0), direction='forward')
    sim_true.solve_global_smatrix()

    with torch.no_grad():
        target = field_intensity(sim_true, x_axis, y_axis)
    print(f'Target  |E|²  range: [{target.min().item():.4f}, {target.max().item():.4f}]')

    # ── optimise ───────────────────────────────────────────────────────────
    thickness_param = torch.tensor(0.35, dtype=torch.float64, device=device,
                                   requires_grad=True)

    optimiser = torch.optim.Adam([thickness_param], lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=40, gamma=0.5)

    n_steps = 100
    print(f"\n{'step':>5s}  {'thickness':>10s}  {'loss':>12s}  {'grad':>12s}")
    print('-' * 46)

    for step in range(n_steps):
        optimiser.zero_grad()

        sim = rcwa(freq, order, L, dtype=dtype, device=device)
        sim.set_incident_angle(0.2, 0.0, 'input')
        sim.add_input_layer(1.0)
        sim.add_output_layer(1.0)
        sim.add_layer(thickness_param, pattern)
        sim.source_planewave(amplitude=(1.0, 0.0), direction='forward')
        sim.solve_global_smatrix()

        intensity = field_intensity(sim, x_axis, y_axis)
        loss = torch.nn.functional.mse_loss(intensity, target)
        loss.backward()

        g = thickness_param.grad.item()
        optimiser.step()
        scheduler.step()

        if step % 10 == 0 or step == n_steps - 1:
            print(f'{step:5d}  {thickness_param.item():10.6f}  {loss.item():12.6e}  {g:12.3e}')

    print()
    print(f'True thickness:  {thickness_true}')
    print(f'Final thickness: {thickness_param.item():.6f}')
    rel_err = abs(thickness_param.item() - thickness_true) / thickness_true * 100
    print(f'Relative error:  {rel_err:.4f}%')
    print(f'Converged: {"YES" if rel_err < 2.0 else "NO"}')


if __name__ == '__main__':
    main()
