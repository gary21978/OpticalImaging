"""
RCWA (Rigorous Coupled Wave Analysis) — PyTorch implementation.

Uses torch for all linear algebra so that autodiff is available.
The FREE method avoids the matrix square root by computing cos(Θ) and
Θ⁻¹·sin(Θ) directly from P·Q via an optimised Taylor expansion.

Reference
---------
- Bader, Blanes & Casas, "Efficient numerical methods for the matrix
  exponential and the matrix cosine and sine", 2019.
"""

import torch
import warnings


# ═══════════════════════════════════════════════════════════════════════════════
#  Taylor-expansion coefficients  (Bader, Blanes & Casas, 2019)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_coeff_table(dtype, device):
    """Order-18 coefficient table."""
    return torch.tensor([
        [ 0.0,                    -0.10036558103014462001, -0.00802924648241156960, -0.00089213849804572995,  0.0                     ],
        [ 0.0,                     0.39784974949964507614,  1.36783778460411719922,  0.49828962252538267755, -0.00063789819459472330  ],
        [-10.9676396052962062593,  1.68015813878906197182,  0.05717798464788655127, -0.00698210122488052084,  0.00003349750170860705  ],
        [-0.0904316832390810561,  -0.06764045190713819075,  0.06759613017704596460,  0.02955525704293155274, -0.00001391802575160607  ],
        [ 0.0,                     0.0,                    -0.09233646193671185927, -0.01693649390020817171, -0.00001400867981820361  ],
    ], dtype=dtype, device=device)


def _make_coeff_table_12(dtype, device):
    """Order-12 coefficient table."""
    return torch.tensor([
        [-0.01860232051462055322, -0.00500702322573317730, -0.57342012296052226390, -0.13339969394389205970],
        [ 4.60000000000000000000,  0.99287510353848683614, -0.13244556105279963884,  0.00172990000000000000],
        [ 0.21169311829980944294,  0.15822438471572672537,  0.16563516943672741501,  0.01078627793157924250],
        [ 0.0,                    -0.13181061013830184015, -0.02027855540589259079, -0.00675951846863086359],
    ], dtype=dtype, device=device)


# ═══════════════════════════════════════════════════════════════════════════════
#  V‑matrix construction and multiplication
# ═══════════════════════════════════════════════════════════════════════════════

def _build_v0(kx_diag, ky_diag, eps):
    """Construct the 4 diagonal vectors of the V₀ matrix.

    Returns v11, v12, v21, v22  (each length n).
    The V matrix is block‑diagonal with 2×2 blocks [v11, v12; v21, v22] at
    positions (i,i) and (i+n,i+n).
    """
    kz = torch.sqrt(eps - kx_diag**2 - ky_diag**2)
    kz = torch.where(kz.imag < 0, kz.conj(), kz)
    inv_kz = torch.where(kz.abs() > 1e-15, 1.0 / kz, torch.zeros_like(kz))
    v11 = -kx_diag * ky_diag * inv_kz
    v12 = -(kz + ky_diag**2 * inv_kz)
    v21 = kz + kx_diag**2 * inv_kz
    v22 = kx_diag * ky_diag * inv_kz
    return v11, v12, v21, v22


def v0_left_mul(v11, v12, v21, v22, B):
    """Left‑multiply by the block‑diagonal V₀ matrix:  C = V₀ @ B.

    B and C are 2n × k.
    """
    n = v11.shape[0]
    C_top = v11.reshape(-1, 1) * B[:n] + v12.reshape(-1, 1) * B[n:]
    C_bot = v21.reshape(-1, 1) * B[:n] + v22.reshape(-1, 1) * B[n:]
    return torch.cat((C_top, C_bot), dim=0)


def v0_right_mul(v11, v12, v21, v22, B):
    """Right‑multiply by the block‑diagonal V₀ matrix:  C = B @ V₀.

    B and C are k × 2n.
    """
    n = v11.shape[0]
    C_left = B[:, :n] * v11.reshape(1, -1) + B[:, n:] * v21.reshape(1, -1)
    C_right = B[:, :n] * v12.reshape(1, -1) + B[:, n:] * v22.reshape(1, -1)
    return torch.cat((C_left, C_right), dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  P and Q construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_pq(eps_conv, Kx, Ky):
    """Construct the 2n×2n block matrices P and Q.

    Parameters
    ----------
    eps_conv : (n, n)  — convolution matrix of relative permittivity
    Kx, Ky   : (n, n)  — diagonal matrices of in‑plane wave‑vector components

    Returns
    -------
    P, Q : (2n, 2n)
    """
    n = eps_conv.shape[0]
    dtype = eps_conv.dtype
    device = eps_conv.device

    I = torch.eye(n, dtype=dtype, device=device)

    Kx2 = Kx @ Kx
    Ky2 = Ky @ Ky
    KxKy = Kx @ Ky

    # E⁻¹·Kx  and  E⁻¹·Ky
    try:
        EiKx = torch.linalg.solve(eps_conv, Kx)
        EiKy = torch.linalg.solve(eps_conv, Ky)
        KxEiKy = Kx @ EiKy
        KyEiKx = Ky @ EiKx
        KxEiKx = Kx @ EiKx
        KyEiKy = Ky @ EiKy
    except RuntimeError:
        KxEiKy = KxKy
        KyEiKx = KxKy
        KxEiKx = Kx2
        KyEiKy = Ky2

    #  P = [  Kx·E⁻¹·Ky          Ky·E⁻¹·Ky − I  ]
    #      [  I − Kx·E⁻¹·Kx     −Ky·E⁻¹·Kx      ]
    #
    # NOTE: the off‑diagonal blocks are swapped relative to the conventional
    # 2×2 block layout.  The resulting matrix equals the transpose of the
    # conventional layout since every block is symmetric.  We mirror this by
    # swapping the block placement below.

    P11 = KxEiKy
    P12 = KyEiKy - I
    P21 = I - KxEiKx
    P22 = -KyEiKx
    P = torch.cat((torch.cat((P11, P12), dim=0), torch.cat((P21, P22), dim=0)), dim=1)

    Q11 = -KxKy
    Q12 = eps_conv - Ky2
    Q21 = Kx2 - eps_conv
    Q22 = KxKy
    Q = torch.cat((torch.cat((Q11, Q12), dim=0), torch.cat((Q21, Q22), dim=0)), dim=1)

    return P, Q


# ═══════════════════════════════════════════════════════════════════════════════
#  Cos(Θ) and Sinc(Θ) = Θ⁻¹·sin(Θ)  directly from P·Q   (no sqrt)
# ═══════════════════════════════════════════════════════════════════════════════

def _matrix_norm1(A):
    """Matrix 1‑norm (max absolute column sum)."""
    return torch.max(torch.sum(torch.abs(A), dim=0))


def compute_cos_sinc_from_pq_order18(P, Q, k0d, coeff):
    """18th‑order Cos(Θ) and Sinc(Θ) directly from P·Q.

    Phases:
      1.  A² = 2⁻²ᵐ·(k₀d)²·(P·Q),  m = ⌈½·log₂ max(‖P·Q‖₁, 1)⌉
      2.  A⁶ = (A²)³
      3.  B⁽ⁱ⁾  = a₀ᵢ·I − a₂ᵢ·A² − a₄ᵢ·A⁶
          B̄⁽ⁱ⁾ = a₁ᵢ·I − a₃ᵢ·A²     (i = 0…4)
      4.  A⁽⁹⁾  = B⁽⁰⁾·B⁽⁴⁾ − A²·(B̄⁽⁰⁾·B̄⁽⁴⁾) + B⁽³⁾
          Ā⁽⁹⁾  = B̄⁽⁰⁾·B⁽⁴⁾ + B⁽⁰⁾·B̄⁽⁴⁾ + B̄⁽³⁾
      5.  C     = B⁽¹⁾ + (B⁽²⁾+A⁽⁹⁾)·A⁽⁹⁾ − A²·(B̄⁽²⁾+Ā⁽⁹⁾)·Ā⁽⁹⁾
          Sinc  = B̄⁽¹⁾ + (B̄⁽²⁾+Ā⁽⁹⁾)·A⁽⁹⁾ + (B⁽²⁾+A⁽⁹⁾)·Ā⁽⁹⁾
      6.  m rounds of double‑angle squaring
    """
    n = P.shape[0]
    dtype = P.dtype
    device = P.device
    I = torch.eye(n, dtype=dtype, device=device)

    # ── Phase 1: A² = 2⁻²ᵐ·(k₀d)²·(P·Q) ────────────────────────────────
    PQ = P @ Q
    with torch.no_grad():
        PQnorm = _matrix_norm1(PQ)
        m_val = int(torch.ceil(0.5 * torch.log2(
            torch.clamp(PQnorm, min=torch.tensor(1.0, device=device)))))
        if m_val < 0:
            m_val = 0

    scale = (2.0 ** (-2 * m_val)) * (k0d.real ** 2)
    A2 = scale * PQ

    # ── Phase 2: A⁶ = (A²)³ ─────────────────────────────────────────────
    A6pow = A2 @ A2 @ A2

    # ── Phase 3: B⁽ⁱ⁾, B̄⁽ⁱ⁾  (i = 0…4) ─────────────────────────────────
    B = []
    Bbar = []
    for i in range(5):
        a0, a1, a2, a3, a4 = coeff[i]
        B.append(a0 * I - a2 * A2 - a4 * A6pow)
        Bbar.append(a1 * I - a3 * A2)

    # ── Phase 4: A⁽⁹⁾, Ā⁽⁹⁾ ────────────────────────────────────────────
    A9 = B[0] @ B[4] - A2 @ (Bbar[0] @ Bbar[4]) + B[3]
    Abar9 = Bbar[0] @ B[4] + B[0] @ Bbar[4] + Bbar[3]

    # ── Phase 5: C (Cos), Sinc ──────────────────────────────────────────
    T1 = B[2] + A9
    T2 = Bbar[2] + Abar9
    C = B[1] + T1 @ A9 - A2 @ (T2 @ Abar9)
    Sinc = Bbar[1] + T2 @ A9 + T1 @ Abar9

    # ── Phase 6: double‑angle squaring ───────────────────────────────────
    for _ in range(m_val):
        Sinc = Sinc @ C
        C = 2.0 * C @ C - I

    return C, Sinc


def compute_cos_sinc_from_pq_order12(P, Q, k0d, coeff):
    """12th‑order Cos(Θ) and Sinc(Θ) directly from P·Q.

    Phases:
      1.  A² = 2⁻²ᵐ·(k₀d)²·(P·Q)
      2.  B⁽ⁱ⁾  = a₀ᵢ·I − a₂ᵢ·A²
          B̄⁽ⁱ⁾ = a₁ᵢ·I − a₃ᵢ·A²     (i = 0…3)
      3.  A⁽⁶⁾  = (B⁽³⁾)² − A²·(B̄⁽³⁾)² + B⁽²⁾
          Ā⁽⁶⁾  = B̄⁽³⁾·B⁽³⁾ + B⁽³⁾·B̄⁽³⁾ + B̄⁽²⁾
      4.  C     = B⁽⁰⁾ + (B⁽¹⁾+A⁽⁶⁾)·A⁽⁶⁾ − A²·(B̄⁽¹⁾+Ā⁽⁶⁾)·Ā⁽⁶⁾
          Sinc  = B̄⁽⁰⁾ + (B̄⁽¹⁾+Ā⁽⁶⁾)·A⁽⁶⁾ + (B⁽¹⁾+A⁽⁶⁾)·Ā⁽⁶⁾
      5.  m rounds of double‑angle squaring
    """
    n = P.shape[0]
    dtype = P.dtype
    device = P.device
    I = torch.eye(n, dtype=dtype, device=device)

    # ── Phase 1 ──────────────────────────────────────────────────────────
    PQ = P @ Q
    with torch.no_grad():
        PQnorm = _matrix_norm1(PQ)
        m_val = int(torch.ceil(0.5 * torch.log2(
            torch.clamp(PQnorm, min=torch.tensor(1.0, device=device)))))
        if m_val < 0:
            m_val = 0

    scale = (2.0 ** (-2 * m_val)) * (k0d.real ** 2)
    A2 = scale * PQ

    # ── Phase 2: B⁽ⁱ⁾, B̄⁽ⁱ⁾  (i = 0…3) ─────────────────────────────────
    B = []
    Bbar = []
    for i in range(4):
        a0, a1, a2, a3 = coeff[i]
        B.append(a0 * I - a2 * A2)
        Bbar.append(a1 * I - a3 * A2)

    # ── Phase 3: A⁽⁶⁾, Ā⁽⁶⁾ ────────────────────────────────────────────
    A6 = B[3] @ B[3] - A2 @ (Bbar[3] @ Bbar[3]) + B[2]
    Abar6 = Bbar[3] @ B[3] + B[3] @ Bbar[3] + Bbar[2]

    # ── Phase 4: C (Cos), Sinc ──────────────────────────────────────────
    T1 = B[1] + A6
    T2 = Bbar[1] + Abar6
    C = B[0] + T1 @ A6 - A2 @ (T2 @ Abar6)
    Sinc = Bbar[0] + T2 @ A6 + T1 @ Abar6

    # ── Phase 5: double‑angle squaring ───────────────────────────────────
    for _ in range(m_val):
        Sinc = Sinc @ C
        C = 2.0 * C @ C - I

    return C, Sinc


# ═══════════════════════════════════════════════════════════════════════════════
#  EF → S‑matrix
# ═══════════════════════════════════════════════════════════════════════════════

def ef_to_s_matrix(E, F):
    """Convert E, F matrices to S‑matrix blocks.

       T₂₂ = (E+F)/2,   T₂₁ = (E−F)/2
       S₁₁ = T₂₂⁻¹,     S₁₂ = −S₁₁·T₂₁
       S₂₁ = S₁₂,       S₂₂ = S₁₁
    """
    T22 = 0.5 * (E + F)
    T21 = 0.5 * (E - F)
    S11 = torch.linalg.inv(T22)
    S12 = -S11 @ T21
    return S11, S12


# ═══════════════════════════════════════════════════════════════════════════════
#  S‑matrix via the FREE path
# ═══════════════════════════════════════════════════════════════════════════════

def compute_s_matrix_free(P, Q, k0d, v0_11, v0_12, v0_21, v0_22, coeff_high, coeff_low,
                          high_order):
    """Compute the layer S‑matrix from P, Q without matrix square root.

    Steps:
      1. [Cos, Sinc] = cos(Θ) and Θ⁻¹·sin(Θ)  from P·Q
      2. V₀Q = V₀·Q
      3. E  = Cos + i·k₀d·V₀Q·Sinc
      4. F  = −(i·k₀d·Sinc·P + V₀Q·Cos·Q⁻¹)·V₀
      5. S‑blocks from E, F
    """
    if high_order:
        Cos, Sinc = compute_cos_sinc_from_pq_order18(P, Q, k0d, coeff_high)
    else:
        Cos, Sinc = compute_cos_sinc_from_pq_order12(P, Q, k0d, coeff_low)

    # 2. V₀Q = V₀ · Q
    V0Q = v0_left_mul(v0_11, v0_12, v0_21, v0_22, Q)

    # 3. E = Cos + i·k₀d·V₀Q·Sinc
    E = Cos + 1.j * k0d.real * (V0Q @ Sinc)

    # 4. F = −(i·k₀d·Sinc·P + V₀Q·Cos·Q⁻¹)·V₀
    part1 = 1.j * k0d.real * (Sinc @ P)
    CosQ = torch.linalg.solve(Q.T, Cos.T).T
    V0Q_CosQ = V0Q @ CosQ
    F = -v0_right_mul(v0_11, v0_12, v0_21, v0_22, part1 + V0Q_CosQ)

    # 5. EF → S
    return ef_to_s_matrix(E, F)


# ═══════════════════════════════════════════════════════════════════════════════
#  RCWA class
# ═══════════════════════════════════════════════════════════════════════════════

class rcwa:
    """Rigorous Coupled Wave Analysis.

    Lorentz‑Heaviside units (c = 1), time harmonics exp(−jωt).

    Parameters
    ----------
    freq : float
        Frequency  (unit: length⁻¹).
    order : (int, int)
        Fourier truncation order  [order_x, order_y].
    L : (float, float)
        Lattice constants  [Lx, Ly]  (unit: length).
    dtype : torch.dtype
        Complex working precision.  complex128 recommended for orders ≥ 6.
    device : torch.device
    high_order : bool
        Use 18th‑order Taylor expansion (True) or 12th‑order (False).
    """

    def __init__(self, freq, order, L, *,
                 dtype=torch.complex128,
                 device=None,
                 high_order=True):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype not in (torch.complex64, torch.complex128):
            warnings.warn(f'dtype {dtype} not recognized, using complex128.', UserWarning)
            dtype = torch.complex128

        self._dtype = dtype
        self._device = device
        self._high_order = high_order

        # physical parameters
        self.freq = torch.as_tensor(freq, dtype=self._dtype, device=self._device)
        self.omega = 2.0 * torch.pi * self.freq
        self.L = torch.as_tensor(L, dtype=torch.float64, device=self._device)

        # Fourier orders
        self.order = order
        ox = 2 * order[0] + 1
        oy = 2 * order[1] + 1
        self.order_x = torch.linspace(-order[0], order[0], ox, dtype=torch.int64, device=self._device)
        self.order_y = torch.linspace(-order[1], order[1], oy, dtype=torch.int64, device=self._device)
        self.order_N = ox * oy

        # reciprocal‑lattice step
        self.Gx_norm = 1.0 / (L[0] * self.freq.real)
        self.Gy_norm = 1.0 / (L[1] * self.freq.real)

        # surrounding media  (default: free space)
        self.eps_in = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.eps_out = torch.tensor(1.0, dtype=self._dtype, device=self._device)

        # internal layers
        self.layer_N = 0
        self.thickness = []
        self.eps_conv = []
        self.layer_S11 = []
        self.layer_S12 = []
        self.layer_S21 = []
        self.layer_S22 = []

        # Taylor coefficients
        self._coeff_high = _make_coeff_table(self._dtype, self._device)
        self._coeff_low = _make_coeff_table_12(self._dtype, self._device)

        # k‑vectors and V₀ — filled by set_incident_angle / _kvectors
        self._kvectors_ready = False

    # ── Public API ────────────────────────────────────────────────────────

    def add_input_layer(self, eps=1.0):
        """Set the relative permittivity of the input half‑space."""
        self.eps_in = torch.as_tensor(eps, dtype=self._dtype, device=self._device)

    def add_output_layer(self, eps=1.0):
        """Set the relative permittivity of the output half‑space."""
        self.eps_out = torch.as_tensor(eps, dtype=self._dtype, device=self._device)

    def set_incident_angle(self, inc_ang, azi_ang, angle_layer='input'):
        """Set the incident direction and build k‑vectors.

        Parameters
        ----------
        inc_ang : float
            Incident angle (radians).
        azi_ang : float
            Azimuthal angle (radians).
        angle_layer : str
            Reference layer: 'input' or 'output'.
        """
        self.inc_ang = torch.as_tensor(inc_ang, dtype=self._dtype, device=self._device)
        self.azi_ang = torch.as_tensor(azi_ang, dtype=self._dtype, device=self._device)
        if angle_layer in ('i', 'in', 'input'):
            self.angle_layer = 'input'
        elif angle_layer in ('o', 'out', 'output'):
            self.angle_layer = 'output'
        else:
            self.angle_layer = 'input'
        self._kvectors()

    def add_layer(self, thickness, eps=1.0):
        """Add an internal layer.

        Parameters
        ----------
        thickness : float
            Layer thickness (length units).
        eps : float | complex | Tensor
            Relative permittivity.  Scalar → homogeneous; tensor → patterned.
        """
        is_homogeneous = (
            isinstance(eps, (float, complex)) or
            (isinstance(eps, torch.Tensor) and
             (eps.dim() == 0 or (eps.dim() == 1 and eps.shape[0] == 1)))
        )
        if is_homogeneous:
            eps_t = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
            E = eps_t * torch.eye(self.order_N, dtype=self._dtype, device=self._device)
        else:
            E = self._material_conv(eps)

        self.eps_conv.append(E)
        self.layer_N += 1
        self.thickness.append(thickness)

        if is_homogeneous:
            self._compute_layer_homogeneous(thickness, eps)
        else:
            self._compute_layer_free(thickness)

    def solve_global_smatrix(self):
        """Solve the global S‑matrix by Redheffer star‑product of all layers."""
        n2 = 2 * self.order_N
        if self.layer_N > 0:
            S11, S21, S12, S22 = (
                self.layer_S11[0], self.layer_S21[0],
                self.layer_S12[0], self.layer_S22[0])
        else:
            S11 = torch.eye(n2, dtype=self._dtype, device=self._device)
            S22 = torch.eye(n2, dtype=self._dtype, device=self._device)
            S12 = torch.zeros(n2, n2, dtype=self._dtype, device=self._device)
            S21 = torch.zeros(n2, n2, dtype=self._dtype, device=self._device)

        for i in range(self.layer_N - 1):
            S11, S21, S12, S22 = self._rs_prod(
                [S11, S21, S12, S22],
                [self.layer_S11[i + 1], self.layer_S21[i + 1],
                 self.layer_S12[i + 1], self.layer_S22[i + 1]])

        if hasattr(self, 'Sin'):
            S11, S21, S12, S22 = self._rs_prod(
                self.Sin, [S11, S21, S12, S22])

        if hasattr(self, 'Sout'):
            S11, S21, S12, S22 = self._rs_prod(
                [S11, S21, S12, S22], self.Sout)

        self.S = [S11, S21, S12, S22]

    def source_planewave(self, *, amplitude=(1.0, 0.0), direction='forward'):
        """Set a plane‑wave source at the (0,0) diffraction order.

        Parameters
        ----------
        amplitude : (float, float)
            [Ex_amp, Ey_amp].
        direction : str
            'forward' or 'backward'.
        """
        self.source_fourier(amplitude=amplitude, orders=[0, 0], direction=direction)

    def source_fourier(self, *, amplitude, orders, direction='forward'):
        """Set a Fourier source at specific diffraction orders.

        Parameters
        ----------
        amplitude : (N, 2) array_like
            [[Ex_amp, Ey_amp], ...] at each order.
        orders : (N, 2) array_like
            [[mx, my], ...] diffraction orders.
        direction : str
            'forward' or 'backward'.
        """
        amplitude = torch.as_tensor(amplitude, dtype=self._dtype, device=self._device)
        amplitude = amplitude.reshape(-1, 2)
        orders = torch.as_tensor(orders, dtype=torch.int64, device=self._device)
        orders = orders.reshape(-1, 2)

        if direction not in ('f', 'forward', 'b', 'backward'):
            direction = 'forward'
        self.source_direction = 'forward' if direction in ('f', 'forward') else 'backward'

        order_indices = self._matching_indices(orders)
        E_i = torch.zeros(2 * self.order_N, 1, dtype=self._dtype, device=self._device)
        E_i[order_indices, 0] = amplitude[:, 0]
        E_i[order_indices + self.order_N, 0] = amplitude[:, 1]
        self.E_i = E_i

    def field_xy(self, x_axis, y_axis):
        """Compute Ex, Ey, Ez on an xy plane.

        Parameters
        ----------
        x_axis : Tensor  1D coordinates along x.
        y_axis : Tensor  1D coordinates along y.

        Returns
        -------
        [Ex, Ey, Ez] : list[Tensor]  each of shape (len(x_axis), len(y_axis)).
        """
        if not isinstance(x_axis, torch.Tensor) or not isinstance(y_axis, torch.Tensor):
            warnings.warn('x_axis and y_axis must be torch.Tensor.')
            return None

        x_axis = x_axis.reshape(-1, 1, 1)
        y_axis = y_axis.reshape(1, -1, 1)

        eps = getattr(self, 'eps_in', None)
        eps = eps if eps is not None else torch.tensor(1.0, dtype=self._dtype, device=self._device)
        if self.angle_layer != 'input':
            eps = getattr(self, 'eps_out', eps)

        if self.angle_layer == 'input':
            port = 1 if self.source_direction == 'forward' else 3
        else:
            port = 0 if self.source_direction == 'forward' else 2

        kz = torch.sqrt(eps - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        if self.angle_layer == 'input':
            kz = torch.where(kz.imag < 0, kz.conj(), kz).reshape(-1, 1)
        else:
            kz = -torch.where(kz.imag < 0, kz.conj(), kz).reshape(-1, 1)

        Exy = self.S[port] @ self.E_i
        Ex_mn = Exy[:self.order_N]
        Ey_mn = Exy[self.order_N:]

        inv_kz = torch.where(kz.abs() > 1e-15, 1.0 / kz, torch.zeros_like(kz))
        Ez_mn = (self.Kx_norm_dn.reshape(-1, 1) * Ex_mn +
                 self.Ky_norm_dn.reshape(-1, 1) * Ey_mn) * inv_kz

        phase = torch.exp(1.j * self.omega * (
            self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y_axis))
        Ex = torch.sum(Ex_mn.reshape(1, 1, -1) * phase, dim=2)
        Ey = torch.sum(Ey_mn.reshape(1, 1, -1) * phase, dim=2)
        Ez = torch.sum(Ez_mn.reshape(1, 1, -1) * phase, dim=2)

        return Ex, Ey, Ez

    # ── Internal helpers ──────────────────────────────────────────────────

    def _matching_indices(self, orders):
        """Map (mx, my) order pairs to flattened indices in [0, order_N)."""
        orders = orders.clone()
        orders[:, 0] = orders[:, 0].clamp(-self.order[0], self.order[0])
        orders[:, 1] = orders[:, 1].clamp(-self.order[1], self.order[1])
        oy_len = len(self.order_y)
        return oy_len * (orders[:, 0] + self.order[0]) + (orders[:, 1] + self.order[1])

    def _kvectors(self):
        """Build Kx, Ky diagonal matrices and V₀ from the incident angles."""
        n_ref = torch.real(torch.sqrt(
            self.eps_in if self.angle_layer == 'input' else self.eps_out))
        self.kx0_norm = n_ref * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
        self.ky0_norm = n_ref * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)

        kx = self.kx0_norm + self.order_x * self.Gx_norm
        ky = self.ky0_norm + self.order_y * self.Gy_norm

        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='ij')
        self.Kx_norm_dn = kx_grid.reshape(-1)
        self.Ky_norm_dn = ky_grid.reshape(-1)
        self.Kx_norm = torch.diag(self.Kx_norm_dn)
        self.Ky_norm = torch.diag(self.Ky_norm_dn)

        # build V₀  (free‑space V matrix, eps = 1)
        self._v0_11, self._v0_12, self._v0_21, self._v0_22 = _build_v0(
            self.Kx_norm_dn, self.Ky_norm_dn,
            torch.tensor(1.0, dtype=self._dtype, device=self._device))

        if hasattr(self, 'eps_in'):
            self._build_port_smatrix('input')
        if hasattr(self, 'eps_out'):
            self._build_port_smatrix('output')

        self._kvectors_ready = True

    def _build_port_smatrix(self, side):
        """Build the port S‑matrix for the input or output half‑space."""
        eps = self.eps_in if side == 'input' else self.eps_out
        vp = _build_v0(self.Kx_norm_dn, self.Ky_norm_dn, eps)
        is_input = (side == 'input')

        n = self.order_N
        n2 = 2 * n

        vp11, vp12, vp21, vp22 = vp
        v011, v012, v021, v022 = self._v0_11, self._v0_12, self._v0_21, self._v0_22

        # Vsum = Vp + V0,  Vdiff = Vp − V0  (per diagonal 2×2 block)
        vs11 = vp11 + v011; vs12 = vp12 + v012
        vs21 = vp21 + v021; vs22 = vp22 + v022
        vd11 = vp11 - v011; vd12 = vp12 - v012
        vd21 = vp21 - v021; vd22 = vp22 - v022

        # invert each 2×2 block of Vsum
        det = vs11 * vs22 - vs12 * vs21
        inv_det = 1.0 / det
        inv11 = vs22 * inv_det;  inv12 = -vs12 * inv_det
        inv21 = -vs21 * inv_det; inv22 = vs11 * inv_det

        # coupling = Vsum⁻¹ @ Vdiff
        c11 = inv11 * vd11 + inv12 * vd21
        c12 = inv11 * vd12 + inv12 * vd22
        c21 = inv21 * vd11 + inv22 * vd21
        c22 = inv21 * vd12 + inv22 * vd22

        if is_input:
            s11_11 = 2.0 * (inv11 * vp11 + inv12 * vp21)
            s11_12 = 2.0 * (inv11 * vp12 + inv12 * vp22)
            s11_21 = 2.0 * (inv21 * vp11 + inv22 * vp21)
            s11_22 = 2.0 * (inv21 * vp12 + inv22 * vp22)
            s12_11 = -c11; s12_12 = -c12
            s12_21 = -c21; s12_22 = -c22
            s21_11 = c11;  s21_12 = c12
            s21_21 = c21;  s21_22 = c22
            s22_11 = 2.0 * (inv11 * v011 + inv12 * v021)
            s22_12 = 2.0 * (inv11 * v012 + inv12 * v022)
            s22_21 = 2.0 * (inv21 * v011 + inv22 * v021)
            s22_22 = 2.0 * (inv21 * v012 + inv22 * v022)
        else:
            s11_11 = 2.0 * (inv11 * v011 + inv12 * v021)
            s11_12 = 2.0 * (inv11 * v012 + inv12 * v022)
            s11_21 = 2.0 * (inv21 * v011 + inv22 * v021)
            s11_22 = 2.0 * (inv21 * v012 + inv22 * v022)
            s12_11 = c11;  s12_12 = c12
            s12_21 = c21;  s12_22 = c22
            s21_11 = -c11; s21_12 = -c12
            s21_21 = -c21; s21_22 = -c22
            s22_11 = 2.0 * (inv11 * vp11 + inv12 * vp21)
            s22_12 = 2.0 * (inv11 * vp12 + inv12 * vp22)
            s22_21 = 2.0 * (inv21 * vp11 + inv22 * vp21)
            s22_22 = 2.0 * (inv21 * vp12 + inv22 * vp22)

        S11 = torch.zeros(n2, n2, dtype=self._dtype, device=self._device)
        S12 = torch.zeros(n2, n2, dtype=self._dtype, device=self._device)
        S21 = torch.zeros(n2, n2, dtype=self._dtype, device=self._device)
        S22 = torch.zeros(n2, n2, dtype=self._dtype, device=self._device)

        idx = torch.arange(n, device=self._device)
        S11[idx, idx] = s11_11;       S11[idx, idx + n] = s11_12
        S11[idx + n, idx] = s11_21;   S11[idx + n, idx + n] = s11_22
        S12[idx, idx] = s12_11;       S12[idx, idx + n] = s12_12
        S12[idx + n, idx] = s12_21;   S12[idx + n, idx + n] = s12_22
        S21[idx, idx] = s21_11;       S21[idx, idx + n] = s21_12
        S21[idx + n, idx] = s21_21;   S21[idx + n, idx + n] = s21_22
        S22[idx, idx] = s22_11;       S22[idx, idx + n] = s22_12
        S22[idx + n, idx] = s22_21;   S22[idx + n, idx + n] = s22_22

        if side == 'input':
            self.Sin = [S11, S21, S12, S22]
        else:
            self.Sout = [S11, S21, S12, S22]

    def _material_conv(self, material):
        """Build the convolution matrix of eps via FFT."""
        material = material.to(self._dtype)
        Nx, Ny = material.shape
        material_N = Nx * Ny

        order_x_grid, order_y_grid = torch.meshgrid(
            self.order_x, self.order_y, indexing='ij')
        ox = order_x_grid.reshape(-1).to(torch.int64)
        oy = order_y_grid.reshape(-1).to(torch.int64)

        ind = torch.arange(self.order_N, device=self._device)
        indx, indy = torch.meshgrid(ind.to(torch.int64), ind.to(torch.int64), indexing='ij')

        material_fft = torch.fft.fft2(material) / material_N
        fx = ox[indx] - ox[indy]
        fy = oy[indx] - oy[indy]
        fx = fx % Nx
        fy = fy % Ny
        return material_fft[fx, fy]

    def _rs_prod(self, Sm, Sn):
        """Redheffer star product: S = Sm ⊗ Sn."""
        n2 = 2 * self.order_N
        I = torch.eye(n2, dtype=self._dtype, device=self._device)
        O = torch.zeros(n2, n2, dtype=self._dtype, device=self._device)

        T1 = I - Sm[2] @ Sn[1]
        T2 = I - Sn[1] @ Sm[2]
        try:
            S11 = Sn[0] @ torch.linalg.solve(T1, Sm[0])
            S21 = Sm[1] + Sm[3] @ torch.linalg.solve(T2, Sn[1] @ Sm[0])
            S12 = Sn[2] + Sn[0] @ torch.linalg.solve(T1, Sm[2] @ Sn[3])
            S22 = Sm[3] @ torch.linalg.solve(T2, Sn[3])
        except RuntimeError:
            S11 = Sn[0] @ Sm[0]
            S21 = Sm[1]
            S12 = Sn[2]
            S22 = O

        return S11, S21, S12, S22

    # ── Layer computation ─────────────────────────────────────────────────

    def _compute_layer_homogeneous(self, thickness, eps):
        """Analytical S‑matrix for a homogeneous layer.

        Computes per‑order 2×2 blocks directly from kx, ky, kz.
        """
        eps_t = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        kz = torch.sqrt(eps_t - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        kz = torch.where(kz.imag < 0, kz.conj(), kz)
        one_c = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        kz0 = torch.sqrt(one_c - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        kz0 = torch.where(kz0.imag < 0, kz0.conj(), kz0)

        n = self.order_N
        n2 = 2 * n
        k0d = self.omega * thickness

        S11 = torch.zeros(n2, n2, dtype=self._dtype, device=self._device)
        S12 = torch.zeros(n2, n2, dtype=self._dtype, device=self._device)
        I2 = torch.eye(2, dtype=self._dtype, device=self._device)

        for i in range(n):
            kx_i = self.Kx_norm_dn[i]
            ky_i = self.Ky_norm_dn[i]
            kz_i = kz[i]
            kz0_i = kz0[i]

            if kz_i.abs() < 1e-12 or kz0_i.abs() < 1e-12:
                continue

            tau = kz_i / kz0_i - kz0_i / kz_i
            a = tau * kx_i**2 - kz_i / kz0_i
            b = tau * kx_i * ky_i
            d = tau * ky_i**2 - kz_i / kz0_i

            U = torch.tensor([[a.detach(), b.detach()], [b.detach(), d.detach()]],
                             dtype=self._dtype, device=self._device)
            u = torch.exp(1.j * kz_i * k0d)

            A_blk = I2 - U
            B_blk = (I2 + U) * u
            G = B_blk @ torch.linalg.inv(A_blk)
            C_blk = 2.0 * torch.linalg.inv(A_blk - G @ B_blk)
            D_blk = -C_blk @ G

            S11_blk = C_blk * u + D_blk
            S12_blk = D_blk * u + C_blk - I2

            # place into sparse 2n × 2n matrix
            for bi, ri in [(0, i), (1, i + n)]:
                for bj, ci in [(0, i), (1, i + n)]:
                    S11[ri, ci] = S11_blk[bi, bj]
                    S12[ri, ci] = S12_blk[bi, bj]

        self.layer_S11.append(S11)
        self.layer_S12.append(S12)
        self.layer_S21.append(S12)
        self.layer_S22.append(S11)

    def _compute_layer_free(self, thickness):
        """Compute the layer S‑matrix via the FREE path."""
        # thickness splitting
        thickness_thres = 2.0 * torch.min(
            self.L[0] / self.order[0], self.L[1] / self.order[1])
        if thickness > thickness_thres:
            n_repeated = int(torch.ceil(
                torch.log2(torch.as_tensor(thickness / thickness_thres)).detach()))
            d_block = thickness / (2 ** n_repeated)
        else:
            n_repeated = 0
            d_block = thickness

        k0d = self.omega * d_block
        n2 = 2 * self.order_N

        E_conv = self.eps_conv[-1]
        P, Q = build_pq(E_conv, self.Kx_norm, self.Ky_norm)

        S11, S12 = compute_s_matrix_free(
            P, Q, k0d,
            self._v0_11, self._v0_12, self._v0_21, self._v0_22,
            self._coeff_high, self._coeff_low, self._high_order)

        if n_repeated > 0:
            S11, S12 = self._repeat_s_matrix(S11, S12, n_repeated, n2)

        self.layer_S11.append(S11)
        self.layer_S12.append(S12)
        self.layer_S21.append(S12)
        self.layer_S22.append(S11)

    def _repeat_s_matrix(self, S11, S12, repeats, n2):
        """Power an S‑matrix by repeated star products."""
        I = torch.eye(n2, dtype=self._dtype, device=self._device)
        S21 = S12
        S22 = S11

        for _ in range(repeats):
            R = I - S12 @ S12
            rhs_left = S12 @ S11
            rhs = torch.cat((rhs_left, S11), dim=1)
            XY = torch.linalg.solve(R, rhs)

            S11_XY = S11 @ XY
            S21_new = S11_XY[:, :n2]
            S21_new = S21_new + S12
            S22_new = S11_XY[:, n2:]

            # swap: S₁₁↔S₂₂_new, S₁₂↔S₂₁_new
            S11, S22 = S22_new, S11
            S12, S21 = S21_new, S12

        return S11, S12
