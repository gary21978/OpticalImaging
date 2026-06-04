# 手动反向传播公式推导

## 概览

优化参数：标量 `θ = thickness`。前向管线：

```
θ → k₀d → Cos,Sinc → E,F → S_layer → S_global → E_xy → fields → I → loss
```

链式法则：

```
d loss
─────  =  Σ  ───── · ────
  d θ      ij ∂ I_ij    d θ
```

下面从 loss 开始逆序推导每一步。**粗体** 标记的是需要在 CUDA kernel 中实现的公式。

---

## 0. 符号约定

- `A*` 或 `conj(A)` ：共轭，逐元素
- `A^H` ：共轭转置
- `A^{-T}` ：`(A^{-1})^T`
- `∂L/∂A` ：标量 L 对矩阵 A 的梯度，**与 A 同形状**
- `⊙` ：逐元素乘
- `Re(z)` ：取实部
- `tr` ：迹

关键恒等式（VJP = vector-Jacobian product）：

```
d L = tr( (∂L/∂A)^T · dA )     ← 标量对矩阵的微分

若 A = B @ C:     ∂L/∂B = (∂L/∂A) @ C^T
                  ∂L/∂C = B^T @ (∂L/∂A)

若 A = B + C:     ∂L/∂B = ∂L/∂A
                  ∂L/∂C = ∂L/∂A

若 A = B^{-1}:    dA = -A · dB · A
                  ∂L/∂B = -A^{-T} · (∂L/∂A) · A^{-T}
                      = -A^{-H} · (∂L/∂A) · A^{-H}

若 X = solve(A, B):   A·X = B,  X = A^{-1}·B
    dX = A^{-1}·(dB - dA·X)
    ∂L/∂B = A^{-H} · (∂L/∂X)
    ∂L/∂A = -A^{-H} · (∂L/∂X) · X^H
```

---

## 1. Loss → Intensity

```
I = |Ex|² + |Ey|² + |Ez|²          (逐像素)

L = MSE(I, I_target) = (1/N) Σ (I_ij - I_target_ij)²
```

**梯度：**
```
∂L                2
────  =  ─── · (I - I_target)       (逐像素标量)
∂ I               N
```

**I 对 Ex/Ey/Ez 的梯度：**
```
∂L           ∂L
────  =  2 · ─── · Re(Ex)           (逐像素)
∂ Ex          ∂I

∂L/∂Ey, ∂L/∂Ez 类似
```

等价写法（复数域）：
```
∂L/∂Ex = (∂L/∂I) · conj(Ex)        ← Wirtinger 导数, ∂|z|²/∂z = conj(z)
```

---

## 2. Fields → Fourier 振幅

```
Ex(x,y) = Σ_mn Ex_mn · exp( i·ω·(Kx_m·x + Ky_n·y) )
```

这是**线性变换**：`Ex = Φ @ Ex_mn`，其中 `Φ_{ij,mn} = exp(i·ω·(Kx_m·x_i + Ky_n·y_j))`

**梯度（反向就是伴随算子）：**
```
∂L                               ∂L
─────  =  Σ   ───── · exp( -i·ω·(Kx_m·x_i + Ky_n·y_j) )
∂Ex_mn    ij  ∂Ex_ij
```

即 `∂L/∂Ex_mn = Φ^H @ (∂L/∂Ex)`，相当于用负相位做反向傅里叶求和。

**Ez 的特殊处理：**
```
Ez_mn = (Kx_m·Ex_mn + Ky_n·Ey_mn) / Kz_mn
                                     ─────
                                     对角
```

这是逐元素运算，梯度：
```
∂L              Kx_m      ∂L
─────  =  ──────── · ────────
∂Ex_mn        conj(Kz_mn)  ∂Ez_mn

∂L              Ky_n      ∂L
─────  =  ──────── · ────────
∂Ey_mn        conj(Kz_mn)  ∂Ez_mn
```

---

## 3. Fourier 振幅 → 全局 S 矩阵

```
Exy = S_port @ E_i         (S_port 是 S11/S21 中的某一个)
```

`E_i` 是固定源向量，不求导。

```
∂L              ∂L
─────  =  ────────── · E_i^T          ← 外积
∂S_port        ∂ Exy

对于 S 的其他块 (非 port 的 S12/S22)，∂L/∂S = 0
```

---

## 4. 全局 S 矩阵 → 各层 S 矩阵

全局 S 由各层 S 通过 Redheffer 星乘串接。以两层为例：

```
S_global = RS_prod( RS_prod(S_in, S_layer), S_out )
```

其中 `S_in`, `S_out` 是端口 S 矩阵（与厚度无关，不求导）。
记 `S = RS_prod(Sm, Sn)`：

```
S11 = Sn11 @ T1^{-1} @ Sm11
S21 = Sm21 + Sm22 @ T2^{-1} @ Sn21 @ Sm11
S12 = Sn12 + Sn11 @ T1^{-1} @ Sm12 @ Sn22
S22 = Sm22 @ T2^{-1} @ Sn22

其中  T1 = I - Sm12 @ Sn21
     T2 = I - Sn21 @ Sm12
```

**梯度通过 RS_prod 反向传播**（链式法则，对 Sm 和 Sn 分别求导）：

对 `Sm11`（出现在 S11 中）：
```
∂L               T     ∂L
─────  =  Sn11^T @ ───── @ T1^{-T}
∂Sm11                  ∂S11
```

对 `Sm22`（出现在 S22 中）：
```
∂L          ∂L
─────  =  ───── @ Sn22^T @ T2^{-T}
∂Sm22       ∂S22
```

对 `Sm12`（出现在 T1 和 S12 中，通过 `Sm12 @ Sn21` 项）：
```
需要同时考虑 ∂S11/∂Sm12、∂S12/∂Sm12、∂S21/∂Sm12、∂S22/∂Sm12
每项展开后合并。核心模式：
- ∂(T1^{-1})/∂Sm12 = T1^{-1} @ Sn21^T @ T1^{-T}
- 然后链到各 S 块
```

这是整个链路中**最繁琐的一步**，但每层的推导模式相同。

---

## 5. 层 S 矩阵 → E, F → Cos, Sinc

### 5a. EF → S (ef_to_s_matrix)

```
T22 = (E + F) / 2
T21 = (E - F) / 2
S11 = T22^{-1}
S12 = -S11 @ T21
```

**梯度：**

```
∂L               1     ∂L
───  =  ─── · ─────── · T21^T @ S11^T    (通过对 T22^{-1} 的梯度)
∂T22             2    ∂S11       ──
                                  从 S12 也有贡献

更直接的推导（用 VJP 恒等式）：

∂L/∂S11 = 已知（来自 RS_prod 反向传播）
∂L/∂S12 = 已知

∂L/∂T22 = -T22^{-H} · (∂L/∂S11 + S12^T · ∂L/∂S12) · T22^{-H}
∂L/∂T21 = -S11^T · ∂L/∂S12

∂L/∂E = (∂L/∂T22 + ∂L/∂T21) / 2
∂L/∂F = (∂L/∂T22 - ∂L/∂T21) / 2
```

### 5b. FREE path: E, F → Cos, Sinc, P, Q

```
E = Cos + i·k₀d · V₀Q @ Sinc
F = -( part1 + V₀Q_CosQ ) @ V₀

其中:
    V₀Q = V₀ @ Q
    part1 = i·k₀d · Sinc @ P
    V₀Q_CosQ = V₀Q @ CosQ
    CosQ = Cos @ Q^{-1}
```

**梯度（关键几步）：**

```
∂L/∂Cos   = ∂L/∂E    (E 中的直接项)
           + ∂L/∂F 通过 F 的贡献

∂L/∂Sinc  = k₀d · (V₀Q)^T · (-i·∂L/∂E)    (E 贡献)
           + k₀d · P^T · (-i·∂L/∂F_pre)    (F 贡献，∂F_pre 再链到 F)

∂L/∂P     = -i·k₀d · (∂L/∂F_pre) · Sinc^T

∂L/∂CosQ  = V₀Q^T · ∂L/∂F_pre

其中 F_pre = part1 + V₀Q_CosQ,  F = -F_pre @ V₀
∂L/∂F_pre = -V₀ · (∂L/∂F)^T  (通过 V₀ 右乘反向传播)
```

### 5c. CosQ = Cos @ Q^{-1}

```
CosQ = solve(Q^T, Cos^T)^T
```

等价于 `Q @ CosQ^T = Cos^T`，即逐列求解。梯度：

```
∂L/∂Cos += ∂L/∂CosQ @ Q^{-H}           (对 Cos 的额外贡献)

∂L/∂Q   += -Q^{-H} @ Cos^T @ (∂L/∂CosQ)^T @ Q^{-H}
```

### 5d. V₀Q = V₀ @ Q

```
∂L/∂Q += V₀^T @ ∂L/∂V₀Q                (从 E 和 F 累积)
```

---

## 6. Cos, Sinc → P, Q → k₀d

### 6a. Cos, Sinc 对 A² 的梯度

Cos 和 Sinc 由 `compute_cos_sinc_from_pq_order18` 计算。核心是：

```
A² = 2^{-2m} · k₀d² · P·Q          ← P, Q 形成 PQ
```

然后经过多轮矩阵乘法和加法构建 Cos 和 Sinc。
每一步都是 `A @ B` 或 `A + B` 形式，梯度直接用矩阵乘法的 VJP 反向传播：

```
若 C = A @ B:   ∂L/∂A = (∂L/∂C) @ B^T
                ∂L/∂B = A^T @ (∂L/∂C)

若 C = A + B:   ∂L/∂A = ∂L/∂C
                ∂L/∂B = ∂L/∂C

若 C = α·A:     ∂L/∂A = α · ∂L/∂C
```

### 6b. PQ = P @ Q

```
∂L/∂P += (∂L/∂PQ) @ Q^T               (从 A² 反向传播)
∂L/∂Q += P^T @ (∂L/∂PQ)
```

### 6c. P, Q 对 eps_conv 的梯度

```
P = [KxEiKy,    KyEiKy-I]
    [I-KxEiKx,  -KyEiKx ]

Q = [-KxKy,   eps_conv - Ky²]
    [Kx²-eps_conv,  KxKy    ]
```

其中 `KxEiKy = Kx @ E^{-1} @ Ky`，`E = eps_conv`。

关键梯度：`∂L/∂(E^{-1} @ Ky)` 涉及 `solve` 的梯度：

```
EiKy = solve(E, Ky)           ← E @ EiKy = Ky

∂L/∂E += -E^{-H} @ Kx^T @ (∂L/∂KxEiKy) @ EiKy^T @ E^{-H}
          ... (从 KxEiKy, KxEiKx 等各项累加到 ∂L/∂E)
```

### 6d. k₀d 对 thickness 的梯度

```
k₀d = ω · d_block
d_block = thickness / 2^{n_repeated}
```

```
∂L             d (k₀d)     ∂L
────────  =  ────────── · ────
∂thickness     d θ        ∂k₀d
```

其中 `d(k₀d)/dθ = ω / 2^{n_repeated}`（当无需厚度分割时，n_repeated=0）。

`∂L/∂k₀d` 累积自：
1. Cos,Sinc 计算中的 `A² = 2^{-2m}·k₀d²·PQ` → `∂L/∂A² · 2·k₀d·2^{-2m}·PQ`
2. E,F 构造中的 `i·k₀d·V₀Q·Sinc` 和 `i·k₀d·Sinc·P`

---

## 7. 梯度汇总：d loss / d thickness

```
∂L            ∂L       d(k₀d)
─────  =   ─────── · ────────
∂θ           ∂k₀d       dθ

∂L
─────  =  tr( (∂L/∂E)^T · ∂E/∂k₀d + (∂L/∂F)^T · ∂F/∂k₀d )
∂k₀d
        + tr( (∂L/∂Cos)^T · ∂Cos/∂k₀d + (∂L/∂Sinc)^T · ∂Sinc/∂k₀d )

其中:
∂E/∂k₀d = i · V₀Q @ Sinc
∂F/∂k₀d = -i · (Sinc @ P + V₀Q @ CosQ_k0d_part) @ V₀    ← 需展开

∂Cos/∂k₀d  和  ∂Sinc/∂k₀d  需通过 CosSinc 内部的双角公式
和 A² 对 k₀d 的依赖来求。
```

---

## 8. 需要的 CUDA kernel 列表

| # | Kernel | 功能 | 复杂度 |
|---|--------|------|--------|
| 1 | `MSEGradKernel` | ∂L/∂I | O(N) |
| 2 | `FieldToFourierGradKernel` | ∂L/∂Ex_mn | O(N·n_orders) |
| 3 | `V0LeftMulGradKernel` | V₀左乘的 VJP | O(n²) |
| 4 | `V0RightMulGradKernel` | V₀右乘的 VJP | O(n²) |
| 5 | `MatInvGradKernel` | A^{-1} 的 VJP：-A^{-H}·G·A^{-H} | O(n³) matmul |
| 6 | `SolveGradKernel` | solve(A,B) 的 VJP | O(n³) matmul |
| 7 | `RSProdGradKernel` | Redheffer 星乘的 VJP | O(n³) matmul × 6 |
| 8 | `EFToSGradKernel` | EF→S 的 VJP | O(n³) matmul |
| 9 | `CosSincGradKernel` | CosSinc 各 Phase 的 VJP | O(n³) matmul × 多轮 |
| 10 | `BuildPQGradKernel` | P,Q 构造的 VJP | O(n³) matmul |

其中 kernel 5-10 都可以用 cuBLAS `gemm` 实现，不需要手写矩阵乘法。

**核心工作量不是写 CUDA kernel，而是正确推导和实现每个 kernel 的输入/输出/数学公式。**

---

## 9. 估算

| 项目 | 工作量 |
|------|--------|
| 链式法则公式推导（已完成） | 已做 |
| 每步 VJP 的 C++/cuBLAS 实现 | ~200 行/步 × 10 步 = ~2000 行 |
| 调试：前向 + 反向的数值差分验证 | 每步独立验证 |
| 总新增代码量 | ~2000-3000 行 |

与有限差分的 **~150 行** 相比，手动反向传播的工作量大 15-20 倍。
