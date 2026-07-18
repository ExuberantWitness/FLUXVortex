"""rVPM 粒子输运(S1,PROJECT_rvpm / research_rvpm_arch.md V4/V6/V8)。

离散输运(Alvarez & Ning,f=0, g=1/5 守恒定值;rvpm-02):
    S   = (∇ū)ᵀ·Γ                     (transposed stretching,默认 scheme)
    Z   = [(f+g)/(1+3f)]·(S·Γ)/|Γ|²   = (1/5)·(S·Γ)/|Γ|²
    dΓ/dt = S − 3·Z·Γ                  (Γ∥ 拉伸的 3/5 移入核收缩)
    dσ/dt = −σ·Z
经典 VPM 消融开关 f=g=0(rvpm-01):Z=0、dσ/dt=0 —— G2 门要求它复现爆断。
散度控制:corrected-Pedrizzetti(保 |Γ|,rlxf=0.3/步;rvpm-05):
    Γ ← |Γ|·normalize((1−r)·Γ̂ + r·ω̂),ω 取粒子处局部涡量(J 反对称部,免费)。
时间积分:Williamson LSRK3(A5-C3)。SFS 本级关(S2 起开)。
|Γ|→0 守卫用光滑分母(autodiff 坑清单;不用条件门)。
诱导/Jacobian 复用已验证 warp_vpm 核(Gaussian-erf,与 FLOWVPM 同族;rvpm-07)。
零拟合:全部常数文献值。"""
import numpy as np

from fluxvortex.warp_vpm import (velocity_from_particles_gpu,
                                 jacobian_from_particles_gpu)

CFG_RVPM = dict(f=0.0, g=0.2)          # 守恒推导定值(rvpm-02)
CFG_CVPM = dict(f=0.0, g=0.0)          # 经典消融(rvpm-01)
_LSRK3_A = (0.0, -5.0 / 9.0, -153.0 / 128.0)     # Williamson 1980
_LSRK3_B = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)
EPS_G2 = 1e-12                          # |Γ|² 光滑守卫(m^4/s^2 量级远小于物理值)


def rhs(X, G, sig, f=0.0, g=0.2, classic_sigma_frozen=None):
    """返回 (dX, dG, dsig)。classic 模式(g=0)下 dsig=0。"""
    u = velocity_from_particles_gpu(X, X, G, sig)                  # (n,3)
    J = jacobian_from_particles_gpu(X, G, X, G, sig)               # (n,3,3) ∂u_i/∂x_j
    S = np.einsum("nji,nj->ni", J, G)                              # transposed: (∇u)ᵀ·Γ
    g2 = np.einsum("ni,ni->n", G, G) + EPS_G2
    cz = (f + g) / (1.0 + 3.0 * f)
    Z = cz * np.einsum("ni,ni->n", S, G) / g2                      # (n,)
    dG = S - 3.0 * Z[:, None] * G
    dsig = -sig * Z
    return u, dG, dsig


def pedrizzetti_corrected(X, G, sig, rlxf=0.3):
    """corrected-Pedrizzetti:Γ 向局部涡量方向松弛,|Γ| 保持(rvpm-05)。"""
    J = jacobian_from_particles_gpu(X, G, X, G, sig)
    om = np.stack([J[:, 2, 1] - J[:, 1, 2],
                   J[:, 0, 2] - J[:, 2, 0],
                   J[:, 1, 0] - J[:, 0, 1]], axis=1)               # ∇×u
    gn = np.linalg.norm(G, axis=1, keepdims=True)
    gh = G / np.maximum(gn, 1e-300)
    oh = om / np.maximum(np.linalg.norm(om, axis=1, keepdims=True), 1e-300)
    mix = (1.0 - rlxf) * gh + rlxf * oh
    mix /= np.maximum(np.linalg.norm(mix, axis=1, keepdims=True), 1e-300)
    return gn * mix


def step_lsrk3(X, G, sig, dt, f=0.0, g=0.2, relax=0.3):
    """一步 Williamson LSRK3;步末 corrected-Pedrizzetti(relax<=0 关闭)。"""
    qX = np.zeros_like(X); qG = np.zeros_like(G); qs = np.zeros_like(sig)
    for a, b in zip(_LSRK3_A, _LSRK3_B):
        u, dG, ds = rhs(X, G, sig, f=f, g=g)
        qX = a * qX + dt * u
        qG = a * qG + dt * dG
        qs = a * qs + dt * ds
        X = X + b * qX
        G = G + b * qG
        sig = np.maximum(sig + b * qs, 1e-9)
    if relax > 0.0:
        G = pedrizzetti_corrected(X, G, sig, rlxf=relax)
    return X, G, sig


def make_ring(R=1.0, Gamma=1.0, a=0.1, n=100, z=0.0, overlap=1.3):
    """薄涡环离散为 n 粒子:环向矩 α=Γ·dl,核 σ=max(a, overlap×间距)。"""
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    X = np.stack([R * np.cos(th), R * np.sin(th), np.full(n, z)], axis=1)
    dl = 2 * np.pi * R / n
    tang = np.stack([-np.sin(th), np.cos(th), np.zeros(n)], axis=1)
    G = Gamma * dl * tang
    sig = np.full(n, max(a, overlap * dl))
    return X, G, sig
