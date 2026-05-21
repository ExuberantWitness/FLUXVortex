"""
Vortex particle Biot-Savart kernels — vectorized NumPy implementation.
Ported from FLOWVPM's Gaussian-erf kernel with Jacobian computation.
"""
import numpy as np
from math import sqrt, pi
from scipy.special import erf


# ── Gaussian-erf regularizing kernel ──────────────────────────────────
def zeta_gauserf(r_bar):
    """Vorticity distribution function ζ(r/σ)."""
    c = 1.0 / (2.0 * pi) ** 1.5
    return c * np.exp(-0.5 * r_bar ** 2)


def g_gauserf(r_bar):
    """Regularizing function g(r/σ) for velocity."""
    c = sqrt(2.0 / pi)
    return erf(r_bar / sqrt(2.0)) - c * r_bar * np.exp(-0.5 * r_bar ** 2)


def dgdr_gauserf(r_bar):
    """Derivative dg/d(r/σ)."""
    c = sqrt(2.0 / pi)
    return c * r_bar ** 2 * np.exp(-0.5 * r_bar ** 2)


def g_dgdr_gauserf(r_bar):
    """Evaluate g and dg/dr simultaneously."""
    c = sqrt(2.0 / pi)
    exp_val = np.exp(-0.5 * r_bar ** 2)
    g = erf(r_bar / sqrt(2.0)) - c * r_bar * exp_val
    dgdr = c * r_bar ** 2 * exp_val
    return g, dgdr


# ── Chunked pairwise Biot-Savart ─────────────────────────────────────
_CHUNK = 512  # chunk size to control memory


def velocity_from_particles(target_points, src_pos, src_gamma, src_sigma):
    """
    Induced velocity at M target points from N vortex particles.
    Uses Gaussian-erf regularization.

    Parameters
    ----------
    target_points : (M, 3) ndarray
    src_pos       : (N, 3) ndarray — particle positions
    src_gamma     : (N, 3) ndarray — vectorial circulation
    src_sigma     : (N,)   ndarray — core sizes

    Returns
    -------
    U : (M, 3) ndarray — induced velocity at each target point
    """
    M = target_points.shape[0]
    N = src_pos.shape[0]
    U = np.zeros((M, 3))

    if N == 0:
        return U

    _C = -1.0 / (4.0 * pi)

    for i0 in range(0, M, _CHUNK):
        i1 = min(i0 + _CHUNK, M)
        t_block = target_points[i0:i1]          # (Bt, 3)
        for j0 in range(0, N, _CHUNK):
            j1 = min(j0 + _CHUNK, N)
            s_pos = src_pos[j0:j1]               # (Bs, 3)
            s_gam = src_gamma[j0:j1]             # (Bs, 3)
            s_sig = src_sigma[j0:j1]             # (Bs,)

            dx = t_block[:, None, :] - s_pos[None, :, :]   # (Bt, Bs, 3)
            r2 = np.sum(dx ** 2, axis=-1)                    # (Bt, Bs)
            r = np.sqrt(r2)
            r = np.maximum(r, 1e-12)

            r_bar = r / s_sig[None, :]          # (Bt, Bs)
            g = g_gauserf(r_bar)                 # (Bt, Bs)

            cross = np.cross(dx, s_gam[None, :, :])  # (Bt, Bs, 3)

            # U = -1/(4π) * g/r³ * (dx × Γ)
            coeff = _C * g / (r ** 3)            # (Bt, Bs)
            U[i0:i1] += np.sum(coeff[:, :, None] * cross, axis=1)

    return U


def jacobian_from_particles(tgt_pos, tgt_gamma, src_pos, src_gamma, src_sigma):
    """
    Compute velocity gradient tensor J = dU/dx at each target particle
    due to all source particles. Used for vortex stretching.

    Parameters
    ----------
    tgt_pos   : (M, 3)  target particle positions (usually = src)
    tgt_gamma : (M, 3)  target particle circulations (unused here, for API compat)
    src_pos   : (N, 3)  source particle positions
    src_gamma : (N, 3)  source vectorial circulation
    src_sigma : (N,)    source core sizes

    Returns
    -------
    J : (M, 3, 3) ndarray — Jacobian dU_i/dx_j at each target
        Stored as J[target, velocity_component, position_component]
    """
    M = tgt_pos.shape[0]
    N = src_pos.shape[0]
    J = np.zeros((M, 3, 3))

    if N == 0:
        return J

    _C = -1.0 / (4.0 * pi)

    for i0 in range(0, M, _CHUNK):
        i1 = min(i0 + _CHUNK, M)
        t_block = tgt_pos[i0:i1]
        for j0 in range(0, N, _CHUNK):
            j1 = min(j0 + _CHUNK, N)
            s_pos = src_pos[j0:j1]
            s_gam = src_gamma[j0:j1]
            s_sig = src_sigma[j0:j1]

            dx = t_block[:, None, :] - s_pos[None, :, :]   # (Bt, Bs, 3)
            r2 = np.sum(dx ** 2, axis=-1)
            r = np.sqrt(np.maximum(r2, 1e-24))

            r_bar = r / s_sig[None, :]
            g, dgdr = g_dgdr_gauserf(r_bar)

            # K × Γ = -1/(4π) * 1/r³ * (dx × Γ)
            r3inv = 1.0 / (r ** 3)
            cross_KG = np.cross(dx, s_gam[None, :, :])  # (Bt, Bs, 3)
            KxG = _C * cross_KG * r3inv[:, :, None]      # (Bt, Bs, 3)

            # aux = dgdr/(σ*r) - 3*g/r²
            aux = dgdr / (s_sig[None, :] * r) - 3.0 * g / (r ** 2)

            # aux2 = -1/(4π) * g / r³
            aux2 = _C * g * r3inv

            # J_ij = aux * (K×Γ)_i * dx_j + δ-correction
            # row 0 (U_x):
            J[i0:i1, 0, 0] += np.sum(aux * KxG[:, :, 0] * dx[:, :, 0], axis=1)
            J[i0:i1, 0, 1] += np.sum(aux * KxG[:, :, 0] * dx[:, :, 1] + aux2 * s_gam[None, :, 2], axis=1)
            J[i0:i1, 0, 2] += np.sum(aux * KxG[:, :, 0] * dx[:, :, 2] - aux2 * s_gam[None, :, 1], axis=1)

            # row 1 (U_y):
            J[i0:i1, 1, 0] += np.sum(aux * KxG[:, :, 1] * dx[:, :, 0] - aux2 * s_gam[None, :, 2], axis=1)
            J[i0:i1, 1, 1] += np.sum(aux * KxG[:, :, 1] * dx[:, :, 1], axis=1)
            J[i0:i1, 1, 2] += np.sum(aux * KxG[:, :, 1] * dx[:, :, 2] + aux2 * s_gam[None, :, 0], axis=1)

            # row 2 (U_z):
            J[i0:i1, 2, 0] += np.sum(aux * KxG[:, :, 2] * dx[:, :, 0] + aux2 * s_gam[None, :, 1], axis=1)
            J[i0:i1, 2, 1] += np.sum(aux * KxG[:, :, 2] * dx[:, :, 1] - aux2 * s_gam[None, :, 0], axis=1)
            J[i0:i1, 2, 2] += np.sum(aux * KxG[:, :, 2] * dx[:, :, 2], axis=1)

    return J
