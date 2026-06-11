"""MATLAB-native UVLM induction (generate_q1234_mat.m verbatim, vectorized).

q1234(targets, rings) -> (Nt, 3*Ns) induced-velocity blocks of unit-Gamma rings
(MATLAB sign: q1234 = -(q1+q2+q3+q4)). Includes BOTH MATLAB regularizations:
  - denominator eps_v added to |r1 x r2|^2     (the 9.3e-5 AIC residual source)
  - algebraic vortex core Kv = h^2/(h^(2Nc)+r_core^(2Nc))^(1/Nc),
    r_core = max(all 4 segment lengths of the SOURCE ring, Length/Nx) * r_eps
Segments: (1->4), (2->1), (3->2), (4->3) per the r11/r21 definitions.
"""
import numpy as np

def _cnorm(v, axis=-1):
    """norm that stays analytic for complex-step (sqrt of plain square sum);
    bit-identical to np.linalg.norm for real input."""
    return np.sqrt(np.sum(v * v, axis=axis))


def _cmax(a, b):
    """maximum by real part (complex-step-safe; identical for real input)."""
    return np.where(np.real(a) >= np.real(b), a, b)



MEPS = np.finfo(float).eps   # MATLAB eps


def q1234_mat(rc, x1, x2, x3, x4, Length, Nx, r_eps, Ncore=2, eps_v=1e-9):
    """rc (Nt,3); x1..x4 (Ns,3) ring corners; r_eps = scalar (fine 1e-6 / rough 0.1).
    Returns (Nt, Ns, 3) induced velocity per unit Gamma (MATLAB -V sign)."""
    Nt = rc.shape[0]; Ns = x1.shape[0]
    R = rc[:, None, :]                       # (Nt,1,3)
    P = [x[None, :, :] for x in (x1, x2, x3, x4)]   # 4x (1,Ns,3)
    # r1/r2 per segment: seg k pairs (a_k, b_k) = (1,4),(2,1),(3,2),(4,3)
    pairs = [(0, 3), (1, 0), (2, 1), (3, 2)]
    # source-ring core radius (per ring): max segment length vs Length/Nx
    seglen = []
    for a, b in pairs:
        seglen.append(_cnorm((P[b] - P[a])[0]))   # (Ns,)
    max_r0 = seglen[0]
    for _s in seglen[1:]:
        max_r0 = _cmax(max_r0, _s)
    max_r0 = _cmax(max_r0, Length / Nx)            # (Ns,)
    r_core = max_r0 * r_eps                              # (Ns,)
    V = np.zeros((Nt, Ns, 3), dtype=np.result_type(rc, x1))
    for a, b in pairs:
        r1 = R - P[a]                                    # (Nt,Ns,3)
        r2 = R - P[b]
        r0 = r1 - r2                                     # = b - a... (x_b - x_a)
        cr = np.cross(r1, r2)
        ncr2 = np.einsum('tsc,tsc->ts', cr, cr)          # |r1xr2|^2
        n1 = _cnorm(r1)
        n2 = _cnorm(r2)
        dot = np.einsum('tsc,tsc->ts',
                        r0, r1 / _cmax(n1, MEPS)[..., None]
                        - r2 / _cmax(n2, MEPS)[..., None])
        q = cr / (ncr2 + eps_v)[..., None] * (dot / (4.0 * np.pi))[..., None]
        # vortex core factor
        n0 = _cnorm(r0)
        h = np.sqrt(ncr2) / n0                           # perpendicular distance
        Kv = h**2 / (h**(2 * Ncore) + r_core[None, :]**(2 * Ncore))**(1.0 / Ncore)
        V += Kv[..., None] * q
    return -V


def aic_from_q1234(V, normals):
    """A[i,j] = V[i,j,:] . n[i]  (MATLAB q_mat_ni)."""
    return np.einsum('tsc,tc->ts', V, normals)
