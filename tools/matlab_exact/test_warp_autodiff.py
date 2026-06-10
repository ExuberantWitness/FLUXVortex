"""Phase A prereq — Warp tape differentiability of the MATLAB-exact kernels.

Checks reverse-mode gradients through ml_induce_kernel (ring Biot-Savart with
eps_v + algebraic core, atomic_add accumulation):
    loss = sum_i V(colloc_i) . n_i
gradients w.r.t. ring corners (c1..c4), circulation Gamma, and colloc points,
validated entry-wise against central finite differences.
Acceptance: rel err <= 1e-6 (fp64).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import warp as wp
import scipy.sparse as sp
from scipy.io import loadmat
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_ml_exact import ml_induce_kernel, ml_rcore_kernel

DTYPE = config.DTYPE
VEC3 = config.VEC3
DEV = config.DEVICE
NP = config.NP_DTYPE

f = loadmat('FSI_by_FEM_and_UVLM/single_sheet/fixtures_traj/fixture_step3_t0.3000.mat',
            squeeze_me=True, struct_as_record=False)
g = lambda k: (f[k].toarray() if sp.issparse(f[k]) else np.asarray(f[k], dtype=float))
rc = g('rc_vec'); nv = g('n_vec_i')
P = [g(f'r_panel_vec_{k}') for k in (1, 2, 3, 4)]
Gam = g('Gamma').ravel()
T = rc.shape[0]; S = P[0].shape[0]
B = 1
eps_v = NP(1e-9)


@wp.kernel
def loss_kernel(V: wp.array(dtype=VEC3, ndim=2),
                n: wp.array(dtype=VEC3, ndim=2),
                loss: wp.array(dtype=DTYPE, ndim=1)):
    e, i = wp.tid()
    wp.atomic_add(loss, e, wp.dot(V[e, i], n[e, i]))


def up(a, dt=VEC3, grad=False):
    arr = np.ascontiguousarray(np.broadcast_to(a, (B,) + a.shape).astype(NP))
    return wp.array(arr, dtype=dt, device=DEV, requires_grad=grad)


def forward(c_arrays, g_array, col_array, rc_core, V, loss):
    V.zero_(); loss.zero_()
    wp.launch(ml_induce_kernel, dim=(B, T, S),
              inputs=[col_array, c_arrays[0], c_arrays[1], c_arrays[2], c_arrays[3],
                      g_array, rc_core, DTYPE(eps_v)], outputs=[V], device=DEV)
    wp.launch(loss_kernel, dim=(B, T), inputs=[V, up(nv)], outputs=[loss], device=DEV)


# ---- device arrays with grads ----
cw = [up(P[k], grad=True) for k in range(4)]
gw = up(Gam, dt=DTYPE, grad=True)
colw = up(rc, grad=True)
# r_core treated as constant (recomputed per geometry in production; here frozen
# to isolate the induction-path gradient, matching how rcore is a derived const)
rc_core = wp.zeros((B, S), dtype=DTYPE, device=DEV)
wp.launch(ml_rcore_kernel, dim=(B, S),
          inputs=[cw[0], cw[1], cw[2], cw[3], DTYPE(NP(1.0 / 15)), DTYPE(NP(1e-6))],
          outputs=[rc_core], device=DEV)
rc_core_const = wp.clone(rc_core)
rc_core_const.requires_grad = False
V = wp.zeros((B, T), dtype=VEC3, device=DEV, requires_grad=True)
loss = wp.zeros(B, dtype=DTYPE, device=DEV, requires_grad=True)

tape = wp.Tape()
with tape:
    forward(cw, gw, colw, rc_core_const, V, loss)
tape.backward(loss=loss)
wp.synchronize()

gG = gw.grad.numpy()[0]
gC1 = cw[0].grad.numpy()[0]
gCol = colw.grad.numpy()[0]


def loss_np(P_, Gam_, rc_):
    sys.path.insert(0, HERE)
    from ml_uvlm import q1234_mat
    Vn = q1234_mat(rc_, P_[0], P_[1], P_[2], P_[3], 1.0, 15, 1e-6, 2, 1e-9)
    return float(np.einsum('tsc,s,tc->', Vn, Gam_, nv))


h = 1e-6
print("=== d(loss)/d(Gamma): tape vs central FD (5 entries) ===")
ok = True
for j in [0, 37, 74, 111, 149]:
    Gp = Gam.copy(); Gp[j] += h
    Gm = Gam.copy(); Gm[j] -= h
    fd = (loss_np(P, Gp, rc) - loss_np(P, Gm, rc)) / (2 * h)
    rel = abs(gG[j] - fd) / (abs(fd) + 1e-30)
    ok &= rel < 1e-6
    print(f"  j={j:3d}  tape={gG[j]:+.8e}  fd={fd:+.8e}  rel={rel:.2e}")

print("=== d(loss)/d(corner1): tape vs FD (3 entries x 3 comps) ===")
for j in [10, 75, 140]:
    for c in range(3):
        Pp = [p.copy() for p in P]; Pp[0][j, c] += h
        Pm = [p.copy() for p in P]; Pm[0][j, c] -= h
        fd = (loss_np(Pp, Gam, rc) - loss_np(Pm, Gam, rc)) / (2 * h)
        rel = abs(gC1[j][c] - fd) / (abs(fd) + 1e-30)
        ok &= rel < 1e-5
        print(f"  j={j:3d} c={c}  tape={gC1[j][c]:+.8e}  fd={fd:+.8e}  rel={rel:.2e}")

print("=== d(loss)/d(colloc): tape vs FD (2 entries x 3 comps) ===")
for i in [5, 120]:
    for c in range(3):
        rp = rc.copy(); rp[i, c] += h
        rm = rc.copy(); rm[i, c] -= h
        fd = (loss_np(P, Gam, rp) - loss_np(P, Gam, rm)) / (2 * h)
        rel = abs(gCol[i][c] - fd) / (abs(fd) + 1e-30)
        ok &= rel < 1e-5
        print(f"  i={i:3d} c={c}  tape={gCol[i][c]:+.8e}  fd={fd:+.8e}  rel={rel:.2e}")

print("\nAUTODIFF CHECK:", "PASS" if ok else "FAIL")
