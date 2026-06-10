"""Solve MATLAB's exact Newmark linear system in Python — using MATLAB-dumped
M, K, F, BC indices — and check whether the result matches MATLAB's q_p1.

If yes: MATLAB's formula DOES give q_p1 = 1.77e-8 and Python's _step_newmark
must be solving a DIFFERENT system (i.e., we've localized a real Python bug).

If no: there's an artifact in the MATLAB dump itself.
"""
import os, sys
import numpy as np
from scipy.io import loadmat
from scipy.sparse import eye as speye, bmat as spbmat, csc_matrix as spcsc
from scipy.sparse.linalg import spsolve

ml = loadmat('FSI_by_FEM_and_UVLM/single_sheet/fixtures/step1_qp1.mat',
             squeeze_me=True, struct_as_record=False)
Nx = int(ml['Nx']); Ny = int(ml['Ny']); N_q_all = int(ml['N_q_all'])
d_t = float(ml['d_t'])

M_global    = ml['M_for_newmark']
K_global    = ml['dq_Qe_mem_only']         # membrane K only (what MATLAB uses by default)
M_global    = M_global.toarray() if hasattr(M_global, 'toarray') else np.asarray(M_global)
K_global    = K_global.toarray() if hasattr(K_global, 'toarray') else np.asarray(K_global)
Qf_pulse    = np.asarray(ml['Qf_pulse'], dtype=float).ravel()
i_vec_bc    = np.asarray(ml['i_vec_bc_full'], dtype=int).ravel() - 1   # 1-indexed → 0-indexed
q_n         = np.asarray(ml['q_vec'], dtype=float).ravel()
dq_n        = np.zeros(N_q_all)

q_p1_ml = np.asarray(ml['q_p1'], dtype=float).ravel()
dq_p1_ml = np.asarray(ml['dq_p1'], dtype=float).ravel()

free = np.setdiff1d(np.arange(N_q_all), i_vec_bc)
nf = len(free)
print(f"N_q_all={N_q_all}, nf={nf}, n_bc={len(i_vec_bc)}, d_t={d_t}")

# Build A1, A2 (MATLAB convention, theta_a=0 → C_damp=2, Qd=0)
alpha_v = 0.5
C_damp = 2.0

M_ff = M_global[np.ix_(free, free)]
K_ff = K_global[np.ix_(free, free)]

I = np.eye(nf)
O = np.zeros((nf, nf))
D_bot_left = C_damp * d_t / 2.0 * K_ff
D_mat = np.block([[I, O], [D_bot_left, M_ff]])
X2_mat = np.block([[O, I], [O, O]])
A1 = D_mat - alpha_v * d_t * X2_mat
A2 = D_mat + (1.0 - alpha_v) * d_t * X2_mat

X_n_free = np.concatenate([q_n[free], dq_n[free]])
RHS = A2 @ X_n_free
RHS_pulse = np.concatenate([np.zeros(nf), Qf_pulse[free]]) * d_t
X_p1_free = np.linalg.solve(A1, RHS + RHS_pulse)

q_p1_py = q_n.copy()
dq_p1_py = dq_n.copy()
q_p1_py[free] = X_p1_free[:nf]
dq_p1_py[free] = X_p1_free[nf:]

# Compare to MATLAB result
print(f"\nTip Z direct system solve:")
tip_node_1idx = Nx * (Ny + 1) + Ny // 2 + 1
z_dof = (tip_node_1idx - 1) * 9 + 2
print(f"  Python solving MATLAB system: q_p1[tip_z] = {q_p1_py[z_dof]:+.5e}")
print(f"  MATLAB's saved q_p1[tip_z]:                {q_p1_ml[z_dof]:+.5e}")
print(f"  Python dq_p1[tip_z] = {dq_p1_py[z_dof]:+.5e}")
print(f"  MATLAB dq_p1[tip_z] = {dq_p1_ml[z_dof]:+.5e}")
print(f"  Trapezoidal check: 0.5*dt*dq_p1_py = {0.5*d_t*dq_p1_py[z_dof]:+.5e}")
print(f"                     q_p1_py        = {q_p1_py[z_dof]:+.5e}")

dq = q_p1_py - q_p1_ml
print(f"\n|q_p1_py - q_p1_ml|_max = {np.max(np.abs(dq)):.4e}")
print(f"|q_p1_py - q_p1_ml|_F   = {np.linalg.norm(dq):.4e}")
