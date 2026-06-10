"""Compare MATLAB and Python tip displacement trajectories step-by-step.

MATLAB fixture stores h_X_vec[dof, time_idx]. We extract tip_z at node 175 (TE
outer corner, matches Python tip_idx) at MATLAB time steps. Then align with
Python's tip_w_history via real time.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from load_matlab_fixture import MatlabFixture


def extract_matlab_trajectory(fx, node_idx=175, dof_offset=2, d_t=2e-3):
    """Return (time_array, tip_z_array) from h_X_vec at given node z-dof."""
    hX = np.asarray(fx._raw['h_X_vec'])  # (2*N_q_all, n_time)
    z_dof = node_idx * 9 + dof_offset
    tip_z = hX[z_dof, :]
    n_steps = len(tip_z)
    t = np.arange(n_steps) * d_t  # nondim
    return t, tip_z


def main():
    fx_path = '/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/FSI_by_FEM_and_UVLM/single_sheet/fixtures/fixture_step3_t0.1995.mat'
    fx = MatlabFixture(fx_path)

    # MATLAB d_t = 2e-3, 151 time steps in fixture → t* ∈ [0, 0.3]
    t_ml, tip_ml = extract_matlab_trajectory(fx, node_idx=175, dof_offset=2, d_t=2e-3)

    print(f"MATLAB tip_z (node 175, TE outer corner):")
    print(f"  Times t* = {t_ml[0]:.4f} ... {t_ml[-1]:.4f} ({len(t_ml)} samples)")
    print(f"  Range: {tip_ml.min():+.4e} ... {tip_ml.max():+.4e}")
    print()
    print(f"  t* = 0.05  →  tip_z = {tip_ml[25]:+.4e}")
    print(f"  t* = 0.10  →  tip_z = {tip_ml[50]:+.4e}")
    print(f"  t* = 0.15  →  tip_z = {tip_ml[75]:+.4e}")
    print(f"  t* = 0.20  →  tip_z = {tip_ml[100]:+.4e}")
    print(f"  t* = 0.25  →  tip_z = {tip_ml[125]:+.4e}")

    # Save for Python comparison
    out_path = '/tmp/matlab_tip_trajectory.npz'
    np.savez(out_path, t_star=t_ml, tip_z=tip_ml)
    print(f"\nSaved MATLAB trajectory to {out_path}")


if __name__ == "__main__":
    main()
