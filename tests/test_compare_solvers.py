"""Direct comparison: StandaloneHybridSolver vs MatlabFSISolver.

Applies small AoA to get non-zero gamma, then compares forces.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fluxvortex.ancf_shell import ANCFShell
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
from fluxvortex.matlab_fsi_solver import MatlabFSISolver


def main():
    np.set_printoptions(precision=6, linewidth=120)

    nx, ny = 5, 3
    L, W = 1.0, 1.0
    V_inf = 10.0
    rho_fluid = 1.225
    alpha_deg = 1.0  # Small AoA for non-zero forces

    thick = 1e-3
    U_star, M_star = 25.0, 1.0
    rho_solid = M_star * rho_fluid * L / thick
    E = 12.0 * rho_solid * L**2 * V_inf**2 / (U_star**2 * thick**2)
    print(f"Yamano: rho_s={rho_solid:.1f}, E={E:.3e}")

    x_vec = np.linspace(0, L, nx + 1)
    y_vec = np.linspace(0, W, ny + 1)
    nn = (nx + 1) * (ny + 1)
    nodes = np.zeros((nn, 3))
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes[j * (nx + 1) + i] = [x_vec[i], y_vec[j], 0.0]
    ne = nx * ny
    quads = np.zeros((ne, 4), dtype=np.int32)
    for j in range(ny):
        for i in range(nx):
            quads[j * nx + i] = [
                j * (nx + 1) + i, j * (nx + 1) + (i + 1),
                (j + 1) * (nx + 1) + (i + 1), (j + 1) * (nx + 1) + i]

    alpha = np.radians(alpha_deg)
    V_inf_vec = np.array([V_inf * np.cos(alpha), 0.0, -V_inf * np.sin(alpha)])

    shell = ANCFShell(nodes, quads, h=thick, rho=rho_solid,
                      Ex=E, Ey=E, nu_xy=0.3, mode='full', n_gauss=5)
    le_nodes = [j * (nx + 1) for j in range(ny + 1)]
    shell.set_bc(le_nodes, fix_slopes=True)

    dx = L / nx
    dt_uvlm = dx / V_inf
    struct_ratio = 45
    structural_dt = dt_uvlm / struct_ratio

    # ── Solver 1: StandaloneHybridSolver ──
    solver1 = StandaloneHybridSolver(
        shell, V_inf_vec, rho_fluid=rho_fluid,
        structural_dt=structural_dt, uvlm_dt_ratio=struct_ratio,
        integrator='implicit', relaxation=0.95,
        newton_tol=1e-4, max_newton=30,
        max_particles=1000, wake_truncation=5.5,
        core_radius=1e-6, coupling='strong',
    )

    # ── Initial UVLM solve with AoA ──
    solver1._uvlm_step_initial()
    gamma1 = solver1.uvlm.gamma.copy()
    forces1_nv = solver1.uvlm.forces_no_vstruct.copy()
    dp_lift2_1 = solver1.uvlm.dp_lift2.copy()

    print(f"\nsolver1 (dimensional) gamma:\n{gamma1}")
    print(f"solver1 gamma sum: {gamma1.sum():.6f}")
    print(f"solver1 forces_no_vstruct Fz per panel:\n{forces1_nv[:,:,2]}")
    print(f"solver1 forces_no_vstruct total Fz: {forces1_nv[:,:,2].sum():.4f} N")
    print(f"solver1 dp_lift2 z per panel:\n{dp_lift2_1[:,:,2]}")

    # ── solver2: compute forces manually (MatlabFSISolver-like) ──
    # Build identical UVLM
    from fluxvortex.standalone_uvlm import StandaloneUVLM
    uvlm2 = StandaloneUVLM(solver1._uvlm_vertices.copy(), V_inf_vec,
                           rho=rho_fluid, core_radius=1e-6)
    uvlm2.build_aic()

    # Verify AIC match
    print(f"\nAIC match: {np.allclose(solver1.uvlm._AIC, uvlm2._AIC)}")
    print(f"Gamma match: {np.allclose(gamma1, uvlm2.gamma)}")

    # Solve UVLM
    V_struct_colloc = np.zeros((nx, ny, 3))
    uvlm2.solve(V_ext_colloc=None, V_struct_colloc=V_struct_colloc)

    # Compute forces using both gradient formulas
    V_wake_colloc = np.zeros((nx, ny, 3))
    V_bound_colloc = uvlm2.compute_bound_induction_at_colloc()
    uvlm2.compute_forces(dt_uvlm, V_ext_colloc=V_wake_colloc + V_bound_colloc,
                        V_struct_colloc=V_struct_colloc)

    gamma2 = uvlm2.gamma
    print(f"\nsolver2 gamma:\n{gamma2}")
    print(f"Gamma diff norm: {np.linalg.norm(gamma2 - gamma1):.2e}")

    forces2 = uvlm2.forces
    forces2_nv = uvlm2.forces_no_vstruct
    dp_lift2_2 = uvlm2.dp_lift2
    print(f"solver2 forces_no_vstruct total Fz: {forces2_nv[:,:,2].sum():.4f} N")

    # ── KEY COMPARISON: What would MATLAB compute? ──
    # MATLAB uses dγ/dx for chordwise gradient
    # Let's compute the MATLAB-style Bernoulli force using the same gamma
    print("\n" + "=" * 70)
    print("FORCE FORMULA COMPARISON (same gamma distribution)")
    print("=" * 70)

    # Extract tangent vectors (same as uvlm compute_forces)
    tau_x_all = np.zeros((nx, ny, 3))
    tau_y_all = np.zeros((nx, ny, 3))
    dx_all = np.zeros((nx, ny))
    dy_all = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            c = uvlm2._corners[i, j]
            r21 = c[1] - c[0]; r34 = c[2] - c[3]
            r14 = c[0] - c[3]; r23 = c[1] - c[2]
            tau_x = (r21 + r34) / 2
            tau_y = (r14 + r23) / 2
            tau_x_norm = np.linalg.norm(tau_x) + 1e-15
            tau_y_norm = np.linalg.norm(tau_y) + 1e-15
            tau_x_all[i,j] = tau_x / tau_x_norm
            tau_y_all[i,j] = tau_y / tau_y_norm
            dx_all[i,j] = tau_x_norm
            dy_all[i,j] = tau_y_norm

    # Method 1: dΦ/dx = γ/dx (our fix) — cumulative potential gradient
    dPhi_dx = gamma2 / dx_all  # Potential jump gradient
    dPhi_dy = np.zeros_like(dPhi_dx)
    gamma_bound = np.cumsum(gamma2, axis=0)
    for j in range(ny):
        if ny == 1:
            dPhi_dy[:, j] = 0
        elif j == 0:
            dPhi_dy[:, j] = gamma_bound[:, j] / dy_all[:, j]
        elif j == ny - 1:
            dPhi_dy[:, j] = -gamma_bound[:, j] / dy_all[:, j]
        else:
            dPhi_dy[:, j] = (gamma_bound[:, j+1] - gamma_bound[:, j-1]) / (2 * dy_all[:, j])

    # Method 2: dγ/dx (MATLAB-style) — ring circulation gradient
    dgamma_dx = np.zeros_like(gamma2)
    dgamma_dx[0, :] = gamma2[0, :] / dx_all[0, :]
    dgamma_dx[1:, :] = (gamma2[1:, :] - gamma2[:-1, :]) / dx_all[1:, :]

    # Compute Bernoulli pressure for each method
    V_ext_only = V_inf_vec + V_wake_colloc + V_bound_colloc
    dp_our = np.zeros((nx, ny))
    dp_matlab = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            V_dot_tau_x = np.dot(V_ext_only[i,j], tau_x_all[i,j])
            dp_our[i,j] = rho_fluid * V_dot_tau_x * dPhi_dx[i,j]
            dp_matlab[i,j] = rho_fluid * V_dot_tau_x * dgamma_dx[i,j]

    # Forces
    Fz_our = np.sum(dp_our * uvlm2._areas)
    Fz_matlab = np.sum(dp_matlab * uvlm2._areas)

    # Also compute the cumulative gamma gradient
    print(f"\nGamma distribution (same for both):")
    for i in range(nx):
        g_sum = gamma2[i,:].sum()
        print(f"  row {i}: gamma sum={g_sum:.6f}, "
              f"gamma bound={gamma_bound[i,:].sum():.6f}, "
              f"dΦ/dx sum={dPhi_dx[i,:].sum():.4f}, "
              f"dγ/dx sum={dgamma_dx[i,:].sum():.4f}")

    print(f"\nBernoulli Fz comparison:")
    print(f"  Our method (dΦ/dx = γ/dx):  {Fz_our:.4f} N")
    print(f"  MATLAB method (dγ/dx):      {Fz_matlab:.4f} N")
    print(f"  Ratio our/MATLAB:            {Fz_our/(Fz_matlab+1e-15):.2f}x")

    # KJ theorem check
    gamma_te = gamma2[-1, :]  # TE ring circulation
    dy_j = W / ny
    Fz_KJ_ring = rho_fluid * V_inf * np.sum(gamma_te * dy_j)
    Gamma_bound_TE = gamma_bound[-1, :]  # Total bound circulation at TE
    Fz_KJ_total = rho_fluid * V_inf * np.sum(Gamma_bound_TE * dy_j)
    print(f"\nKJ theorem check:")
    print(f"  Fz from γ_TE only:     {Fz_KJ_ring:.4f} N")
    print(f"  Fz from Γ_bound_total: {Fz_KJ_total:.4f} N")
    print(f"  Our/dΦ matches Γ_total KJ: {abs(Fz_our/Fz_KJ_total - 1):.6f}")
    print(f"  MATLAB/dγ matches γ_TE KJ:  {abs(Fz_matlab/Fz_KJ_ring - 1):.6f}")


if __name__ == '__main__':
    main()
