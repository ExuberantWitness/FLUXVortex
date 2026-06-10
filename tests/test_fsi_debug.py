"""Debug FSI forces: check sign/magnitude of aero forces during coupling."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fluxvortex.ancf_shell import ANCFShell
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver


def build_test_shell(nx=3, ny=2, L=1.0, W=1.0):
    x_vec = np.linspace(0, L, nx + 1)
    y_vec = np.linspace(0, W, ny + 1)
    nn = (nx + 1) * (ny + 1)
    nodes = np.zeros((nn, 3))
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes[j*(nx+1)+i] = [x_vec[i], y_vec[j], 0.0]
    ne = nx * ny
    quads = np.zeros((ne, 4), dtype=np.int32)
    for j in range(ny):
        for i in range(nx):
            quads[j*nx+i] = [j*(nx+1)+i, j*(nx+1)+(i+1), (j+1)*(nx+1)+(i+1), (j+1)*(nx+1)+i]
    shell = ANCFShell(nodes, quads, h=1e-3, rho=3125.0,
                      Ex=3.125e6, Ey=3.125e6, nu_xy=0.3, mode='full', n_gauss=2)
    le_nodes = [j*(nx+1) for j in range(ny+1)]
    shell.set_bc(le_nodes, fix_slopes=True)
    return shell


def main():
    np.set_printoptions(precision=6, linewidth=120)

    nx, ny = 3, 2
    L, W = 1.0, 1.0
    shell = build_test_shell(nx, ny, L, W)
    ndof = shell.ndof
    print(f"Mesh: {nx}×{ny}, DOF={ndof}, ne={shell.ne}")

    V_inf = 10.0
    rho_fluid = 1.225
    V_inf_vec = np.array([V_inf, 0.0, 0.0])

    dx = L / nx
    dt_uvlm = dx / V_inf
    struct_ratio = 45
    structural_dt = dt_uvlm / struct_ratio
    print(f"dx={dx:.4f}, dt_uvlm={dt_uvlm:.6f}, dt_struct={structural_dt:.2e}")

    solver = StandaloneHybridSolver(
        shell, V_inf_vec, rho_fluid=rho_fluid,
        structural_dt=structural_dt, uvlm_dt_ratio=struct_ratio,
        integrator='implicit', relaxation=0.95,
        newton_tol=1e-4, max_newton=30,
        max_particles=1000, wake_truncation=5.5,
        core_radius=1e-6, coupling='strong',
    )

    # Pulse
    T_dur = 0.2 * L / V_inf
    force_density_ref = rho_fluid * V_inf**2 / 1e-3
    F_body_peak = np.array([0.0, 0.0, -0.5 * force_density_ref])
    pulse_peak = shell.distributed_load(F_body_peak)
    solver.set_pulse_distributed(pulse_peak, amplitude=1.0, duration=T_dur)

    # Instead of using solver.run(), manually step through to inspect forces
    print("\nManual step-by-step with force diagnostics:")

    # Initial UVLM solve
    solver._uvlm_step_initial()
    print(f"\nt=0: gamma range = [{solver.uvlm.gamma.min():.6f}, {solver.uvlm.gamma.max():.6f}]")
    print(f"  forces_no_vstruct Fz sum = {np.sum(solver.uvlm.forces_no_vstruct[:,:,2]):.6f}")

    forces_nv = solver.uvlm.forces_no_vstruct.copy()
    dp_lift2 = solver.uvlm.dp_lift2.copy()

    # Run a few structural steps manually, checking forces
    for step in range(10):
        V_struct = solver._compute_structural_velocity_at_colloc()
        F_bernoulli = solver._load_transfer(forces_nv)
        F_lift2 = solver._compute_lift2_force(V_struct, dp_lift2)
        F_pulse = solver._pulse_force()
        F_total = F_bernoulli + F_lift2 + F_pulse

        # Sum z forces
        Fz_bernoulli = F_bernoulli[2::9].sum()
        Fz_lift2 = F_lift2[2::9].sum()
        Fz_pulse = F_pulse[2::9].sum()
        Fz_total_z = Fz_bernoulli + Fz_lift2 + Fz_pulse

        tip_idx = (shell.nn - 1) * 9 + 2
        w_before = shell.q[tip_idx]

        shell.step_newmark(F_total, structural_dt, newton_tol=1e-4, max_newton=30)

        w_after = shell.q[tip_idx]

        if step < 5 or step % 2 == 0:
            print(f"  step {step:2d}: w={w_after:.6e}, Fz_bern={Fz_bernoulli:+.4f}, "
                  f"Fz_lift2={Fz_lift2:+.4f}, Fz_pulse={Fz_pulse:+.1f}, Fz_tot={Fz_total_z:+.1f}")

        solver._update_uvlm_vertices()
        solver._displacement_transfer()
        solver._pulse_elapsed += structural_dt
        solver.sim_time += structural_dt

    # Now do a UVLM solve and check new forces
    print(f"\n--- UVLM solve at step 10 (t*={solver.sim_time*V_inf/L:.4f}) ---")
    print(f"  Before solve: gamma = {solver.uvlm.gamma[:,0]}")
    solver._uvlm_step()
    print(f"  After solve:  gamma = {solver.uvlm.gamma[:,0]}")
    print(f"  forces_no_vstruct Fz sum = {np.sum(solver.uvlm.forces_no_vstruct[:,:,2]):.4f}")
    print(f"  dp_lift2 z sum = {np.sum(solver.uvlm.dp_lift2[:,:,2]):.4f}")
    print(f"  plate tip z = {shell.q[tip_idx]:.6f}")

    # Check: is V_struct · dp_lift2 the correct sign?
    V_struct = solver._compute_structural_velocity_at_colloc()
    for i in range(solver._nx):
        for j in range(solver._ny):
            v_dot_dp = np.dot(V_struct[i,j], solver.uvlm.dp_lift2[i,j])
            if abs(v_dot_dp) > 1e-10:
                print(f"  panel({i},{j}): V_struct_z={V_struct[i,j,2]:.4e}, dp_lift2_z={solver.uvlm.dp_lift2[i,j,2]:.4e}, V·dp={v_dot_dp:.4e}")


if __name__ == '__main__':
    main()
