"""FSI test with sign fix + MATLAB gradient, correct Yamano parameters."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fluxvortex.ancf_shell import ANCFShell
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver


def main():
    np.set_printoptions(precision=6, linewidth=120)

    U_star, M_star = 25.0, 1.0
    AR, V_inf = 1.0, 10.0
    rho_fluid = 1.225
    nu, thickness = 0.3, 1e-3
    Length, Width = 1.0, AR * 1.0

    mu_m = 1.0 / M_star
    rho_solid = M_star * rho_fluid * Length / thickness
    eta_m = mu_m / U_star**2
    E = 12.0 * rho_solid * Length**2 * V_inf**2 / (U_star**2 * thickness**2)

    print(f"Yamano: U*={U_star}, M*={M_star}, AR={AR}")
    print(f"rho_s={rho_solid:.1f} kg/m³, E={E:.3e} Pa, h={thickness*1000:.1f}mm")

    for nx, ny in [(8, 5)]:
        print(f"\n{'='*60}")
        print(f"Mesh: {nx}x{ny}")
        print(f"{'='*60}")

        x_vec = (np.arange(nx+1)/nx) * Length
        y_vec = np.arange(ny+1)/ny * Width

        nn = (nx+1)*(ny+1)
        nodes = np.zeros((nn, 3))
        for j in range(ny+1):
            for i in range(nx+1):
                nodes[j*(nx+1)+i] = [x_vec[i], y_vec[j], 0.0]

        ne = nx*ny
        quads = np.zeros((ne, 4), dtype=np.int32)
        for j in range(ny):
            for i in range(nx):
                quads[j*nx+i] = [j*(nx+1)+i, j*(nx+1)+(i+1), (j+1)*(nx+1)+(i+1), (j+1)*(nx+1)+i]

        shell = ANCFShell(nodes, quads, h=thickness, rho=rho_solid,
                          Ex=E, Ey=E, nu_xy=nu, mode='full', n_gauss=5)
        le_nodes = [j*(nx+1) for j in range(ny+1)]
        shell.set_bc(le_nodes, fix_slopes=True)

        V_inf_vec = np.array([V_inf, 0.0, 0.0])
        dx = Length / nx
        dt_uvlm = dx / V_inf
        struct_ratio = 45
        structural_dt = dt_uvlm / struct_ratio

        print(f"dx={dx:.4f}, dt_uvlm={dt_uvlm:.4f}, dt_struct={structural_dt:.2e}")

        solver = StandaloneHybridSolver(
            shell, V_inf_vec, rho_fluid=rho_fluid,
            structural_dt=structural_dt, uvlm_dt_ratio=struct_ratio,
            integrator='implicit', relaxation=0.95,
            newton_tol=1e-4, max_newton=30,
            max_particles=1000, wake_truncation=5.5,
            core_radius=1e-6, coupling='strong',
        )

        # Yamano pulse
        T_dur = 0.2 * Length / V_inf
        force_density_ref = rho_fluid * V_inf**2 / thickness
        F_body_peak = np.array([0.0, 0.0, -0.5 * force_density_ref])
        pulse_peak = shell.distributed_load(F_body_peak)
        total_Fz = pulse_peak[2::9].sum()
        print(f"Pulse: peak Fz={total_Fz:.1f} N, T_dur={T_dur:.4f}s")
        solver.set_pulse_distributed(pulse_peak, amplitude=1.0, duration=T_dur)

        n_struct = 3 * struct_ratio  # 135 structural → 3 UVLM → t*≈0.2
        print(f"Running {n_struct} structural steps...")
        solver.run(n_struct, print_every=struct_ratio)

        results = solver.get_results()
        if len(results['tip_w']) > 0:
            w_final = results['tip_w'][-1]
            w_star = w_final / Length
            t_star = results['sim_time'] * V_inf / Length
            print(f"\nFinal: t*={t_star:.4f}, w*={w_star:.6f}, w={w_final:.6f}m")

            matlab_w_star = -0.001489
            ratio = w_star / matlab_w_star if abs(matlab_w_star) > 1e-12 else 0
            print(f"MATLAB reference: w*={matlab_w_star:.6f} at t*=0.1995")
            print(f"Ratio our/MATLAB: {ratio:.2f}x")
            print(f"Discrepancy: {abs(ratio-1)*100:.1f}%")


if __name__ == '__main__':
    main()
