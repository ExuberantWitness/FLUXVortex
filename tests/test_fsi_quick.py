"""Quick FSI coupling test — smallest mesh that can capture fluid effects.

Runs standalone hybrid solver with fluid coupling on a 5×3 mesh for a few
UVLM steps. Compares structural-only vs FSI tip displacement at t*=0.2.
"""
import sys, os, time as time_mod
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fluxvortex.ancf_shell import ANCFShell
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver


def build_test_shell(nx=5, ny=3, L=1.0, W=1.0):
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
            elem = j * nx + i
            quads[elem] = [
                j * (nx + 1) + i,
                j * (nx + 1) + (i + 1),
                (j + 1) * (nx + 1) + (i + 1),
                (j + 1) * (nx + 1) + i,
            ]

    shell = ANCFShell(
        nodes, quads, h=1e-3, rho=3125.0,
        Ex=3.125e6, Ey=3.125e6, nu_xy=0.3,
        mode='full', n_gauss=2,
    )
    le_nodes = [j * (nx + 1) for j in range(ny + 1)]
    shell.set_bc(le_nodes, fix_slopes=True)
    return shell


def main():
    np.set_printoptions(precision=6, linewidth=120)

    for (nx, ny) in [(5, 3), (8, 5)]:
        print(f"\n{'='*70}")
        print(f"Mesh: {nx}×{ny}")
        print(f"{'='*70}")

        L, W = 1.0, 1.0
        shell = build_test_shell(nx, ny, L, W)
        print(f"  DOF={shell.ndof}, ne={shell.ne}")

        V_inf = 10.0
        rho_fluid = 1.225
        V_inf_vec = np.array([V_inf, 0.0, 0.0])

        dx = L / nx
        dt_uvlm = dx / V_inf
        struct_ratio = 45
        structural_dt = dt_uvlm / struct_ratio

        print(f"  dx={dx:.4f}m, dt_uvlm={dt_uvlm:.6f}s, dt_struct={structural_dt:.2e}s")

        # Target: 3 UVLM steps (3×45=135 structural steps), t* ≈ 3*0.0015*45=0.2025
        n_uvlm_steps = 3
        n_struct = n_uvlm_steps * struct_ratio  # 135

        solver = StandaloneHybridSolver(
            shell, V_inf_vec,
            rho_fluid=rho_fluid,
            structural_dt=structural_dt,
            uvlm_dt_ratio=struct_ratio,
            integrator='implicit',
            relaxation=0.95,
            newton_tol=1e-4,
            max_newton=30,
            max_particles=1000,
            wake_truncation=5.5,
            core_radius=1e-6,
            coupling='strong',
        )

        # Pulse force (same as Yamano)
        T_dur_nondim = 0.2
        T_dur = T_dur_nondim * L / V_inf
        q_in_norm_max = 0.5
        force_density_ref = rho_fluid * V_inf**2 / 1e-3
        F_body_peak = np.array([0.0, 0.0, -q_in_norm_max * force_density_ref])
        pulse_peak_force = shell.distributed_load(F_body_peak)
        total_Fz = pulse_peak_force[2::9].sum()
        print(f"  Pulse peak Fz = {total_Fz:.2f} N, T_dur = {T_dur:.4f}s")

        solver.set_pulse_distributed(pulse_peak_force, amplitude=1.0, duration=T_dur)

        t_start = time_mod.time()
        solver.run(n_struct, print_every=90)
        elapsed = time_mod.time() - t_start
        print(f"  Elapsed: {elapsed:.1f}s")

        results = solver.get_results()
        tip_w = results['tip_w']
        if len(tip_w) > 0:
            w_final = tip_w[-1]
            w_star = w_final / L
            t_star = results['sim_time'] * V_inf / L
            print(f"\n  Final: t*={t_star:.4f}, w*={w_star:.6f}, w={w_final:.6f}m")
            print(f"  MATLAB FSI ref:  w* = -0.001489 at t*=0.1995")
            ratio = abs(w_star / -0.001489) if abs(w_star) > 1e-12 else 0
            print(f"  Ratio Python/MATLAB: {ratio:.2f}x")


if __name__ == '__main__':
    main()
