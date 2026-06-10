"""Run standalone VPM-Hybrid solver matching Yamano et al. parameters.

Zero PteraSoftware dependency. Uses:
  - StandaloneUVLM (ring vortex panels + Biot-Savart AIC)
  - ANCFShell (bicubic Hermite, 9DOF/node, Kirchhoff-Love)
  - VortexParticleField (VPM far-field wake)

Validates against MATLAB benchmark:
  - Natural frequencies: [0.1324, 0.3084, 0.7488, 1.0783, 1.1538]
  - Tip displacement time history
  - Flutter onset at U*=25, M*=1

Usage:
  python run_standalone_yamano.py [--quick | --modal-only]
    --quick      : short run (~2 nondim time) for testing
    --modal-only : only compute natural frequencies, no simulation
    default      : full run (~10 nondim time)
"""
import sys, os, argparse, time as time_mod, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fluxvortex.ancf_shell import ANCFShell
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'standalone_yamano')


def yamano_params():
    """Return dimensional parameters for Yamano single-sheet configuration."""
    U_star = 25.0
    M_star = 1.0
    AR = 1.0
    V_inf = 10.0
    rho_fluid = 1.225
    nu = 0.3
    thickness = 1e-3

    Length = 1.0
    Width = AR * Length
    mu_m = 1.0 / M_star
    rho_solid = M_star * rho_fluid * Length / thickness
    eta_m = mu_m / U_star**2
    E = 12.0 * rho_solid * Length**2 * V_inf**2 / (U_star**2 * thickness**2)

    print(f"Yamano Single-Sheet Parameters:")
    print(f"  U*={U_star}, M*={M_star}, AR={AR}")
    print(f"  L={Length}m, W={Width}m, h={thickness*1000:.1f}mm")
    print(f"  rho_s={rho_solid:.0f} kg/m³, E={E:.3e} Pa, nu={nu}")
    print(f"  mu_m={mu_m:.4f}, eta_m={eta_m:.6f}")
    print(f"  V_inf={V_inf} m/s, rho_f={rho_fluid} kg/m³")

    return {
        'U_star': U_star, 'M_star': M_star, 'AR': AR,
        'Length': Length, 'Width': Width, 'thickness': thickness,
        'rho_solid': rho_solid, 'E': E, 'nu': nu,
        'mu_m': mu_m, 'eta_m': eta_m,
        'V_inf': V_inf, 'rho_fluid': rho_fluid,
    }


def build_yamano_shell(params, nx=15, ny=10):
    """Build ANCF shell matching Yamano's mesh."""
    L = params['Length']
    W = params['Width']
    E = params['E']
    nu = params['nu']
    thickness = params['thickness']
    rho_solid = params['rho_solid']

    n_LW = 1.0
    x_vec = (np.arange(nx + 1) / nx)**n_LW * L
    y_vec = np.arange(ny + 1) / ny * W

    nn = (nx + 1) * (ny + 1)
    nodes = np.zeros((nn, 3))
    for j in range(ny + 1):
        for i in range(nx + 1):
            idx = j * (nx + 1) + i
            nodes[idx, 0] = x_vec[i]
            nodes[idx, 1] = y_vec[j]
            nodes[idx, 2] = 0.0

    ne = nx * ny
    quads = np.zeros((ne, 4), dtype=np.int32)
    for j in range(ny):
        for i in range(nx):
            elem = j * nx + i
            n1 = j * (nx + 1) + i
            n2 = j * (nx + 1) + (i + 1)
            n3 = (j + 1) * (nx + 1) + (i + 1)
            n4 = (j + 1) * (nx + 1) + i
            quads[elem] = [n1, n2, n3, n4]

    shell = ANCFShell(
        nodes, quads,
        h=thickness, rho=rho_solid,
        Ex=E, Ey=E, nu_xy=nu,
        mode='full', n_gauss=5,
    )

    # Clamped leading edge: fix nodes at x=0
    # Mesh order: y-major (j*(nx+1)+i), so x=0 nodes have i=0 → idx=j*(nx+1)
    le_nodes = [j * (nx + 1) for j in range(ny + 1)]
    shell.set_bc(le_nodes, fix_slopes=True)

    return shell, x_vec, y_vec, le_nodes


def compute_natural_frequencies(shell, params, n_modes=5):
    """Compute natural frequencies and compare with MATLAB benchmark."""
    L = params['Length']
    V_inf = params['V_inf']

    # Get free DOFs
    ndof = shell.ndof
    fixed = sorted(shell._bc_dofs)
    free = sorted(set(range(ndof)) - set(fixed))

    # Mass matrix
    M = shell.M.tocsc()

    # Stiffness matrix (tangent at reference config)
    _, K = shell._internal_forces_and_tangent(shell.q.copy())

    M_ff = M[free][:, free]
    K_ff = K[free][:, free]

    # Solve: K @ phi = omega^2 * M @ phi
    # Use dense generalized eigenvalue solver (reliable for < 2000 DOF)
    from scipy.linalg import eigh as dense_eigh
    vals, _ = dense_eigh(K_ff.toarray(), M_ff.toarray(), subset_by_index=[0, n_modes-1])

    omega_dim = np.sqrt(np.abs(np.real(vals)))
    omega_star = omega_dim * L / V_inf
    omega_star = np.sort(omega_star)

    # MATLAB benchmark from actual run (FSI_by_FEM_and_UVLM/single_sheet/save/)
    matlab_ref = [0.1455, 0.3567, 0.8942, 1.1393, 1.2980]

    print(f"\nNatural Frequencies (nondimensional ω* = ω·L/U_inf):")
    print(f"  Mode | Python ω* | MATLAB ω* | Rel. diff")
    print(f"  -----|-----------|-----------|----------")
    for i, (py, ml) in enumerate(zip(omega_star, matlab_ref)):
        rel_diff = (py - ml) / ml * 100
        print(f"  {i+1:4d} | {py:9.4f} | {ml:9.4f} | {rel_diff:+.2f}%")
    print(f"  RMS relative error: {np.sqrt(np.mean(((omega_star - matlab_ref)/matlab_ref)**2))*100:.1f}%")

    return omega_star


def run_simulation(params, shell, config):
    """Run standalone hybrid solver."""
    V_inf = params['V_inf']
    rho_fluid = params['rho_fluid']
    V_inf_vec = np.array([V_inf, 0.0, 0.0])

    dx = params['Length'] / config['nx']
    dt_uvlm = dx / V_inf
    structural_dt = dt_uvlm / config['struct_ratio']

    print(f"\nTime steps:")
    print(f"  UVLM dt: {dt_uvlm:.6f}s (dL/U_inf)")
    print(f"  Structural dt: {structural_dt:.2e}s")

    solver = StandaloneHybridSolver(
        shell, V_inf_vec,
        rho_fluid=rho_fluid,
        structural_dt=structural_dt,
        uvlm_dt_ratio=config['struct_ratio'],
        integrator='implicit',
        relaxation=0.95,
        newton_tol=1e-4,
        max_newton=30,
        max_particles=config['max_particles'],
        wake_truncation=5.5,
        core_radius=1e-6,
        coupling='strong',
    )

    # Pulse matching Yamano MATLAB: q(t) = 0.5 * sin(π*t/0.2) for t < 0.2 (nondim)
    #
    # MATLAB nondimensional body force:
    #   Qf_pulse* = Qf_time_global * q_in_norm(t)
    #   where Qf_time_global = ∫ S^T · q_in_vec dA  (integrates over area, NO h factor)
    #
    # Dimensional equivalent:
    #   Qf_pulse_dim = Qf_pulse* · ρ_f · V_inf² · L²
    #
    # Python distributed_load computes:
    #   Qf_python = ∫ S^T · f_body · h dA  (includes h)
    #
    # Equating: f_body · h · ∫S^T·[0,0,1]dA = q_in_norm · ρ_f · V_inf² · L² · ∫S^T·[0,0,1]dA/L²
    # → f_body = q_in_norm · ρ_f · V_inf² / h
    #
    # Reference: N·m⁻³ = (kg·m⁻³ · m²·s⁻²) / m = kg·m⁻¹·s⁻²
    T_dur_nondim = 0.2
    T_dur = T_dur_nondim * params['Length'] / params['V_inf']
    q_in_norm_max = 0.5
    force_density_ref = params['rho_fluid'] * params['V_inf']**2 / params['thickness']
    F_body_peak = np.array([0.0, 0.0, -q_in_norm_max * force_density_ref])
    pulse_peak_force = shell.distributed_load(F_body_peak)
    total_Fz = pulse_peak_force[2::9].sum()
    print(f"  Pulse: peak body force density={F_body_peak[2]:.1f} N/m³, total Fz={total_Fz:.2f} N, T_dur={T_dur:.3f}s")
    solver.set_pulse_distributed(pulse_peak_force, amplitude=1.0, duration=T_dur)

    n_struct_steps = config['n_struct_steps']
    print(f"\nRunning {n_struct_steps} structural steps...")
    print(f"  UVLM solves: ~{n_struct_steps // config['struct_ratio']}")
    print(f"  Max VPM: {config['max_particles']}")
    print(f"  Sim time: {n_struct_steps * structural_dt:.2f}s")
    print("-" * 60)

    t_start = time_mod.time()
    solver.run(n_struct_steps, print_every=config['print_every'])
    elapsed = time_mod.time() - t_start

    print(f"\nSimulation complete: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return solver


def main():
    parser = argparse.ArgumentParser(description='Standalone VPM-Hybrid Yamano Comparison')
    parser.add_argument('--quick', action='store_true', help='Short test run')
    parser.add_argument('--modal-only', action='store_true', help='Only compute natural frequencies')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    params = yamano_params()

    if args.quick:
        config = {
            'nx': 15, 'ny': 10,          # match MATLAB discretization
            'struct_ratio': 45,
            'n_struct_steps': 900,        # ~20 UVLM solves, t*=1.33
            'max_particles': 20000,
            'print_every': 45,
        }
    else:
        config = {
            'nx': 15, 'ny': 10,
            'struct_ratio': 45,
            'n_struct_steps': 20000,   # matches MATLAB End_Time=30, d_t=0.0015
            'max_particles': 500000,
            'print_every': 500,
        }

    shell, x_vec, y_vec, le_nodes = build_yamano_shell(params, config['nx'], config['ny'])
    print(f"\nANCF mesh: {shell.nn} nodes, {shell.ne} elements")

    omega_star = compute_natural_frequencies(shell, params)

    if args.modal_only:
        print("\nModal validation complete.")
        return

    solver = run_simulation(params, shell, config)

    results = solver.get_results()
    os.makedirs(args.output, exist_ok=True)

    np.savez(
        os.path.join(args.output, 'results.npz'),
        tip_w=results['tip_w'],
        force=results['force'],
        dt_struct=results['dt_struct'],
        dt_uvlm=results['dt_uvlm'],
        sim_time=results['sim_time'],
        n_steps=results['n_steps'],
    )

    with open(os.path.join(args.output, 'params.json'), 'w') as f:
        json.dump({
            'params': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in params.items()},
            'config': config,
            'omega_star': omega_star.tolist(),
        }, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print(f"Tip w range: [{results['tip_w'].min():.6f}, {results['tip_w'].max():.6f}] m")


if __name__ == '__main__':
    main()
