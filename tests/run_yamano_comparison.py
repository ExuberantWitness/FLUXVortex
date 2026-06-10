"""Run ANCF Hybrid aeroelastic simulation matching Yamano et al. parameters.

Generates comparison data for:
  1. Streamlines around flapping sheets
  2. Flow velocity distributions (slice contours)
  3. Wake behind sheets (VPM particle field)
  4. Snapshot of flapping sheets (ANCF surface deformation)

References:
  - Yamano et al., J. Sound and Vibration (2020): flutter boundary, LCO amplitude
  - Yamano et al., MEJ (2021): energy harvesting, aspect ratio effects
  - Yamano et al., IJSSD (2022): spanwise plate deformation
  - https://github.com/KRproject-tech/FSI_by_FEM_and_UVLM

Usage:
  python run_yamano_comparison.py [--quick] [--full]
    --quick : short run (~5 chord lengths, coarse mesh) for testing
    --full  : full run (~40 chord lengths, refined mesh) for publication
    default : medium run (~15 chord lengths)
"""
import sys
import os
import argparse
import time as time_mod
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

import pterasoftware as ps
from fluxvortex.ancf_hybrid_coupling import (
    ANCFHybridAeroelasticSolver,
    yamano_single_sheet_params,
    print_yamano_params,
    compute_velocity_field,
)
from fluxvortex.ancf_aero_coupling import build_ancf_wing, build_uvlm_problem
from fluxvortex.ancf_shell import ANCFShell, NDOF_NODE

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'yamano_comparison')


def create_yamano_setup(args):
    """Build ANCF mesh + UVLM problem matching Yamano's single_sheet.

    Key parameters from Yamano et al. (J Sound Vib 2020, Mech Eng J 2021):
      - Nx=15, Ny=10 (150 elements, 176 nodes), clamped LE
      - U*=25, M*=1.0, AR=1.0, alpha=0°
      - d_t = 0.0015 (nondim structural dt)
      - End_Time = 30 (nondim, ~3s dimensional at V=10m/s)
      - dt_fluid = dL/U_inf = 0.00667s (wake shed per element-length)
      - struct_ratio = 45 (structural sub-steps per fluid step)
      - Wake truncation at 5.5 chords downstream
      - Half-sine initial z-force pulse for t < 0.2
    """

    # ── Parameter selection ──
    if args.quick:
        config = {
            'nx': 6, 'ny': 4,
            'num_chords': 6,       # End_Time ~ 6 nondim (quick test)
            'dt': None,            # auto-computed from dL/U_inf
            'struct_ratio': 30,    # reduced for speed in quick mode
            'max_steps': 100,
            'snapshot_every': 10,
            'U_star': 25.0,
            'M_star': 1.0,
            'V_inf': 10.0,
            'alpha': 0.0,          # matched to Yamano
            'rho_fluid': 1.225,
            'AR': 1.0,
        }
    elif args.full:
        config = {
            'nx': 15, 'ny': 10,    # EXACT Yamano mesh
            'num_chords': 30,      # End_Time = 30 nondim
            'dt': None,            # auto-computed
            'struct_ratio': 45,    # EXACT Yamano sub-cycling
            'max_steps': None,
            'snapshot_every': 50,
            'U_star': 25.0,
            'M_star': 1.0,
            'V_inf': 10.0,
            'alpha': 0.0,
            'rho_fluid': 1.225,
            'AR': 1.0,
        }
    else:
        config = {
            'nx': 10, 'ny': 8,
            'num_chords': 15,      # End_Time = 15 nondim
            'dt': None,            # auto-computed
            'struct_ratio': 45,    # EXACT Yamano sub-cycling
            'max_steps': None,
            'snapshot_every': 30,
            'U_star': 25.0,
            'M_star': 1.0,
            'V_inf': 10.0,
            'alpha': 0.0,
            'rho_fluid': 1.225,
            'AR': 1.0,
        }

    # Auto-compute UVLM dt from dL/U_inf (Yamano's approach)
    if config['dt'] is None:
        dL = 1.0 / config['nx']  # chordwise element length
        config['dt'] = dL / config['V_inf']  # wake shed per element-length

    # ── Compute dimensional parameters ──
    params = yamano_single_sheet_params(
        U_star=config['U_star'], M_star=config['M_star'],
        AR=config['AR'], V_inf=config['V_inf'],
        rho_fluid=config['rho_fluid'],
    )
    print_yamano_params(params)

    # ── Build ANCF shell ──
    t0 = time_mod.time()
    shell, le_nodes = build_ancf_wing(
        Length=params['Length'], Width=params['Width'],
        thickness=params['thickness'],
        nx=config['nx'], ny=config['ny'],
        rho=params['rho'], E=params['E'], nu=params['nu'],
        bc_type='clamped',
    )
    print(f"  ANCF mesh: {shell.nn} nodes, {shell.ne} elements "
          f"({config['nx']}×{config['ny']})")
    print(f"  Build time: {time_mod.time() - t0:.1f}s")

    # ── Build UVLM problem ──
    t0 = time_mod.time()
    mv, op = build_uvlm_problem(
        shell, config['V_inf'], rho=config['rho_fluid'],
        alpha=config['alpha'], dt=config['dt'],
        num_chords=config['num_chords'],
    )
    print(f"  UVLM: {config['num_chords']} chord lengths, dt={config['dt']}s")
    print(f"  Build time: {time_mod.time() - t0:.1f}s")

    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)
    n_steps = len(prob.steady_problems)
    sim_time = n_steps * config['dt']
    print(f"  Total steps: {n_steps}, simulation time: {sim_time:.1f}s")

    return config, params, shell, prob, mv, op


class RecordingHybridSolver(ANCFHybridAeroelasticSolver):
    """ANCFHybridAeroelasticSolver with periodic snapshot capture."""
    def __init__(self, *args, snapshot_every=20, print_every=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot_every = snapshot_every
        self._print_every = print_every
        self._t_start = None
        self._n_steps = None

    def _populate_next_airplanes_wake(self):
        super()._populate_next_airplanes_wake()
        step = self._current_step
        if step >= 1 and self._t_start is not None:
            if step % self._snapshot_every == 0:
                self.capture_snapshot()
            if step % self._print_every == 0:
                tip_w = self.tip_w_history[-1] if self.tip_w_history else 0.0
                elapsed = time_mod.time() - self._t_start
                print(f"  Step {step:5d}/{self._n_steps} | "
                      f"tip_w={tip_w:+.4e} m | "
                      f"VPM: {self._vpm_field.np:6d} | "
                      f"elapsed: {elapsed:.0f}s")


def run_simulation(config, shell, prob):
    """Run ANCFHybridAeroelasticSolver with snapshot capture."""

    solver = RecordingHybridSolver(
        prob, shell,
        integrator='implicit',
        relaxation=0.7,
        structural_dt_ratio=config['struct_ratio'],
        newton_tol=1e-4,
        max_newton=30,
        max_particles=100000,
        nu=0.0,
        rlxf=0.3,
        snapshot_every=config['snapshot_every'],
        print_every=max(1, config['snapshot_every'] // 4),
    )

    dt_struct = config['dt'] / config['struct_ratio']
    print(f"\nRunning simulation:")
    print(f"  UVLM dt: {config['dt']}s, struct dt: {dt_struct:.2e}s")
    print(f"  Integrator: implicit Newmark-β (tol=1e-4, max_iter=30)")
    print(f"  Relaxation: 0.7, sub-cycles: {config['struct_ratio']}")
    print(f"  Max VPM particles: 100000")
    print(f"  Snapshot every: {config['snapshot_every']} steps")
    print("-" * 60)

    # Half-sine force pulse matching Yamano: q_in(t) = 0.5*sin(pi*t/0.2)* (t<0.2)
    # Dimensional: 0.02s duration (0.2 * L/U_inf), moderate amplitude
    solver.set_initial_pulse(amplitude=0.5, duration=0.02)

    solver._t_start = time_mod.time()
    solver._n_steps = len(prob.steady_problems)

    try:
        solver.run(prescribed_wake=True, calculate_streamlines=False,
                   show_progress=False)
        elapsed = time_mod.time() - solver._t_start
        print(f"\nSimulation complete: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Snapshots captured: {len(solver.snapshots)}")

    except Exception as e:
        elapsed = time_mod.time() - solver._t_start
        print(f"\nSimulation interrupted after {elapsed:.0f}s: {e}")
        import traceback
        traceback.print_exc()

    return solver


def analyze_results(solver, config, params):
    """Analyze simulation results: flutter detection, LCO amplitude."""

    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)

    tip_w = np.array(solver.tip_w_history)
    dt = config['dt']  # tip history recorded once per UVLM step

    if len(tip_w) < 10:
        print("  Insufficient data for analysis")
        return {}

    # ── Tip displacement statistics ──
    max_abs_w = np.max(np.abs(tip_w))
    rms_w = np.sqrt(np.mean(tip_w**2))

    # ── Flutter detection: envelope growth rate ──
    from fluxvortex.ancf_aero_coupling import compute_envelope_growth
    sigma_w = compute_envelope_growth(tip_w, dt)

    # ── Frequency analysis ──
    # Use second half of signal for steady-state frequency
    half = len(tip_w) // 2
    tip_second_half = tip_w[half:] - np.mean(tip_w[half:])

    if len(tip_second_half) > 20:
        from scipy.fft import rfft, rfftfreq
        n_fft = len(tip_second_half)
        freqs = rfftfreq(n_fft, dt)
        spectrum = np.abs(rfft(tip_second_half * np.hanning(n_fft)))
        dominant_idx = np.argmax(spectrum[1:]) + 1
        dominant_freq = freqs[dominant_idx]
        dominant_amp = spectrum[dominant_idx] / n_fft * 2
    else:
        dominant_freq = params.get('freq1_beam', 0.0)
        dominant_amp = 0.0

    print(f"  Tip displacement:")
    print(f"    Max |w|:        {max_abs_w:.6f} m")
    print(f"    RMS w:          {rms_w:.6f} m")
    print(f"    Growth rate σ:  {sigma_w:+.4f} 1/s ({'FLUTTER' if sigma_w > 0 else 'STABLE'})")
    print(f"    Dominant freq:  {dominant_freq:.3f} Hz")
    print(f"    Dominant amp:   {dominant_amp:.6f} m")
    print(f"    Beam freq (ref): {params.get('freq1_beam', 0):.3f} Hz")

    # ── Wake statistics ──
    if solver._vpm_field.np > 0:
        vpm = solver.get_vpm_particles()
        total_circulation = np.sum(np.linalg.norm(vpm['gamma'], axis=1))
        print(f"  Wake:")
        print(f"    VPM particles:  {vpm['np']}")
        print(f"    Total |Γ|:      {total_circulation:.4f} m²/s")

    results = {
        'max_abs_w': float(max_abs_w),
        'rms_w': float(rms_w),
        'sigma_w': float(sigma_w),
        'dominant_freq': float(dominant_freq),
        'dominant_amp': float(dominant_amp),
        'vpm_np': int(solver._vpm_field.np),
        'n_steps': len(tip_w),
        'sim_time': len(tip_w) * dt,
    }
    return results


def save_all_data(solver, config, params, results, output_dir):
    """Save simulation data for later plotting."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Save tip history ──
    np.savez(
        os.path.join(output_dir, 'tip_history.npz'),
        tip_w=np.array(solver.tip_w_history),
        tip_theta=np.array(solver.tip_theta_history),
        force=np.array(solver.force_history),
        dt_uvlm=config['dt'],  # tip history is per UVLM step
    )

    # ── Save snapshots ──
    solver.save_snapshots(output_dir)

    # ── Save final state ──
    nodes, quads = solver.get_sheet_surface()
    np.savez(
        os.path.join(output_dir, 'final_surface.npz'),
        nodes=nodes,
        quads=quads,
    )

    # ── Save VPM final state ──
    vpm = solver.get_vpm_particles()
    np.savez(
        os.path.join(output_dir, 'vpm_final.npz'),
        positions=vpm['positions'],
        gamma=vpm['gamma'],
        sigma=vpm['sigma'],
        np=vpm['np'],
    )

    # ── Save config & results ──
    with open(os.path.join(output_dir, 'run_info.json'), 'w') as f:
        json.dump({
            'config': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in config.items()},
            'params': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in params.items()},
            'results': results,
        }, f, indent=2)

    print(f"\nAll data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='ANCF Hybrid Aeroelastic Simulation — Yamano Comparison')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run (~5 chord lengths)')
    parser.add_argument('--full', action='store_true',
                        help='Full publication run (~40 chord lengths)')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--plot-only', type=str, default=None,
                        help='Skip simulation, only plot from saved data dir')
    args = parser.parse_args()

    output_dir = args.output

    if args.plot_only:
        print(f"Plot-only mode: loading data from {args.plot_only}")
        # Import and run plotting
        from plot_yamano_results import generate_all_plots
        generate_all_plots(args.plot_only)
        return

    ps.set_up_logging(level="Warning")

    print("=" * 60)
    print("ANCF Hybrid Aeroelastic Simulation")
    print("Comparison target: Yamano et al. (2020, 2021, 2022)")
    print("=" * 60)

    # ── Setup ──
    config, params, shell, prob, mv, op = create_yamano_setup(args)
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({
            'config': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in config.items()},
            'params': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in params.items()},
        }, f, indent=2)

    # ── Run ──
    solver = run_simulation(config, shell, prob)

    # ── Analyze ──
    results = analyze_results(solver, config, params)

    # ── Save ──
    save_all_data(solver, config, params, results, output_dir)

    # ── Plot ──
    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    from plot_yamano_results import generate_all_plots
    generate_all_plots(output_dir)

    print(f"\nDone! Results in: {output_dir}")


if __name__ == '__main__':
    main()
