"""
Goland Wing Aeroelastic Benchmark — Flutter Detection via Envelope Growth.

Method:
  1. Apply initial tip perturbation (heave + twist)
  2. Run time-domain simulation at each velocity
  3. Measure oscillation envelope growth rate from tip displacement history
  4. Flutter = velocity where envelope transitions from decay to growth
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pterasoftware as ps
import time


def build_goland_wing(V_inf, dt=0.005, num_chords=100, alpha=2.0):
    chord = 1.8288
    semi_span = 6.096

    airplane = ps.geometry.airplane.Airplane(
        wings=[
            ps.geometry.wing.Wing(
                wing_cross_sections=[
                    ps.geometry.wing_cross_section.WingCrossSection(
                        num_spanwise_panels=8, chord=chord,
                        airfoil=ps.geometry.airfoil.Airfoil(name='naca0012', n_points_per_side=200),
                        spanwise_spacing='uniform'),
                    ps.geometry.wing_cross_section.WingCrossSection(
                        num_spanwise_panels=None, chord=chord,
                        Lp_Wcsp_Lpp=(0.0, semi_span, 0.0),
                        airfoil=ps.geometry.airfoil.Airfoil(name='naca0012', n_points_per_side=200),
                        spanwise_spacing=None),
                ],
                name='Goland Wing',
                Ler_Gs_Cgs=(0.0, 0.0, 0.0),
                angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
                symmetric=False, mirror_only=False,
                num_chordwise_panels=4, chordwise_spacing='uniform',
            ),
        ],
        name='Goland Wing Model',
    )

    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=V_inf, alpha=alpha, beta=0.0, nu=15.06e-6)
    opm = ps.movements.operating_point_movement.OperatingPointMovement(base_operating_point=op)
    wm = ps.movements.wing_movement.WingMovement(
        base_wing=airplane.wings[0],
        wing_cross_section_movements=[
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(base_wing_cross_section=wcs)
            for wcs in airplane.wings[0].wing_cross_sections
        ],
    )
    am = ps.movements.airplane_movement.AirplaneMovement(base_airplane=airplane, wing_movements=[wm])
    mv = ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_chords=num_chords, delta_time=dt)

    return mv, op


def compute_envelope_growth(signal, dt):
    """Compute exponential growth rate of the signal envelope.

    Returns growth rate σ (1/s): positive = growing, negative = decaying.
    """
    if len(signal) < 10:
        return 0.0

    abs_signal = np.abs(signal)
    peaks = []
    for i in range(1, len(abs_signal) - 1):
        if abs_signal[i] > abs_signal[i-1] and abs_signal[i] > abs_signal[i+1]:
            peaks.append((i * dt, abs_signal[i]))

    if len(peaks) < 3:
        return 0.0

    t_peaks = np.array([p[0] for p in peaks])
    a_peaks = np.maximum(np.array([p[1] for p in peaks]), 1e-15)

    # Skip first peak (initial transient)
    if len(t_peaks) > 4:
        log_a = np.log(a_peaks[1:])
        t_fit = t_peaks[1:]
    else:
        log_a = np.log(a_peaks)
        t_fit = t_peaks

    if len(t_fit) >= 2:
        coeffs = np.polyfit(t_fit, log_a, 1)
        return coeffs[0]
    return 0.0


def run_at_velocity(V, beam_params, dt=0.003, num_chords=100):
    from fluxvortex.aeroelastic_solver import AeroelasticSolver

    mv, op = build_goland_wing(V, dt=dt, num_chords=num_chords)
    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)
    n_steps = len(prob.steady_problems)

    solver = AeroelasticSolver(prob, beam_params=beam_params, relaxation=1.0)

    # Apply initial perturbation
    tip_node = solver.beam.nnodes - 1
    initial_tip_w = 0.05
    initial_tip_theta = np.radians(2.0)
    solver.beam.d[3 * tip_node] = initial_tip_w
    solver.beam.d[3 * tip_node + 2] = initial_tip_theta

    # Initialize acceleration
    K_r, M_r, _, free = solver.beam.apply_bc(solver.beam.K, solver.beam.M)
    a0_r = np.linalg.solve(M_r, -K_r @ solver.beam.d[free])
    solver.beam.a[free] = a0_r

    t0 = time.time()
    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    elapsed = time.time() - t0

    tip_w = np.array(solver.tip_w_history)
    tip_theta = np.array(solver.tip_theta_history)

    sigma_w = compute_envelope_growth(tip_w, dt)
    sigma_theta = compute_envelope_growth(tip_theta, dt)
    sigma = max(sigma_w, sigma_theta)

    max_w = np.max(np.abs(tip_w)) if len(tip_w) > 0 else 0

    return {
        'V': V, 'sigma': sigma, 'sigma_w': sigma_w, 'sigma_theta': sigma_theta,
        'max_w': max_w, 'elapsed': elapsed, 'n_steps': n_steps,
        'n_history': len(tip_w),
    }


def run_flutter_sweep():
    chord = 1.8288
    semi_span = 6.096

    beam_params = {
        'length': semi_span, 'n_elements': 8,
        'EI': 9.773e6, 'GJ': 0.988e6,
        'm_per_length': 35.72,
        'Ip': 35.72 * (chord**2) / 24,
        'x_ea_cg': 0.10 * chord,
        'structural_damping': 0.005,
    }

    dt = 0.003
    num_chords = 100

    # Phase 1: Coarse sweep
    coarse_velocities = [80, 100, 110, 120, 130, 140, 160, 180]

    print("=" * 80)
    print("Goland Wing Flutter Sweep")
    print(f"  EI={beam_params['EI']:.2e}, GJ={beam_params['GJ']:.2e}")
    print(f"  m={beam_params['m_per_length']}, Ip={beam_params['Ip']:.2f}")
    print(f"  x_ea_cg={beam_params['x_ea_cg']:.4f}m")
    print(f"  dt={dt}s, num_chords={num_chords}")
    print("=" * 80)

    results = []
    for V in coarse_velocities:
        print(f"  V={V:3d} m/s ... ", end="", flush=True)
        try:
            r = run_at_velocity(V, beam_params, dt, num_chords)
            status = "FLUTTER" if r['sigma_w'] > 0 else "stable"
            print(f"{status} (σ_w={r['sigma_w']:+.3f}, σ_θ={r['sigma_theta']:+.3f}, "
                  f"{r['elapsed']:.1f}s)")
            r['status'] = status
            results.append(r)
        except Exception as e:
            print(f"error: {e}")
            results.append({'V': V, 'sigma': 0, 'sigma_w': 0, 'sigma_theta': 0,
                           'max_w': 0, 'status': 'error'})

    # Find σ_w zero crossing region and refine
    valid = [r for r in results if r['status'] != 'error']
    crossing_lo, crossing_hi = None, None
    for i in range(len(valid) - 1):
        if valid[i]['sigma_w'] < 0 and valid[i+1]['sigma_w'] > 0:
            crossing_lo = valid[i]['V']
            crossing_hi = valid[i+1]['V']
            break

    if crossing_lo is not None and crossing_hi - crossing_lo > 4:
        print(f"\n  Refining σ_w=0 crossing between V={crossing_lo} and V={crossing_hi} m/s ...")
        step = max(2, (crossing_hi - crossing_lo) // 5)
        fine_vs = list(range(crossing_lo + step, crossing_hi, step))
        for V in fine_vs:
            print(f"  V={V:3d} m/s ... ", end="", flush=True)
            try:
                r = run_at_velocity(V, beam_params, dt, num_chords)
                status = "FLUTTER" if r['sigma_w'] > 0 else "stable"
                print(f"{status} (σ_w={r['sigma_w']:+.3f}, σ_θ={r['sigma_theta']:+.3f}, "
                      f"{r['elapsed']:.1f}s)")
                r['status'] = status
                results.append(r)
            except Exception as e:
                print(f"error: {e}")

    # Sort results
    results.sort(key=lambda r: r['V'])

    # Summary
    print(f"\n{'='*80}")
    print("FLUTTER SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"  {'V (m/s)':>10s} {'σ_w (1/s)':>12s} {'σ_θ (1/s)':>12s} {'Status':>10s}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*10}")

    flutter_speed = None
    for r in results:
        if r['status'] != 'error':
            print(f"  {r['V']:10d} {r['sigma_w']:+12.4f} {r['sigma_theta']:+12.4f} "
                  f"{r['status']:>10s}")

    # Interpolate flutter speed from σ_w=0 crossing
    valid = [r for r in results if r['status'] != 'error']
    for i in range(len(valid) - 1):
        if valid[i]['sigma_w'] < 0 and valid[i+1]['sigma_w'] > 0:
            s0, s1 = valid[i]['sigma_w'], valid[i+1]['sigma_w']
            V0, V1 = valid[i]['V'], valid[i+1]['V']
            flutter_speed = V0 - s0 * (V1 - V0) / (s1 - s0)
            break

    if flutter_speed:
        ref_flutter = 137.0
        err = abs(flutter_speed - ref_flutter) / ref_flutter * 100
        print(f"\n  Predicted flutter speed: {flutter_speed:.1f} m/s")
        print(f"  Reference (Goland & Luke): ~{ref_flutter} m/s")
        print(f"  Error: {err:.1f}%")
    else:
        print(f"\n  No flutter transition found")

    print(f"{'='*80}")
    return results


if __name__ == '__main__':
    ps.set_up_logging(level="Warning")
    run_flutter_sweep()
