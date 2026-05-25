"""
Goland Wing Flutter Animation — Stable vs Flutter side-by-side comparison.

Runs aeroelastic simulations at two velocities (stable + flutter),
then creates an animated GIF showing wing surface deformation over time.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pterasoftware as ps
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time


def build_goland_wing(V_inf, dt=0.003, num_chords=60, alpha=2.0):
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


def run_simulation(V, beam_params, dt=0.003, num_chords=60):
    from fluxvortex.aeroelastic_solver import AeroelasticSolver

    mv, op = build_goland_wing(V, dt=dt, num_chords=num_chords)
    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)

    solver = AeroelasticSolver(prob, beam_params=beam_params, relaxation=1.0)

    tip_node = solver.beam.nnodes - 1
    solver.beam.d[3 * tip_node] = 0.05
    solver.beam.d[3 * tip_node + 2] = np.radians(2.0)

    K_r, M_r, _, free = solver.beam.apply_bc(solver.beam.K, solver.beam.M)
    a0_r = np.linalg.solve(M_r, -K_r @ solver.beam.d[free])
    solver.beam.a[free] = a0_r

    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)

    return {
        'w_history': np.array(solver.w_history),
        'theta_history': np.array(solver.theta_history),
        'y_nodes': solver.beam.y_nodes,
        'dt': dt,
        'n_steps': len(prob.steady_problems),
    }


def make_wing_surface(y_nodes, w, theta, chord=1.8288, x_ea_frac=0.33, nx=20):
    """Reconstruct wing surface from beam nodal displacements."""
    x_ea = x_ea_frac * chord
    x = np.linspace(0, chord, nx)
    y = y_nodes

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for j, yj in enumerate(y):
        w_j = w[j]
        theta_j = theta[j]
        for i, xi in enumerate(x):
            Z[j, i] = w_j + (xi - x_ea) * np.sin(theta_j)

    return X, Y, Z


def create_animation():
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

    V_stable = 100
    V_flutter = 160
    dt = 0.003
    num_chords = 60

    # Check for cached data
    cache_file = os.path.join(os.path.dirname(__file__), '..', 'figures', 'flutter_data.npz')
    cache_file = os.path.normpath(cache_file)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file):
        print("Loading cached simulation data...")
        data = np.load(cache_file, allow_pickle=True)
        data_stable = {
            'w_history': data['w_stable'],
            'theta_history': data['theta_stable'],
            'y_nodes': data['y_nodes'],
            'dt': float(data['dt']),
        }
        data_flutter = {
            'w_history': data['w_flutter'],
            'theta_history': data['theta_flutter'],
            'y_nodes': data['y_nodes'],
            'dt': float(data['dt']),
        }
    else:
        print(f"Running simulation V={V_stable} m/s (stable)...")
        t0 = time.time()
        data_stable = run_simulation(V_stable, beam_params, dt, num_chords)
        print(f"  Done in {time.time()-t0:.1f}s ({len(data_stable['w_history'])} frames)")

        print(f"Running simulation V={V_flutter} m/s (flutter)...")
        t0 = time.time()
        data_flutter = run_simulation(V_flutter, beam_params, dt, num_chords)
        print(f"  Done in {time.time()-t0:.1f}s ({len(data_flutter['w_history'])} frames)")

        # Cache
        np.savez_compressed(cache_file,
            w_stable=data_stable['w_history'],
            theta_stable=data_stable['theta_history'],
            w_flutter=data_flutter['w_history'],
            theta_flutter=data_flutter['theta_history'],
            y_nodes=data_stable['y_nodes'],
            dt=dt,
        )
        print(f"  Cached to {cache_file}")

    # Build animation
    print("Building animation...")

    w_s = data_stable['w_history']
    th_s = data_stable['theta_history']
    w_f = data_flutter['w_history']
    th_f = data_flutter['theta_history']
    y_nodes = data_stable['y_nodes']

    n_frames = min(len(w_s), len(w_f))
    # Target ~100 frames for compact GIF
    target_frames = 100
    skip = max(1, n_frames // target_frames)
    frame_indices = list(range(0, n_frames, skip))
    if len(frame_indices) > target_frames:
        frame_indices = frame_indices[:target_frames]
    t_all = np.arange(n_frames) * dt

    fig = plt.figure(figsize=(12, 6), facecolor='#1a1a2e')

    # 3D wing views
    ax3d_s = fig.add_subplot(2, 2, 1, projection='3d', facecolor='#1a1a2e')
    ax3d_f = fig.add_subplot(2, 2, 2, projection='3d', facecolor='#1a1a2e')

    # Time history plots
    ax_ts = fig.add_subplot(2, 2, 3, facecolor='#16213e')
    ax_tf = fig.add_subplot(2, 2, 4, facecolor='#16213e')

    z_limit = max(np.max(np.abs(w_s)), np.max(np.abs(w_f))) * 1.3

    def style_3d(ax, title, color):
        ax.set_xlim(-0.5, chord + 0.5)
        ax.set_ylim(-0.5, semi_span + 0.5)
        ax.set_zlim(-z_limit, z_limit)
        ax.set_xlabel('x (m)', color='white', fontsize=8, labelpad=1)
        ax.set_ylabel('y (m)', color='white', fontsize=8, labelpad=1)
        ax.set_zlabel('z (m)', color='white', fontsize=8, labelpad=1)
        ax.set_title(title, color=color, fontsize=13, fontweight='bold', pad=5)
        ax.view_init(elev=25, azim=-60)
        ax.tick_params(colors='white', labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333')
        ax.yaxis.pane.set_edgecolor('#333')
        ax.zaxis.pane.set_edgecolor('#333')
        ax.grid(True, alpha=0.15)

    def style_ts(ax, title, color):
        ax.set_xlim(0, n_frames * dt)
        ax.set_ylim(-z_limit, z_limit)
        ax.set_xlabel('Time (s)', color='white', fontsize=9)
        ax.set_ylabel('Tip w (m)', color='white', fontsize=9)
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
        ax.tick_params(colors='white', labelsize=8)
        ax.axhline(0, color='#444', linewidth=0.5)
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def update(frame_idx):
        fi = frame_indices[frame_idx]

        for ax in [ax3d_s, ax3d_f]:
            ax.cla()

        # Stable wing
        X, Y, Z = make_wing_surface(y_nodes, w_s[fi], th_s[fi], chord)
        ax3d_s.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.85,
                            vmin=-z_limit, vmax=z_limit,
                            edgecolor='#ffffff30', linewidth=0.3)
        # Elastic axis line
        ea_x = 0.33 * chord
        ea_z = [w_s[fi][j] for j in range(len(y_nodes))]
        ax3d_s.plot([ea_x]*len(y_nodes), y_nodes, ea_z,
                    'w-', linewidth=2, alpha=0.8, label='Elastic Axis')
        style_3d(ax3d_s, f'V = {V_stable} m/s  STABLE', '#4fc3f7')

        # Flutter wing
        X, Y, Z = make_wing_surface(y_nodes, w_f[fi], th_f[fi], chord)
        ax3d_f.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.85,
                            vmin=-z_limit, vmax=z_limit,
                            edgecolor='#ffffff30', linewidth=0.3)
        ea_z = [w_f[fi][j] for j in range(len(y_nodes))]
        ax3d_f.plot([ea_x]*len(y_nodes), y_nodes, ea_z,
                    'w-', linewidth=2, alpha=0.8, label='Elastic Axis')
        style_3d(ax3d_f, f'V = {V_flutter} m/s  FLUTTER', '#ff5252')

        # Time histories
        ax_ts.cla()
        ax_tf.cla()

        t = t_all[:fi+1]
        tip_s = w_s[:fi+1, -1]
        tip_f = w_f[:fi+1, -1]

        ax_ts.plot(t, tip_s, color='#4fc3f7', linewidth=1.5)
        ax_ts.fill_between(t, 0, tip_s, alpha=0.15, color='#4fc3f7')
        style_ts(ax_ts, f'Tip Heave — V={V_stable} m/s (stable)', '#4fc3f7')

        ax_tf.plot(t, tip_f, color='#ff5252', linewidth=1.5)
        ax_tf.fill_between(t, 0, tip_f, alpha=0.15, color='#ff5252')
        style_ts(ax_tf, f'Tip Heave — V={V_flutter} m/s (flutter)', '#ff5252')

        # Time indicator
        current_t = fi * dt
        fig.suptitle(
            f'Goland Wing Aeroelastic Flutter  |  t = {current_t:.3f}s  |  '
            f'V_flutter = 140.2 m/s (ref 137 m/s)',
            color='white', fontsize=12, fontweight='bold', y=0.98
        )

        return []

    anim = FuncAnimation(
        fig, update,
        frames=len(frame_indices),
        interval=33,  # ~30 fps
        blit=False,
    )

    output_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'flutter_animation.gif')
    output_path = os.path.normpath(output_path)

    print(f"Saving animation ({len(frame_indices)} frames) to {output_path}...")
    anim.save(output_path, writer='pillow', fps=24, dpi=80,
              savefig_kwargs={'facecolor': '#1a1a2e'})
    print(f"Done! Saved to {output_path}")
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"File size: {size_mb:.1f} MB")

    plt.close(fig)


if __name__ == '__main__':
    ps.set_up_logging(level="Warning")
    create_animation()
