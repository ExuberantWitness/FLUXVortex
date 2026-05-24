"""
Animated GIF demo: Hybrid Panel-Particle Wake solver.

Shows: wing panels (gray), near-field wake panels (blue lines),
far-field VPM particles (colored by |Gamma|), and CL time history.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")
from fluxvortex.particles import VortexParticleField
from experiment_hybrid_panel_particle import (
    HybridSolver, make_wing, make_plunge, extract_cl, get_Vvec
)


def collect_frame_data(solver, step):
    """Extract wing panels, wake panels, and particle data at one timestep."""
    wings_data = []
    for airplane in solver.current_airplanes:
        for wing in airplane.wings:
            # Wing panels
            panels = wing.panels
            if panels is None: continue
            panel_verts = []
            for panel in panels.ravel():
                rv = panel.ring_vortex
                if rv is None: continue
                panel_verts.append([
                    rv.Frrvp_GP1_CgP1.copy(),
                    rv.Flrvp_GP1_CgP1.copy(),
                    rv.Blrvp_GP1_CgP1.copy(),
                    rv.Brrvp_GP1_CgP1.copy(),
                ])
            # Wake ring vortices (non-zero strength = near-field)
            wake_lines = []
            wake_fr, wake_old = [], []
            wrv = wing.wake_ring_vortices
            if wrv is not None and wrv.shape[0] > 0:
                for i in range(wrv.shape[0]):
                    for j in range(wrv.shape[1]):
                        rv = wrv[i, j]
                        if rv is None: continue
                        fr, fl = rv.Frrvp_GP1_CgP1, rv.Flrvp_GP1_CgP1
                        bl, br = rv.Blrvp_GP1_CgP1, rv.Brrvp_GP1_CgP1
                        segs = [[fr, fl], [fl, bl], [bl, br], [br, fr]]
                        if abs(rv.strength) > 1e-15:
                            wake_fr.extend(segs)
                        else:
                            wake_old.extend(segs)

            wings_data.append({
                'panels': panel_verts,
                'wake_active': wake_fr,
                'wake_converted': wake_old,
            })

    # VPM particles
    vpm_pos, vpm_gam = None, None
    if solver._vpm.np > 0:
        vpm_pos = solver._vpm.positions.copy()
        vpm_gam = np.linalg.norm(solver._vpm.gammas, axis=1)

    return wings_data, vpm_pos, vpm_gam


class RecordingHybridSolver(HybridSolver):
    """HybridSolver that records frame data at each step."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.frames = []

    def _populate_next_airplanes_wake(self):
        super()._populate_next_airplanes_wake()
        if self._current_step >= 1:
            wings_data, vpm_pos, vpm_gam = collect_frame_data(self, self._current_step)
            self.frames.append({
                'step': self._current_step,
                'wings': wings_data,
                'vpm_pos': vpm_pos,
                'vpm_gam': vpm_gam,
            })


def create_animation(k=0.5, h0c=0.1, V=10.0, chord=1.0, n_keep=10,
                     free_vpm=True, skip=2, output='hybrid_wake_demo.gif'):
    """Run simulation and create animated GIF."""
    omega = 2 * k * V / chord
    period = 2 * np.pi / omega
    h0 = h0c * chord

    print(f"Running simulation: k={k}, N_keep={n_keep}, free_vpm={free_vpm}")
    wing = make_wing(chord)
    _, mv = make_plunge(wing, h0, period, V)
    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)

    solver = RecordingHybridSolver(prob, n_keep=n_keep, free_vpm=free_vpm)
    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    print(f"  Simulation done: {len(solver.frames)} frames recorded")

    # Extract CL
    t_arr, cl_arr = extract_cl(solver, mv)

    # Setup figure
    fig = plt.figure(figsize=(14, 6), facecolor='white')
    ax3d = fig.add_subplot(121, projection='3d')
    ax_cl = fig.add_subplot(122)

    half_span = 5.0
    x_range = [min(-chord * 0.5, -chord * 3), chord * 12]
    y_range = [-half_span * 0.3, half_span * 1.1]
    z_range = [-h0 * 4, h0 * 4]

    # Pre-compute CL plot limits
    cl_valid = cl_arr[len(cl_arr) // 3:]
    if len(cl_valid) > 0:
        cl_min, cl_max = cl_valid.min(), cl_valid.max()
        cl_margin = max(0.05, (cl_max - cl_min) * 0.3)
    else:
        cl_min, cl_max, cl_margin = -0.5, 0.5, 0.1

    frames = solver.frames[::skip]
    print(f"  Animating {len(frames)} frames (skip={skip})")

    def update(frame_idx):
        ax3d.cla()
        ax_cl.cla()

        frame = frames[frame_idx]
        step = frame['step']

        for wing_data in frame['wings']:
            # Draw wing panels (semi-transparent gray)
            if wing_data['panels']:
                polys = Poly3DCollection(wing_data['panels'], alpha=0.6,
                                         facecolor='#4a90d9', edgecolor='#2c5f8a',
                                         linewidth=0.3)
                ax3d.add_collection3d(polys)

            # Draw active (near-field) wake panels (blue lines)
            if wing_data['wake_active']:
                lc = Line3DCollection(wing_data['wake_active'], colors='#6baed6',
                                      linewidths=0.4, alpha=0.7)
                ax3d.add_collection3d(lc)

            # Draw converted (zeroed) wake outlines (very faint)
            if wing_data['wake_converted']:
                lc = Line3DCollection(wing_data['wake_converted'], colors='#d9d9d9',
                                      linewidths=0.15, alpha=0.2)
                ax3d.add_collection3d(lc)

        # Draw VPM particles
        vpm_pos = frame['vpm_pos']
        vpm_gam = frame['vpm_gam']
        if vpm_pos is not None and len(vpm_pos) > 0:
            gam_abs = np.abs(vpm_gam)
            gam_max = gam_abs.max() if gam_abs.max() > 0 else 1.0
            gam_norm = gam_abs / gam_max
            colors = plt.cm.hot(gam_norm)
            # Subsample if too many particles
            if len(vpm_pos) > 3000:
                idx = np.random.choice(len(vpm_pos), 3000, replace=False)
                vpm_pos = vpm_pos[idx]
                colors = colors[idx]
                sizes = 3 + 8 * gam_norm[idx]
            else:
                sizes = 3 + 8 * gam_norm
            ax3d.scatter(vpm_pos[:, 0], vpm_pos[:, 1], vpm_pos[:, 2],
                        c=colors, s=sizes, alpha=0.5, depthshade=True)

        ax3d.set_xlim(x_range)
        ax3d.set_ylim(y_range)
        ax3d.set_zlim(z_range)
        ax3d.set_xlabel('x (chordwise)')
        ax3d.set_ylabel('y (spanwise)')
        ax3d.set_zlabel('z')
        ax3d.set_title(f'Hybrid Panel-Particle Wake\n'
                       f'k={k:.1f}, N_keep={n_keep}, step={step}',
                       fontsize=10)
        ax3d.view_init(elev=22, azim=-60 + frame_idx * 0.3)

        # CL time history
        if len(t_arr) > 0:
            dt = mv.delta_time
            current_t = step * dt
            mask = t_arr <= current_t
            ax_cl.plot(t_arr, -cl_arr, 'b-', linewidth=0.8, alpha=0.3, label='CL')
            if mask.any():
                ax_cl.plot(t_arr[mask], -cl_arr[mask], 'b-', linewidth=1.5, label='CL (current)')
            ax_cl.axvline(current_t, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_cl.set_xlabel('Time (s)')
        ax_cl.set_ylabel('-CL')
        ax_cl.set_title(f'Lift Coefficient (k={k:.1f})', fontsize=10)
        ax_cl.set_ylim(cl_min - cl_margin, cl_max + cl_margin)
        ax_cl.legend(fontsize=8, loc='upper right')
        ax_cl.grid(True, alpha=0.3)

        return []

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=80, blit=False)

    filepath = os.path.join(os.path.dirname(__file__), '..', 'figures', output)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"  Saving to {filepath} ...")
    anim.save(filepath, writer='pillow', fps=12, dpi=80)
    print(f"  Done! {filepath}")
    plt.close(fig)
    return filepath


if __name__ == '__main__':
    print("=" * 60)
    print("FLUXVortex: Animated GIF Demo")
    print("=" * 60)

    # Demo 1: k=0.5, free wake (shows dynamic particle motion)
    create_animation(k=0.5, n_keep=10, free_vpm=True, skip=5,
                     output='hybrid_k05_free.gif')
