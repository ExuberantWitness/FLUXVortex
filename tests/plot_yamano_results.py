"""Generate Yamano-comparison visualizations from saved simulation data.

Four plot types matching Yamano et al. publications:
  1. Streamlines around flapping sheets
  2. Flow velocity distributions (slice contours)
  3. Wake behind sheets (VPM particle field)
  4. Snapshot of flapping sheets (ANCF surface deformation)

Usage:
  python plot_yamano_results.py <data_dir>
  or imported: generate_all_plots(data_dir)
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp

from fluxvortex.kernel import velocity_from_particles


# ─── Global plot settings (matching Yamano: Times New Roman, jet colormap) ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'figure.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'image.cmap': 'jet',
})


def load_run_data(data_dir):
    """Load all saved simulation data."""
    data = {}

    # Tip history
    tip_file = os.path.join(data_dir, 'tip_history.npz')
    if os.path.exists(tip_file):
        d = np.load(tip_file, allow_pickle=True)
        data['tip_w'] = d['tip_w']
        data['tip_theta'] = d['tip_theta']
        data['force'] = d['force']
        data['dt_uvlm'] = float(d['dt_uvlm'])

    # Final surface
    surf_file = os.path.join(data_dir, 'final_surface.npz')
    if os.path.exists(surf_file):
        d = np.load(surf_file, allow_pickle=True)
        data['nodes'] = d['nodes']
        data['quads'] = d['quads']

    # VPM final state
    vpm_file = os.path.join(data_dir, 'vpm_final.npz')
    if os.path.exists(vpm_file):
        d = np.load(vpm_file, allow_pickle=True)
        data['vpm_positions'] = d['positions']
        data['vpm_gamma'] = d['gamma']
        data['vpm_sigma'] = d['sigma']
        data['vpm_np'] = int(d['np'])

    # Run info
    info_file = os.path.join(data_dir, 'run_info.json')
    if os.path.exists(info_file):
        with open(info_file) as f:
            data['info'] = json.load(f)

    # Snapshots
    snap_file = os.path.join(data_dir, 'snapshots.npz')
    if os.path.exists(snap_file):
        d = np.load(snap_file, allow_pickle=True)
        data['snapshots'] = dict(zip(d['steps'], d['snapshots']))

    return data


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Streamlines around flapping sheet
# ══════════════════════════════════════════════════════════════════════════

def compute_streamlines(vpm_positions, vpm_gamma, vpm_sigma,
                        seed_points, V_inf_vec, z_plane=0.0,
                        max_t=10.0, dt_stream=0.01, direction='both'):
    """Compute 3D streamlines through velocity field (freestream + VPM particles).

    Uses scipy.integrate.solve_ivp with RK45 for accurate integration.

    Parameters
    ----------
    vpm_positions : (N, 3) array — VPM particle positions
    vpm_gamma : (N, 3) array — vector circulation
    vpm_sigma : (N,) array — core sizes
    seed_points : (M, 3) array — streamline starting positions
    V_inf_vec : (3,) array — freestream velocity vector
    z_plane : float — z-coordinate to constrain to (if > -inf)
    max_t : float — integration time
    dt_stream : float — output time step
    direction : str — 'forward', 'backward', or 'both'

    Returns
    -------
    lines : list of (T, 3) arrays
    """

    def velocity_field(t, y):
        x = y.reshape(-1, 3)
        v = np.broadcast_to(V_inf_vec, x.shape).copy()
        if len(vpm_positions) > 0:
            v += velocity_from_particles(x, vpm_positions, vpm_gamma, vpm_sigma)
        return v.ravel()

    lines = []
    for seed in seed_points:
        segments = []

        if direction in ('forward', 'both'):
            sol = solve_ivp(velocity_field, [0, max_t], seed,
                            method='RK45', t_eval=np.arange(0, max_t, dt_stream),
                            rtol=1e-6, atol=1e-8, max_step=0.05)
            if sol.y.shape[1] > 1:
                segments.append(sol.y.T)

        if direction in ('backward', 'both'):
            sol = solve_ivp(velocity_field, [0, -max_t], seed,
                            method='RK45', t_eval=np.arange(0, -max_t, -dt_stream),
                            rtol=1e-6, atol=1e-8, max_step=0.05)
            if sol.y.shape[1] > 1:
                segments.insert(0, sol.y.T[::-1])

        if segments:
            full = np.vstack(segments) if len(segments) > 1 else segments[0]
            lines.append(full)

    return lines


def plot_streamlines(data, output_dir):
    """Figure 1: Streamlines around the flapping sheet (mid-span slice)."""
    print("  Plotting: Streamlines around flapping sheet...")

    nodes = data.get('nodes')
    quads = data.get('quads')
    vpm_pos = data.get('vpm_positions')
    vpm_gam = data.get('vpm_gamma')
    vpm_sig = data.get('vpm_sigma')
    info = data.get('info', {})

    if nodes is None:
        print("    Skipped: no surface data")
        return

    # Get freestream vector from config
    cfg = info.get('config', {})
    V_inf = cfg.get('V_inf', 10.0)
    alpha = np.radians(cfg.get('alpha', 2.0))
    V_inf_vec = np.array([V_inf * np.cos(alpha), 0.0, -V_inf * np.sin(alpha)])

    # Mid-span slice plane (Yamano: X*=[-1,5], Z*=[-1.5,1.5] at y=Width/2)
    y_mid = (nodes[:, 1].min() + nodes[:, 1].max()) / 2
    L = nodes[:, 0].max()

    # Seed points for streamlines (Yamano: 50 seeds at X* = -1 + eps)
    n_seeds = 50
    seed_z = np.linspace(-1.5 * L, 1.5 * L, n_seeds)
    seed_points = np.column_stack([
        np.full(n_seeds, -L + 1e-6),  # just upstream of LE
        np.full(n_seeds, y_mid),
        seed_z,
    ])

    # Compute streamlines
    if vpm_pos is not None and len(vpm_pos) > 0:
        lines = compute_streamlines(
            vpm_pos, vpm_gam, vpm_sig,
            seed_points, V_inf_vec, max_t=5.0, dt_stream=0.02,
        )
    else:
        # Freestream only
        lines = []
        for seed in seed_points:
            t_vals = np.linspace(0, 5.0, 100)
            pts = seed + V_inf_vec * t_vals[:, None]
            lines.append(pts)

    # ── Plot (Yamano style: blue streamlines, red quiver, sheet outline) ──
    fig, ax = plt.subplots(figsize=(10, 6))

    # Wing cross-section at mid-span (blue line, thick)
    mid_nodes = nodes[np.abs(nodes[:, 1] - y_mid) < 0.05]
    if len(mid_nodes) > 0:
        sort_idx = np.argsort(mid_nodes[:, 0])
        mid_sorted = mid_nodes[sort_idx]
        ax.plot(mid_sorted[:, 0] / L, mid_sorted[:, 2] / L,
                'b-', linewidth=4, label='Sheet section')

    # Streamlines (blue)
    for line in lines:
        ax.plot(line[:, 0] / L, line[:, 2] / L,
                linewidth=1.0, color='blue', alpha=0.7)

    # Velocity quiver on grid (red, subsampled)
    n_x, n_z = 120, 80
    x_grid = np.linspace(-L, 5 * L, n_x)
    z_grid = np.linspace(-1.5 * L, 1.5 * L, n_z)
    Xq, Zq = np.meshgrid(x_grid, z_grid)
    pts = np.column_stack([Xq.ravel(), np.full(Xq.size, y_mid), Zq.ravel()])
    Vq = compute_velocity_on_grid(pts, vpm_pos, vpm_gam, vpm_sig, V_inf_vec)
    skip = 3
    ax.quiver(Xq[::skip, ::skip] / L, Zq[::skip, ::skip] / L,
              Vq[:, 0].reshape(Xq.shape)[::skip, ::skip] / V_inf,
              Vq[:, 2].reshape(Xq.shape)[::skip, ::skip] / V_inf,
              scale=20, width=0.003, color='red', alpha=0.5)

    ax.set_xlabel('x*')
    ax.set_ylabel('z*')
    ax.set_title('Velocity field (mid-span, y* = Width/2)')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    filepath = os.path.join(output_dir, 'fig1_streamlines.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"    -> {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Flow velocity distributions (slice contours)
# ══════════════════════════════════════════════════════════════════════════

def compute_velocity_on_grid(points, vpm_positions, vpm_gamma, vpm_sigma, V_inf_vec):
    """Compute velocity (freestream + VPM) at arbitrary points."""
    V = np.broadcast_to(V_inf_vec, points.shape).copy()
    if vpm_positions is not None and len(vpm_positions) > 0:
        V += velocity_from_particles(points, vpm_positions, vpm_gamma, vpm_sigma)
    return V


def plot_velocity_distributions(data, output_dir):
    """Figure 2: Flow velocity distributions on slice planes."""
    print("  Plotting: Flow velocity distributions...")

    nodes = data.get('nodes')
    vpm_pos = data.get('vpm_positions')
    vpm_gam = data.get('vpm_gamma')
    vpm_sig = data.get('vpm_sigma')
    info = data.get('info', {})

    if nodes is None:
        print("    Skipped: no surface data")
        return

    cfg = info.get('config', {})
    V_inf = cfg.get('V_inf', 10.0)
    alpha = np.radians(cfg.get('alpha', 2.0))
    V_inf_vec = np.array([V_inf * np.cos(alpha), 0.0, -V_inf * np.sin(alpha)])

    # Two slice planes:
    # (a) x-z plane at mid-span
    # (b) x-y plane at z=0 (plan view)
    y_mid = (nodes[:, 1].min() + nodes[:, 1].max()) / 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── (a) Mid-span x-z slice (Yamano: 120×80 grid, jet colormap, 40 levels) ──
    ax = axes[0]
    n_x, n_z = 120, 80  # EXACT Yamano grid
    L = nodes[:, 0].max()  # chord length
    x_vals = np.linspace(-L, 5 * L, n_x)  # X*: [-1, 5]
    z_vals = np.linspace(-1.5 * L, 1.5 * L, n_z)  # Z*: [-1.5, 1.5]
    X, Z = np.meshgrid(x_vals, z_vals)
    Y = np.full(X.size, y_mid)
    X, Z = np.meshgrid(x_vals, z_vals)
    Y = np.full(X.size, y_mid)
    points = np.column_stack([X.ravel(), Y, Z.ravel()])

    V = compute_velocity_on_grid(points, vpm_pos, vpm_gam, vpm_sig, V_inf_vec)
    V_mag = np.sqrt(V[:, 0]**2 + V[:, 2]**2).reshape(X.shape)
    V_u = V[:, 0].reshape(X.shape)
    V_w = V[:, 2].reshape(X.shape)

    # Nondimensional velocity
    V_mag_nondim = V_mag / V_inf
    V_u_nondim = V_u / V_inf
    V_w_nondim = V_w / V_inf

    # Velocity magnitude contour (Yamano: 40 levels, jet colormap)
    levels = 40
    cf = ax.contourf(X / L, Z / L, V_mag_nondim, levels=levels,
                      cmap='jet', extend='both')
    ax.quiver(X[::3, ::3] / L, Z[::3, ::3] / L,
              V_u_nondim[::3, ::3], V_w_nondim[::3, ::3],
              scale=20, width=0.003, color='red', alpha=0.6)

    # Wing outline (mid-span section)
    mid_mask = np.abs(nodes[:, 1] - y_mid) < 0.05
    if mid_mask.any():
        mid_n = nodes[mid_mask]
        sort_idx = np.argsort(mid_n[:, 0])
        ax.plot(mid_n[sort_idx, 0] / L, mid_n[sort_idx, 2] / L,
                'b-', linewidth=4, label='Sheet')

    ax.set_xlabel('x*')
    ax.set_ylabel('z*')
    ax.set_title('|u*| distribution (mid-span)')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='|u*|', shrink=0.8)

    # ── (b) x-y plan view ──
    ax = axes[1]
    n_x, n_y = 80, 60
    x_vals = np.linspace(-0.5 * L, 5 * L, n_x)
    y_vals = np.linspace(-0.5 * L, 1.5 * L, n_y)
    X, Y = np.meshgrid(x_vals, y_vals)
    z_plane = 0.0
    points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_plane)])

    V = compute_velocity_on_grid(points, vpm_pos, vpm_gam, vpm_sig, V_inf_vec)
    V_mag = np.sqrt(V[:, 0]**2 + V[:, 1]**2).reshape(X.shape)

    cf = ax.contourf(X / L, Y / L, V_mag / V_inf, levels=40,
                      cmap='jet', extend='both')

    # Wing outline (plan view)
    wing_rect = plt.Rectangle(
        (nodes[:, 0].min() / L, nodes[:, 1].min() / L),
        nodes[:, 0].ptp() / L, nodes[:, 1].ptp() / L,
        fill=True, color='#333333', alpha=0.5,
    )
    ax.add_patch(wing_rect)

    ax.set_xlabel('x*')
    ax.set_ylabel('y*')
    ax.set_title('|u*| (plan view, z*=0)')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='|u*|', shrink=0.8)

    plt.suptitle('Flow velocity distributions', fontsize=15, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'fig2_velocity_distributions.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"    -> {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Wake behind sheets (VPM particle field)
# ══════════════════════════════════════════════════════════════════════════

def plot_wake_particles(data, output_dir):
    """Figure 3: Wake visualization — VPM particle field + sheet."""
    print("  Plotting: Wake behind sheets (VPM particles)...")

    nodes = data.get('nodes')
    quads = data.get('quads')
    vpm_pos = data.get('vpm_positions')
    vpm_gam = data.get('vpm_gamma')
    vpm_np = data.get('vpm_np', 0)

    if nodes is None:
        print("    Skipped: no surface data")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    L = nodes[:, 0].max()
    # ── ANCF surface mesh (Yamano: red semi-transparent patch) ──
    if quads is not None:
        for q in quads:
            verts = nodes[q]
            poly = Poly3DCollection([verts], alpha=0.5,
                                    facecolor='red', edgecolor='#8b0000',
                                    linewidth=0.2)
            ax.add_collection3d(poly)

    # ── VPM particles, colored by |Γ| ──
    if vpm_pos is not None and len(vpm_pos) > 0 and vpm_np > 0:
        pos = vpm_pos[:vpm_np]
        gam = vpm_gam[:vpm_np]
        gam_mag = np.linalg.norm(gam, axis=1)

        max_display = 8000
        if len(pos) > max_display:
            idx = np.random.choice(len(pos), max_display, replace=False)
            pos = pos[idx]
            gam_mag = gam_mag[idx]

        gam_norm = plt.Normalize(gam_mag.min(), gam_mag.max())
        colors = plt.cm.jet(gam_norm(gam_mag))
        sizes = 2 + 10 * gam_norm(gam_mag)

        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c=colors, s=sizes, alpha=0.4, depthshade=True)

    # ── View settings (Yamano: X*=[-1,5], Y*=[-1,2], Z*=[-1.5,1.5], view [1,-2,1]) ──
    ax.set_xlim(-L, 5 * L)
    ax.set_ylim(-L, 2 * L)
    ax.set_zlim(-1.5 * L, 1.5 * L)
    ax.set_xlabel('x*')
    ax.set_ylabel('y*')
    ax.set_zlabel('z*')
    ax.set_title('Wake behind flapping sheet',
                 fontsize=15, fontweight='bold')
    ax.view_init(elev=20, azim=-60)

    filepath = os.path.join(output_dir, 'fig3_wake_particles.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"    -> {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Snapshot of flapping sheet (ANCF surface deformation)
# ══════════════════════════════════════════════════════════════════════════

def plot_sheet_deformation(data, output_dir):
    """Figure 4: ANCF surface deformation colored by z-displacement."""
    print("  Plotting: Sheet deformation snapshot...")

    nodes = data.get('nodes')
    quads = data.get('quads')
    tip_w_history = data.get('tip_w')
    tip_theta_history = data.get('tip_theta')
    dt_uvlm = data.get('dt_uvlm', 0.001)

    if nodes is None:
        print("    Skipped: no surface data")
        return

    # Reference flat sheet (undeformed positions)
    ref_nodes = nodes.copy()
    ref_nodes[:, 2] = 0.0

    fig = plt.figure(figsize=(16, 7))

    L = nodes[:, 0].max()
    # ── (a) 3D deformed surface (Yamano: red semi-transparent, view [1,2,1]) ──
    ax = fig.add_subplot(121, projection='3d')

    if quads is not None:
        for q in quads:
            verts = nodes[q]
            poly = Poly3DCollection([verts], alpha=0.5,
                                     facecolor='red', edgecolor='#8b0000',
                                     linewidth=0.2)
            ax.add_collection3d(poly)

    # Reference outline
    if quads is not None:
        for i, q in enumerate(quads):
            if i % 5 == 0:
                verts = ref_nodes[q]
                poly = Poly3DCollection([verts], alpha=0.15,
                                         facecolor='#cccccc', edgecolor='#999999',
                                         linewidth=0.2, linestyle='--')
                ax.add_collection3d(poly)

    ax.set_xlabel('x*')
    ax.set_ylabel('y*')
    ax.set_zlabel('z*')
    ax.set_title('Deformed ANCF sheet')
    ax.set_xlim(-0.5 * L, 1.5 * L)
    ax.set_ylim(-1.5 * L, 2 * L)
    z_max_disp = max(abs(nodes[:, 2]).max(), 0.1 * L)
    ax.set_zlim(-0.5 * L, 0.5 * L)
    ax.view_init(elev=25, azim=-55)

    # ── (b) Tip displacement time history (Yamano: nondimensional t*, z*) ──
    ax = fig.add_subplot(122)

    if tip_w_history is not None and len(tip_w_history) > 1:
        info = data.get('info', {})
        cfg = info.get('config', {})
        V_inf = cfg.get('V_inf', 10.0)
        L_ref = cfg.get('Length', nodes[:, 0].max()) if nodes is not None else 1.0

        t_dim = np.arange(len(tip_w_history)) * dt_uvlm
        t_nd = t_dim * V_inf / L_ref  # nondimensional t* = t * U_inf / L
        w_nd = np.array(tip_w_history) / L_ref  # nondimensional z* = z / L

        ax.plot(t_nd, w_nd, 'b-', linewidth=0.8, label='Tip w*')

        abs_w = np.abs(w_nd)
        if len(abs_w) > 20:
            from scipy.ndimage import maximum_filter1d
            window = max(len(abs_w) // 10, 5)
            envelope = maximum_filter1d(abs_w, window)
            ax.plot(t_nd, envelope, 'r--', linewidth=1.2, alpha=0.7,
                    label='Envelope')
            ax.plot(t_nd, -envelope, 'r--', linewidth=1.2, alpha=0.7)

        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xlabel('t*')
        ax.set_ylabel('z* (tip)')
        ax.set_title('Tip heave response')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Flutter status
        if len(w_nd) > 30:
            mid = len(w_nd) // 2
            first_amp = np.max(np.abs(w_nd[:mid]))
            second_amp = np.max(np.abs(w_nd[mid:]))
            ratio = second_amp / max(first_amp, 1e-15)
            status = 'FLUTTER' if ratio > 2.0 else \
                     'LCO' if 0.8 < ratio < 1.25 else \
                     'STABLE' if ratio < 0.8 else 'TRANSIENT'
            color = 'red' if ratio > 2.0 else 'green' if ratio < 0.8 else 'orange'
            ax.text(0.98, 0.95, status, transform=ax.transAxes,
                    ha='right', va='top', fontsize=14, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.suptitle('Snapshot of flapping sheet', fontsize=15, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'fig4_sheet_deformation.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"    -> {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Multi-snapshot comparison (period sequence)
# ══════════════════════════════════════════════════════════════════════════

def plot_multi_snapshot(data, output_dir):
    """Figure 5: Sequence of sheet deformations through a flapping period."""
    print("  Plotting: Multi-snapshot comparison...")

    snapshots = data.get('snapshots', {})
    if not snapshots:
        print("    Skipped: no snapshot data")
        return

    n_snaps = len(snapshots)
    if n_snaps < 2:
        print("    Skipped: need at least 2 snapshots")
        return

    # Pick evenly-spaced snapshots through the simulation
    step_keys = sorted(snapshots.keys())
    n_display = min(6, len(step_keys))
    indices = np.linspace(0, len(step_keys) - 1, n_display, dtype=int)
    selected_steps = [step_keys[i] for i in indices]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                              subplot_kw={'projection': '3d'})
    axes = axes.ravel()

    for ax_idx, step in enumerate(selected_steps):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        snap = snapshots[step]
        nodes = snap.get('nodes')
        quads = snap.get('quads')

        if nodes is None:
            continue

        # Deformed surface
        z_disp = nodes[:, 2]
        z_max = max(abs(z_disp).max(), 0.01)
        z_norm = plt.Normalize(-z_max, z_max)

        for q in quads:
            verts = nodes[q]
            z_mean = z_disp[q].mean()
            color = plt.cm.RdBu_r(z_norm(z_mean))
            poly = Poly3DCollection([verts], alpha=0.8,
                                     facecolor=color, edgecolor='#555555',
                                     linewidth=0.2)
            ax.add_collection3d(poly)

        ax.set_xlim(nodes[:, 0].min() - 0.1, nodes[:, 0].max() + 0.1)
        ax.set_ylim(nodes[:, 1].min() - 0.1, nodes[:, 1].max() + 0.1)
        ax.set_zlim(-z_max * 2, z_max * 2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'Step {step}, t={snap["time"]:.2f}s\n'
                     f'tip_w={snap.get("tip_w", 0)*1000:.2f}mm',
                     fontsize=9)
        ax.view_init(elev=30, azim=-45 - ax_idx * 15)

    # Hide unused subplots
    for i in range(n_display, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Sheet Deformation Sequence', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'fig5_deformation_sequence.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"    -> {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Summary dashboard
# ══════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(data, output_dir):
    """Figure 6: Summary dashboard with key metrics."""
    print("  Plotting: Summary dashboard...")

    tip_w = data.get('tip_w')
    tip_theta = data.get('tip_theta')
    force = data.get('force')
    dt_uvlm = data.get('dt_uvlm', 0.001)
    info = data.get('info', {})
    vpm_np = data.get('vpm_np', 0)
    vpm_pos = data.get('vpm_positions')

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # ── (a) Tip heave time history ──
    ax = fig.add_subplot(gs[0, :])
    if tip_w is not None and len(tip_w) > 1:
        t = np.arange(len(tip_w)) * dt_uvlm
        ax.plot(t, tip_w * 1000, 'b-', linewidth=0.6, alpha=0.8, label='Tip w')
        ax.set_ylabel('Tip heave [mm]')
        ax.set_title('Tip Displacement Time History', fontweight='bold')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate max amplitude
        max_idx = np.argmax(np.abs(tip_w))
        ax.annotate(f'Max: {tip_w[max_idx]*1000:.2f}mm',
                    xy=(t[max_idx], tip_w[max_idx]*1000),
                    xytext=(t[max_idx] + 0.1*max(t), tip_w[max_idx]*1000*1.1),
                    fontsize=8, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))

    # ── (b) Pitch angle history ──
    ax = fig.add_subplot(gs[1, 0])
    if tip_theta is not None and len(tip_theta) > 1:
        t = np.arange(len(tip_theta)) * dt_uvlm
        ax.plot(t, np.degrees(tip_theta), 'r-', linewidth=0.6, alpha=0.8)
        ax.set_ylabel('Tip pitch [deg]')
        ax.set_xlabel('Time [s]')
        ax.set_title('Tip Pitch Angle', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # ── (c) FFT spectrum ──
    ax = fig.add_subplot(gs[1, 1])
    if tip_w is not None and len(tip_w) > 30:
        half = len(tip_w) // 2
        signal = tip_w[half:] - np.mean(tip_w[half:])
        from scipy.fft import rfft, rfftfreq
        n = len(signal)
        freqs = rfftfreq(n, dt_uvlm)
        spec = np.abs(rfft(signal * np.hanning(n)))
        spec_db = 20 * np.log10(np.maximum(spec / n, 1e-15))

        ax.semilogx(freqs[1:], spec_db[1:], 'b-', linewidth=0.8)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude [dB]')
        ax.set_title('Tip Displacement Spectrum', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark dominant frequency
        peak_idx = np.argmax(spec[1:]) + 1
        ax.axvline(freqs[peak_idx], color='red', linestyle='--', linewidth=1,
                   label=f'{freqs[peak_idx]:.2f} Hz')
        ax.legend(fontsize=8)

    # ── (d) Aerodynamic force history ──
    ax = fig.add_subplot(gs[2, 0])
    if force is not None and len(force) > 1:
        t = np.arange(len(force)) * dt_uvlm
        ax.plot(t, force, 'g-', linewidth=0.6, alpha=0.8)
        ax.set_ylabel('|F_aero| [N]')
        ax.set_xlabel('Time [s]')
        ax.set_title('Total Aerodynamic Force', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # ── (e) Parameters & stats table ──
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')

    # Build stats text
    stats = []
    cfg = info.get('config', {})
    params = info.get('params', {})
    results = info.get('results', {})

    stats.append(['Parameter', 'Value'])
    stats.append(['U*', f"{cfg.get('U_star', 'N/A'):.1f}"])
    stats.append(['M*', f"{cfg.get('M_star', 'N/A'):.2f}"])
    stats.append(['V_inf', f"{cfg.get('V_inf', 'N/A'):.1f} m/s"])
    stats.append(['α', f"{cfg.get('alpha', 'N/A'):.1f}°"])
    stats.append(['E', f"{params.get('E', 0):.2e} Pa"])
    stats.append(['ρ_struct', f"{params.get('rho', 0):.1f} kg/m³"])
    stats.append(['f₁ (beam)', f"{params.get('freq1_beam', 0):.3f} Hz"])
    stats.append(['', ''])
    stats.append(['Max |w|', f"{results.get('max_abs_w', 0)*1000:.2f} mm"])
    stats.append(['σ_w', f"{results.get('sigma_w', 0):+.4f} 1/s"])
    stats.append(['f_dominant', f"{results.get('dominant_freq', 0):.3f} Hz"])
    stats.append(['VPM particles', f"{vpm_np:,}"])

    table = ax.table(cellText=stats, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for i in range(2):
        table[0, i].set_text_props(fontweight='bold')
        table[0, i].set_facecolor('#2c5f8a')
        table[0, i].set_text_props(color='white')

    for i in range(1, len(stats)):
        if i % 2 == 0:
            table[i, 0].set_facecolor('#f0f0f0')
            table[i, 1].set_facecolor('#f0f0f0')

    ax.set_title('Simulation Parameters & Results', fontweight='bold')

    plt.suptitle('Yamano Comparison — Summary Dashboard',
                 fontsize=15, fontweight='bold')
    filepath = os.path.join(output_dir, 'fig6_summary_dashboard.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"    -> {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# Master function
# ══════════════════════════════════════════════════════════════════════════

def generate_all_plots(data_dir):
    """Generate all comparison plots from saved simulation data.

    Parameters
    ----------
    data_dir : str
        Path to directory containing saved .npz and .json files.
    """
    print(f"\nLoading data from: {data_dir}")
    data = load_run_data(data_dir)

    if not data:
        print("ERROR: No data found.")
        return

    # Print data summary
    info = data.get('info', {})
    if info:
        results = info.get('results', {})
        print(f"  Steps: {results.get('n_steps', '?')}")
        print(f"  Max tip |w|: {results.get('max_abs_w', 0)*1000:.2f} mm")
        print(f"  Growth rate σ: {results.get('sigma_w', 0):+.4f} 1/s")
        print(f"  VPM particles: {data.get('vpm_np', 0):,}")

    print("\nGenerating plots...")

    plot_streamlines(data, data_dir)
    plot_velocity_distributions(data, data_dir)
    plot_wake_particles(data, data_dir)
    plot_sheet_deformation(data, data_dir)
    plot_multi_snapshot(data, data_dir)
    plot_summary_dashboard(data, data_dir)

    print(f"\nAll plots saved to: {data_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_yamano_results.py <data_dir>")
        sys.exit(1)
    generate_all_plots(sys.argv[1])
