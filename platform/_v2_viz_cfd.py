"""CFD-STYLE interactive viz (Plotly HTML): wing + wake/LEV vortex rings + a sampled VELOCITY-MAGNITUDE
field (Biot-Savart from all rings on the GPU), shown as (a) a mid-span x-z SLICE colored by |V| (XFLOW-like)
and (b) sparse 3D cones (induced-flow vectors). Mouse rotate/zoom + play/pause + slider. Self-contained.

  python _v2_viz_cfd.py [nc] [spc]      # default nc=8 spc=240
"""
import sys, os, numpy as np
import warp as wp
import plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v2_robo as R
from diff_uvlm_unsteady_gpu import ring_vel, V3, DTYPE
from grid_indep import MODEL, COND

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SPC = int(sys.argv[2]) if len(sys.argv) > 2 else 240


@wp.kernel
def field_vel_kernel(pts: wp.array(dtype=V3), rings: wp.array(dtype=V3, ndim=2), g: wp.array(dtype=DTYPE),
                     nr: int, Vinf: V3, vel: wp.array(dtype=V3)):
    i = wp.tid()
    P = pts[i]; v = Vinf
    for k in range(nr):
        v = v + g[k] * ring_vel(P, rings[k, 0], rings[k, 1], rings[k, 2], rings[k, 3])
    vel[i] = v


def sample_field(pts_np, rings_np, g_np, Vinf):
    dev = 'cuda'
    pts = wp.array(pts_np.astype(np.float64), dtype=V3, device=dev)
    rings = wp.array(rings_np.astype(np.float64), dtype=V3, device=dev)
    g = wp.array(g_np.astype(np.float64), dtype=DTYPE, device=dev)
    vel = wp.zeros(len(pts_np), dtype=V3, device=dev)
    Vw = V3(float(Vinf[0]), float(Vinf[1]), float(Vinf[2]))
    wp.launch(field_vel_kernel, dim=len(pts_np), inputs=[pts, rings, g, len(rings_np), Vw], outputs=[vel], device=dev)
    return vel.numpy()


def all_rings(f):
    """combine bound(gamma) + TEV(wg) + LEV(lev_g) rings into one (N,4,3)+(N,) set for field sampling."""
    Rs = [f['bound'].reshape(-1, 4, 3)]; Gs = [f['gam'].reshape(-1)]
    if len(f['wr']): Rs.append(f['wr']); Gs.append(f['wg'])
    if len(f['lev_rings']): Rs.append(f['lev_rings']); Gs.append(f['lev_g'])
    return np.concatenate(Rs, 0), np.concatenate(Gs, 0)


wp.init()
frames_data = []
res = R.gpu_run_twist(**{**MODEL, 'n_cycle': 2}, nc=NC, ns=16, steps_per_cycle=SPC, wake_rows=SPC,
                      **COND, frames_out=frames_data, frame_skip=max(1, SPC // 24))
print(f"recorded {len(frames_data)} frames; L_wind={res['L_wind']:.2f}N", flush=True)
U = COND['U']; Vinf = np.array([U, 0.0, 0.0])

# mid-span x-z slice grid (y = mid of the wing span)
allb = np.concatenate([f['bound'].reshape(-1, 3) for f in frames_data])
ymid = 0.40
xs = np.linspace(allb[:, 0].min() - 0.15, allb[:, 0].max() + 0.6, 90)
zs = np.linspace(allb[:, 2].min() - 0.35, allb[:, 2].max() + 0.35, 70)
XX, ZZ = np.meshgrid(xs, zs)
slice_pts = np.stack([XX.ravel(), np.full(XX.size, ymid), ZZ.ravel()], 1)

# precompute |V| on the slice per frame
speed_frames = []
for f in frames_data:
    rr, gg = all_rings(f)
    v = sample_field(slice_pts, rr, gg, Vinf)
    spd = np.linalg.norm(v, axis=1).reshape(XX.shape)
    speed_frames.append(spd)
smax = float(np.percentile(np.concatenate([s.ravel() for s in speed_frames]), 99))
print(f"slice {XX.shape}, |V| cmax≈{smax:.1f} m/s", flush=True)


def wing_quads(bound):
    V = bound.reshape(-1, 4, 3); npan = V.shape[0]
    verts = V.reshape(-1, 3); i0 = np.arange(npan) * 4
    I = np.concatenate([i0, i0]); J = np.concatenate([i0 + 1, i0 + 2]); K = np.concatenate([i0 + 2, i0 + 3])
    return verts, I, J, K

def ring_lines(rings, g, sub=1):
    def pack(idx):
        xs, ys, zs = [], [], []
        for k in idx:
            r = rings[k]
            for a in [0, 1, 2, 3, 0]:
                xs.append(r[a, 0]); ys.append(r[a, 1]); zs.append(r[a, 2])
            xs += [None]; ys += [None]; zs += [None]
        return xs, ys, zs
    return pack([k for k in range(0, len(rings), sub) if g[k] > 0]), pack([k for k in range(0, len(rings), sub) if g[k] <= 0])


def frame_traces(f, spd):
    v, I, J, Kf = wing_quads(f['bound'])
    wsub = max(1, len(f['wr']) // 160)
    (lpx, lpy, lpz), (lnx, lny, lnz) = ring_lines(f['lev_rings'], f['lev_g']) if len(f['lev_rings']) else (([],[],[]),([],[],[]))
    return [
        go.Surface(x=XX, y=np.full(XX.shape, ymid), z=ZZ, surfacecolor=spd, colorscale='Turbo',
                   cmin=0, cmax=smax, opacity=0.92, showscale=True, colorbar=dict(title='|V| m/s', x=0.0, len=0.6),
                   name='|V| midspan'),
        go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=I, j=J, k=Kf, color='black', opacity=1.0, flatshading=True, name='wing'),
        go.Scatter3d(x=lpx, y=lpy, z=lpz, mode='lines', line=dict(color='rgba(0,200,0,1)', width=5), name='LEV +Γ'),
        go.Scatter3d(x=lnx, y=lny, z=lnz, mode='lines', line=dict(color='rgba(230,0,230,1)', width=5), name='LEV -Γ'),
    ]

f0 = frame_traces(frames_data[0], speed_frames[0])
pframes = [go.Frame(data=frame_traces(f, s), name=str(k)) for k, (f, s) in enumerate(zip(frames_data, speed_frames))]
xr = [xs.min(), xs.max()]; yr = [-0.05, 0.85]; zr = [zs.min(), zs.max()]
fig = go.Figure(data=f0, frames=pframes)
fig.update_layout(
    title=f"FLUXVortex CFD-view — mid-span |V| slice + LEV (nc={NC}, spc={SPC}, 8m/s ±45°flap 22.5°twist) "
          f"| L={res['L_wind']:.1f}N  [drag to rotate]",
    scene=dict(xaxis=dict(range=xr, title='x (flow→)'), yaxis=dict(range=yr, title='y (span)'),
               zaxis=dict(range=zr, title='z'), aspectmode='data', camera=dict(eye=dict(x=0.2, y=-2.2, z=0.6))),
    updatemenus=[dict(type='buttons', showactive=False, x=0.05, y=0.05, xanchor='left',
        buttons=[dict(label='▶ Play', method='animate',
                      args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True, mode='immediate')]),
                 dict(label='⏸ Pause', method='animate',
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])],
    sliders=[dict(active=0, x=0.15, len=0.8, currentvalue=dict(prefix='frame '),
        steps=[dict(method='animate', label=str(k), args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True))])
               for k in range(len(pframes))])])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', f'_v2_cfd_nc{NC}.html')
fig.write_html(out, include_plotlyjs='cdn', auto_play=False)
print(f"saved {out} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True); print("DONE", flush=True)
