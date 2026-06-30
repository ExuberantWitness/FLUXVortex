"""INTERACTIVE 3D (Plotly, self-contained HTML) of the Ansari-LEV flapping run — mouse rotate/zoom/pan
+ play/pause + time slider, CFD-viewer style. Wing = paneled surface (colored by per-panel loading |Γ_net|);
TEV wake + Ansari LEV = vortex rings colored by circulation sign. Opens in any browser / VSCode.

  python _v2_viz_interactive.py [nc] [spc]      # default nc=8 spc=240 (fast dev); pass 12 720 for production
"""
import sys, os, numpy as np
import plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v2_robo as R
from grid_indep import MODEL, COND

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SPC = int(sys.argv[2]) if len(sys.argv) > 2 else 240
NCYC = 2
frames_data = []
res = R.gpu_run_twist(**{**MODEL, 'n_cycle': NCYC}, nc=NC, ns=16, steps_per_cycle=SPC, wake_rows=SPC,
                      **COND, frames_out=frames_data, frame_skip=max(1, SPC // 30))
print(f"recorded {len(frames_data)} frames; L_wind={res['L_wind']:.2f}N T_wind={res['T_wind']:.2f}N", flush=True)

# keep both wings (mirror) for a symmetric look
def mirror(arr):
    m = arr.copy(); m[..., 1] *= -1; return m

def wing_mesh(bound, gam, ns):
    """paneled wing surface from the bound rings (quad per panel); intensity = |Γ_net| loading."""
    V = bound.reshape(-1, 4, 3)
    npan = V.shape[0]; nc = npan // ns
    gm = gam.reshape(nc, ns)
    gnet = gm.copy(); gnet[1:] = gm[1:] - gm[:-1]               # net chordwise loading per panel
    inten = np.abs(gnet).reshape(-1)
    verts = V.reshape(-1, 3)                                    # 4 verts per panel
    i0 = np.arange(npan) * 4
    # two triangles per quad (0,1,2) (0,2,3)
    I = np.concatenate([i0 + 0, i0 + 0]); J = np.concatenate([i0 + 1, i0 + 2]); K = np.concatenate([i0 + 2, i0 + 3])
    inten_v = np.repeat(inten, 4)
    return verts, I, J, K, inten_v

def ring_lines(rings, g, sub=1):
    """closed quad rings -> x,y,z with None separators; split by sign for two-color line traces."""
    def pack(idx):
        xs, ys, zs = [], [], []
        for k in idx:
            r = rings[k]
            for a in [0, 1, 2, 3, 0]:
                xs.append(r[a, 0]); ys.append(r[a, 1]); zs.append(r[a, 2])
            xs.append(None); ys.append(None); zs.append(None)
        return xs, ys, zs
    pos = [k for k in range(0, len(rings), sub) if g[k] > 0]
    neg = [k for k in range(0, len(rings), sub) if g[k] <= 0]
    return pack(pos), pack(neg)

# ---- build per-frame trace data (fixed trace order: wing, wing_mirror, wake+, wake-, lev+, lev-) ----
def frame_traces(f):
    v, I, J, Kf, inten = wing_mesh(f['bound'], f['gam'], f['ns'])
    vm = mirror(v)
    wsub = max(1, len(f['wr']) // 220)                          # cap wake rings drawn (HTML size)
    (wpx, wpy, wpz), (wnx, wny, wnz) = ring_lines(f['wr'], f['wg'], wsub) if len(f['wr']) else (([],[],[]),([],[],[]))
    (lpx, lpy, lpz), (lnx, lny, lnz) = ring_lines(f['lev_rings'], f['lev_g']) if len(f['lev_rings']) else (([],[],[]),([],[],[]))
    return [
        go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=I, j=J, k=Kf, intensity=inten, colorscale='Viridis',
                  cmin=0, cmax=0.6, showscale=False, opacity=1.0, flatshading=True, name='wing'),
        go.Mesh3d(x=vm[:,0], y=vm[:,1], z=vm[:,2], i=I, j=J, k=Kf, intensity=inten, colorscale='Viridis',
                  cmin=0, cmax=0.6, showscale=False, opacity=0.45, flatshading=True, name='wing(mirror)'),
        go.Scatter3d(x=wpx, y=wpy, z=wpz, mode='lines', line=dict(color='rgba(210,60,60,0.45)', width=2), name='TEV +Γ'),
        go.Scatter3d(x=wnx, y=wny, z=wnz, mode='lines', line=dict(color='rgba(60,90,210,0.45)', width=2), name='TEV -Γ'),
        go.Scatter3d(x=lpx, y=lpy, z=lpz, mode='lines', line=dict(color='rgba(0,180,0,1)', width=5), name='LEV +Γ'),
        go.Scatter3d(x=lnx, y=lny, z=lnz, mode='lines', line=dict(color='rgba(200,0,200,1)', width=5), name='LEV -Γ'),
    ]

f0 = frame_traces(frames_data[0])
plotly_frames = [go.Frame(data=frame_traces(f), name=str(k)) for k, f in enumerate(frames_data)]

# axis range (fixed so rotation is stable)
allb = np.concatenate([f['bound'].reshape(-1, 3) for f in frames_data])
xr = [allb[:,0].min()-0.1, allb[:,0].max()+0.4]; yr = [-0.85, 0.85]; zr = [allb[:,2].min()-0.1, allb[:,2].max()+0.1]

fig = go.Figure(data=f0, frames=plotly_frames)
fig.update_layout(
    title=f"FLUXVortex RoboEagle — Ansari-LEV  (nc={NC}, spc={SPC}, 8 m/s, ±45° flap, 22.5° twist) "
          f"| L_wind={res['L_wind']:.1f}N  [drag to rotate]",
    scene=dict(xaxis=dict(range=xr, title='x (chord, flow→)'), yaxis=dict(range=yr, title='y (span)'),
               zaxis=dict(range=zr, title='z'), aspectmode='data',
               camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
    updatemenus=[dict(type='buttons', showactive=False, x=0.05, y=0.05, xanchor='left',
        buttons=[dict(label='▶ Play', method='animate',
                      args=[None, dict(frame=dict(duration=70, redraw=True), fromcurrent=True, mode='immediate')]),
                 dict(label='⏸ Pause', method='animate',
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])],
    sliders=[dict(active=0, x=0.15, len=0.8, currentvalue=dict(prefix='frame '),
        steps=[dict(method='animate', label=str(k),
                    args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True))])
               for k in range(len(plotly_frames))])])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', f'_v2_interactive_nc{NC}.html')
fig.write_html(out, include_plotlyjs='cdn', auto_play=False)
print(f"saved {out} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True); print("DONE", flush=True)
