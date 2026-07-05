"""P2-S0: pre-tension de-risk. The ANCF shell diverged because the membrane has no pre-tension
(floppy plate, not drum-skin). set_pre_tension(N0) now seeds a constant 2nd-PK pre-stress ->
taut membrane (geometric stiffness K_sigma). Sweep N0 to see if the FSI stabilizes.
Root-cause test: if a taut membrane is stable, pre-tension IS the divergence fix (element type
secondary); if still diverges, ANCF thin-plate has other issues -> standard FEM (S1/S2)."""
import warp as wp; wp.init()
import numpy as np, warnings
warnings.filterwarnings('ignore')
import _v2_flex_robo as fr
from newton_pc import WindowPredictorCorrector
from newton_pc.adapters.flap import FlapKinematics, FlapUVLMProvider, NodalForceSet


def trial(N0, Ey=50e9, damping=0.2, nc=6, ns=12, n_cycles=2):
    Ex, k, _ = fr.calibrate_Ex(nc, ns, Ey, '11b', n_it=3)
    period = 1.0/2.3; CHORD = 0.287; dtw = (CHORD/nc)/8.0
    wpc = int(round(period/dtw)); nw = int(round(n_cycles*period/dtw))
    kin = FlapKinematics(np.deg2rad(45.0), period)
    entry = fr.FlapEntryRobo(nc, ns, kin, Ex, Ey, damping=damping, three_mat=False, pre_tension=N0)
    V = 8.0*np.array([np.cos(np.deg2rad(5)), 0, np.sin(np.deg2rad(5))])
    prov = FlapUVLMProvider(V, 1.225, dtw, K=6, nu=15.06e-6, chord=CHORD,
                            particles=False, max_particles=1, added_mass_operator=True)
    pc = WindowPredictorCorrector(entry=entry, provider=prov, substeps=10, dt=dtw/10, mode='two-pass')
    pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))
    try: pc.advance(n_substeps=1)
    except Exception as e: return f'init ERR {type(e).__name__}: {str(e)[:40]}'
    lift = []; bend = []; fin = True
    for w in range(nw):
        try: pc.advance()
        except Exception as e: return f'w={w} ERR {type(e).__name__}'
        st = entry.state()
        if not np.all(np.isfinite(st['verts'])): fin = False; break
        F = pc._F_cur.payload['f_panel'].sum(axis=(0, 1)) if pc._F_cur.payload else np.zeros(3)
        lift.append(float(-F[0]*np.sin(np.deg2rad(5)) + F[2]*np.cos(np.deg2rad(5))))
        th = kin.angles(pc._t)[0]; zr = (entry.nodes0[:, 1]*np.sin(th)).reshape(ns+1, nc+1).T
        b = st['verts'][..., 2] - zr; bend.append(float(b.max() - b.min()))
    Lm = 2.0*np.mean(lift[-wpc:]) if len(lift) >= wpc else float('nan')
    return f'N0={N0:>6.0f}N/m: Ex={Ex/1e6:>5.1f}MPa L={Lm:+7.2f} bend={max(bend) if bend else 0:.3f} fin={fin} n={len(lift)}'


if __name__ == '__main__':
    print('P2-S0 pre-tension de-risk (uniform orthotropic, Ey=50GPa, added_mass, damp=0.2):')
    for N0 in (0.0, 100.0, 500.0, 2000.0, 10000.0):
        print(trial(N0), flush=True)
