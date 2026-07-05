"""P2a: UNIFORM orthotropic stabilization sweep (the 3-material shell diverges — thin shell can't
represent beam bending). Test if strong Ey (spar) + heavy damping + added_mass stabilizes the
smeared model enough to port closures + hit the fidelity gate (discrete beam = P3+ refinement)."""
import warp as wp; wp.init()
import numpy as np, warnings
warnings.filterwarnings('ignore')
import _v2_flex_robo as fr
from newton_pc import WindowPredictorCorrector
from newton_pc.adapters.flap import FlapKinematics, FlapUVLMProvider, NodalForceSet


def trial(Ey, am, dmp, nc=6, ns=12, n_cycles=2):
    Ex, k, _ = fr.calibrate_Ex(nc, ns, Ey, '11b', n_it=3)
    period = 1.0/2.3; CHORD = 0.287; dtw = (CHORD/nc)/8.0
    wpc = int(round(period/dtw)); nw = int(round(n_cycles*period/dtw))
    kin = FlapKinematics(np.deg2rad(45.0), period)
    entry = fr.FlapEntryRobo(nc, ns, kin, Ex, Ey, damping=dmp, three_mat=False)
    V = 8.0*np.array([np.cos(np.deg2rad(5)), 0, np.sin(np.deg2rad(5))])
    prov = FlapUVLMProvider(V, 1.225, dtw, K=6, nu=15.06e-6, chord=CHORD,
                            particles=False, max_particles=1, added_mass_operator=am)
    pc = WindowPredictorCorrector(entry=entry, provider=prov, substeps=10, dt=dtw/10, mode='two-pass')
    pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))
    try: pc.advance(n_substeps=1)
    except Exception as e: return f'init ERR {type(e).__name__}'
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
    return f'Ey={Ey/1e9:>3.0f}GPa am={int(am)} damp={dmp:.1f}: Ex={Ex/1e6:>5.1f}MPa L={Lm:+7.2f} bend={max(bend) if bend else 0:.3f} fin={fin} n={len(lift)}'


if __name__ == '__main__':
    print('UNIFORM orthotropic stabilization (calibrated Ex to k=166.9):')
    for Ey in (50e9, 100e9, 200e9):
        for dmp in (0.3, 0.6):
            print(trial(Ey, True, dmp), flush=True)
