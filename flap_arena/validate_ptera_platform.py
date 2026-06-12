"""Platform-config validation vs PteraSoftware (case 1 of the suite)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from newton_pc.case import CaseConfig, run_case

for amp, bl in [(5.0, 'ptera_baseline5.npz'), (15.0, 'ptera_baseline.npz')]:
    base = np.load(f'flap_arena/out/{bl}')
    dtw = float(base['dt_free'])
    cfg = CaseConfig(kin_amp_deg=amp, window_dt=dtw, K=8, particles=True,
                     cycles=3)
    res = run_case(cfg)
    np.savez(f'flap_arena/out/flap_PLAT{int(amp)}.npz', lift=res.lift,
             dtw=dtw, n_windows=len(res.lift))
    print(f'[platform {amp}deg] windows={len(res.lift)} wall={res.wall_s:.0f}s '
          f'solves={res.n_solves}', flush=True)
