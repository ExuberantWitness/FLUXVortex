"""Summarize the particle population control study (A/B/C lanes)."""
import os, sys
import numpy as np
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

A = np.load('flap_arena/out/pcs_A.npz')
rows = {}
for tag in sys.argv[1:] or ['A','B','C3']:
    try: rows[tag] = np.load(f'flap_arena/out/pcs_{tag}.npz')
    except FileNotFoundError: print(f'({tag} missing)')

ref = A['lift']
print(f"{'lane':>5} {'scheme':>6} {'eps':>8} {'n_p end':>8} {'ms/win mean':>11} "
      f"{'ms/win last100':>14} {'NRMSE vs A':>10} {'ledger drift':>12}")
for tag, d in rows.items():
    L = d['lift']; n = min(len(L), len(ref))
    err = L[:n] - ref[:n]
    nrmse = np.sqrt((err**2).mean()) / np.sqrt((ref[:n]**2).mean())
    led = d['ledger']
    drift = np.linalg.norm(led[-1] - led[min(len(led)-1, 400)])
    print(f"{tag:>5} {str(d['scheme']):>6} {float(d['eps']):8.0e} "
          f"{int(d['npart'][-1]):8d} {d['wall'].mean()*1000:11.0f} "
          f"{d['wall'][-100:].mean()*1000:14.0f} {nrmse*100:9.2f}% "
          f"{drift:12.3e}")
