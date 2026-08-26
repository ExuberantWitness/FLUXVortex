"""Compare the frozen 50-substep reference against a diagnostic substep
override on the same 110-step window: time-convergence evidence for the
structural clock.  Usage:
    python compare_substep_convergence.py REFERENCE.json PROBE.json
"""
import json, sys
import numpy as np

def load(path):
    p = json.load(open(path))
    cn = np.array([r["cn"] for r in p["records"]], dtype=np.float64)
    z = np.array([r["instantaneous_zmax_over_c"] for r in p["records"]], dtype=np.float64)
    return p, cn, z

ref_path, probe_path = sys.argv[1], sys.argv[2]
ref, cn_r, z_r = load(ref_path)
probe, cn_p, z_p = load(probe_path)
n = min(len(cn_r), len(cn_p))
f = np.array([r["freestream_factor"] for r in ref["records"][:n]])
cn_scale = max(float(np.abs(cn_r).max()), 1e-12)
z_scale = max(float(np.abs(z_r).max()), 1e-12)
d_cn = np.abs(cn_p[:n] - cn_r[:n])
d_z = np.abs(z_p[:n] - z_r[:n])
print(f"window: {n} steps; substeps ref={ref.get('structural_substeps')} probe={probe.get('structural_substeps')}")
print(f"Cn:  max|dCn|={d_cn.max():.3e} ({100*d_cn.max()/cn_scale:.3f}% of max|Cn|={cn_scale:.3f})")
print(f"z/c: max|dz/c|={d_z.max():.3e} ({100*d_z.max()/z_scale:.3f}% of max|z/c|={z_scale:.3f})")
for i in range(0, n, max(1, n//8)):
    print(f"  step {i+1:3d} f={f[i]:.3f}: Cn {cn_r[i]:+.4f} vs {cn_p[i]:+.4f} (d={d_cn[i]:.1e})  z/c {z_r[i]:.5f} vs {z_p[i]:.5f} (d={d_z[i]:.1e})")
