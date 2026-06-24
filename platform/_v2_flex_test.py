import warp as wp; wp.init()
import _v2_flex_robo as fr
import numpy as np
print("Diagnostic: stable run with physical stiff E (rule out soft-Ex vs degenerate-tip geometry)", flush=True)
print("  if stable -> soft Ex was the issue; if diverges -> raked-tip elements degenerate\n", flush=True)
for E, ss in [(50e9, 16), (10e9, 30)]:
    r = fr.run_fsi(E_override=E, substeps=ss, n_cycles=3, nc=6, ns=12)
    print(f"  E={E/1e9:.0f}GPa substeps={ss}: L={r['L']:+.2f}N bend_max={r['bend_max']:.4f}m finite={r['finite']} ({r['n']}w)", flush=True)
print("DONE", flush=True)
