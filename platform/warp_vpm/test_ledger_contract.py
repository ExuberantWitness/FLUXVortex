"""P0-1 regression: ledger_step()['total'] must equal t1+t2+t3 after G6b clip.

Also verifies the audit's specific scenario: a high-separation record that
triggers the Rayleigh ceiling clip must return a closed total.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bing_drag_ledger import LedgerConfig, ledger_step

cfg = LedgerConfig(
    lesp_crit=0.11, aspect_ratio=4.0, rho=998.0, cd0=0.04,
    enable_t1=True, enable_t2=True, enable_t3=True)

# Synthetic record with extreme separation to trigger G6b
S = 8
np.random.seed(42)
rec = {
    "step": 0,
    "lesp": np.full(S, 5.0),          # >> crit, max separation
    "chords": np.full(S, 0.076),
    "areas": np.full(S, 0.076 * 0.6 / S),
    "v_rel_st": np.tile([0.3, 0.0, 0.1], (S + 1, 1)),
    "v_inf": np.array([0.3, 0.0, 0.0]),
    "le_now": np.column_stack([
        np.full(S + 1, 0.0),
        np.linspace(0, 0.6, S + 1),
        np.full(S + 1, 0.05)]),
    "te_now": np.column_stack([
        np.full(S + 1, 0.076),
        np.linspace(0, 0.6, S + 1),
        np.full(S + 1, 0.0)]),
    "cn_strip": np.full(S, 10.0),     # very high attached CN -> big T2
    "dt": 0.1,
}

out = ledger_step(rec, cfg)

# P0-1: total must close
t_sum = out["t1"] + out["t2"] + out["t3"]
closure = np.max(np.abs(out["total"] - t_sum))
assert closure < 1e-12, f"P0-1 FAIL: total closure error {closure:.3e}"
print(f"P0-1 PASS: total == t1+t2+t3, max|d| = {closure:.2e}")

# Verify the clip actually engaged (t1+t2 should be bounded by Rayleigh)
sep_sum = out["t1"] + out["t2"]
alpha = out["alpha_eff"]
rayleigh = 2.0 * np.sin(alpha)**2 * out["q"] * rec["areas"]
assert np.all(sep_sum <= rayleigh + 1e-10), \
    f"G6b FAIL: separation {sep_sum.max():.3e} > Rayleigh {rayleigh.max():.3e}"
print(f"G6b PASS: separation bounded by Rayleigh ceiling")

# Normal case (attached, no clip): also must close
rec2 = dict(rec, lesp=np.full(S, 0.05))  # < crit, attached
out2 = ledger_step(rec2, cfg)
closure2 = np.max(np.abs(out2["total"] - (out2["t1"] + out2["t2"] + out2["t3"])))
assert closure2 < 1e-15, f"attached FAIL: closure {closure2:.3e}"
print(f"Attached PASS: closure = {closure2:.2e}")
print("ALL LEDGER CONTRACT TESTS PASS")
