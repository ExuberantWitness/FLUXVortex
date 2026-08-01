"""Three-point E1/E2 reproduction gate for the frozen 2026-07-25 v4.1 cache."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from lb_sweep118 import BASE
from _v2_repro_nc12 import spc_of


def main():
    import warp as wp

    wp.init()
    from _v2_robo import gpu_run_twist

    cache_path = os.path.join(HERE, "docs", "s6_sweep_v41.json")
    with open(cache_path) as stream:
        cache = json.load(stream)

    spc = spc_of(8.0, 2.6)
    worst = 0.0
    for aoa in (15.0, 0.0, 5.0):
        result = gpu_run_twist(
            U=8.0, aoa_deg=aoa, freq=2.6,
            twist_amp_deg=11.25, nc=12, ns=16, n_cycle=4,
            steps_per_cycle=spc, wake_rows=spc, closure="v41", **BASE
        )
        key = f"8_2.6_22.5_{aoa:g}"
        reference = cache[key]
        delta_l = float(result["L_wind"]) - reference["L"]
        delta_t = float(result["T_wind"]) - reference["T"]
        worst = max(worst, abs(delta_l), abs(delta_t))
        print(
            f"{key}: L={result['L_wind']:+.3f} (d={delta_l:+.3f}), "
            f"T={result['T_wind']:+.3f} (d={delta_t:+.3f})"
        )

    tolerance = 0.15
    print(f"worst_abs_delta={worst:.3f} N; tolerance={tolerance:.2f} N")
    if worst > tolerance:
        raise SystemExit("FAIL: v41 does not reproduce the frozen cache")
    print("PASS: v41 identity reproduces the frozen cache")


if __name__ == "__main__":
    main()
