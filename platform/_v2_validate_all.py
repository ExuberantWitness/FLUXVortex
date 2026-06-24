"""Exhaustive validation of the strip-LDVM RoboEagle model vs the full data.md battery (Fig 17/18/19).

Parses docs/data.md directly into structured (condition -> measured lift/thrust). Baselines inferred by
cross-referencing (Fig18@8m/s == Fig19@5deg -> Fig18 is AoA=5deg, Fig19 is 8m/s; Fig19 c/d twist sweeps
at 2.6Hz). Column labels in (c)/(d) are MISLABELED -> classify lift/thrust by sign (lift=large +, thrust=-).

Model: _v2_flap_strip.flapping_wing (per-strip 2D LDVM, lift_p + saturated LE suction + NACA-2406 camber,
3D-downwash x AR/(AR+2)). 2D strip has NO induced drag -> NET thrust is under-resolved (gross suction only);
LIFT is the meaningful quantitative test. Honest: report both, flag the induced-drag gap on thrust.
"""
import re, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GF = 9.80665e-3  # 1 gram-force in Newtons
AR = (2 * 0.80) / 0.287
C3D = AR / (AR + 2.0)            # finite-wing downwash correction ~0.74

# figure -> (kind, which params are swept / fixed). Verified by cross-referencing the data.
FIG_SPEC = {
    ("17", "a"): dict(kind="T", sweep="twist", wind=8.0, aoa=5.0),    # freq from header
    ("17", "b"): dict(kind="L", sweep="twist", wind=8.0, aoa=5.0),
    ("18", "a"): dict(kind="T", sweep="freq", aoa=5.0, twist=0.0),    # wind from header
    ("18", "b"): dict(kind="L", sweep="freq", aoa=5.0, twist=0.0),
    ("18", "c"): dict(kind="T", sweep="twist", aoa=5.0),             # wind,freq from header
    ("18", "d"): dict(kind="L", sweep="twist", aoa=5.0),
    ("19", "a"): dict(kind="T", sweep="freq", wind=8.0, twist=0.0),   # aoa from header
    ("19", "b"): dict(kind="L", sweep="freq", wind=8.0, twist=0.0),
    ("19", "c"): dict(kind="T", sweep="twist", wind=8.0, freq=2.6),   # aoa from header
    ("19", "d"): dict(kind="L", sweep="twist", wind=8.0, freq=2.6),
}


def _parse_header(h):
    """'8m/s，2.3Hz' / '1.4HZ' / '15度' -> dict of whatever it specifies."""
    out = {}
    m = re.search(r"([\d.]+)\s*m/s", h)
    if m: out["wind"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*[Hh][Zz]", h)
    if m: out["freq"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*度", h)
    if m: out["aoa"] = float(m.group(1))
    return out


def parse_data_md(path=None):
    path = path or os.path.join(HERE, "docs", "data.md")
    blocks = []
    fig = sub = None
    cur = None
    for line in open(path, encoding="utf-8"):
        mf = re.search(r"Figure\s+(\d+)\.?\s*\(([a-d])\)", line)
        if mf:
            fig, sub = mf.group(1), mf.group(2)
            continue
        if "工况" in line:
            if cur and cur["points"]:
                blocks.append(cur)
            hdr = _parse_header(line.split("：", 1)[-1] if "：" in line else line)
            spec = dict(FIG_SPEC.get((fig, sub), {}))
            spec.update(hdr)
            cur = dict(fig=fig, sub=sub, kind=spec.get("kind", "?"), sweep=spec.get("sweep"),
                       wind=spec.get("wind"), aoa=spec.get("aoa"), freq=spec.get("freq"),
                       twist=spec.get("twist"), points=[])
            continue
        m = re.match(r"\s*(-?[\d.]+e[+-]\d+)\s+(-?[\d.]+e[+-]\d+)", line)
        if m and cur is not None:
            cur["points"].append((float(m.group(1)), float(m.group(2))))
    if cur and cur["points"]:
        blocks.append(cur)
    return blocks


def run_case(wind, aoa, freq, twist, settings):
    from _v2_flap_strip import flapping_wing
    r = flapping_wing(U=wind, aoa_deg=aoa, flap_amp_deg=45.0, twist_amp_deg=twist, freq=freq,
                      lesp_crit=0.20, lev_shed=False, camber_m=0.02, **settings)
    return r["L"] * C3D, r["T"] * C3D


def stats(jpath=None):
    """Per-axis trend (correlation) + magnitude (ratio) stats from a validation JSON."""
    import json
    jpath = jpath or os.path.join(HERE, "_v2_validation.json")
    d = json.load(open(jpath))
    rows = d["rows"]
    def corr(xs, ys):
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 3 or xs.std() < 1e-9 or ys.std() < 1e-9: return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])
    # group rows by (fig, kind, the fixed-params signature) -> a single sweep curve
    from collections import defaultdict
    curves = defaultdict(list)
    for r in rows:
        key = (r["fig"], r["kind"], r["sweep"], r["wind"], r["aoa"], r["freq"])
        curves[key].append(r)
    print(f"{'fig':5s}{'kind':>5}{'sweep':>7}{'cond':>22}  {'trendCorr':>9} {'meanRatio':>9} {'n':>3}")
    agg = defaultdict(lambda: [[], [], []])   # by (kind,sweep): corrs, ratios, all (meas,model)
    for key, rs in sorted(curves.items()):
        fig, kind, sweep, wind, aoa, freq = key
        rs = sorted(rs, key=lambda r: r["x"])
        meas = [r["meas_N"] for r in rs]; model = [r["model_N"] for r in rs]
        c = corr([r["x"] for r in rs], meas) and corr(meas, model)
        cc = corr(meas, model)
        ratios = [m / d for m, d in zip(model, meas) if abs(d) > 0.5]
        mr = float(np.mean(ratios)) if ratios else float("nan")
        cond = f"w{wind} a{aoa} f{freq}"
        if sweep in ("freq",):   # only print the trend-critical freq sweeps in full
            print(f"{fig:5s}{kind:>5}{sweep:>7}{cond:>22}  {cc:>+9.2f} {mr:>9.2f} {len(rs):>3}")
        agg[(kind, sweep)][0].append(cc)
        agg[(kind, sweep)][1].append(mr)
        agg[(kind, sweep)][2].extend(zip(meas, model))
    print("\n=== aggregate by (kind, sweep) ===")
    print(f"{'kind':>5}{'sweep':>7} {'nCurves':>7} {'medTrendCorr':>13} {'medRatio':>9} {'RMSE_N':>8}")
    for (kind, sweep), (corrs, ratios, mm) in sorted(agg.items()):
        cc = np.nanmedian(corrs); mr = np.nanmedian(ratios)
        mm = np.array(mm); rmse = float(np.sqrt(np.mean((mm[:, 1] - mm[:, 0]) ** 2)))
        print(f"{kind:>5}{sweep:>7} {len(corrs):>7} {cc:>+13.2f} {mr:>9.2f} {rmse:>8.2f}")


def _cond_of(b, x):
    """The (wind,aoa,freq,twist) condition for sweep-point x of block b (exact x; a/b pairs dedupe)."""
    wind, aoa, freq, twist = b["wind"], b["aoa"], b["freq"], b["twist"]
    if b["sweep"] == "twist":
        twist = max(0.0, x)
    elif b["sweep"] == "freq":
        freq = x
    return (round(wind, 3), round(aoa, 3), round(freq, 3), round(twist, 2))


def _job(args):
    (wind, aoa, freq, twist), settings = args
    L, T = run_case(wind, aoa, freq, twist, settings)
    return (wind, aoa, freq, twist), (L, T)


def full_sweep(blocks, settings, nproc=14):
    """Deduplicate to unique conditions, run each once (multiprocessing -> both L and T), map back."""
    conds = {}
    for b in blocks:
        for (x, _) in b["points"]:
            conds[_cond_of(b, x)] = None
    jobs = [((w, a, f, t), settings) for (w, a, f, t) in conds]
    from multiprocessing import Pool
    with Pool(nproc) as p:
        for cond, LT in p.map(_job, jobs):
            conds[cond] = LT
    rows = []
    for b in blocks:
        for (x, meas_g) in b["points"]:
            L, T = conds[_cond_of(b, x)]
            model = L if b["kind"] == "L" else T
            rows.append(dict(fig=b["fig"] + b["sub"], kind=b["kind"], sweep=b["sweep"],
                             wind=b["wind"], aoa=b["aoa"], freq=b["freq"], twist=b["twist"],
                             x=round(x, 2), meas_N=round(meas_g * GF, 3), model_N=round(model, 3)))
    return rows


if __name__ == "__main__":
    blocks = parse_data_md()
    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        import json
        from collections import defaultdict
        d = json.load(open(os.path.join(HERE, "_v2_validation.json")))
        rows = d["rows"]
        groups = defaultdict(list)
        for r in rows:
            # group a sweep curve by (fig, the fixed params)
            key = (r["fig"], r["kind"], r["sweep"], r["wind"], r["aoa"], r["freq"], r["twist"])
            groups[key].append(r)
        def corr(a, b):
            a = np.array(a); b = np.array(b)
            if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9: return float("nan")
            return float(np.corrcoef(a, b)[0, 1])
        print(f"{'fig':>5}{'k':>2} {'sweep':>5} {'fixed(w/a/f/t)':>16} {'n':>3} {'meanRatio':>9} "
              f"{'trendCorr':>9} {'meas[lo..hi]N':>16} {'model[lo..hi]N':>16}")
        agg = defaultdict(lambda: [0, 0.0])
        for key in sorted(groups):
            g = sorted(groups[key], key=lambda r: r["x"])
            fig, kind, sweep, w, a, f, t = key
            xs = [r["x"] for r in g]; me = [r["meas_N"] for r in g]; mo = [r["model_N"] for r in g]
            if kind == "L":
                ratios = [mo[i] / me[i] for i in range(len(g)) if abs(me[i]) > 0.3]
                mr = float(np.mean(ratios)) if ratios else float("nan")
            else:
                mr = float("nan")   # thrust: ratio meaningless (sign), use offset/trend
            tc = corr(me, mo)
            fixed = f"{w}/{a}/{f}/{t}"
            print(f"{fig:>5}{kind:>2} {sweep:>5} {fixed:>16} {len(g):>3} {mr:>9.3f} {tc:>9.2f} "
                  f"[{min(me):6.1f}..{max(me):6.1f}] [{min(mo):6.1f}..{max(mo):6.1f}]")
            if kind == "L" and ratios:
                agg[("L", sweep)][0] += len(ratios); agg[("L", sweep)][1] += sum(ratios)
        print("\n=== LIFT mean ratio by sweep type ===")
        for k, (n, s) in sorted(agg.items()):
            print(f"  {k[1]:>6}: mean model/meas ratio = {s/n:.3f}  (n={n})")
    elif len(sys.argv) > 1 and sys.argv[1] == "--run":
        import json
        # first-principles strip (lift_p attached pressure + Garrick LE suction; no empirical Polhamus).
        SET = dict(ns=6, nc=30, steps_per_cycle=120, max_wake=200, n_cycle=4)
        rows = full_sweep(blocks, SET, nproc=14)
        json.dump(dict(settings=SET, C3D=C3D, rows=rows), open(os.path.join(HERE, "_v2_validation.json"), "w"), indent=0)
        print(f"ran {len({(_cond_of(b,x)) for b in blocks for (x,_) in b['points']})} unique conds, {len(rows)} rows -> _v2_validation.json")
    elif len(sys.argv) > 1 and sys.argv[1] == "--list":
        nL = nT = 0
        for b in blocks:
            n = len(b["points"])
            nL += n if b["kind"] == "L" else 0
            nT += n if b["kind"] == "T" else 0
            print(f"Fig{b['fig']}{b['sub']} kind={b['kind']} sweep={b['sweep']:5s} "
                  f"wind={b['wind']} aoa={b['aoa']} freq={b['freq']} twist={b['twist']} "
                  f"npts={n}  x[{b['points'][0][0]:.1f}..{b['points'][-1][0]:.1f}] "
                  f"meas[{b['points'][0][1]:.0f}..{b['points'][-1][1]:.0f}]g")
        print(f"\nTOTAL: {len(blocks)} blocks, {nL} lift pts, {nT} thrust pts")
