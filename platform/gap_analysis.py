"""STEP-1 of the gap->research->implement loop (user-mandated methodology):
Systematic GAP structure analysis — how does (model - experiment) vary with operating
condition (U, f, twist, aoa) and with cycle phase? Outputs scaling-law fits + structured
tables/plots that become the phenomenology fed to the literature search (step 2).

  python gap_analysis.py --models fix,H4,H13     # means: per-axis gap laws + global regression
  (phase-dimension gap uses docs/diag/fig16_series.npz if present)

Outputs: docs/diag/gap_structure_<model>.png + gap_laws.md (printed + saved)
"""
import os, json, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
OD = os.path.join(HERE, 'docs', 'diag'); os.makedirs(OD, exist_ok=True)
from _v2_repro_nc12 import cond_of, ckey


def load_pts(model):
    """-> list of (U, aoa, f, tw, kind, exp, mod) matched points."""
    R = json.load(open('docs/repro_data.json'))
    C = json.load(open(f'docs/repro_nc12/cache_nc4_cyc3_{model}.json'))
    pts = []
    for key in R:
        kind = R[key]['kind']
        for xi, e in zip(R[key]['x'], R[key]['exp']):
            U, aoa, f, tw = cond_of(key, xi); ck = ckey(U, aoa, f, tw)
            v = C.get(ck)
            if v is None or not np.isfinite(v[0]): continue
            pts.append((U, aoa, f, tw, kind, e, v[0] if kind == 'L' else v[1]))
    return pts


def axis_fits(pts, kind):
    """Per-axis gap laws: slope/intercept of gap along each swept axis at fixed others."""
    rows = []
    P = [p for p in pts if p[4] == kind]
    # gap vs f at tw=0, per (U, aoa)
    for (U, aoa) in sorted({(p[0], p[1]) for p in P if p[3] == 0}):
        s = sorted([(p[2], p[6] - p[5]) for p in P if p[0] == U and p[1] == aoa and p[3] == 0])
        if len(s) < 3: continue
        x = np.array([q[0] for q in s]); g = np.array([q[1] for q in s])
        A = np.vstack([np.ones_like(x), x ** 2]).T
        c, res, *_ = np.linalg.lstsq(A, g, rcond=None)
        r2 = 1 - (res[0] if len(res) else 0) / (np.var(g) * len(g) + 1e-12)
        rows.append(('f@tw0', f'U{U:g}/aoa{aoa:g}', f'gap = {c[0]:+.2f} {c[1]:+.3f}·f²', f'{r2:.2f}', f'{len(s)}'))
    # gap vs tw at f=2.6, per (U, aoa)
    for (U, aoa) in sorted({(p[0], p[1]) for p in P if abs(p[2] - 2.6) < .01}):
        s = sorted([(p[3], p[6] - p[5]) for p in P if p[0] == U and p[1] == aoa and abs(p[2] - 2.6) < .01])
        if len(s) < 4: continue
        x = np.radians(np.array([q[0] for q in s])); g = np.array([q[1] for q in s])
        A = np.vstack([np.ones_like(x), x ** 2]).T
        c, res, *_ = np.linalg.lstsq(A, g, rcond=None)
        r2 = 1 - (res[0] if len(res) else 0) / (np.var(g) * len(g) + 1e-12)
        rows.append(('tw@2.6Hz', f'U{U:g}/aoa{aoa:g}', f'gap = {c[0]:+.2f} {c[1]:+.2f}·tw²[rad]', f'{r2:.2f}', f'{len(s)}'))
    # gap vs aoa at tw0 (Fig19 a/b axis), per f
    for f in sorted({p[2] for p in P if p[0] == 8.0 and p[3] == 0}):
        s = sorted([(p[1], p[6] - p[5]) for p in P if p[0] == 8.0 and p[3] == 0 and abs(p[2] - f) < .01])
        if len({q[0] for q in s}) < 3: continue
        x = np.radians(np.array([q[0] for q in s])); g = np.array([q[1] for q in s])
        A = np.vstack([np.ones_like(x), x]).T
        c, res, *_ = np.linalg.lstsq(A, g, rcond=None)
        r2 = 1 - (res[0] if len(res) else 0) / (np.var(g) * len(g) + 1e-12)
        rows.append((f'aoa@tw0', f'U8/f{f:g}', f'gap = {c[0]:+.2f} {c[1]:+.2f}·aoa[rad]', f'{r2:.2f}', f'{len(s)}'))
    return rows


def global_fit(pts, kind):
    """One regression over ALL points: gap ~ 1, f², tw², f²tw², aoa, U² — which terms dominate."""
    P = [p for p in pts if p[4] == kind]
    U = np.array([p[0] for p in P]); aoa = np.radians([p[1] for p in P])
    f = np.array([p[2] for p in P]); tw = np.radians([p[3] for p in P])
    g = np.array([p[6] - p[5] for p in P])
    feats = {'1': np.ones_like(g), 'f²': f ** 2, 'tw²': tw ** 2, 'f²·tw²': (f * tw) ** 2,
             'aoa': aoa, 'U²/64': U ** 2 / 64.0, 'f²·U²/64': f ** 2 * U ** 2 / 64}
    X = np.vstack(list(feats.values())).T
    c, res, *_ = np.linalg.lstsq(X, g, rcond=None)
    pred = X @ c
    r2 = 1 - np.sum((g - pred) ** 2) / (np.sum((g - g.mean()) ** 2) + 1e-12)
    # term importance: |coef| * std(feature)
    imp = {k: abs(c[i]) * np.std(X[:, i]) for i, k in enumerate(feats)}
    order = sorted(imp, key=lambda k: -imp[k])
    return c, list(feats), r2, order, float(np.sqrt(np.mean((g - pred) ** 2))), float(np.std(g))


def phase_gap(model):
    """Phase-dimension: harmonic decomposition of the Fig16 instantaneous gap per twist."""
    RAW = os.path.join(OD, 'fig16_series.npz')
    if not os.path.exists(RAW): return []
    from fig16_compare import parse_fig16, butter8
    d = dict(np.load(RAW)); E = parse_fig16(); out = []
    SHIFT = 187  # global phase shift established by tw0 lift xcorr (fig16_compare)
    for tw in (0.0, 22.5, 45.0):
        for kind, idx in (('T', 0), ('L', 1)):
            k = f'{model}_{tw:g}_{kind}'
            if k not in d: continue
            s = np.roll(butter8(d[k], 2.0), SHIFT)
            te, fe = E[(kind, tw)]
            tp = (np.arange(len(s)) + 0.5) / len(s)
            gap = s - np.interp(tp, te, fe, period=1.0)
            # harmonics 0..3 of the gap
            H = []
            for n in range(4):
                cn = 2 * np.mean(gap * np.cos(2 * np.pi * n * tp)) if n else np.mean(gap)
                sn = 2 * np.mean(gap * np.sin(2 * np.pi * n * tp)) if n else 0.0
                amp = np.hypot(cn, sn); ph = np.degrees(np.arctan2(sn, cn))
                H.append((amp, ph))
            tworst = tp[int(np.argmax(np.abs(gap)))]
            out.append((kind, tw, H, float(np.max(np.abs(gap))), float(tworst)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default='fix,H4,H13')
    a = ap.parse_args()
    lines = ['# GAP structure (model − experiment) — step-1 of gap→research→implement\n']
    for m in a.models.split(','):
        pts = load_pts(m)
        lines.append(f'\n## model {m}  ({len(pts)} matched pts)\n')
        for kind in ('T', 'L'):
            c, names, r2, order, rmse, gstd = global_fit(pts, kind)
            lines.append(f'\n### {kind}: global gap regression (R²={r2:.2f}, resid RMSE {rmse:.2f}N vs gap std {gstd:.2f}N)')
            lines.append('| term | coef | rank |')
            lines.append('|---|---|---|')
            for i, nm in enumerate(names):
                lines.append(f'| {nm} | {c[i]:+.3f} | {order.index(nm) + 1} |')
            lines.append(f'\n### {kind}: per-axis laws (gap fits along swept axes)')
            lines.append('| axis | fixed | law | R² | n |')
            lines.append('|---|---|---|---|---|')
            for r in axis_fits(pts, kind):
                lines.append('| ' + ' | '.join(r) + ' |')
        ph = phase_gap(m)
        if ph:
            lines.append(f'\n### phase-dimension gap (Fig16 8/5/2Hz, filtered, harmonics of model−exp)')
            lines.append('| kind | tw | mean(N) | 1/rev amp | 2/rev amp | 3/rev amp | max|gap| | @t/T |')
            lines.append('|---|---|---|---|---|---|---|---|')
            for kind, tw, H, mx, tt in ph:
                lines.append(f'| {kind} | {tw:g} | {H[0][0]:+.2f} | {H[1][0]:.2f}∠{H[1][1]:.0f}° | '
                             f'{H[2][0]:.2f}∠{H[2][1]:.0f}° | {H[3][0]:.2f} | {mx:.2f} | {tt:.2f} |')
    md = '\n'.join(lines)
    open(os.path.join(OD, 'gap_laws.md'), 'w').write(md)
    print(md)
    print(f"\nsaved {os.path.join(OD, 'gap_laws.md')}")


if __name__ == '__main__':
    main()
