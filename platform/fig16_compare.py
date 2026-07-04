"""Fig16 INSTANTANEOUS (phase-resolved) validation: exp (datav2.md, 8m/s AoA5 2Hz, tw 0/22.5/45)
vs model per-step wind-frame forces. The phase signature discriminates mechanisms (mid-stroke vs
reversal, upstroke vs downstroke) far better than cycle means.

  python fig16_compare.py --models K0,H4,H9   # ~25min GPU (9 runs, nc4/ns16/spc240/ncyc3)

Phase alignment: ONE cyclic shift, chosen by cross-correlating the tw0 LIFT curve (largest clean
signal), applied to ALL cases/models. Output: docs/diag/fig16_compare.png + per-case RMSE table.
"""
import os, re, json, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
OD = os.path.join(HERE, 'docs', 'diag'); os.makedirs(OD, exist_ok=True)
from _v2_repro_nc12 import CFG_PRESETS, FIX

G2N = 9.81 / 1000.0
COND = dict(U=8.0, aoa_deg=5.0, freq=2.0)
NC, NS, NCYC, SPC = 4, 16, 3, 240          # spc = 15*8*4/2.0 = 240 (production U*dt/chord ratio)
MODELS = {
    'K0': dict(CFG_PRESETS['K0']),
    'H4': dict(CFG_PRESETS['H4']),
    'H9': dict(CFG_PRESETS['H4'], **FIX, geo_stall_vec=True),
    'K0s': dict(CFG_PRESETS['K0'], stall=True),      # + CL_max saturation of the Bernoulli VECTOR (a_stall~11deg airfoil)
    'H4s': dict(CFG_PRESETS['H4'], stall=True),
    'H10': dict(CFG_PRESETS['H10']),                 # unified alpha_eff-Kirchhoff closure (kirch_cn + les_att)
    'H11': dict(CFG_PRESETS['H11']),                 # + ring strength capped at the exact Kelvin excess
    'H12': dict(CFG_PRESETS['H12']),                 # stall cap (sees induced loads) + les_att (kills fake pulses)
    'H13': dict(CFG_PRESETS['H13']),                 # + Goman-Khrabrov separation lag (tau*=4.5 literature)
    'K1':  dict(CFG_PRESETS['K1']),                  # kelvin path + closure suite + fp_lev DSV lift
    'K1g': dict(CFG_PRESETS['K1g']),                 # + legacy geo_stall
    'H14': dict(CFG_PRESETS['H14']),                 # literature-faithful Hirato (ansari LEV sheet + LESP constraint, vnf off)
    'H15': dict(CFG_PRESETS['H15']),                 # H14 + stall cap (production candidate)
    'H16': dict(CFG_PRESETS['H16']),                 # Li/Feng vortex-impulse LEV force (grid-independent)
    'H17': dict(CFG_PRESETS['H17']),                 # H16 + stall cap (non-conflicting: cap on Fb, impulse separate)
}
ZCH = ['Lh_bern', 'Lh_les', 'Lh_vtx', 'Lh_pd', 'Lh_vis', 'Lh_stall', 'Lh_vimp']
XCH = ['Xh_bern', 'Xh_les', 'Xh_vtx', 'Xh_pd', 'Xh_vis', 'Xh_stall', 'Xh_vimp']


def parse_fig16():
    """-> {('T'|'L', twist): (t_norm, force_N)} from datav2.md."""
    txt = open('docs/datav2.md').read()
    out = {}
    # split into the (a) thrust and (b) lift halves
    ia = txt.index('Figure 16. (a)'); ib = txt.index('Figure 16. (b)', ia + 10)
    for kind, seg in (('T', txt[ia:ib]), ('L', txt[ib:])):
        for m in re.finditer(r'Twist amplitude\((\d+(?:\.\d+)?)°?\)', seg):
            tw = float(m.group(1))
            tail = seg[m.end():]
            nxt = re.search(r'Twist amplitude\(', tail)
            block = tail[:nxt.start()] if nxt else tail
            pairs = re.findall(r'(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)', block, re.I)
            arr = np.array([[float(a), float(b)] for a, b in pairs])
            i = np.argsort(arr[:, 0])
            out[(kind, tw)] = (arr[i, 0], arr[i, 1] * G2N)
    return out


def model_series(kw, tw):
    import warp as wp; wp.init()
    from _v2_robo import gpu_run_twist
    r = gpu_run_twist(twist_amp_deg=tw, twist_phase_deg=90.0, nc=NC, ns=NS, n_cycle=NCYC,
                      steps_per_cycle=SPC, wake_rows=SPC, **COND, **kw)
    last = slice((NCYC - 1) * SPC, NCYC * SPC)
    Fz = 2.0 * sum(np.asarray(r[k], float) for k in ZCH)[last]     # x2: channel series are HALF-wing sums
    Fx = 2.0 * sum(np.asarray(r[k], float) for k in XCH)[last]
    ca, sa = np.cos(np.radians(COND['aoa_deg'])), np.sin(np.radians(COND['aoa_deg']))
    dpar = kw.get('d_para', 0.0) * (COND['U'] / 8.0) ** 2          # d_para is already a both-wings total
    Fx = Fx + dpar * ca; Fz = Fz + dpar * sa
    T = -(Fx * ca + Fz * sa); L = Fz * ca - Fx * sa
    # de-spike (single-step near-singular DVM artifacts), same spirit as production _robmean
    def clip8(a):
        m = np.median(a); mad = np.median(np.abs(a - m)) + 1e-12
        return np.clip(a, m - 8 * 1.4826 * mad, m + 8 * 1.4826 * mad)
    return clip8(T), clip8(L)


def butter8(sig, freq):
    """Same conditioning as the experiment: 5th-order Butterworth low-pass, 8 Hz cutoff (datav2.md).
    Periodic signal -> filtfilt over 3 tiled cycles, keep the middle one."""
    from scipy.signal import butter, filtfilt
    n = len(sig); fs = n * freq                       # samples per second (n per cycle x freq)
    b, a = butter(5, 8.0 / (fs / 2.0))
    return filtfilt(b, a, np.tile(sig, 3))[n:2 * n]


def main():
    global NC, SPC
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default='K0,H4,H9')
    ap.add_argument('--nc', type=int, default=NC)
    a = ap.parse_args(); names = a.models.split(',')
    if a.nc != NC:
        NC = a.nc; SPC = int(round(15.0 * COND['U'] * NC / COND['freq'] / 60.0)) * 60
        print(f'[nc{NC}] spc={SPC} (purge fig16_series.npz for clean nc-tagged cache)')
    E = parse_fig16()
    TWS = [0.0, 22.5, 45.0]
    sim = {}; RAW = os.path.join(OD, f'fig16_series_nc{NC}.npz')
    old = dict(np.load(RAW)) if os.path.exists(RAW) else {}                  # resumable across script edits
    for nm in names:
        for tw in TWS:
            kT, kL = f'{nm}_{tw:g}_T', f'{nm}_{tw:g}_L'
            if kT in old: sim[(nm, tw)] = (old[kT], old[kL]); continue
            print(f'[run] {nm} tw{tw:g}', flush=True)
            sim[(nm, tw)] = model_series(MODELS[nm], tw)
            old[kT], old[kL] = sim[(nm, tw)]; np.savez(RAW, **old)           # incremental save
    for (nm, tw), (T, L) in list(sim.items()):                               # exp was Butterworth-filtered: match it
        sim[(nm, tw)] = (butter8(T, COND['freq']), butter8(L, COND['freq']))
    tphase = (np.arange(SPC) + 0.5) / SPC
    # ---- phase alignment on tw0 LIFT of the FIRST model: one shift for everything ----
    te, fe = E[('L', 0.0)]
    exp0 = np.interp(tphase, te, fe, period=1.0)
    L0 = sim[(names[0], 0.0)][1]
    xc = [np.dot(np.roll(L0, s), exp0 - exp0.mean()) for s in range(SPC)]
    shift = int(np.argmax(xc))
    print(f'phase shift (tw0 lift xcorr, {names[0]}): {shift}/{SPC} steps = {shift/SPC:.3f} T')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    COLS = {'K0': '#1b9e77', 'H4': '#d95f02', 'H9': '#7570b3', 'H10': '#e7298a', 'H11': '#66a61e',
            'K0s': '#a6d854', 'H4s': '#e78ac3', 'H12': '#386cb0', 'H13': '#bf5b17', 'K1': '#f0027f', 'K1g': '#666666', 'H14': '#1f78b4', 'H15': '#b2df8a', 'H16': '#ff7f00', 'H17': '#cab2d6'}
    stats = []
    for j, tw in enumerate(TWS):
        for i, kind in enumerate(('T', 'L')):
            axx = ax[i, j]
            tt, ff = E[(kind, tw)]
            axx.plot(tt, ff, 'k-', lw=2.2, label='exp')
            axx.axhline(ff.mean(), color='k', ls=':', lw=0.8)
            expi = np.interp(tphase, tt, ff, period=1.0)
            for nm in names:
                s = np.roll(sim[(nm, tw)][0 if kind == 'T' else 1], shift)
                axx.plot(tphase, s, color=COLS.get(nm, 'gray'), lw=1.3, label=nm)
                axx.axhline(s.mean(), color=COLS.get(nm, 'gray'), ls=':', lw=0.8)
                stats.append((kind, tw, nm, float(np.sqrt(np.mean((s - expi) ** 2))),
                              float(s.mean() - expi.mean())))
            axx.set_title(f'Fig16{"a" if kind == "T" else "b"} {kind} tw{tw:g}°'); axx.grid(alpha=0.3)
            if i == 0 and j == 0: axx.legend(fontsize=8)
            axx.set_xlabel('t/T'); axx.set_ylabel(f'{kind} (N)')
    fig.suptitle('Fig16 instantaneous exp vs models  [8 m/s, AoA5, 2 Hz]  (one global phase shift)', fontsize=12)
    fig.tight_layout(); p = os.path.join(OD, 'fig16_compare.png'); fig.savefig(p, dpi=115); plt.close(fig)
    print(f'saved {p}')
    print(f"{'kind':>4} {'tw':>5} {'model':>6} {'RMSE(N)':>8} {'dMean(N)':>9}")
    for kind, tw, nm, rmse, dm in stats:
        print(f'{kind:>4} {tw:>5g} {nm:>6} {rmse:>8.2f} {dm:>+9.2f}')
    json.dump({f'{k}_{tw:g}_{nm}': dict(rmse=r, dmean=d) for k, tw, nm, r, d in stats},
              open(os.path.join(OD, 'fig16_stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
