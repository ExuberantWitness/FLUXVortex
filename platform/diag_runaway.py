"""P0 runaway diagnosis: H2 blows up ONLY at 8/15/2.6/tw45 (L=+555/T=+2109 in cache); H4 (a0_crit
0.27) is stable there. Single-variable matrix to locate the mechanism (cap self-growth vs S
conditioning vs LEV-ring near-field feedback vs geo_stall damping):

  python diag_runaway.py --variant H2        # reproduce the runaway (baseline)
  python diag_runaway.py --variant H2gs      # + geo_stall ON        (damping hypothesis)
  python diag_runaway.py --variant H2it3     # lev_iter=3            (S under-convergence hypothesis)
  python diag_runaway.py --variant H2core08  # lev_core_ring 0.4->0.8 (near-field feedback hypothesis)
  python diag_runaway.py --variant H4        # a0_crit 0.27 stable reference

Captures the per-step HIRATO_DBG stream (|A0pre|max, |gL|max, shed count) in-process plus the
returned per-step force channels, prints a per-cycle summary + takeoff step, saves docs/diag/*.json.
"""
import sys, os, io, re, json, argparse, numpy as np
os.environ['HIRATO_DBG'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
OD = os.path.join(HERE, 'docs', 'diag'); os.makedirs(OD, exist_ok=True)
from _v2_repro_nc12 import CFG_PRESETS, FIX     # reuse the exact sweep presets (no drift)

COND = dict(U=8.0, aoa_deg=15.0, freq=2.6, twist_amp_deg=45.0)   # the ONLY true runaway condition
NC, NS, NCYC = 4, 16, 3
SPC = int(round(15.0 * COND['U'] * NC / COND['freq'] / 60.0)) * 60   # ==180, same as cache run

VARIANTS = {
    'H2':       dict(CFG_PRESETS['H2']),
    'H2gs':     dict(CFG_PRESETS['H2'], **FIX),
    'H2it3':    dict(CFG_PRESETS['H2'], lev_iter=3),
    'H2core08': dict(CFG_PRESETS['H2'], lev_core_ring=0.8),
    'H4':       dict(CFG_PRESETS['H4']),
}

DBG_RE = re.compile(r"\[hirato t=\s*(\d+)\] shed=\s*(\d+)/\d+\s+\|A0_pre\|max=([\d.eE+-]+).*?\|gL\|max=([\d.eE+-]+)")


class Tee(io.TextIOBase):                       # capture the HIRATO_DBG stream while still showing progress
    def __init__(self, real): self.real, self.buf, self.skip = real, [], False
    def write(self, s):                          # suppress the 540-line DBG spam on console (kept in buf)
        self.buf.append(s)
        if '[hirato' in s: self.skip = True
        elif self.skip and s == '\n': self.skip = False
        else: self.real.write(s)
        return len(s)
    def flush(self): self.real.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    ap.add_argument('--ncyc', type=int, default=NCYC)
    a = ap.parse_args()
    kw = VARIANTS[a.variant]
    import warp as wp; wp.init()
    from _v2_robo import gpu_run_twist
    print(f"[diag] {a.variant} cond={COND} nc={NC} ns={NS} ncyc={a.ncyc} spc={SPC}", flush=True)
    tee = Tee(sys.stdout); old = sys.stdout; sys.stdout = tee
    try:
        r = gpu_run_twist(nc=NC, ns=NS, n_cycle=a.ncyc, steps_per_cycle=SPC, wake_rows=SPC,
                          twist_phase_deg=90.0, **COND, **kw)
    finally:
        sys.stdout = old
    txt = ''.join(tee.buf)
    steps = [(int(t), int(sh), float(a0), float(gl)) for t, sh, a0, gl in DBG_RE.findall(txt)]
    N = a.ncyc * SPC
    a0m = np.zeros(N); glm = np.zeros(N); shd = np.zeros(N)
    for t, sh, a0v, gl in steps:
        if t < N: a0m[t], glm[t], shd[t] = a0v, gl, sh
    ch = {k: np.asarray(r[k], float) for k in ('Lh_bern', 'Xh_bern', 'Lh_les', 'Xh_les', 'Lh_vtx', 'Xh_vtx', 'Lh')}
    print(f"\n=== {a.variant}: L_wind={r['L_wind']:+.2f} T_wind={r['T_wind']:+.2f} "
          f"(cache H2: +554.66/+2108.54, H4: +22.85/-15.90)")
    print(f"{'cyc':>4} {'max|A0pre|':>11} {'max|gL|':>9} {'shed%':>6} {'max|Lbern|':>11} {'max|Lvtx|':>10} {'max|Lles|':>10}")
    for c in range(a.ncyc):
        s = slice(c * SPC, (c + 1) * SPC)
        print(f"{c:>4} {a0m[s].max():>11.3f} {glm[s].max():>9.3f} {100 * shd[s].mean() / NS:>5.0f}% "
              f"{np.abs(ch['Lh_bern'][s]).max():>11.1f} {np.abs(ch['Lh_vtx'][s]).max():>10.1f} "
              f"{np.abs(ch['Lh_les'][s]).max():>10.1f}")
    # takeoff step: first t where |Lh_bern| exceeds 10x the cycle-0 median
    med0 = np.median(np.abs(ch['Lh_bern'][:SPC])) + 1e-9
    off = np.where(np.abs(ch['Lh_bern']) > 10 * med0)[0]
    print(f"takeoff(|Lbern|>10x cyc0-med {med0:.1f}N): step {off[0] if len(off) else 'NONE'} of {N}")
    if len(off):
        t0 = int(off[0]); lo = max(0, t0 - 6)
        print(f"{'t':>5} {'|A0pre|':>8} {'|gL|':>8} {'shed':>4} {'Lbern':>10} {'Lvtx':>8} {'Lles':>8}  (around takeoff)")
        for t in range(lo, min(t0 + 6, N)):
            print(f"{t:>5} {a0m[t]:>8.3f} {glm[t]:>8.3f} {int(shd[t]):>4} {ch['Lh_bern'][t]:>10.1f} "
                  f"{ch['Lh_vtx'][t]:>8.1f} {ch['Lh_les'][t]:>8.1f}")
    out = dict(variant=a.variant, cond=COND, nc=NC, ns=NS, ncyc=a.ncyc, spc=SPC,
               L_wind=float(r['L_wind']), T_wind=float(r['T_wind']),
               a0max=a0m.tolist(), glmax=glm.tolist(), shed=shd.tolist(),
               **{k: v.tolist() for k, v in ch.items()})
    jp = os.path.join(OD, f"runaway_{a.variant}.json"); json.dump(out, open(jp, 'w'))
    print(f"saved {jp}", flush=True)


if __name__ == '__main__':
    main()
