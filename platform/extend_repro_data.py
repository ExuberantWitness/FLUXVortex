"""One-shot: merge Fig19 (thrust/lift vs freq & vs twist at AoA 0/5/10/15, 8 m/s) from docs/data.md into
docs/repro_data.json, so the comparison set covers ALL measured data (Fig17+18+19).

Robustness notes (verified against data.md):
- data.md FIGURE captions for (c)/(d) are SWAPPED vs the actual column headers (18c caption says Lift but the
  column header is 'Thurst/g'); the COLUMN HEADER is authoritative -> kind is parsed from the header line of
  each 工况 block, never from the caption.
- Units: grams-force -> N via g*9.80665e-3, sign preserved (negative thrust = net drag).
- Fig18 thrust provenance warning: the original digitization note assigned the source plot's lower
  curve to U=6 and upper curve to U=10.  This is reversed relative to the PDF legend and正文.
  ``correct_fig18_curve_identity.py`` repairs the measured x/exp identities.  Do not restore the
  obsolete check ``18|a|6.0 == -323g``; the corrected U=6 anchor is about -116g.
- Fig19 fixed params (from _v2_validate_all.FIG_SPEC, condition text in data.md): a/b: wind=8, twist=0, sweep=freq;
  c/d: wind=8, freq=2.6, sweep=twist; aoa from the 工况 header (度).
Key format: '19|a|<aoa>' etc., value {kind:'T'|'L', x:[...], exp:[...]} — matches existing consumer code.

  python extend_repro_data.py          # merge + verify (idempotent; rewrites 19|* keys)
"""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'docs', 'data.md')
JP = os.path.join(HERE, 'docs', 'repro_data.json')
G2N = 9.80665e-3


def parse_fig19():
    blocks = []   # (sub, aoa, kind, [(x, val_g)])
    fig = sub = None; cur = None; expect_hdr = False
    for line in open(DATA, encoding='utf-8'):
        mf = re.search(r"Figure\s+(\d+)\s*\.?\s*\(([a-d])\)", line)
        if mf:
            fig, sub = mf.group(1), mf.group(2); continue
        if '工况' in line and fig == '19':
            ma = re.search(r"([\d.]+)\s*度", line)
            cur = dict(sub=sub, aoa=float(ma.group(1)) if ma else None, kind=None, pts=[])
            blocks.append(cur); expect_hdr = True; continue
        if cur is not None and expect_hdr and re.search(r"/g", line):
            # column header line: '... Thurst/g' or '... lift/g' -> authoritative kind
            cur['kind'] = 'T' if re.search(r"[Tt]h?urst|[Tt]hrust", line) else ('L' if re.search(r"[Ll]ift", line) else None)
            expect_hdr = False; continue
        m = re.match(r"\s*(-?[\d.]+e[+-]\d+)\s+(-?[\d.]+e[+-]\d+)", line)
        if m and cur is not None and fig == '19':
            cur['pts'].append((float(m.group(1)), float(m.group(2))))
    return [b for b in blocks if b['pts'] and b['aoa'] is not None]


def main():
    R = json.load(open(JP))
    blocks = parse_fig19()
    assert blocks, 'no Fig19 blocks parsed'
    n_new = 0
    for b in blocks:
        assert b['kind'] in ('T', 'L'), f"block sub={b['sub']} aoa={b['aoa']}: kind not found in column header"
        key = f"19|{b['sub']}|{b['aoa']:g}"
        R[key] = dict(kind=b['kind'], x=[p[0] for p in b['pts']], exp=[p[1] * G2N for p in b['pts']])
        n_new += 1
        print(f"  {key}: kind={b['kind']} n={len(b['pts'])} x=[{b['pts'][0][0]:.2f}..{b['pts'][-1][0]:.2f}] "
              f"exp=[{b['pts'][0][1]*G2N:+.2f}..{b['pts'][-1][1]*G2N:+.2f}]N")
    json.dump(R, open(JP, 'w'))
    print(f"merged {n_new} Fig19 curves into {os.path.basename(JP)} ({len(R)} keys total)")
    # spot checks vs data.md raw values
    a = R['19|a|15']; assert a['kind'] == 'T' and abs(a['exp'][0] - (-386.206896551724 * G2N)) < 1e-6
    print(f"  spot1 19|a|15 exp[0]={a['exp'][0]:+.3f}N (=-386.2g) OK")
    d = R['19|d|0']; assert d['kind'] in ('T', 'L')
    print(f"  spot2 19|d|0 kind={d['kind']} (from column header, caption ignored)")
    b19 = R['19|b|5']
    print(f"  spot3 19|b|5 kind={b19['kind']} exp range [{min(b19['exp']):+.2f},{max(b19['exp']):+.2f}]N "
          f"(lift at aoa5 should be positive ~2-8N)")


if __name__ == '__main__':
    main()
