"""Overlay exp vs K0 vs H4 (best balanced) for Fig17/18/19. H2 skipped (runaway at AoA15+twist).
Reads docs/repro_data.json + the three caches directly (no global-CFG coupling).
"""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
for f in ['Noto Sans CJK SC', 'Noto Sans CJK HK', 'AR PL UMing CN']:
    try:
        font_manager.findfont(f, fallback_to_default=False); rcParams['font.sans-serif'] = [f]; break
    except Exception: pass
rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, 'docs')
OD = os.path.join(DOCS, 'repro_nc12')
import importlib.util
spec = importlib.util.spec_from_file_location("rr", os.path.join(HERE, '_v2_repro_nc12.py'))
rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)

R = json.load(open(os.path.join(DOCS, 'repro_data.json')))
CACHES = {
    'K0': json.load(open(os.path.join(OD, 'cache_nc4_cyc3_fix.json'))),
    'H4': json.load(open(os.path.join(OD, 'cache_nc4_cyc3_H4.json'))),
    'H16': json.load(open(os.path.join(OD, 'cache_nc4_cyc3_H16.json'))) if os.path.exists(os.path.join(OD, 'cache_nc4_cyc3_H16.json')) else {},
    'H19': json.load(open(os.path.join(OD, 'cache_nc4_cyc3_H19.json'))) if os.path.exists(os.path.join(OD, 'cache_nc4_cyc3_H19.json')) else {},
    'H20': json.load(open(os.path.join(OD, 'cache_nc4_cyc3_H20.json'))) if os.path.exists(os.path.join(OD, 'cache_nc4_cyc3_H20.json')) else {},
}
COL = {'exp': 'k', 'K0': '#1b9e77', 'H4': '#d95f02', 'H16': '#1f78b4', 'H19': '#e31a1c', 'H20': '#6a3d9a'}
LW  = {'exp': 2.4, 'K0': 1.3, 'H4': 1.3, 'H16': 1.5, 'H19': 1.3, 'H20': 1.5}

def pred_line(cache, key):
    kind = R[key]['kind']; out = []
    for xi in R[key]['x']:
        U, aoa, freq, tw = rr.cond_of(key, xi); v = cache.get(rr.ckey(U, aoa, freq, tw))
        out.append(np.nan if (v is None or not np.isfinite(v[0])) else v[0 if kind == 'L' else 1])
    return np.array(out, float)

def mae(a, b):
    m = np.isfinite(a) & np.isfinite(b); return float(np.mean(np.abs(a[m]-b[m]))) if m.sum() else float('nan')

def draw(ax, keys, title, xlabel):
    n = {c: len(CACHES.get(c, {})) for c in ('H16', 'H19', 'H20')}
    for i, k in enumerate(keys):
        if k not in R: continue
        x = R[k]['x']; exp = np.array(R[k]['exp'], float)
        shade = i == 0
        ax.plot(x, exp, '-', color=COL['exp'], lw=LW['exp'], alpha=0.9, label='实测' if shade else None)
        for c in ('K0', 'H16', 'H19', 'H20'):
            if c in n and n[c] == 0: continue
            p = pred_line(CACHES[c], k)
            ls = '--' if c == 'K0' else (':' if c == 'H16' else ('-' if c in ('H19','H20') else '-.'))
            lab = f'{c} ({n[c]}/122)' if (c in n and shade) else (c if shade else None)
            ax.plot(x, p, ls, color=COL[c], lw=LW[c], alpha=0.9, label=lab)
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel); ax.set_ylabel('力 (N)'); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=5)

# ---- Fig17: thrust(a)+lift(b) vs twist, 5 freqs (8m/s AoA5) ----
FREQS = [1.4, 1.7, 2.0, 2.3, 2.6]
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for f in FREQS:
    draw(ax[0], [f'17|a|{f}'], '', ''); draw(ax[1], [f'17|b|{f}'], '', '')
ax[0].set_title('Fig17a 净推力 vs 扭转 (每频率)'); ax[1].set_title('Fig17b 升力 vs 扭转 (每频率)')
for a in ax: a.set_xlabel('扭转角 (deg)')
fig.suptitle('Fig17  实测(黑实线) vs K0(绿虚=kelvin基线) vs H4(橙点划=hirato) vs H16(蓝实点=涡冲量)  [8m/s AoA5]', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, 'compare_fig17.png'), dpi=120); plt.close(fig)

# ---- Fig18: thrust(a)+lift(b) vs freq, 3 winds (twist0, AoA5) ----
WINDS = [6.0, 8.0, 10.0]
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for U in WINDS:
    draw(ax[0], [f'18|a|{U}'], '', ''); draw(ax[1], [f'18|b|{U}'], '', '')
ax[0].set_title('Fig18a 净推力 vs 频率 (每风速)'); ax[1].set_title('Fig18b 升力 vs 频率 (每风速)')
for a in ax: a.set_xlabel('频率 (Hz)')
fig.suptitle('Fig18  实测(黑) vs K0(绿虚) vs H4(橙点划) vs H16(蓝点) vs H19(红实)  [扭转0, AoA5]', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, 'compare_fig18.png'), dpi=120); plt.close(fig)

# ---- Fig19: 2x2  a/b vs freq @tw0, c/d vs twist @2.6Hz, 4 AoAs (8m/s) ----
AOAS = [0, 5, 10, 15]
fig, ax = plt.subplots(2, 2, figsize=(16, 11))
for aoa in AOAS:
    draw(ax[0,0], [f'19|a|{aoa:g}'], '', ''); draw(ax[0,1], [f'19|b|{aoa:g}'], '', '')
    draw(ax[1,0], [f'19|c|{aoa:g}'], '', ''); draw(ax[1,1], [f'19|d|{aoa:g}'], '', '')
ax[0,0].set_title('Fig19a 净推力 vs 频率 (扭转0)'); ax[0,1].set_title('Fig19b 升力 vs 频率 (扭转0)')
ax[1,0].set_title('Fig19c 净推力 vs 扭转 (2.6Hz)'); ax[1,1].set_title('Fig19d 升力 vs 扭转 (2.6Hz)')
ax[0,0].set_xlabel('频率 (Hz)'); ax[0,1].set_xlabel('频率 (Hz)')
ax[1,0].set_xlabel('扭转角 (deg)'); ax[1,1].set_xlabel('扭转角 (deg)')
for a in ax.flat: a.set_ylabel('力 (N)')
fig.suptitle('Fig19  实测(黑) vs K0(绿虚) vs H4(橙点划) vs H16(蓝实点)  [8m/s, 每攻角] — 攻角泛化轴', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, 'compare_fig19.png'), dpi=120); plt.close(fig)

# ---- Summary MAE bar (L/T split) ----
groups = {'Fig17': [k for k in R if k.startswith('17')],
          'Fig18': [k for k in R if k.startswith('18')],
          'Fig19': [k for k in R if k.startswith('19')]}
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
labels = list(groups.keys()); x = np.arange(len(labels))
models_bar = [c for c in ('K0', 'H4', 'H16') if c == 'K0' or c == 'H4' or len(CACHES.get('H16', {})) > 0]
w = 0.8 / len(models_bar)
for idx, c in enumerate(models_bar):
    Lm = []; Tm = []
    for g in labels:
        la = []; ta = []
        for k in groups[g]:
            exp = np.array(R[k]['exp'], float); p = pred_line(CACHES[c], k)
            m = np.isfinite(exp) & np.isfinite(p)
            if R[k]['kind'] == 'L' and m.sum(): la.append(mae(p, exp))
            elif R[k]['kind'] == 'T' and m.sum(): ta.append(mae(p, exp))
        Lm.append(np.mean(la) if la else np.nan); Tm.append(np.mean(ta) if ta else np.nan)
    off = (idx - (len(models_bar) - 1) / 2) * w
    ax[0].bar(x + off, Lm, w, color=COL[c], label=f'{c}' + (f' ({len(CACHES[c])}/122)' if c == 'H16' else ''))
    ax[1].bar(x + off, Tm, w, color=COL[c], label=f'{c}' + (f' ({len(CACHES[c])}/122)' if c == 'H16' else ''))
ax[0].set_title('升力 MAE (N) 越低越好'); ax[1].set_title('推力 MAE (N) 越低越好')
for a in ax: a.set_xticks(x); a.set_xticklabels(labels); a.grid(alpha=0.3, axis='y'); a.legend()
fig.suptitle('按图分组 MAE:K0 vs H4 vs H16(H16 仅部分工况)', fontsize=11); fig.tight_layout()
fig.savefig(os.path.join(DOCS, 'compare_mae.png'), dpi=120); plt.close(fig)

print("saved: compare_fig17/18/19.png + compare_mae.png")
# print MAE table
print(f"\n{'group':8} {'K0_L':>6} {'H4_L':>6} {'K0_T':>6} {'H4_T':>6}")
for g in labels:
    la_k=[]; la_h=[]; ta_k=[]; ta_h=[]
    for k in groups[g]:
        exp=np.array(R[k]['exp'],float)
        if R[k]['kind']=='L': la_k.append(mae(pred_line(CACHES['K0'],k),exp)); la_h.append(mae(pred_line(CACHES['H4'],k),exp))
        else: ta_k.append(mae(pred_line(CACHES['K0'],k),exp)); ta_h.append(mae(pred_line(CACHES['H4'],k),exp))
    print(f"{g:8} {np.mean(la_k):6.2f} {np.mean(la_h):6.2f} {np.mean(ta_k):6.2f} {np.mean(ta_h):6.2f}")
