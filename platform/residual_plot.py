"""Thrust-axis RESIDUAL analysis: model - experiment, how it varies with operating condition.
Lift is already accurate (~1N); the thrust/drag axis is the open problem. This plots the residual
STRUCTURE (vs twist, freq, aoa, U, and vs the measured value) to expose the systematic error.

  python residual_plot.py                       # K0, H4 (full), H16 (partial) thrust residuals
  python residual_plot.py --models fix,H4 --kind L   # lift instead
"""
import os, json, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
DOCS = os.path.join(HERE, 'docs')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
for f in ['Noto Sans CJK SC', 'Noto Sans CJK HK', 'AR PL UMing CN']:
    try: font_manager.findfont(f, fallback_to_default=False); rcParams['font.sans-serif'] = [f]; break
    except Exception: pass
rcParams['axes.unicode_minus'] = False
from _v2_repro_nc12 import cond_of, ckey

NAMES = {'fix': 'K0', 'H4': 'H4', 'H16': 'H16', 'H13': 'H13'}
COL = {'K0': '#1b9e77', 'H4': '#d95f02', 'H16': '#1f78b4', 'H13': '#bf5b17'}


def load_pts(model, kind):
    R = json.load(open(os.path.join(DOCS, 'repro_data.json')))
    C = json.load(open(os.path.join(DOCS, 'repro_nc12', f'cache_nc4_cyc3_{model}.json')))
    pts = []
    for key in R:
        if R[key]['kind'] != kind: continue
        for xi, e in zip(R[key]['x'], R[key]['exp']):
            U, aoa, f, tw = cond_of(key, xi); ck = ckey(U, aoa, f, tw)
            v = C.get(ck)
            if v is None or not np.isfinite(v[0]): continue
            pts.append((U, aoa, f, tw, e, v[1] if kind == 'T' else v[0]))  # (U,aoa,f,tw,exp,model)
    return np.array(pts, float)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default='fix,H4,H16'); ap.add_argument('--kind', default='T')
    a = ap.parse_args()
    models = [m for m in a.models.split(',') if os.path.exists(os.path.join(DOCS, 'repro_nc12', f'cache_nc4_cyc3_{m}.json'))]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    summary = []
    for m in models:
        P = load_pts(m, a.kind)
        if len(P) == 0: continue
        nm = NAMES.get(m, m); c = COL.get(nm, 'gray')
        U, aoa, f, tw, exp, mod = P.T
        res = mod - exp                                                    # residual = model - experiment
        # stats
        mean, std = float(np.mean(res)), float(np.std(res))
        mae = float(np.mean(np.abs(res)))
        sign_agree = float(np.mean(np.sign(mod) == np.sign(exp)))          # % same sign (net-thrust direction)
        sys_bias = mean                                                    # systematic: model over- (+) / under- (-) predicts
        summary.append((nm, len(P), mean, std, mae, sign_agree))
        # ---- 6 panels ----
        ax[0, 0].scatter(tw, res, s=14, color=c, alpha=0.6, label=f'{nm} ({len(P)})')
        ax[0, 1].scatter(f, res, s=14, color=c, alpha=0.6)
        ax[0, 2].scatter(aoa, res, s=14, color=c, alpha=0.6)
        ax[1, 0].scatter(U, res, s=14, color=c, alpha=0.6)
        ax[1, 1].scatter(exp, mod, s=14, color=c, alpha=0.6)
        ax[1, 2].hist(res, bins=30, range=(-15, 15), color=c, alpha=0.4, density=True)
    ax[0, 0].axhline(0, color='k', lw=0.8); ax[0, 0].set_xlabel('扭转 (deg)'); ax[0, 0].set_ylabel(f'{a.kind} 残差 (model−exp, N)')
    ax[0, 0].set_title('残差 vs 扭转'); ax[0, 0].legend(fontsize=8)
    for a_ in ax[0, 1:]: a_.axhline(0, color='k', lw=0.8)
    ax[0, 1].set_xlabel('频率 (Hz)'); ax[0, 1].set_title('残差 vs 频率')
    ax[0, 2].set_xlabel('攻角 (deg)'); ax[0, 2].set_title('残差 vs 攻角')
    ax[1, 0].axhline(0, color='k', lw=0.8); ax[1, 0].set_xlabel('风速 (m/s)'); ax[1, 0].set_ylabel(f'{a.kind} 残差 (N)'); ax[1, 0].set_title('残差 vs 风速')
    lo = min(-0.5, float(exp.min()) - 1) if len(models) else -1; hi = max(0.5, float(exp.max()) + 1) if len(models) else 1
    ax[1, 1].plot([lo, hi], [lo, hi], 'k--', lw=0.8); ax[1, 1].set_xlabel(f'实测 {a.kind} (N)'); ax[1, 1].set_ylabel(f'模型 {a.kind} (N)'); ax[1, 1].set_title('模型 vs 实测(对角=完美)')
    ax[1, 1].axhline(0, color='gray', lw=0.5); ax[1, 2].set_xlabel(f'{a.kind} 残差 (N)'); ax[1, 2].set_title('残差直方(0=完美)')
    ax[1, 2].axvline(0, color='k', lw=0.8)
    fig.suptitle(f'{"推力/阻力" if a.kind == "T" else "升力"} 残差结构分析  (残差 = 模型 − 实测;  正=模型高估, 负=低估)', fontsize=12)
    fig.tight_layout(); p = os.path.join(DOCS, f'residual_{a.kind}.png'); fig.savefig(p, dpi=115); plt.close(fig)
    print(f'saved {p}\n')
    print(f"{'model':>6} {'n':>4} {'均值(系统偏置)':>14} {'std':>6} {'MAE':>6} {'符号一致率':>10}")
    for nm, n, mean, std, mae, sg in summary:
        print(f"{nm:>6} {n:>4} {mean:>+14.2f} {std:>6.2f} {mae:>6.2f} {100*sg:>9.0f}%")
    print('\n读法:均值>0 = 模型系统性高估(给的阻力/推力偏正);均值<0 = 系统性低估(偏负=净阻力给多了)。')
    print('符号一致率 = 模型与实测同号(都推力或都阻力)的比例;推力轴 <50% 说明连方向都错。')


if __name__ == '__main__':
    main()
