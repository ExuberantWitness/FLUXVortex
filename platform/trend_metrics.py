"""四轴趋势记分卡 harness(GAP 修正闭环的发现引擎,计划 2026-07-19 v2 T1)。

输入:模型 118 结果 JSON("U_f_tw_aoa" -> {L, T},v4eff 口径:推力已含显式去皮)
     + repro_data.json(实测曲线;映射与 p2_s6_fig_sweep.py 权威口径一致)。
每曲线指标:斜率符号对错(一票判)、Pearson r、斜率相对误差、MAE。
跨族斜率:dT/dU、dL/dU(Fig18 跨速)、dL/daoa、dT/daoa(Fig19 跨攻角)。
输出:终端记分卡 + 病灶排序表(选题依据:符号错 > |斜率误差| > 低 r > MAE)
     + 版本化 JSON(docs/scorecards/scorecard_<tag>.json)。
自检门:对 s6_sweep_v4eff.json 必须复现三判决(dT/dU 符号反;aoa10 dL/df 实测-模型
差 ~1.5-1.9 N/Hz;升力族 MAE≈0.8)——不复现则 harness 有 bug,禁用其结论。

用法:python platform/trend_metrics.py <sweep.json> [tag]
"""
import ast
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
TWS = [0, 5, 10, 15, 20, 22.5, 25, 27.5, 30, 35, 40, 45]
FS = [1.4, 1.7, 2.0, 2.3, 2.6]


def load(sweep_path):
    M = json.load(open(os.path.join(DOCS, "repro_data.json")))
    SW = json.load(open(sweep_path))
    return M, SW


def scorecard(sweep_path, tag=None, quiet=False):
    M, SW = load(sweep_path)

    def mod(U, f, tw, aoa):
        v = SW.get(f"{U:g}_{f:g}_{tw:g}_{aoa:g}")
        return None if (v is None or "fail" in v) else v

    rows = []

    def ev(mkey, pts, chan):
        d = M.get(mkey)
        if d is None:
            return
        xs, vs = [], []
        for x, cond in pts:
            v = mod(*cond)
            if v is not None:
                xs.append(x); vs.append(v[chan])
        if len(xs) < 3:
            return
        xs = np.array(xs, float); vs = np.array(vs, float)
        mv = np.interp(xs, d["x"], d["exp"])
        mae = float(np.mean(np.abs(vs - mv)))
        r = (float(np.corrcoef(vs, mv)[0, 1])
             if np.std(vs) > 1e-9 and np.std(mv) > 1e-9 else 0.0)
        sm = float(np.polyfit(xs, vs, 1)[0]); se = float(np.polyfit(xs, mv, 1)[0])
        rows.append(dict(curve=mkey, n=len(xs), mae=mae, r=r, slope_model=sm,
                         slope_meas=se, sign_ok=bool(np.sign(sm) == np.sign(se)),
                         slope_err=float(abs(sm - se) / (abs(se) + 1e-9))))

    for f in FS:
        ev(f"17|a|{f:.1f}", [(tw, (8, f, tw, 5)) for tw in TWS], "T")
        ev(f"17|b|{f:.1f}", [(tw, (8, f, tw, 5)) for tw in TWS], "L")
    for U in (6.0, 8.0, 10.0):
        ev(f"18|a|{U}", [(f, (U, f, 22.5, 5)) for f in FS], "T")
        ev(f"18|b|{U}", [(f, (U, f, 22.5, 5)) for f in FS], "L")
    for k in sorted(M):
        if k.startswith("18|c|") or k.startswith("18|d|"):
            w, f = ast.literal_eval(k.split("|")[2])
            ev(k, [(tw, (float(w), float(f), tw, 5)) for tw in TWS],
               "T" if "|c|" in k else "L")
    for a in (0.0, 5.0, 10.0, 15.0):
        ev(f"19|a|{a:g}", [(f, (8, f, 22.5, a)) for f in FS], "T")
        ev(f"19|b|{a:g}", [(f, (8, f, 22.5, a)) for f in FS], "L")
        ev(f"19|c|{a:g}", [(tw, (8, 2.6, tw, a)) for tw in TWS], "T")
        ev(f"19|d|{a:g}", [(tw, (8, 2.6, tw, a)) for tw in TWS], "L")

    # 跨族斜率
    cross = {}
    for chan, tag2, sub, xs_ref in (("T", "dT_dU", "a", (6.0, 8.0, 10.0)),
                                    ("L", "dL_dU", "b", (6.0, 8.0, 10.0))):
        sm_l, se_l = [], []
        for f in FS:
            vs = [mod(U, f, 22.5, 5) for U in xs_ref]
            if any(v is None for v in vs):
                continue
            vs = [v[chan] for v in vs]
            me = [np.interp(f, M[f"18|{sub}|{U}"]["x"], M[f"18|{sub}|{U}"]["exp"])
                  for U in xs_ref]
            sm_l.append(np.polyfit(xs_ref, vs, 1)[0])
            se_l.append(np.polyfit(xs_ref, me, 1)[0])
        cross[tag2] = dict(model=float(np.mean(sm_l)), meas=float(np.mean(se_l)),
                           sign_ok=bool(np.sign(np.mean(sm_l)) == np.sign(np.mean(se_l))))
    for chan, tag2, sub in (("T", "dT_daoa", "a"), ("L", "dL_daoa", "b")):
        sm_l, se_l = [], []
        aoas = (0.0, 5.0, 10.0, 15.0)
        for f in FS:
            vs = [mod(8, f, 22.5, a) for a in aoas]
            if any(v is None for v in vs):
                continue
            vs = [v[chan] for v in vs]
            me = [np.interp(f, M[f"19|{sub}|{a:g}"]["x"], M[f"19|{sub}|{a:g}"]["exp"])
                  for a in aoas]
            sm_l.append(np.polyfit(aoas, vs, 1)[0])
            se_l.append(np.polyfit(aoas, me, 1)[0])
        cross[tag2] = dict(model=float(np.mean(sm_l)), meas=float(np.mean(se_l)),
                           sign_ok=bool(np.sign(np.mean(sm_l)) == np.sign(np.mean(se_l))))

    # 汇总 + 病灶排序(符号错 > 斜率误差 > 低 r > MAE)
    Ls = [x for x in rows if "|b|" in x["curve"] or "|d|" in x["curve"]]
    Ts = [x for x in rows if "|a|" in x["curve"] or "|c|" in x["curve"]]
    summary = dict(n_curves=len(rows),
                   lift_mae=float(np.mean([x["mae"] for x in Ls])),
                   lift_rbar=float(np.mean([x["r"] for x in Ls])),
                   thrust_mae=float(np.mean([x["mae"] for x in Ts])),
                   thrust_rbar=float(np.mean([x["r"] for x in Ts])),
                   sign_errors=sum(1 for x in rows if not x["sign_ok"]),
                   sign_error_curves=[x["curve"] for x in rows if not x["sign_ok"]],
                   trend_score=float(np.mean([x["sign_ok"] for x in rows])))
    diseases = sorted(rows, key=lambda x: (x["sign_ok"], -x["slope_err"], x["r"], -x["mae"]))

    if not quiet:
        print(f"{'curve':<16}{'n':>3}{'MAE':>7}{'r':>7}{'slp_m':>8}{'slp_e':>8}{'sign':>6}")
        for x in rows:
            print(f"{x['curve']:<16}{x['n']:>3}{x['mae']:>7.2f}{x['r']:>7.2f}"
                  f"{x['slope_model']:>8.3f}{x['slope_meas']:>8.3f}"
                  f"{'OK' if x['sign_ok'] else '✗':>6}")
        print(f"\n== 汇总: {summary['n_curves']} 曲线 | 趋势得分(符号正确率) "
              f"{summary['trend_score']:.2f} | 升力 MAE {summary['lift_mae']:.2f} "
              f"r̄ {summary['lift_rbar']:+.2f} | 推力 MAE {summary['thrust_mae']:.2f} "
              f"r̄ {summary['thrust_rbar']:+.2f} | 符号错 {summary['sign_errors']}")
        for k, v in cross.items():
            print(f"  {k}: model {v['model']:+.3f} vs meas {v['meas']:+.3f} "
                  f"{'OK' if v['sign_ok'] else '✗'}")
        print("\n== 病灶排序(top 8):")
        for x in diseases[:8]:
            print(f"  {x['curve']:<16} sign={'OK' if x['sign_ok'] else '✗'} "
                  f"slope {x['slope_model']:+.3f}/{x['slope_meas']:+.3f} "
                  f"r={x['r']:+.2f} MAE={x['mae']:.2f}")

    out = dict(sweep=os.path.basename(sweep_path), rows=rows, cross=cross,
               summary=summary, diseases=[x["curve"] for x in diseases])
    if tag:
        od = os.path.join(DOCS, "scorecards"); os.makedirs(od, exist_ok=True)
        json.dump(out, open(os.path.join(od, f"scorecard_{tag}.json"), "w"), indent=1)
        if not quiet:
            print(f"\nsaved docs/scorecards/scorecard_{tag}.json")
    return out


def selfcheck(out):
    """自检门:v4 基线必须复现三判决。"""
    ok1 = not out["cross"]["dT_dU"]["sign_ok"]                       # dT/dU 反向
    r10 = [x for x in out["rows"] if x["curve"] == "19|b|10"][0]
    gap = r10["slope_meas"] - r10["slope_model"]
    ok2 = 1.2 < gap < 2.5                                            # dL/df 缺口带
    ok3 = 0.6 < out["summary"]["lift_mae"] < 1.1                     # 升力 MAE 带
    print(f"自检门: dT/dU反向={ok1} dL/df缺口={gap:+.2f}∈(1.2,2.5)={ok2} "
          f"liftMAE带={ok3} -> {'PASS' if (ok1 and ok2 and ok3) else 'FAIL'}")
    return ok1 and ok2 and ok3


if __name__ == "__main__":
    # 默认口径 = 生产真身 s6_sweep_v4.json(d_para=0.5,机构钝体物理值 + Blasius visc,
    # 滑翔锚验证成立,gap_t3_dpara.md)。s6_sweep_v4eff.json(d_para=3.0)是 T3 案卷
    # 判废的补偿显示口径:它把 dT/dU 从生产真身的 −0.195 虚放大到 −0.82(纯 U² 惩罚
    # 伪影,非模型缺陷)——不作趋势评估基线,仅供旧图对照。
    default = os.path.join(DOCS, "s6_sweep_v4.json")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    out = scorecard(path, tag=tag)
    if os.path.basename(path) == "s6_sweep_v4.json":
        selfcheck(out)
