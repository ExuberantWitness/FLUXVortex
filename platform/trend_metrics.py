"""轨迹相似性趋势记分卡 harness v2(2026-07-25 升级,GAP 修正闭环的发现引擎)。

用户裁定(2026-07-25,入记忆 feedback_rom_law_capture):降阶模型近似实测载荷的核心底线
= **关键规律(轨迹形状)捕获为主**,绝对误差 <10% 或尽可能小,**趋势贴近时 MAE 门可放宽**。
实测规律本非单调,逐点微分斜率(d/dx 符号投票)在近平坦曲线上脆弱——主裁判改为
**轨迹相似性矩阵**,微分斜率降为次要参考。

度量分类:
  形状类(主判):皮尔逊 r / 夹角余弦 / Hausdorff(min-max 归一 (x,y) 点集) /
                标准欧氏 / 马氏(diag 协方差)
  量级类(参考):欧氏 / 曼哈顿 / 切比雪夫 / 闵可(p=3) / BrayCurtis
  不适用(剔除):汉明、杰卡德(集合/分类变量度量,连续轨迹无意义)

趋势捕获判据(每曲线):
  - 实测近平坦(振幅 < 5% 量级):模型振幅 < max(2×实测振幅, 10% 量级) -> 捕获
  - 否则:Pearson r >= 0.5 -> 捕获

六族(轨迹级):dL/df、dT/df(f 扫,19|a/b aoa 族 + 18|a/b U 族)
             dL/dU、dT/dU(跨 U,18 族采样);dL/dAOA、dT/dAOA(跨 aoa,19 族采样)

输入:模型 118 结果 JSON("U_f_tw_aoa" -> {L, T}) + repro_data.json(实测)。
输出:终端记分卡 + 病灶排序(未捕获 > 低 r > MAE)+ 版本化 JSON(docs/scorecards/)。
自检门:对 s6_sweep_v4.json 必须复现四判决(dT/dU 反向;19|b10 dL/df 缺口带;
升力 MAE 带;dL/df 族形状失败 r̄<0.5)——不复现则 harness 有 bug,禁用其结论。

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
TRAJ_KEYS = ["euc", "man", "che", "mink", "se", "mah", "cos", "pr", "bray", "haus"]
TRAJ_HDR = {"euc": "欧氏↓", "man": "曼哈顿↓", "che": "切比雪夫↓", "mink": "闵可p3↓",
            "se": "标准欧氏↓", "mah": "马氏diag↓", "cos": "夹角余弦↑", "pr": "皮尔逊r↑",
            "bray": "BrayCurtis↓", "haus": "Hausdorff↓"}
SHAPE_KEYS = ["pr", "cos", "haus", "se", "mah"]     # 形状类(主判)
LEVEL_KEYS = ["euc", "man", "che", "mink", "bray"]  # 量级类(参考)


def load(sweep_path):
    M = json.load(open(os.path.join(DOCS, "repro_data.json")))
    SW = json.load(open(sweep_path))
    return M, SW


def traj(a, b, xv):
    """轨迹相似性矩阵:model a vs measured b(共同自变量 xv)。返回 10 度量。"""
    a = np.asarray(a, float); b = np.asarray(b, float); xv = np.asarray(xv, float)
    out = {}
    out["euc"] = float(np.linalg.norm(a - b))
    out["man"] = float(np.abs(a - b).sum())
    out["che"] = float(np.abs(a - b).max())
    out["mink"] = float((np.abs(a - b) ** 3).sum() ** (1 / 3))
    sd = np.std(np.concatenate([a, b]), ddof=1) + 1e-9
    out["se"] = float(np.sqrt((((a - b) / sd) ** 2).sum()))
    cov = np.var(np.vstack([a, b]), axis=1, ddof=1) + 1e-9
    out["mah"] = float(np.sqrt((((a - b) ** 2) / np.mean(cov)).sum()))
    out["cos"] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    out["pr"] = (float(np.corrcoef(a, b)[0, 1])
                 if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0)
    out["bray"] = float(np.abs(a - b).sum() / (np.abs(a + b).sum() + 1e-12))
    X = np.vstack([xv, a]).T; Y = np.vstack([xv, b]).T
    allp = np.vstack([X, Y]); rng = np.ptp(allp, axis=0) + 1e-9
    Xn = (X - allp.min(0)) / rng; Yn = (Y - allp.min(0)) / rng

    def hd(A, B):
        return max(max(min(np.linalg.norm(p - B, axis=1)) for p in A),
                   max(min(np.linalg.norm(p - B, axis=1)) for p in B))
    out["haus"] = float(hd(Xn, Yn))
    return out


def captured_of(r, vs, mv):
    """趋势捕获判据:实测近平坦 -> 比振幅;否则 Pearson r>=0.5。"""
    scale = max(np.abs(mv).max(), 1e-9)
    rng_e = float(mv.max() - mv.min()); rng_m = float(vs.max() - vs.min())
    if rng_e < 0.05 * scale:                    # 实测近平坦:符号投票无意义,比振幅
        return bool(rng_m < max(2.0 * rng_e, 0.10 * scale))
    return bool(r >= 0.5)


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
        tm = traj(vs, mv, xs)
        sm = float(np.polyfit(xs, vs, 1)[0]); se_ = float(np.polyfit(xs, mv, 1)[0])
        cap = captured_of(tm["pr"], vs, mv)
        rows.append(dict(curve=mkey, n=len(xs), mae=mae, r=tm["pr"],
                         slope_model=sm, slope_meas=se_,
                         sign_ok=bool(np.sign(sm) == np.sign(se_)),
                         slope_err=float(abs(sm - se_) / (abs(se_) + 1e-9)),
                         captured=cap,
                         traj={k: tm[k] for k in TRAJ_KEYS}))

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

    # 跨族斜率(保留,次要参考)
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

    # ---- 六族轨迹相似性矩阵(主判) ----
    def fam_traj(curves):
        """curves: [(xs, vs, xe, ye)];返回度量均值字典。"""
        agg = {k: [] for k in TRAJ_KEYS}
        for xs, vs, xe, ye in curves:
            if len(xs) < 3:
                continue
            tm = traj(vs, np.interp(xs, xe, ye), xs)
            for k in TRAJ_KEYS:
                agg[k].append(tm[k])
        return {k: float(np.mean(v)) for k, v in agg.items() if v}

    def curve_xy(mkey, pts, chan):
        d = M.get(mkey)
        if d is None:
            return None
        xs, vs = [], []
        for x, cond in pts:
            v = mod(*cond)
            if v is not None:
                xs.append(x); vs.append(v[chan])
        if len(xs) < 3:
            return None
        return (np.array(xs, float), np.array(vs, float),
                np.array(d["x"], float), np.array(d["exp"], float))

    families = {}
    for chan, sub, famL in (("L", "b", "dL/df"), ("T", "a", "dT/df")):
        curves = []
        for a in (0, 5, 10, 15):
            c = curve_xy(f"19|{sub}|{a:g}", [(f, (8, f, 22.5, a)) for f in FS], chan)
            if c:
                curves.append(c)
        for U in (6, 8, 10):
            c = curve_xy(f"18|{sub}|{U}", [(f, (U, f, 22.5, 5)) for f in FS], chan)
            if c:
                curves.append(c)
        families[famL] = fam_traj(curves)
    for chan, sub, famL, xs_ref in (("L", "b", "dL/dU", (6.0, 8.0, 10.0)),
                                    ("T", "a", "dT/dU", (6.0, 8.0, 10.0))):
        curves = []
        for f0 in FS:
            xs, vs, xe, ye = [], [], [], []
            for U in xs_ref:
                v = mod(U, f0, 22.5, 5)
                if v is None:
                    continue
                xs.append(U); vs.append(v[chan]); xe.append(U)
                ye.append(float(np.interp(f0, M[f"18|{sub}|{U}"]["x"],
                                          M[f"18|{sub}|{U}"]["exp"])))
            if len(xs) >= 3:
                curves.append((np.array(xs), np.array(vs), np.array(xe), np.array(ye)))
        families[famL] = fam_traj(curves)
    for chan, sub, famL in (("L", "b", "dL/dAOA"), ("T", "a", "dT/dAOA")):
        curves = []
        aoas = (0.0, 5.0, 10.0, 15.0)
        for f0 in FS:
            xs, vs, xe, ye = [], [], [], []
            for a in aoas:
                v = mod(8, f0, 22.5, a)
                if v is None:
                    continue
                xs.append(a); vs.append(v[chan]); xe.append(a)
                ye.append(float(np.interp(f0, M[f"19|{sub}|{a:g}"]["x"],
                                          M[f"19|{sub}|{a:g}"]["exp"])))
            if len(xs) >= 3:
                curves.append((np.array(xs), np.array(vs), np.array(xe), np.array(ye)))
        families[famL] = fam_traj(curves)

    # 汇总 + 病灶排序(未捕获 > 低 r > MAE)
    Ls = [x for x in rows if "|b|" in x["curve"] or "|d|" in x["curve"]]
    Ts = [x for x in rows if "|a|" in x["curve"] or "|c|" in x["curve"]]
    summary = dict(n_curves=len(rows),
                   lift_mae=float(np.mean([x["mae"] for x in Ls])),
                   lift_rbar=float(np.mean([x["r"] for x in Ls])),
                   thrust_mae=float(np.mean([x["mae"] for x in Ts])),
                   thrust_rbar=float(np.mean([x["r"] for x in Ts])),
                   sign_errors=sum(1 for x in rows if not x["sign_ok"]),
                   sign_error_curves=[x["curve"] for x in rows if not x["sign_ok"]],
                   trend_score=float(np.mean([x["captured"] for x in rows])),
                   trend_capture=float(np.mean([x["captured"] for x in rows])),
                   trend_sign_legacy=float(np.mean([x["sign_ok"] for x in rows])),
                   uncaptured=[x["curve"] for x in rows if not x["captured"]])
    diseases = sorted(rows, key=lambda x: (x["captured"], x["r"], -x["mae"]))

    if not quiet:
        print(f"{'curve':<16}{'n':>3}{'MAE':>7}{'r':>7}{'cos':>7}{'haus':>7}"
              f"{'slp_m':>8}{'slp_e':>8}{'捕获':>5}")
        for x in rows:
            print(f"{x['curve']:<16}{x['n']:>3}{x['mae']:>7.2f}{x['r']:>7.2f}"
                  f"{x['traj']['cos']:>7.2f}{x['traj']['haus']:>7.2f}"
                  f"{x['slope_model']:>8.3f}{x['slope_meas']:>8.3f}"
                  f"{'OK' if x['captured'] else '✗':>5}")
        print(f"\n== 汇总: {summary['n_curves']} 曲线 | 趋势捕获率 "
              f"{summary['trend_capture']:.2f} (旧符号率 {summary['trend_sign_legacy']:.2f}) | "
              f"升力 MAE {summary['lift_mae']:.2f} r̄ {summary['lift_rbar']:+.2f} | "
              f"推力 MAE {summary['thrust_mae']:.2f} r̄ {summary['thrust_rbar']:+.2f} | "
              f"未捕获 {len(summary['uncaptured'])}")
        print("  (门径:规律捕获为主;MAE<10% 为目标,趋势贴近可放宽——用户裁定 2026-07-25)")
        for k, v in cross.items():
            print(f"  {k}(参考): model {v['model']:+.3f} vs meas {v['meas']:+.3f} "
                  f"{'OK' if v['sign_ok'] else '✗'}")
        print("\n== 六族轨迹相似性矩阵(主判) ==")
        for fam, tm in families.items():
            print(f"  --- {fam} ---")
            for k in TRAJ_KEYS:
                print(f"    {TRAJ_HDR[k]:>12} {tm[k]:>9.3f}")
        print("\n== 病灶排序(top 8,未捕获优先):")
        for x in diseases[:8]:
            print(f"  {x['curve']:<16} 捕获={'OK' if x['captured'] else '✗'} "
                  f"r={x['r']:+.2f} cos={x['traj']['cos']:+.2f} haus={x['traj']['haus']:.2f} "
                  f"slope {x['slope_model']:+.3f}/{x['slope_meas']:+.3f} MAE={x['mae']:.2f}")

    out = dict(sweep=os.path.basename(sweep_path), rows=rows, cross=cross,
               families=families, summary=summary,
               diseases=[x["curve"] for x in diseases])
    if tag:
        od = os.path.join(DOCS, "scorecards"); os.makedirs(od, exist_ok=True)
        json.dump(out, open(os.path.join(od, f"scorecard_{tag}.json"), "w"), indent=1)
        if not quiet:
            print(f"\nsaved docs/scorecards/scorecard_{tag}.json")
    return out


def selfcheck(out):
    """自检门:v4 基线必须复现四判决。"""
    ok1 = not out["cross"]["dT_dU"]["sign_ok"]                       # dT/dU 反向
    r10 = [x for x in out["rows"] if x["curve"] == "19|b|10"][0]
    gap = r10["slope_meas"] - r10["slope_model"]
    ok2 = 1.2 < gap < 2.5                                            # dL/df 缺口带
    ok3 = 0.6 < out["summary"]["lift_mae"] < 1.1                     # 升力 MAE 带
    dldf = out["families"]["dL/df"]
    ok4 = dldf["pr"] < 0.5                                           # dL/df 族形状失败(v4 病历)
    print(f"自检门: dT/dU反向={ok1} dL/df缺口={gap:+.2f}∈(1.2,2.5)={ok2} "
          f"liftMAE带={ok3} dL/df形状r̄={dldf['pr']:+.2f}<0.5={ok4} -> "
          f"{'PASS' if (ok1 and ok2 and ok3 and ok4) else 'FAIL'}")
    return ok1 and ok2 and ok3 and ok4


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
