"""UIUC 实测极曲线 CD(alpha, Re) 查表(零拟合;u-trend 候选 1b 实现,research_utrend.md §4.2)。

数据:researchpaper/uiuc_polars/SD7003.DRG(UIUC LSAT Vol.1 原始风洞实测,
Selig et al.;Re 块 61k/102k/202k/303k)。薄弯度截面代理(本翼膜+圆管前缘,
文献无直接极曲线;代理选择与外推域在 constants_registry 登记)。
插值:alpha 线性 × logRe 线性;超出表域 CLAMP 到边界(不做外推拟合——
后失速段表内无数据,壁垒如实保留;若门 B 因此不达 → 按报告回落到候选 2)。"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRG = os.path.join(_HERE, "..", "researchpaper", "uiuc_polars", "SD7003.DRG")


def _parse(path=_DRG):
    blocks = []
    with open(path) as fh:
        lines = fh.read().split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Average Reynolds"):
            re_val = float(lines[i + 1].strip())
            j = i + 2
            while not lines[j].strip().startswith("alpha"):
                j += 1
            rows = []
            j += 1
            while j < len(lines):
                parts = lines[j].split()
                try:
                    vals = [float(x) for x in parts[:3]]
                    if len(vals) == 3:
                        rows.append(vals)
                        j += 1
                        continue
                except ValueError:
                    pass
                break
            if len(rows) >= 5:
                blocks.append((re_val, np.array(rows)))
            i = j
        else:
            i += 1
    return blocks


class CdTable:
    def __init__(self, path=_DRG):
        blocks = _parse(path)
        self.res = np.array([b[0] for b in blocks])
        # 公共 alpha 栅格(并集范围,各块线性插到栅格,边界 clamp)
        amin = max(b[1][:, 0].min() for b in blocks)
        amax = min(b[1][:, 0].max() for b in blocks)
        self.alphas = np.linspace(amin, amax, 48)
        self.cd = np.stack([np.interp(self.alphas, b[1][:, 0], b[1][:, 2])
                            for b in blocks])          # (nre, nalpha)
        self.logre = np.log(self.res)

    def __call__(self, alpha_deg, re):
        """向量化 CD(α[deg], Re);超域 clamp。对称化:负攻角用 |α| 段近似
        (薄弯度截面近似;表内已含 −5°..0° 数据,超出才对称)。"""
        a = np.asarray(alpha_deg, float)
        a = np.clip(np.abs(a) * np.sign(a), self.alphas[0], self.alphas[-1])
        lr = np.clip(np.log(np.maximum(np.asarray(re, float), 1.0)),
                     self.logre[0], self.logre[-1])
        ia = np.clip(np.searchsorted(self.alphas, a) - 1, 0, len(self.alphas) - 2)
        ta = (a - self.alphas[ia]) / (self.alphas[ia + 1] - self.alphas[ia])
        ir = np.clip(np.searchsorted(self.logre, lr) - 1, 0, len(self.logre) - 2)
        tr = (lr - self.logre[ir]) / (self.logre[ir + 1] - self.logre[ir])
        c00 = self.cd[ir, ia]; c01 = self.cd[ir, ia + 1]
        c10 = self.cd[ir + 1, ia]; c11 = self.cd[ir + 1, ia + 1]
        return (c00 * (1 - ta) + c01 * ta) * (1 - tr) + (c10 * (1 - ta) + c11 * ta) * tr


if __name__ == "__main__":
    t = CdTable()
    print("Re 块:", t.res, " α 域:", t.alphas[0], "→", t.alphas[-1])
    for re in (1.1e5, 1.5e5, 1.9e5):
        print(f"Re={re:.0f}: CD(0°)={t(0, re):.4f} CD(5°)={t(5, re):.4f} "
              f"CD(9°)={t(9, re):.4f} CD(12°|clamp)={t(12, re):.4f}")
