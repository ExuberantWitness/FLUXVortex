"""RoboEagle 双曲柄-滑块机构精确运动学(零拟合,Drones 2025, 9, 535 eq.(2)-(7)).

论文印刷设计参数: c=62, z=90, n=11, m=21, a=34, k=26 (mm)。
两曲柄同半径 r、安装相位差 2α(论文原文);k = 2 r cosα (eq.2) 把 r 与 α 耦合,
保持扑幅恒定。推杆(长 l2)顶点高度:
    z_Fi(t) = sqrt(l2^2 - (r cos φi ∓ e)^2) ± r sin φi,  φ1,2 = Ωt ± α
扭转 sinψ = (z_F2 - z_F1)/l2 (eq.6);扑角波形取滑块平均高度经 asin 归一
(幅值 ±22.5° 由论文 Fig10 实测钉定;摇臂传递近似单调)。
l2 未印刷,由 z=90 推定 l2 = sqrt(z^2+(k/2)^2) ≈ 90.93 mm(名义顶点高度),
经 Fig8(r、2α 随扭幅曲线)与 Fig10(波形峰位 t/T≈0.17、幅值 ±12.5°)交叉验证。
本模块只用论文数据;无任何对本机测力数据的拟合。"""
import numpy as np

C, Z, N_, M_, A_, K_ = 62.0, 90.0, 11.0, 21.0, 34.0, 26.0   # mm, paper-printed
L2 = float(np.sqrt(Z ** 2 + (K_ / 2) ** 2))                  # inferred from z (see docstring)
FLAP_HALF_DEG = 22.5                                          # paper Fig10 measured


def solve_crank(tw_label_deg):
    """tw 标称(峰-峰)→ (r, alpha):eq.2 约束 r cosα = k/2;扭幅由行程极限差给出。
    行程极限(φ=±α? 论文 eq.4/5 在极限位):Δz_max = z_F2 - z_F1 (eq.4/5),
    sin(ψ_half) = Δz_max / l2,ψ_half = tw_label/2(峰-峰口径)。解 e 与 r。"""
    psi_half = np.radians(tw_label_deg / 2.0)
    tgt = np.sin(psi_half) * L2                              # 需要的 Δz_max
    # e 由扑幅定(eq.3, β=扑角半幅):
    e = C * (1.0 - np.cos(np.radians(FLAP_HALF_DEG))) / 2.0
    # 扫 r 解 Δz_max(r) = tgt(单调),α = acos(k/(2r))
    rs = np.linspace(K_ / 2 + 1e-6, 40.0, 4000)
    best = None
    for r in rs:
        al = np.arccos(np.clip(K_ / (2 * r), -1, 1))
        zf1 = np.sqrt(max(L2 ** 2 - (r * np.cos(al) + e) ** 2, 1e-9)) - r * np.sin(al)
        zf2 = np.sqrt(max(L2 ** 2 - (r * np.cos(al) - e) ** 2, 1e-9)) + r * np.sin(al)
        d = (zf2 - zf1) - tgt
        if best is None or abs(d) < abs(best[2]):
            best = (r, al, d)
    return best[0], best[1]


def waveforms(tw_label_deg, nphase=720):
    """返回 (phase[0,1), theta_flap_rad, psi_twist_rad) 机构精确波形。
    相位约定与生产一致:phase=0 为冲程顶(θ=+半幅)。"""
    r, al = solve_crank(tw_label_deg)
    e = C * (1.0 - np.cos(np.radians(FLAP_HALF_DEG))) / 2.0
    t = np.linspace(0.0, 1.0, nphase, endpoint=False)
    om_t = 2 * np.pi * t
    p1 = om_t + al
    p2 = om_t - al
    zf1 = np.sqrt(np.maximum(L2 ** 2 - (r * np.cos(p1) + e) ** 2, 1e-9)) - r * np.sin(p1)
    zf2 = np.sqrt(np.maximum(L2 ** 2 - (r * np.cos(p2) - e) ** 2, 1e-9)) + r * np.sin(p2)
    psi = np.arcsin(np.clip((zf2 - zf1) / L2, -1, 1))
    psi = psi - psi.mean()                                    # 去均值(安装零位)
    zm = 0.5 * (zf1 + zf2)
    zm = (zm - zm.mean())
    th = zm / np.max(np.abs(zm)) * np.radians(FLAP_HALF_DEG)  # 幅值钉 ±22.5(论文)
    # 相位规约:滚动使 θ 峰(冲程顶)在 phase=0
    i0 = int(np.argmax(th))
    th = np.roll(th, -i0)
    psi = np.roll(psi, -i0)
    return t, th, psi


if __name__ == "__main__":
    print("交叉验证一:Fig8 曲线(r 12→17mm、相位差 2α 0→80°)")
    for tw in (0.1, 5, 10, 15, 22.5, 25, 30, 45):
        r, al = solve_crank(tw)
        print(f"  tw{tw:>5}: r={r:.1f}mm  2α={2 * np.degrees(al):.1f}°")
    print("交叉验证二:Fig10(标称 25:扭幅 ±12.5°,扭峰 t/T≈0.15-0.20,滞后 90°±畸变)")
    t, th, psi = waveforms(25.0)
    ipk = int(np.argmax(psi))
    print(f"  扭幅 ±{np.degrees(psi.max()):.1f}°/{np.degrees(psi.min()):.1f}°, "
          f"扭峰 t/T={t[ipk]:.3f}, 扑幅 ±{np.degrees(th.max()):.1f}°")
    # 谐波含量
    c1 = np.fft.rfft(psi)
    print(f"  扭转谐波 |H2/H1|={abs(c1[2]) / abs(c1[1]):.3f} |H3/H1|={abs(c1[3]) / abs(c1[1]):.3f}"
          f"  基频相位滞后={np.degrees(np.angle(c1[1]) - np.angle(np.fft.rfft(th)[1])):.1f}°")
