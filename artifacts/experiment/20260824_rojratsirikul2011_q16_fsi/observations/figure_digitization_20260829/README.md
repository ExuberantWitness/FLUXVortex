# Rojratsirikul et al. (2011) Figure 6/9/12--15 数字化对比数据

## 1. 结论和适用范围

可以绘制用户列出的对比，但必须把两类科学对象分开：

1. **Figure 6、Figure 9 是当前柔性膜翼 FSI 的直接定量 oracle**：
   - Figure 6：时间平均位移场的最大值 `max_xy(mean_t(|z|))/c`；
   - Figure 9：柔性膜翼和刚性平板翼的时间平均法向力系数 `Cn`。
2. **Figure 12--15 是刚性平板翼的尾流脱涡数据**，不是柔性膜位移谱：
   - Figure 12：`AR=2, alpha=15 deg, Re=48,700` 的刚性翼尾流速度脉动谱；
   - Figure 13：有限展弦比刚性翼的 `St=fc/U_inf`；
   - Figure 14：其他文献的名义二维翼/尖锐前缘平板数据以及 `St=0.17/sin(alpha)`；
   - Figure 15：有限翼的修正 Strouhal 数 `St*=St sin(alpha)`。

因此，Figure 12--15 可用于验证 V5M/自由尾迹预测的脱涡频率，并检查柔性膜是否接近尾流基频或一阶谐波锁频；不得把它们直接当作 Q16 柔性膜的固有振动实验值。

## 2. 文件

| 文件 | 内容 | 推荐用途 |
|---|---|---|
| `figure06_displacement_digitized.csv` | 三个来流速度、`alpha=0..25 deg` 的 `zmax/c` | FSI 变形主对比 |
| `figure09_normal_force_digitized.csv` | 三个来流速度的柔性/刚性 `Cn`，`alpha=0..30 deg` | 载荷和柔性增升对比 |
| `figure12_wake_spectrum_digitized.csv` | `AR=2, alpha=15 deg, Re=48,700` 尾流谱线 | 谱形、主峰和宽带能量对比 |
| `figure13_15_rigid_wake_reference.csv` | `AR=2` 的 Figure 13/15 交叉读取结果 | 刚性翼脱涡频率对比 |
| `figure14_2d_reference_relation.csv` | 论文给出的 `St=0.17/sin(alpha)` 及 `St*=0.15--0.20` 带 | 外部物理关系/诊断带 |
| `rojratsirikul2011_fig06_09_12_15_digitized.png` | 统一预览图 | 快速人工复核 |
| `digitize_rojratsirikul_figures.py` | PDF SHA 校验、提取和重绘脚本 | 可追溯再生成 |

所有 CSV 数值均为从论文图中反演的 `digitized_approx`，不是作者发布的数据表。

## 3. 来源冻结

- 论文：P. Rojratsirikul, M. S. Genc, Z. Wang, I. Gursul, “Flow-induced vibrations of low aspect ratio rectangular membrane wings,” *Journal of Fluids and Structures* 27(8), 1296--1309, 2011。
- DOI：`10.1016/j.jfluidstructs.2011.06.007`
- PDF：`../../references/Rojratsirikul2011_JFS.pdf`
- SHA-256：`c9d8f59b4fefafd846fae77fdda6376424b70032db6ae6c40f1f28d51aa9a6a4`

脚本在提取前强制校验 SHA；PDF 漂移时非零退出。

## 4. 数字化方法和不确定度

PDF 中 Figure 6/9/13/15 不是可直接读取坐标的矢量 path，而是按竖向条带存储的高分辨率栅格/蒙版。脚本按 PDF 中的 image rectangle 顺序重组原生图像，再用主刻度做线性坐标标定。

- Figure 6：填充圆/方/三角标记中心，建议将 `zmax/c` 图读误差视为约 `±0.0007`。
- Figure 9：柔性填充标记约 `±0.012 Cn`；刚性三组空心标记大量重叠，建议按 `±0.02 Cn` 使用。
- Figure 12：连续谱线逐列跟踪；PSD 绝对读数约 `±0.00035`，主峰横坐标约 `±0.02 St`。
- Figure 13/15：同一点同时满足 `St*=St sin(alpha)` 后才保存；建议 `St` 使用 `±0.012`、`St*` 使用 `±0.003`。

Figure 14 的散点是作者从 Rojratsirikul 2009、Fage and Johansen 1927、Chen and Fang 1996、Abernathy 1962 等外部文献重新绘制的数据。当前数据包保存论文明确给出的拟合式和文字总结的 `St*=0.15--0.20` 带；这些外部散点不应冒充 Rojratsirikul 2011 本次实验的 GT。若要逐系列严格复现 Figure 14，应优先回到各原始论文取表或数字化原图，而不是对二次重绘继续累积图读误差。

## 5. 与统一框架输出的比较合同

### 5.1 Figure 6

模型量必须是：

```text
z_mean(x,y) = mean_t(z(x,y,t))
zmax_over_c = max_xy(abs(z_mean(x,y))) / c
```

不能用 `mean_t(max_xy(abs(z)))`，两者在振动状态下不等价。统计窗口必须排除启动斜坡并通过稳态/周期稳定性门。

### 5.2 Figure 9

```text
Cn_mean = mean_t(F dot n_chord) / (0.5*rho*U_inf^2*S)
```

`n_chord` 是旋转后弦面法向；不能直接取世界坐标 `Fz`。柔性和刚性曲线应分别报告，不得用刚性曲线代替柔性 FSI。

推荐指标：

- 逐速度 `MAE`、`RMSE`、最大绝对误差；
- 升力曲线线性段斜率；
- 峰值 `Cn` 和对应失速攻角；
- 柔性相对刚性的 `Delta Cn`。

### 5.3 Figure 12--15

从统一框架输出的刚性 `AR=2` 尾流探针速度或全局力/环量时序计算：

```text
St = f_peak*c/U_inf
St_modified = St*sin(alpha)
```

Figure 12 的直接频谱对比必须使用 `alpha=15 deg, Re=48,700`，且与实验一样在尾缘后 `1c/2c`、四分之一展向附近设置尾流探针并对多个位置的主频做汇总。仅用膜面位移谱去对比 Figure 12 科学对象不一致。

推荐门：

- Figure 12 主峰：`|Delta St| <= 0.05`；
- Figure 13 AR=2 同工况：报告逐点 `MAE(St)`，不得用拟合线调参后再评分；
- Figure 15：优先检查 `St*` 是否落入 `0.15--0.20`，并检查 `alpha≈14 deg` 局部低谷和 `alpha≈20 deg` 局部高点是否存在；
- Figure 14：只作为跨文献物理关系，不作为 Q16 FSI 主准确率门。

## 6. 再生成

从仓库根目录执行：

```bash
python artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/observations/figure_digitization_20260829/digitize_rojratsirikul_figures.py \
  --pdf artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/references/Rojratsirikul2011_JFS.pdf
```

脚本只进行 PDF 数据提取和绘图，不运行气动或 FSI 求解器，因此不违反正式 CASE 的 GPU-only 执行要求。
