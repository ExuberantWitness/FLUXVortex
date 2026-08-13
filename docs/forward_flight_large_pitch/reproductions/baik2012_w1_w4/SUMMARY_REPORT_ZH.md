# Baik 2012 W1--W4 复现与 FluxV 对比总结

## 1. 结论摘要

W1--W4 的几何、非简谐俯仰--升沉运动、corrected-total 实验载荷及论文
标准 Theodorsen 升力参考均已重建。主实验数据来自 Baik 博士论文
Fig. 5.24--5.27，而不是早期 AIAA 稿中去除了初始 8 度稳态水动力的
relative-load 曲线。

在与实验一致的 1 Hz Fourier 带宽下，full 计算得到：

| 模型 | CL 四工况宏平均 RMSE | CD 四工况宏平均 RMSE |
|---|---:|---:|
| FluxV old | 0.69484 | 0.40728 |
| FluxV v4b，Ramesh 正文平板 `Lcrit=0.11` transfer | 0.65754 | 0.34515 |
| 论文 standard Theodorsen（仅 CL） | 0.82062 | -- |

因此 v4b 相对 old 的 CL RMSE 下降 5.37%，CD RMSE 下降 15.25%。更重要的
是，W1--W4 每个工况的 CL 与 CD RMSE 都分别下降，并非由宏平均掩盖局部
退化。这个结果支持“v4b 的分离/LEV 机制迁移在 Baik W1--W4 上有正向效果”。

独立只读审计已从 20,800 个逐点样本零差重算全部 52 组指标；canonical
full 的 17/17 源码哈希和 7/7 结果哈希均匹配。审计总体判定为
`WARN / qualified only`：数值工件与算术通过，范围和数值收敛证据仍有限。

它不支持“Baik 已被高精度完全复现”。W2 的绝对误差仍最大；当前 Ptera
只能用自由端零厚度中面替代实验的 6.25% 厚圆边平板、壁面和自由面端板，
而且 Baik 专属 LESP 阈值没有公开。

## 2. 工况与运动

共同参数：弦长 0.076 m、跨度 0.600 m、厚度比 6.25%、前后缘圆角半径
0.002375 m、四分之一弦俯仰轴、Re=5000。试验翼几乎横跨 0.61 m 水槽，
底部约 1 mm 间隙并带自由面端板，所以是准二维壁面约束实验，而非自由端
AR=7.895 的有限翼。

| Case | k | h0/c | 标称 St | 表中俯仰幅值 | T |
|---|---:|---:|---:|---:|---:|
| W1 | 0.5 | 0.50 | 0.16 | 13.16 deg | 7.13 s |
| W2 | 1.0 | 0.50 | 0.32 | 33.73 deg | 3.56 s |
| W3 | 1.0 | 0.25 | 0.16 | 13.16 deg | 3.56 s |
| W4 | 0.5 | 1.00 | 0.32 | 33.73 deg | 7.13 s |

W3 的 `k=1.0`；AIAA 表中 `k=0.5` 是已由周期、正文和博士论文交叉排除的
误印。升沉位移不能写成正弦。实现先规定

```text
h_dot/U = -tan(alpha_pl,max sin(2 pi t/T))
```

再周期积分得到位移。由位移幅值约束解得 `alpha_pl,max=27.182110 deg`
（W1/W3）和 `47.755954 deg`（W2/W4）。四个关键相位的有效迎角均为
`8, 22, 8, -6 deg`。Ptera 实际移动网格反算有效迎角与解析式最大差小于
0.03 度。

## 3. 实验数据冻结

实验真值是博士论文 Fig. 5.24--5.27 的 corrected-total 直接测力曲线。原图
是 1318x1602、220 dpi 的嵌入 JPEG，不是矢量图。固定像素坐标标定和统一
动态规划中心线规则得到每工况 401 点 CL/CD；评分时去掉 phase=1 的重复
端点，使用 400 个唯一等间隔点。

没有做相位、振幅或均值拟合。原始像素中心线（重新插值到公共 401 点网格
之前）的周期积分均值与论文印刷均值分别为：

| Case | digitized CL / printed CL | digitized CD / printed CD |
|---|---:|---:|
| W1 | 1.0436 / 1.0400 | 0.0311 / 0.0315 |
| W2 | 2.1357 / 2.1100 | -0.1264 / -0.1270 |
| W3 | 1.1433 / 1.1400 | 0.1263 / 0.1270 |
| W4 | 1.3856 / 1.3700 | -0.3072 / -0.3080 |

公共 401 点配对文件会因左右面板原始像素宽度不同而产生很小的插值均值
变化；评分再去掉 phase=1 重复端点。这两种离散均值只用于数据管线审计，
均不用于修改波形。实验约有 +/-0.02 系数不确定度；数字化读图误差是另
一项，不能把二者当作允许模型平移或缩放的容差。

## 4. Full 逐工况精度

下表为 source-matched 1 Hz 过滤后的相位 RMSE；括号为 v4b 相对 old 的
下降比例。

| Case | CL old | CL v4b | CL 改善 | CD old | CD v4b | CD 改善 |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 0.53323 | 0.51566 | 3.29% | 0.22984 | 0.16091 | 29.99% |
| W2 | 1.07838 | 1.03231 | 4.27% | 0.78905 | 0.72568 | 8.03% |
| W3 | 0.38651 | 0.37432 | 3.15% | 0.33097 | 0.26259 | 20.66% |
| W4 | 0.78124 | 0.70788 | 9.39% | 0.27925 | 0.23143 | 17.12% |

MAE 的四工况宏平均也从 CL 0.59320、CD 0.35393 降到 CL 0.54886、
CD 0.29776，分别下降 7.47% 和 15.87%。

论文 standard Theodorsen 是 lift-only 外部参考，CL 宏平均 RMSE 0.82062。
这不能解释为 FluxV 在所有线性附着场景都优于 Theodorsen；W1 与 W3 的
Theodorsen 结果仍很有竞争力，宏平均主要受到 W2 大幅非定常工况影响。

## 5. LESP 来源冲突

Ramesh 博士论文 flat-plate Re=1000 的详细正文与 Fig. 4.19/4.21 使用
`Lcrit=0.11`，Table 4.1 却打印 0.19。主结果在看 Baik 误差前固定使用正文值
0.11；0.19 仅作为来源冲突敏感性，不因它在某些指标更好而改为主值。

0.19 敏感性的宏平均 RMSE 为 CL 0.59394、CD 0.34636。它显示 CL 对阈值
来源很敏感，也说明下一步应从独立的 6.25% 圆边平板数据或局部流场确定
Baik 专属 onset，而不是用 W1--W4 载荷反标阈值。

## 6. 数值设置与证据边界

Full 主设置：UVLM 4x8 面元、128 步/周期、3 周期、保留 2 周期尾迹；LDVM
512 步/周期、最多 256 步材料尾迹。主评分先对模型作论文同样的 1 Hz
sharp Fourier low-pass；原始数值曲线另存，不用滤波隐藏不稳定性。

W2 单因素数值敏感性记录在 `NUMERICAL_SENSITIVITY.md`。8→12 展向面元和
3→4 周期变化很小，受控 LDVM 256→512→1024 步变化也小于 old→v4b
差异；但 UVLM 64→128 步变化不可忽略，LDVM 保留 0.25/0.50/0.75 周期
材料尾迹也明显改变结果。因此 full 是冻结生产分辨率结果，不是网格/时间/
尾迹收敛解，精确改善百分比仍是 provisional。

## 7. 可声称与不可声称

可声称：

- W1--W4 corrected-total 实验曲线、几何和非简谐运动已建立可审计复现；
- 在固定的自由端中面替代和 1 Hz 评分口径下，v4b 对四个工况的 CL/CD
  RMSE 均优于 old；
- v4b 的总体改善在阻力通道更明显，说明引入分离/LEV 机制方向是有效的；
- standard Theodorsen 的 lift-only 外部曲线已独立数字化并参与对比。

不可声称：

- 当前 UVLM 已模拟真实水槽壁面/端板或 6.25% 厚圆边截面；
- `Lcrit=0.11` 或 0.19 是 Baik 工况经实验确认的材料常数；
- W1--W4 是完全 held-out 泛化验证；该 case 在开发阶段已被查看；
- 相位载荷已经达到实验不确定度水平；
- 一次 full 计算等于数值收敛证明；
- CD 的符号可以直接称作推力系数。论文定义 `CD<0` 才表示推力。

## 8. 主要文件

- 工况/模型：`platform/forward_flight_benchmarks/baik2012.py`
- 主 runner：`platform/forward_flight_benchmarks/run_baik2012_benchmark.py`
- 数值敏感性：`platform/forward_flight_benchmarks/run_baik2012_sensitivity.py`
- 测试：`platform/tests/test_baik2012.py`
- 来源审计：`SOURCE_AUDIT.md`
- 数字化说明：`source_data/DIGITIZATION_AND_PROVENANCE.md`
- full 数值：`runs/20260813_baik2012_w1_w4_full_reproducible/`
- 主图：`runs/20260813_baik2012_w1_w4_full_reproducible/baik2012_w1_w4_filtered_old_v4b.png`
- UVLM 单因素：`sensitivity/20260813_w2_one_factor_reproducible/`
- 受控 LDVM：`sensitivity/20260813_w2_ldvm_controlled_reproducible/`

不带 `_reproducible` 后缀的早期目录保留为开发历史，但其 manifest 生成于
最终 provenance collector 之前，不是本报告引用的规范证据。
