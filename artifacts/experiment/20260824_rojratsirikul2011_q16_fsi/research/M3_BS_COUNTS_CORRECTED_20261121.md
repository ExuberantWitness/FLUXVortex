# M3：粒子 Biot–Savart 调用数与对数 —— 按 propose 累计的改正测量

> 脚本 `/tmp/count_targets_v2.py 110`；110 步（`wake_rows = 110 > wake_free_rows = 100`，已越过门控阈值）。
> 只读 monkeypatch，未改生产代码。`step_wall_mean = 8.460 s/步`（110 步仍在填充，150 步为 9.136）。

## 1. 实测（最后 20 个 propose 的平均）

| 次数/propose | 类型 | 调用者 | targets |
|---|---|---|---|
| **3.00** | `velocity_at_cuda` | `_advance_material_state` | **12,000** |
| 2.00 | `velocity_at_cuda` | `propose` | 450（配点）|
| 1.00 | `velocity_at_cuda` | `propose` | 1,800（涡丝中点）|
| ~3.0 | `velocity_self_cuda` | `velocity`（`advance_wrk3` 三阶）| ≈26,700（= n）|

`proposals / outer step = 2.9727`。

计数器把 `velocity_self_cuda` 与其内部的 `velocity_at_cuda` 各记一次（同尺寸成对出现），因此自诱导的真实次数为 **~3/propose**，不是表里的两倍。

## 2. 三处对我此前记录的纠正

| 我此前的记录 | 实测 |
|---|---|
| 尾迹顶点粒子诱导"targets 到 300 行为 36,000"、"`wake_free_rows` 门控未作用于粒子求值" | **错**。`active = original[:freeze_rows*ns]` ⇒ targets 封顶 **12,000**；门控生效 |
| "每步 1 次调用" | **错**。每个 propose 内 RK3 **三阶各一次** ⇒ 3 次/propose |
| 每步对数 **1.94×10⁹** | **错，低估约 5 倍** |

## 3. 改正后的每步对数（n ≈ 26,700）

| 目标集 | 次/步（= 3 × 2.9727）| 对数 |
|---|---|---|
| 尾迹顶点（free rows）| 8.92 | **2.87×10⁹** |
| 粒子自诱导 | 8.92 | **6.36×10⁹** |
| 配点（2/propose）| 5.94 | 4.7×10⁷ |
| 涡丝中点（1/propose）| 2.97 | 1.4×10⁸ |
| **合计** | | **≈ 9.4×10⁹ 对/步** |

与审计方独立估算的 9.6×10⁹ 一致（差异来自 n 取 27,371 还是 26,700）。

## 4. 对提速的含义（只做算术）
- 粒子自诱导占 **68%** 的对数，尾迹顶点占 **31%**；
- 二者都是每 propose 三次。若能在 FSI 耦合迭代/`formal replay` 中复用已对流的场（审计方 C2-5 的建议），对数按 proposal 数线性下降：2.97 → 1 次即降 **≈3×**；
- 仅靠"砍冻结行"没有收益（门控已生效）。

## 5. 边界
- 110 步时 n ≈ 26,700，仍在向稳态填充；150 步的 n = 27,371；
- 本测量只计数，未改任何代码路径；
- 未测 `velocity_at_cuda_chunked`（非生产路径）与 joint 路径特有的 ribbon-basis scratch 调用。
