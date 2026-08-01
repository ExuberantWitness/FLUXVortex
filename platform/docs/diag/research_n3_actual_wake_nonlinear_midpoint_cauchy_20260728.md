# N3 S3x：actual nonlinear midpoint 多步时间 Cauchy

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2b`  
状态：**EXECUTED / GO**

## ① 病因定位

S3w 的单步 algebraic/scalar/geometry 组成达到 roundoff，但单步正确不能推出
时间算法正确。尤其 projection residual 在 midpoint 从 `55.20%` 增至
`56.53%`；它可能只是连续状态变化，也可能是步长相关积累。

## ② 学科机理

HERK 的阶必须同时检查 differential geometry/scalar 与 algebraic body trace。
Roccia 的 explicit previous-time 分类只说明“不必迭代”，并不替代时间收敛。
Krebs 的 distributed wake 又使 geometry 和 strength 共同反馈 induced field，
因此不能只比较总 circulation 或单一 geometry 坐标。

## ③ 缺件还是错件

| 组成 | 判定 |
|---|---|
| S3w 单步组成 | 已验证，冻结 |
| 多步 geometry Cauchy | 缺证据 |
| 多步 global-P2 Cauchy | 缺证据 |
| algebraic body-trace Cauchy | 缺证据 |
| projection/normal diagnostic 步长独立性 | 缺证据 |
| 加 smoothing/damping | 无病因支持，禁止 |

## ④ S3x 预登记

固定 `T=0.01`，运行 `1/2/4` 步族。每个 midpoint 和 endpoint 都重算 actual
四通道速度；所有轨迹共享同一不可变初态。

终态分别比较：

- free-P1 geometry；
- 完整 global-P2 scalar；
- body attachment trace。

三个状态都必须呈二阶 Cauchy，最细差还需小于实际演化的 `2%`。projection
residual 与 relative-normal 不设拟合目标，只要求 coarse→medium 与
medium→fine 的终点差发生收缩。全部 stage residual、topology、area 和
mutation 门继续执行。

通过后也只授权 newborn-band/longer-history 门；不直接进入力拟合。

## 执行结果

预登记门原样执行，`10/10 GO`：

| 指标 | 结果 | 门 |
|---|---:|---:|
| geometry Cauchy | 4.0789 | ≥3.4 |
| global-P2 scalar Cauchy | 4.2133 | ≥3.4 |
| body trace Cauchy | 4.4283 | ≥3.4 |
| geometry 最细差/演化 | 2.2531e-4 | ≤0.02 |
| scalar 最细差/演化 | 2.5731e-4 | ≤0.02 |
| body trace 最细差/演化 | 1.2332e-5 | ≤0.02 |
| 最大具名残差 | 7.9656e-15 | ≤2e-11 |
| projection diagnostic 收缩 | 4.7320 | ≥1 |
| relative-normal diagnostic 收缩 | 3.7801 | ≥1 |
| topology mismatch / mutation / geometry iteration | 0 / 0 / 0 | 0 / 0 / 0 |

这证实的是：在冻结的两带 actual wake、`T=0.01` 短时窗内，
previous-time constraint-consistent midpoint 对 geometry、global-P2
材料强度和 algebraic body trace 同时保持二阶，并且此前保留的
projection/normal 指纹趋向步长无关终值。

## Claim 裁决

- `N3.1j3b6d18c2b3b3b2b`：由 open 改为 validated/frozen。
- `N3.1j3b6d18c2b3b3b2`：仍为 partial。S3x 没有验证新生带插入、
  多带长期 chronology、整扑翼周期或压力。
- `N3.1j3b6d18c2b3b3c`：不晋升。当前没有 actual stiffness 或显式
  HERK 失败证据要求 present/averaged-time 隐式联立。
- 下一可动空间仅为 newborn-band/longer-history 组成门；禁止借此
  引入 pressure、force、LESP、目标载荷、平滑或阻尼。
