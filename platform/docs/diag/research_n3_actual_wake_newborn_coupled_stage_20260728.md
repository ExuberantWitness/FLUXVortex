# N3 S3aa：coupled newborn half/full trace

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2c2a`  
状态：**EXECUTED / GO**

## ① 病因定位

S3y/S3z 已经闭合 newborn 的几何与状态容器，但没有闭合 strength：

- S3y 的 `midpoint_trace/current_trace` 是显式反事实；
- S3z 只给出收敛的 weak P1 normal release geometry；
- 新增 18 个 P2 DOFs 中，upstream 9 个来自 old released trace，仍缺 half/full
  stage 的 current body/newborn trace；
- S3u 已证明“全量推进后再 copy/clamp body trace”遗漏 DAE 耦合。

所以当前可动空间不是调一个 shed-strength 常数，而是在每个 stage 解九维
actual body-wake algebraic trace。

## ② 学科机理

Krebs (2021, Fig.3.1, §3.3) 明确说明 surface 与 newly-created wake
strengths closely intertwined；每步 wake relaxation 前后都要 resolve，才能同时
满足 tangency 和 Kutta continuity。2022 Aerospace 论文进一步确认新 row 的
strength 在创建步赋值，之后保持 material identity。

Dumoulin–Eldredge–Chatelain (JFM 2023) 把 shed strength 与 body
no-through-flow、Kelvin 和 unsteady Kutta 放在同一 algebraic system。
Arnold–Strehmel–Weiner 的 HERK 结果则要求 algebraic variables 在 stage
一致求解。四者共同排除：

- old trace 直接复制到 half；
- half trace 直接复制到 endpoint；
- 端点平均或事后 clamp。

## ③ 缺件还是错件

| 组成 | 判定 |
|---|---|
| S3z weak release geometry | validated/frozen |
| S3y P1/P2 augmentation | validated/frozen |
| old/free P2 state injection | validated/frozen |
| counterfactual midpoint/current rows | 仅测试输入，不是物理 closure |
| half/full coupled algebraic trace | 缺件 |
| old free-state material transport | 下一门 |
| copy/average/clamp | 已有 DAE 证据反对 |

## ④ S3aa 预登记

固定 `dt=0.01`：

1. 用 S3z q7 weak release 形成 half/full finite geometries。
2. half rows 为 `[g_old, g_old, g_half]`，只把 `g_half` 作为九维未知量。
3. full rows 为 `[g_old, g_half, g_end]`，只把 `g_end` 作为九维未知量。
4. 每阶段以 zero＋9 unit bases 构造精确 affine residual，再独立验证一次。
5. 所有 54 个 non-body P2 values 在每次 solve 中必须原样保留。
6. 同时执行 copy-old/copy-half 反事实，证明 stage solve 不是空操作。

GO 只冻结 newborn stage algebraic composition；不宣称 old-state transport、
多次插入、时间阶、pressure 或 force 已完成。

## 执行结果

预登记门原样执行，`10/10 GO`：

| 指标 | 结果 |
|---|---:|
| weak q7 finest relative change | 0.02519 |
| half/full newborn min area | 6.78e-5 / 1.36e-4 |
| augmented P2 DOFs | 63 / 63 |
| rank deficiency / condition | 0 / 1.0000 |
| algebraic trace residual | 1.39e-17 |
| 54 free-state preservation | 0 |
| actual weak / attachment | 3.16e-16 / 0 |
| seam / P2 round-trip | 0 / 0 |
| solved half trace change | 0.025768 |
| solved full trace change | 0.005134 |
| minimum copy-counterfactual residual | 0.005134 |
| inference / geometry iteration / mutation | 0 / 0 / 0 |

half/full trace 的非零变化以及 copy 反例残差证明 coupled solve 不是形式上的
恒等操作。`[g_old,g_old,g_half]` 与 `[g_old,g_half,g_end]` 的 stage
composition 在 actual boundary equation 下成立。

## Claim 裁决

- `N3.1j3b6d18c2b3b3b2c2a`：validated/frozen；
- `c2` 父节点仍为 partial；
- `c2b` 保持 open，因为 S3aa 刻意冻结全部 old/free P2 state，尚未证明
  material transport、连续插入或时间 Cauchy；
- pressure、force、LESP 和 production 继续禁止。
