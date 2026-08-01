# N3 S3w：actual nonlinear previous-time midpoint 单步组成门

日期：2026-07-28  
Claim：`N3.1j3b6d18c2b3b3b2`  
状态：**EXECUTED / GO**

## ① 病因定位

S3v 证明 fixed-geometry actual DAE 上 explicit midpoint 二阶且无局部刚性证据，
但它冻结了两项真正的非线性：

- wake geometry 改变 owner-aware physical velocity；
- physical velocity 改变 normal geometry 和 relative-tangential P2 transport。

同时 S3u 已证明 body trace 不能在最后 clamp。故下一缺件不是“再选一个积分器”，
而是一个同时满足 geometry stage、P2 increment 和 algebraic body trace 的
nonlinear stage composition。

## ② 学科机理

HERK 的 algebraic stage 与 Roccia 的 previous-time geometry 并不矛盾：

- geometry 使用已有 stage velocity 显式预测，不做 geometry fixed point；
- 在预测出的固定 stage geometry 上，body/wake potential equation 必须重解；
- P2 free block必须显式包含 `M_fb (g_stage-g_n)`，不能 endpoint overwrite。

Krebs 的“移动 wake 后再恢复 strength/tangency”也支持把 geometry update 与
algebraic flow solve分开，但不支持省略后者。

## ③ 缺件还是错件

| 组成 | 判定 |
|---|---|
| previous-time geometry prediction | S3v授权进入原型 |
| fixed-stage actual velocity/P2 matrices | S3t已验证 |
| fixed-stage body trace affine response | S3v已验证 |
| endpoint scalar clamp | S3u已证伪 |
| nonlinear half/end stage组成 | 缺件 |
| geometry fixed point | 当前无证据需要 |

## ④ S3w 预登记

在 `dt=0.01` 上：

1. base actual velocity给 `X_mid=X_n+dt*w_n/2`；
2. 在固定 `X_mid` 上联立 half-step P2 increment 与九维 body trace；
3. 重算 midpoint actual velocity；
4. 用 `X_1=X_n+dt*w_mid` 和 midpoint P2 flux 联立 endpoint trace。

每个九维 trace residual 对固定 geometry 都是 affine；用 zero + 9 unit bases
建精确矩阵，再做独立 verification solve。它不是松弛迭代，也没有 ridge 或
容差拟合。

本门冻结 actual-boundary、scalar increment、attachment、seam、P2 round-trip、
四通道 ledger、非零更新、面积和输入不可变门。只在全部通过后才允许多步时间
Cauchy；仍不计算压力或力。

## ⑤ 执行结果

冻结 `dt=0.01` 与全部门限未改，10/10 GO：

| 指标 | 结果 |
|---|---:|
| actual-boundary weak residual | `4.161e-16` |
| half/end affine stage rank deficiency | `0` |
| 最大 stage condition | `1.13416` |
| algebraic trace residual | `2.498e-15` |
| half/full scalar residual | `2.349e-15 / 1.332e-15` |
| geometry identity / body attachment | `0 / 0` |
| chronological seam / velocity ledger | `0 / 0` |
| P2 round-trip | `2.498e-15` |
| free geometry / scalar change | `1.1448e-3 / 1.8931e-3` |
| minimum face-area ratio | `0.999871` |
| input mutation / geometry iterations | `0 / 0` |

每个 half/end stage 使用 `11` 个 actual solves（zero、九个 unit basis、独立
verification），未使用收敛容差、ridge 或 endpoint clamp。

## ⑥ Claim 边界

新增 validated/frozen 子节点仅声称：

> 一个 `dt=0.01` 的 nonlinear previous-time midpoint step 能同时闭合
> geometry、global-P2 scalar 与 actual body algebraic trace。

`N3.1j3b6d18c2b3b3b2` 仍为 partial，因为单步代数闭合不证明多步时间阶。
而且 geometry projection residual 从 `55.20%` 增至 `56.53%`，
relative-normal diagnostic 从 `0.3295` 增至 `0.3305`。这些量没有进入拟合，
必须在下一多步门中检查随步长的收敛与累积。
