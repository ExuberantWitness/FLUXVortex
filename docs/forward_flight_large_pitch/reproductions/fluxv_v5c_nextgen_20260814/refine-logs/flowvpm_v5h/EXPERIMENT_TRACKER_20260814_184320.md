# FluxV v5h 实验跟踪表

生成时间：2026-08-14 18:43 CST  
当前裁决：`PLAN_READY / IMPLEMENTATION_NOT_STARTED`

## 固定输入

| 项目 | 固定值 | 状态 |
|---|---|---|
| FLOWVPM.jl | `4f433fb09f6baad25db65c9905e0d9cbb09663ce` | 已审源码 |
| FLOWUnsteady | `b7283db2e94a5f44a7ef2d57f223b0bdb8d0dec7` | 已审源码 |
| VortexLattice.jl | `63e8c363389f90b00176ff67675bdfd6f2498c58` | 已审公开分支；不依赖未发布 API |
| 生产路线 | Python/Warp direct，Julia offline oracle | 已裁决 |
| 初始 formulation | reformulated VPM，Gaussian-erf，inviscid，no-SFS | 已预注册 |
| 初始 overlap | `lambda=2.125`；另扫 1.5/2.0/2.5 | 已预注册，非目标拟合 |
| surface force owner | Ptera KJ+dGamma only | 已预注册 |
| target cases | Yang 6 + Fig14 12 unique/14 marker + Baik W1–W4 | 冻结但未授权评分 |

## 运行队列

| ID | Block | Run | 状态 | 通过条件 | 失败动作 |
|---|---|---|---|---|---|
| R-1 | source diagnostic | v5f2 revised-birth M5 | optional/pending | 20/40/80/160 有共同极限 | 归档 ring-source 反例 |
| R0.1 | B1 | 隔离 Julia 1.10 + upstream tests | pending | 官方测试通过，环境/hash冻结 | 停止全部 v5h |
| R0.2 | B1 | 导出 U/J、RK3、ring、leapfrog oracle | pending | Float64 finite、schema/hash闭合 | 停止 R1 |
| R1.1 | B1 | Python direct U/J parity | pending | `U<=1e-12`,`J<=1e-11` relative | 修核/坐标，不进入耦合 |
| R1.2 | B1 | RK3/rVPM/relaxation parity | pending | 每 stage state `<=1e-11` | 修离散时间层 |
| R2.1 | B2 | 单边/共享边 deposition | pending | incidence/Kelvin/vector moment `<=1e-12` | 修全局边图 |
| R2.2 | B2 | TE ring→particle shadow | pending | finest probe L2 `<=1%` | 修 deposition/kernel |
| R2.3 | B2 | TE exclusive owner | pending | owner=1；disabled bitwise | 回退 shadow，不接 Ptera |
| R3.1 | B3 | manufactured LE, fixed cloud dt scan | blocked_on_R2 | 160/80 peaks `<=1.25` | STOP，不评分 |
| R3.2 | B3 | fixed h, transport subcycling | blocked_on_R2 | finest fields/integrals `<=5%` | STOP，不评分 |
| R3.3 | B3 | lambda/formulation matrix | blocked_on_R2 | no clip；守恒门全过 | 保留最简单通过配置 |
| R4.1 | B4 | particles-off/no-release exact reduction | blocked_on_R3 | stepwise `np.array_equal` | STOP |
| R4.2 | B4 | TE-only two-way Ptera coupling | blocked_on_R3 | U/J/owner/load ledger闭合 | STOP |
| R4.3 | B4 | 非目标 AR6/heaving-wing | blocked_on_R3 | time/h/lambda 收敛 | STOP |
| R4.4 | B4 | direct↔FMM parity | deferred | U/J `<=1e-4`, load `<=0.2%` | 继续 direct |
| R5.1 | B5 | Ramesh/LDVM source qualification | blocked_on_R4 | source `O(dt)`、Kelvin/onset闭合 | STOP at transport result |
| R5.2 | B5 | frozen all-22 | blocked_on_R5.1 | 三篇 primary no-regression | 保留 v4b |
| R5.3 | B5 | 全工况图、manifest、独立审计 | blocked_on_R5.2 | hashes/metrics/claims闭合 | 不发布性能声明 |

## 每次运行必须记录

- git commit、dirty diff hash、上游 commit、Python/Julia/NumPy/Ptera/FastMultipole 版本；
- formulation、kernel、precision、direct/FMM 参数、lambda、h、sigma、dt_release、dt_transport；
- stable particle IDs、lineage、source edge、birth step、owner transition；
- U/J provider 集合和 RK stage；
- Kelvin、vector moment、impulse proxy、`Gamma*sigma^2`、min spacing/sigma；
- clip/nonfinite/owner-conflict counts，必须全为 0；
- 运行时、峰值内存、粒子数历史；
- 输入、结果、图、summary、manifest SHA-256。

## 当前开放问题

1. Julia 1.10 隔离安装及 FLOWVPM 官方测试尚未执行。
2. Ramesh/LDVM source flux 到三维全局 edge graph 的唯一映射尚未资格审查。
3. Ptera bound/near-wake 的解析 `J` provider 尚未实现；只提供 U 不可过门。
4. newborn 对 Ptera 的当步可见性需用离散方程与解析测试冻结。
5. 粒子 deletion/merge、CoreSpreading/RBF、SFS、FMM、GPU、restart 均后置。

## Stop/Go 快照

- Code-reading gate: `PASS`。
- Plan gate: `PASS`。
- Julia numerical parity: `NOT_RUN`。
- Conservative bridge: `NOT_RUN`。
- Manufactured LE convergence: `NOT_RUN`。
- Ptera two-way coupling: `NOT_RUN`。
- Source-law qualification: `NOT_RUN`。
- Cross-paper performance: `BLOCKED_NOT_SCORED`。
