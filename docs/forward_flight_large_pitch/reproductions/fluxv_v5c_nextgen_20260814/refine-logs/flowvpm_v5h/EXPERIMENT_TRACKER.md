# FluxV v5h 实验跟踪表

更新时间：2026-08-14 22:07 CST  
当前裁决：`FLUXV_PTERA_REUSED / R0_R1_PASS / R2_G0_G4_PASS / D0_EVENT_PASS / D1_SOURCE_EDGE_PASS / SINGLE_CLOUD_PASS / FRONTIER_HANDOFF_PENDING`

## 固定输入

| 项目 | 固定值 | 状态 |
|---|---|---|
| FLOWVPM.jl | `4f433fb09f6baad25db65c9905e0d9cbb09663ce` | 已审源码 |
| FastMultipole.jl | `adc4f264732de3dbbd492758e729af0b35db54b2` | 已冻结；v2.2.0 |
| Julia | `1.10.11`, official tar SHA256 `fb49c6...37bf` | 已隔离安装 |
| FLOWUnsteady | `b7283db2e94a5f44a7ef2d57f223b0bdb8d0dec7` | 已审源码 |
| VortexLattice.jl | `63e8c363389f90b00176ff67675bdfd6f2498c58` | 已审公开分支；不依赖未发布 API |
| 生产路线 | FluxV/Ptera UVLM + DVM source + Python/Warp rVPM transport；Julia offline oracle | 已裁决 |
| 初始 formulation | reformulated VPM，Gaussian-erf，inviscid，no-SFS | 已预注册 |
| 初始 overlap | `lambda=sigma/h>=2.125`；固定 sigma 后按边长反算粒子数 | FLOWUnsteady static-sheet-inspired development transfer；不是动态 LEV core 定律 |
| surface force owner | Ptera KJ+dGamma only | 已预注册 |
| target cases | Yang 6 + Fig14 12 unique/14 marker + Baik W1–W4 | 冻结但未授权评分 |

## 运行队列

| ID | Block | Run | 状态 | 通过条件 | 失败动作 |
|---|---|---|---|---|---|
| R-1 | source diagnostic | v5f2 revised-birth M5 | optional/pending | 20/40/80/160 有共同极限 | 归档 ring-source 反例 |
| R0.1 | B1 | 隔离 Julia 1.10 + upstream tests | passed | 14/14官方testsets；环境/hash冻结 | — |
| R0.2 | B1 | 导出 U/J、RK3、relaxation oracle；记录ring/leapfrog官方证据 | passed | Float64 finite、schema/hash闭合 | — |
| R1.1 | B1 | Python direct U/J parity | passed | `U=1.315e-16`,`J=8.136e-17` relative | — |
| R1.2 | B1 | RK3/rVPM/relaxation parity | passed | worst relative `2.456e-16`; schema-v2 29/29 tests | — |
| R2.1 | B2 | 单边/共享边 deposition | passed | incidence/vector moment `<=1e-12`; 38/38 bridge+R0 tests | — |
| R2.2 | B2 | pure ring→particle shadow | passed | finest probe L2 `1.26e-4` segment / `9.79e-5` ring | — |
| R2.3 | B2 | Ptera TE read-only shadow G4 | passed | 44 tests；80 native records bitwise；owner=ring；feedback=0 | — |
| D0.1 | source | author Fortran v2.5 exact external replay | event_history_pass_rowwise_strength_qualified | 499 rows, 174 LEV, onset/Kelvin/source row parity | exact motion only in external harness; in-repo clean-room motion is non-row-identical; 保留Newton-vs-linear限定 |
| D0.2 | source | Python 172→174 first-divergence closure | passed | camber ordinate + first-TEV initialization + provisional-TEV LE velocity + per-step dt | 无target调参 |
| D1.1 | source | source-only DVM API | passed_noncanonical | 49 joint tests；Kelvin solve `2.78e-16`、persistence `3.33e-16`；no force/load fields | 保持canonical=false，不反馈 |
| D1.2 | source | strip source→global node-owned edge shadow | passed_noncanonical | 122 full-stack tests；shared node / full incidence / event-chain / first-continuous-restart；独立审计PASS | 保持feedback=0，不评分 |
| R3.0 | B3 | DVM first-release, fixed-sigma single-cloud shadow | passed_mechanical | straight/taper/twist；invariant drift `<=1.22e-15`；`max(dt||J||)<=2.93e-4`；zero clip | 只证明单云机械稳定，不声明精度 |
| R3.1 | B3 | advected `NodeFrontierFact` + repeated release | in_progress | continuous node只消费同一rVPM时层的已对流frontier；防伪/回滚/重放门 | STOP，不评分 |
| R3.2 | B3 | fixed h, repeated-release transport subcycling | blocked_on_R3.1 | finest fields/integrals `<=5%` | STOP，不评分 |
| R3.3 | B3 | lambda/formulation matrix | blocked_on_R2_D1 | no clip；守恒门全过 | 保留最简单通过配置 |
| R4.1 | B4 | particles-off/no-release exact reduction | blocked_on_R3 | stepwise `np.array_equal` | STOP |
| R4.2 | B4 | TE-only two-way Ptera coupling | blocked_on_R3 | U/J/owner/load ledger闭合 | STOP |
| R4.3 | B4 | 非目标 AR6/heaving-wing | blocked_on_R3 | time/h/lambda 收敛 | STOP |
| R4.4 | B4 | direct↔FMM parity | deferred | U/J `<=1e-4`, load `<=0.2%` | 继续 direct |
| R5.1 | B5 | frozen all-22 | blocked_on_R4_D1 | 三篇 primary no-regression | 保留 v4b |
| R5.2 | B5 | 全工况图、manifest、独立审计 | blocked_on_R5.1 | hashes/metrics/claims闭合 | 不发布性能声明 |

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

1. Ptera G4 已在单翼、prescribed-wake、单瞬时 kernel channel 上通过；多翼、异核与反馈仍不在已验范围。
2. Ramesh/LDVM 事件历史已在隔离 `source_parity` 模式中闭合为 174/174；D1 source/Kelvin/单位账本和 node-owned 3-D ribbon 均已独立通过。当前阻塞是把 rVPM 已对流的材料前沿作为下一次 continuous birth 的唯一事实源，不是再调 LESP 阈值。
3. 旧单云失败已定位为逐边 `sigma=lambda L_edge/n` 使 O(dt) 短边 core 塌缩约 496 倍，造成 stretching/Jacobian 显式RK刚性；不是普通对流 CFL，也没有发现 FLOWVPM 方程移植错误。固定空间 sigma 后机械门通过，但尚无 h 收敛或气动精度资格。
4. Ptera bound/near-wake 的解析 `J` provider 尚未实现；只提供 U 不可过门。
5. newborn 对 Ptera 的当步可见性需用离散方程与解析测试冻结。
6. 粒子 deletion/merge、CoreSpreading/RBF、SFS、FMM、GPU、restart 均后置。

## Stop/Go 快照

- Code-reading gate: `PASS`。
- Plan gate: `PASS`。
- Julia numerical parity: `PASS`，仅 direct/inviscid/no-SFS 输运层。
- Conservative pure bridge G0–G3: `PASS_MECHANICAL_SHADOW_ONLY`。
- Ptera TE extraction G4: `PASS_READ_ONLY_NOT_FORCE_COUPLED`。
- Manufactured first-release fixed-cloud gate: `PASS_MECHANICAL_SINGLE_LAYER_ONLY`。
- Advected-frontier repeated release: `IN_PROGRESS_NOT_SCORED`。
- Ptera two-way coupling: `NOT_RUN`。
- DVM author-source event parity: `PASS_174_OF_174_ROW_STRENGTH_QUALIFIED`。
- DVM source-only interface: `PASS_MECHANICAL_NONCANONICAL`。
- DVM node-owned ribbon: `PASS_MECHANICAL_NONCANONICAL`。
- Cross-paper performance: `BLOCKED_NOT_SCORED`。
