# S3ai-v2.1 可执行定义审计（正式运行前）

时间：2026-07-28 13:32:18 +08:00  
节点：`N3.1j3b6d18c2b3b3b2c2b2b3e3b`  
裁决：**v2 定义有歧义；v2.1 已在任何正式 history 前补冻**

## 审计发现

独立审计没有运行 31-history canonical，也没有看到 pressure-law
物理结果。它发现的是定义层问题：

1. v2 的 quadrature plateau 写成 `u_q=delta2+u_round`，总式又包含
   `+u_round`，会双计 floating budget；
2. 空向量/`0x0` mass 可被数值 primitive 当成“精确收敛”；
3. negative controls 没有固定 DOF、幅值和 pressure-change 判据；
4. mirror parity、condition norm 和 floating-vector 集合不够确定；
5. 31 条 history 的现有 S3e 实际工作量应为 380 measured steps、
   411 total steps、822 half/full solves、791 observed stages，而不是
   共享 pre-step 后的 387 steps。

这些问题会改变 go/no-go，所以不能在实现里自行选择。v2 时间戳文件保持
不变；`actual_wake_reachable_pressure_obstruction_cases_20260728_133218.yaml`
作为 v2.1 可执行 addendum，在正式运行前固定唯一解释。

## v2.1 修正

- Richardson、q-tail 和 mixed allowance 内都不含 floating floor；
  `u_round` 只在总误差中加入一次。
- active space 必须恰为 7 维，`M` 必须为 bit-exact symmetric 的
  `7x7` SPD，并在全部 histories 中完全相同。
- 精确固定 stored-phi 污染 DOF 与 `2^-10` 幅值、非对称九节点 trace、
  wrong sign、row/surface mutation 和局部抵消负对照。
- span mirror 固定为数组反向的 even parity；零向量单独返回零残差。
- condition 固定为 `kappa_2(B)` 乘 componentwise backward error，只作
  screen，不伪称严格 forward bound。
- body/direct-W q 由 typed runner 同时传入；由于 S3e solution 不保存
  body-q，本门只声明 typed-call+spy 证据，不夸大 provenance。
- 每条 history 独占一个 zero pre-step；正式 runner 必须在首个 march
  前断言 `31/380/411/822/791`。

## 当前边界

v2.1 仍是预登记，不是结果。正式执行还需：

1. actual previous-full→half→full inventory 测试；
2. seam/time/geometry/internal-trace 缺陷测试；
3. runner 独立实现审计；
4. 通过后才允许一次正式执行并生成带时间戳 JSON。

无论结果为何，它仍没有空间 mesh `h/p` 细化，不能直接裁决
massless forming geometry、finite VES 或 production。
