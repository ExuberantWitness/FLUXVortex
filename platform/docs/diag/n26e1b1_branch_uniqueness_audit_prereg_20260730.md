# N2.6e1b1 双支唯一性独立审计预登记

日期：2026-07-30  
对象：正式结果
`n26e1b1_source_faithful_te_refinement_result_20260730.json`  
性质：实验完整性审计，不是新气动候选

## 病因

正式 runner 在底层求解成功返回后无条件写入
`branch_unambiguous=true`。底层确实尝试 upper/lower 两个尾缘方向，但当
一支成功且符号一致、另一支因收敛、范围或验证错误退出时，仍可返回成功支。
因此该 JSON 字段没有证明两个候选支路都被完整审清。

这个缺口不会把正式 `NO-GO` 变成 `GO`：末级局部出生状态已有多个
`7--9% > 2%` 的独立失败。本审计只决定原结果中的 branch 子门应记为
`PASS` 还是 `FAIL`。

## 冻结方法

保持正式运行的源码、网格、运动、时间步和 core 不变：

- `64/128/256 panels-per-side`；
- `32` 个半余弦 ramp steps；
- `U=9 m/s`、`c=1 m`、`alpha: 0 -> 6 deg`、`T=0.4 s`；
- `core=0.02c`；
- 同一 `svi_dw_unsteady_outer_2d.py` 和正式 runner。

只读包装 `_select_physical_orientation_branch`，在不改变其输入、返回值或
异常的前提下，对每次选择记录：

- lower/upper 是否各自返回；
- 每支的 `DeltaGamma_B`、`sign_tolerance`、roundoff-no-birth 和
  nonzero-sign-consistent 标志；
- 每支错误的类型和消息；
- 原函数选择的方向或抛出的错误。

禁止改 solver、正式 runner、阈值、分支顺序或任何物理量；禁止读取
Figure 12、Fig17/18/19 或载荷。

## 判据

每个空间层应有恰好 `33` 次选择（32 次 march 加终态 solve）。每次必须：

1. lower 和 upper 两支都成功返回；
2. 恰有一支为 nonzero sign-consistent，且原函数选择该支；或两支均为
   roundoff-no-birth、原 `_no_birth_branch_solutions_agree` 为真并按原
   规则选择 lower；
3. 包装前后的数值返回不受改变，正式 case 指标可重现至浮点逐位相等。

全部 `99` 次满足才将 branch 子门记为 `PASS`。任一次缺支、双支一致、
零支一致、no-birth 场不一致、选择不匹配、次数不符或 case 指标漂移，
branch 子门即 `FAIL`。

无论本审计结果如何，`N2.6e1b1` 保持 `falsified/frozen`，正式总体
`NO-GO` 不得被逆转。

