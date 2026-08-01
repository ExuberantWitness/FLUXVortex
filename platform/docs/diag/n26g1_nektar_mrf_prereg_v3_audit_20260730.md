# N2.6g1 Nektar++ MRF 预登记 v3 独立审计

日期：2026-07-30  
对象：`n26g1_nektar_mrf_source_prereg_v3_20260730.md`  
对象 SHA-256：
`35703a0b55a2aaff1ea507eb23dcc460d35491d5324f9c885e857e8f8f04a782`  
裁决：`BLOCKED — NOT PASS TO EXECUTE`

审计同时核验：

- base v2：
  `14d29b5b6523ed523efae255b345edab61a1a5b2f31e54d265f986963d3c9b0b`；
- v2 audit：
  `95c6f42a75c73fefb5e6e04eccff15f85690b4021deef08baf1e0e89e6f73527`。

审计全程只读；没有解包、构建、生成 mesh、运行 source flow 或观察
target。

## P0-1：source immutability 门不可满足

v3 把 `NEK_SRC` 设为容器根，同时把 build、dependency sysroot、receipt
和 wrapper 放入同一目录；但 base v2 又要求 extracted source 的
post-extract/post-build manifest 完全相同。按 v3 的“替换”语义，旧门会
被意外删除；按继承语义，新增 build/wrapper 又必然使门失败。

唯一允许的修正是把 immutable source payload 与 build、receipt、sysroot、
wrapper 分开，并对同一 source-payload scope 恢复 pre/post file manifest
相等和 zero-patch 断言。

## P0-2：Hermite--Bezier 端点变量未绑定

v3 只定义了 \(s_i\) 与 \(\mathbf r(s)\)，但端点写成未定义的
\(\mathbf r(\theta_i)\)、\(\mathbf r(\theta_{i+1})\)。唯一机械修正是：

\[
\mathbf P_0=\mathbf r(s_i),\qquad
\mathbf P_3=\mathbf r(s_{i+1}).
\]

## P0-3：hybrid Physical Surface 与 composite 双射冲突

v3 固定 BL quadrilateral、fan/outer triangle，并把二者置于同一 Gmsh
Physical Surface 301；同时又要求 Physical ID 与 Nektar composite 双射。
固定 commit 的 `InputGmsh.cpp` 会按 shape 拆分同一 physical tag，第二种
shape 获得新 composite ID，所以旧门必然失败。

唯一允许的修正是保留 hybrid mesh，预登记 `301 -> {triangle,
quadrilateral}` 的 shape split；要求恰好两个 volume composites，冻结
remap 表，并让 `DOMAIN` 和所有 `EXPANSIONS` 同时覆盖二者。

## P0-4：单场 checkpoint 不能连续重放 IMEXOrder2

固定 commit 的 `FilterCheckpoint` 只写当前 `pFields`，没有写时间积分历史。
固定 `IMEXOrder2` 是二阶 multistep，需要两个 solution values 和两个
explicit derivatives。因而即使补齐 restart time/argv，单场 restart 仍会
重新经历启动相位，不能满足逐步 `1e-10` 连续重放门。

唯一允许的修正是：

1. 44/55 deg 已知时刻在五个 formal runs 内直接保存；
2. H2-fine 的 CL/CD first-argmax extrema 在 formal 结束后，从 `t=0`
   用相同 binary、mesh、session 数值配置和初值做至多两个去重后的
   deterministic prefix reruns；
3. prefix 每一步对 formal 的 `CL/CD/CM` 差 `<=1e-10`，运动量
   bitwise-identical；
4. prefix 输出不进入正式响应评分；
5. 资源门按“五个 formal + 至多两个 H2-fine prefix reruns”估算。

## 已关闭项

- Scotch sysroot、OpenBLAS realpath/SHA 和完整 CMake token；
- exact MRF XML 和 K0/K1/K2；
- Cauchy 公式、support 和 reference ranges；
- cross-mass、consistent load、刚体力/矩及八个非刚体虚功模式的核心定义。

## P2 澄清

v3 的 `LEVEL_SPECIFIC_INTEGER` 由前文的 `800/400/800` 已可机械替换，本身
不是 P0；采用从 `t=0` prefix rerun 后应删除该 restart 模板。v4 还应把
结构基明确写为 analytic `r_ref/n_sf,ref`，冻结矩阵/残差范数、独立
accumulator、旋转模式力矩尺度和 `Mesh.RandomSeed=1`。

## 审计结论

四个 P0 都在任何数值输出之前发现，均是执行规范错件，不改变 N2.6g
候选、物理命题、阈值或正式数值方案。只允许形成一个版本化 v4 mechanical
errata；v4 差分复核通过前继续保持：

```text
archive extraction = OFF
dependency acquisition = OFF
build = OFF
mesh generation = OFF
source flow = OFF
target = OFF
```
