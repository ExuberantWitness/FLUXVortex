# N2.6g1 Nektar++ MRF 来源门预登记 v1 审计

日期：2026-07-30  
审计对象：`n26g1_nektar_mrf_source_prereg_20260730.md`  
对象 SHA-256：`8cbfcc6e4d3cf3d070537d2a7f1ee407d8b417fe32067e4a0f8d869401cf6d33`  
裁决：`P0 BLOCKED — 不得解包、构建或产生流场输出`  
审计角色：同线程只读反方审计；因并发线程上限，独立性为 provisional，
必须由未参与 v2 起草的实现审计者再审后解除阻塞。

## 1. 审计范围

审计只回答：

1. v1 是否在看到任何 Nektar 自建工况输出之前唯一冻结了候选；
2. 每个 go/no-go 指标是否可计算、无零分母且无事后解释自由；
3. 来源域数值门是否足以支持材料面分布 traction，而不只支持合力；
4. closed-NACA0015 source 与 open-TE NACA2406-like target 是否有明确角色转换；
5. 构建、网格、求解器和人审图是否足以复现。

本审计不改变 V4.1、Fig17/18/19 数据、N1/N4、N2.6f1 失败状态或任何
气动公式。

## 2. P0 阻塞项

### P0-1：运动导数不是完整分段式

v1 的 \(\dot\alpha,\ddot\alpha\) 只写了 \(t\geq0.2\) 分支形式，却将其
用于“所有计划 solver times”。若直接在 \(t<0.2\) 评价，
\(\dot\alpha\) 和 \(\ddot\alpha\) 分别不是零和静止段导数，破坏 source
运动身份。必须显式定义 \(t<0.2\) 两者均为零，并规定切换点采用右支。

### P0-2：相对/归一化误差存在零分母

zero-motion、uniform-flow、力矩或导数核对都可能以零为参考值。v1 的
`relative L2` 和 `normalized max error` 未冻结尺度，结果可以是
`NaN/Inf`，也可在输出后任意选择归一化。必须使用预先固定的物理尺度和
`max(scale floor, reference norm)` 定义。

### P0-3：正式 session 未唯一确定

v1 未完全冻结：

- `SpectralVanishingViscosity` 类型和系数；
- boundary/initial conditions、pressure gauge；
- advection form、projection、dealiasing；
- basis、quadrature、velocity/pressure expansion；
- global linear solver、preconditioner、absolute/relative tolerance；
- startup、checkpoint、AeroForces/traction 输出时刻。

这些自由度足以显著改变 Re=10000 rapid-pitch 响应，故不能先运行后补写。

### P0-4：力、阻力和升力的方向与系数尺度未定义

v1 没有固定惯性轴、正攻角方向、物面法向方向、Nektar 原生法向与
fluid-on-body traction 的对应关系，也没有明确 `AeroForces` 原始量如何
除以 \(q_\infty c\)。因此即使曲线数值相同，仍可发生符号或二倍尺度错误。

### P0-5：G2 只闭合合力/合矩，不足以支持 co-design

总力和 quarter-chord moment 相同，不能排除上/下表面或弦向 traction
分配错误。必须把 H1/H2 与中/细时间轴投影到冻结的共同材料网格，比较
pressure/full-viscous traction 的全支撑分布，并用非刚体虚功模式验证
保守结构传力；否则不能声称“统一压力让路”已产生可用于结构 co-design
的载荷。

### P0-6：source/target 尾缘身份发生冲突

source 明确冻结为 closed NACA0015，而继承的 target 是 open-TE
NACA2406-like 数值代理。v1 同时禁止改变 TE 又在 G4 引用该 target，
状态机含混。必须明确：source 与 target 是两个预先声明的不同验证角色；
source 内绝不改 TE，G0--G3 全过后才允许按继承 SHA 切换到固定 target，
且不得把 source 网格或数值结果复用为 target 证据。

## 3. P1 可复现性问题

1. 网格只给了尺寸摘要，没有冻结几何脚本、field 叠加规则、曲线分段、
   高阶优化、转换命令和质量下限；原 wake box 若在 H2 全域采用
   `0.02c`，最低三角形数量已达约 \(3.9\times10^5\)，与本机资源不匹配。
2. Richardson observed order 的公式、范数、union knots 和退化差分判据
   未定义。
3. source 文献快照应是 nominal 44/55 deg；旧资产文件名中的 `54` 不能
   被误写成文献角度，且需记录实际最近时间及实际攻角。
4. reference CSV、图、scorer、继承 target prereg 和支持域没有在 v1
   内固定 SHA。
5. 人工视觉检查未冻结裁剪、色标、取样时刻和 rubric，不能复现。
6. manufactured/frame-invariance cases 只有名称，没有输入、输出和尺度。
7. G4 只引用可变文件名，未引用其 SHA。
8. “唯一选择 Nektar++”超出证据；准确说法只能是“本轮唯一预登记
   successor”，不能声称学术上不存在其他后端。
9. G3 可证明的是数值一致的材料 traction，不是逐点 traction 的物理真值。

## 4. P2 记录问题

1. Chandar & Sitaraman 的 *Computer Physics Communications* 卷号应为
   `273`，不是 `275`。
2. source archive 还需记录 HTTP headers、ETag/request ID、bytes 和
   archive SHA；protected tag 不是密码学签名。
3. 网格必须冻结正 Jacobian 以外的尺度化质量下限和硬资源上限。

## 5. 解除阻塞的必要条件

只有新的、另存为 v2 的预登记同时满足下列条件，才可解除本审计阻塞：

- 保留 v1 原文和 SHA，不覆盖历史；
- 在任何解包、编译、自建网格或 flow output 之前修复全部 P0/P1；
- 把 build/mesh/resource failure 写成 fail-closed no-go，而不是运行后
  换 solver、换 mesh 或换版本；
- 由未参与 v2 起草的审计者给出 `PASS TO EXECUTE`；
- claim tree 同时记录 v1 blocked、审计案卷和 v2 身份。

在此之前：

```text
N2.6g1 = open / preregistration blocked
SOURCE FLOW = OFF
TARGET = OFF
Fig17/18/19 improvement = NOT TESTED
```
