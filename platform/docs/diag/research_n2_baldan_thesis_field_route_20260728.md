# N2.6b4f3b：Baldan 2026 博士论文带来的两级取数路线

## 1. 新的一手证据

Politecnico di Milano 于 2026-02-12 公开 Giacomo Baldan 博士论文：

```text
Rotor Dynamic Stall from High-Fidelity Simulations
to Deep Learning Surrogate Models
hdl:10589/249257
```

本次审计下载的公开 PDF：

```text
文件大小   15,475,519 bytes
页数       238
SHA256     8c2efec8cf8cc053e78bab6c2cd38159f4a706d2ee51908fe5a6b51d3e44a616
```

POLITesi 条目只有该 PDF 附件，没有体场文件或数据归档链接。论文 Appendix C
重印 Re=135k、span/c=1.2 的三维 WRLES 研究，原论文数据声明仍是
`available from the corresponding author upon reasonable request`。

## 2. 一个此前未显式利用的低成本入口

论文 Appendix F.B.3 明确给出 PBFM dynamic-stall 数据的上游生成过程：

```text
2D compressible URANS, NACA0012
ANSYS Fluent 2024R2
O-grid: 512 surface × 128 wall-normal nodes
2048 time steps / pitching cycle
SST + intermittency
128 nominal train + 16 nominal test conditions
32 perturbations / nominal condition
```

这证明公开的 `128×128` HDF5 表面位置×周期相位数组来自更高维的原生 Fluent
体解；公开输出只保留表面压力、密度、温度、壁面剪切和切向速度梯度，并未
发布法向 128 层上的速度场。

因此存在两条不同证据角色的作者取数路线：

### A0：一个 PBFM 原生二维 RANS 体场周期

请求一个 test condition 的 native Fluent case/data 或导出的 O-grid
`x,y,u,v,rho,p,T` 连续周期。它不能晋升三维生产节点，但可以：

- 实际执行 wall-normal profile/edge 抽取；
- 检查无滑移、材料 flow-map 时间插值和 IBL 同源投影；
- 提前证伪四加二状态是否连二维 RANS 剖面都无法条件重构；
- 在不接触载荷的情况下验证数据管线。

### A1：Re=135k 三维 WRLES 连续 onset 子窗

仍是目标 Re 的首选物理证据。它可检验三维涡破裂、展向非相干性和材料
分离，但因周期翼段无有限翼翼尖横流，只能作为生产前的三维提前证伪包。

## 3. claim 裁决

这项发现不改变 `N2.6b4f3b` 的 **open** 状态：没有实际体场文件，因而
不能晋升任何 profile decoder、材料 flow map 或分离 backbone。

它改变的是取数策略：

1. 不再只请求庞大的完整 WRLES；
2. 允许作者先提供一个连续 onset 子窗；
3. 若 WRLES 暂时无法共享，接受一个原生 PBFM 2D RANS test case 作为
   明确降级的 A0 证伪资产；
4. 两者都必须保留原生网格、时间、壁面运动、侧别和法向层身份；
5. 表面 HDF5、论文等值图或相位平均图仍不是体场。

未经张明昊明确授权，不发送邮件或外部消息。

## 4. 原始来源

- Baldan, G., *Rotor Dynamic Stall from High-Fidelity Simulations to
  Deep Learning Surrogate Models*, PhD thesis, Politecnico di Milano,
  2026, hdl:10589/249257, especially Appendix C and Appendix F.B.3.
- Baldan, G. & Guardone, A., *Wall-resolved large eddy simulations of a
  pitching airfoil in deep dynamic stall*, Physics of Fluids 37, 024112
  (2025), doi:10.1063/5.0252828.
- Baldan, G., Liu, Q., Guardone, A. & Thuerey, N., *Flow Matching Meets
  PDEs: A Unified Framework for Physics-Constrained Generation*,
  arXiv:2506.08604v4 / ICLR 2026, Appendix C.3.

