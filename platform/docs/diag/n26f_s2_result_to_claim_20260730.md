# N2.6f1 Basilisk moving-EB：result-to-claim

日期：2026-07-30  
Claim：`N2.6f1-BASILISK-MOVING-EB-SOURCE-GATE`  
裁决：`FALSIFIED / FROZEN / TARGET PROHIBITED`

## 1. 被证伪的精确命题

本结果证伪并冻结：

> 固定 SHA 的 Basilisk moving embedded-boundary 路径，在该 closed
> NACA0015 快速俯仰来源算例中，能够同时提供独立收敛的 CL/CD、跨相位水密
> 的移动物面，以及可作为后续材料面板压力/剪切证据的一次牵引积分。

失败是三重且独立的：

1. 有效替代时间轴最后两级 CD/CL 指标越过 3%；
2. 空间最后两级 CD/CL waveform 指标越过 3%；
3. 512 points/c 的 44/54 度物面闭合仍分别为
   `9.15e-5/8.26e-4`，而门为 `1e-8`。

完整日志与 official formal run 字节相同、fragment force ledger 为机器零，
所以不能把失败转移给 instrumentation 或求和代码。

## 2. 没有被证伪的命题

本结果不证伪：

- 不可压缩 Navier--Stokes 作为分离压力和壁面剪切的证据生成方程；
- 全域二维 observer 相对局部人工 NS--outer interface 的方向性优势；
- 贴体、水密、机体固连网格上的移动参考系或 ALE 方法；
- 统一双侧压力/剪切到材料面板的 co-design 载荷表示；
- S1 formal response 已通过这一事实。

它也没有观察 RoboEagle 目标条带，故不能证伪或验证
“V4.1 在 U10 条带缺失分离压力阻力”这一目标物理命题。

## 3. Claim tree 回写

- `N2.6f1: open -> falsified, freeze=true`；
- `N2.6f2` 保持 `open / NOT-RUN`，但因其必要依赖已证伪而禁止执行；
- 父级“全域黏性 NS 证据 observer”方向保持开放，不能复用本实现重跑；
- V4.1、N1、N2/N3 公式及 Fig17/18/19 基线完全不变。

正式机器证据：

- `s2_manifest.json` SHA
  `b5633481359d76f1285adf14c3c1540e2f6ca44c9923902acd0a25383dc5b156`；
- `s2_family_result.json` SHA
  `a4bebb7d417c4ecbcf7d5c11b027b45b9ae99178d942d1f49dc7f1a0d0d8b008`；
- 人读数据案卷：
  `n26f_s2_source_numerics_result_20260730.md`。

## 4. 禁止重走与唯一可动空间

以下均不能作为 N2.6f1 rescue：

- 继续减 DT、改变 3% 门或挑选较好时间层；
- 平滑/重归一化 CL/CD，改峰值窗口或参考支持域；
- 圆钝、打开或人工封口 trailing edge；
- 对缺口 traction 作后验守恒修正或总力重分配；
- 调 AMR、黏度、pitch law 或来源 Re 后重新比较；
- 越过来源门直接看 U10/Fig18 目标。

后继只能另立一个有证据的新实现命题，并同时满足：

1. 真实贴体且拓扑固定的水密物面，不使用 moving PLIC/cut-cell；
2. 规定运动在机体固连网格中表达，删除 overset/nonconservative interface；
3. pressure 与完整 viscous traction 来自同一离散场并保留有序材料边界；
4. 先过版本官方回归、运动学、独立 p/h/dt 轴和来源 NACA0015 门；
5. 任一来源门失败即证伪该新实现，不观察目标。

该边界把允许方向收缩到“贴体 body-fixed moving-reference-frame
observer”；它是新 claim，不是 Basilisk 参数补丁。

