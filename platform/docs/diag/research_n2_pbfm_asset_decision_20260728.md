# PBFM 动态失速公开资产：空间涡态资格裁决

日期：2026-07-28

## 1. 病因与节点

当前阻塞仍挂在 `N2.6b4f3b → N2.6c1b2b/N2.6c1b3b`：缺少同一目标域
资产中的近壁连续三维速度场，因而无法用真实材料身份检验

```text
flow map → material ridge/backbone → spatial LEV state。
```

PBFM 公开页称其包含 dynamic-stall dataset，且作者与目标候选
Baldan/Guardone 重合，必须先确认它是否已经公开了所需 WRLES 场，不能仅凭
标题把它当作合格速度场。

## 2. 一手来源与文件指纹

审计只采用作者的 ICLR 2026 论文、官方 GitHub 仓库及官方 Hugging Face
数据页：

- 论文：Baldan et al., *Physics vs Distributions: Pareto Optimal Flow
  Matching with Physics Constraints*, arXiv `2506.08604v4`；
- GitHub commit：
  `2a702673bae0cc4778d63af1c0243ad99e728b78`；
- Hugging Face revision：
  `022378d965c92611edcd01cc79044e9a230128f6`；
- `dynamic_stall_test.h5`：`268,453,888` bytes，LFS SHA-256
  `027f91e47c38952132ea562add0f0f8682021dd0369f1ac65999034015a86df2`；
- `dynamic_stall_train.h5`：`2,147,616,768` bytes，LFS SHA-256
  `c18cefd617a7327099f234151e5392077a1a3eb431ad2bfad3070eea0a647aa8`。

论文 Appendix C.3 明确生成模型是围绕正弦俯仰 NACA0012 的**二维非定常
可压缩 RANS**，使用 SST 加 intermittency transport。每周期原始求解
2048 步，但公开样本被后处理为 `128×128` 的
**翼面位置 × 周期相位**数组。

官方 loader 将 8 个通道拆成：

1. 六个物理通道：绝对压力、壁面切向速度梯度的 x/y 分量、温度、密度、
   signed wall shear；
2. 两个几何通道：翼面切向的 cosine/sine。

这里没有体速度分量，也没有 O-grid/Cartesian 体坐标。`128×128` 不是二维
空间速度平面。

## 3. 缺件还是错件

### 错件

把“spatio-temporal dynamic-stall fields”理解为“时序二维/三维体速度
场”是表示身份错误。该资产的两个轴是翼面位置和相位，不是两个空间坐标；
切向速度**梯度**也不能替代近壁速度剖面。

### 缺件

PBFM 没有填补当前缺失的：

- 显式时间坐标与完整壁面运动；
- wall-normal 坐标、双侧法向射线和 edge；
- 体速度场；
- 三维 spanwise 速度与横流；
- `Re=1.1e5--1.9e5` 的已声明数据身份。

因此缺失的仍是 Baldan/Guardone WRLES 原始三维场或独立生成的目标有限翼
场，不是需要再换一种插值器。

## 4. 有证据的方案裁决

预注册结果由 `pbfm_dynamic_stall_asset_cases.yaml` 固定，
`audit_pbfm_dynamic_stall_asset.py` 只按既有 `N2.6b4f3` 联合合同审计：

- `N2.6b4f3b` 空间场取得：**NO-GO，保持 open**；
- `N2.6c1b2b` 材料 flow map：**NO-GO**；
- `N2.6c1b3b` 真实场 backbone：**NO-GO**；
- `N3.1i` 统一面板压力：保留为**带二维可压缩 URANS 域标签的表面时空
  测试资产**，但不能据此晋升空间 LEV 物理。

没有下载 2.4 GB HDF5 来制造更强结论：官方论文、数据卡和 loader 已经
唯一给出发布变量身份。未来真正实现统一面板压力后，可以再下载 test split
检查翼面压力的相位拓扑、符号和守恒积分；在此之前下载全量文件不改变
`velocity=false`。

## 5. 来源

- [官方论文](https://arxiv.org/abs/2506.08604)
- [官方代码仓库](https://github.com/tum-pbs/PBFM)
- [官方数据集](https://huggingface.co/datasets/thuerey-group/PBFM)

