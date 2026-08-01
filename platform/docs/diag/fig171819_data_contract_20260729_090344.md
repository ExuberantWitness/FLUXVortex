# Fig17/18/19 Gate 0：数据身份与评分合同

**时间**：2026-07-29 09:03:44 +0800  
**阶段**：研究流程步骤①之前的数据身份门  
**裁决**：`CONDITIONAL / NO-GO for promotion`

## 1. 本门回答的问题

本文件只确定“用什么实验数据、在哪些物理工况、如何评分”。它不修改
V4.1 气动公式，不选择 claim 病灶，也不授权任何候选模型。

完整终点仍为 Fig17/18/19 的 50 条曲线。当前必须区分：

- **184**：在 Fig19(c/d) 暂按 2.6 Hz 解释时的条件性 solver grid；
- **530**：`platform/docs/data.md` 中 50 条曲线的全部原始数字化测点，
  其中 thrust/lift 各 265 点。

## 2. 权威输入

| 资产 | SHA-256 | 角色 |
|---|---|---|
| `platform/docs/data.md` | `ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1` | 原始数字化实验测点 |
| `platform/docs/repro_data.json` | `808ffeed36be0071850e954231417fa7007167c59eb5730cd8cb6829ff18101c` | 历史兼容资产，不再作为官方 ground truth |
| `platform/docs/s6_sweep_v41.json` | `965da388863dc57b390d58b49fe3b8978bdc77c3603b4a3276c97d4d17f94c73` | 冻结 V4.1 118 点种子 |
| `researchpaper/Meng2025_Drones_FlappingTwist_RoboEagle_SOURCE.pdf` | `eccaf750707a693fd58c0e38476a2b8ce2c694bfbf40b910f3bdf10017aa0a66` | 论文条件与图义的一手来源 |

## 3. 冻结评分规则

1. 解析 `data.md` 的全部 50 条曲线和 530 个原始测点。
2. 模型仅在 solver grid 上求解，再将**模型曲线插值到实验测点 x**。
3. 禁止将稀疏实验点插值到 solver grid；实验 force 值不被插值或重造。
4. 数字化端点若仅因取图误差轻微越过公开物理坐标轴，可投影到轴端点并逐点留痕：
   - twist tolerance：0.25°；
   - frequency tolerance：0.01 Hz。
5. 当前共有 18 个端点投影；最大 twist 偏移 0.19149°，最大 frequency
   偏移 0.005455 Hz，均小于相邻采样间距。超过容差的点硬失败，禁止外推。
6. Fig18(c) 按数据列头判为 thrust，Fig18(d) 判为 lift；论文 caption 的
   c/d 文本互换不作为通道依据。
7. Fig18(a,c) thrust 的 U=6/U=10 曲线身份采用源 PDF 图例和既有 D1 审计的
   修正；原始测点不重写，转换在 scorecard provenance 中逐条记录。

实现与确定性验证：

- `platform/fig171819_benchmark.py`
- `platform/tests/test_fig171819_benchmark.py`
- 11 项测试通过；冻结 118 点仍覆盖 38/50 曲线、118/184 条件，并对
  386 个原始测点评分。

## 4. Fig19(c/d) 固定频率身份

源 PDF 的 §4.5、Figure 19 图例与 caption 均未声明 c/d 扭转扫的固定频率；
`data.md` 也没有该元数据。MDPI 官方页面和版本记录未提供补充文件或勘误。
论文 Data Availability Statement 表明数据可向通讯作者合理请求。

### 4.1 同图 `twist=22.5°` 交叉锚

四个 AoA 汇总：

| 通道 | f=2.0 MAE | f=2.3 MAE | f=2.6 MAE |
|---|---:|---:|---:|
| thrust | **0.1477 N** | 0.4556 N | 0.8969 N |
| lift | 0.6223 N | 0.3675 N | **0.2879 N** |

AoA=5° 单点：

| 通道 | c/d 锚值 | f=2.0 误差 | f=2.3 误差 | f=2.6 误差 |
|---|---:|---:|---:|---:|
| thrust | −1.4467 N | **0.0682 N** | 0.3782 N | 0.8922 N |
| lift | +8.4072 N | 0.9630 N | 0.6172 N | **0.5909 N** |

### 4.2 AoA=5° 整条扭转曲线交叉核对

与 Fig18 显式频率曲线相比：

| 通道 | f=2.0 MAE | f=2.3 MAE | f=2.6 MAE |
|---|---:|---:|---:|
| thrust | **0.1252 N** | 0.4841 N | 0.9699 N |
| lift | **0.2918 N** | 0.3329 N | 0.3253 N |

因此 thrust 强指向 2.0 Hz；lift 的交叉锚弱偏 2.6 Hz，但整曲线不能唯一
识别 2.6 Hz。不存在足以冻结 c/d 共用单一频率的证据。

## 5. 对工况合同的影响

| 解释 | solver 条件总数 | 冻结118中可复用 | 新运行 |
|---|---:|---:|---:|
| c/d 均暂按 2.6 Hz | 184 | 118 | 66 |
| 作者确认 c/d 均为 2.0 或 2.3 Hz | 184 | 85 | 99 |
| 按当前通道指纹 T@2.0、L@2.6 | 217 | 118 | 99 |

当前 `184` 合同只能标为
`fig19_cd_frequency.status = unresolved`、`conditional_assumption_Hz = 2.6`。
即使完成 184/184，也必须由 `promotion_eligible = false` 阻止晋升。

## 6. 当前授权边界

**允许：**

- 补算 Fig18(c/d) U=6/10 的 66 个缺失条件。这 66 点在上述所有解释中都需要，
  不依赖 Fig19 的频率裁决。
- 生成条件性 scorecard，用于检查数值完整性和后续数据身份恢复。

**禁止：**

- 将条件性 184 宣称为完整、权威的 Fig17/18/19 基线；
- 用 Fig19(c/d) 的条件性评分选择、晋升或证伪气动 claim；
- 用“最像某一频率”的模型拟合结果反推实验工况；
- 在取得权威元数据前覆盖 fixed-name production baseline。

## 7. 解锁条件

取得作者原始工况元数据或书面澄清后，冻结 Fig19(c/d) 的频率身份，重新计算
唯一条件集合与缺失点，再打开完整基线的 publication/promotion gate。

一手来源：

- [MDPI official article](https://www.mdpi.com/2504-446X/9/8/535)
- [MDPI version notes](https://www.mdpi.com/2504-446X/9/8/535/notes)
- [Official article PDF](https://mdpi-res.com/d_attachment/drones/drones-09-00535/article_deploy/drones-09-00535.pdf)
