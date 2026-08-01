# Fig. 19(c,d) 固定频率身份：一手资产穷尽核查

**核查日期**：2026-07-29（Asia/Shanghai）
**论文**：Meng et al., *Drones* 2025, 9(8), 535
**DOI**：10.3390/drones9080535
**范围**：只核查 Fig. 19(c,d) 扭转幅值扫描采用的固定扑动频率及两面板是否同频。
**裁决**：`UNRESOLVED / NO-GO for promotion`
**修改边界**：本次只新增本核查文档；未修改代码、实验数据、模型参数或 claim 状态，也未联系作者。

## 1. 结论先行

| 对象 | 一手公开资料能确认的事实 | 当前身份裁决 |
|---|---|---|
| Fig. 19(c) | 图内纵轴为 `T_net/g`；caption 却把 (c) 写成 Lift；正文只说“frequency is held constant”，未给数值 | 固定频率 `unresolved`；论文内部曲线指纹强烈偏向 2.0 Hz |
| Fig. 19(d) | 图内纵轴为 `Lift/g`；caption 却把 (d) 写成 Net thrust；正文未给固定频率 | 固定频率 `unresolved`；整曲线偏向 2.0 Hz，但交点证据偏向 2.5/2.6 Hz |
| c/d 是否同频 | 公开正文、图例、caption、JATS、EPUB、原始 TIFF、版本记录均未声明 | `not established` |

最简设计意图假设是“c/d 来自同一批同时测得升力和推力的工况，且共同为
2.0 Hz”。它能解释 AoA=5° 的整条曲线冗余，但不能解释全部跨 AoA 交点，
因此只能登记为：

```text
H_shared_2p0 = plausible internal inference, not authoritative metadata
```

不得把它写入 frozen ground truth，也不得用条件性 Fig. 19(c,d) 评分晋升或证伪
气动 claim。

## 2. 证据等级

- **P：一手公开事实**——出版商正式资产、作者 ORCID、机构仓储或正式注册元数据。
- **I：论文内部数值推断**——只使用同一论文图中的冗余曲线及其数字化值。
- **N：负向检索结果**——说明“本次未找到”，不等价于证明资产不存在。

下文不会用 I 或 N 覆盖 P，也不会把“最像某频率”写成实验工况事实。

## 3. MDPI 一手资产与内容哈希

抓取时间为 2026-07-29 15:07–15:09 +08:00。SHA-256 是本次取得的响应正文
内容哈希；HTML 的 403 错误页不作为论文资产，因此不接受其哈希。

| 资产 | 直接 URL / 本地路径 | 状态、字节数、SHA-256 | 与频率身份有关的结果 |
|---|---|---|---|
| Article HTML | [MDPI article](https://www.mdpi.com/2504-446X/9/8/535) | 本地 `curl` 为 HTTP 403；未获得可接受的 article HTML 哈希 | 可由出版商索引读取正文；§4.5 未给 c/d 固定频率 |
| Version notes | [MDPI version notes](https://www.mdpi.com/2504-446X/9/8/535/notes) | 本地 `curl` 为 HTTP 403；未获得可接受的 notes HTML 哈希 | 记录仅有 2025-07-30 的原始 HTML、PDF、XML；未见 correction 或 dataset/supplement 更新 |
| Version of Record PDF | [official PDF](https://mdpi-res.com/d_attachment/drones/drones-09-00535/article_deploy/drones-09-00535.pdf)；本地 `researchpaper/Meng2025_Drones_FlappingTwist_RoboEagle_SOURCE.pdf` | HTTP 200；6,718,611 B；`eccaf750707a693fd58c0e38476a2b8ce2c694bfbf40b910f3bdf10017aa0a66`；本地文件同哈希 | §4.5、Figure 19、caption 均不含 c/d 固定频率 |
| JATS XML | [official JATS](https://mdpi-res.com/d_attachment/drones/drones-09-00535/article_deploy/drones-09-00535.xml) | HTTP 200；134,243 B；`ecd6b1a3ef7d3bf03e58a70c8ed2a7f04c598661f59b19b16248a880e5f66fbc` | §4.5 重复同一省略；`supplementary-material` 节点数 0，`related-object` 节点数 0 |
| EPUB | [official EPUB](https://mdpi-res.com/d_attachment/drones/drones-09-00535/article_deploy/drones-09-00535.epub) | HTTP 200；13,769,299 B；`65aeb21dd217e6fd356c95035aa3cb763e02d0a09eb98dd585eadf933326359a` | 68 个归档条目；manifest/文件名中 supplement、dataset、CSV、XLSX、ZIP 命中数 0 |
| Figure 19 原始 TIFF | [official Fig. 19 TIFF](https://mdpi-res.com/d_attachment/drones/drones-09-00535/article_deploy/drones-09-00535-g019.tif) | HTTP 200；642,060 B；`49f5fa2f18ddfc87ed7c79fadc24469ed5133f72ab5851b601ea792721f65696` | 3364×2782、600 dpi；图内 (c)=`T_net/g`、(d)=`Lift/g`；EXIF 无工况、源文件名或频率元数据 |
| Data Availability | [official JATS, Data Availability Statement](https://mdpi-res.com/d_attachment/drones/drones-09-00535/article_deploy/drones-09-00535.xml) | 被上述 JATS/PDF 哈希共同覆盖 | 数据集仅声明可向通讯作者合理请求；未给 repository、accession、DOI 或下载链接 |
| Supplement | JATS/EPUB 为权威 manifest；另负向探测同目录 `drones-09-00535-s001.{pdf,zip,xlsx,csv,docx,txt}` | JATS 节点 0；EPUB 命中 0；六个候选直链均 HTTP 404 | 未发现公开 supplement；404 错误页不做资产哈希 |

六个补充文件负向探测采用的共同直接 URL 前缀为：

```text
https://mdpi-res.com/d_attachment/drones/drones-09-00535/article_deploy/
```

该负向探测只作旁证；“无 supplement”的主要依据是正式 JATS 与 EPUB
manifest，而不是猜测文件名。

## 4. 一手资产明确写了什么、没有写什么

### 4.1 正文和 Table 3

JATS §4.5 明确给 Fig. 19(a,b)：

- 固定风速 `U=8 m/s`；
- 已确定扭转幅值 `22.5°`；
- 横轴为扑动频率。

紧接着正文只写“When the flapping frequency is held constant”，随后讨论
Figure 19(c,d)，但没有给出所保持的频率数值。Figure 19 图例只有 AoA，
caption 也没有频率。

Table 3 给出的实验频率集合是：

```text
1.4, 1.7, 2.0, 2.3, 2.6 Hz
```

但官方 Figure 19(a,b) 图中可见的频率标记还包括 1.5 和 2.5 Hz。这是论文内部
另一个工况表/绘图不一致，意味着不能仅凭 Table 3 假定 c/d 必属五个离散值之一。

### 4.2 panel/caption 互换

官方 TIFF 直接显示：

- 物理 panel (c) 的纵轴是 `T_net/g`；
- 物理 panel (d) 的纵轴是 `Lift/g`。

JATS caption 却写：

- (c) Lift versus twist amplitude；
- (d) Net thrust versus twist amplitude。

因此后续数值核对按**图内纵轴**认定通道，而不是按 caption。该互换是出版物
本身的可见事实，也降低了用 caption 补全缺失工况元数据的可信度。

## 5. 外部一手注册表与仓储检索范围

检索日期均为 2026-07-29。下表是 N 级负向检索，不把“未检出”表述为
“不存在”。

| 来源 | 直接查询或范围 | 本次结果 |
|---|---|---|
| Crossref DOI deposit | [Crossref work record](https://api.crossref.org/works/10.3390%2Fdrones9080535) | `relation={}`、`update_to=null`；唯一 link 为 MDPI Version of Record PDF，无 dataset/code 关系 |
| DataCite | [DOI/full-text query](https://api.datacite.org/dois?query=10.3390%2Fdrones9080535&page%5Bsize%5D=100) | `meta.total=0`，未检出引用该 DOI 的 DataCite 数据/软件记录 |
| ORCID：Rui Meng | [0009-0003-9808-810X works](https://pub.orcid.org/v3.0/0009-0003-9808-810X/works) | 1 项，即论文 DOI；无数据或代码作品 |
| ORCID：Bifeng Song | [0000-0002-1511-8381 works](https://pub.orcid.org/v3.0/0000-0002-1511-8381/works) | 公开 works 返回 0 项 |
| ORCID：Jianlin Xuan | [0000-0001-7550-7306 works](https://pub.orcid.org/v3.0/0000-0001-7550-7306/works) | 9 项，包含论文 DOI；未见关联数据或代码作品 |
| ORCID：Yugang Zhang | [0000-0001-5358-9283 works](https://pub.orcid.org/v3.0/0000-0001-5358-9283/works) | 公开 works 返回 0 项 |
| NWPU Pure | [官方门户](https://pure.nwpu.edu.cn/en/)；[advanced search](https://pure.nwpu.edu.cn/en/searchAll/advanced/)；精确 DOI、标题、四作者组合 | advanced search 被 Cloudflare captcha 阻断；站点限定精确 DOI/标题检索未返回本论文的数据或代码资产 |
| GitHub repositories | [exact DOI repository search](https://github.com/search?q=%2210.3390%2Fdrones9080535%22&type=repositories)；另查 `"drones-09-00535"` 与精确标题 | 三个 repository 查询均为 0；匿名 REST code search 要求认证，因此不能声称已穷尽私有仓库或全部代码内容 |
| Preprints.org | 精确 DOI、精确标题、作者组合的站点限定检索 | 未检出对应 preprint 或附件 |
| Zenodo / Figshare | 精确 DOI、精确标题、作者组合的站点限定检索；同时由 DataCite 查询覆盖带 DOI 的公开记录 | 未检出对应公开 dataset/code 记录 |

检索边界：

- 覆盖公开、可索引资产；不覆盖私有仓库、未公开实验盘、邮件附件或搜索引擎
  尚未索引的对象。
- 搜索结果中只有论文页面、正式 PDF 及指向出版商的副本；副本没有高于
  Version of Record 的工况元数据权威性，故未用于裁决。

## 6. 本地输入及可复现数值推断

| 本地资产 | SHA-256 | 角色 |
|---|---|---|
| `platform/docs/data.md` | `ca4274e0b5c4af4f8fa526f2403579ee5eda1a1cdceabfac5682286f8c3cf3a1` | 官方 Figure 17/18/19 的本地数字化曲线 |
| `platform/fig171819_benchmark.py` | `ae18ef391d1e68f114911d8d0a7341d3ae43752197a9009d6fb639ea01007179` | 曲线身份、单位换算和基准解析 |
| `researchpaper/Meng2025_Drones_FlappingTwist_RoboEagle_SOURCE.pdf` | `eccaf750707a693fd58c0e38476a2b8ce2c694bfbf40b910f3bdf10017aa0a66` | 与官方 PDF 字节哈希一致的一手论文 |

以下全部是 I 级推断。做法是把待比较的两条数字化曲线分别线性插值到论文的
共同名义扭转网格

```text
[0, 5, 10, 15, 20, 22.5, 25, 27.5, 30, 35, 40, 45] deg
```

再计算 12 点平均绝对差：

```text
MAE_g = mean_i |F_19(x_i) - F_reference,f(x_i)|
MAE_N = MAE_g × 9.80665e-3
```

这比较的是论文内部重复工况的一致性，不是用 FLUXV 模型反推实验条件。

### 6.1 AoA=5°：Fig. 19(c,d) 对 Fig. 17 全曲线

两组公开条件均为 `U=8 m/s`、`AoA=5°`、扭转幅值扫描；Fig. 17 额外明确标出
五条频率，因此构成最直接的内部冗余。

| 候选频率 | Fig.19(c) T 对 Fig.17(a) T，MAE | Fig.19(d) L 对 Fig.17(b) L，MAE |
|---:|---:|---:|
| 1.4 Hz | 56.213 g = 0.551266 N | 68.339 g = 0.670175 N |
| 1.7 Hz | 34.392 g = 0.337272 N | 39.779 g = 0.390099 N |
| **2.0 Hz** | **15.470 g = 0.151712 N** | **15.750 g = 0.154452 N** |
| 2.3 Hz | 46.024 g = 0.451340 N | 28.720 g = 0.281642 N |
| 2.6 Hz | 92.451 g = 0.906636 N | 44.571 g = 0.437097 N |

这一证据对两个通道都把 2.0 Hz 排在第一。

### 6.2 AoA=5°：再对 Fig. 18(c,d) 的 U=8 全曲线

Fig. 18(c,d) 也提供 `U=8 m/s`、`AoA=5°` 的显式频率扭转扫，形成第二个
论文内部冗余。

| 候选频率 | Fig.19(c) T 对 Fig.18(c) T，MAE | Fig.19(d) L 对 Fig.18(d) L，MAE |
|---:|---:|---:|
| **2.0 Hz** | **12.768 g = 0.125209 N** | **29.756 g = 0.291804 N** |
| 2.3 Hz | 49.362 g = 0.484075 N | 33.949 g = 0.332930 N |
| 2.6 Hz | 98.899 g = 0.969865 N | 33.174 g = 0.325327 N |

T 通道再次强指 2.0 Hz；L 通道虽仍以 2.0 Hz 最小，但 2.0/2.3/2.6 的
差距很小，识别力弱。

### 6.3 同一 Fig. 19 的 `twist=22.5°` 跨 AoA 交点

若 (a,b) 频率扫与 (c,d) 扭转扫来自同一物理工况，四个 AoA 在
`twist=22.5°` 应各自相交。对 AoA=0/5/10/15 的绝对差取平均：

| 候选频率 | T：19(c) 对 19(a) | L：19(d) 对 19(b) |
|---:|---:|---:|
| 2.0 Hz | **15.058 g = 0.147666 N** | 63.455 g = 0.622277 N |
| 2.3 Hz | 46.462 g = 0.455639 N | 37.478 g = 0.367530 N |
| 2.5 Hz（图中标记） | 76.260 g = 0.747856 N | **22.570 g = 0.221339 N** |
| 2.6 Hz | 91.456 g = 0.896873 N | 29.357 g = 0.287891 N |

冲突是实质性的：

- T 的整曲线和交点都强指 2.0 Hz；
- L 的整曲线最接近 2.0 Hz，但同图交点明显更接近 2.5/2.6 Hz；
- 各 AoA 单点也不完全一致：T 的 AoA=0/5/10 偏 2.0，而 AoA=15 更接近
  1.4；L 的 AoA=0/5/10 偏高频，AoA=15 更接近 2.5。

因此数据可能包含面板拼接、重复试验差异、数字化误差、caption/通道装配错误，
或 c/d 并非同一固定频率。公开资料不足以区分这些原因。

## 7. 184 与 217 的真实 condition-union 含义

令：

- `C` 为 Fig.17 + Fig.18 + Fig.19(a,b) 已确认的 solver 条件集合，
  `|C|=151`；
- `S(f)` 为一个固定频率下 Fig.19(c/d) 所需的
  `4 AoA × 12 twist = 48` 个条件。

当 `f` 属于当前 confirmed 离散频率集合时，
`S(f)` 与 `C` 重合 15 个条件：

- AoA=5° 的整条 12 点扭转扫；
- 其余 AoA=0/10/15 在 twist=22.5° 的 3 点；
- AoA=5°、twist=22.5° 已包含在前 12 点中，不重复计数。

### 7.1 c/d 同频

两个力通道共享同一个 solver 支撑集合 `S(f)`：

```text
|C ∪ S(f)| = 151 + (48 - 15) = 184
```

所以 **184 隐含“c/d 同频”假设**，不是原始论文直接给出的工况总数。

### 7.2 c/d 分频

若 T 与 L 使用不同频率 `f_T != f_L`，两个 48 点集合彼此不重合，而各自与
`C` 重合 15 点：

```text
|C ∪ S(f_T) ∪ S(f_L)|
  = 151 + (48 - 15) + (48 - 15)
  = 217
```

因此作者若确认分频，完整端点应改为 **217 unique solver conditions**；
曲线数和实验测点数仍分别是 50 和 530。

### 7.3 离散频率集合之外的保留项

上述 184/217 推导假设作者确认的频率属于当前 confirmed solver 集合
`{1.4,1.7,2.0,2.3,2.6}`。由于 Figure 19(a,b) 还画出 1.5/2.5 Hz 标记，
若作者确认 c/d 使用这类集合外频率，必须重新计算交集：

- 同频且 `f not in C`：`151+48=199`；
- 分频，一条在 `C`、一条不在：`151+33+48=232`；
- 两条均不在 `C` 且彼此不同：`151+48+48=247`。

所以最终权威 endpoint 必须由作者工况表生成，而不能预先锁死为 184 或 217。

## 8. 给通讯作者的最小两问（不实际发送）

通讯作者：Jianlin Xuan，`xuan@nwpu.edu.cn`。以下恰为两问。

1. **中文**：请确认 Figure 19(c)（图内纵轴为 `T_net`）扭转幅值扫描采用的固定扑动频率是多少 Hz？

   **English**: What fixed flapping frequency (Hz) was used for the twist-amplitude sweep in Figure 19(c), whose plotted y-axis is `T_net`?

2. **中文**：请确认 Figure 19(d)（图内纵轴为 `Lift`）是否与 Figure 19(c) 使用完全相同的固定频率；若不同，请分别给出两者频率，并确认 caption 中 (c)/(d) 的 Lift 与 Net thrust 描述是否互换？

   **English**: Did Figure 19(d), whose plotted y-axis is `Lift`, use exactly the same fixed frequency as Figure 19(c)? If not, please provide both frequencies and confirm whether the Lift/Net-thrust descriptions of panels (c) and (d) were interchanged in the caption.

## 9. 请求的原始工况表最小字段

作者回复若只给一句频率，能解锁身份，但不足以审计完整数据合同。建议请求一张
每个测量工况一行的表，至少包含：

| 字段 | 最低要求 / 用途 |
|---|---|
| `figure_panel` | `19c` 或 `19d` |
| `plotted_force_channel` | `T_net` 或 `Lift`，用于解决 caption 互换 |
| `curve_id` / `run_id` | 将面板曲线映射回原始试验 |
| `AoA_deg` | 0、5、10、15 |
| `U_m_s` | 预期 8，但需由原始表确认 |
| `flapping_frequency_Hz` | 本次身份门的关键字段；保留原始精度 |
| `nominal_twist_amplitude_deg` | 每个扭转扫点；同时注明其是总幅值、峰值还是峰峰值 |
| `flapping_amplitude_deg` | 数值及幅值定义 |
| `twist_phase_deg` | 扭转与扑动的相位约定 |
| `wing_configuration_id` | 刚性翼/机构配置及是否在 c/d 间相同 |
| `repeat_index` | 重复试验编号，区分重测差异与拼图错误 |
| `averaging_window_cycles` | 周期平均使用的周期数或起止区间 |
| `Fx_raw_N`, `Fz_raw_N` | 若可提供，保留传感器原始体轴均值 |
| `Lift_N`, `T_net_N` | 最终风轴结果，注明符号和单位 |
| `tare_gravity_correction` | 空载、重力与 Equation (11) 变换版本 |
| `timestamp_or_source_file` | 追溯到原始采集文件 |

## 10. 当前可执行裁决

```text
fig19_cd_frequency.status = unresolved
H_shared_2p0.status = internal_inference_only
promotion_eligible = false
author_metadata_required = true
```

在作者原始工况表或书面澄清到达前：

- Fig. 19(c,d) 可用于展示条件性敏感性，但不得参与 claim 病因排序、参数选择、
  模型晋升或证伪；
- confirmed 主线保持 151 个唯一 solver 条件；
- “184 完整测试”必须明确写成 shared-frequency conditional；
- 若要报告分频假设，则报告 217，而不是仍沿用 184。
