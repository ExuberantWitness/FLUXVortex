# L-B 集成到 UVLM 环格的先例与做法(LB-R-C,2026-07-20)

统一动态失速闭合升级的 **Part C(集成)**,来源:专门调研代理(CONFIRMED=读一手源摘录,
PLAUSIBLE=推断,UNKNOWN=未找到;WebFetch 该轮不可用,CONFIRMED 基于搜索表面摘录非全文)。

## 核心裁定:REPLACE 语义 + 怎么装不双计

**风机 AeroDyn/OpenFAST(BEM):非定常模型 REPLACE 静态极曲线截面力,不在 wake 力之上叠加。**
wake 只供诱导速度(α_eff 修正),L-B/HGM 从修正 α 算全截面 CN/CC/CM。BEM 无独立 wake 力,
无双计。CONFIRMED(OpenFAST UA 理论页 + NREL/TP-5000-66347 Damiani & Hayman 2019)。
HGM 4 态:α_E=α₃₄(1−A₁−A₂)+x₁+x₂;Kirchhoff CN=C_nα(α−α₀)[(1+√f)/2]²;
弦向 CC=η_e C_nα(α−α₀)√f tanα;涡升力 CC=CC^fs+CN^v tan(α_e)(1−τ_V/T_VL)。

**双计只在宿主 wake 显式对流脱涡时出现(free-wake 升力线 / UVLM)**:L-B 附着段的环量
deficiency 函数(Theodorsen/Wagner=脱涡记忆)与 free-wake 已用 Biot-Savart 解析的脱涡诱导
重复。**修法 = 剥掉 L-B 附着环量项,只留分离(Kirchhoff f)+ LEV 涡升 CN^v + (可选)附加质量。**
CONFIRMED(Wendler et al. GT2016-57184,QBlade ATEFlap"excludes contribution of the wake in
the attached flow region")——这是与本案最直接的先例。

**附加质量可能也在 UVLM(非定常 Bernoulli)里**,需一并去掉非环量项——QBlade 只剥环量脱涡项,
留冲量项;是否剥附加质量 = PLAUSIBLE(由本实现定)。

## 对本实现的装配建议(PLAUSIBLE 重建,非文献逐字)

UVLM 已算环量升力(环上 KJ)+ 附加质量。故:
1. 剥掉 L-B 附着环量态 + 冲量/附加质量项;
2. **用 Kirchhoff 分离因子 f(α) 乘性缩放 UVLM 法向力**;
3. **加 LEV 涡升 CN^v + 弦向/前缘吸力修正 CC=η·C_nα(α−α₀)√f·tanα**(FAST-8 CC 式 +
   QBlade 剥附着环量规则的组合;f 缩放 UVLM 环量是标准 Kirchhoff 混合,见 UVLM-K
   AIAA 2022-0132)。**确切 add-vs-scale 约定 = 实现决策,对 UIUC 极曲线验证,非文献定案。**

## 其他集成形态

- **DUST(PoliMi)**:UVLM↔2D 粘性极曲线迭代有效α(van Dam 环),Cl_inv/2π 查极曲线+松弛
  α_local;失速用 Kirchhoff 混合(UVLM-K)。无动态失速时间滞后;是否加 L-B 层 = UNKNOWN。
- **CAMRAD II**:L-B 作为二阶升力线+翼型表的 per-section 选项,配 free wake;附着模型=
  可压薄翼近似,与 3D trailing-vortex 分离处理,两者都留。显式减去 free-wake 脱涡诱导
  防双计 = UNKNOWN。
- **扑翼**:Djojodihardjo 组(Ramli/Djojodihardjo,UPM)分条+ L-B 修正+LE 吸力+粘性,
  对 CFD 验证——结构与本案同但 **strip theory 非 UVLM 耦合**(无本案双计问题)。
  Ansari/Żbikowski/Knowles 2006 = 物理 LEV 环量/尾迹积分模型(非 L-B),高 k 区替代路线。

## ★ L-B 在本 regime 的失效边界(决定主攻方向)

L-B 是 2D、轻失速、中 k、per-翼型拟合模型。CONFIRMED 失效:
(a) **高 k 过预测失速延迟**(TNO:k≈0.065 即超 max-lift 角 ~5°,我们 k=0.13-0.39 是 2-6 倍);
(b) 无二次涡脱(后期 CN 峰漏);(c) **深失速/负α再附差**(直接命中我们上冲程负α);
(d) 3D 旋翼失效(UH-60A 俯仰力矩峰漏);(e) r′>0.05 即失真。
来源:RSER 2024 批判综述、TNO、Nguyen & Johnson UH-60A、JFS 2021。

**→ 含义(重要)**:RoboEagle k 至 0.39 已远超 L-B 验证包络。**LEVP 主导升力应作主物理
(Ansari 物理路线),L-B 只用于后缘分离/再附调制,不作主升力模型。** 否则会在 k 验证带外
重蹈"半经验失速延迟过预测"。

## 高价值一手源(待全文)

1. OpenFAST theory_ua + NREL/TP-5000-66347(确切 CN/CC/LEV 式);
2. Wendler GT2016-57184(双计修法,最直接类比);
3. W. Johnson CIIaerodynamics.pdf(CAMRAD II 架构);
4. Ramli/Djojodihardjo UPM 2015(分条+L-B 扑翼先例);
5. Ansari/Żbikowski/Knowles 2006 两部分(高 k 物理 LEV 替代)。
