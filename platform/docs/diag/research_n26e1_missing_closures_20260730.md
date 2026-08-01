# N2.6e1 缺失闭合的一手来源审计

日期：2026-07-30  
对象：`N2.6e1`，Riziotis--Voutsinas 二维 strong
viscous--inviscid interaction / double-wake 来源方法  
状态：`FORMULA-LAYER GO / ORIGINAL-PROGRAM IDENTITY NO-GO /
REMESH-STATE-TRANSFER UNDISCLOSED / TARGET FIT PROHIBITED`

## 1. 裁决

本次检索改变了此前“2008 论文未给出，因此只能借用 Ramos
二手闭合”的判断。Riziotis 本人 2003 年博士论文第 7 章是 2008
论文方法的上游一手来源，公开了以下此前阻断项：

1. 法向动量厚度 `Theta_n` 的积分定义和 East 一阶闭合；
2. `H*`、`H**`、`Cf`、`CD` 的层流/湍流闭合；
3. `e^N` 转捩方程、`n_crit=9` 和
   `sqrt(C_tau,tr)=0.7 sqrt(C_tau,eq)`；
4. double-wake 的非定常 Bernoulli、势函数参考、总压差 `Delta h`
   以及压力/摩擦载荷积分；
5. 分离点是 `Cf=0` 的位置，且分离点更新发生在边界层收敛之后。

因此：

- **来源方程层：GO。** `N2.6e1` 不再需要用 Ramos (2011)
  作为这些闭合公式的主来源。
- **2008 原程序同一性：仍为 NO-GO。** 博士论文是同一作者、同一方法谱系的
  直接披露，但不能证明 2008 可执行程序逐行使用了完全相同版本。
- **重网格状态转移：仍为 NO-GO。** 2008 论文和 2003 博士论文均未说明整翼重网格后
  `delta* / theta / n / C_tau` 如何投影。不得把自选插值称为来源方法。
- **Figure 12 实验工况：试验为 untripped。** 完整 run ID
  `05012721` 中 `c=0`，即 Orthodox experiment；不是 trip wire
  (`c=1`) 或 roughness transition strips (`c=8`)。
- **Figure 12 自由来流湍流强度：不可识别。** Glasgow 报告没有披露 TI；
  不得由 Figure 12 或 Fig17/18/19 反求 TI 或 `Ncrit`。
- **唯一 H0 决策：采用 `400/Re_theta`。** 这是根据开发者源代码、分段连续性和
  后续同行评审文献解决论文排印错误，不是根据任何响应曲线选参。

本报告只授权一个独立 shadow/source-response 实现进入数值收敛门。
它不授权修改 V4.1，也不授权查看 Fig17/18/19 后选择公式、常数、
网格或转捩参数。

## 2. 来源身份、稳定链接和审计哈希

### 2.1 直接来源

| 资产 | 稳定入口 | 本次审计副本与 SHA256 | 证据等级 |
|---|---|---|---|
| Riziotis & Voutsinas (2008), *Dynamic stall modelling on airfoils based on strong viscous--inviscid interaction coupling* | DOI: <https://doi.org/10.1002/fld.1525>; NTUA 元数据：<https://dspace.lib.ntua.gr/xmlui/handle/123456789/18752> | 临时审计副本 `/tmp/riziotis2008_full.pdf`; `cc4970b38b3586affc4805a84e526fcb0049ba2dfa42219c01379e2a8f48fa84` | 同行评审主方法来源 |
| Riziotis (2003), *Aerodynamic and aeroelastic analysis of dynamic stall on wind turbine rotors* | 国家论文库：<https://www.didaktorika.gr/eadd/handle/10442/16690>; DOI: <https://doi.org/10.12681/eadd/16690>; Handle: <http://hdl.handle.net/10442/hedi/16690>; 官方阅读器：<https://freader.ekt.gr/eadd/index.php?doc=16690&lang=el> | 官方 reader 配置临时副本 `/tmp/riziotis_book_config.js`; `27bc7de99609a3540821ce111a25693f10e6ae5847a03e55a0d75b1385a3b4cc` | 同一作者、同一方法的上游一手详细披露；不是 bitwise 身份证明 |
| Galbraith, Gracey & Leitch (1992), G.U. Aero Report 9221 | 记录：<https://eprints.gla.ac.uk/262672/>; PDF：<https://eprints.gla.ac.uk/262672/1/262672.pdf> | 临时审计副本 `/tmp/glasgow9221.pdf`; `0eb8842385c9f7e85c10826b87dac726cea4c707a54aac62609ed1c7797de8e9` | Figure 12 实验元数据一手来源 |
| XFOIL 6.99 开发者发布 | MIT：<https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz> | `/tmp/xfoil6.99.tgz`: `5c0250643f52ce0e75d7338ae2504ce7907f2d49a30f921826717b8ac12ebe40`; `Xfoil/src/xblsys.f`: `da0fa508a9f4a739eac0bcdc9ff3e9f28d8f217929a0ab61b529983ca6a8d46d` | 上游开发者代码；本报告只用它裁决 `H0` 排印冲突 |
| Agrawal et al. (2024), *An extension of Thwaites' method for turbulent boundary layers* | DOI：<https://doi.org/10.1017/flo.2024.27>; Cambridge：<https://www.cambridge.org/core/journals/flow/article/an-extension-of-thwaites-method-for-turbulent-boundary-layers/A372BFAC4896C1E126A6BC75664DBF69> | 公开同行评审正文，Eqs. (3.5)--(3.6) | `H0=3+400/Re_theta` 的独立同行评审佐证 |

`/tmp` 路径只记录本次审计工作副本，不是持久资产。持久可复核身份由稳定链接、
页号和 SHA256 组成；未把 118 MB 论文复制进仓库。

### 2.2 论文官方 reader 精确页资产

官方页图 URL 模式为：

```text
https://freader.ekt.gr/getfile.php?lib=eadd&path=large&doc=HqfTEv0%3D&item=<ITEM>.jpg
```

| 印刷页 | item | 本报告使用的内容 | 页图 SHA256 |
|---|---:|---|---|
| 7.5 | 249 | `(s,n)` 运动坐标、`Omega` 和 Coriolis 符号 | `9a2b6519ae60c51f25919b57de65e70166fadcf569b8248a3ec058378bf40cd0` |
| 7.8 | 252 | `Theta_n`，Eq. (7.22) | `d6c33640dd9cdfeb8f6ebbbce71f2e50d34b719a612c96d92f999d7592e9c0ae` |
| 7.9 | 253 | 完整 kinetic-energy equation，Eqs. (7.30)--(7.32) | `c2f8924c60e1b663ebcee13a7ae1d9f63b004b8b13c514c1a0f106e7ec8ab06e` |
| 7.10 | 254 | `e^N`，Eqs. (7.33)--(7.35)，`n_crit=9` | `4bdb86b72b72ebdc883281d1692f591028afc16fc2dc6ec501cde14b79ba6d5b` |
| 7.11 | 255 | Eqs. (7.36)--(7.42) | `d65ff390b56638d0c999ea2c1d9ad86026efaed66e8cf5559c47018540e5be4d` |
| 7.12 | 256 | Eqs. (7.43)--(7.48)，含 East `Theta_n` 闭合 | `7267da47f6e5518d546b8c80fa573487c9db85aa4a119c766bc9270ff29dcffe` |
| 7.14 | 258 | `Hk`、`H**`，Eqs. (7.54)--(7.56) | `632a5a17378f53bcd0db64538f5f2938d89256a94b1e6232adb896f571695f26` |
| 7.15 | 259 | 层流闭合和湍流 `Cf`，Eqs. (7.57)--(7.61) | `c3ddb6ceed9dcea45204d80adf66bc2d71a0fce5488ae87749096c60777d584c` |
| 7.16 | 260 | 湍流 `H*` 和原文 `H0`，Eq. (7.62) | `3509a61d2b1ee4b213d01f2ca43f409ffb602376ca3085a549e525aa0a594c03` |
| 7.22 | 266 | double-wake `Delta h`，Eqs. (7.85)--(7.88) | `ba76fa1c7170ba7bca06dff6204cfd4da3604c43c6b03fd40d01a8cf6c3f0441` |
| 7.23 | 267 | Kelvin/新生尾迹强度，Eqs. (7.89)--(7.98) | `77b43c615f1d9837e7e6b65c8086d11f2aae146dc5310ff7b4eddf338e7fefbf` |
| 7.28 | 272 | 面板中点 control point 与尾缘加密要求 | `11ee8cf57fcbd34362912724111bc6a65b07c6158b74c51cc29028765bc6eb14` |
| 7.36 | 280 | 转捩区间离散及线性插值 | `98633e5d17d6832b052faa7d8f09f0c99fc7a3b7f2eab51134d846e453a9cc7e` |
| 7.37 | 281 | `sqrt(C_tau,tr)=0.7 sqrt(C_tau,eq)` | `0cff0cefd1e009e5f83a81b0ae8c5c152853973cc0a17f74a60f5ecad5f8741d` |
| 7.40 | 284 | single-wake 最近上下 control-point TE 速度，Eq. (7.126) | `f36b0d8ded6cce61e7e272d5ea9a98826b7a6dc5817b2215fa52cccb338bb6e0` |
| 7.42 | 286 | double-wake 最近 control-point 释放强度，Eqs. (7.131)--(7.133) | `0301924c92a224d43aff45623f42c059b8ba5bcdc7d2bd5d9c9fbd7b2f0ab71b` |
| 7.48 | 292 | 每时步近尾迹/分离点内迭代，从上一步初始化 | `85948a0ca79d2e863f7ac6f9ad467c8f67cffb196deb843ac7b9860faa16b7aa` |
| 7.49 | 293 | 分离尾迹置于 `Cf=0` 位置 | `1f8806e212a8772581f7a30755cfd8ec2e596d19aa5093cdad7f23c9c375b739` |
| 7.50 | 294 | 非定常 Bernoulli，Eqs. (7.143)--(7.147) | `691b32c8129f13c0c81529cebac3d2f347c5d3a3d889d6aeb59831346d91234a` |
| 7.51 | 295 | 势跳、dipole gauge、后向时间差分，Eqs. (7.148)--(7.151) | `e5cd65fc2208629493a836b67299735e18bb62e05de59f4aca11b57096bef9fe` |
| 7.52 | 296 | 压力力和摩擦力沿翼面一次积分 | `047884a45db34f711a23de1834f709d1146aeb4b5007675fdefc4e6eb0624a36` |

### 2.3 尾缘速度的离散身份

Riziotis 来源没有把 Eqs. (7.94)--(7.95) 中的
`\bar u_{ew}^{+/-}` 定义成数学 cusp 上的单侧连续极限。其离散身份是：

1. 印刷页 7.28 将每个直线面板的中点定义为 control point，并明确要求
   尾缘采用更密的离散，以更好逼近尾缘速度及尾迹量；
2. 印刷页 7.40，Eq. (7.126) 直接用最靠近尾缘的上、下两个 control
   point 相对切向速度近似尾缘两侧速度，并说明翼面速度只在 control
   points 计算；
3. 印刷页 7.42，Eq. (7.131) 在 double-wake 离散中继续采用相同定义；
4. 2008 论文 article p.190 Figure 3 在尾缘上下相邻位置标出 `(c.p.)`。

因此 `svi_dw_unsteady_outer_2d.py` 现有“最近上下 panel midpoint”采样与
来源一致；把它改成 cusp 外推、固定距离采样或 finite-part 都将是新的数值
claim，不能冒充来源复现。来源未披露的是原程序的面板数和具体尾缘聚类率，
故必须通过独立网格渐近门，而不能用代数 Kelvin 闭合替代。

## 3. `Theta_n`、pitching term 和符号

### 3.1 直接定义

博士论文印刷页 7.8，Eq. (7.22)：

```math
\Theta_n =
\frac{1}{\rho_e u_e^2}
\int_0^\delta
\left(\rho_e u_e v_e-\rho u v\right)\,dn .
```

这里 `s` 是沿翼面边界层积分方向，`n` 从壁面向流体为正，`u,v`
是随体曲线坐标中的切向、法向速度，`u_e,v_e` 是边界层边缘值。
这是法向动量通量亏损，不是可调 pitching gain。

博士论文印刷页 7.12，Eq. (7.48)，明确称为 East (1981) 的
“first approximation”：

```math
\Theta_n \simeq (\theta+\delta^*)\frac{d\delta^*}{ds}.
```

因此该闭合的符号由 `d(delta*)/ds` 决定；不得取绝对值、不得限为正数，
也不得以目标载荷缩放。它是近似式而非精确恒等式。

### 3.2 完整 kinetic-energy equation 中的位置

博士论文印刷页 7.9，Eq. (7.32)：

```math
\begin{aligned}
&\frac{1}{\rho_e u_e^3}\frac{d}{dt}(\rho_e u_e^2\theta)
+\frac{1}{\rho_e u_e}\frac{d}{dt}(\rho_e\delta^*)
+\frac{2}{u_e^2}\frac{du_e}{dt}H^{**}\theta \\
&-\frac{H^*}{\rho_e u_e^2}
  \frac{d}{dt}(\rho_e u_e\delta^*)
+\theta\frac{dH^*}{ds}
+\left[2H^{**}+H^*(1-H)\right]
  \frac{\theta}{u_e}\frac{du_e}{ds}
-\frac{4\Omega}{u_e}\Theta_n \\
&=2C_D+\frac{2a}{u_e^2}\delta^*
-H^*\frac{C_f}{2}.
\end{aligned}
```

旋转项的来源符号是：

```math
-\frac{4\Omega}{u_e}\Theta_n .
```

博士论文印刷页 7.5 的切向动量方程中 Coriolis 项同样为 `-2 rho Omega v`。
所以 `Omega` 必须是与 `(s,n)` 和整体俯仰坐标一致的**带符号角速度**，
不能使用 `abs(pitch_rate)`。来源没有用一句独立文字指定 FLUXV
全局俯仰角的正方向；从 FLUXV 角度到 `Omega` 的符号映射必须作为坐标约定
显式测试，不能靠 Figure 12 的误差方向反选。

同页组给出的随体加速度项不是自由参数：

```math
a =
\Omega^2 R_{OP,s}
+\frac{\partial\Omega}{\partial t}R_{OP,n}
-\frac{\partial^2 R_{O,s}}{\partial t^2},
```

并在薄层内使用含局部 `n` 的近似形式。`a` 项与
`-(4 Omega/u_e) Theta_n` 是两个不同来源，不能相互吸收。

### 3.3 适用边界

直接来源的适用条件是二维、薄、沿翼面发展的积分边界层，在随运动物体的
曲线坐标中求解；2008 论文明确把实际应用限制为不可压/低速风机流。
在 `u_e -> 0` 的驻点，`Theta_n` 定义及 Eq. (7.32) 的归一化退化，
需要来源方法自己的驻点初始化。用任意 epsilon floor 改写方程不属于
来源闭合。

## 4. `H / H* / H** / Cf / CD` 完整闭合

### 4.1 定义和共同压缩性变换

```math
H=\frac{\delta^*}{\theta},\qquad
H^*=\frac{\theta^*}{\theta},\qquad
H^{**}=\frac{\delta^{**}}{\theta},\qquad
Re_\theta=\frac{\rho_e u_e\theta}{\mu_e}.
```

相应厚度定义为：

```math
\delta^*=\int_0^\delta
\left(1-\frac{\rho u}{\rho_eu_e}\right)dn,
```

```math
\theta=\int_0^\delta
\frac{\rho u}{\rho_eu_e}\left(1-\frac{u}{u_e}\right)dn,
```

```math
\theta^*=\int_0^\delta
\frac{\rho u}{\rho_eu_e}\left(1-\frac{u^2}{u_e^2}\right)dn,
```

```math
\delta^{**}=\int_0^\delta
\frac{\rho u}{\rho_eu_e}\left(\frac{\rho_e}{\rho}-1\right)dn.
```

博士论文印刷页 7.14，Eqs. (7.55)--(7.56)：

```math
H_k=\frac{H-0.290M_e^2}{1+0.113M_e^2},
```

```math
H^{**}=\left(\frac{0.064}{H_k-0.8}+0.251\right)M_e^2.
```

对于本来源的不可压算例，`M_e=0`，所以 `H_k=H`、`H**=0`；
仍应保留这一有来源的退化，而不是删除状态后再增加经验项。

### 4.2 层流闭合

博士论文印刷页 7.15，Eqs. (7.57)--(7.59)：

```math
H^*=
\begin{cases}
1.515+0.076(4-H_k)^2/H_k,&H_k<4,\\
1.515+0.040(H_k-4)^2/H_k,&H_k>4,
\end{cases}
```

```math
Re_\theta\frac{C_f}{2}=
\begin{cases}
-0.067+0.01977(7.4-H_k)^2/(H_k-1),&H_k<7.4,\\
-0.067+0.022\left(1-\frac{1.4}{H_k-6}\right)^2,&H_k>7.4,
\end{cases}
```

```math
Re_\theta\frac{2C_D}{H^*}=
\begin{cases}
0.207+0.00205(4-H_k)^{5.5},&H_k<4,\\
0.207-\dfrac{0.003(H_k-4)^2}
{1+0.02(H_k-4)^2},&H_k>4.
\end{cases}
```

### 4.3 湍流闭合

博士论文印刷页 7.15，Eq. (7.61)：

```math
\begin{aligned}
F_c C_f={}&0.3e^{-1.33H_k}
\left[\log_{10}\left(\frac{Re_\theta}{F_c}\right)\right]^{
-1.74-0.31H_k}\\
&+0.00011\left[
\tanh\left(4-\frac{H_k}{0.875}\right)-1\right],
\qquad
F_c=(1+0.2M_e^2)^{1/2}.
\end{aligned}
```

博士论文印刷页 7.16，Eq. (7.62)；注意常数为 `1.505`，不是层流式中的
`1.515`：

```math
H^*=
\begin{cases}
1.505+\dfrac{4}{Re_\theta}
+\left(0.165-\dfrac{1.6}{\sqrt{Re_\theta}}\right)
\dfrac{(H_0-H_k)^{1.6}}{H_k},&H_k<H_0,\\[6pt]
1.505+\dfrac{4}{Re_\theta}
+(H_k-H_0)^2
\left[
\dfrac{0.04}{H_k}
+\dfrac{0.007\ln Re_\theta}
{\left(H_k-H_0+4/\ln Re_\theta\right)^2}
\right],&H_k>H_0.
\end{cases}
```

湍流耗散不是一个静态 `CD(H)` 补丁。博士论文印刷页 7.11--7.12，
Eqs. (7.41)--(7.45)：

```math
C_D=\frac{C_f}{2}U_s+C_\tau(1-U_s),
```

```math
U_s=\frac{H^*}{2}
\left[1-\frac{4}{3}\frac{H_k-1}{H}\right],
```

```math
\frac{\delta}{C_\tau}\frac{dC_\tau}{ds}
=5.6\left(\sqrt{C_{\tau,eq}}-\sqrt{C_\tau}\right)
+2\delta\left\{
\frac{4}{3\delta^*}
\left[\frac{C_f}{2}
-\left(\frac{H_k-1}{6.7H_k}\right)^2\right]
-\frac{1}{u_e}\frac{du_e}{ds}
\right\},
```

```math
\delta=\theta\left(3.15+\frac{1.72}{H_k-1}\right)+\delta^*,
```

```math
C_{\tau,eq}
=H^*\frac{0.015}{1-U_s}
\frac{(H_k-1)^3}{H_k^2H}.
```

因此 `C_tau` 是有空间记忆的状态，`CD` 是它的派生量；不能用一个
瞬时拟合阻力项代替。

## 5. 转捩和 `C_tau` 初始化

博士论文印刷页 7.10，Eqs. (7.33)--(7.35)：

```math
\tilde n=
\frac{d\tilde n}{dRe_\theta}(H_k)
\left[Re_\theta-Re_{\theta0}(H_k)\right],
```

```math
\frac{d\tilde n}{dRe_\theta}
=0.01\left\{
\left[2.4H_k-3.7+2.5\tanh(1.5H_k-4.65)\right]^2+0.25
\right\}^{1/2},
```

```math
\log_{10}Re_{\theta0}
=\left(\frac{1.415}{H_k-1}-0.489\right)
\tanh\left(\frac{20}{H_k-1}-12.9\right)
+\frac{3.295}{H_k-1}+0.44.
```

非相似流使用博士论文印刷页 7.11 的空间积分形式：

```math
\frac{d\tilde n}{ds}
=\frac{d\tilde n}{dRe_\theta}\frac{dRe_\theta}{ds}.
```

来源直接把转捩事件定义为：

```math
\tilde n_{tr}=9.
```

博士论文印刷页 7.36--7.37 还规定：

1. 含转捩点的离散区间按 `n=9` 分成层流、湍流两段分别求解；
2. 转捩点处 `delta*` 和 `theta` 由相邻控制点线性插值，以避免不连续；
3. 湍流最大剪切状态初始化为

```math
\sqrt{C_{\tau,tr}}=0.7\sqrt{C_{\tau,eq}}.
```

这是 Riziotis 来源方法的直接边界条件。当前 XFOIL 6.99
`xblsys.f:1391--1402` 使用的是
`CTR=CTRCON exp[-CTRCEX/(Hk-1)]`，且 `xbl.f:1587--1588`
给出 `CTRCON=1.8, CTRCEX=3.3`；那是后续 XFOIL 版本规则，
不能混入 Riziotis 候选，也不能拿它与 `0.7` 做 Figure 12 选优。

## 6. `H0=4/Re_theta` 与 `400/Re_theta` 的唯一预登记

博士论文印刷页 7.16 字面印为：

```math
H_0=
\begin{cases}
4,&Re_\theta<400,\\
3+4/Re_\theta,&Re_\theta>400.
\end{cases}
```

该字面式在阈值处从约 `4` 跳到约 `3.01`。相反：

- MIT 官方 XFOIL 6.99 `Xfoil/src/xblsys.f:2399--2405` 为
  `H0=3+400/Re_theta`（`Re_theta>400`），否则 `H0=4`；
- 两段在 `Re_theta=400` 连续；
- Agrawal et al. (2024), Eqs. (3.5)--(3.6) 在同行评审正文中明确复述
  `H0=4` for `Re_theta<400`, otherwise `3+400/Re_theta`。

所以唯一实现决策预登记为：

```math
H_0=
\begin{cases}
4,&Re_\theta\le 400,\\
3+400/Re_\theta,&Re_\theta>400.
\end{cases}
```

这被登记为论文漏印两个零的勘误裁决。禁止实现 `4/Re_theta`
和 `400/Re_theta` 两个候选后，用 Figure 12 或 Fig17/18/19 选择。

### 6.1 与 XFOIL 6.99 的逐式版本边界

对 MIT 官方 `xblsys.f` 的逐式核对表明，除 `H0` 外不能把差异解释为
论文排印错误：

| thesis equation | Riziotis 2003 | XFOIL 6.99 | e1 冻结裁决 |
|---|---|---|---|
| 7.55 `Hk` | `0.290/0.113` | `HKIN` 相同 | 原式 |
| 7.56 `H**` | `0.064/0.251` | `HCT` 相同 | 原式 |
| 7.57 laminar `H*` | `Hk=4`, `1.515`, `0.076/0.040` | 新相关式 `Hk=4.35`, `1.528` 等 | thesis 旧式 |
| 7.58 laminar `Cf` | `Hk=7.4`, `0.01977/0.022` | `Hk=5.5`, `0.0727/0.015` | thesis 旧式 |
| 7.59 laminar `2CD/H*` | 分离支 `-0.003` | 分离支 `-0.0016` | thesis `-0.003` |
| 7.61 turbulent `Cf` | Swafford 式 | 代数相同但有数值 floor | 原式，无隐藏 floor |
| 7.62 turbulent `H*` | `1.505`, `0.165`, `0.04` 旧式 | 1991 后新 `HST`、`Re_theta>=200` clamp | thesis 旧式 |
| 7.41--45 `C_tau` | `1-Us` 与原 Green 输运 | `0.995-Us`、低 Re 和附加修正 | thesis 旧式 |
| transition init | `0.7 sqrt(C_tau,eq)` | `1.8 exp[-3.3/(Hk-1)]` | thesis `0.7` |

XFOIL 的版本说明明确记录过黏性闭合、转捩和低 Reynolds 数规则的更新。
因此 e1 禁止“顺手升级”为当前 XFOIL：禁止 `RTZ=max(Re_theta,200)`、
`0.995-Us`、新版 `HST`、新版层流式或新版 transition ramp。只有
`H0=3+400/Re_theta` 同时满足开发者源码、阈值连续性和独立同行评审
复述，因而是唯一授权勘误。

## 7. double-wake Bernoulli、gauge、`Delta h`

### 7.1 非定常压力式

博士论文印刷页 7.50，Eqs. (7.143)--(7.147)。以总势 `Phi` 和物体速度
`U_b` 表示：

```math
\frac{\partial\Phi}{\partial t}
+\frac{|\mathbf u_e|^2}{2}
+\mathbf u_e\cdot\mathbf U_b+\frac{p}{\rho}=c(t).
```

定义翼面相对速度
`\bar{\mathbf u}_e=\mathbf u_e-\mathbf U_b`：

```math
\frac{\partial\Phi}{\partial t}
+\frac{|\bar{\mathbf u}_e|^2}{2}
-\frac{|\mathbf U_b|^2}{2}
+\frac{p}{\rho}=c(t).
```

分离泡内部多出总压差：

```math
\frac{\partial\Phi}{\partial t}
+\frac{|\bar{\mathbf u}_e|^2}{2}
-\frac{|\mathbf U_b|^2}{2}
+\frac{p}{\rho}+\Delta h=c(t).
```

以无穷远为参考后，泡外：

```math
\frac{p-p_\infty}{\rho}
=\frac{|\mathbf U_\infty|^2}{2}
-\frac{|\bar{\mathbf u}_e|^2}{2}
+\frac{|\mathbf U_b|^2}{2}
-\frac{\partial(\Phi-\Phi_\infty)}{\partial t},
```

泡内：

```math
\frac{p-p_\infty}{\rho}
=\frac{|\mathbf U_\infty|^2}{2}
-\frac{|\bar{\mathbf u}_e|^2}{2}
+\frac{|\mathbf U_b|^2}{2}
-\frac{\partial(\Phi-\Phi_\infty)}{\partial t}
-\Delta h,
```

其中来源明确固定：

```math
\Phi_\infty=U_{\infty x}x+U_{\infty y}y.
```

所以压力需要的是 `partial_t(Phi-Phi_inf)`，不是任意 gauge 下的
`partial_t Phi`。共同的时间函数 gauge 在这一差值和势跳中消失。

### 7.2 势函数离散 gauge

来源分解：

```math
\Phi=\Phi_\infty+\phi_\sigma+\phi_\gamma+
\phi_{\gamma,\mathrm{wake}}+\phi^*.
```

博士论文印刷页 7.51，Eqs. (7.148)--(7.150)：

```math
\gamma=[u]=[\nabla_s\phi]=\nabla_s[\phi]=-\nabla_s\mu,
```

```math
\mu^{i+1}=\mu^i-\gamma^i\Delta S^i.
```

double-wake 两条新生 sheet 的积分起点均为：

```math
\mu_w^1=0,\qquad \mu_s^1=0.
```

时间导数直接规定一阶后向差分：

```math
\frac{\partial\Phi}{\partial t}
\simeq\frac{\Phi(t)-\Phi(t-\Delta t)}{\Delta t}.
```

来源未进一步说明当控制点因整翼重网格而移动时，旧势值如何重采样。
这一点与边界层状态转移一起保留为数值身份 NO-GO，不能静默解释。

### 7.3 `Delta h`

博士论文印刷页 7.22 将 `Delta h` 定义为两条尾迹分开的内、外区域的总压差。
分离点压力连续并假定分离点下游侧为相对速度驻点，Eq. (7.86)：

```math
\frac{|\bar{\mathbf u}_{es}^{+}|^2}{2}
=\frac{\partial}{\partial t}
(\Phi_s^- -\Phi_s^+)+\Delta h.
```

尾缘压力连续，Eq. (7.88)：

```math
\frac{|\bar{\mathbf u}_{ew}^{-}|^2}{2}
-\frac{|\bar{\mathbf u}_{ew}^{+}|^2}{2}
=\frac{\partial}{\partial t}
(\Phi_w^+ -\Phi_w^-)+\Delta h.
```

因此：

```math
\Delta h=
\frac{|\bar{\mathbf u}_{ew}^{-}|^2-
      |\bar{\mathbf u}_{ew}^{+}|^2}{2}
-\frac{\partial}{\partial t}(\Phi_w^+-\Phi_w^-).
```

同一套符号还给出：

```math
\gamma_w=\bar u_{ew}^{+}-\bar u_{ew}^{-},\qquad
\gamma_s=\bar u_{es}^{+},
```

并由 Eqs. (7.97)--(7.98) 保持既有 wake vorticity 的物质守恒。
`Delta h` 不是额外力通道，而是 bubble 内压力式中的总压差状态。

### 7.4 压力只积分一次

博士论文印刷页 7.52 直接说明：翼型气动载荷由压力力和摩擦力沿翼面积分得到。
将其写成守恒 traction 的实现表达为：

```math
\mathbf F=
\oint_{\partial B}
\left[-(p-p_\infty)\mathbf n+\tau_w\mathbf t\right]\,ds,
```

```math
\mathbf M_O=
\oint_{\partial B}
(\mathbf x-\mathbf x_O)\times
\left[-(p-p_\infty)\mathbf n+\tau_w\mathbf t\right]\,ds.
```

这两个向量式是对来源文字的守恒转写，不是原文编号公式；
法向取物体指向流体。关键账本约束是：由 Eqs. (7.146)--(7.147)
得到统一表面压力后只积分一次，不能再叠加一个独立
“dynamic-stall lift” 或 “LEV normal-force” 修正。

## 8. 分离判据、重网格和状态转移

### 8.1 直接披露

博士论文印刷页 7.49 明确写明：第二条涡量 sheet 放置在
`Cf` 变为零的位置，即边界层分离条件。因此此前
“Riziotis 没有打印分离判据”的表述应撤销：

```math
C_f(s_s,t)=0.
```

这只定义该二维来源方法的分离事件，不能晋升为生产三维分离流形。

博士论文印刷页 7.48 说明每个时间步：

1. 近尾迹长度/方向和分离点位置的内迭代从上一时刻状态开始；
2. 求解耦合 inviscid/IBL 方程；
3. 重新估计近尾迹几何和分离点；
4. 重复直到收敛。

2008 论文 PDF p.9 / journal p.193 又明确说明：

- `Ps` 改变时必须重网格，使 `Ps` 始终是网格点；
- 为维持合理间距，整个翼面都重网格；
- `Ps` 更新和重网格总是在边界层解收敛后执行。

### 8.2 未披露项及 NO-GO 边界

两份来源均没有说明整翼重网格后如何把旧网格上的：

```text
delta*, theta, n 或 sqrt(C_tau), 以及上一时刻 Phi
```

转移到新网格。博士论文印刷页 7.36 的线性插值只用于**一个固定离散区间内的
层流--湍流转捩点**，不能作为全翼 remesh transfer 的来源证据。

因此：

- 原程序/bitwise 身份保持 `NO-GO`；
- 不得凭空写一个插值然后标为 “Riziotis implementation”；
- 若要继续新的 source-response shadow，必须先另行预登记一个与
  Figure 12、Fig17/18/19 无关的数值转移规则，并通过独立网格/时间步
  收敛与质量、动量亏损账本守卫；
- 最自然但仍属**推断性数值候选**的是对
  `rho_e u_e delta*` 与 `rho_e u_e^2 theta` 做守恒 remap，
  在各流态区间内单调转移 `n` 或 `sqrt(C_tau)`，再按 `n=9`
  重建转捩事件。该候选尚未由本报告授权，也不能以 Figure 12
  表现来选择或调节。

## 9. Glasgow Figure 12：trip 和 TI

### 9.1 完整 run ID

Glasgow Report 9221：

- PDF physical page 15 / report printed p.7：每个试验有
  `abcdefgh` 八位 ID；
- `ab` 是模型号；
- `c=0` Orthodox，`c=1` Trip Wire，`c=8` Roughness Transition
  Strips；
- `d=1` Sinusoidal Oscillation；
- `efg` 是 run sequence，`h` 是 attempt；
- PDF physical page 91 / printed p.83：NACA0015、`c=0.55 m`
  是 Model 5；
- PDF physical page 101 / printed p.93，Table 7.5.5：
  run `12721` 为 mean `11 deg`、amplitude `8 deg`、
  `k=0.051`、`Re=1.47e6`。

所以完整 ID 为：

```text
ab c d efg h = 05 0 1 272 1 = 05012721.
```

直接结论：该实验是 `c=0` 的 **Orthodox/untripped** 工况，
不是 trip-wire 或 roughness-strip 工况。2008 论文的名义标注
`Re=1.5e6, Ma=0.12, k=0.05` 与报告的实际
`Re=1.47e6, k=0.051` 应分别保留，不能混成一个“精确”值。

### 9.2 不可识别量

对完整 Report 9221 检索：

```text
turbulence intensity
free-stream turbulence
turbulence level
```

均无披露。报告对 `c=0` 的定义只证明没有列出的 trip/hot-film/
roughness 等非标准处理，不能推出一个数值 TI。

2008 论文也没有为 Figure 12 明确声明计算曲线的 transition/trip
输入。论文关于 moment overshoot 的讨论不足以反推出这一输入。
因此冻结：

```text
experiment_trip_status = untripped
experiment_TI = unknown
paper_curve_transition_setting = unknown
source_method_ncrit = 9
```

`ncrit=9` 来自同作者来源方法，不是从实验 TI 映射得到。
禁止用 Figure 12 拟合 TI、`Ncrit`、trip location，亦禁止把
这些量带到 Fig17/18/19 做响应选择。

## 10. 对现有 claim/contract 的证据性改写建议

本报告不直接编辑 claim YAML；建议主线按下列单一证据改写：

1. `N2.6e1` 的公式闭合从 `open/missing closure` 改为
   `source-disclosed, implementation pending`。
2. `n26e1_source_response_contract_20260730.md` 第 4 节的 Ramos
   supplement 降为交叉核查，不再作为主闭合来源。
3. 撤销该 contract 第 5 节“Riziotis does not print a separation
   criterion”；替换为论文 Eq./p.7.49 的 `Cf=0`。
4. 将 Figure 12 “trip unknown” 拆成：
   `experiment untripped`、`paper-curve transition setting unknown`、
   `TI unknown`。
5. 固定 `H0=3+400/Re_theta` 勘误和
   `sqrt(C_tau,tr)=0.7 sqrt(C_tau,eq)`；二者都不得成为响应候选。
6. 保留 `remesh state transfer` 为唯一未闭合的来源身份缺口。

实现边界仍然是：

```text
source equations -> independent shadow -> source Figure 12 convergence gate
-> only then target representation -> only then frozen Fig17/18/19 comparison
```

任何先看 Fig17/18/19、再选择 `H0`、`Ncrit`、TI、`C_tau,tr`、
`Theta_n` 系数或 remap 形式的路径，都违反本次预登记。
