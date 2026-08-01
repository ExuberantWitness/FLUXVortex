# N2.6f1 S0 Basilisk 来源冒烟结果

日期：2026-07-30  
节点：`N2.6f1`  
裁决：`S0 PASS / MOVING-PHYSICS NOT YET TESTED`

## 1. 本门回答的问题

本门只回答固定 Basilisk 快照是否能在本机完成
`qcc -> embedded geometry -> AMR -> pressure/viscous force -> vorticity`
这一条官方执行链。它不验证移动边界，不验证 Schneiders 响应，也不授权
RoboEagle/Fig17--19 目标运行。

## 2. 冻结资产

- tarball：
  `platform/external/basilisk-fe6b4b58/basilisk.tar.gz`
- tarball size：`9,610,797 bytes`
- tarball SHA-256：
  `fe6b4b5821517d792c58f0413ea6de4b5bd6b1d337578bb5ceb2fa6f07f8f193`
- official source：
  `platform/data_external/n26f_runs/source_s0_naca2414/naca2414-starting.c`
- source SHA-256：
  `5566e55f643ca0f0a0f4e26a79beb8bd17afc9bbb84faf0a5dd9ea4df6a2fe32`
- solver build：
  `platform/external/basilisk-fe6b4b58/basilisk/src/qcc`

固定快照首先执行了上游构建。`make -j8` 暴露上游 Makefile 的
`ast/libast.a` 并行依赖竞争；随后不改变源码执行串行 `make -k && make`
完成构建。该问题属于构建协议，不是物理解失败。

工作命令从算例目录使用相对源码路径，并显式设置 `BASILISK`：

```bash
export BASILISK="$PWD/platform/external/basilisk-fe6b4b58/basilisk/src"
cd platform/data_external/n26f_runs/source_s0_naca2414
"$BASILISK/qcc" -O2 -Wall -autolink naca2414-starting.c \
  -o naca2414-starting -lfb_tiny -lm
./naca2414-starting >stdout.log 2>force.log
```

首次尝试使用绝对输入路径时，`qcc` 没有在预期位置找到生成的预处理文件；
首次链接又因未显式加入 `-lfb_tiny` 而失败。两项都在正式运行前修正，
没有产生或筛选物理结果。

## 3. 运行事实

官方 NACA2414 starting-vortex 算例完整运行至无量纲时间 `t=1`：

- 计算：`462 steps`，`25.2067 CPU s`，`25.25 real s`；
- numeric force rows：`463`；
- 所有记录有限，无 NaN/Inf；
- `dt` range：`0.000284091--0.00226933`；
- `CD` range：`0.0689175--0.221965`；
- `CL` range：`0.0145804--0.535631`；
- final：`CD=0.0689388`，`CL=0.532308`。

证据文件：

| file | SHA-256 |
|---|---|
| `force.log` | `2fda77072fbf4aa36f790c346b094d3e4edf64dbdee278abc091caffd818b2ff` |
| `stdout.log` | `028e67fab571794b0ed24a4b58db49029b03873282900faea60d68ea94afd235` |
| `vorticity.png` | `3a5cfbc7a5f5782709cd9ae17075f3173c9eeb652f9e8d79d8e5d1d9a81777e4` |

主线程视觉核对 `vorticity.png`：翼型表面上下两侧剪切层符号相反，尾缘后方
出现单个卷起的起动涡，且没有明显几何穿透、反向尾迹或图像截断。该观察只
支持 S0 的执行链完整性。

系统没有 `ppm2mp4`/`ffmpeg`，Basilisk 因而回退生成
`vorticity.mp4.ppm` 原始帧流（`105,600,704 bytes`，SHA-256
`70ebeb46c8a48e2c58d5854760593ac19c53fb5bf31ffa0007da4556b5329cda`）。
它是已由最终 PNG 覆盖的意外中间缓存，不作为科学证据保留。

## 4. 裁决

S0 预登记条件全部满足，因此：

> 固定 Basilisk 快照的编译、AMR、静止 embedded-boundary、压力/黏性力和
> 涡量输出链 `PASS`。

仍未回答、不可外推的命题：

1. moving embedded boundary 是否正确；
2. CL/CD 是否复现 Schneiders 一手参考；
3. 空间、时间和域尺度是否独立；
4. N2.6 的缺失分离压力病因是否成立；
5. Fig17/18/19 是否改善。

下一门仍严格是冻结的 S1 moving NACA0015 source run；不得越级运行目标点。
