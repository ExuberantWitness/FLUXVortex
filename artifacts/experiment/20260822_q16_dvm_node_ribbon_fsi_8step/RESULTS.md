# 结果

原 `Lcrit=0.11` 的 DVM-Q16 FSI 已连续通过 8 个接受步。4 步前缀被原样
保留并恢复到 8 步；第 9 个不存在的气动坐标被事务性拒绝，已提交 parent
不变。

| 验收项 | 结果 |
|---|---:|
| 接受步 | 8/8 PASS |
| 初始/最终 LEV 粒子 | 206 / 1917 |
| DVM source / solver step | 9 / 9 |
| free-wake / frontier advance | 8 / 8 |
| 最大耦合残差 | 4.0602e-10 |
| 最大结构残差 | 2.1119e-8 |
| 最大虚功相对残差 | 3.3218e-13 |
| 生产 impulse 计数/范数 | 0 / 0 |
| GPU 峰值 allocated | 277705728 bytes |
| 受影响回归 | 74/74 PASS |

本工况中所有三个展向 cell 在全部气动坐标都自然越过阈值，所以它验证的
是持续 active 条件释放，而不是释放开关循环。node 的 raw 阈值在一个
坐标上为 3/4 active，但有效拓扑为 4/4；这是共享端点继承相邻 active
cell 的正确行为，不是额外释放。

结果哈希：

- result：`0fbdfca788708a10dcdc9cd6ea03a3604c3354dbcf6170f12e2f50d918f1d388`
- trajectory chain：`c5f3383b979af8f621b0d6e32729dcc873af18ded6827beefb48269b023f0f36`
