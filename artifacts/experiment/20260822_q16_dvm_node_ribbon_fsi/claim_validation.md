# 声明校验

| 声明 | 结论 | 证据与边界 |
|---|---|---|
| LEV 始终集成但释放有条件 | 支持 | 5° 原阈值零释放仍完成 FSI；20° 原阈值自然释放 |
| DVM node-ribbon 已进入 Q16 强耦合事务 | 支持 | 两个连续接受步推进同一 owner lineage，source/solver step 同为 3 |
| predictor 推进真实 LEV 和自由尾迹 | 支持（两步） | 粒子 `206 -> 413 -> 571`，wake commit `0 -> 1 -> 2` |
| 失败 trial 不污染 parent | 支持（单个强制失败门） | 非收敛测试比较结构哈希、solver 哈希、对象身份、粒子和 source step |
| 气动—结构载荷不重复计冲量 | 支持（该夹具） | owner 全为 `ptera_kj_plus_dgamma`；生产 impulse=0，诊断 impulse 非零 |
| node/cell connected-ribbon 拓扑一致 | 支持（当前门） | cell 拥有事件，node 继承邻接并集；投影后 mismatch 始终为 0 |
| DVM Q16 FSI 已达到长时/多周期稳定 | 不支持 | 当前只运行两个接受步 |
| 已完成论文 CASE 精度复现 | 不支持 | 本轮是开发夹具，未访问论文 GT 或评分器 |
| 已证明所有可能 LESP 切换边界 | 不支持 | 仅覆盖一个亚临界和一个持续 active 工况；自然开/关切换待 4--8 步门 |
