# 声明校验

| 声明 | 证据 | 判定 |
|---|---|---|
| 原 `Lcrit=0.11` 的 DVM-Q16 FSI 可连续接受 8 步 | 4+4 prefix/resume，8/8 接受 | 支持 |
| predictor/corrector 推进同一 DVM source、LEV 和 free wake lineage | source/solver/wake=`9/9/8`，owner/aero generation=`8/8` | 支持 |
| LEV 仍是条件释放而非强制释放 | cell release 由 `LESPmax>0.11` 触发；未改阈值 | 支持（本工况持续 active） |
| node 不独立制造三维物质释放 | 一个坐标 raw/topology=3/4，cell=3；有效 node 仅继承相邻 cell | 支持 |
| 气动—结构载荷没有重复 impulse | 全时层 owner=`ptera_kj_plus_dgamma`，生产 impulse=0，诊断 impulse 非零 | 支持 |
| 失败坐标不污染已提交 parent | 第 9 坐标失败，结构/气动 owner、对象和哈希不变 | 支持 |
| 已验证同一轨迹 release off/on/restart | 全部 cell 在 9 个坐标均 active | 不支持 |
| 已验证多周期稳定或论文精度 | 静态 inflow、8 步 dev fixture、无论文 GT | 不支持 |
| 两步与八步载荷/位移可直接比较 | 八步使用声明的 `E=1e9`、阻尼 `20 s^-1` 长时结构夹具 | 不支持 |
