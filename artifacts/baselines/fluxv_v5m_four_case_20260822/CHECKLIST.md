# FLUX-V5M 四 CASE 复现检查表

- [x] 明确核心目标为 CASE 复现，而非继续学术探索
- [x] 锁定四 CASE 身份和冻结指标文件
- [x] 确认无遗留周期 FSI GPU 进程
- [x] 隔离会改变默认 CASE 数值的周期探索改动
- [x] 将三周期长测改为显式 opt-in，不进入默认回归
- [x] 运行 mandatory separated-LEV/joint-TEV/free-wake GPU 冒烟门
- [x] 建立 Mancini 2017 mandatory-mode 生产入口与命令
- [x] Mancini fast smoke 完成（RMSE `6.18865`，FAIL）
- [x] Mancini fast full 完成（RMSE `316.16453`，FAIL）
- [x] Mancini slow smoke 完成（RMSE `4.25877`，FAIL）
- [x] 已按停止规则取消 slow full，避免在已知 NO-GO 表示上浪费计算
- [x] 修复长时 GPU 尾迹归约 OOM：粒子/尾迹查询统一 CUDA 分块
- [x] runner 增加冻结 reference 误差门，FAIL 改为非零退出
- [x] 建立 Baik W1--W4 mandatory CUDA 生产入口
- [x] fresh 复算 Baik W1--W4；四个物理门和宏 RMSE 非退化门全部 PASS
- [x] CUDA LDVM 导出 source-only LEV/TEV 出生包（含作者首 TEV 持久化语义）
- [x] CUDA node-owned 连通涡带沉积原语通过共享边/闭合/固定核门
- [x] 将 CUDA source-only 包接入节点放置、predictor 运输和 Ptera RHS/load
- [x] 修复 DVM predictor 序列化与展向批处理 source bank
- [x] 修复 DVM 实体粒子使用最终物理束缚环量推进
- [x] DVM 生产载荷改为唯一 Ptera `KJ+dGamma` owner，禁止重复冲量力
- [x] 将“新生 LEV 事件”和“Ptera 分离边界状态”拆开，消除开关假峰
- [x] Mancini fast smoke 通过非退化门（RMSE `0.84303`）
- [x] Mancini fast full 通过（RMSE `1.04886`）
- [x] Mancini slow smoke/full 通过（RMSE `0.21476/0.22527`）
- [x] 受影响 DVM/Hirato/Q16/GPU-only 回归 `41 passed`
- [ ] 建立 Yang、Izraelevitz 生产入口与命令
- [ ] 按冻结合同执行四 CASE fresh GPU 复验
- [ ] 生成 expected/observed/差值与 PASS/FAIL 报告

当前判定：`baik_and_mancini_mandatory_pass_yang_izraelevitz_pending`。Baik 与
Mancini 可作为本次 fresh 结果；常强度环路径仍维持 NO-GO，Mancini 的通过结果
只属于 DVM source + node-owned ribbon + Ptera LESP/载荷路径。Yang、Izraelevitz
尚未完成 mandatory 迁移，不得写成四篇全部通过。
