# fresh151 外部进程中断与 resume 记录

**campaign**：`20260729_135128`  
**最后原子 checkpoint**：2026-07-29 14:28:02 +08:00  
**核验时间**：2026-07-29 14:38:52 +08:00  
**分类**：EXTERNAL PROCESS/PTY INTERRUPTION；非 solver/guard failure

## 事实

- PTY 监控句柄失效后，原 Python PID 已不存在，GPU 已空闲；
- manifest 状态仍为 `running`，`failures={}`；
- 正式完成 `34/151`；
- result、contributions、case_guards 均为 34 个 key，三者集合完全相同；
- 无 `.partial` 文件；
- 对 34 个 saved case 重新执行 condition、resolved call、runtime、
  graph identity、guards、raw contributions、ledger 和 source closure 校验：
  `discarded={}`；
- solver/control source 数为 `140/11`，五个授权对象 SHA 均未变化。

## 裁决

旧 checkpoint 可按原预登记显式 resume。resume 必须：

1. 使用同一 timestamp `20260729_135128`；
2. 不复用任何未在这 34 个有效 checkpoint 中的结果；
3. 重新执行 discarded cold + formal warm anchor；
4. warm anchor 继续满足 0.15 N 身份门；
5. 从第 35 个正式工况继续。

为避免会话句柄再次终止长任务，本次 resume 使用独立后台进程和版本化日志；
数值 runner、运行合同及源代码保持不变。
