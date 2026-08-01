# V4.1 fresh confirmed151 启动授权凭据

**日期**：2026-07-29  
**角色**：POST-IMPLEMENTATION LAUNCH RECEIPT  
**状态**：FINAL CANDIDATE FROZEN；等待外部双重复审签发 GO

## 1. 历史身份

`v41_confirmed151_resume_integrity_addendum_20260729.md` 中的
`d5bd099c...` 明确表示触发修复的旧 runner，而不是修复后的授权版本。该历史
SHA 保持不变，避免把事后实现冒充预登记对象。

## 2. 最终授权对象

runner 已将本凭据纳入 control-source closure。最终候选冻结为：

- runner：`78b80cf9a04caf4bfe9040c893966538efac85dbc88d6ae625e75f769183081a`
- witness：`601128ed48ae3f8bec364c812cc71db02acf207df314c3e2ddf38eedd45b6925`
- tests：`1eea1e1286869a3095b831fe0c9b93229ecb477e52649ec51c1197186eef24e2`
- prereg addendum：
  `0c41ba621bcded7d6671cd94e45a63cbf316787c7ac3b0dabe9432838b493e2c`

## 3. 授权条件

- 20/20 无 GPU 测试必须在最终哈希上通过；
- 必须同时获得独立 schema/integrity 审计与 launch code review 的 GO；
- runner、witness、tests、prereg addendum 或本凭据任一内容变化，授权自动失效；
- 本凭据只授权 fresh V4.1 confirmed151 基线，不授权任何气动公式修改或
  V4.2 candidate。

独立审阅者以本文件 SHA 和上述对象 SHA 在外部审计报告中签发 GO；为避免自指
哈希，本文件不写入自身 SHA，也不在审阅后回写结论。
