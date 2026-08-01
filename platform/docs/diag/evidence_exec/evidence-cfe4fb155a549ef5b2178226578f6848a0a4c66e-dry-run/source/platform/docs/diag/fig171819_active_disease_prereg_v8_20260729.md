# Fig17/18/19 Active Disease 研究完整性威胁模型 v8

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取 fresh
精度统计  
**用途**：终止与 Fig17/18/19 科学裁决无关的供应链安全扩张

## 0. 规范组成

active-disease 权威规范按顺序叠加 v3–v8；v7 SHA：

```text
9cf9d345f6d12bdbbb45baa0d927f085718b002f769cff6c4220cce82f1c8815
```

v3–v6 定义科学选择和归因；v7 的 raw-blob/isolated-Python 控制用于防止意外或
事后工作树漂移。本 v8 明确研究完整性威胁模型，并替换 v7 中未闭合的 receipt
“相互绑定”文字。

## 1. 明确可信计算基

本项目的 scientific-integrity threat model 信任：

- Linux kernel、当前文件系统实现与硬件；
- 当前登录用户会话中不存在恶意并发进程；
- `/usr/bin/git`、系统动态加载器和 Git object database；
- `/bin/bash` 与 Codex 受控命令执行通道；
- `/home/exuber/anaconda3/envs/fluxvortex/bin/python`、其动态库、标准库和已冻结
  dependency environment；
- 仓库父目录及其祖先未被恶意用户并发替换。

本协议要防止的是：

- fresh 完成后事后改 selector/threshold/source；
- 错读 partial/旧结果；
- 意外导入当前工作树或 user-site；
- Git replace、hook/filter、普通 checkout 和未登记源码漂移；
- 非原子输出或错误 resume。

本协议不声称抵抗已经控制 kernel、Git/Python binary、dynamic loader、当前 UID
或文件系统祖先的攻击者。`LD_PRELOAD`、恶意 Git wrapper、原生库植入和并发
symlink/rename 攻击属于 trusted-computing-base breach，不是 Fig17/18/19
scientific go/no-go 的研究变量。审计可以记录这些环境身份，但不得把本研究无限
升级为主机供应链安全证明。

## 2. 外层可信 bootstrap

外层 bootstrap 明确属于上述可信计算基，由 Codex 受控命令通道直接调用绝对路径
`/usr/bin/git` 与固定 Python 解释器；不从仓库工作树导入 wrapper。

启动时仍必须：

- 清除 `LD_PRELOAD`、`LD_LIBRARY_PATH`、`PYTHONPATH`、`PYTHONHOME`、
  `GIT_DIR`、`GIT_WORK_TREE`、`GIT_OBJECT_DIRECTORY`、
  `GIT_ALTERNATE_OBJECT_DIRECTORIES` 和 `GIT_EXEC_PATH`；
- 使用 v7 的 no-replace、no-checkout、raw-blob 和 exact path/hash closure；
- 记录 Git/Python realpath、version 和可获得的 binary SHA256；
- 验证物化根及所有祖先不是 symlink，物化目录独占创建且启动前不存在。

这些检查防止操作错误和普通环境污染；它们不改变 §1 的 TCB 边界。

## 3. 无环 receipt DAG

删除 v7“outer/inner receipts 相互绑定”的文字。固定单向 DAG：

```text
outer_preflight_receipt (immutable) -> H0
inner_launcher_receipt  (contains H0) -> H1
outer_completion_receipt(contains H0,H1,post-run closure,cleanup status) -> H2
selector/Prepare/Evaluate receipt(contains H0,H1,H2)
```

`outer_preflight_receipt` 写入后永不重写。任何 receipt 不得包含自己的 SHA 或引用
尚未生成的下游 SHA。

## 4. Python 与物化目录的实用闭合

解释器命令固定增加 `-B`：

```text
/home/exuber/anaconda3/envs/fluxvortex/bin/python -I -S -B <verified-launcher>
```

launcher 首行设置 `sys.dont_write_bytecode=True`。源 closure 完整验证后：

- 源文件和源目录改为只读；
- scientific outputs/receipts 只写入独立授权 output root；
- import 后、清理前再次枚举并 hash 源 closure；
- 任一新增 `__pycache__`、`.pyc`、路径或字节漂移为
  `INVALID_EVIDENCE`。

物化根记录 realpath/device/inode；清理只删除 outer-completion receipt 中逐项
列出的本次文件，随后删除本次空目录，禁止删除父目录。

## 5. 科学验收边界

只要：

1. v3–v6 的数学/数据协议通过零数据审计；
2. v7/v8 在上述明确 TCB 内由实现与负控测试验证；
3. evidence/attestation literal commits 在 fresh 完成前公布；

即可冻结 selector/Prepare/Evaluate。TCB 外的主机攻陷假设不得继续阻塞病灶选择、
Fig17/18/19 对照或 claim research pipeline。

任何未来安全加固只能作为非阻断 defense-in-depth，不得修改科学阈值、disease
winner、parent truth table 或候选晋升门。
