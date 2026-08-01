# S3ai-v2.2：one-shot 执行层预登记

时间：2026-07-28 14:30:34 +08:00  
节点：`N3.1j3b6d18c2b3b3b2c2b2b3e3b`  
裁决：**定义层通过；冻结 runner 永久不可执行；允许实现外部 one-shot
wrapper，但第二次有界审计前仍禁止正式 history**

## ① 数据指纹、树节点与可动空间

首次外部语义审计尝试的实际后端为 GLM-5.2（Claude Code CLI stdin 路由），thread
`38480c70-a3dc-4b24-a6aa-601baca48f18`。首次审计确认 v2.2 的
continuum-zero、typed parity、单次 round、逐 stage 非抵消、负对照和
fail-closed 语义与代码一致；但追审自报 `single-family, single-run`，
因此只授予 wrapper 实现许可，同时撤回“当前快照可执行”的过强结论。

原因是同一执行链存在联合不可满足条件：

```text
_load_frozen_contract: formal_execution_allowed must be False
run_preregistered_observation: formal_execution_allowed must be True
```

所以当前 runner 不是“少一枚开关”，而是作为冻结定义锚而永久不可执行。
直接修改 runner 或 v2.2 YAML 会破坏此前证据身份。

此外，冻结 runner 自报 15 个源码指纹。干净的 `python -B` 导入实测表明，
`claim_runtime/__init__.py` 的 eager imports 会使 runner 在尚未调用 loader、
mesh 或 history 前加载 62 个本地 Python 源码；包含 package initializer 的
递归 AST 闭包独立得到同一集合，双向差集为空。先前登记的“21 个”只追踪了
显式递归边，漏掉了 package import 语义，因此已被证伪并在冻结前撤回。

因此可动空间只允许新增执行运输层：调用冻结 loader/collector/aggregator，
不改公式，不重写冻结资产，并把完整 62 文件闭包加 wrapper 本身共 63 个
源码纳入指纹。精确有序清单及其换行终止清单 SHA256 已写入 YAML。

## ② 学科/数值机理

这一步没有新增气动机理。它解决的是可复现实验的 provenance 和 one-shot
语义：

- 定义资产回答“测什么、怎样判”；
- 授权票据回答“哪一个已审源码快照可以运行一次”；
- 永久 `O_EXCL` marker 回答“这次授权是否已经消耗”；
- 原子 no-replace 发布回答“并发或外部文件是否会被覆盖”。

审稿建议中的 `tmp + os.replace` 不满足 one-shot：POSIX replace 会覆盖在
检查后并发创建的 canonical result。故最终发布采用同目录临时 inode、
`fsync`、`os.link(tmp, final)`；目标已存在时 link 原子失败，不会覆盖。

授权 YAML 的原始 SHA 无法无矛盾地写进自身。方案使用 canonical digest：
解析 YAML，将 `authorization.canonical_sha256` 归零，再以 sorted compact
JSON 序列化并计算 SHA；原始文件 SHA 只在结果中观察和记录。

## ③ 缺组成部分还是组成部分错

本轮判定是两个执行治理组件缺失：

1. 冻结定义和一次正式运行之间缺独立 one-shot wrapper；
2. 15 文件指纹清单缺完整本地执行闭包。

这不是 pressure law、forming geometry 或 VES 组件错误。尤其不能把执行
授权缺口解释成需要新增涡状态。

审稿意见要求“预先绑定每条 history 的数组 value hash”不可直接采用：
这些数值只有正式 observation 后才存在，预绑定等于预造结果。正确做法是
在授权票据中冻结**必须出现的五类 value-hash schema**，运行后由冻结
aggregator 产生实际值并写入唯一结果。

## ④ 有证据方案与 go/no-go

时间戳预登记为
`actual_wake_reachable_pressure_execution_wrapper_cases_20260728_143034.yaml`。
阶段顺序固定：

1. 实现 wrapper 和纯定义测试，所有测试不得调用正式 collector；
2. 以分片、不过上下文上限的跨族审计重新核对 wrapper 与八组 v2.2 语义；
3. 只有第二审计 accepted 后才可生成预留 authorization ticket；
4. ticket 绑定冻结四资产、完整 63 文件源码闭包、31 case identity、审计
   trace、唯一 result/marker 和 one-shot token；正式调用者还必须从票据
   外提供 ticket 原始 SHA256 和第二审计 token 的 preimage；
5. 执行前以 `O_EXCL` 创建永久 marker；崩溃也消耗授权，禁止静默重跑；
6. 将 frozen aggregator 的任何结果（包括 `PROTOCOL-NO-GO`）以
   no-replace 原子方式持久化。

当前裁决仍是：

```text
wrapper_implementation_allowed = true
authorization_ticket_creation_allowed_before_second_bounded_audit = false
authorization_ticket_creation_requires_accepted_second_bounded_audit = true
formal_execution_allowed = false
production_activation_allowed = false
```

第二审计不能只留下自由文本结论。它必须返回无 Markdown 包裹的结构化 JSON
clearance，并同时绑定 wrapper、预登记、63-source map、runtime、31-case
registry、冻结 definition chain 和外部 bearer-token commitment。wrapper
直接解析该 response；ticket 自报的 `accepted/cross-family` 字符串不构成许可。

正式进程还须以 `-B -X pycache_prefix=<repo>/.one_shot_no_bytecode_cache`
启动，且该 cache 路径必须不存在。wrapper 在导入冻结 runner 前记录这一
条件，以排除“磁盘哈希是新源码、实际执行却来自旧 `.pyc`”的部署竞态。
runtime 身份同时绑定工作树绝对路径与 device/inode，以及已加载 native/BLAS
二进制；因此一次授权只适用于这一工作树，不可复制到另一 clone 重放。
正式调用由固定的 `python -c` bootstrap 以 `O_NOFOLLOW` 打开 wrapper，
同一份 bytes 同时用于 SHA256 与 `compile/exec`，并把观察到的 SHA 注入
wrapper；wrapper 再以 import 前后两次 62-source snapshot 排除正常原子部署
造成的“内存旧代码/磁盘新哈希”。argv 只携带外部 ticket SHA，32-byte token
preimage 以 64 位十六进制从 stdin 输入，不新增 launcher 文件。

marker 与 result 从 claim 到 publish 共用一个持续打开的 output-directory
descriptor；每次关键操作都把 canonical 路径重新解析到该 descriptor 的
device/inode。因而正常目录 rename/swap 只能令本次已消耗尝试 fail closed，
不能把 marker 留在旧目录却把 result 发布到新目录。runtime 还绑定 BLAS/OMP
选择变量、CPU affinity 与浮点舍入模式，授权不跨数值执行路径漂移。
marker 的内容摘要和创建时 inode/size 会在发布前复核；result 临时 inode
保持打开直到 no-replace hard link 完成，并核对 final 与 temp 的 inode 和
实际字节，避免路径名检查和实际发布对象分离。

## 边界

- 未修改冻结 v2/v2.1/v2.2 或原 runner。
- 未运行 31-history，未生成正式结果。
- 未进行空间 h/p、状态身份、forming geometry/VES 裁决。
- 未计算力、生产 118 工况或 Fig17/18/19。
- one-shot 只承诺当前未快照、未回滚的文件系统实例；跨 VM/文件系统快照的
  全局一次性需要外部 append-only/CAS 消费账本，不由本地 wrapper 冒充实现。
- wrapper 门控的是“可接受的 canonical 正式证据发布”，不是 Python 私有函数
  的能力安全边界；绕过 wrapper 直接调用 private collector 不带授权回执，
  不能晋升为本战役的正式证据。
- trust boundary 假设源码和包环境在进程 bootstrap 期间不被恶意双重切换；
  对能同时改写代码、票据、trace 和目录的本地攻击者不作密码学安全声明。
