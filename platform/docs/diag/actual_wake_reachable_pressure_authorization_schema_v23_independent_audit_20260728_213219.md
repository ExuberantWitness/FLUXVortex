# transport-v2.3 authorization schema successor 第七次独立复审

**时间**：2026-07-28T21:32:19+08:00  
**review independence**：two fresh-context same-family read-only audits plus sidecar cross-check  
**被审 MD SHA**：`5c0aad0637aeda05db0884ffaf73154e4b3d5c7c54415aed801a1b59f5d10e2c`  
**被审 JSON SHA**：`bd743d6e251cb76aaf6b914aff9d2c7c66936a0cbbe8f624c6e7c24f3fab45f6`

## 裁决

```text
FAIL / STRICT_PARSER_DEFINITION_LACKS_OWNED_FIXTURE_AND_RACE_SAFE_G6
remaining blocker groups = 3
```

MD/JSON raw SHA 命中，JSON 严格解析通过。未实现 parser/consumer/G6，未生成
production Q/M/A/C/Z、credential、ticket、marker、result，未运行 history。

## 已闭环

- Q observed path exact 绑定 namespace profile；
- fixture/production 双向 component-prefix不可比与 trace/file隔离；
- root dirfd pin、fresh reopen与 parent re-walk方向；
- pre-Q SourceMap 与 final-ticket SourceMap分支，不再引用 future TKT；
- nominal G6 `nlink=1→2→1`、S5 后 token release；
- Build/P/raw(TKT)/guard/S snapshots均未回归。

## Blocker 1：fixture owned-root capability 缺失

profile fields 不含 `fixture_id`，也不携带 exclusive mkdir 后的 base/root dirfd、
`(st_dev,st_ino)` 或等价 capability。seam 无法证明当前同名 root仍是本次 fixture
owner创建的 inode，亦无法把 child walk与 cleanup ledger持续绑定到该 inode。
fixed fixture base当前也不存在，provisioning主体和步骤未冻结。

最小闭合要求：唯一 factory 以 pinned existing base做 exclusive mkdir，立即
no-follow open root并返回 typed `_FixtureNamespaceLeaseV1`；profile显式携带该
lease，seam入口、child walk、S_2、return和cleanup均验证同一 inode。fixture id
必须是 exact field或从 isolation root相对 fixed base唯一派生。

## Blocker 2：SourceMap pre-plan freeze 边界循环

`plan.bound_artifacts` 被要求在 plan构造前已经通过 `SourceMap_B`，而
`SourceMap_B` 又只从已构造 plan 的 `bound_artifacts` 投影。虽已消除 future
ticket，仍缺独立 pre-plan producer。

最小闭合顺序：

```text
candidate_bound_artifacts := strict observed pre-Q map
SourceMap_seed := positive projection(candidate_bound_artifacts)
validate source trace and anchor against SourceMap_seed
plan.bound_artifacts := frozen deep copy(candidate_bound_artifacts)
SourceMap_B := positive projection(plan.bound_artifacts)
SourceMap_B = SourceMap_seed
```

accepted 再验证 `SourceMap_T=SourceMap_B`；rejected不引用 TKT。

## Blocker 3：G6 early failure、pathname race与 sidecar冲突

- `owned_identity` 只在 write+fsync后定义；create成功、write前失败无 cleanup
  comparison identity；
- `stat(name)==owned → unlinkat(name)` 不是条件 unlink；active name-swap可使
  publisher删除第三方 pathname。若保留“绝不删除第三方”强 claim，当前协议不能
  支撑；若 assurance profile不覆盖 malicious writer，则必须显式撤销该强 claim
  和相应 race test；
- JSON 声称所有 post-link failure均保留 temp link，但 MD S4可在 temp已 unlink
  后的 fsync/re-walk失败，二者直接冲突。

最小闭合要求：create后、write前记录 `created_identity=(dev,ino)`，complete
write后再绑定 expected size；明确 active malicious name-swap不在 assurance
claim，或改为不按可交换 pathname删除的 publication primitive；sidecar改为
canonical总保留、temp保持 failure发生时的 observed state且不再 cleanup。

## 最大允许结论

当前只允许建立新 timestamped successor并再次只读审计。parser/consumer、
production G3/G5/G6、credential/ticket/history继续锁定；科学裁决仍为
`UNKNOWN`，三点和118均未启动。
