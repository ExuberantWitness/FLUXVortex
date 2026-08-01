# transport-v2.3 authorization schema successor 第六次独立复审

**时间**：2026-07-28T21:18:29+08:00  
**review independence**：two fresh-context same-family read-only audits  
**被审 MD SHA**：`1e0c8e577a0ad02eded625443b5ba69cc579b4f4cea73a96afa58d600fdd49f6`  
**被审 JSON SHA**：`82af94de8898c543cdaefdaa7ade44f085322b55c57b57bfecb97d83c7cf39c4`

## 裁决

```text
FAIL / STRICT_PARSER_DEFINITION_NOT_YET_NAMESPACE_CAUSAL_OR_LINK_STATE_COMPLETE
remaining blocker groups = 3
```

被审 MD/JSON 的 raw SHA 命中，JSON 可解析且其 normative Markdown binding
命中。未实现 parser/consumer，未生成 production Q/M/A/C/Z、credential、
ticket、marker、result，未运行 history。

## 已闭环

- pre-Q `BuildProjectionV1` 与 final `P(TKT)` 是两条独立输入路径；
- rejection branch 只使用 Build/Q，不引用不存在的 P/TKT；
- external expected SHA hard-equal `raw(TKT)`；
- 已加载 guard module/path/source hash与 registry pure computation；
- `_loaded_dependency_state()` 的 `S_pre/S_1/S_2` constructor、时点和 full-object
  equality；
- Q/M/A/C/Z/TKT 主等式、bearer raw-byte domain、run-directory和 temp-name
  grammar。

## Blocker 1：fixture namespace 未双向封闭

`NamespaceProfileV1` 没有 `issuance_request_path` 或统一动态制品根，因此
`path(Q)` 没有 profile ownership。fixture 与 production 的 overlap 禁令也未
覆盖 `isolation_root` 自身及双向 component-prefix；`allowed_trace_root` 与
ticket/result/marker 之间没有前缀隔离。可构造 Q 落在 production trace tree、
其余对象落在 fixture root 的反例，并通过当前矩阵。

最小闭合要求：

- profile 增加机械派生的 exact Q path；
- fixture isolation root、Q/ticket/result/marker/trace root与全部 production
  dynamic namespace做双向 component-prefix不可比校验；
- trace directory与 sibling dynamic files做 type-aware prefix隔离；
- 固定 repo-root dirfd 的 open flags、`fstat` identity和全流程 re-walk。

## Blocker 2：source trace 仍引用未来 ticket

`ImplementationSourceArtifactMapV1` 被写成从 final ticket
`bound_artifacts` 投影，但 anchor/Build/Q 在 Q 前就使用它；rejection branch
根本没有 final ticket。

最小闭合要求：pre-Q 从 frozen `plan.bound_artifacts` 重算并验证 source map；
accepted branch 在 final ticket 后从 `TKT.bound_artifacts` 重复同一投影并要求
等值，rejection branch只依赖 plan/Build/Q。

## Blocker 3：G6 inode/link 状态机不唯一

`verify inode/nlink` 未冻结比较对象和精确状态，也未定义 owned temp 创建后的
pre-link failure cleanup。

最小闭合要求：

```text
create: owned temp is regular, nlink=1
link:   temp fd/temp name/canonical name share (st_dev,st_ino), nlink=2
unlink: canonical keeps owned inode, nlink=1; temp is ENOENT
```

所有 pre-link failure 只清理由本次 `O_EXCL` 创建且 name仍指向 owned inode 的
temp，随后 fsync；cleanup失败即 poison。post-link failure保留 canonical并
poison。从固定 repo-root重新遍历 lexical ticket parent，必须仍指向 pinned
directory inode，才允许 release token。恶意 writer hardening仍保持 nonclaim。

## 最大允许结论

本轮证明核心因果投影和 standalone snapshot 来源已闭环，但 filesystem ownership、
pre-Q source-trace producer与 hard-link state machine仍不唯一。当前只允许修订
新 timestamped successor并再次只读审计；parser/consumer/G3/G5/G6、production
credential/ticket/history继续锁定，科学裁决仍为 `UNKNOWN`。
