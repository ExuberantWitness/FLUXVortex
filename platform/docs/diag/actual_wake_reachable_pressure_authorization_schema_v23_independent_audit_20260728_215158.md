# transport-v2.3 authorization schema successor 第九次独立复审

**时间**：2026-07-28T21:51:58+08:00  
**review independence**：two fresh-context same-family read-only audits plus sidecar cross-check  
**被审 MD SHA**：`ee60675e9cae3a1c83889d6fd16ff82c151047c0808f8641fe5827ae321ac4bf`  
**被审 JSON SHA**：`3cdb31e17e755646f9c481ebdd9d69a6c165ba2653d0a235c952c3a89de6d204`

## 裁决

```text
FAIL / TWO_REMAINING_BINDING_AND_HANDOFF_BOUNDARY_BLOCKERS
remaining blocker groups = 2
```

MD/JSON raw SHA与JSON parse通过。未实现 parser/G6，未创建 production
credential/ticket/result，未运行 history。

## 已闭环

- IdentityCore内容→source trace→anchor五字段内容链；
- typed DirectoryLease、fixture ownership与双向 namespace；
- SourceMap seed→candidate→plan与 accepted/rejected branches；
- G6 S0、O_TMPFILE、proc-fd linkat、nonzero canonical
  absent/same/foreign、close/re-walk/observation和post-link poison；
- Q/Build/P、raw ticket、guard、S snapshots未回归。

## Blocker 1：IdentityCore pre-candidate binding seed缺失

core内容已有 pre-anchor producer，但其唯一 path/raw authority仍被写成未来
`candidate_bound_artifacts.implementation_identity_core`。step 0 无法先定位和
raw-bind core；source trace因此仍隐含使用未来 candidate map。

最小闭合：

```text
IdentityCoreBindingSeedV1 :=
  {
    path: exact profile-owned/fixed orchestrator output path,
    raw_sha256: externally supplied SHA256(exact core raw bytes)
  }
strict no-follow read/parse core from seed
source trace core binding = seed
candidate_bound_artifacts.implementation_identity_core = seed
```

该 external digest不得从 candidate或core自身推断。最终步骤必须写成完成
`0–7`。

## Blocker 2：G6 handoff后 launch failure边界过宽

S6 custodian成功接受 T 后 flag=true，broker finally不再销毁 custody；failure C
却仍把 post-link launch failure纳入“broker销毁”承诺。launch在 accept/flag前后
未定义，sidecar的 any-failure摘要因而过强。

最小闭合：G6只负责 ticket commit与 custody handoff。`custodian_accept(T)` 返回
成功 receipt后，flag=true并立即返回该预构造 receipt；flag后无 launch、I/O、
hash或其他协议步骤。formal wrapper launch是 S6之后的外部阶段，不属于 G6
failure table；pre-accept任一 failure仍必须先销毁 broker custody且不能到 S6。

## 最大允许结论

仅允许建立新 successor并再次只读审计。parser/G6/production
Q/M/A/C/Z/token/ticket/marker/result/history继续锁定，科学裁决仍为
`UNKNOWN`。
