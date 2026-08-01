# transport-v2.3 authorization schema successor 第三次独立复审

**时间**：2026-07-28T20:40:18+08:00  
**review independence**：fresh-context same-family, read-only  
**被审 MD SHA**：`d15db9d628362358792533a69bb0cb98665b21f77e0e2ec1187f5d23ae5bbf47`  
**被审 JSON SHA**：`fee0fe15b5c3d7c6dc42109a26b9b6d4db9ee90378d999ad96787276c50ae6b8`

## 裁决

```text
FAIL / PARSER_AND_TRANSPORT_CONSUMER_IMPLEMENTATION_STILL_LOCKED
remaining blocker groups = 1
```

未实现 parser/consumer，未生成 credential、ticket、marker、result，未运行
history。timestamped/latest MD、JSON 各自 byte-identical；JSON normative Markdown
SHA 命中且无 duplicate key；reserved ticket/result/marker 均不存在。

## 已闭环

- Q→M→A→C→Z→ticket 已形成无未来 digest 回边的单向链；
- `response_canonical_sha256 == clearance_canonical_sha256` 已硬等同；
- 唯一 private seam 只信 ticket path，并逐层派生 Q/M/A/C/Z child paths；
- projection 正向 allowlist 不再包含自身或未来 audit digest；
- nested transport consumer/provenance 迁移的职责边界可落地。

## 唯一剩余 blocker group

**exact implementation/request metadata contract 仍不可执行：**

1. §2.3 lines 220–230 将 ticket/projection/Q/M/A 的
   `implementation_identity` 冻结为 7 字段；§3.1 lines 426–451 又把 anchor
   file 冻结为 10 字段，并要求二者 canonical object exact equal。集合精确差
   `{artifact,schema_version,version}`，因此不存在满足契约的对象。
2. `source_trace_metadata_sha256` 未声明绑定 raw 还是 canonical bytes。
3. `source_artifact_map_sha256` 所称 final wrapper/G2/G4 map 未冻结 exact keys、
   value shape 和 canonical 编码，仓库中无另一权威定义。
4. `required_checks` 只列语义类别，未冻结可机器校验的 ordered literal array。

非独立硬 blocker但应同步修正：summary-only JSON 把 MD 中实际
`publication_pre_link_gate` 写成 `_pre_link_gate`，容易制造审计歧义。

## 最小闭合要求

- 定义唯一 `ImplementationIdentityBodyV1`；
- anchor 使用 envelope，并规定 ticket/projection/Q/M/A exact 等于
  `anchor.identity`；
- 明确 source trace digest 的 raw/canonical 语义；
- 冻结 source artifact map exact object 与 canonical SHA 算法；
- 枚举 `required_checks` 的 ordered literal IDs；
- 同步 summary-only JSON 的 consumer 名称后，再做第四次只读复审。

## 最大允许结论

本 successor 已关闭循环依赖，但仍不唯一可执行。parser、consumer、G3、G5、
G6、credential、ticket 和 history 继续锁定；科学裁决仍为 `UNKNOWN`。
