# v23 authorization schema 独立审计

**时间**：2026-07-28T20:24:40+08:00  
**review independence**：fresh-context same-family, read-only  
**被审工件**：

- MD SHA256 `228e00afed4393fa4d677fd18efabb560684e32c2844b85119146c98965fa621`
- JSON SHA256 `6ac63bd33166957a8409bd9c6e26ecc2aa960a702351454125101d6d5ab10a54`

## 裁决

```text
FAIL / AUTHORIZATION_SCHEMA_NO_GO
blockers = 7
```

不得据此实现 final parser、生成 token/ticket、进入 G5/G6 或执行 history。

## 通过项

- timestamped/latest MD 和 JSON 分别 byte-identical；
- JSON duplicate-safe、NaN-reject 只读解析通过；
- ticket self-zero canonical digest、external raw SHA、strict
  `64 lowercase hex + LF + EOF`、parser zero-delta 和 issuance lock 方向正确；
- `_VerifiedAuthorization` 现有字段都能由计划 loader inputs 填充；
- 本审计没有写文件、生成 token/ticket 或运行 history。

## Blockers

1. **semantic projection 自引用**：projection 包含
   `ticket_semantic_projection_sha256` 或其 issuance binding，形成
   `P contains H(P)`。必须用正向 allowlist，排除整个 ticket
   `issuance_request`、整个 `second_bounded_audit`、authorization self 和所有
   projection-digest 字段。
2. **MD/JSON schema 不一致**：JSON clearance grant 缺
   `authorization_id/single_use_token_sha256`，clearance scope 未定义；“至少”
   binding keys 也不能生成 exact parser。
3. **运行后 consumer 不闭合**：现有 contract verification、augmentation、
   serialized validator 和 provenance 读取旧 clearance/second-audit 形状。
   successor 必须显式授权并冻结这些 transport-only migration 及新 provenance
   schema，不能用隐式 flatten。
4. **token presentation limit 不可强制**：marker 在 auth/contract 之后，当前
   没有 durable pre-marker receipt；broker governance 不能阻止重复调用。必须
   增加预注册的 O_EXCL presentation receipt/retired commitment registry，或删除
   `token_presentation_limit` 和 pre-marker no-reuse 强 claim。
5. **trace metadata 顺序不闭合**：reviewer 无法绑定调用完成后才生成的 metadata
   hash。必须使用 review 前冻结、且不含 response hash 的 invocation metadata，
   并冻结 allowed root、same-prefix 和 cross-family comparison 规则。
6. **parser—G3 只部分破环**：须冻结同一 private loader seam、synthetic isolated
   positive fixture、loader 前后三时点 fixed point 和 formal tripwires。
7. **G6 temp hardlink 收尾缺失**：canonical link 核验后必须 unlink 自有 temp、
   再 fsync dir并验证 temp absent；失败则 namespace poisoned，绝不删 canonical。

另需明确 `audit_files` 是否覆盖所有 bound artifacts、issuance request、
request/response/pre-frozen metadata；old quarantine exact map/digest；positive
roundtrip、projection idempotence 和 projection 中不存在自身 digest 的负控。

## 三个循环裁决

```text
parser—G3—ticket: PARTIAL
token—G5: data graph feasible, execution semantics FAIL
clearance—ticket: FAIL due hard self-reference
```

## 最大允许结论

authorization 方向和 trust boundary 已形成草案，但 schema 还不可构造、不可由
现有 consumers 完整消费。reserved authorization/result/marker 仍不存在，physics
保持 `UNKNOWN`。
