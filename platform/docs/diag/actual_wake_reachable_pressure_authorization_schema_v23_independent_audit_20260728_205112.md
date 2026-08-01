# transport-v2.3 authorization schema successor 第四次独立复审

**时间**：2026-07-28T20:51:12+08:00  
**review independence**：two fresh-context same-family read-only audits  
**被审 MD SHA**：`60330e24ea21dac3db69b04727c630e5cc61bc5abdca9a51a7527844f88aea9a`  
**被审 JSON SHA**：`b5fc3f9d0ff3f0ef028bc8c03829386c099874b122a2fb25f978dd9d14525f1c`

## 裁决

```text
FAIL / STRICT_PARSER_DEFINITION_NOT_YET_UNIQUE
remaining blocker groups = 2
```

未实现 parser/consumer，未生成 credential、ticket、marker、result，未运行
history。timestamped/latest MD、JSON 各自 byte-identical；JSON normative Markdown
SHA 命中且 JSON 合法。

## 已闭环

- 7-field `ImplementationIdentityBodyV1` 与 4-field anchor envelope；
- source-trace raw-SHA byte domain与 3-key source artifact map；
- exact ordered `required_checks`；
- 全部 bound-manifest、B/U/R 和 clearance binding digest inputs；
- Q→M→A→C→Z→ticket 无未来 digest 回边；
- ticket-derived 单一 private seam；
- response/clearance canonical equality和 projection 自引用消除。

## 剩余 blocker group 1：projection 与跨对象授权语义

1. `P(ticket)` 只列 output fields，未逐字段定义
   `ticket_artifact := ticket.artifact`、`ticket_version := ticket.version`、
   `authorization_plan := projection(ticket.authorization)` 等来源；
   `P-compatible reconstructed ticket` 也未定义。
2. bearer 的 64 lowercase hex、hex-decode 后 32 bytes、LF 与
   `single_use_token_sha256` 的 hash byte domain未冻结。
3. Q/P/T/C/Z 间重复语义没有 pointer equality/inequality matrix，因此以下值
   可在局部 schema 全部合规而互相矛盾：authorization/result/marker path，
   authorization id、token、decision、formal flag、execution limit、retry，
   attempt id、registry digest、implementation identity与 reviewer identity。
4. accepted/rejected grant 的镜像来源未指向 exact request path；
   clearance scope 反向写成等于未来 ticket scope，应改为绑定 Q projection，
   再由 final ticket绑定同一 projection。

## 剩余 blocker group 2：observed nested inputs

`definition_chain`、`execution_sources`、`stable_runtime_identity`、
ordered registry/case identity和 old-failure quarantine仍只被称为“完整对象”；
没有明确等于哪个 seam-observed object，也未列出 quarantine 七个键。严格
unknown/missing-field rejection因此不能机械实现。

## 最小闭合要求

- 给出 `P(ticket)` 的逐字段正向构造并删除未定义 reconstruction；
- 冻结 `T = hex-decode(stdin[0:64])`、`SHA256(T)` 及 LF exclusion；
- 给出跨 Q/M/A/C/Z/ticket 的完整 equality/inequality matrix；
- clearance scope 绑定 Q projection，不引用未来 ticket；
- 将 nested inputs exact 等同于 wrapper-definition projection、
  `source_fingerprints`、`stable_runtime_identity`、audited case registry
  computation，并枚举 quarantine keys；
- 新 timestamped successor 后再做第五次只读复审。

## 最大允许结论

identity、digest inputs和 artifact DAG 已闭合，但 strict parser 仍可能产生不同
合法结果。parser/consumer、G3/G5/G6、credential、ticket 和 history继续锁定；
科学裁决仍为 `UNKNOWN`。
