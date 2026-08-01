# v23 authorization schema successor 独立复审

**时间**：2026-07-28T20:31:07+08:00  
**review independence**：fresh-context same-family, read-only  
**被审 MD SHA**：`768532b369fcbc9374d0e77e247b5b3191573e0ecffc6b2590dcbb02d47217a2`  
**被审 JSON SHA**：`1e49f0b7bf0a91b9148d4470be0b99c6718364e9f4f122d559d6210989a895c9`

## 裁决

```text
FAIL / PARSER_IMPLEMENTATION_STILL_LOCKED
remaining blocker groups = 3
```

未写文件、未生成 credential、未运行 history。

## 已通过

- normative Markdown / JSON index binding；
- positive-allowlist projection 已消除 self digest；
- nested consumers/provenance 迁移定义；
- pre-marker presentation 强 claim 已降级；
- G6 canonical hardlink/temp cleanup；
- reserved issuance、ticket、result、marker 均不存在。

## Blockers

1. **exact child schemas 缺失**：
   - review request 没有 exact artifact/version/top/nested schema；
   - invocation/post-review/implementation metadata 没冻结固定字面值和全部类型；
   - 同一 clearance response 的 `response_canonical_sha256` 与
     `clearance_canonical_sha256` 未硬等同。
2. **Q↔M 时序互引用**：issuance request `Q` 绑定 invocation metadata `M`
   raw/canonical，M 又绑定 Q raw；M 还预先绑定未来 review request。必须冻结单向：

   ```text
   Q → M → A(review request) → C(clearance) → post-review trace → ticket
   ```

   Q 只含 planned M path/identity，不含 M digest；M 可绑定 Q，不含未来 A digest；
   A 绑定 Q+M；ticket 事后绑定全部。
3. **private seam 参数不唯一**：public wrapper 在 G3 freeze 时不知道 G5 才生成的
   child paths。唯一 seam 必须只接 ticket path（外加 expected SHA/token/observed
   inputs），内部 strict parse ticket，再逐层 derive/parse issuance、invocation、
   request、response、trace path；synthetic ticket 只通过自身字段指向隔离 fixture。

另需冻结 `call_prefix` 机械派生、implementation identity exact object schema，并
要求 identity artifact object 与 ticket/projection/issuance 中对象 exact equal。

## 最大允许结论

projection、consumer、presentation 和 G6 四类问题已闭环，但 schema 尚不能唯一
构造或实现；parser-only implementation、G5/G6 和 history 继续锁定。
