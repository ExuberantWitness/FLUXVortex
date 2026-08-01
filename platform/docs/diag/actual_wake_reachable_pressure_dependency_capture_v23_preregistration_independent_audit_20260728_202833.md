# G3 dependency-capture protocol successor 独立复审

**时间**：2026-07-28T20:28:33+08:00  
**review independence**：fresh-context same-family, read-only  
**被审 MD SHA**：`2b673462d7946f762ef04b43a868c3d1bc0df94dce1fc718e8e656e38bd172a1`  
**被审 JSON SHA**：`56cb52aad717154cfdbffc6de453e83aa7c4bc8d941f7f037612f18a24dda981`

## 裁决

```text
FAIL / PROTOCOL_DEFINITION_NOT_YET_ACCEPTED
remaining blockers = 2
```

未写文件、未运行 history、未生成 manifest/auth/token/marker/result。

## 已闭环

原八项中的 seed Discovery→candidate Replay、pre-wrapper budget、完整 load-site
inventory、`O=∅`、`S_pre/S_1/S_2`、clean/instrumented twins、wheel/conda/usrmerge
方向、native nullable identity 及 normative Markdown authority 均基本闭环。

## 剩余 blockers

1. **raw event chain 比较矛盾**：协议一处要求 C/I twins raw event chain exact，
   另一处又允许 instrumented observer ledger 多出 `observed_import_request`。
   必须分别比较 C 组和 I 组的 raw chain；C↔I 只比较 membership/fingerprint/
   native/first-seen phase，以及过滤 observer-request 后的 canonical checkpoint
   projection，并冻结 filter/projection algorithm 和 SHA。
2. **distribution grammar 非无歧义**：
   `conda:<name>:<build>:<channel>:<subdir>` 中 channel URL 可含冒号；
   wheel normalized-name 规则也未冻结。必须对每 component 使用相同 uppercase
   `%HH` encoding，冻结 PEP 503 name normalization，并明确 RECORD distribution
   root 的确定算法，禁止启发式 split/rsplit。

## 最大允许结论

该 successor 已关闭绝大部分原缺陷，但仍不能唯一执行；B/U/R 未建立，production
capture 和 reserved manifest 保持锁定，physics=`UNKNOWN`。
