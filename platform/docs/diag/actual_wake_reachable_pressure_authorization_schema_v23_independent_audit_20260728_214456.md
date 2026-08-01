# transport-v2.3 authorization schema successor 第八次独立复审

**时间**：2026-07-28T21:44:56+08:00  
**review independence**：two fresh-context same-family read-only audits plus sidecar cross-check  
**被审 MD SHA**：`d6da25243e6d821e5bb6e20b384d1ee2fc8bea9f09638bd59dbf05f6ca802cf9`  
**被审 JSON SHA**：`cae6bddddce673d7b6f25ab8f32528d28e17c997d7380e3b10a2c39e017da3ca`

## 裁决

```text
FAIL / TWO_REMAINING_CAUSAL_AND_FAILURE_BRANCH_BLOCKERS
remaining blocker groups = 2
```

MD/JSON raw SHA、JSON parse与 capability-probe binding均通过。未实现
parser/lease/G6，未生成 production credential/ticket/result，未运行 history。

## 已闭环

- typed DirectoryLease factory、exclusive fixture mkdir、fixed `platform/tests`
  base及持续 inode recheck；
- accepted/rejected 的 Q profile ownership与双向 prefix隔离；
- SourceMap seed→candidate→plan→accepted T/rejected no-TKT；
- evidence-bound `O_TMPFILE`、proc-fd linkat、`nlink=0→1`、无 named-temp cleanup；
- Q/Build/P、raw ticket、guard和 `S_pre=S_1=S_2`未回归；
- MD/JSON未发现独立 sidecar冲突。

## Blocker 1：pre-anchor IdentityCore 来源缺失

source trace 在 anchor前就必须写入
`provider/model/model_family/agent_id/trace_id`，当前唯一规则却只是事后等于
future `anchor.identity`。anchor又必须绑定 source-trace raw SHA，不能调到 trace
之前。

最小闭合顺序：

```text
ImplementationIdentityCoreV1
  := externally frozen five-field implementation trace identity
→ source trace(identity_core, SourceMap_seed)
→ anchor(identity_core + source-trace path/raw SHA)
→ candidate_bound_artifacts
→ plan
```

source trace与anchor分别只从同一 core复制五字段并 direct equality核验。

## Blocker 2：G6 EEXIST/failure custody branch未闭合

S4 将 `EEXIST`/任意 nonzero linkat归为 failure；统一 PRELINK handler又要求
canonical `ENOENT`，两者在 race/collision时直接冲突。custody destruction排在
该断言后，可能被异常跳过。

最小闭合：

```text
every G6 failure:
  finally destroy broker custody first (cannot be bypassed)
  close unnamed fd if open

linkat nonzero:
  if canonical ENOENT and close/re-walk checks pass:
    CLEAN_PRELINK_ABORT
  else:
    AUTHORIZATION_NAMESPACE_POISONED
  never delete/rename/overwrite canonical
  never reach token release
```

若 nonzero return 后 canonical恰好同 tmp inode，也按 ambiguous committed
namespace poison处理并保留 canonical。

## 最大允许结论

只允许建立新 timestamped successor并再次只读审计。capability probe不是执行
授权；parser/G6/production Q/M/A/C/Z/token/ticket/marker/result/history继续锁定，
科学裁决仍为 `UNKNOWN`。
