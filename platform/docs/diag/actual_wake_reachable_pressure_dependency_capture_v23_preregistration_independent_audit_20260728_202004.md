# G3 dependency-capture protocol-only prereg 独立审计

**时间**：2026-07-28T20:20:04+08:00  
**review independence**：fresh-context same-family, read-only  
**被审工件**：

- `actual_wake_reachable_pressure_dependency_capture_v23_preregistration_20260728_200850.md`
  SHA256 `6b5ab72b9f6b0002e57e730a67585f989162943a240998c108b1c61bf4faf97f`
- `actual_wake_reachable_pressure_dependency_capture_v23_preregistration_20260728_200850.json`
  SHA256 `beac3e0bbb6473a0765612f5ccd7cde111781d40929d4731a3e2001fdbf1bae3`

## 裁决

```text
FAIL / G3_DEPENDENCY_CLOSURE_CAPTURE_NO_GO
blockers = 8
```

本审计没有运行 physical history、没有写 reserved manifest/auth/token/marker/result，
没有修改模型或 claim state。

## 通过项

- timestamped 与 fixed latest 的 MD、JSON 分别逐字节一致；
- wrapper、tests、transport prereg、wrapper definition、frozen guard 引用 SHA
  全部匹配；
- 七项 retired quarantine SHA 匹配；
- reserved manifest、新 auth/result/marker、旧 canonical/latest 均不存在；
- protocol-only 边界、三 Discovery/三 Replay、G0/G2/G4 前置、失败语义与 hard
  nonclaims 方向正确。

## Blockers

1. **Discovery/candidate 循环**：D1–D3 后才生成 candidate，但 Discovery 又要求
   candidate parser/full-member fingerprint。必须改成 manifest-free/seed-bound
   Discovery，consensus 后机械 candidate，再由 V1–V3 执行完整 candidate path，
   并要求全部 `B_D == B_V`。
2. **bootstrap 可污染 B**：只限制 repository-local preload 不够。必须冻结
   pre-wrapper `sys.modules`/file-backed state，禁止预导入 `importlib.util`、
   `importlib.metadata`、pkgutil/packaging/conda/readelf 等。当前最小 bootstrap
   实测 `209/218`，预导入 `importlib.util` 会变成 `210/219`。
3. **加载位点 inventory 不完整**：不能只扫第三方调用；必须覆盖完整 63-source
   rooted closure 中的 Import/ImportFrom、函数内 import、`__import__/importlib`、
   lazy attribute、ctypes/dlopen、extension/native loader side effect。
4. **owner 算法未冻结**：必须规定 wheel RECORD 的 CSV/词法/prefix 规则、conda
   noarch materialization 和 usrmerge `/lib↔/usr/lib` 同 dev/inode alias 证明；
   全部候选收集后要求唯一 owner，不能用优先级吞冲突。
5. **native identity 不可执行**：SONAME/build-id 均可能缺失，多 module alias
   必须完整排序。需要 kind-discriminated canonical identity；ELF 证据不能代替
   package owner。
6. **optional 语义不足**：本 campaign 应冻结 `O=∅`、`U=B∪R`、`E_final=U`；
   optional 需另开带每分支三复本和多 phase schema 的协议。
7. **non-interference 不充分**：须记录 measurement 前、第一次后、第二次后状态；
   三者 exact equal，另做 clean twin vs instrumented twin，以及 observer
   enter/exit 零增量。
8. **MD/JSON 非完整双权威**：successor 必须将全部规则复制进 JSON，或指定 MD
   为唯一 normative authority，并在 JSON 绑定其 SHA。可执行 successor 还须绑定
   G0、最终 G2/G4、bootstrap/matrix/fixture/supervisor 的确切 accepted path/SHA。

## Owner/native 实测补充

当前诊断 U=218 的 owner 可 `218/218` 唯一闭合，unknown=0、ambiguous=0；但这不
证明最终 U。39 个 native 中，23 个有 Python module identity，另 16 个有
SONAME；build-id 仅 11/39。enrichment 若在 target 内运行会把 closure 从
218 污染到 259，因此必须 out-of-process。

## 最大允许结论

当前工件是方向正确且 fail-closed 的 protocol-only 草案，但不是可执行、可晋升
的 dependency-capture 协议。`209/218/+9` 仍只是诊断，B/U/R 未建立，reserved
manifest 不得写入，科学结果保持 `UNKNOWN`。
