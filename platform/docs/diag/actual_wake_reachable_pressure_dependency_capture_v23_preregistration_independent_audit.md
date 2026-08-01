# G3 dependency-capture protocol definition 最终独立复审

**时间**：2026-07-28T20:35:25+08:00  
**review independence**：fresh-context same-family, read-only  
**被审 MD SHA**：`1b07c694bc1a2b6f24d04adad379931f5b3216ce8b09bb4ae9cad9f9d4ea9316`  
**被审 JSON SHA**：`c2b9cf3b2f560177449804440a9ea794110ce71fe328d2c67ca6312a04be9471`

## 裁决

```text
PASS / G3_PROTOCOL_DEFINITION_ACCEPTED
blockers = 0
```

该 PASS 只接受 protocol definition，可进入 executable instantiation；不代表
bootstrap/matrix/harness 已生成，不代表 B/U/R、reserved manifest、authorization
或 history 已存在。

## 核对结果

- timestamped/latest MD、JSON 各自 byte-identical；
- JSON 正确绑定 normative Markdown path/SHA；
- raw event chain 只在 D/V × clean/instrumented 同类组三复本内比较；
- C↔I 使用冻结 `checkpoint_event_projection_v1`，只过滤
  `observed_import_request`，保留 successful-load/removal、membership、phase、
  fingerprint并重建 chain；不能隐藏 removal；
-所有 12 child projection object/SHA 必须 exact equal；
- wheel/conda/dpkg component 使用 UTF-8 uppercase `%HH`，literal `:` 只作固定
  分隔，不再依赖 URL-aware heuristic split；
- wheel name 冻结 PEP 503，来源唯一 METADATA Name；
- RECORD root 唯一定义为 `.dist-info` lexical parent，row lexical join/normpath，
  environment-prefix escape 失败；
- conda noarch、usrmerge、unique owner、native nullable identity、seed→candidate→
  replay、O空集和三时点 non-interference 均未被破坏；
- formal history、collector、mesh、march 均未运行。

## 最大允许结论

G3 dependency-capture 的协议定义已经接受。下一步只能生成并审计 executable
instantiation；在其完成前，当前 `209/218/+9` 仍只是诊断，production capture、
reserved manifest 和科学结论保持锁定。
