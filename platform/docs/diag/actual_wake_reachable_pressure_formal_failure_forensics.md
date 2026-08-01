# S3ai-v2.2 正式 one-shot 终止取证：NumPy lazy-import 协议错件

**取证时间**：2026-07-28T18:33:14+08:00  
**状态**：`FORMAL TRANSPORT FAILURE / RESULT ABSENT / RETRY UNAUTHORIZED`  
**物理裁决**：`UNKNOWN`  
**claim-state change**：`false`

## 0. 结论

本次 one-shot 不是科学意义上的 `PROTOCOL-NO-GO`；现存 checkpoint 也没有
观察到授权源文件漂移。它在完成 31-history collector、aggregation 和
`_validate_frozen_result()` 之后，因 wrapper 自身的 runtime-identity 规则误把
NumPy 的正常 lazy import 当成“dependency source set drift”而拒绝发布。
失败进程没有保存 end-fingerprint 差分，因此不能排除一个同时发生、随后恢复的
瞬时漂移；但正式路径必然发生的 170→179 lazy-import set growth 已经是本次
invariant 失败的充分原因。

可重复的最小指纹为：

```text
wrapper import-time external Python source set = 170
call np.polynomial.legendre.leggauss(8)
end external Python source set = 179
added = 9 numpy.polynomial pure-Python source files
removed = 0
changed bytes/identity among the original set = 0
```

正式 pressure observation 的必经函数
`_value_only_pressure_model()` 调用同一个
`np.polynomial.legendre.leggauss(line_quadrature_order)`。wrapper 的 import
warmup 只调用 `host_platform.platform()` 和 `np.__config__.show()`，没有预载
`numpy.polynomial`。因此“import-time exact loaded-source set 必须等于 end
set”的 invariant 对这条正式执行路径从设计上必然失败。

## 1. 原始事实

### 1.1 进程与产物

- permanent marker claimed：
  `2026-07-28T08:19:49.411432+00:00`
  （北京时间 16:19:49）；
- 最后确认运行：2026-07-28 18:27:14+08:00；
- failure log mtime：2026-07-28 18:28:41.771405648+08:00；
- 进程随后不存在；
- canonical result：不存在；
- latest result：不存在；
- 临时 result/core dump：未发现；
- 只有 881-byte traceback log。

### 1.2 锁定证据

| Artifact | SHA256 |
|---|---|
| formal failure log | `61ae80785419e6502c0e4544ba8d9757785c36fbb6e24bb7cac8bba988c8c60d` |
| permanent attempt marker | `42f9cb852128b24d2e28330ebf2ad9911764c845fba6370b4220aadbd5d6d778` |
| execution authorization | `39ebcd7b5a51e9ccd400c211cc3025952b9f27be9c09ad296f1dfc1a0bf5a75e` |

marker 明确记录：

```text
retry_allowed = false
```

所以不能把 wrapper bug 当作重新消耗同一授权的理由。

### 1.3 Traceback

失败点：

```text
run_authorized_once
  → runtime_end = _runtime_identity()
  → loaded Python dependency source set or bytes changed after wrapper import
```

代码顺序证明在该调用之前已经正常返回：

```text
observations = guard._collect_frozen_histories(contract)
result = guard.aggregate_frozen_histories(observations, contract, ...)
_validate_frozen_result(result, observations, ...)
```

因此可证明“collector、aggregate 和 frozen-result structure/value-hash
validation 已经运行到结束”；不能证明 15 个 scientific checks 全通过，也不能
知道 `stage_decision`。该值只存在于已退出进程的内存中，没有序列化。

## 2. Checkpoint 漂移检查及证据边界

### 2.1 Repository execution closure

execution authorization 绑定的 63 个 Python execution source：

```text
changed = 0
missing = 0
```

wrapper、guard、62-file source closure 和所有 claim-runtime 源在事后
checkpoint 的 SHA256 均与授权一致。`find` 也未发现任何 formal Python source
在 attempt 开始后改变 mtime。

### 2.2 Import-time external Python closure

authorization 保存的 170 个 external pure-Python dependencies：

```text
changed SHA/device/inode/size = 0
missing = 0
```

所以事后 checkpoint 没有观察到原有文件 bytes/identity 改变；独立最小复现则
证明正式路径必然使 **set grow**。由于失败进程没有序列化 end-fingerprint
diff，不能把“没有观察到”加强为“运行期间不可能存在随后恢复的瞬时漂移”。

## 3. 确定性最小复现

使用同一 Python 环境、`-B`、只 import wrapper，不读取 token、不 claim marker、
不运行 history：

```python
before = wrapper._IMPORT_TIME_PYTHON_DEPENDENCY_FINGERPRINTS
wrapper.np.polynomial.legendre.leggauss(8)
after = wrapper._loaded_python_dependency_fingerprints()
```

结果：

```text
before_count = 170
after_count = 179
changed = []
removed = []
```

新增 9 个文件：

| Path suffix | SHA256 |
|---|---|
| `numpy/polynomial/__init__.py` | `8064b02cda4f0a95df3e08894ac815a15b09d004b573efcc5a518e7a21b9e6c2` |
| `numpy/polynomial/_polybase.py` | `6f49028938149bc0f9402fcb5929bac8dbf00bbf67a4301e92c2b4bd03dc8824` |
| `numpy/polynomial/chebyshev.py` | `1621159a3acd55b85c6301b951012af66a910584cf86f1ab902ff2e80b314688` |
| `numpy/polynomial/hermite.py` | `8bc58c5e0038edc2792e214a8077556299a14326abbaceda26bb04debe096e6a` |
| `numpy/polynomial/hermite_e.py` | `244c341316423611ca5a98fb0b948a3f7dd04ebe88e307a04d4c8fbbd50ac8a3` |
| `numpy/polynomial/laguerre.py` | `2d8568238e6285355703b4af0dff07fcaac0bff2fac055dd0fbba3f830f6d7eb` |
| `numpy/polynomial/legendre.py` | `65667892aeceb7e239982303ed6cb910a06d23c8dcf3caac74f1b2d0c65a160f` |
| `numpy/polynomial/polynomial.py` | `9da743f753387a41e92d56c6c5763307e83b8932bdc8ce039408e0f7926c8cff` |
| `numpy/polynomial/polyutils.py` | `68064957a6f465962e2b520a0cf1e188981299757f7948a48d7714a132c7c3b6` |

这与 formal pressure 路径的 `leggauss` 调用完全同构，足以证实 deterministic
protocol bug；无需、也禁止为取证重跑 31 histories。

## 4. 病因挂树与裁决

### 数据指纹

```text
all 63 bound repository sources match authorization at post-failure checkpoint
all 170 import-time external sources match authorization at post-failure checkpoint
one required lazy import adds exactly 9 sources
failure occurs only at runtime_end equality check
no result publication
coincident transient drift is unrecoverable and not ruled out
```

### Claim 映射

- aerodynamic target `…b3e3b`：保持 `open`，没有新物理证据可晋升；
- 15-check formal result：`UNOBSERVED`，不是 false；
- one-shot transport invariant
  “import-time exact loaded Python source set equals end set”：
  **falsified as implemented for the declared execution path**；
- permanent attempt：已消耗且不可重试；
- typed quadrature/hp/governance research artifacts：不受该 failure 反转。

### 缺件还是错件

这是 **protocol/transport 组成部分错**：

- 正确目标应是“所有被实际加载的 dependency 都属于预先声明/授权的 closure，
  且各自 bytes/device/inode 在运行中不变”；
- 当前实现错误地把“运行前已经加载的集合必须与运行后完全相等”当作稳定性；
- 它没有预载正式路径必需的 `numpy.polynomial`，也没有声明允许且绑定的 lazy
  dependency closure。

## 5. 合法修复方向

任何修复都需要新的 wrapper 定义、独立审计和新的 one-shot authorization。
现授权不得复用。

可接受的两个方向：

1. **完全 warmup + frozen exact set**
   - 在冻结 snapshot 前显式预载正式路径所有外部 pure-Python/native lazy
     dependencies；
   - 证明 warmup 具有 execution-path exact coverage；
   - start/end 对每个文件做 bytes/device/inode equality。
2. **预声明 allowlisted dependency closure（更稳健）**
   - authorization 绑定允许加载的完整外部 dependency manifest；
   - start/end 的实际 loaded set 都必须是该 manifest 的子集；
   - end 新增项只有在 manifest 中且 hash/identity 与授权一致才允许；
   - 未声明新增、删除/替换已加载文件、bytes/identity drift 均 fail closed。

禁止：

- 删除 end check；
- 忽略所有新增模块；
- 只按包版本而不绑定文件；
- 改 marker 或伪造 result；
- 用内存已丢失的 stage decision 猜测物理结论；
- 静默重跑。

## 6. 下一步

1. 将本次 attempt 记录为 `FORMAL TRANSPORT FAILURE`，不解释为
   `PROTOCOL-NO-GO`；
2. 对本取证做 fresh-context 独立审计；
3. 继续不依赖该 result 的 fail-closed claim governance 与数值证书研究；
4. 若要重新获得 formal observation，必须提交修正版 wrapper 的独立
   authorization 请求；
5. 新授权到位前，h/p/forming/VES 与 Fig17/18/19 生产晋升仍保持锁定。
