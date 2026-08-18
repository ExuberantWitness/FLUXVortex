# V5H12 amendment：§9.3 联合回归命令的进程隔离重定义

日期：2026-08-17（UTC）
性质：显式治理 amendment（PLAN.md 不可回写；依 HANDOFF §6 “若发现治理合同本身
错误，必须先形成显式 amendment，再实现”）
提出依据：`PAPER_REPRODUCTION_STATUS_AND_NEXT_PLAN_20260816.md` P2 判定 +
本目录 tracker G1-006 证据链
批准：仓库所有者于 2026-08-17 指示“推进计划尽快实现论文 case 复现”，采用
下述推荐选项。

## 1. 问题

HANDOFF §9.3 的字面联合命令要求六个测试文件在**同一个 pytest 进程**内按
序执行并全绿。实测证明该要求在“8 个 V5H11 文件不可变 + 只允许修改 4 个
V5H12 文件”的边界内不可满足：

- 六文件同进程：12 failed / 168 passed，全部 12 个失败位于冻结 V5H11 文件；
- **仅冻结 V5H11 三文件（executor+runner+coupling，无任何 V5H12 文件参与）
  即可复现完全相同的 12 个失败**；
- 机制：`test_fluxv_v5h11_baik_coupling.py` 在模块顶层真实 import
  pterasoftware/forward_flight_benchmarks/fluxvortex；pytest 在任何测试运行
  前完成全部收集，因此这些模块从第一个测试起就在 `sys.modules` 中，令
  V5H11 executor 的 origin-attest 测试与 V5H11 runner 的 runtime-inventory
  测试必然失败。V5H11 文件冻结，无法为其加隔离。

## 2. 采纳的选项（三选一中的选项 1）

**联合回归的合格执行方式 = 逐文件独立进程**：按 §9.3 的文件顺序，每个测试
文件在独立 pytest 进程中执行，六次全部通过即视为联合回归 PASS；另需附上
V5H12 executor+runner 两文件**同进程** 0 失败的补充证据（证明 V5H12 自身
无跨文件干扰）。

被否决的选项：conftest.py（超出 4 文件改动地图，需更强授权）；解冻 V5H11
测试（破坏不可变历史控制）。

## 3. 通过证据（2026-08-16/17）

逐文件独立进程（命令见 HANDOFF §9.2/§9.3 环境变量约定）：

```text
test_fluxv_v5h12_baik_w2_executor.py   33 passed
test_run_fluxv_v5h12_baik_w2.py        51 passed
test_fluxv_v5h11_baik_w2_executor.py   20 passed
test_run_fluxv_v5h11_baik_w2.py        42 passed
test_fluxv_v5h11_baik_coupling.py      11 passed
test_rvpm_ir_wrk3_stream.py            25 passed
```

补充同进程证据：V5H12 executor+runner 单进程 `84 passed / 0 failed`
（sys.modules 隔离 fixture 仅恢复测试期进程环境，未触碰任何
runner/executor gate）。

静态门：py_compile / Black / Ruff / git diff --check（含 no-index 逐文件）
全绿。V5H12↔V5H11 namespace 归一化 diff 经两轮独立 reviewer 复核仅含
prereg 许可 delta。

## 4. 效力与边界

- 本 amendment 仅重定义“联合回归命令”的执行方式，不改变任何科学合同、
  阈值、N、W2 输入、A/B 规则或 GT/scorer 封存状态。
- H4/G1-006 在本 amendment 下判定为 **PASS**，smoke（G2-002）的执行前置
  条件（原预注册联合命令全绿 + 最终四叶重冻 + 新 token verified）自此
  全部满足。
- 观测边界不变：`observation_access=none`；GT/scorer 在 inherited unlock
  gate（outer convergence + unlock token）之前保持 sealed。

## 5. 冻结引用

最终四叶（本 amendment 生效时的绑定对象）：

- runner `5e0777d82147827a0ebcd9520f3a6cfdade592bc392d3306ac77e7a9085f05fe`
- executor `5c74a9ffe245a0212aacf06067c477c6ddcc384e1c71b97a2aa1bd017bfb7053`
- runner test `e8b3de1271cc8cffa09e6e1252a68595662a2b2328bbdc943a18bdb5b3c3b2fa`
- executor test `3990bb32309eb69d858e768a4d476ff42e815524afdf38108aaf29314f875b07`

依赖闭合（G2-001）：manifest/token `/tmp/fluxv-v5h12-audit-20260816-4W8c03/`
（41 leaves + 56 runtime modules）已 verified；旧 V5H11 token fail closed。
