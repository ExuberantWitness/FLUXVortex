# 运行日志摘要

## 1. 计划与静态检查

- 完整读取 `experiment` 技能及计划、清单、证据层级模板。
- 专用 `bash_exec`、artifact/memory 接口不可用，改用本地非交互终端；
  没有因此放宽任何物理或数值门。
- 新增长时测试后，Ruff、`py_compile`、`git diff --check` 通过。

## 2. 4 步 pilot

执行 `_long_active_build()` 后调用
`Q16CudaRealFSITrajectory.advance(step_count=4, delta_time=0.04)`。

- PASS，30.0299 s；
- 粒子 `206 -> 413 -> 624 -> 836 -> 1050`；
- 最大耦合残差 `4.2901450044696315e-11`；
- 5 个气动坐标 cell 释放均为 3；
- 第 3 个坐标 node raw/topology 为 3/4，证明共享拓扑投影实际发生；
- 无生产代码修改。

## 3. 8 步 focused gate

```text
PYTHONPATH=platform:platform/warp_vpm:src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q platform/warp_vpm/test_q16_dvm_node_ribbon_fsi.py::\
test_eight_step_active_dvm_fsi_resumes_one_conditional_release_lineage
```

结果：`1 passed, 14 warnings in 59.91s`。警告均来自 PyTorch 对
`torch.jit.script_method` 的弃用提示。

## 4. 指标证据运行

用相同 `_long_active_build()` 执行 4 步 prefix、恢复 4 步、验证结果哈希，
再请求第 9 个耗尽坐标并比较失败前后 parent。结果 8/8 PASS，57.8235 s；
完整逐步数值见 `metrics.json` 和 `metrics.md`。

## 5. 受影响回归

运行 mandatory mode、载荷包、冲量转移、Ptera resolved transfer、DVM
source/ribbon、气动事务、增量几何、Q16 单步/轨迹及新增长时门共 13 个
测试文件。结果：`74 passed, 14 warnings in 121.61s`。

原始终端会话由当前工具托管，但因缺少 `bash_exec` 无法导出专用
`bash.log` session；关键命令、输出、指标和哈希已保存在本证据包。
