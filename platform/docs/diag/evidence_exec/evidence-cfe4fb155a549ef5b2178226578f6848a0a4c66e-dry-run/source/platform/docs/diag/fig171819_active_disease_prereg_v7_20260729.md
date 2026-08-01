# Fig17/18/19 fresh V4.1 Active Disease 冻结协议 v7

**日期**：2026-07-29  
**状态**：PRE-REGISTERED；写于 fresh151 `status=running`，未读取 fresh
精度统计  
**阶段**：G1/G2 证据启动根；不授权数据解释或模型修改

## 0. 规范组成

权威 active-disease 规范按顺序叠加 v3、v4、v5、v6 和本 v7。v6 SHA：

```text
b2ea263cc786791a46de194ee538d0354d294187c2d1262848bf98f2105f78c7
```

parent 规范继承本启动根。本 v7 只替换 detached execution/bootstrap 条款，
关闭 Git replace、checkout transform 和 Python startup contamination；科学门、
阈值、排名和 parent truth table 均不改变。

## 1. 所有 Git object 操作禁用 replacement

bootstrap 的每一个 Git 命令必须同时使用：

```text
GIT_NO_REPLACE_OBJECTS=1
git --no-replace-objects
```

并显式配置：

```text
-c core.hooksPath=/dev/null
-c filter.lfs.required=false
```

同时设置 `GIT_CONFIG_NOSYSTEM=1` 和 `GIT_CONFIG_GLOBAL=/dev/null`。禁止调用未带
这些选项的 `show`、`cat-file`、`rev-parse`、`diff-tree`、`ls-tree`、`archive`
或 worktree 命令。receipt 保存每条 bootstrap command 的 argv 和退出码。

commit、parent、tree 和 blob identity 只能由 no-replace raw object traversal
得到；不得使用可能跟随 replace refs 的 porcelain 输出。

## 2. 禁止 checkout：raw blob 直接物化

v5/v6 的 detached worktree/checkout 文字作废。bootstrap 不运行
`git checkout`、`git worktree`、`git archive`、clean/smudge filters 或任何 Git
hook。

固定 authorization 必须列出完整 runtime source closure：

```text
ordered[(repo_relative_path,
         git_mode,
         git_blob_oid,
         sha256,
         size_bytes)]
```

物化算法：

1. 用 no-replace `git ls-tree`/`cat-file` 从 raw evidence tree 逐项取得 mode、
   blob OID 和 exact bytes；
2. 只接受 regular blob modes `100644` 或 `100755`；拒绝 symlink `120000`、
   submodule `160000`、缺失路径和未知 mode；
3. 在仓库内由 launcher 专属的、预先不存在的目录创建文件；禁止 `/tmp`；
4. 写入 exact blob bytes，计算 size/SHA256，并与 authorization 逐项相等；
5. 重新枚举物化目录，路径集合必须与 runtime closure 精确相等；拒绝新增文件、
   `.pth`、软链接、目录穿越和大小写别名；
6. 在任何 project import 前再次逐字节 hash 全 closure。

attestation payload 和 launcher 本身也通过 no-replace raw blob 取得。系统
bootstrap 先验证 exact launcher blob SHA，再把该 exact blob 物化到专属目录；
不得从当前工作树执行同名文件。

## 3. 隔离 Python 启动

唯一允许解释器：

```text
/home/exuber/anaconda3/envs/fluxvortex/bin/python
```

launcher 首次启动固定为：

```text
/home/exuber/anaconda3/envs/fluxvortex/bin/python -I -S <verified-launcher>
```

其中：

- `-I` 忽略所有 `PYTHON*` 环境路径并隔离 user site；
- `-S` 禁止自动加载 `site`、`.pth`、`sitecustomize` 和 `usercustomize`；
- bootstrap 还必须拒绝环境中非空的 `PYTHONPATH`/`PYTHONHOME` 被转发；
- 当前工作树和当前工作目录不得出现在 `sys.path`；
- launcher 只能在验证 runtime environment manifest 后，以固定绝对路径显式
  加入授权的 standard-library/site-packages roots；
- 加入 roots 不得调用 `site.addsitedir()`，只允许 `sys.path.insert()`，因此不
  执行 `.pth`；
- 首次 project import 前再次检查 `sitecustomize`、`usercustomize` 和任何
  project module 均未出现在 `sys.modules`。

runtime environment manifest 必须绑定解释器 realpath/version、NumPy/PyYAML
等允许 dependency 的 distribution files/hashes；出现未授权 dependency path
或源码漂移即 `INVALID_EVIDENCE`。

## 4. Launcher 自身不得自证

外层 system bootstrap 在启动 Python 前完成并记录：

- no-replace attestation/evidence commit、parent 和 exact one-path tree delta；
- payload、authorization 和 launcher exact blob hashes；
- launcher materialized-file hash/mode/size 与 raw blob 相等；
- runtime closure path/mode/blob/hash/size 全等；
- 物化目录无附加成员。

verified launcher 只能重复这些检查，不能以自己的报告替代外层检查。外层
bootstrap receipt 和内层 launcher receipt 必须分别落档并相互绑定。

## 5. Receipt 必需字段

所有 selector/Prepare/Evaluate receipts 增加：

```yaml
git_no_replace_objects: true
git_config_nosystem: true
git_config_global: /dev/null
git_hooks_disabled: true
checkout_used: false
raw_blob_materialization: true
runtime_source_closure_verified: true
runtime_source_closure_sha256: <64 hex>
python_executable_realpath: /home/exuber/anaconda3/envs/fluxvortex/bin/python
python_isolated_flag: true
python_no_site_flag: true
python_startup_contamination_check: PASS
outer_bootstrap_receipt_sha256: <64 hex>
inner_launcher_receipt_sha256: <64 hex>
```

任一字段缺失或 false 为 `INVALID_EVIDENCE`。

## 6. 清理边界

物化目录路径必须同时满足：位于仓库
`platform/docs/diag/evidence_exec/` 下、包含 literal evidence SHA、由本次
launcher receipt 记录且启动前不存在。只有在 receipt 已关闭文件句柄、确认没有
子进程、逐项核对路径集合后，才允许删除该次目录；不得递归删除其父目录或任何
未登记路径。

## 7. 阶段边界

authorization、launcher、selector/Prepare/Evaluate 和测试必须绑定
active v3+v4+v5+v6+v7 及全部 parent 规范。v7 未获独立零数据 GO 前不得创建
evidence/attestation commits，不得运行 selector 或读取 contribution。
