# transport-v2.3 G6 unnamed-inode capability probe

**时间**：2026-07-28T21:36:11+08:00  
**role**：read-only protocol-selection evidence; no parser/ticket/history authority  
**claim-state change**：`false`

## 病因

第七次 authorization-schema audit 证明 named temp 的
`stat(name)==owned → unlinkat(name)` 不是条件删除：active pathname swap 可使
cleanup删除第三方同名 entry。继续增强 check-then-unlink不能闭合该反例。

## 隔离探针

探针只在 `platform/tests/.v23-otmpfile-*-probe-213409` 创建 owner-only临时目录，
不使用 production Q/ticket/result/marker namespace；退出前删除其唯一 owned
link和空目录。

当前 filesystem：

```text
statfs type = ext2/ext3 family
os.O_TMPFILE available = true
```

结果：

```text
openat(ticket_dir, ".", O_RDWR|O_TMPFILE|O_CLOEXEC) = PASS
linkat(tmp_fd, "", ticket_dir_fd, canonical, AT_EMPTY_PATH) = FAIL ENOENT
linkat(AT_FDCWD, "/proc/self/fd/<tmp_fd>",
       ticket_dir_fd, canonical, AT_SYMLINK_FOLLOW) = PASS
fstat(tmp_fd).(st_dev,st_ino)
  == lstat(canonical).(st_dev,st_ino) = true
post-link st_nlink = 1
```

## 裁决

本环境存在可执行的 deletion-free no-replace commit primitive：

1. `O_TMPFILE` 创建无 pathname inode；
2. 写入并 `fsync(tmp_fd)`；
3. 通过 kernel-owned `/proc/self/fd/<fd>` 与 `AT_SYMLINK_FOLLOW` 直接
   `linkat` 到 absent canonical name；
4. canonical 已存在时 `linkat` fail，不覆盖；
5. pre-link failure只关闭 unnamed fd，无 pathname cleanup；
6. post-link failure保留 canonical。

因此下一版 schema应删除 named temp、hard-link-then-unlink和
`nlink=1→2→1`，改写为 unnamed `nlink=0→1` state machine。该探针不授权实现
parser/G6，不生成 credential、ticket、marker、result或 history。
