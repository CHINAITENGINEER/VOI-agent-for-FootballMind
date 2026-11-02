# Agent 同一运行的跨进程互斥

`football_agent.audit.RunStore.lock(run_id, timeout=0.0)` 为同一存储目录下的同一运行提供单写者锁。默认立即返回；占用时抛出 `RunBusyError`，异常包含 `code="RUN_BUSY"` 和 `run_id`。正数 `timeout` 可等待指定秒数，等待线程锁与操作系统锁共用这一期限。

```python
from football_agent.audit import RunStore, RunBusyError

store = RunStore("var/parallel/agent/round2/runs")
try:
    with store.lock(run_id):
        record = store.load(run_id)
        # 完整的读取、恢复判断、工具执行、费用更新和存储必须在这里完成。
except RunBusyError:
    # HTTP 返回 409；CLI 返回非零退出状态和 RUN_BUSY。
    pass
```

Workflow 已将锁覆盖到 `get_run`、完整 `step`、完整 `run_to_completion` 以及新运行的首次持久化。同线程可在同一或不同 RunStore 实例间重入，所以 `run_to_completion` 内部调用 `get_run/step` 保持同一锁的所有权。其他线程和进程被排除；不同运行独立。`get_run` 在获得锁以后才判断是否发生工具中断，避免把其他进程正在执行的调用误判为崩溃。

Windows 使用 `msvcrt.locking`；POSIX 使用 `fcntl.flock`。锁文件保留在 `<store>/.locks/<run_id>.lock`，运行 ID 仍严格校验为 UUID。程序既不删除锁文件，也不根据 PID、mtime 或超时抢占所有权。进程退出后操作系统释放所有权；随后由既有审计恢复逻辑记录工具中断和未知费用预留。低层 `RunStore.load/verify` 可只读完整原子快照；需要修改或执行恢复时应经 Workflow 或显式持有上述锁。

当前在本机 Windows 完成真实子进程验证，POSIX 分支待 Linux 执行。共享网络文件系统的锁语义尚未验收；当前交付范围为本地展示存储。

验证命令：

```powershell
python -m pytest tests/agent/test_round2_lock.py tests/agent/test_workflow.py -q
```

实测 65 项通过，包含原有审计完整性与崩溃恢复回归。锁专项 11 项通过：跨线程重入/互斥、不同运行并行、有限等待、真实子进程崩溃释放、实际 CLI resume 与 HTTP 409、占用期间不恢复活动工具、不重复调用或计费。所有构造工具与概率都保留 `mode=test` 标记，未发出网络请求。

保存的验收清单为 `var/parallel/agent/round2/lock-validation/validation.json`，JUnit 为同目录 `pytest.xml`；`fixtures/` 保留正常完成一次工具及工具中途进程退出两份运行、输入输出引用和外部测试调用计数。两份记录均只有一次调用、0.1 测试费用；崩溃案例仅保守预留一次未知费用，审计链校验均为 OK。
