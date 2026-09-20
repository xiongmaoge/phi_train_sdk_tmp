import time
from dataclasses import replace
from typing import Callable

from phi_train_slurm.job import registry
from phi_train_slurm.job.job_spec import JobSpec
from phi_train_slurm.job.status import JobStatus
from phi_train_slurm.slurm.launcher import render_sbatch_script

# 轮询退避上限：长训练作业没必要 30s 一查
MAX_POLL_INTERVAL = 3600.0


class JobHandle:
    """已提交作业的操作句柄：wait / logs / cancel。

    所有操作经 SlurmBackend 走登录节点，上层不直接碰 Slurm 命令。
    """

    def __init__(
        self,
        backend,
        job_id: str,
        spec: JobSpec,
        attempt: int = 1,
        registry_dir: str | None = None,
    ):

        self.backend = backend

        # root_job_id：首次提交的 id（注册表主键，CLI 用户持有的就是它）
        # job_id：当前 attempt 实际在跑的 id（requeue 后会变）
        self.root_job_id = job_id
        self.job_id = job_id
        self.spec = spec
        self.attempt = attempt
        self.registry_dir = registry_dir

    @property
    def status(self) -> JobStatus:
        return self.backend.status(self.job_id)

    def wait(
        self,
        timeout: float | None = None,
        poll_interval: float = 30,
        on_status: Callable[[JobStatus], None] | None = None,
    ) -> JobStatus:
        """轮询直到作业结束，返回最终状态。

        requeue.enabled 时：出现可重试状态（FAILED/REQUEUED，见
        job/status.py 的 RETRYABLE_SACCT_STATES）且 attempt < max_retries，
        自动用同一 JobSpec 重提交（注入 --resume）并继续等待。

        on_status 每次轮询到状态时回调（CLI 用它打印进度；requeue 触发的
        重提交在回调之后进行）。
        timeout 超时抛 TimeoutError；作业不受影响，仍可继续 wait。
        """

        deadline = (
            time.monotonic() + timeout if timeout is not None else None
        )

        interval = poll_interval

        while True:

            status = self.backend.status(self.job_id)

            if on_status is not None:
                on_status(status)

            if self._should_requeue(status):
                self._requeue()
                interval = poll_interval
                continue

            if status.is_terminal():
                return status

            if deadline is not None:

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    raise TimeoutError(
                        f"job {self.job_id} still {status.value}"
                    )

                time.sleep(min(interval, remaining))

            else:
                time.sleep(interval)

            interval = min(interval * 2, MAX_POLL_INTERVAL)

    def _should_requeue(self, status: JobStatus) -> bool:

        return (
            status in (JobStatus.FAILED, JobStatus.REQUEUED)
            and self.spec.requeue.enabled
            and self.attempt < self.spec.requeue.max_retries
        )

    def _requeue(self) -> None:
        """用同一 spec 重提交并原地更新句柄（job_id 换新，attempt +1）。

        job_name 保持不变：日志文件名带 %j（job id），不会互相覆盖。
        resume_from 设置且命令里还没有 --resume 时追加进去（训练代码契约：
        --resume <dir> 存在则从最近 checkpoint 续训）。
        """

        if (
            self.spec.requeue.resume_from
            and "--resume" not in self.spec.command
        ):
            self.spec = replace(
                self.spec,
                command=(
                    f"{self.spec.command.strip()} "
                    f"--resume {self.spec.requeue.resume_from}"
                ),
            )

        self.job_id = self.backend.submit(render_sbatch_script(self.spec))

        self.attempt += 1

        if self.registry_dir:
            registry.save(
                self.root_job_id,
                self.spec,
                current_job_id=self.job_id,
                attempt=self.attempt,
                registry_dir=self.registry_dir,
            )

    def logs(
        self,
        follow: bool = False,
        tail: int = 100,
    ):
        """逐行 yield 作业日志（生成器）。

        日志文件在 CPFS：`<log_dir>/<job_name>-<job_id>.out`，
        登录节点可读，backend 负责 SSH tail。follow=True 时持续输出，
        直到生成器被关闭（如 Ctrl-C）。err 日志为同名 .err 文件。
        """

        path = f"{self.spec.log_dir}/{self.spec.job_name}-{self.job_id}.out"

        return self.backend.read_log(path, tail=tail, follow=follow)

    def cancel(self) -> None:
        """取消作业（scancel）。"""

        self.backend.cancel(self.job_id)
