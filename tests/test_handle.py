"""JobHandle 单测：假 backend 驱动 wait 轮询 / requeue / 超时。"""

import pytest

from phi_train_slurm.job import registry
from phi_train_slurm.job.handle import JobHandle
from phi_train_slurm.job.status import JobStatus
from .test_spec import make_spec


class FakeBackend:
    """status() 按序弹出预设状态；只剩一个时持续返回它（模拟稳定状态）。"""

    def __init__(self, statuses):

        self.statuses = list(statuses)
        self.submitted = []
        self._next_id = 100

    def status(self, job_id):

        if len(self.statuses) > 1:
            return self.statuses.pop(0)

        return self.statuses[0]

    def submit(self, script):

        self.submitted.append(script)
        self._next_id += 1
        return str(self._next_id)

    def read_log(self, path, tail=100, follow=False):

        yield f"LOG {path}\n"


# ---- wait ----


def test_wait_success():

    backend = FakeBackend([JobStatus.RUNNING, JobStatus.SUCCEEDED])
    handle = JobHandle(backend, "10", make_spec())

    assert handle.wait(poll_interval=0) is JobStatus.SUCCEEDED


def test_wait_on_status_callback():

    backend = FakeBackend([JobStatus.RUNNING, JobStatus.SUCCEEDED])
    handle = JobHandle(backend, "10", make_spec())

    seen = []

    assert (
        handle.wait(poll_interval=0, on_status=seen.append)
        is JobStatus.SUCCEEDED
    )

    assert seen == [JobStatus.RUNNING, JobStatus.SUCCEEDED]


def test_wait_requeue_then_success():

    spec = make_spec(
        requeue={
            "enabled": True,
            "max_retries": 3,
            "resume_from": "/ckpt/x",
        }
    )
    backend = FakeBackend(
        [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.SUCCEEDED]
    )
    handle = JobHandle(backend, "10", spec)

    assert handle.wait(poll_interval=0) is JobStatus.SUCCEEDED

    assert len(backend.submitted) == 1
    assert "--resume /ckpt/x" in backend.submitted[0]
    assert handle.attempt == 2
    assert handle.job_id == "101"
    assert handle.root_job_id == "10"


def test_wait_requeue_on_requeued_status():

    spec = make_spec(requeue={"enabled": True, "max_retries": 3})
    backend = FakeBackend([JobStatus.REQUEUED, JobStatus.SUCCEEDED])
    handle = JobHandle(backend, "10", spec)

    assert handle.wait(poll_interval=0) is JobStatus.SUCCEEDED
    assert handle.attempt == 2


def test_wait_requeue_exhausted():

    spec = make_spec(requeue={"enabled": True, "max_retries": 2})
    backend = FakeBackend([JobStatus.FAILED])
    handle = JobHandle(backend, "10", spec)

    assert handle.wait(poll_interval=0) is JobStatus.FAILED
    assert len(backend.submitted) == 1  # attempt 1 重试一次后 attempt 2 不再重试


def test_wait_no_requeue_when_disabled():

    spec = make_spec(requeue={"enabled": False})
    backend = FakeBackend([JobStatus.FAILED])
    handle = JobHandle(backend, "10", spec)

    assert handle.wait(poll_interval=0) is JobStatus.FAILED
    assert backend.submitted == []


def test_wait_timeout():

    backend = FakeBackend([JobStatus.RUNNING])
    handle = JobHandle(backend, "10", make_spec())

    with pytest.raises(TimeoutError):
        handle.wait(timeout=0, poll_interval=0)


def test_requeue_no_resume_when_no_resume_from():

    spec = make_spec(requeue={"enabled": True, "max_retries": 3})
    backend = FakeBackend([JobStatus.FAILED, JobStatus.SUCCEEDED])
    JobHandle(backend, "10", spec).wait(poll_interval=0)

    assert "--resume" not in backend.submitted[0]


# ---- logs / 注册表 ----


def test_logs_path():

    backend = FakeBackend([JobStatus.RUNNING])
    handle = JobHandle(backend, "10", make_spec())

    lines = list(handle.logs())

    assert "/mnt/cpfs/logs/vla_test_job-10.out" in lines[0]


def test_requeue_updates_registry(tmp_path):

    spec = make_spec(requeue={"enabled": True, "max_retries": 3})
    backend = FakeBackend([JobStatus.FAILED, JobStatus.SUCCEEDED])
    handle = JobHandle(backend, "10", spec, registry_dir=str(tmp_path))

    handle.wait(poll_interval=0)

    record = registry.load("10", registry_dir=str(tmp_path))

    assert record is not None
    assert record.current_job_id == "101"
    assert record.attempt == 2
