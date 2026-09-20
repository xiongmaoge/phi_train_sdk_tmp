"""SlurmJobManager 单测：假 backend 驱动提交流程与注册表。"""

import pytest

from phi_train_slurm.job import registry
from phi_train_slurm.job.manager import SlurmJobManager
from phi_train_slurm.job.status import JobStatus
from .test_spec import make_spec


class FakeManagerBackend:

    def __init__(self, existing_paths=None):

        self.host = "login01"
        self.existing_paths = set(existing_paths or [])
        self.mkdir_calls = []
        self.submitted_scripts = []
        self.status_calls = []
        self.cancel_calls = []

    def path_exists(self, path):
        return path in self.existing_paths

    def mkdir(self, path):
        self.mkdir_calls.append(path)

    def submit(self, script):
        self.submitted_scripts.append(script)
        return "123"

    def status(self, job_id):
        self.status_calls.append(job_id)
        return JobStatus.RUNNING

    def cancel(self, job_id):
        self.cancel_calls.append(job_id)

    def read_log(self, path, tail=100, follow=False):
        yield f"LOG {path}\n"


# ---- submit ----


def test_submit_renders_and_persists(tmp_path):

    backend = FakeManagerBackend(existing_paths={"/mnt/cpfs/data"})
    manager = SlurmJobManager(backend, registry_dir=str(tmp_path))

    spec = make_spec(mounts=["/mnt/cpfs/data:/data"])
    handle = manager.submit(spec)

    assert handle.job_id == "123"
    assert "--job-name=vla_test_job" in backend.submitted_scripts[0]
    assert tmp_path.joinpath("123.json").is_file()
    assert registry.load("123", registry_dir=str(tmp_path)).spec == spec

    # log_dir 不存在也应被创建（幂等 mkdir）
    assert "/mnt/cpfs/logs" in backend.mkdir_calls


def test_submit_rejects_missing_mount(tmp_path):

    manager = SlurmJobManager(FakeManagerBackend(), registry_dir=str(tmp_path))

    with pytest.raises(ValueError):
        manager.submit(make_spec(mounts=["/nope:/data"]))


# ---- status / logs / cancel 经注册表换算当前 job id ----


def test_status_without_record_uses_given_id(tmp_path):

    backend = FakeManagerBackend()
    manager = SlurmJobManager(backend, registry_dir=str(tmp_path))

    manager.status("777")

    assert backend.status_calls == ["777"]


def test_status_uses_current_job_id(tmp_path):

    backend = FakeManagerBackend()
    registry.save(
        "123", make_spec(), current_job_id="456", registry_dir=str(tmp_path)
    )
    manager = SlurmJobManager(backend, registry_dir=str(tmp_path))

    assert manager.status("123") is JobStatus.RUNNING
    assert backend.status_calls == ["456"]


def test_logs_requires_local_record(tmp_path):

    manager = SlurmJobManager(
        FakeManagerBackend(), registry_dir=str(tmp_path)
    )

    with pytest.raises(ValueError):
        list(manager.logs("999"))


def test_logs_reads_recorded_path(tmp_path):

    backend = FakeManagerBackend()
    registry.save("123", make_spec(), registry_dir=str(tmp_path))
    manager = SlurmJobManager(backend, registry_dir=str(tmp_path))

    lines = list(manager.logs("123"))

    assert "/mnt/cpfs/logs/vla_test_job-123.out" in lines[0]


def test_cancel_uses_current_job_id(tmp_path):

    backend = FakeManagerBackend()
    registry.save(
        "123", make_spec(), current_job_id="456", registry_dir=str(tmp_path)
    )
    manager = SlurmJobManager(backend, registry_dir=str(tmp_path))

    manager.cancel("123")

    assert backend.cancel_calls == ["456"]
