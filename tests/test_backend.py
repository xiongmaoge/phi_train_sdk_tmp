"""SlurmBackend 单测：不打真 SSH，用假 _run/_exec 替换，只测命令组装与解析。"""

import pytest

from phi_train_slurm.job.status import JobStatus
from phi_train_slurm.slurm.backend import SlurmBackend, SlurmError


def make_backend(**kwargs) -> SlurmBackend:

    return SlurmBackend(host="login01", **kwargs)


def patch_run(backend, fake):
    """把 _run 换成 fake(remote_cmd, timeout, stdin) -> stdout。"""

    backend._run = fake


# ---- ssh 命令组装 ----


def test_ssh_argv():

    backend = make_backend(user="alice", port=2222, identity_file="/tmp/id")

    argv = backend._ssh_argv("squeue -j 1")

    assert argv[0] == "ssh"
    assert argv[:3] == ["ssh", "-p", "2222"]
    assert "-o" in argv and "BatchMode=yes" in argv
    assert "-i" in argv and "/tmp/id" in argv
    assert argv[-2] == "alice@login01"
    assert argv[-1] == "squeue -j 1"


def test_ssh_argv_defaults():

    argv = make_backend()._ssh_argv("true")

    assert argv[-2] == "login01"
    assert "-i" not in argv


# ---- submit ----


def test_submit_parses_job_id():

    backend = make_backend()

    patch_run(backend, lambda cmd, timeout=60, stdin=None: "Submitted batch job 12345\n")

    assert backend.submit("#!/bin/bash\ntrue\n") == "12345"


def test_submit_script_via_stdin():

    captured = {}

    def fake(cmd, timeout=60, stdin=None):
        captured["cmd"] = cmd
        captured["stdin"] = stdin
        return "Submitted batch job 1\n"

    backend = make_backend()
    patch_run(backend, fake)

    backend.submit("SCRIPT")

    assert captured["stdin"] == "SCRIPT"
    assert "cat >" in captured["cmd"]
    assert "sbatch" in captured["cmd"]


def test_submit_unparseable_output():

    backend = make_backend()

    patch_run(backend, lambda cmd, timeout=60, stdin=None: "garbage")

    with pytest.raises(SlurmError):
        backend.submit("x")


# ---- status ----


def test_status_prefers_squeue():

    def fake(cmd, timeout=60, stdin=None):
        if cmd.startswith("squeue"):
            return "RUNNING\n"
        return "COMPLETED|0:0\n"

    backend = make_backend()
    patch_run(backend, fake)

    assert backend.status("42") is JobStatus.RUNNING


def test_status_falls_back_to_sacct():

    def fake(cmd, timeout=60, stdin=None):
        if cmd.startswith("squeue"):
            return ""
        return "NODE_FAIL|1:0\n"

    backend = make_backend()
    patch_run(backend, fake)

    assert backend.status("42") is JobStatus.FAILED


def test_status_squeue_error_falls_to_sacct():
    # 作业已结束时 squeue 会报 Invalid job id（rc!=0），应转查 sacct

    def fake(cmd, timeout=60, stdin=None):
        if cmd.startswith("squeue"):
            raise SlurmError("Invalid job id specified")
        return "COMPLETED|0:0\n"

    backend = make_backend()
    patch_run(backend, fake)

    assert backend.status("42") is JobStatus.SUCCEEDED


def test_status_cancelled_by():

    def fake(cmd, timeout=60, stdin=None):
        if cmd.startswith("squeue"):
            return ""
        return "CANCELLED by 12345|0:0\n"

    backend = make_backend()
    patch_run(backend, fake)

    assert backend.status("42") is JobStatus.CANCELLED


def test_status_unknown():

    backend = make_backend()
    patch_run(backend, lambda cmd, timeout=60, stdin=None: "")

    assert backend.status("42") is JobStatus.UNKNOWN


def test_status_rejects_injection():

    backend = make_backend()

    with pytest.raises(ValueError):
        backend.status("1; rm -rf ~")


# ---- 路径 / 日志 ----


def test_path_exists():

    backend = make_backend()

    backend._exec = lambda cmd, timeout=60, stdin=None: (0, "", "")
    assert backend.path_exists("/data")

    backend._exec = lambda cmd, timeout=60, stdin=None: (1, "", "")
    assert not backend.path_exists("/data")


def test_dir_nonempty():

    backend = make_backend()

    backend._exec = lambda cmd, timeout=60, stdin=None: (0, "", "")
    assert backend.dir_nonempty("/data")

    backend._exec = lambda cmd, timeout=60, stdin=None: (1, "", "")
    assert not backend.dir_nonempty("/data")


def test_read_log_tail():

    backend = make_backend()
    patch_run(backend, lambda cmd, timeout=60, stdin=None: "line1\nline2\n")

    lines = list(backend.read_log("/logs/a.out", tail=2))

    assert lines == ["line1\n", "line2\n"]


def test_read_log_tail_command():

    captured = {}

    def fake(cmd, timeout=60, stdin=None):
        captured["cmd"] = cmd
        return ""

    backend = make_backend()
    patch_run(backend, fake)

    list(backend.read_log("/logs/a.out", tail=50))

    assert "tail -n 50" in captured["cmd"]
    assert "/logs/a.out" in captured["cmd"]
