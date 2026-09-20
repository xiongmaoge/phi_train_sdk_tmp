"""submit 入口：username 解析与 --wait 交互输出格式。"""

import re

import pytest

from phi_train_slurm.job.status import JobStatus
from phi_train_slurm.submitter.submit import (
    _fit,
    _queue_job,
    _wait_header,
    _wait_row,
    resolve_username,
)


# ---- username 解析 ----


def test_resolve_username_prefers_argument(monkeypatch):

    monkeypatch.setenv("PHI_USERNAME", "env-user")

    assert resolve_username("arg-user") == "arg-user"


def test_resolve_username_falls_back_to_env(monkeypatch):

    monkeypatch.setenv("PHI_USERNAME", "env-user")

    assert resolve_username() == "env-user"


def test_resolve_username_required(monkeypatch):

    monkeypatch.delenv("PHI_USERNAME", raising=False)

    with pytest.raises(ValueError, match="username is required"):
        resolve_username()


# ---- --wait 输出格式 ----


def test_wait_header_has_columns():

    header = _wait_header()

    for column in ("Time", "Job", "Name", "Status", "Detail"):
        assert column in header

    assert "\u2500" in header


def test_wait_row_format_no_color(monkeypatch):

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    row = _wait_row("4", "phi-smoke-cpu", JobStatus.SUCCEEDED)

    assert re.match(r"^\d{2}:\d{2}:\d{2}  ", row)
    assert "4" in row
    assert "phi-smoke-cpu" in row
    assert "SUCCEEDED" in row
    assert "\033" not in row


def test_wait_row_detail_appended(monkeypatch):

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    row = _wait_row("4", "n", JobStatus.PENDING, detail="comp001")

    assert row.endswith("comp001")


def test_fit_truncates_long_name(monkeypatch):

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    row = _wait_row("4", "very-long-job-name", JobStatus.RUNNING)

    assert "very-long-jo.." in row


def test_wait_row_colored_on_tty(monkeypatch):

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)

    row = _wait_row("4", "n", JobStatus.RUNNING)

    assert "\033[33m" in row


def test_wait_row_no_color_for_unknown(monkeypatch):

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert "\033" not in _wait_row("4", "n", JobStatus.UNKNOWN)


# ---- squeue 详情查询 ----


class QueueBackend:

    def __init__(self, jobs=None, fail=False):

        self.jobs = jobs
        self.fail = fail

    def list_jobs(self, user=None):

        if self.fail:
            raise RuntimeError("ssh down")

        return self.jobs


def test_queue_job_found():

    backend = QueueBackend(
        [
            {"job_id": "4", "name": "x", "reason": "comp001", "nodes": "comp001"},
            {"job_id": "5", "name": "y", "reason": "None", "nodes": "comp002"},
        ]
    )

    assert _queue_job(backend, "5")["name"] == "y"


def test_queue_job_missing_returns_none():

    assert _queue_job(QueueBackend([]), "4") is None


def test_queue_job_failure_returns_none():

    assert _queue_job(QueueBackend(fail=True), "4") is None
