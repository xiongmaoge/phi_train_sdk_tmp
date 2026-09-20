"""bootstrap（phi-train init）单测：全部 mock 掉 ssh 子进程。"""

from pathlib import Path

import pytest

from phi_train_slurm.submitter import bootstrap
from phi_train_slurm.submitter.bootstrap import bootstrap_cluster


@pytest.fixture(autouse=True)
def no_real_ssh(monkeypatch):
    """拦掉所有 subprocess 调用，防误连真实主机。"""

    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "batchmode_ok", lambda *a, **k: False)


# ---- 密钥发现 / 生成 ----


def test_find_existing_key(tmp_path):

    key = tmp_path / "id_ed25519"
    key.write_text("key")

    assert bootstrap.find_or_create_key(ssh_dir=tmp_path) == key


def test_find_prefers_ed25519(tmp_path):

    (tmp_path / "id_rsa").write_text("rsa")
    key = tmp_path / "id_ed25519"
    key.write_text("ed")

    assert bootstrap.find_or_create_key(ssh_dir=tmp_path) == key


def test_create_key_when_missing(tmp_path, monkeypatch):

    generated = tmp_path / "id_ed25519"

    calls = {}

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        generated.write_text("new key")
        return subprocess_result(0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    assert bootstrap.find_or_create_key(ssh_dir=tmp_path) == generated
    assert "ssh-keygen" in calls["argv"]


def subprocess_result(rc):
    import subprocess as sp

    return sp.CompletedProcess(args=[], returncode=rc)


# ---- bootstrap 流程 ----


def test_bootstrap_already_passwordless(monkeypatch):

    monkeypatch.setattr(
        bootstrap.apiserver,
        "get_cluster",
        lambda name=None: {
            "name": "c",
            "host": "h",
            "user": "root",
            "port": 22,
        },
    )
    monkeypatch.setattr(bootstrap, "batchmode_ok", lambda *a, **k: True)
    monkeypatch.setattr(
        bootstrap, "find_or_create_key", lambda ssh_dir=None: Path("/k")
    )

    result = bootstrap_cluster("c")

    assert result == {
        "cluster": "c",
        "host": "h",
        "key": "/k",
        "installed": False,
    }


def test_bootstrap_installs_key_with_password(monkeypatch):

    monkeypatch.setattr(
        bootstrap.apiserver,
        "get_cluster",
        lambda name=None: {
            "name": "c",
            "host": "h",
            "user": "root",
            "port": 22,
            "password": "pw",
        },
    )

    # 首次 batchmode 失败，装完 key 后成功
    monkeypatch.setattr(
        bootstrap,
        "batchmode_ok",
        lambda *a, **k: False,
    )

    attempts = iter([False, True])
    monkeypatch.setattr(
        bootstrap, "batchmode_ok", lambda *a, **k: next(attempts)
    )

    monkeypatch.setattr(
        bootstrap, "find_or_create_key", lambda ssh_dir=None: Path("/k")
    )
    monkeypatch.setattr(
        bootstrap, "_install_key_askpass", lambda *a, **k: True
    )

    result = bootstrap_cluster("c")

    assert result["installed"] is True


def test_bootstrap_requires_password_when_locked_out(monkeypatch):

    monkeypatch.setattr(
        bootstrap.apiserver,
        "get_cluster",
        lambda name=None: {"name": "c", "host": "h", "user": "root"},
    )
    monkeypatch.setattr(bootstrap, "batchmode_ok", lambda *a, **k: False)
    monkeypatch.setattr(
        bootstrap, "find_or_create_key", lambda ssh_dir=None: Path("/k")
    )

    with pytest.raises(ValueError):
        bootstrap_cluster("c")


def test_bootstrap_raises_when_install_fails(monkeypatch):

    monkeypatch.setattr(
        bootstrap.apiserver,
        "get_cluster",
        lambda name=None: {
            "name": "c",
            "host": "h",
            "user": "root",
            "password": "bad",
        },
    )
    monkeypatch.setattr(bootstrap, "batchmode_ok", lambda *a, **k: False)
    monkeypatch.setattr(
        bootstrap, "find_or_create_key", lambda ssh_dir=None: Path("/k")
    )
    monkeypatch.setattr(
        bootstrap, "_install_key_askpass", lambda *a, **k: False
    )
    monkeypatch.setattr(
        bootstrap, "_install_key_expect", lambda *a, **k: False
    )

    with pytest.raises(RuntimeError):
        bootstrap_cluster("c")
