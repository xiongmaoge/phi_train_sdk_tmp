"""新机器一键初始化：从 apiserver 拉集群连接信息并装 SSH 公钥。

用法（每台新提交机一次性执行）：

    phi-train init --cluster ehpc_hz

流程：
  1. apiserver 查集群配置（host / user / port / password）
  2. 本机没有 SSH 密钥则生成（ed25519）
  3. 已能 BatchMode 免密登录 → 直接完成
  4. 否则用配置里的 password 装 公钥到登录节点：
     - 优先 SSH_ASKPASS（OpenSSH >= 8.4，纯标准库）
     - 失败回退 expect（macOS / 多数 Linux 自带）
  5. BatchMode 复验免密

密码只在这一次 ssh-copy-id 的内存里使用，不写任何本地文件；
之后所有链路走密钥认证，password 字段不再参与。
"""

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from phi_train_slurm.client import apiserver

KEY_CANDIDATES = ("id_ed25519", "id_rsa")


def _ssh_target(host: str, user: str | None) -> str:
    return f"{user}@{host}" if user else host


def _ssh_common_args(port: int = 22, identity_file: str | None = None) -> list[str]:

    argv = ["-p", str(port), "-o", "ConnectTimeout=10"]

    if identity_file:
        argv += ["-i", identity_file]

    return argv


def batchmode_ok(
    host: str,
    user: str | None,
    port: int = 22,
    identity_file: str | None = None,
) -> bool:
    """免密登录是否已可用（BatchMode，与 SlurmBackend 的认证方式一致）。"""

    argv = (
        ["ssh", "-o", "BatchMode=yes"]
        + _ssh_common_args(port, identity_file)
        + [_ssh_target(host, user), "true"]
    )

    return subprocess.run(argv, capture_output=True).returncode == 0


def find_or_create_key(ssh_dir: Path | None = None) -> Path:
    """返回本机默认私钥路径；没有则生成一个 ed25519。"""

    ssh_dir = ssh_dir or Path.home() / ".ssh"

    ssh_dir.mkdir(mode=0o700, exist_ok=True)

    for name in KEY_CANDIDATES:

        key = ssh_dir / name

        if key.is_file():
            return key

    key = ssh_dir / "id_ed25519"

    subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(key),
            "-C",
            "phi-train",
        ],
        check=True,
        capture_output=True,
    )

    return key


def _install_key_askpass(
    key: Path,
    host: str,
    user: str | None,
    port: int = 22,
    password: str = "",
) -> bool:
    """SSH_ASKPASS 方式装公钥（OpenSSH >= 8.4，纯标准库）。"""

    with tempfile.NamedTemporaryFile(
        "w", suffix=".sh", delete=False, encoding="utf-8"
    ) as f:

        f.write(f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(password)}\n")

        askpass = f.name

    try:

        os.chmod(askpass, 0o700)

        env = dict(
            os.environ,
            SSH_ASKPASS=askpass,
            SSH_ASKPASS_REQUIRE="force",
            DISPLAY="phi-train",
        )

        rc = subprocess.run(
            [
                "ssh-copy-id",
                "-i",
                f"{key}.pub",
                "-o",
                "StrictHostKeyChecking=accept-new",
            ]
            + _ssh_common_args(port)
            + [_ssh_target(host, user)],
            env=env,
            capture_output=True,
            text=True,
        ).returncode

        return rc == 0

    finally:
        os.unlink(askpass)


def _install_key_expect(
    key: Path,
    host: str,
    user: str | None,
    port: int = 22,
    password: str = "",
) -> bool:
    """expect 方式装公钥（SSH_ASKPASS 不可用时的回退）。"""

    env = dict(os.environ, PHI_PW=password)

    script = (
        "set timeout 30\n"
        "spawn ssh-copy-id -i KEYPUB -o StrictHostKeyChecking=accept-new "
        "PORTARGS TARGET\n"
        "expect {\n"
        '  "*assword*" { send "$env(PHI_PW)\\r"; exp_continue }\n'
        "  eof\n"
        "}\n"
        "catch wait result\n"
        "exit [lindex $result 3]\n"
    ).replace(
        "KEYPUB", shlex.quote(f"{key}.pub")
    ).replace(
        "PORTARGS", f"-p {port}"
    ).replace(
        "TARGET", shlex.quote(_ssh_target(host, user))
    )

    rc = subprocess.run(
        ["expect", "-c", script],
        env=env,
        capture_output=True,
        text=True,
    ).returncode

    return rc == 0


def bootstrap_cluster(cluster_name: str | None = None) -> dict:
    """初始化本机到集群的免密登录；返回 {cluster, host, key, installed}。

    apiserver 集群配置里必须有 password 字段（首次装公钥用）。
    """

    cluster = apiserver.get_cluster(cluster_name)

    host = cluster["host"]
    user = cluster.get("user")
    port = int(cluster.get("port") or 22)

    key = find_or_create_key()

    if batchmode_ok(host, user, port, cluster.get("identity_file")):
        return {
            "cluster": cluster.get("name"),
            "host": host,
            "key": str(key),
            "installed": False,
        }

    password = cluster.get("password")

    if not password:
        raise ValueError(
            f"cluster '{cluster.get('name')}' has no password on apiserver "
            "and this machine cannot log in without a password; "
            "either add password on the apiserver or run "
            f"ssh-copy-id {_ssh_target(host, user)} manually"
        )

    installed = _install_key_askpass(key, host, user, port, password)

    if not installed:
        installed = _install_key_expect(key, host, user, port, password)

    if not installed:
        raise RuntimeError(
            f"failed to install ssh key on {host} (tried SSH_ASKPASS "
            "and expect); check the password on apiserver / network"
        )

    if not batchmode_ok(host, user, port):
        raise RuntimeError(
            f"key installed on {host} but BatchMode login still fails; "
            "check sshd config (PubkeyAuthentication) on the login node"
        )

    return {
        "cluster": cluster.get("name"),
        "host": host,
        "key": str(key),
        "installed": True,
    }
