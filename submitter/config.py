"""提交侧配置：登录节点连接信息。

集群配置的主数据源在 apiserver（按集群名拉取，见 client/apiserver.py）。
本模块的环境变量 / .env 只在 apiserver 不可达时兜底，读取优先级
（高到低，已存在的环境变量不被低优先级来源覆盖）：

1. 进程环境变量（CI / 脚本注入）
2. 当前目录 `.env`（与 phi_data_sdk 约定对齐的历史兜底）

| 环境变量 | 说明 |
| --- | --- |
| `PHI_SLURM_HOST` | 登录节点地址（必填，兜底用） |
| `PHI_SLURM_USER` | SSH 用户（缺省用本机用户） |
| `PHI_SLURM_IDENTITY_FILE` | SSH 私钥路径（缺省用 ~/.ssh 默认） |
| `PHI_SLURM_PORT` | SSH 端口（缺省 22） |
| `PHI_SLURM_REMOTE_TMP_DIR` | sbatch 脚本在登录节点的暂存目录（缺省 /tmp） |

SSH 必须免密可用（SDK 以 `BatchMode` 执行，密钥不通会直接失败而不是挂住
等密码）；apiserver 集群条目里的 password 字段仅供 `phi-train init`
首次装公钥用，日常链路走密钥认证。
"""

import os
from pathlib import Path


def load_env() -> None:
    """加载当前目录的 .env；已存在的环境变量不覆盖。"""

    env_file = Path.cwd() / ".env"

    if not env_file.is_file():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():

        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")

        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


def slurm_host() -> str:
    """登录节点地址；未配置抛 KeyError（提交前必须知道连哪）。"""

    return os.environ["PHI_SLURM_HOST"]


def slurm_user() -> str | None:

    return os.environ.get("PHI_SLURM_USER")


def slurm_port() -> int:

    return int(os.environ.get("PHI_SLURM_PORT", "22"))


def slurm_identity_file() -> str | None:

    return os.environ.get("PHI_SLURM_IDENTITY_FILE")


def remote_tmp_dir() -> str:

    return os.environ.get("PHI_SLURM_REMOTE_TMP_DIR", "/tmp")
