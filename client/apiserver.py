"""apiserver 门面：SlurmBackend 连接配置按集群名从远端拉取。

与 phi_data_sdk 的 client/apiserver.py 同模式：
- 主配置源是远端 apiserver（GET /conf/query?name=<cluster>，即 apiserver
  的 conf.json），按集群名缓存
- 远端不可达时兜底使用环境变量（PHI_SLURM_HOST 等），打印 Warning
- apiserver 上已登记但未授权（401）/ 未登记（found=false）不兜底，直接报错
- 提交鉴权走 GET /auth?username=...（check_user_authorized），fail closed

环境变量：
| 变量 | 说明 |
| --- | --- |
| `APISERVER_URL` | apiserver 地址（默认 http://106.15.232.143:8080） |
| `APISERVER_API_KEY` | X-Api-Key（.env 兜底） |
| `PHI_SLURM_CLUSTER` | 默认集群名（不传 --cluster 时用它，再缺省 "default"） |
| `PHI_SLURM_FALLBACK` | "1" 时集群未登记也退回本地配置（集群尚未在 apiserver 登记的过渡期用；默认关闭） |
"""

import os

from phi_train_slurm.client.api_server_client import (
    ApiServerClient,
    ApiServerUnreachable,
)
from phi_train_slurm.submitter.config import (
    load_env,
    slurm_host,
    slurm_identity_file,
    slurm_port,
    slurm_user,
)

load_env()

APISERVER_URL = os.environ.get(
    "APISERVER_URL",
    "http://106.15.232.143:8080",
)

DEFAULT_APISERVER_API_KEY = "foresee123"
APISERVER_API_KEY = os.environ.get(
    "APISERVER_API_KEY",
    DEFAULT_APISERVER_API_KEY,
)

APISERVER_TIMEOUT_SECONDS = 10

DEFAULT_CLUSTER_NAME = os.environ.get("PHI_SLURM_CLUSTER", "default")

_cluster_cache = {}


def set_api_server(url: str | None = None, api_key: str | None = None) -> None:
    """CLI --server / --api-key 传入时覆盖（优先级高于环境变量）。"""

    global APISERVER_URL, APISERVER_API_KEY

    if url:
        APISERVER_URL = url
        os.environ["APISERVER_URL"] = url

    if api_key:
        APISERVER_API_KEY = api_key
        os.environ["APISERVER_API_KEY"] = api_key


def _client() -> ApiServerClient:

    return ApiServerClient(
        base_url=APISERVER_URL,
        api_key=APISERVER_API_KEY,
        timeout_seconds=APISERVER_TIMEOUT_SECONDS,
    )


def get_cluster(cluster_name: str | None = None) -> dict:
    """查询集群配置（host/user/port/identity_file），按集群名缓存。

    远端不可达时兜底走环境变量（PHI_SLURM_HOST 等）并打印 Warning；
    远端正常但集群未登记 / key 被拒则抛错（不兜底），除非
    PHI_SLURM_FALLBACK=1（集群尚未登记的过渡期开关）。
    """

    name = cluster_name or DEFAULT_CLUSTER_NAME

    if name not in _cluster_cache:

        try:
            _cluster_cache[name] = _client().get_cluster(name)

        except ApiServerUnreachable as error:

            print(
                f"Warning: cannot reach apiserver at {APISERVER_URL} "
                f"({error}); falling back to PHI_SLURM_* env vars"
            )

            _cluster_cache[name] = _env_cluster(name)

        except RuntimeError as error:

            if os.environ.get("PHI_SLURM_FALLBACK") != "1":
                raise

            print(
                f"Warning: {error}; PHI_SLURM_FALLBACK=1, "
                f"falling back to PHI_SLURM_* env vars"
            )

            _cluster_cache[name] = _env_cluster(name)

    return _cluster_cache[name]


def check_user_authorized(username: str) -> None:
    """提交前经 apiserver /auth 鉴权（与 phi_data_sdk 同协议）。

    未授权 / apiserver 不可达 / 响应异常一律抛 ValueError（fail closed）——
    鉴权不过不允许提交，不兜底。
    """

    try:
        authorized = _client().check_user(username)

    except (ApiServerUnreachable, RuntimeError) as error:
        raise ValueError(str(error)) from error

    if not authorized:
        raise ValueError(
            f"user '{username}' is not authorized to submit training "
            "jobs; contact the apiserver admin"
        )


def _env_cluster(name: str) -> dict:

    host = slurm_host()  # 未配置时抛 KeyError，属预期

    return {
        "name": name,
        "host": host,
        "user": slurm_user(),
        "port": slurm_port(),
        "identity_file": slurm_identity_file(),
    }
