"""apiserver HTTP 客户端：按集群名从远端配置中心拉取 Slurm 集群配置。

与 phi_data_sdk 共用同一个 apiserver：slurm 集群登记在 apiserver 的
conf.json 里（GET /conf/query?name=<cluster>），与 OSS bucket / CPFS 文件
系统等其他配置同文件、按顶层 key 区分。cluster 字段约定：

    name          集群名（回显）
    host          登录节点地址（必填）
    user          SSH 用户（可选，缺省本机用户）
    port          SSH 端口（可选，缺省 22）
    identity_file SSH 私钥路径（可选，缺省 ~/.ssh 默认）
    password      SSH 密码（可选，仅供 phi-train init 首次装公钥用；
                  SlurmBackend 不读取，日常链路走密钥认证）
    desc          描述（可选）

另提供提交鉴权（与 phi_data_sdk 同协议）：

    GET /auth?username=<username>
    -> {"authorized": bool, "username": str}
"""

import json
import urllib.error
import urllib.parse
import urllib.request

# 登录节点连接字段：host 必填，其余可选
REQUIRED_FIELDS = ("host",)

OPTIONAL_FIELDS = ("user", "port", "identity_file", "password", "desc")


class ApiServerUnreachable(RuntimeError):
    """apiserver 无法连接（网络不通 / 未部署，区别于 404 / 401 等服务端错误）。"""


class ApiServerClient:
    """远端 apiserver 客户端（零依赖，urllib 实现）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 10,
    ):

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _get(
        self,
        path: str,
        query_params: dict,
    ) -> dict:

        query = urllib.parse.urlencode(query_params)

        url = f"{self.base_url}{path}?{query}"

        headers = {}

        if self.api_key:
            headers["X-Api-Key"] = self.api_key

        request = urllib.request.Request(url, headers=headers)

        try:

            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as e:

            if e.code == 401:
                raise RuntimeError(
                    f"apiserver rejected the API key (HTTP 401): "
                    f"{self.base_url}; check APISERVER_API_KEY in .env"
                )

            raise RuntimeError(
                f"apiserver returned HTTP {e.code}: {url}"
            )

        except urllib.error.URLError as e:

            raise ApiServerUnreachable(
                f"cannot reach apiserver at {self.base_url} "
                f"({e.reason}); check APISERVER_URL / network"
            )

    def get_cluster(
        self,
        cluster_name: str,
    ) -> dict:
        """按集群名查配置（GET /conf/query?name=...）；返回配置 dict。

        slurm 集群与 OSS/CPFS 存储登记在同一个 conf.json 数据源里
        （与 phi_data_sdk 的 phi-sync 共用 apiserver），slurm 条目按字段
        区分（有 host / user，无 access_key_id）。
        未登记抛 RuntimeError；连接不上抛 ApiServerUnreachable（上层兜底）。
        """

        payload = self._get("/conf/query", {"name": cluster_name})

        cluster = (
            payload.get("conf") if isinstance(payload, dict) else None
        )

        if (
            not isinstance(payload, dict)
            or not payload.get("found")
            or not isinstance(cluster, dict)
        ):
            raise RuntimeError(
                f"slurm cluster not found: {cluster_name} "
                f"(apiserver: {self.base_url})"
            )

        missing = [
            field
            for field in REQUIRED_FIELDS
            if not cluster.get(field)
        ]

        if missing:
            raise RuntimeError(
                f"invalid cluster config from apiserver, "
                f"missing {', '.join(missing)}: {self.base_url}"
            )

        return cluster

    def check_user(
        self,
        username: str,
    ) -> bool:
        """校验用户是否有提交权限（GET /auth?username=...，与 phi_data_sdk 同协议）。

        返回 True/False；响应格式异常抛 RuntimeError；连接不上抛
        ApiServerUnreachable（由上层处理）。
        """

        payload = self._get("/auth", {"username": username})

        if isinstance(payload, dict) and payload.get("authorized") is True:
            return True

        if isinstance(payload, dict) and payload.get("authorized") is False:
            return False

        raise RuntimeError(
            f"invalid auth response from apiserver: {payload}"
        )
