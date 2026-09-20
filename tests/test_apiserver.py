"""apiserver 接入单测：不打真 HTTP，用假 client / 假 URL 替换。"""

import pytest

from phi_train_slurm.client import apiserver as facade
from phi_train_slurm.client.api_server_client import (
    ApiServerClient,
    ApiServerUnreachable,
)


class FakeClient:

    def __init__(self, clusters=None, unreachable=False):

        self.clusters = clusters or {}
        self.unreachable = unreachable
        self.queries = []

    def get_cluster(self, name):

        self.queries.append(name)

        if self.unreachable:
            raise ApiServerUnreachable("connection refused")

        if name not in self.clusters:
            raise RuntimeError(f"slurm cluster not found: {name}")

        return self.clusters[name]


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):

    monkeypatch.setattr(facade, "_cluster_cache", {})

    # 每个用例重置后恢复真实 _client，避免用例间泄漏
    yield


# ---- client 层：URL / 鉴权头组装 ----


def test_client_get_url_and_headers(monkeypatch):

    import urllib.request

    captured = {}

    class FakeResponse:

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"found": true}'

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["key"] = request.get_header("X-api-key")
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = ApiServerClient(
        base_url="http://apiserver:8080/", api_key="secret"
    )

    client._get("/conf/query", {"name": "gpu_cluster"})

    assert (
        captured["url"]
        == "http://apiserver:8080/conf/query?name=gpu_cluster"
    )
    assert captured["key"] == "secret"


def test_client_get_cluster_not_found(monkeypatch):

    client = ApiServerClient(base_url="http://x")

    def fake_get(path, params):
        return {"found": False, "conf": None}

    monkeypatch.setattr(client, "_get", fake_get)

    with pytest.raises(RuntimeError):
        client.get_cluster("nope")


def test_client_get_cluster_missing_host(monkeypatch):

    client = ApiServerClient(base_url="http://x")

    def fake_get(path, params):
        return {"found": True, "conf": {"name": "c", "user": "root"}}

    monkeypatch.setattr(client, "_get", fake_get)

    with pytest.raises(RuntimeError):
        client.get_cluster("c")


# ---- 门面层：远端优先 + 兜底 ----


def test_facade_returns_cluster_from_server(monkeypatch):

    fake = FakeClient(
        clusters={
            "gpu_cluster": {
                "name": "gpu_cluster",
                "host": "login01",
                "user": "root",
                "port": "22",
            }
        }
    )

    monkeypatch.setattr(facade, "_client", lambda: fake)

    cluster = facade.get_cluster("gpu_cluster")

    assert cluster["host"] == "login01"
    assert fake.queries == ["gpu_cluster"]


def test_facade_caches_per_name(monkeypatch):

    fake = FakeClient(clusters={"c1": {"name": "c1", "host": "h1"}})

    monkeypatch.setattr(facade, "_client", lambda: fake)

    facade.get_cluster("c1")
    facade.get_cluster("c1")

    assert fake.queries == ["c1"]


def test_facade_falls_back_to_env_on_unreachable(monkeypatch, capsys):

    monkeypatch.setattr(facade, "_client", lambda: FakeClient(unreachable=True))
    monkeypatch.setenv("PHI_SLURM_HOST", "env-login")
    monkeypatch.setenv("PHI_SLURM_USER", "env-user")

    cluster = facade.get_cluster("gpu_cluster")

    assert cluster["host"] == "env-login"
    assert cluster["user"] == "env-user"
    assert "Warning" in capsys.readouterr().out


def test_facade_raises_when_not_registered(monkeypatch):

    monkeypatch.delenv("PHI_SLURM_FALLBACK", raising=False)

    monkeypatch.setattr(facade, "_client", lambda: FakeClient(clusters={}))

    with pytest.raises(RuntimeError):
        facade.get_cluster("unknown_cluster")


def test_facade_fallback_flag_allows_unregistered(monkeypatch, capsys):

    monkeypatch.setenv("PHI_SLURM_FALLBACK", "1")
    monkeypatch.setenv("PHI_SLURM_HOST", "env-login")

    monkeypatch.setattr(facade, "_client", lambda: FakeClient(clusters={}))

    cluster = facade.get_cluster("unregistered")

    assert cluster["host"] == "env-login"
    assert "PHI_SLURM_FALLBACK" in capsys.readouterr().out


def test_set_api_server_overrides(monkeypatch):

    facade.set_api_server(url="http://override:9000", api_key="k1")

    assert facade.APISERVER_URL == "http://override:9000"
    assert facade.APISERVER_API_KEY == "k1"


# ---- client 层：/auth 鉴权 ----


def test_client_check_user_authorized(monkeypatch):

    client = ApiServerClient(base_url="http://x")

    def fake_get(path, params):
        assert path == "/auth"
        assert params == {"username": "alice"}
        return {"authorized": True, "username": "alice"}

    monkeypatch.setattr(client, "_get", fake_get)

    assert client.check_user("alice") is True


def test_client_check_user_not_authorized(monkeypatch):

    client = ApiServerClient(base_url="http://x")

    monkeypatch.setattr(
        client,
        "_get",
        lambda path, params: {"authorized": False, "username": "bob"},
    )

    assert client.check_user("bob") is False


def test_client_check_user_invalid_payload(monkeypatch):

    client = ApiServerClient(base_url="http://x")

    monkeypatch.setattr(client, "_get", lambda path, params: {"weird": 1})

    with pytest.raises(RuntimeError):
        client.check_user("alice")


# ---- 门面层：check_user_authorized ----


class AuthFake:

    def __init__(self, authorized=None, unreachable=False):

        self.authorized = authorized
        self.unreachable = unreachable
        self.queries = []

    def check_user(self, username):

        self.queries.append(username)

        if self.unreachable:
            raise ApiServerUnreachable("connection refused")

        if self.authorized is None:
            raise RuntimeError("invalid auth response")

        return self.authorized


def test_facade_check_user_authorized_ok(monkeypatch):

    fake = AuthFake(authorized=True)

    monkeypatch.setattr(facade, "_client", lambda: fake)

    facade.check_user_authorized("alice")  # 不抛即通过

    assert fake.queries == ["alice"]


def test_facade_check_user_not_authorized(monkeypatch):

    monkeypatch.setattr(facade, "_client", lambda: AuthFake(authorized=False))

    with pytest.raises(ValueError, match="not authorized"):
        facade.check_user_authorized("bob")


def test_facade_check_user_unreachable_fails_closed(monkeypatch):

    monkeypatch.setattr(facade, "_client", lambda: AuthFake(unreachable=True))

    with pytest.raises(ValueError, match="refused"):
        facade.check_user_authorized("alice")
