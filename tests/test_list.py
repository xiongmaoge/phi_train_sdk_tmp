"""list_nodes / list_jobs 解析单测（假 _run）。"""

from phi_train_slurm.slurm.backend import SlurmBackend


def patch_run(backend, fake):
    backend._run = fake


def test_list_nodes():

    backend = SlurmBackend(host="login01")

    patch_run(
        backend,
        lambda cmd, timeout=60, stdin=None: (
            "comp001|comp*|4|7208|idle~\n"
            "comp002|comp*|4|7208|mixed#\n"
        ),
    )

    nodes = backend.list_nodes()

    assert nodes[0] == {
        "name": "comp001",
        "partition": "comp",
        "cpus": 4,
        "mem_mb": 7208,
        "state": "idle~",
    }
    assert nodes[1]["state"] == "mixed#"


def test_list_jobs():

    backend = SlurmBackend(host="login01")
    captured = {}

    def fake(cmd, timeout=60, stdin=None):
        captured["cmd"] = cmd
        return "1|phi-smoke-cpu|root|RUNNING|(null)|comp001\n"

    patch_run(backend, fake)

    jobs = backend.list_jobs()

    assert jobs == [
        {
            "job_id": "1",
            "name": "phi-smoke-cpu",
            "user": "root",
            "state": "RUNNING",
            "reason": "(null)",
            "nodes": "comp001",
        }
    ]
    assert "-u self" in captured["cmd"]


def test_list_jobs_all_users():

    backend = SlurmBackend(host="login01")
    captured = {}

    def fake(cmd, timeout=60, stdin=None):
        captured["cmd"] = cmd
        return ""

    patch_run(backend, fake)

    backend.list_jobs(user="all")

    assert "-u" not in captured["cmd"]


def test_list_jobs_specific_user():

    backend = SlurmBackend(host="login01")
    captured = {}

    def fake(cmd, timeout=60, stdin=None):
        captured["cmd"] = cmd
        return ""

    patch_run(backend, fake)

    backend.list_jobs(user="alice")

    assert "-u alice" in captured["cmd"]


def test_list_empty_output():

    backend = SlurmBackend(host="login01")
    patch_run(backend, lambda cmd, timeout=60, stdin=None: "\n\n")

    assert backend.list_nodes() == []
    assert backend.list_jobs() == []
