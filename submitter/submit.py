"""phi-train CLI 与 SDK 入口函数。

用法：

    phi-train [--cluster <name>] [--server URL] [--api-key KEY] <command>
    phi-train submit exp_vla.yaml --username <name> [--wait]
    phi-train status <job-id>
    phi-train logs <job-id> [--no-follow] [--tail N]
    phi-train cancel <job-id>
    phi-train jobs [--all-users]        # 队列中的作业（排队+运行中）
    phi-train nodes                     # 集群节点及省电状态

连接配置：--cluster 按集群名从 apiserver 拉取（GET /conf/query?name=...，
即 apiserver 的 conf.json，见 client/apiserver.py）；apiserver 不可达时兜底
submitter/config.py 的本地配置（PHI_SLURM_HOST 等环境变量 / .env）。

提交鉴权：submit 需 username（--username 或环境变量 PHI_USERNAME），提交前
经 apiserver GET /auth?username=... 鉴权（见 client/apiserver.py），未授权
或不可达直接拒绝提交（fail closed）。
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

# 日志可能含控制台无法编码的字符，替换为 ? 避免 print 崩溃
for _stream in (sys.stdout, sys.stderr):

    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

if __package__ in (None, ""):

    # 直接以脚本方式运行（python submitter/submit.py）时，把仓库父目录加入
    # sys.path 使 phi_train_slurm 包可导入；要求仓库目录名为 phi_train_slurm，
    # 目录名不同请 pip install -e . 后使用 phi-train
    if PROJECT_ROOT.name != "phi_train_slurm":

        print(
            "Error: repo dir must be named 'phi_train_slurm' to run "
            "submit.py directly; use 'pip install -e .' and "
            "'phi-train' instead"
        )

        sys.exit(1)

    sys.path.insert(
        0,
        str(PROJECT_ROOT.parent)
    )

from phi_train_slurm.job.handle import JobHandle
from phi_train_slurm.job.job_spec import JobSpec
from phi_train_slurm.job.manager import SlurmJobManager
from phi_train_slurm.job.status import JobStatus
from phi_train_slurm.slurm.backend import SlurmBackend
from phi_train_slurm.client import apiserver
from phi_train_slurm.submitter.config import (
    load_env,
    remote_tmp_dir,
    slurm_host,
    slurm_identity_file,
    slurm_port,
    slurm_user,
)


def create_backend(cluster_name: str | None = None) -> SlurmBackend:
    """按集群名组装 SlurmBackend（登录节点连接）。

    连接信息优先从 apiserver 按 cluster_name 拉取（GET /conf/query?name=...，
    即 apiserver 的 conf.json，见 client/apiserver.py）；apiserver 不可达时
    兜底 PHI_SLURM_* 环境变量 / .env（优先级：环境变量 > .env）。
    """

    load_env()

    cluster = apiserver.get_cluster(cluster_name)

    return SlurmBackend(
        host=cluster["host"],
        user=cluster.get("user") or slurm_user(),
        port=int(cluster.get("port") or 0) or slurm_port(),
        identity_file=cluster.get("identity_file") or slurm_identity_file(),
        remote_tmp_dir=remote_tmp_dir(),
    )


# =====================================================
# SDK 入口函数
# =====================================================

# 提交时的用户名来源：--username 参数，其次环境变量 PHI_USERNAME
# （.env 里的 PHI_USERNAME 会由 client/apiserver.py 的 load_env 加载）
USERNAME_ENV_VAR = "PHI_USERNAME"


def resolve_username(username: str | None = None) -> str:
    """参数优先，其次环境变量 PHI_USERNAME；都没有则报错并给出指引。"""

    name = username or os.environ.get(USERNAME_ENV_VAR)

    if not name:
        raise ValueError(
            f"username is required to submit; pass --username or set "
            f"{USERNAME_ENV_VAR} (e.g. export {USERNAME_ENV_VAR}=lushuai)"
        )

    return name


def submit(
    spec: JobSpec,
    wait: bool = True,
    cluster_name: str | None = None,
    username: str | None = None,
) -> JobHandle:
    """提交作业。wait=True 阻塞到结束；False 提交后立即返回 job id（等价不带 --wait）。

    username（参数 > PHI_USERNAME 环境变量）必填，提交前经 apiserver
    /auth 鉴权（check_user_authorized），未授权 / 不可达拒绝提交。
    """

    apiserver.check_user_authorized(resolve_username(username))

    handle = SlurmJobManager(create_backend(cluster_name)).submit(spec)

    if wait:
        handle.final_status = handle.wait()

    return handle


def status(
    job_id: str,
    cluster_name: str | None = None,
) -> JobStatus:

    return SlurmJobManager(create_backend(cluster_name)).status(job_id)


def logs(
    job_id: str,
    follow: bool = True,
    tail: int = 100,
    cluster_name: str | None = None,
):
    """打印作业日志；follow=True 时持续输出直到 Ctrl-C。"""

    stream = SlurmJobManager(create_backend(cluster_name)).logs(
        job_id,
        follow=follow,
        tail=tail,
    )

    for line in stream:
        sys.stdout.write(line)
        sys.stdout.flush()


def cancel(
    job_id: str,
    cluster_name: str | None = None,
):

    return SlurmJobManager(create_backend(cluster_name)).cancel(job_id)


def nodes(cluster_name: str | None = None) -> list[dict]:
    """列出集群节点（含省电关机状态）。"""

    return create_backend(cluster_name).list_nodes()


def jobs(
    user: str | None = None,
    cluster_name: str | None = None,
) -> list[dict]:
    """列出队列中的作业；user=None 只看自己，"all" 看全部。"""

    return create_backend(cluster_name).list_jobs(user=user)


# =====================================================
# CLI
# =====================================================

# --wait 轮询表格：终端态着色（非 tty / NO_COLOR 时输出纯文本）
STATUS_COLORS = {
    "SUCCEEDED": "32",   # green
    "FAILED": "31",      # red
    "CANCELLED": "31",
    "RUNNING": "33",     # yellow
    "PENDING": "33",
    "REQUEUED": "33",
}


def _colorize(text: str, status: str) -> str:

    code = STATUS_COLORS.get(status)

    if code is None or not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text

    return f"\033[{code}m{text}\033[0m"


def _fit(text: str, width: int) -> str:
    """超长截断加 ..，保证表格列对齐。"""

    return text if len(text) <= width else text[: width - 2] + ".."


def _wait_header() -> str:

    parts = [
        "Time".ljust(8),
        "Job".ljust(4),
        "Name".ljust(14),
        "Status".ljust(10),
        "Detail",
    ]

    return (
        "  ".join(parts)
        + "\n"
        + "  ".join("\u2500" * len(part) for part in parts)
    )


def _wait_row(
    job_id: str,
    name: str,
    status: JobStatus,
    detail: str = "",
) -> str:

    cells = (
        time.strftime("%H:%M:%S"),
        str(job_id).ljust(4),
        _fit(name, 14).ljust(14),
        _colorize(status.value.ljust(10), status.value),
        detail,
    )

    return "  ".join(cells)


def _queue_job(backend, job_id: str) -> dict | None:
    """squeue 里按 job id 找作业；不在队列 / 查询失败返回 None（Detail 留空）。"""

    try:
        jobs = backend.list_jobs(user=None)
    except Exception:
        return None

    for job in jobs:

        if job["job_id"] == job_id:
            return job

    return None


def main(
    argv: list[str] | None = None,
):

    parser = argparse.ArgumentParser(
        prog="phi-train",
        description="submit and manage training jobs on EHPC-Slurm",
    )

    parser.add_argument(
        "--cluster",
        help="slurm cluster name registered on the apiserver "
        "(default: env PHI_SLURM_CLUSTER, or 'default')",
    )
    parser.add_argument(
        "--server",
        help="apiserver base url (default: env APISERVER_URL)",
    )
    parser.add_argument(
        "--api-key",
        help="X-Api-Key for the apiserver (default: env APISERVER_API_KEY)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="submit a training job")
    p_submit.add_argument(
        "config",
        help="path to job YAML (JobSpec), e.g. examples/smoke_cpu.yaml",
    )
    p_submit.add_argument(
        "--username",
        help="username for apiserver /auth authorization "
        "(default: env PHI_USERNAME; required to submit)",
    )
    p_submit.add_argument(
        "--wait",
        action="store_true",
        help="block until the job reaches a terminal status "
        "(default: submit only, return job id immediately)",
    )

    p_status = sub.add_parser("status", help="query job status")
    p_status.add_argument("job_id")

    p_logs = sub.add_parser("logs", help="fetch job logs")
    p_logs.add_argument("job_id")
    p_logs.add_argument(
        "--no-follow", action="store_true", help="print once instead of tail -f"
    )
    p_logs.add_argument("--tail", type=int, default=100)

    p_cancel = sub.add_parser("cancel", help="cancel a job")
    p_cancel.add_argument("job_id")

    sub.add_parser("nodes", help="list cluster nodes and their states")

    p_jobs = sub.add_parser(
        "jobs", help="list jobs in the queue (running and pending)"
    )
    p_jobs.add_argument(
        "--all-users",
        action="store_true",
        help="show jobs of all users (default: only yours)",
    )

    sub.add_parser(
        "init",
        help="one-time setup on a new machine: fetch cluster config from "
        "the apiserver and install the ssh public key on the login node",
    )

    args = parser.parse_args(argv)

    apiserver.set_api_server(url=args.server, api_key=args.api_key)

    kwargs = {"cluster_name": args.cluster}

    if args.command == "init":

        from phi_train_slurm.submitter.bootstrap import bootstrap_cluster

        result = bootstrap_cluster(args.cluster)

        if result["installed"]:
            print(
                f"installed {result['key']}.pub on {result['host']} "
                f"(cluster '{result['cluster']}'); passwordless login ready"
            )
        else:
            print(
                f"cluster '{result['cluster']}' ({result['host']}): "
                "passwordless login already works, nothing to do"
            )

    elif args.command == "submit":

        username = resolve_username(args.username)

        apiserver.check_user_authorized(username)

        print(f"username: {username} (authorized)")

        spec = JobSpec.from_yaml(args.config)

        handle = submit(
            spec,
            wait=False,
            username=username,
            **kwargs,
        )

        if not args.wait:
            # 不带 --wait 只输出 job id，便于 shell 直接取用（NFR-5）
            print(handle.job_id)
            return

        print(f"submitted: job {handle.job_id} ({spec.job_name})")
        print(
            f"status: phi-train status {handle.job_id}   "
            f"logs: phi-train logs {handle.job_id} --follow"
        )
        print(
            "waiting for terminal status "
            "(ctrl-c detaches; the job keeps running)"
        )
        print(_wait_header())

        def _show(status: JobStatus) -> None:

            detail = ""

            if status in (JobStatus.PENDING, JobStatus.RUNNING):

                job = _queue_job(handle.backend, handle.job_id)

                if job:

                    if status is JobStatus.RUNNING:
                        nodes = (job.get("nodes") or "").strip()
                        detail = f"on {nodes}" if nodes else ""

                    else:
                        detail = (job.get("reason") or "").strip()

            print(
                _wait_row(handle.job_id, spec.job_name, status, detail),
                flush=True,
            )

        handle.final_status = handle.wait(on_status=_show)

        print(f"job {handle.job_id}: {handle.final_status.value}")

        if handle.final_status is not JobStatus.SUCCEEDED:
            sys.exit(1)

    elif args.command == "status":

        print(status(args.job_id, **kwargs).value)

    elif args.command == "logs":

        logs(
            args.job_id,
            follow=not args.no_follow,
            tail=args.tail,
            **kwargs,
        )

    elif args.command == "cancel":

        cancel(args.job_id, **kwargs)

        print(f"cancelled {args.job_id}")

    elif args.command == "nodes":

        for node in nodes(**kwargs):
            print(
                f"{node['name']:<10} {node['partition']:<8} "
                f"cpus={node['cpus']:<4} mem={node['mem_mb']}MB "
                f"{node['state']}"
            )

    elif args.command == "jobs":

        user = "all" if args.all_users else None

        for job in jobs(user):
            print(
                f"{job['job_id']:<10} {job['name']:<20} {job['user']:<10} "
                f"{job['state']:<10} nodes={job['nodes']:<12} {job['reason']}"
            )


if __name__ == "__main__":
    main()
