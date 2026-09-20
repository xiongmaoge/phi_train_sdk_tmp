"""本地作业注册表：把 job_id -> JobSpec 的映射落盘。

CLI 的 `phi-train status/logs/cancel <job-id>` 只拿 job id，而拉日志需要
spec（log_dir / job_name），requeue 重提交后原始 job id 也会过期（真正在跑的
是新的 job id）。因此在提交 / 重提交时把元数据写到 `.phi_jobs/<job_id>.json`，
后续按原始 job id 查询时先经注册表换算成「当前 job id + spec」。

只在本机生效：换机器执行 CLI 时没有记录，logs 会明确报错而不是猜路径。
"""

import json
from dataclasses import dataclass
from pathlib import Path

from phi_train_slurm.job.job_spec import JobSpec

DEFAULT_REGISTRY_DIR = ".phi_jobs"


@dataclass
class JobRecord:
    """注册表中的一条作业记录。"""

    job_id: str

    # requeue 重提交后正在跑的 job id（未重提交时等于 job_id）
    current_job_id: str

    # 第几次尝试（首次提交为 1）
    attempt: int

    spec: JobSpec


def _record_path(job_id: str, registry_dir: str) -> Path:

    return Path(registry_dir) / f"{job_id}.json"


def save(
    job_id: str,
    spec: JobSpec,
    current_job_id: str | None = None,
    attempt: int = 1,
    registry_dir: str = DEFAULT_REGISTRY_DIR,
) -> None:
    """提交 / requeue 后写入（覆盖）作业记录。"""

    path = _record_path(job_id, registry_dir)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "job_id": job_id,
        "current_job_id": current_job_id or job_id,
        "attempt": attempt,
        "spec": spec.to_dict(),
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load(
    job_id: str,
    registry_dir: str = DEFAULT_REGISTRY_DIR,
) -> JobRecord | None:
    """按原始 job id 读取记录；无记录返回 None。"""

    path = _record_path(job_id, registry_dir)

    if not path.is_file():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))

    return JobRecord(
        job_id=payload["job_id"],
        current_job_id=payload.get("current_job_id", payload["job_id"]),
        attempt=payload.get("attempt", 1),
        spec=JobSpec.from_dict(payload["spec"]),
    )
