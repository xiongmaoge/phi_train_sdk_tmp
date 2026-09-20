from enum import Enum

# 归一后的 SDK 作业状态：上层只认这些，不关心 Slurm 原始状态差异


class JobStatus(str, Enum):

    PENDING = "PENDING"

    RUNNING = "RUNNING"

    SUCCEEDED = "SUCCEEDED"

    FAILED = "FAILED"

    CANCELLED = "CANCELLED"

    # 内部容错状态：Slurm 侧 PREEMPTED/REQUEUED 时，JobHandle.wait 会重提交
    REQUEUED = "REQUEUED"

    UNKNOWN = "UNKNOWN"

    def is_terminal(self) -> bool:
        return self in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )


# squeue 输出（%T）→ JobStatus：作业仍在队列/运行中时用
SQUEUE_STATE_MAP = {
    "PENDING": JobStatus.PENDING,
    "CONFIGURING": JobStatus.PENDING,
    "RESV_DEL_HOLD": JobStatus.PENDING,
    "SUSPENDED": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETING": JobStatus.RUNNING,
    "STOPPED": JobStatus.PENDING,
}

# sacct 输出（State 字段）→ JobStatus：作业已结束时用
SACCT_STATE_MAP = {
    "COMPLETED": JobStatus.SUCCEEDED,
    "FAILED": JobStatus.FAILED,
    "NODE_FAIL": JobStatus.FAILED,
    "OUT_OF_MEMORY": JobStatus.FAILED,
    "OOM": JobStatus.FAILED,
    "TIMEOUT": JobStatus.FAILED,
    "CANCELLED": JobStatus.CANCELLED,
    "REVOKED": JobStatus.CANCELLED,
    "PREEMPTED": JobStatus.REQUEUED,
    "REQUEUED": JobStatus.REQUEUED,
}

# requeue 自动重提交时视为「可重试失败」的 sacct 状态：
# 用户主动取消（CANCELLED）不算失败，不重试
RETRYABLE_SACCT_STATES = {
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "OOM",
    "TIMEOUT",
    "PREEMPTED",
    "REQUEUED",
}
