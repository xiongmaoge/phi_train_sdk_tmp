from dataclasses import dataclass


@dataclass
class ResourceSpec:
    """作业在 Slurm 集群上的资源申请，字段与 sbatch 指令一一对应。

    - partition / qos: EHPC 侧已创建好的分区与 QOS
    - nodes: 节点数；多机训练时由 Slurm 保证 gang scheduling（全起或不起）
    - gpus_per_node: 每节点 GPU 数，映射到 `--gres=gpu:N`
    - cpus_per_task: 每 srun 任务的 CPU 核数
    - mem_per_node_gb: 每节点内存上限（GB），None 表示不限制
    - time_limit: 最长运行时间（DD-HH:MM:SS 或 HH:MM:SS），超时 Slurm 终止作业
    """

    partition: str = "gpu"

    qos: str | None = None

    nodes: int = 1

    gpus_per_node: int = 8

    cpus_per_task: int = 16

    mem_per_node_gb: int | None = None

    time_limit: str = "72:00:00"

    def to_dict(self) -> dict:
        """序列化，用于 requeue 重提交与作业元数据落盘。"""

        return {
            "partition": self.partition,
            "qos": self.qos,
            "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node,
            "cpus_per_task": self.cpus_per_task,
            "mem_per_node_gb": self.mem_per_node_gb,
            "time_limit": self.time_limit,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ResourceSpec":
        return cls(
            partition=payload.get("partition", "gpu"),
            qos=payload.get("qos"),
            nodes=payload.get("nodes", 1),
            gpus_per_node=payload.get("gpus_per_node", 8),
            cpus_per_task=payload.get("cpus_per_task", 16),
            mem_per_node_gb=payload.get("mem_per_node_gb"),
            time_limit=payload.get("time_limit", "72:00:00"),
        )
