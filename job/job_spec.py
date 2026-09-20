import re
from dataclasses import dataclass, field
from pathlib import Path

from phi_train_slurm.model.resource import ResourceSpec

# Slurm 作业名仅允许字母、数字、下划线、连字符
JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class RequeueConfig:
    """故障自动重提交配置（Slurm 不原生做自动续训，由 SDK 在 JobHandle.wait 里实现）。

    resume_from 是训练代码与 SDK 之间的契约：训练脚本读该路径，
    存在 checkpoint 则从最近一步续训，否则从头训。
    """

    enabled: bool = True

    max_retries: int = 3

    resume_from: str | None = None

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "max_retries": self.max_retries,
            "resume_from": self.resume_from,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RequeueConfig":
        return cls(
            enabled=payload.get("enabled", True),
            max_retries=payload.get("max_retries", 3),
            resume_from=payload.get("resume_from"),
        )


@dataclass
class JobSpec:
    """训练作业的声明式配置：纯数据、可序列化，团队共享训练模板。

    由 SlurmJobManager.submit 渲染成 sbatch 脚本提交；
    requeue 重提交时用同一个 spec（保证一致性）。
    """

    job_name: str

    resource: ResourceSpec

    command: str

    # pyxis/enroot 容器镜像（None 表示直接用宿主机 Python 环境，需 EHPC 已装 pyxis）
    image: str | None = None

    # "host:container" 列表；host 侧路径必须在集群（CPFS）上存在
    mounts: list[str] = field(default_factory=list)

    # 注入作业的环境变量；值含 ${VAR} 时从提交者本地环境展开，避免密钥进 YAML
    env: dict[str, str] = field(default_factory=dict)

    # 多机启动器：torchrun（默认）| deepspeed
    launch_backend: str = "torchrun"

    # 日志目录（CPFS 路径）：sbatch --output/--error 指向这里，
    # 登录节点即可读，SDK 拉日志无需 SSH 进计算节点
    log_dir: str = "/mnt/cpfs/logs"

    requeue: RequeueConfig = field(default_factory=RequeueConfig)

    def validate(self) -> None:
        """提交前校验，不合法抛 ValueError。"""

        if not self.job_name or not JOB_NAME_PATTERN.match(self.job_name):
            raise ValueError(
                f"invalid job_name {self.job_name!r}: "
                "only letters, digits, underscore and hyphen allowed"
            )

        if self.resource.nodes < 1 or self.resource.gpus_per_node < 0:
            raise ValueError("nodes must be >= 1, gpus_per_node must be >= 0")

        if not self.command.strip():
            raise ValueError("command must not be empty")

        if self.launch_backend not in ("torchrun", "deepspeed", "plain"):
            raise ValueError(
                f"unsupported launch_backend {self.launch_backend!r}: "
                "only torchrun / deepspeed / plain"
            )

        # sbatch 头 / 模板注入防护：拼进脚本的字段不允许空白与引号
        for name, value in (
            ("partition", self.resource.partition),
            ("qos", self.resource.qos),
            ("time_limit", self.resource.time_limit),
            ("log_dir", self.log_dir),
            ("image", self.image),
        ):
            if value and re.search(r"[\s\"']", value):
                raise ValueError(
                    f"{name} must not contain whitespace or quotes: "
                    f"{value!r}"
                )

        for mount in self.mounts:
            if re.search(r"[\s,]", mount):
                raise ValueError(
                    f"invalid mount {mount!r}: no whitespace or commas"
                )

        for key in self.env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key or ""):
                raise ValueError(f"invalid env key {key!r}")

    def to_dict(self) -> dict:
        return {
            "job_name": self.job_name,
            "resource": self.resource.to_dict(),
            "command": self.command,
            "image": self.image,
            "mounts": list(self.mounts),
            "env": dict(self.env),
            "launch_backend": self.launch_backend,
            "log_dir": self.log_dir,
            "requeue": self.requeue.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "JobSpec":
        return cls(
            job_name=payload["job_name"],
            resource=ResourceSpec.from_dict(payload.get("resource", {})),
            command=payload["command"],
            image=payload.get("image"),
            mounts=payload.get("mounts", []),
            env=payload.get("env", {}),
            launch_backend=payload.get("launch_backend", "torchrun"),
            log_dir=payload.get("log_dir", "/mnt/cpfs/logs"),
            requeue=RequeueConfig.from_dict(payload.get("requeue", {})),
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        defaults: dict | None = None,
    ) -> "JobSpec":
        """从 YAML 文件加载（可选依赖：pip install "phi-train-slurm[yaml]"）。

        defaults 用于回填 YAML 未显式写出的 partition / log_dir
        （CLI 从 apiserver 的集群配置 defaults 段取，见 submitter/submit.py）；
        回填优先级：YAML 显式值 > defaults > 代码默认值。
        """

        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for YAML configs: "
                'pip install "phi-train-slurm[yaml]"'
            ) from exc

        with open(path, encoding="utf-8") as f:
            payload = yaml.safe_load(f)

        cls._backfill_defaults(payload, defaults or {})

        spec = cls.from_dict(payload)
        spec.validate()
        return spec

    @staticmethod
    def _backfill_defaults(payload: dict, defaults: dict) -> None:
        """把 defaults 里缺失的字段回填进 payload（显式值优先，就地修改）。"""

        if defaults.get("partition"):
            resource = payload.get("resource")
            if resource is None:
                resource = {}
                payload["resource"] = resource
            if isinstance(resource, dict) and not resource.get("partition"):
                resource["partition"] = defaults["partition"]

        if defaults.get("log_dir") and not payload.get("log_dir"):
            payload["log_dir"] = defaults["log_dir"]
