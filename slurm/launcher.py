import os
from pathlib import Path
from string import Template

from phi_train_slurm.job.job_spec import JobSpec

TEMPLATE_NAME = "sbatch.sh.tpl"

# torchrun rendezvous 端口（模板与启动命令共用）
MASTER_PORT = "29500"


def build_launch_command(
    spec: JobSpec,
) -> str:
    """把 spec.command 包成 srun + <launch_backend> 的多机启动命令。

    标准 GPU 训练模式：sbatch 侧 --ntasks-per-node=1（每节点 1 个 srun 任务），
    由 torchrun / deepspeed 在节点内再拉起 gpus_per_node 个进程。
    """

    if spec.launch_backend == "torchrun":

        return (
            "srun --label torchrun "
            "--nnodes=$SLURM_NNODES "
            f"--nproc_per_node={spec.resource.gpus_per_node} "
            f"--rdzv_endpoint=$MASTER_ADDR:{MASTER_PORT} "
            f"{spec.command.strip()}"
        )

    if spec.launch_backend == "deepspeed":

        return (
            "srun --label deepspeed "
            "--num_nodes=$SLURM_NNODES "
            f"--num_gpus={spec.resource.gpus_per_node} "
            f"{spec.command.strip()}"
        )

    if spec.launch_backend == "plain":

        return f"srun --label {spec.command.strip()}"

    raise ValueError(
        f"unsupported launch_backend {spec.launch_backend!r}"
    )


def _build_header(
    spec: JobSpec,
) -> str:
    """sbatch 指令头：有条件的指令（qos/mem/容器）在代码里拼，模板只放脚本体。"""

    r = spec.resource

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec.job_name}",
        f"#SBATCH --partition={r.partition}",
    ]

    if r.qos:
        lines.append(f"#SBATCH --qos={r.qos}")

    lines += [
        f"#SBATCH --nodes={r.nodes}",
        "#SBATCH --ntasks-per-node=1",
    ]

    # gpus_per_node=0：CPU 作业，不申请 GPU
    if r.gpus_per_node > 0:
        lines.append(f"#SBATCH --gres=gpu:{r.gpus_per_node}")

    lines += [
        f"#SBATCH --cpus-per-task={r.cpus_per_task}",
        f"#SBATCH --time={r.time_limit}",
        # 日志落 CPFS：登录节点可读，SDK 拉日志不用进计算节点
        f"#SBATCH --output={spec.log_dir}/{spec.job_name}-%j.out",
        f"#SBATCH --error={spec.log_dir}/{spec.job_name}-%j.err",
    ]

    if r.mem_per_node_gb:
        lines.append(f"#SBATCH --mem={r.mem_per_node_gb * 1024}")

    if spec.image:
        # 容器模式（需 EHPC 已装 pyxis/enroot）
        lines.append(f"#SBATCH --container-image={spec.image}")
        if spec.mounts:
            lines.append(
                f"#SBATCH --container-mounts={','.join(spec.mounts)}"
            )

    return "\n".join(lines) + "\n"


def _expand_env(
    env: dict[str, str],
) -> str:
    """生成 export 行；值含 ${VAR} 时从提交者本地环境展开（缺失抛 KeyError），
    避免把密钥写进 YAML。"""

    lines = []

    for key, value in env.items():

        if value is not None and "${" in value:
            value = Template(value).substitute(os.environ)

        lines.append(f'export {key}="{value}"')

    return "\n".join(lines) if lines else "# (no extra env)"


def _load_template() -> str:
    """优先从安装包读模板；仓库内直接跑时回退到源码目录。"""

    try:

        from importlib import resources

        return (
            resources.files("phi_train_slurm.slurm")
            .joinpath("templates", TEMPLATE_NAME)
            .read_text(encoding="utf-8")
        )

    except (ImportError, ModuleNotFoundError, FileNotFoundError):

        path = (
            Path(__file__).resolve().parent / "templates" / TEMPLATE_NAME
        )

        return path.read_text(encoding="utf-8")


def render_sbatch_script(
    spec: JobSpec,
) -> str:
    """JobSpec → 完整 sbatch 脚本文本（纯字符串渲染，离线可测试）。

    用占位符替换而非 string.Template.substitute：
    脚本体里含 $SLURM_NNODES 等 shell 变量，不能被当占位符展开。
    """

    spec.validate()

    body = _load_template()

    body = body.replace("$env_exports", _expand_env(spec.env))
    body = body.replace("$master_port", MASTER_PORT)
    body = body.replace("$command", build_launch_command(spec))

    return _build_header(spec) + "\n" + body
