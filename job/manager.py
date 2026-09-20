import os

from phi_train_slurm.job import registry
from phi_train_slurm.job.handle import JobHandle
from phi_train_slurm.job.job_spec import JobSpec
from phi_train_slurm.job.status import JobStatus
from phi_train_slurm.slurm.launcher import render_sbatch_script


class SlurmJobManager:
    """对上层暴露的作业管理入口：JobSpec -> sbatch 脚本 -> sbatch -> JobHandle。

    与 phi_data_sdk.job.manager 同构：Manager 只做编排与转换，
    真正的集群交互全部委托给 backend（这里是 SlurmBackend）。

    registry_dir：本地作业注册表目录（默认 .phi_jobs/，可用环境变量
    PHI_JOBS_DIR 覆盖）。status/logs/cancel 按原始 job id 经注册表定位
    requeue 后真正在跑的作业；本机没有记录时 status/cancel 退化为直接
    用给定 id 查询，logs 因无法定位日志路径而报错。
    """

    def __init__(
        self,
        backend,
        registry_dir: str | None = None,
    ):

        self.backend = backend
        self.registry_dir = registry_dir or os.environ.get(
            "PHI_JOBS_DIR", registry.DEFAULT_REGISTRY_DIR
        )

    def submit(
        self,
        spec: JobSpec,
    ) -> JobHandle:
        """提交作业。

        流程：
          1. spec.validate()（作业名/资源/命令合法性）
          2. 提交前 sanity check：spec.mounts 的 host 路径在登录节点存在；
             log_dir 不存在则创建（--output 指向的目录缺失会导致提交即失败）
          3. render_sbatch_script(spec) 生成脚本
          4. backend.submit(script) 提交，解析 job id
          5. 注册表落盘（CLI 后续 status/logs/cancel 依赖）
          6. 返回 JobHandle
        """

        spec.validate()

        self._sanity_check(spec)

        job_id = self.backend.submit(render_sbatch_script(spec))

        registry.save(job_id, spec, registry_dir=self.registry_dir)

        return JobHandle(
            self.backend,
            job_id,
            spec,
            registry_dir=self.registry_dir,
        )

    def _sanity_check(self, spec: JobSpec) -> None:

        # CPFS 数据由 phi-sync 预先同步好，host 路径不存在说明配置有误
        for mount in spec.mounts:

            host_path = mount.split(":", 1)[0]

            if not self.backend.path_exists(host_path):
                raise ValueError(
                    f"mount path {host_path!r} not found on login node "
                    f"{self.backend.host}"
                )

        self.backend.mkdir(spec.log_dir)

    def _current_job_id(self, job_id: str) -> str:
        """requeue 后注册表里记录的当前 job id；无记录则原样返回。"""

        record = registry.load(job_id, registry_dir=self.registry_dir)

        return record.current_job_id if record else job_id

    def status(
        self,
        job_id: str,
    ) -> JobStatus:

        return self.backend.status(self._current_job_id(job_id))

    def logs(
        self,
        job_id: str,
        follow: bool = False,
        tail: int = 100,
    ):
        """逐行 yield 作业日志（生成器）；需要本机注册表里的 spec 定位路径。"""

        record = registry.load(job_id, registry_dir=self.registry_dir)

        if record is None:
            raise ValueError(
                f"no local record for job {job_id} under "
                f"{self.registry_dir}: logs need the job's spec "
                "(was it submitted on this machine?)"
            )

        path = (
            f"{record.spec.log_dir}/"
            f"{record.spec.job_name}-{record.current_job_id}.out"
        )

        return self.backend.read_log(path, tail=tail, follow=follow)

    def cancel(
        self,
        job_id: str,
    ) -> None:

        self.backend.cancel(self._current_job_id(job_id))
