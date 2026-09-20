import re
import shlex
import subprocess
import uuid

from phi_train_slurm.job.status import (
    SACCT_STATE_MAP,
    SQUEUE_STATE_MAP,
    JobStatus,
)

# Slurm job id 恒为纯数字；拒绝其他形式，防止拼进远程命令造成注入
_JOB_ID_PATTERN = re.compile(r"^\d+$")


class SlurmError(Exception):
    """Slurm 命令执行失败（非零退出码 / 解析失败）。"""


class SlurmBackend:
    """Slurm 交互的唯一出口：所有命令经登录节点 SSH 执行。

    命令约定（EHPC）：
    - sbatch <script>                       提交，stdout 形如 "Submitted batch job 12345"
    - squeue -j <id> -h -o %T               运行中作业状态（快，仅队列内可见）
    - sacct -j <id> -n -P --format=State,ExitCode   已结束作业状态（含退出码）
    - scancel <id>                          取消
    - scontrol hold/release <id>            排队期占位/放行

    status() 优先级：squeue 命中就用；否则查 sacct；两者都无 → UNKNOWN。
    状态归一映射见 job/status.py（SQUEUE_STATE_MAP / SACCT_STATE_MAP）。

    零依赖约束：不走 paramiko，subprocess 调系统 ssh 命令（与 phi_data_sdk
    「wheel 自包含」原则对齐）。BatchMode=yes：密钥不通时立即失败而不是挂住等密码。
    """

    def __init__(
        self,
        host: str,
        user: str | None = None,
        port: int = 22,
        identity_file: str | None = None,
        ssh_bin: str = "ssh",
        remote_tmp_dir: str = "/tmp",
    ):

        self.host = host
        self.user = user
        self.port = port
        self.identity_file = identity_file
        self.ssh_bin = ssh_bin
        self.remote_tmp_dir = remote_tmp_dir

    # ---- 底层 ----

    def _ssh_argv(self, remote_cmd: str) -> list[str]:

        target = f"{self.user}@{self.host}" if self.user else self.host

        argv = [
            self.ssh_bin,
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
        ]

        if self.identity_file:
            argv += ["-i", self.identity_file]

        return argv + [target, remote_cmd]

    def _exec(
        self,
        remote_cmd: str,
        timeout: float = 60,
        stdin: str | None = None,
    ) -> tuple[int, str, str]:
        """在登录节点执行命令，返回 (returncode, stdout, stderr)；不抛错。"""

        proc = subprocess.run(
            self._ssh_argv(remote_cmd),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return proc.returncode, proc.stdout, proc.stderr

    def _run(
        self,
        remote_cmd: str,
        timeout: float = 60,
        stdin: str | None = None,
    ) -> str:
        """在登录节点执行命令，返回 stdout；非零退出码抛 SlurmError。"""

        rc, out, err = self._exec(remote_cmd, timeout=timeout, stdin=stdin)

        if rc != 0:
            raise SlurmError(
                f"ssh {self.host} {remote_cmd!r} failed (rc={rc}): "
                f"{err.strip()}"
            )

        return out

    @staticmethod
    def _check_job_id(job_id: str) -> None:

        if not _JOB_ID_PATTERN.match(job_id or ""):
            raise ValueError(f"invalid slurm job id {job_id!r}")

    # ---- 作业操作 ----

    def submit(
        self,
        script: str,
    ) -> str:
        """提交 sbatch 脚本，返回 job id。

        脚本经 stdin 写到登录节点 remote_tmp_dir（避免额外 scp），
        `sbatch <path>` 后解析 "Submitted batch job <id>"。
        """

        remote_path = (
            f"{self.remote_tmp_dir}/phi-train-{uuid.uuid4().hex[:8]}.sh"
        )

        out = self._run(
            f"cat > {shlex.quote(remote_path)} "
            f"&& sbatch {shlex.quote(remote_path)}",
            timeout=120,
            stdin=script,
        )

        match = re.search(r"Submitted batch job (\d+)", out)

        if not match:
            raise SlurmError(f"cannot parse sbatch output: {out.strip()!r}")

        return match.group(1)

    def status(
        self,
        job_id: str,
    ) -> JobStatus:
        """查询作业状态：squeue 优先，sacct 兜底（见类 docstring）。"""

        self._check_job_id(job_id)

        state = self._squeue_state(job_id)

        if state:
            return SQUEUE_STATE_MAP.get(state, JobStatus.UNKNOWN)

        state = self._sacct_state(job_id)

        if state:
            return SACCT_STATE_MAP.get(state, JobStatus.UNKNOWN)

        return JobStatus.UNKNOWN

    def _squeue_state(self, job_id: str) -> str:
        """squeue 原始状态；作业不在队列时报错属预期，返回空串转查 sacct。"""

        try:
            return self._run(f"squeue -j {job_id} -h -o %T").strip()
        except SlurmError:
            # "Invalid job id specified" 等：视为无队列记录。
            # 真正的 ssh 连通性错误会在下一步 sacct 上原样抛出。
            return ""

    def _sacct_state(self, job_id: str) -> str:
        """sacct 原始状态；取首个非空行（作业本身，后续行是 job step）。"""

        out = self._run(
            f"sacct -j {job_id} -n -P --format=State,ExitCode"
        )

        for line in out.splitlines():

            if line.strip():
                # "CANCELLED by 12345" 截断成 "CANCELLED"
                return line.split("|")[0].strip().split()[0]

        return ""

    def cancel(
        self,
        job_id: str,
    ) -> None:
        """scancel 取消作业；作业已结束时报错即可（上层按 UNKNOWN 处理）。"""

        self._check_job_id(job_id)

        self._run(f"scancel {job_id}")

    def hold(
        self,
        job_id: str,
    ) -> None:

        self._check_job_id(job_id)

        self._run(f"scontrol hold {job_id}")

    def release(
        self,
        job_id: str,
    ) -> None:

        self._check_job_id(job_id)

        self._run(f"scontrol release {job_id}")

    # ---- 集群 / 队列查询 ----

    def list_nodes(self) -> list[dict]:
        """列出分区全部节点：name / partition / cpus / mem_mb / state。

        state 含省电标记（idle~ 表示已关机待唤醒，idle# 表示正在唤醒），
        serverless 集群空闲节点显示为关机属正常。
        """

        out = self._run('sinfo -h -o "%n|%P|%c|%m|%T"')

        nodes = []

        for line in out.splitlines():

            if not line.strip():
                continue

            name, partition, cpus, mem_mb, state = line.split("|", 4)

            nodes.append(
                {
                    "name": name,
                    "partition": partition.rstrip("*"),
                    "cpus": int(cpus),
                    "mem_mb": int(mem_mb),
                    "state": state,
                }
            )

        return nodes

    def list_jobs(self, user: str | None = None) -> list[dict]:
        """列出队列中（未结束）的作业：job_id / name / user / state / reason / nodes。

        只覆盖排队/运行中的作业；已结束作业走 sacct（暂不提供，CLI status
        按单个 job id 可查）。user=None 时用 -u self 只看自己，"all" 看全部。
        """

        user_flag = "" if user == "all" else f" -u {user or 'self'}"

        out = self._run(
            'squeue -h -o "%i|%j|%u|%T|%R|%N"' + user_flag
        )

        jobs = []

        for line in out.splitlines():

            if not line.strip():
                continue

            job_id, name, job_user, state, reason, nodes = line.split("|", 5)

            jobs.append(
                {
                    "job_id": job_id,
                    "name": name,
                    "user": job_user,
                    "state": state,
                    "reason": reason,
                    "nodes": nodes,
                }
            )

        return jobs

    # ---- 日志 ----

    def read_log(
        self,
        path: str,
        tail: int = 100,
        follow: bool = False,
    ):
        """逐行 yield 登录节点上的日志文件（CPFS 路径，sbatch --output 指定）。

        follow=False: `tail -n <tail>` 读完即止。
        follow=True: `tail -n <tail> -F` 持续输出（-F 在日志轮转后仍能跟上），
        直到生成器被关闭（如 CLI 里 Ctrl-C），关闭时终止 ssh 进程。
        """

        quoted = shlex.quote(path)

        if not follow:
            out = self._run(f"tail -n {tail} {quoted}")
            for line in out.splitlines(keepends=True):
                yield line
            return

        proc = subprocess.Popen(
            self._ssh_argv(f"tail -n {tail} -F {quoted}"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        try:
            for line in proc.stdout:
                yield line
        finally:
            proc.terminate()
            proc.wait()

    # ---- 提交前 sanity check ----

    def path_exists(
        self,
        path: str,
    ) -> bool:
        """检查登录节点上路径是否存在（提交前校验 CPFS 挂载/数据目录）。"""

        rc, _, _ = self._exec(f"test -e {shlex.quote(path)}")

        return rc == 0

    def dir_nonempty(
        self,
        path: str,
    ) -> bool:
        """检查登录节点上目录是否存在且非空（避免训空数据）。"""

        rc, _, _ = self._exec(
            f"test -d {shlex.quote(path)} "
            f'&& test -n "$(ls -A {shlex.quote(path)})"'
        )

        return rc == 0

    def mkdir(
        self,
        path: str,
    ) -> None:
        """在登录节点上创建目录（幂等）；用于确保 log_dir 存在。"""

        self._run(f"mkdir -p {shlex.quote(path)}")
