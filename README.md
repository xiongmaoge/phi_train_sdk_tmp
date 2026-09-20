# phi_train_slurm

训练 infra sdk：在 EHPC-Slurm 集群上提交、查询、管理**具身 VLA 模型预训练**作业。

集群连接配置按集群名从 apiserver 拉取（与 phi_data_sdk 共用），新提交机
`phi-train init` 一步接入；已在真实 serverless 集群上验证完整链路。

与 [phi_data_sdk](https://github.com/phi-robotics/phi_data_sdk) 分工：

- **phi_data_sdk（数据侧）**：数据预处理、OSS -> CPFS 数据同步（phi-sync，NAS 数据流动 OpenAPI）
- **phi_train_slurm（训练侧，本仓库）**：把训练任务提交到 Slurm、查询状态、拉日志、取消、故障自动续训

## 核心抽象

```
JobSpec (声明式配置)  →  SlurmBackend (执行器)  →  JobHandle (作业句柄)
  YAML 描述训练           sbatch/squeue/sacct        wait / logs / cancel
  长什么样                唯一碰 Slurm 的地方          操作一个已提交作业
```

```python
from phi_train_slurm import JobSpec, SlurmJobManager, SlurmBackend
from phi_train_slurm.client import apiserver

cluster = apiserver.get_cluster("ehpc_hz")   # apiserver 拉登录节点配置
spec = JobSpec.from_yaml("examples/exp_vla.yaml")
manager = SlurmJobManager(
    SlurmBackend(host=cluster["host"], user=cluster.get("user"))
)

job = manager.submit(spec)     # spec -> sbatch 脚本 -> sbatch -> JobHandle
job.wait(timeout=72 * 3600)    # 阻塞到结束（含故障自动重提）
print(job.status)              # SUCCEEDED / FAILED / ...
```

上层入口 `phi_train_slurm.submitter.submit()`（CLI 同路径）在提交前还会用
username 经 apiserver `/auth` 鉴权（见「CLI」一节）；manager 层不涉及鉴权。

## 整体结构

| 路径 | 说明 |
| --- | --- |
| `__init__.py` / `__main__.py` | 顶层 API 导出（`from phi_train_slurm import ...`）/ `python -m phi_train_slurm` 入口 |
| `model/resource.py` | `ResourceSpec`：Slurm 资源申请（partition / qos / nodes / gpus / mem / time_limit...） |
| `job/job_spec.py` | `JobSpec` + `RequeueConfig`：作业声明式配置 + YAML 解析 + 校验（含 sbatch 注入防护）+ 序列化 |
| `job/status.py` | `JobStatus` 状态枚举 + `squeue`/`sacct` 原始状态 → SDK 状态归一映射 + 可重试状态集合 |
| `job/manager.py` | `SlurmJobManager`：submit（校验 + sanity check）/ status / logs / cancel |
| `job/handle.py` | `JobHandle`：wait（轮询 + requeue 自动续训）/ logs / cancel |
| `job/registry.py` | 本地作业注册表（`.phi_jobs/<job_id>.json`）：job id → spec / 当前 attempt 映射 |
| `slurm/backend.py` | `SlurmBackend`：登录节点 SSH，封装 sbatch / squeue / sacct / scancel / scontrol hold·release，以及日志读取与路径检查 |
| `slurm/launcher.py` | srun + torchrun / deepspeed / plain 多机启动命令生成 + sbatch 脚本渲染 |
| `slurm/templates/sbatch.sh.tpl` | sbatch 脚本体模板（占位符替换，`$SLURM_*` 变量保留不展开） |
| `client/api_server_client.py` | apiserver HTTP 客户端（`GET /conf/query?name=`、`GET /auth?username=`，X-Api-Key 鉴权，零依赖） |
| `client/apiserver.py` | 门面：集群配置远端优先（按集群名缓存，不可达兜底本地环境变量）+ 提交鉴权 check_user_authorized（fail closed） |
| `submitter/config.py` | 本地兜底配置（`PHI_SLURM_HOST` 等环境变量优先，当前目录 `.env` 兜底） |
| `submitter/bootstrap.py` | `phi-train init`：新机器从 apiserver 拉配置 + 自动装 SSH 公钥 |
| `submitter/submit.py` | `phi-train` CLI 与 SDK 入口函数（init / submit / status / logs / cancel / jobs / nodes） |
| `examples/exp_vla.yaml` | VLA 预训练作业配置示例（8 节点 x 8 卡，容器模式） |
| `examples/smoke_cpu.yaml` | 链路冒烟作业示例（1 节点 CPU、不占 GPU，几十秒跑完） |
| `docs/系统设计文档.md` | 系统设计文档（数据模型、模块设计、关键流程、部署架构） |
| `docs/后续改进计划.md` | 后续改进清单（安全 / 配置一致性 / 功能缺口 / 成本运维 / 文档） |
| `tests/` | 离线单测（不连集群、不联网；backend 用假 SSH 替身） |

## 作业配置示例

完整示例见 `examples/exp_vla.yaml`（注意 `resource` 是嵌套结构）：

```yaml
job_name: vla-pretrain-pi0-v1

resource:
  partition: gpu
  qos: normal
  nodes: 8                      # 8 节点 x 8 卡 = 64 GPU
  gpus_per_node: 8
  cpus_per_task: 16
  mem_per_node_gb: null         # null = 不限制（用满节点内存）
  time_limit: "72:00:00"

image: registry-vpc.cn-hangzhou.aliyuncs.com/vla/train:0.1   # pyxis/enroot 容器；省略则用宿主机环境
mounts:                         # host:container（数据由 phi-sync 同步好）
  - /mnt/cpfs/vla_dataset_v1:/data
  - /mnt/cpfs/checkpoints:/ckpt
  - /mnt/cpfs/logs:/logs

env:
  WANDB_API_KEY: ${WANDB_API_KEY}   # 提交时从本地环境展开，不落明文

launch_backend: torchrun        # torchrun | deepspeed | plain（见 examples/smoke_cpu.yaml）

command: |
  train.py --config configs/pi0_v1.yaml --data /data --ckpt /ckpt/vla-pi0-v1

log_dir: /mnt/cpfs/logs         # sbatch --output/--error 目录，登录节点可读

requeue:
  enabled: true                 # 故障自动重提交（Slurm 不原生做，SDK 做）
  max_retries: 3
  resume_from: /ckpt/vla-pi0-v1 # 重提时自动给 command 追加 --resume <dir>
```

## CLI

所有命令都支持全局参数 `--cluster <name>`（缺省 `default`）、`--server`、
`--api-key`（覆盖 apiserver 地址与密钥）：

```bash
phi-train init [--cluster <name>]               # 新机器一次性初始化（见下节）
phi-train submit examples/exp_vla.yaml --username <name> --wait   # 提交并等待结束（不带 --wait 只拿 job id）
phi-train status <job-id>                            # 查状态（requeue 后自动跟到最新 attempt）
phi-train logs <job-id> --follow                     # 追日志（--no-follow 只打最近 --tail 行）
phi-train cancel <job-id>                            # 取消
phi-train jobs [--all-users]                         # 队列中的作业（排队 + 运行中）
phi-train nodes                                      # 集群节点及省电状态
```

提交必须带用户名：`--username` 或环境变量 `PHI_USERNAME`（.env 里的
`PHI_USERNAME` 会自动加载），提交前经 apiserver `GET /auth?username=...`
鉴权（与 phi_data_sdk 同协议）；未授权 / apiserver 不可达直接拒绝提交，
不兜底。授权名单维护在 apiserver 侧。

真实集群（`ehpc_hz`，serverless）上的实测输出示例：

```
$ phi-train nodes
comp001    comp     cpus=4    mem=7208MB idle~     # ~ = 省电关机，有作业自动唤醒
...

$ phi-train submit examples/smoke_cpu.yaml --username lushuai --wait
username: lushuai (authorized)
submitted: job 2 (phi-smoke-cpu)
status: phi-train status 2   logs: phi-train logs 2 --follow
waiting for terminal status (ctrl-c detaches; the job keeps running)
Time      Job  Name            Status       Detail
────────  ───  ──────────────  ──────────   ──────
15:02:11  2    phi-smoke-cpu   PENDING      comp001
15:02:41  2    phi-smoke-cpu   RUNNING      on comp001
15:03:11  2    phi-smoke-cpu   SUCCEEDED
job 2: SUCCEEDED

$ phi-train status 2 && phi-train logs 2 --no-follow
SUCCEEDED
0: host: comp001
cpus: 4
DONE
```

不带 `--wait` 时 submit 只输出 job id（便于 shell 取用）；`--wait` 输出如上
（终端下状态带颜色，`NO_COLOR=1` 或重定向时纯文本），作业非 SUCCEEDED
结束时退出码为 1。

`nodes` 的 state 带 Slurm 省电标记：`idle~` = 已关机待唤醒（serverless 集群
空闲节点显示为关机属正常，有作业时自动开机，实测唤醒约 1-5 分钟），
`idle#` = 正在唤醒。

等价入口：`python -m phi_train_slurm ...`；Python 侧可直接调
`phi_train_slurm.submitter` 里的 init/submit/status/logs/cancel/jobs/nodes
函数。首次接入建议先跑冒烟作业验证全链路（1 节点 CPU，几十秒结束）：
`phi-train submit examples/smoke_cpu.yaml --username lushuai --wait`。

## 新机器接入（phi-train init）

每台新提交机的完整接入只需三步，无需问任何人要密码：

```bash
pip install -e .                                  # 或离线装 wheel
export APISERVER_API_KEY=<key>                    # 唯一需要的本地配置
phi-train init --cluster ehpc_hz                  # 一次性初始化
```

`init` 做的事：

1. 从 apiserver 拉集群配置（含 password 字段）；
2. 本机没有 SSH 密钥则自动生成 ed25519；
3. 已能免密登录则直接完成；否则用 password 自动 `ssh-copy-id`（优先
   SSH_ASKPASS（OpenSSH ≥ 8.4），回退 expect）；
4. BatchMode 复验免密。

密码只在那一次装公钥的内存里使用，不落任何本地文件；之后所有链路走密钥
认证，`SlurmBackend` 不读 password。init 之后所有命令只需
`APISERVER_API_KEY`，集群连接信息全部来自 apiserver（换登录节点 / 加集群
只改 apiserver，提交机零改动）。

## 集群配置：apiserver 优先

登录节点连接信息与 phi_data_sdk（phi-sync 的 OSS/CPFS storage 配置）共用
同一个 apiserver 数据源：按集群名 `GET /conf/query?name=<cluster>`
（请求头 `X-Api-Key`）拉取 apiserver 的 `conf.json`。slurm 集群和
OSS bucket / CPFS 文件系统登记在同一个 conf.json 里，按顶层 key 区分
（slurm 条目有 `host`/`user`，无 `access_key_id`）。当前已登记
`ehpc_hz`（以及别名 `default`）。字段约定：

| 字段 | 说明 |
| --- | --- |
| `host` | 登录节点地址（必填） |
| `user` | SSH 用户（可选，缺省用 `PHI_SLURM_USER` / 本机用户） |
| `port` | SSH 端口（可选，缺省 22） |
| `identity_file` | SSH 私钥路径（可选，缺省 ~/.ssh 默认） |
| `password` | SSH 密码（可选，仅供 `phi-train init` 首次装公钥；日常链路走密钥不读取） |
| `desc` | 描述（可选） |

调用链：CLI `--cluster <name>`（缺省环境变量 `PHI_SLURM_CLUSTER`，再缺省
`default`）→ `client/apiserver.py` 门面（按集群名缓存）→
`client/api_server_client.py` HTTP 客户端。

**降级语义**（对齐 phi_data_sdk）：apiserver 不可达（网络不通/未部署）时打
Warning 并兜底用本地配置（环境变量 > 当前目录 `.env`）；apiserver
在线但**未登记该集群**或 **key 被拒（401）**则直接报错，不兜底——避免
「以为在用集群 A，实际连了默认机器」。过渡期可用 `PHI_SLURM_FALLBACK=1`
显式放开：集群尚未在 apiserver 登记时也退回本地配置（打 Warning）。

apiserver 相关环境变量：`APISERVER_URL`（默认 `http://106.15.232.143:8080`）、
`APISERVER_API_KEY`（可放当前目录 `.env`）、`PHI_USERNAME`（提交用户名，
等价 submit 的 `--username`）；CLI 可用 `--server` / `--api-key` 覆盖。

本地兜底配置（apiserver 不可达时使用，优先级：环境变量 > 当前目录
`.env`）：

| 环境变量 | 说明 |
| --- | --- |
| `PHI_SLURM_HOST` | 登录节点地址（必填） |
| `PHI_SLURM_USER` | SSH 用户（缺省用本机用户） |
| `PHI_SLURM_IDENTITY_FILE` | SSH 私钥路径（缺省用 ~/.ssh 默认） |
| `PHI_SLURM_PORT` | SSH 端口（缺省 22） |
| `PHI_SLURM_REMOTE_TMP_DIR` | sbatch 脚本在登录节点的暂存目录（缺省 /tmp） |

SSH 必须免密可用（SDK 以 `BatchMode` 执行，密钥不通会直接失败而不是挂住等密码）。

## 设计要点

1. **声明式 JobSpec**：纯数据、可序列化（`to_dict`/`from_dict`），团队共享训练模板，也是 requeue 重提交的依据。
2. **状态归一**：作业在队列时查 `squeue`，已结束查 `sacct`，统一映射为 `JobStatus`（`PENDING/RUNNING/SUCCEEDED/FAILED/CANCELLED/REQUEUED/UNKNOWN`），上层不用关心 Slurm 状态差异。
3. **gang scheduling 交给 Slurm**：`--nodes=N` 由 Slurm 保证要么全起要么不起；每节点 1 个 srun 任务，内部由 `launch_backend` 拉起 GPU 进程（`--ntasks-per-node=1 + srun torchrun|deepspeed`，`plain` 则直接跑命令）；`gpus_per_node: 0` 时不申请 GPU（CPU 作业，见 `examples/smoke_cpu.yaml`）。
4. **日志落 CPFS**：sbatch 的 `--output/--error` 指向 CPFS 日志目录（`<log_dir>/<job_name>-<job_id>.out/.err`），登录节点即可读，SDK 拉日志无需进计算节点。`log_dir` 缺失时提交前自动创建。
5. **故障自动续训（requeue）**：`wait()` 发现可重试失败（`RETRYABLE_SACCT_STATES`，见 `job/status.py`）时，按 `requeue.max_retries` 自动用同一 JobSpec 重提交；`resume_from` 设置且命令里没有 `--resume` 时自动追加。训练代码契约：`--resume <dir>` 存在则从最近 checkpoint 续训。重提交沿用同一 job_name，日志文件名带 job id 不会互相覆盖。
6. **本地作业注册表**：提交 / 重提交时把 `job_id -> {spec, current_job_id, attempt}` 落盘到 `.phi_jobs/`（可用 `PHI_JOBS_DIR` 覆盖）。CLI 按原始 job id 的 status/logs/cancel 自动换算到 requeue 后真正在跑的作业；`logs` 依赖注册表定位日志路径，换机器执行会明确报错。
7. **零运行时依赖**：核心链路只用标准库，SSH 走系统 `ssh` 命令（subprocess，脚本经 stdin 传到登录节点）；`pyyaml` 为可选 extra（`pip install "phi-train-slurm[yaml]"`）。对齐 phi_data_sdk 的「wheel 自包含、离线可装」原则。
8. **提交前 sanity check + 注入防护**：`submit()` 前校验挂载路径在登录节点存在；`JobSpec.validate()` 拒绝 partition/log_dir/mounts/env key 等拼进 sbatch 脚本的字段含空白、引号等危险字符，job id 非纯数字直接拒绝。
9. **配置中心化（apiserver）**：集群连接信息（host/user/port/password）按集群名从 apiserver `/conf/query`（conf.json）拉取，与 phi_data_sdk 共用一套配置服务；提交机本地只需 `APISERVER_API_KEY`，`phi-train init` 用托管密码自动完成首次公钥安装，之后永久免密。远端不可达才兜底本地配置，在线但未登记则明确报错（不静默换机器）。
10. **提交鉴权（fail closed）**：submit 必须带 username（`--username` > `PHI_USERNAME`），提交前经 apiserver `GET /auth?username=...` 校验（与 phi_data_sdk 同协议，授权名单在 apiserver 侧）；未授权 / apiserver 不可达直接拒绝提交，不兜底。requeue 重提沿用已授权过的同一 JobSpec，不重复鉴权。

## 安装与开发

```bash
# 仓库内开发
pip install -e ".[dev,yaml]"

# 跑测试（全部离线）
pytest
```

打包分发：`python -m pip wheel . --no-deps -w dist`，产物为纯 Python wheel。

## 已知边界

- 已在真实 EHPC serverless 集群（slurm 22.05.8，`ehpc_hz`）上验证过完整链路：
  submit → 节点自动扩容唤醒 → RUNNING → SUCCEEDED → sacct 终态 → CPFS 拉日志，
  以及 `phi-train init` / `nodes` / `jobs` / apiserver 配置拉取。GPU 队列、
  pyxis 容器指令（`--container-image`）、torchrun 多机启动尚未在真实 GPU 集群
  上跑过，接入 GPU 队列时可能需要微调。
- serverless 集群空闲节点自动关机（`idle~`），作业提交后首次排队需等节点
  唤醒，实测约 1-5 分钟，属正常现象。
- requeue 对**任何** FAILED（含训练代码自身的确定性错误）都会重试到
  `max_retries` 次；确认是代码问题请 `requeue.enabled: false` 或先 `cancel`。
  用户主动取消（CANCELLED）不触发重试。
- `wait()` 的重提交只发生在调用方进程活着的时候；进程挂掉后需要用原 spec
  手动重新 submit（checkpoint 续训契约仍在）。
- `scontrol hold/release`（排队期占位/放行）已在 `SlurmBackend` 实现，但
  manager / handle / CLI 尚未透出，暂只能直接调 backend。
