"""离线单测：不连集群、不联网，只覆盖纯数据 / 纯字符串渲染部分。

SlurmBackend 的 SSH 交互为 TODO，不在本文件覆盖（实现后补 test_backend.py，
参考 phi_data_sdk 用假 client 替换集群交互的做法）。
"""

import os

import pytest

from phi_train_slurm.job.job_spec import JobSpec
from phi_train_slurm.job.status import (
    RETRYABLE_SACCT_STATES,
    SACCT_STATE_MAP,
    SQUEUE_STATE_MAP,
    JobStatus,
)
from phi_train_slurm.slurm.launcher import render_sbatch_script


def make_spec(**overrides) -> JobSpec:

    payload = {
        "job_name": "vla_test_job",
        "resource": {
            "partition": "gpu",
            "nodes": 8,
            "gpus_per_node": 8,
            "cpus_per_task": 16,
            "time_limit": "72:00:00",
        },
        "command": "train.py --data /data",
        "launch_backend": "torchrun",
    }

    payload.update(overrides)

    return JobSpec.from_dict(payload)


# ---- JobSpec ----


def test_spec_roundtrip():

    spec = make_spec()

    assert JobSpec.from_dict(spec.to_dict()) == spec


def test_validate_rejects_bad_job_name():

    with pytest.raises(ValueError):
        make_spec(job_name="bad-name!").validate()


def test_validate_rejects_empty_command():

    with pytest.raises(ValueError):
        make_spec(command="   ").validate()


def test_validate_rejects_header_injection():

    with pytest.raises(ValueError):
        make_spec(resource={"partition": "gpu\n# injected"}).validate()

    with pytest.raises(ValueError):
        make_spec(log_dir="/logs # x").validate()


def test_validate_rejects_bad_env_key():

    with pytest.raises(ValueError):
        make_spec(env={"BAD KEY": "1"}).validate()


# ---- 状态归一 ----


def test_status_maps():

    assert SQUEUE_STATE_MAP["RUNNING"] is JobStatus.RUNNING
    assert SQUEUE_STATE_MAP["PENDING"] is JobStatus.PENDING
    assert SACCT_STATE_MAP["COMPLETED"] is JobStatus.SUCCEEDED
    assert SACCT_STATE_MAP["NODE_FAIL"] is JobStatus.FAILED
    assert SACCT_STATE_MAP["CANCELLED"] is JobStatus.CANCELLED
    assert SACCT_STATE_MAP["PREEMPTED"] is JobStatus.REQUEUED


def test_status_is_terminal():

    assert JobStatus.SUCCEEDED.is_terminal()
    assert JobStatus.FAILED.is_terminal()
    assert not JobStatus.RUNNING.is_terminal()


def test_retryable_excludes_user_cancel():

    assert "NODE_FAIL" in RETRYABLE_SACCT_STATES
    assert "CANCELLED" not in RETRYABLE_SACCT_STATES


# ---- sbatch 渲染 ----


def test_render_sbatch_header():

    script = render_sbatch_script(make_spec())

    assert "--job-name=vla_test_job" in script
    assert "--partition=gpu" in script
    assert "--nodes=8" in script
    assert "--gres=gpu:8" in script
    assert "--time=72:00:00" in script
    assert "--output=/mnt/cpfs/logs/vla_test_job-%j.out" in script


def test_render_launch_command():

    script = render_sbatch_script(make_spec())

    assert "srun --label torchrun" in script
    assert "--nnodes=$SLURM_NNODES" in script
    assert "train.py --data /data" in script
    # 模板里的 shell 变量不能被当成占位符展开
    assert "$SLURM_NNODES" in script


def test_render_no_image_no_container():

    script = render_sbatch_script(make_spec())

    assert "--container-image" not in script


def test_render_with_container():

    script = render_sbatch_script(
        make_spec(
            image="registry.example.com/vla:0.1",
            mounts=["/mnt/cpfs/data:/data"],
        )
    )

    assert "--container-image=registry.example.com/vla:0.1" in script
    assert "--container-mounts=/mnt/cpfs/data:/data" in script


def test_render_env_expand(monkeypatch):

    monkeypatch.setenv("MY_SECRET", "secret-value")

    script = render_sbatch_script(
        make_spec(env={"TOKEN": "${MY_SECRET}", "PLAIN": "1"})
    )

    assert 'export TOKEN="secret-value"' in script
    assert 'export PLAIN="1"' in script


def test_render_env_missing_var_raises():

    with pytest.raises(KeyError):
        render_sbatch_script(make_spec(env={"TOKEN": "${NOT_SET_VAR_XYZ}"}))
