"""本地作业注册表单测。"""

from phi_train_slurm.job import registry
from .test_spec import make_spec


def test_save_load_roundtrip(tmp_path):

    registry.save(
        "123",
        make_spec(),
        current_job_id="456",
        attempt=2,
        registry_dir=str(tmp_path),
    )

    record = registry.load("123", registry_dir=str(tmp_path))

    assert record.job_id == "123"
    assert record.current_job_id == "456"
    assert record.attempt == 2
    assert record.spec == make_spec()


def test_save_defaults(tmp_path):

    registry.save("123", make_spec(), registry_dir=str(tmp_path))

    record = registry.load("123", registry_dir=str(tmp_path))

    assert record.current_job_id == "123"
    assert record.attempt == 1


def test_load_missing(tmp_path):

    assert registry.load("404", registry_dir=str(tmp_path)) is None
