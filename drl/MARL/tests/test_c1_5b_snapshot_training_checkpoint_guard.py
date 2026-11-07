"""CPU regression: Phase-D snapshot sampler must not trigger catalog resume."""

import pytest
from drl.MARL.c1_longtrain_resume_artifacts import supports_staged_resume_artifact


def test_existing_catalog_sampler_retains_resumable_update_artifacts():
    assert supports_staged_resume_artifact({"schema": "mssr.c1_train_sampler.v1"}) is True


def test_snapshot_sampler_uses_periodic_model_checkpoints_not_catalog_resume():
    assert supports_staged_resume_artifact({"schema": "mssr.c1.snapshot_sampler.v1"}) is False


@pytest.mark.parametrize("manifest", [None, {}, {"schema": "unknown"}])
def test_unknown_or_missing_sampler_fails_closed(manifest):
    with pytest.raises(ValueError):
        supports_staged_resume_artifact(manifest)
