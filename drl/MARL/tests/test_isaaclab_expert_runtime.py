"""UNIT reset regressions over the existing primitive/controller backend."""
import importlib.util
from pathlib import Path

import pytest


_fixture_path = Path(__file__).resolve().parents[3] / "scripts/smores_ep/tests/test_free_module_reset.py"
_fixture_spec = importlib.util.spec_from_file_location("native_reset_test_fixture", _fixture_path)
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
setup_backend = _fixture_module.setup_backend
from smores_ep.control.teleop import SmoresCommand
from smores_ep.primitives.model import PrimitiveState


def test_native_episode_reset_retires_executor_posture_and_group_memory():
    from drl.MARL.isaaclab_expert_runtime import reset_free_assembly_modules

    states, _, _, router, executor = setup_backend()
    executor._retained_internal_commands.update(free=SmoresCommand(), anchor=SmoresCommand())
    executor._pan_reference_offset_rad.update(free=6., anchor=2.)
    executor._group_module_ids = {'old-fold': {'free'}, 'anchor-fold': {'anchor'}}
    executor._released_joint_groups.update(executor._group_module_ids)
    executor._completed_joint_groups.update(executor._group_module_ids)
    # The previous native reset touched only the router; stale primitive memory
    # remains observable and can bypass a subsequent coordinated fold barrier.
    router.reset_free_modules(('free',))
    assert 'free' in executor._retained_internal_commands
    assert 'old-fold' in executor._completed_joint_groups

    status = reset_free_assembly_modules(dict(router=router, primitive_executor=executor),
        ('free',), stamp=1., execution_id='native-env0-generation2')
    assert status.state is PrimitiveState.SUCCEEDED
    assert status.goal_id == 'native-env0-generation2-reset-free'
    assert set(executor._retained_internal_commands) == {'anchor'}
    assert executor._pan_reference_offset_rad == {'anchor': 2.}
    assert executor._completed_joint_groups == {'anchor-fold'}
    assert executor._released_joint_groups == {'anchor-fold'}
    assert executor._group_module_ids == {'anchor-fold': {'anchor'}}
    assert not executor.active_goals and not executor._resource_owners
    assert states['free'].targets['left_wheel_velocity_rad_s'] == 0


def test_native_reset_fails_closed_when_drive_reset_application_fails():
    from drl.MARL.isaaclab_expert_runtime import reset_free_assembly_modules

    def fail(ids):
        raise RuntimeError('measured backend reset failed')
    _, _, _, router, executor = setup_backend(fail)
    with pytest.raises(RuntimeError, match='Native free-module reset failed'):
        reset_free_assembly_modules(dict(router=router, primitive_executor=executor),
            ('free',), stamp=1., execution_id='native-env0-generation2')
