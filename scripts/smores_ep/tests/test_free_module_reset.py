"""Reset barriers exercise real executor/router/controller code without PhysX."""
import json
from types import SimpleNamespace

import pytest

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.control.teleop import SmoresCommand, InternalMotionMode
from smores_ep.isaac.dynamic_stage import DynamicDriveController
from smores_ep.isaac.command_router import IsaacMultiModuleCommandRouter
from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
from smores_ep.primitives.file_channel import ActionFileChannel, PrimitiveFileChannel
from smores_ep.primitives.model import PrimitiveGoal, PrimitiveState


class State:
    def __init__(self):
        self.measured = SimpleNamespace(pan_joint_rad=.7, tilt_joint_rad=-.3)
        self.targets = {}
        self.modes = []
        self.contact = 'wheel'

    def read(self):
        return self.measured

    def set_targets(self, **targets):
        self.targets = targets

    def configure_fully_passive_mode(self):
        self.modes.append('passive')

    def set_pan_contact_mode(self, mode):
        self.contact = mode


def setup_backend(callback=None):
    states = {m: State() for m in ('free', 'anchor')}
    drives = {m: DynamicDriveController(s, SmoresGeometry(), 2.4) for m, s in states.items()}
    docking = SimpleNamespace(module_ids=tuple(states), connections=())
    router = IsaacMultiModuleCommandRouter(states, drives, docking)
    executor = IsaacPrimitiveExecutor(object(), {m: '/' + m for m in states}, states, docking)
    executor.reset_free_modules_callback = callback or router.reset_free_modules
    return states, drives, docking, router, executor


def reset_goal(ids=('free',)):
    return PrimitiveGoal.from_dict(dict(goal_id='reset', primitive='reset_free_modules', module_ids=ids))


def test_reset_is_selective_fresh_initialization_and_ack_follows_application():
    states, drives, _, router, executor = setup_backend()
    drives['free'].apply(SmoresCommand(internal_motion=InternalMotionMode.PAN, pan_target_rad=12.0))
    anchor_before = drives['anchor'].internal_targets_rad, dict(states['anchor'].targets), list(states['anchor'].modes)
    before = dict(vars(states['free'].measured))
    executor._pan_reference_offset_rad.update(free=6., anchor=2.)
    executor._retained_internal_commands.update(free=SmoresCommand(), anchor=SmoresCommand())
    executor._passive_internal_module_ids.update(('free', 'anchor'))
    applied = []
    def apply(ids):
        assert executor.active_goal.goal_id == 'reset'
        router.reset_free_modules(ids)
        applied.append(ids)
    executor.reset_free_modules_callback = apply
    assert executor.submit(reset_goal(), 0).state is PrimitiveState.ACCEPTED
    assert applied == []
    result = executor.step(.1)
    assert result.statuses[0].state is PrimitiveState.SUCCEEDED
    assert applied == [('free',)]
    fresh = DynamicDriveController(states['free'], SmoresGeometry(), 2.4)
    assert drives['free'].internal_targets_rad == fresh.internal_targets_rad
    assert vars(states['free'].measured) == before
    assert anchor_before == (drives['anchor'].internal_targets_rad, states['anchor'].targets, states['anchor'].modes)
    assert executor._pan_reference_offset_rad == {'anchor': 2.}
    assert set(executor._retained_internal_commands) == {'anchor'}
    assert executor._passive_internal_module_ids == {'anchor'}
    assert not executor._resource_owners
    assert states['free'].targets['pan_joint_velocity_rad_s'] == 0
    assert states['free'].contact == 'pan_face'


def test_attached_reset_is_rejected_without_mutation():
    states, _, docking, router, executor = setup_backend()
    docking.connections = (SimpleNamespace(first_face=SimpleNamespace(module_id='free', face_name='TOP'), second_face=SimpleNamespace(module_id='anchor', face_name='BOTTOM')),)
    before = dict(states['free'].targets)
    assert executor.submit(reset_goal(), 0).state is PrimitiveState.REJECTED
    with pytest.raises(ValueError, match='attached'):
        router.reset_free_modules(('free',))
    assert states['free'].targets == before


def test_reset_refuses_active_shared_owner_and_missing_backend():
    _, _, _, _, executor = setup_backend()
    executor._resource_owners['locomotion:free'] = {'old-shared': 'shared'}
    assert executor.submit(reset_goal(), 0).code == 'RESOURCE_BUSY'
    executor._resource_owners.clear()
    executor.reset_free_modules_callback = None
    assert executor.submit(reset_goal(), 0).state is PrimitiveState.ACCEPTED
    assert executor.step(.1).statuses[0].state is PrimitiveState.FAILED


def test_reset_payload_accepts_multiple_released_modules():
    assert reset_goal(('free', 'another')).module_ids == ('free', 'another')


def test_action_reset_blocks_old_payload_rewrites_until_new_command_handoff(tmp_path):
    path = tmp_path / 'actions.json'
    channel = ActionFileChannel(path)
    def write(command_id):
        PrimitiveFileChannel._write_atomic(path, json.dumps({'locomotion': {'free': {'vx': .1}, 'anchor': {'vx': .2}}, 'expert': {'debug': {'command_id': command_id}, 'pan_traction_module_ids': ['free', 'anchor']}}))
    write('source')
    assert set(channel.commands(0)) == {'free', 'anchor'}
    channel.invalidate_modules(('free',))
    assert set(channel.commands(.1)) == {'anchor'}
    write('source')
    assert set(channel.commands(.2)) == {'anchor'}
    assert channel.diagnostics.pan_traction_module_ids == ('anchor',)
    write('')
    assert set(channel.commands(.3)) == {'anchor'}
    write('target-behavior')
    assert set(channel.commands(.4)) == {'free', 'anchor'}


def test_backend_failure_never_acknowledges_reset():
    def fail(ids):
        raise RuntimeError("drive application failed")
    _, _, _, _, executor = setup_backend(fail)
    assert executor.submit(reset_goal(), 0).state is PrimitiveState.ACCEPTED
    result = executor.step(.1)
    assert result.statuses[0].state is PrimitiveState.FAILED
    assert not executor._resource_owners


def test_action_reset_rejects_delayed_source_even_after_new_handoff(tmp_path):
    path = tmp_path / 'actions.json'
    channel = ActionFileChannel(path)
    def write(command_id):
        PrimitiveFileChannel._write_atomic(path, json.dumps({'locomotion': {'free': {'vx': .1}}, 'expert': {'debug': {'command_id': command_id}}}))
    write('source')
    channel.commands(0)
    channel.invalidate_modules(('free',))
    write('target')
    assert 'free' in channel.commands(.1)
    write('source')
    assert 'free' not in channel.commands(.2)


def test_reset_retires_mixed_groups_and_preserves_unrelated_anchor_memory():
    _, _, _, _, executor = setup_backend()
    executor._group_module_ids = {'free-group': {'free'}, 'anchor-group': {'anchor'}, 'shared': {'free', 'anchor'}}
    executor._released_joint_groups.update(executor._group_module_ids)
    executor._released_undock_groups.update(executor._group_module_ids)
    executor._completed_joint_groups.update(executor._group_module_ids)
    executor.submit(reset_goal(), 0)
    assert executor.step(.1).statuses[0].state is PrimitiveState.SUCCEEDED
    assert executor._released_joint_groups == {'anchor-group'}
    assert executor._released_undock_groups == {'anchor-group'}
    assert executor._completed_joint_groups == {'anchor-group'}
    assert executor._group_module_ids == {'anchor-group': {'anchor'}}
    new_tilt = PrimitiveGoal.from_dict(dict(
        goal_id='new-tilt', primitive='set_tilt', module_ids=['free'],
        parameters=dict(angle_rad=.8, coordination_group='shared', coordination_size=2),
    ))
    assert executor.submit(new_tilt, .2).state is PrimitiveState.ACCEPTED
    active = executor._active['new-tilt']
    assert not executor._coordinated_tilt_ready(active, .3)
    assert not executor._coordinated_tilt_complete(active)


@pytest.mark.parametrize('reset_first', [False, True])
def test_reset_requires_handoff_from_active_member_of_mixed_group(reset_first):
    _, _, _, _, executor = setup_backend()
    executor._group_module_ids = {'shared': {'free', 'anchor'}}
    if reset_first:
        assert executor.submit(reset_goal(), 0).state is PrimitiveState.ACCEPTED
    anchor_tilt = PrimitiveGoal.from_dict(dict(
        goal_id='anchor-tilt', primitive='set_tilt', module_ids=['anchor'],
        parameters=dict(angle_rad=.8, coordination_group='shared', coordination_size=2),
    ))
    assert executor.submit(anchor_tilt, 0).state is PrimitiveState.ACCEPTED
    if reset_first:
        assert executor.step(.1).statuses[0].state is PrimitiveState.FAILED
        assert executor._group_module_ids == {'shared': {'free', 'anchor'}}
    else:
        assert executor.submit(reset_goal(), .1).code == 'RESOURCE_BUSY'


def test_runtime_barrier_drops_held_commands_and_baseline_before_drive_reset(tmp_path):
    from smores_ep.scenarios import parallel_self_assembly as runtime
    states, drives, _, router, _ = setup_backend()
    path = tmp_path / 'actions.json'
    channel = ActionFileChannel(path)
    PrimitiveFileChannel._write_atomic(path, json.dumps({'locomotion': {'free': {'vx': .1}, 'anchor': {'vx': .2}}, 'expert': {'debug': {'command_id': 'source'}}}))
    channel.commands(0)
    held = {'free': SmoresCommand(linear_x_m_s=.3), 'anchor': SmoresCommand()}
    anchor_command = held['anchor']
    runtime.apply_free_module_reset(
        ('free',), action_channel=channel,
        held_primitive_commands=held, command_router=router,
    )
    assert held == {'anchor': anchor_command}
    assert set(channel.commands(.1)) == {'anchor'}
    assert drives['free'].internal_targets_rad == (.7, .3)
    assert states['free'].targets['left_wheel_velocity_rad_s'] == 0

def test_gravity_settle_invalidates_stale_behavior_command_sources_once(tmp_path):
    """Legacy reconfiguration must quarantine the previous RC-Car PAN command."""

    states, drives, _, router, executor = setup_backend()

    path = tmp_path / "actions.json"
    channel = ActionFileChannel(path)

    def write(command_id, pan_rate):
        PrimitiveFileChannel._write_atomic(
            path,
            json.dumps(
                {
                    "locomotion": {
                        "free": {
                            "pan_rate_rad_s": pan_rate,
                        },
                    },
                    "expert": {
                        "debug": {
                            "command_id": command_id,
                        },
                        "pan_traction_module_ids": ["free"],
                    },
                }
            ),
        )

    write("source-rc-car", 1.5)
    source_commands = channel.commands(0.0)

    assert source_commands["free"].internal_motion is InternalMotionMode.PAN_VELOCITY
    assert source_commands["free"].pan_velocity_rad_s == pytest.approx(1.5)

    held = {
        "free": source_commands["free"],
    }
    invalidations = []

    def invalidate_command_sources(module_ids):
        invalidations.append(module_ids)
        channel.invalidate_modules(module_ids)

        for module_id in module_ids:
            held.pop(module_id, None)

    # This is the runtime callback that the gravity-settle primitive must
    # execute exactly once at the locomotion -> reconfiguration handoff.
    executor.invalidate_module_command_sources_callback = (
        invalidate_command_sources
    )

    settle = PrimitiveGoal.from_dict(
        {
            "goal_id": "legacy-handoff-settle",
            "primitive": "gravity_settle",
            "module_ids": ["free"],
            "parameters": {
                "passive_module_ids": ["free"],
                "duration_s": 0.5,
                "clear_passive_policy_on_finish": True,
            },
            "timeout_s": 2.0,
        }
    )

    assert executor.submit(settle, 0.0).state is PrimitiveState.ACCEPTED

    first = executor.step(0.1)
    second = executor.step(0.2)

    assert invalidations == [("free",)]
    assert held == {}

    # The previous RC-Car producer must stay quarantined.
    assert channel.commands(0.2) == {}

    # Even a delayed rewrite carrying the same old command ID stays blocked.
    write("source-rc-car", 2.0)
    assert channel.commands(0.3) == {}

    # A genuinely new behavior owner may take over later.
    write("target-behavior", 0.7)
    fresh = channel.commands(0.4)

    assert fresh["free"].internal_motion is InternalMotionMode.PAN_VELOCITY
    assert fresh["free"].pan_velocity_rad_s == pytest.approx(0.7)

    assert len(first.statuses) == 1
    assert first.statuses[0].goal_id == "legacy-handoff-settle"
    assert first.statuses[0].state is PrimitiveState.RUNNING

    assert len(second.statuses) == 1
    assert second.statuses[0].goal_id == "legacy-handoff-settle"
    assert second.statuses[0].state is PrimitiveState.RUNNING
