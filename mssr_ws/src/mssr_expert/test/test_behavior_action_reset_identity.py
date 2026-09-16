"""Exercise ROS action production against the backend reset quarantine."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

pytest.importorskip('rclpy')
from mssr_expert.nodes.smores_morphology_behavior_node import SmoresMorphologyBehaviorNode
from mssr_expert.behaviors.morphology_library import (
    AssignedModule, BehaviorProgramStep, LongitudinalPositionGoal, MorphologyLibrary,
)
from mssr_expert.behaviors.morphology_dof_model import SmoresMorphologyDofAnalyzer
from mssr_expert.execution.morphology_behavior_executor import MorphologyBehaviorExecutor, MorphologyCommand
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph, GraphNode
from smores_ep.primitives.file_channel import ActionFileChannel, PrimitiveFileChannel

ROLES = ('snake_tail', 'snake_rear', 'snake_hip', 'snake_center_rear',
         'snake_center_front', 'snake_shoulder', 'snake_neck', 'snake_head')


class ActionHarness:
    # Keep real producer logic without constructing DDS participants or timers.
    _step = SmoresMorphologyBehaviorNode._step
    _step_cmd_vel = SmoresMorphologyBehaviorNode._step_cmd_vel
    _publish_actions = SmoresMorphologyBehaviorNode._publish_actions

    def __init__(self, action_path):
        self._library = MorphologyLibrary.load(Path(__file__).parents[1] / 'config/smores_morphology_behaviors.json')
        self._assignments = tuple(AssignedModule(f'm{i}', f'v{i}', r) for i, r in enumerate(ROLES))
        self._morphology_name = 'snake8'
        self._assembly_ready = True
        self._latest_robot_graph = AttributedRobotGraph(nodes=tuple(
            GraphNode(f'm{i}', {'pose': {'position': [0.0, 0.0, 0.03]}}) for i in range(8)))
        self._dof_inventory = SmoresMorphologyDofAnalyzer().analyze(self._latest_robot_graph)
        self._executor = MorphologyBehaviorExecutor(self._library)
        self._latest_primitive_status = {}
        self._last_terminal_command_id = ''
        self._last_cmd_vel_s = time.monotonic()
        self._latest_cmd_vel = (0.02, 0.0, 0.0)
        self._latest_nav2_route_status = {'route_id': 'route-before-reset'}
        self._nav2_terminal_status_consumed = False
        self._cmd_vel_output_active = False
        self._actions_publisher = SimpleNamespace(publish=lambda message: PrimitiveFileChannel._write_atomic(action_path, message.data))
        self.records = []

    def get_parameter(self, name):
        return SimpleNamespace(value={'cmd_vel_timeout_s': 60.0, 'behavior_dataset_episode_id': 'episode'}[name])

    def _record_behavior_transition(self, decision, locomotion):
        self.records.append(decision)

    def _publish_status(self, *args):
        pass

    def start_position_hold(self, command_id):
        self._executor.start(MorphologyCommand(command_id, 'snake8', 'crawl'), self._assignments,
            program_steps_override=(BehaviorProgramStep(phase='POSITION_HOLD', linear_m_s=0.02,
                active_target_roles=ROLES, position_goal=LongitudinalPositionGoal('m0', 1.0, 0.01)),))


@pytest.mark.parametrize('mode', ['nav2', 'position_hold'])
def test_new_command_resumes_released_module_but_repeated_old_ticks_do_not(tmp_path, mode):
    path = tmp_path / 'actions.json'
    node = ActionHarness(path)
    channel = ActionFileChannel(path)
    if mode == 'position_hold':
        node.start_position_hold('hold-before-reset')
    step = node._step_cmd_vel if mode == 'nav2' else node._step
    step()
    assert channel.commands(0.0)['m0'].linear_x_m_s > 0
    channel.invalidate_modules(('m0',))
    for tick in (0.1, 0.2):
        step()
        assert 'm0' not in channel.commands(tick)
    if mode == 'nav2':
        node._latest_nav2_route_status = {'route_id': 'route-after-reset'}
    else:
        node.start_position_hold('hold-after-reset')
    step()
    assert channel.commands(0.3)['m0'].linear_x_m_s > 0
    expected = 'route-after-reset' if mode == 'nav2' else 'hold-after-reset'
    assert channel.diagnostics.command_id == expected
    assert node.records[-1].command_id == expected


def test_nav2_fallback_identity_is_stable_through_watchdog_and_terminal(tmp_path):
    path = tmp_path / 'actions.json'
    node = ActionHarness(path)
    node._latest_nav2_route_status = {}
    node._step_cmd_vel()
    first = json.loads(path.read_text())['expert']['debug'].get('command_id')
    node._last_cmd_vel_s = None
    node._step_cmd_vel()
    stopped = json.loads(path.read_text())
    node._latest_nav2_route_status = {'done': True, 'success': True}
    node._step_cmd_vel()
    terminal = json.loads(path.read_text())
    assert first == 'episode-nav2'
    assert stopped['expert']['debug']['command_id'] == first
    assert terminal['expert']['debug']['command_id'] == first
    assert terminal['expert']['done'] is True
