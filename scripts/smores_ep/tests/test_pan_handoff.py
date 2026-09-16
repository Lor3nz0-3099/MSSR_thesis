"""Explicit reconfiguration reset; preserve the legacy assembly controller."""
from __future__ import annotations
import pytest

import math
from types import SimpleNamespace
import unittest

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.control.teleop import InternalMotionMode as Mode, SmoresCommand
from smores_ep.isaac.dynamic_stage import DynamicDriveController
from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
from smores_ep.primitives.model import PrimitiveGoal, PrimitiveState


class FakeArticulation:
    def __init__(self):
        self.state = SimpleNamespace(pan_joint_rad=0.0, tilt_joint_rad=0.0,
                                     pan_joint_rad_s=0.0, tilt_joint_rad_s=0.0)
        self.targets = {}

    def read(self):
        return self.state

    def set_targets(self, **targets):
        self.targets = targets

    def set_pan(self, angle):
        self.state.pan_joint_rad = math.atan2(math.sin(angle), math.cos(angle))


class PanHandoffTests(unittest.TestCase):
    def setUp(self):
        self.state = FakeArticulation()
        self.drive = DynamicDriveController(self.state, SmoresGeometry(), 2.4)
        self.executor = IsaacPrimitiveExecutor(
            stage=object(), module_roots={"m": "/M"}, states={"m": self.state},
            docking=SimpleNamespace(module_ids=("m",), connections=()),
        )

    def drive_turns(self, angle):
        start = self.state.state.pan_joint_rad
        for tick in range(1, 501):
            self.state.set_pan(start + (angle - start) * tick / 500)
            self.drive.apply(SmoresCommand(
                internal_motion=Mode.PAN_VELOCITY, pan_velocity_rad_s=1.5,
                tilt_target_rad=0.4,
            ))

    def submit_settle(self, name="settle", start=0.0, duration=2.0):
        goal = PrimitiveGoal.from_dict({
            "goal_id": name, "primitive": "gravity_settle", "module_ids": ["m"],
            "parameters": {"passive_module_ids": ["m"], "duration_s": duration,
                           "clear_passive_policy_on_finish": True},
            "timeout_s": 5.0,
        })
        self.assertIs(self.executor.submit(goal, start).state, PrimitiveState.ACCEPTED)

    def settle_frame(self, now, baseline=None):
        result = self.executor.step(now)
        commands = self.executor.compose_with_baseline(baseline or {}, result.commands)
        command = commands["m"]
        self.assertTrue(command.reset_internal_targets)
        self.drive.apply(command)
        self.assertEqual(self.state.targets["pan_joint_velocity_rad_s"], 0.0)
        self.assertEqual(self.state.targets["left_wheel_velocity_rad_s"], 0.0)
        self.assertEqual(self.state.targets["right_wheel_velocity_rad_s"], 0.0)
        return result

    def test_assembly_pan_correction_survives_ordinary_passive_approach(self):
        # Baseline behavior from the source archive: retain the clocking
        # target during a normal PASSIVE wheel approach, including its servo.
        self.drive.apply(SmoresCommand(
            internal_motion=Mode.PAN, pan_target_rad=0.2, tilt_target_rad=0.4,
        ))
        self.state.set_pan(0.5)
        self.state.state.tilt_joint_rad = -0.1
        approach = SmoresCommand(linear_x_m_s=-0.02)
        self.assertFalse(approach.reset_internal_targets)
        rates = self.drive.apply(approach)
        self.assertAlmostEqual(self.state.targets["pan_logical_target_rad"], 0.2)
        self.assertAlmostEqual(self.state.targets["pan_joint_velocity_rad_s"], -1.2)
        self.assertAlmostEqual(self.state.targets["tilt_joint_position_rad"], -0.4)
        self.assertTrue(all(rate < 0 for rate in rates))

    def test_ordinary_tilt_keeps_legacy_pan_semantics(self):
        self.drive.apply(SmoresCommand(internal_motion=Mode.PAN, pan_target_rad=0.2))
        self.drive.apply(SmoresCommand(
            internal_motion=Mode.TILT, pan_target_rad=0.5, tilt_target_rad=0.7,
        ))
        self.assertAlmostEqual(self.drive.internal_targets_rad[0], 0.2)
        self.assertAlmostEqual(self.drive.internal_targets_rad[1], 0.7)

    def test_settle_discards_turn_counts_without_changing_physical_state(self):
        self.executor._joint_positions("m")
        self.drive_turns(4 * math.pi + 0.4)
        self.assertGreater(self.drive.pan_position_rad, 12.0)
        self.state.state.tilt_joint_rad = -0.15
        before = vars(self.state.state).copy()
        self.submit_settle()
        self.settle_frame(0.1)
        self.assertEqual(vars(self.state.state), before)
        pan, tilt = self.executor._joint_positions("m")
        self.assertAlmostEqual(pan, 0.4)
        self.assertAlmostEqual(self.drive.pan_position_rad, pan)
        self.assertAlmostEqual(self.drive.internal_targets_rad[1], tilt)

    def test_settle_follows_backdrive_and_suppresses_stale_baseline(self):
        self.drive_turns(4 * math.pi + 0.4)
        baseline = {"m": SmoresCommand(
            internal_motion=Mode.PAN_VELOCITY, pan_velocity_rad_s=1.5,
            linear_x_m_s=0.03,
        )}
        self.executor.compose_with_baseline(baseline, {})
        self.submit_settle()
        for now, pan, tilt in [(0.1, 0.4, 0.15), (1.0, 0.6, 0.05), (2.1, 0.7, 0.0)]:
            self.state.set_pan(pan)
            self.state.state.tilt_joint_rad = -tilt
            result = self.settle_frame(now, baseline)
            self.assertAlmostEqual(self.drive.internal_targets_rad[0], pan)
            self.assertAlmostEqual(self.drive.internal_targets_rad[1], tilt)
            self.assertNotIn("m", self.executor._retained_internal_commands)
        self.assertIs(result.status.state, PrimitiveState.SUCCEEDED)
        # The baseline producer is stopped at the handoff. The next lift
        # uses the fresh PAN hold even with unchanged legacy TILT behavior.
        goal = PrimitiveGoal.from_dict({
            "goal_id": "lift", "primitive": "set_tilt", "module_ids": ["m"],
            "parameters": {"angle_rad": 0.7}, "timeout_s": 5.0,
        })
        self.executor.submit(goal, 2.2)
        step = self.executor.step(2.3)
        command = self.executor.compose_with_baseline({}, step.commands)["m"]
        self.assertFalse(command.reset_internal_targets)
        self.drive.apply(command)
        self.assertAlmostEqual(self.state.targets["pan_logical_target_rad"], 0.7)
        self.assertEqual(self.state.targets["pan_joint_velocity_rad_s"], 0.0)

    def test_reset_is_delivered_even_if_first_settle_step_is_terminal(self):
        self.drive_turns(4 * math.pi + 0.4)
        self.submit_settle(duration=0.1)
        result = self.settle_frame(0.2)
        self.assertIs(result.status.state, PrimitiveState.SUCCEEDED)
        self.assertAlmostEqual(self.drive.internal_targets_rad[0], 0.4)

    def test_new_traction_can_be_reset_again_in_the_next_reconfiguration(self):
        for number, angle in enumerate((4 * math.pi + 0.4, -4 * math.pi - 0.5)):
            self.drive_turns(angle)
            start = 10.0 * number
            self.submit_settle(name=f"settle-{number}", start=start, duration=0.1)
            self.settle_frame(start + 0.2)
            pan, _ = self.executor._joint_positions("m")
            self.assertAlmostEqual(self.drive.internal_targets_rad[0], pan)

    def test_explicit_relative_pan_still_requests_multiple_turns_after_reset(self):
        self.drive_turns(4 * math.pi + 0.4)
        self.submit_settle(duration=0.1)
        self.settle_frame(0.2)
        goal = PrimitiveGoal.from_dict({
            "goal_id": "turn", "primitive": "rotate_pan_by", "module_ids": ["m"],
            "parameters": {"delta_rad": 4 * math.pi}, "timeout_s": 30.0,
        })
        self.executor.submit(goal, 0.3)
        self.drive.apply(self.executor.step(0.4).commands["m"])
        self.assertAlmostEqual(self.drive.internal_targets_rad[0], 0.4 + 4 * math.pi)
        self.assertEqual(self.state.targets["pan_joint_velocity_rad_s"], 2.4)


    def test_ik_periodic_relative_pan_survives_equivalent_branch_jump(self):
        """An IK PAN must not unwind when the angle representation jumps 2*pi."""

        # Same physical PAN, but deliberately represented on the -2*pi branch.
        self.state.set_pan(0.4)

        tracker = self.executor._pan_trackers["m"]
        tracker._previous_raw_rad = 0.4
        tracker._continuous_rad = 0.4 - 2.0 * math.pi

        goal = PrimitiveGoal.from_dict({
            "goal_id": "ik-periodic",
            "primitive": "rotate_pan_by",
            "module_ids": ["m"],
            "parameters": {
                "delta_rad": 0.1,
                "periodic_equivalent": True,
            },
            "timeout_s": 30.0,
        })

        status = self.executor.submit(goal, 0.0)
        self.assertIs(status.state, PrimitiveState.ACCEPTED)

        # Reproduce the observed runtime condition: another reader/reference
        # now reports the exact same physical PAN on the +0.4 rad branch.
        fresh_tracker = type(self.executor._pan_trackers["m"])()
        fresh_tracker.update(0.4)
        self.executor._pan_trackers["m"] = fresh_tracker

        result = self.executor.step(0.1)

        # Target -2*pi+0.5 and current +0.4 are physically only 0.1 rad apart.
        self.assertAlmostEqual(
            result.status.feedback["error_rad"],
            0.1,
            places=6,
        )
        self.assertAlmostEqual(
            result.commands["m"].pan_target_rad,
            0.5,
            places=6,
        )


    def test_reset_requires_an_explicit_passive_command(self):
        for mode in (Mode.PAN, Mode.PAN_VELOCITY, Mode.TILT, Mode.STRUCTURAL_HOLD):
            with self.assertRaises(ValueError):
                SmoresCommand(internal_motion=mode, reset_internal_targets=True)


if __name__ == "__main__":
    unittest.main()

def test_pan_position_target_uses_nearest_periodic_branch() -> None:
    import math
    from types import SimpleNamespace

    import pytest

    from smores_ep.control.teleop import InternalMotionMode, SmoresCommand
    from smores_ep.isaac.dynamic_stage import (
        DynamicDriveController,
        DynamicJointState,
    )

    current = 2.0 * math.pi - 0.05

    class FixedTracker:
        def update(self, _raw):
            return current

    class FakeArticulation:
        module_root = "/World/smores_06"

        def __init__(self):
            self.last_targets = None

        def read(self):
            return DynamicJointState(
                left_wheel_rad=0.0,
                right_wheel_rad=0.0,
                left_wheel_rad_s=0.0,
                right_wheel_rad_s=0.0,
                tilt_joint_rad=0.0,
                tilt_joint_rad_s=0.0,
                pan_joint_rad=-0.05,
                pan_joint_rad_s=0.0,
            )

        def set_targets(self, **kwargs):
            self.last_targets = kwargs

    articulation = FakeArticulation()

    controller = object.__new__(DynamicDriveController)
    controller._articulation = articulation
    controller._geometry = SimpleNamespace(
        wheel_radius_m=0.0315,
        track_width_m=0.080,
        tilt_min_rad=-math.pi,
        tilt_max_rad=math.pi,
    )
    controller._max_wheel_speed_rad_s = 10.0
    controller._pan_servo_gain_s = 4.0
    controller._pan_angle = FixedTracker()
    controller._pan_position_rad = current
    controller._pan_target_rad = current
    controller._tilt_target_rad = 0.0

    controller.apply(
        SmoresCommand(
            pan_target_rad=0.0,
            tilt_target_rad=0.0,
            internal_motion=InternalMotionMode.PAN,
        )
    )

    applied, _ = controller.internal_targets_rad

    # Requested 0 rad is physically equivalent to +2*pi here.
    # The position controller must travel ~0.05 rad, not ~6.23 rad.
    assert applied == pytest.approx(2.0 * math.pi, abs=1e-9)
    assert abs(applied - current) == pytest.approx(0.05, abs=1e-9)
    assert articulation.last_targets is not None
    assert articulation.last_targets["pan_logical_target_rad"] == pytest.approx(
        applied
    )


def test_physx_pan_target_preserves_unwrapped_continuous_branch() -> None:
    """PhysX must receive the continuous PAN branch selected upstream."""
    from smores_ep.isaac.dynamic_stage import ArticulationStateReader

    class FakeArticulation:
        def __init__(self) -> None:
            self.position_call = None
            self.velocity_call = None

        def set_dof_velocity_targets(self, values, *, dof_indices):
            self.velocity_call = (list(values), list(dof_indices))

        def set_dof_position_targets(self, values, *, dof_indices):
            self.position_call = (list(values), list(dof_indices))

    articulation = FakeArticulation()

    reader = object.__new__(ArticulationStateReader)
    reader._articulation = articulation
    reader._indices = {
        "left_wheel": 0,
        "right_wheel": 1,
        "tilt": 2,
        "pan": 3,
    }
    reader._position_targets = {"tilt": 0.0, "pan": 0.0}

    requested_pan = 4.911509

    reader.set_targets(
        left_wheel_velocity_rad_s=0.0,
        right_wheel_velocity_rad_s=0.0,
        tilt_joint_position_rad=-0.14,
        pan_joint_velocity_rad_s=0.0,
        pan_logical_target_rad=requested_pan,
    )

    assert articulation.position_call is not None
    position_values, position_indices = articulation.position_call

    assert position_indices == [2, 3]
    assert position_values[0] == pytest.approx(-0.14)

    # Critical regression:
    # +4.911509 and -1.371676 are geometrically equivalent, but for a
    # continuous joint they are different physical branches.  The branch
    # choice belongs to DynamicDriveController, not this PhysX writer.
    assert position_values[1] == pytest.approx(requested_pan)
    assert reader.target_positions()[1] == pytest.approx(requested_pan)
