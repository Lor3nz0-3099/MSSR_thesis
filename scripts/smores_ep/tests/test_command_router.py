from __future__ import annotations

from smores_ep.control.teleop import InternalMotionMode, SmoresCommand
from smores_ep.isaac.command_router import IsaacMultiModuleCommandRouter


class _State:
    def __init__(self) -> None:
        self.modes: list[str] = []

    def configure_fully_passive_mode(self) -> None:
        self.modes.append("passive")

    def configure_wheel_drive_with_passive_internals(self) -> None:
        self.modes.append("wheels_only")

    def configure_structural_hold_mode(self, _faces) -> None:
        self.modes.append("structural_hold")

    def configure_internal_drive_with_braked_wheels(self, _faces) -> None:
        self.modes.append("internal_with_wheel_brake")

    def configure_pan_velocity_drive_with_braked_wheels(self, _faces) -> None:
        self.modes.append("pan_velocity_with_braked_support")

    def configure_controlled_docking_mode(self, _faces) -> None:
        self.modes.append("wheels_and_internal")


class _Drive:
    def __init__(self) -> None:
        self.commands: list[SmoresCommand] = []

    def apply(self, command: SmoresCommand) -> tuple[float, float]:
        self.commands.append(command)
        return (1.0, 1.0)


class _Docking:
    module_ids = ("locomotor", "payload")
    connections = ()


class _Face:
    def __init__(self, module_id: str, face_name: str) -> None:
        self.module_id = module_id
        self.face_name = face_name


class _Connection:
    def __init__(self, first: _Face, second: _Face) -> None:
        self.first_face = first
        self.second_face = second


class _DockedDocking:
    module_ids = ("locomotor", "payload")
    connections = (
        _Connection(_Face("payload", "BOTTOM"), _Face("locomotor", "TOP")),
    )


def _router():
    states = {"locomotor": _State(), "payload": _State()}
    drives = {"locomotor": _Drive(), "payload": _Drive()}
    return (
        IsaacMultiModuleCommandRouter(states, drives, _Docking()),
        states,
        drives,
    )


def test_missing_modules_and_dead_man_release_every_joint() -> None:
    router, states, _ = _router()

    rates = router.apply(
        {"locomotor": SmoresCommand(linear_x_m_s=0.03)}
    )

    assert rates == {"locomotor": (1.0, 1.0)}
    assert states["locomotor"].modes == ["wheels_only"]
    assert states["payload"].modes == ["passive"]

    router.apply({})
    assert states["locomotor"].modes == ["wheels_only", "passive"]
    assert states["payload"].modes == ["passive"]


def test_docked_module_without_a_command_stays_structurally_held() -> None:
    """A structural link (no posture target of its own) must not sag."""

    states = {"locomotor": _State(), "payload": _State()}
    drives = {"locomotor": _Drive(), "payload": _Drive()}
    router = IsaacMultiModuleCommandRouter(states, drives, _DockedDocking())

    router.apply({})

    assert states["locomotor"].modes == ["structural_hold"]
    assert states["payload"].modes == ["structural_hold"]


def test_internal_primitive_brakes_wheels_for_reaction_support() -> None:
    router, states, _ = _router()

    router.apply(
        {
            "locomotor": SmoresCommand(
                tilt_target_rad=0.6,
                internal_motion=InternalMotionMode.TILT,
            )
        }
    )

    assert states["locomotor"].modes == ["internal_with_wheel_brake"]
    assert states["payload"].modes == ["passive"]


def test_continuous_pan_releases_only_pan_with_braked_supports() -> None:
    router, states, _ = _router()

    router.apply(
        {
            "locomotor": SmoresCommand(
                internal_motion=InternalMotionMode.PAN_VELOCITY,
                pan_velocity_rad_s=0.8,
            )
        }
    )

    assert states["locomotor"].modes == [
        "pan_velocity_with_braked_support"
    ]
    assert states["payload"].modes == ["passive"]


def test_legacy_hold_keeps_original_internal_drive_route() -> None:
    """Holonomic structural retention must not redefine teleop HOLD."""

    router, states, _ = _router()

    router.apply(
        {
            "locomotor": SmoresCommand(
                pan_target_rad=0.2,
                tilt_target_rad=-0.3,
                internal_motion=InternalMotionMode.HOLD,
            )
        }
    )

    assert states["locomotor"].modes == ["internal_with_wheel_brake"]
    assert states["payload"].modes == ["passive"]


def test_hold_captures_structure_without_commanding_free_wheels() -> None:
    router, states, _ = _router()

    router.apply(
        {
            "payload": SmoresCommand(
                pan_target_rad=0.4,
                tilt_target_rad=-1.35,
                internal_motion=InternalMotionMode.STRUCTURAL_HOLD,
            )
        }
    )

    assert states["payload"].modes == ["structural_hold"]
    assert states["locomotor"].modes == ["passive"]


def test_structural_hold_with_motion_powers_the_locomotor_wheels() -> None:
    router, states, drives = _router()

    router.apply(
        {
            "locomotor": SmoresCommand(
                linear_x_m_s=0.02,
                pan_target_rad=0.0,
                tilt_target_rad=0.9,
                internal_motion=InternalMotionMode.STRUCTURAL_HOLD,
            )
        }
    )

    assert states["locomotor"].modes == ["wheels_and_internal"]
    assert drives["locomotor"].commands[-1].linear_x_m_s == 0.02


def test_fold_pusher_drives_wheels_and_holds_internal_structure() -> None:
    router, states, drives = _router()

    router.apply(
        {
            "locomotor": SmoresCommand(
                linear_x_m_s=0.025,
                pan_target_rad=0.0,
                tilt_target_rad=0.4,
                internal_motion=InternalMotionMode.STRUCTURAL_HOLD,
            )
        }
    )

    assert states["locomotor"].modes == ["wheels_and_internal"]
    assert drives["locomotor"].commands[-1].pan_target_rad == 0.0
    assert drives["locomotor"].commands[-1].tilt_target_rad == 0.4
