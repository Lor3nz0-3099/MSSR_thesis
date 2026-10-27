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
        self.posture_captures = 0

    def initialize_from_measured_posture(self) -> None:
        self.posture_captures += 1

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



def test_emergency_stop_captures_once_and_overrides_all_motion_commands() -> None:
    router, states, drives = _router()

    router.set_emergency_stop(True)

    # Entry captures the physically reached posture exactly once.
    assert drives["locomotor"].posture_captures == 1
    assert drives["payload"].posture_captures == 1

    # Every module is put into braked-wheel/internal-hold mode.
    assert states["locomotor"].modes[-1] == "internal_with_wheel_brake"
    assert states["payload"].modes[-1] == "internal_with_wheel_brake"

    # Capturing the stop must not synthesize a normal morphology command.
    assert drives["locomotor"].commands == []
    assert drives["payload"].commands == []

    # Even a later locomotion/internal-motion command cannot bypass E-STOP.
    rates = router.apply(
        {
            "locomotor": SmoresCommand(
                linear_x_m_s=0.08,
                angular_z_rad_s=0.4,
                internal_motion=InternalMotionMode.PAN_VELOCITY,
                pan_velocity_rad_s=1.0,
            )
        }
    )

    assert rates == {
        "locomotor": (0.0, 0.0),
        "payload": (0.0, 0.0),
    }
    assert drives["locomotor"].commands == []
    assert drives["payload"].commands == []

    # Repeated STOP delivery must not recapture a slightly changed posture.
    router.set_emergency_stop(True)
    assert drives["locomotor"].posture_captures == 1
    assert drives["payload"].posture_captures == 1


def test_emergency_stop_clear_does_not_move_robot_and_fresh_command_can_resume() -> None:
    router, _, drives = _router()

    router.set_emergency_stop(True)

    capture_counts = {
        module_id: drive.posture_captures
        for module_id, drive in drives.items()
    }
    command_counts = {
        module_id: len(drive.commands)
        for module_id, drive in drives.items()
    }

    router.set_emergency_stop(False)

    # Clearing the latch must not itself touch actuator targets.
    assert {
        module_id: drive.posture_captures
        for module_id, drive in drives.items()
    } == capture_counts
    assert {
        module_id: len(drive.commands)
        for module_id, drive in drives.items()
    } == command_counts

    # A later fresh command may pass normally.
    router.apply(
        {
            "locomotor": SmoresCommand(
                linear_x_m_s=0.03,
            )
        }
    )

    assert drives["locomotor"].commands[-1].linear_x_m_s == 0.03


def test_pan_wheel_coactuation_final_targets_and_drive_gains():
    """Real router/reader/drive; fake only the external PhysX articulation."""
    import pytest
    from types import SimpleNamespace
    from smores_ep.config.geometry import SmoresGeometry
    from smores_ep.config.physics import SmoresActuatorConfig
    from smores_ep.isaac.dynamic_stage import ArticulationStateReader,DynamicDriveController

    class Articulation:
        def __init__(self):
            self.velocity=[0.]*4;self.position=[0.]*4
            self.stiffness=[0.]*4;self.damping=[0.]*4;self.events=[]
        def get_dof_positions(self):return SimpleNamespace(numpy=lambda:[[0.]*4])
        def get_dof_velocities(self):return SimpleNamespace(numpy=lambda:[self.velocity])
        def set_dof_gains(self,*,stiffnesses,dampings,dof_indices):
            for i,k,d in zip(dof_indices,stiffnesses,dampings):self.stiffness[i]=k;self.damping[i]=d
            self.events.append(('gains',tuple(self.stiffness)))
        def set_dof_position_targets(self,values,*,dof_indices):
            for i,v in zip(dof_indices,values):self.position[i]=v
            self.events.append(('position',tuple(self.position)))
        def set_dof_velocity_targets(self,values,*,dof_indices):
            for i,v in zip(dof_indices,values):self.velocity[i]=v
            self.events.append(('velocity',tuple(self.velocity)))
        def set_dof_max_efforts(self,values,*,dof_indices):pass
        def set_dof_max_velocities(self,values,*,dof_indices):pass

    art=Articulation();state=object.__new__(ArticulationStateReader)
    state._articulation=art;state._stage=None;state._wheel_contact_mode=None
    state._indices=dict(left_wheel=0,right_wheel=1,tilt=2,pan=3)
    state._position_targets=dict(tilt=0.,pan=0.);state._drive=SmoresActuatorConfig()
    drive=DynamicDriveController(state,SmoresGeometry(),12.)
    router=IsaacMultiModuleCommandRouter({'m':state},{'m':drive},SimpleNamespace(module_ids=('m',),connections=()))
    pan=SmoresCommand(internal_motion=InternalMotionMode.PAN_VELOCITY,pan_velocity_rad_s=1.)
    # Existing PAN-only support braking stays intact.
    router.apply({'m':pan});assert all(k>0 for k in art.stiffness[:2])
    from dataclasses import replace
    together=replace(pan,linear_x_m_s=.063)
    for command in (together,together,pan,together):
        router.apply({'m':command})
        assert art.velocity[3]==1.
        assert art.stiffness[3]==0.
        if command.linear_x_m_s:
            assert art.velocity[:2]==pytest.approx([2.,2.])
            # Nonzero velocities must not fight retained position brakes.
            assert art.stiffness[:2]==[0.,0.]
            assert art.damping[:2]==[state._drive.wheel_damping_nm_s_per_rad]*2
        else:
            assert art.velocity[:2]==[0.,0.]
            assert all(k>0 for k in art.stiffness[:2])
    router.set_emergency_stop(True);router.apply({'m':together})
    assert art.velocity[:2]==[0.,0.] and all(k>0 for k in art.stiffness[:2])
