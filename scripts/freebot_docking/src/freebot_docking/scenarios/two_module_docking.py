from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from freebot_docking.config.geometry import ShellGeometry
from freebot_docking.config.magnet import MagnetConfig
from freebot_docking.config.simulation import ShellContactFrictionConfig
from freebot_docking.physics.external_magnet import (
    ExternalMagneticInteraction,
    compute_external_magnetic_interaction,
)
from freebot_docking.physics.geometry import (
    compute_magnet_inner_shell_geometry,
    compute_shell_pair_geometry,
)
from freebot_docking.physics.state import (
    MagnetState,
    ShellState,
    Vector3,
    as_vector3,
)
from freebot_docking.physics.wrench import Wrench


class ContactRegime(str, Enum):
    """State of the reduced tangential shell contact."""

    FREE = "free"
    STICK = "stick"
    SLIP = "slip"


@dataclass(frozen=True)
class TangentialContactState:
    """Elastic displacement retained while a contact patch sticks."""

    displacement_world: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "displacement_world",
            as_vector3(self.displacement_world),
        )

    @classmethod
    def zero(cls) -> "TangentialContactState":
        return cls(np.zeros(3, dtype=np.float64))


@dataclass(frozen=True)
class ShellContactFrictionResult:
    """Tangential action-reaction pair at the shell contact patch."""

    regime: ContactRegime
    next_state: TangentialContactState
    signed_shell_gap_m: float
    contact_point_world: Vector3
    relative_tangent_velocity_world: Vector3
    normal_load_n: float
    static_friction_limit_n: float
    dynamic_friction_limit_n: float
    force_on_first_world: Vector3
    force_on_second_world: Vector3
    first_shell_wrench: Wrench
    second_shell_wrench: Wrench

    def __post_init__(self) -> None:
        for name in (
            "contact_point_world",
            "relative_tangent_velocity_world",
            "force_on_first_world",
            "force_on_second_world",
        ):
            object.__setattr__(self, name, as_vector3(getattr(self, name)))


def update_shell_contact_friction(
    first_state: ShellState,
    first_geometry: ShellGeometry,
    second_state: ShellState,
    second_geometry: ShellGeometry,
    normal_load_n: float,
    time_step_s: float,
    previous_state: TangentialContactState | None = None,
    config: ShellContactFrictionConfig | None = None,
) -> ShellContactFrictionResult:
    """Advance a tangential spring with a static/dynamic Coulomb cap.

    The normal collision constraint is deliberately not reproduced here:
    PhysX will resolve it from the actual colliders.  ``normal_load_n`` is the
    compressive load available to generate tangential friction.
    """

    load = float(normal_load_n)
    time_step = float(time_step_s)
    if not math.isfinite(load) or load < 0.0:
        raise ValueError("Normal contact load must be finite and non-negative")
    if not math.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("Time step must be finite and positive")

    parameters = ShellContactFrictionConfig() if config is None else config
    old_state = TangentialContactState.zero() if previous_state is None else previous_state
    pair = compute_shell_pair_geometry(
        first_state,
        first_geometry,
        second_state,
        second_geometry,
    )
    normal = pair.normal_first_to_second_world
    contact_point = 0.5 * (
        pair.point_on_first_world + pair.point_on_second_world
    )
    relative_velocity = (
        second_state.velocity_at(contact_point)
        - first_state.velocity_at(contact_point)
    )
    relative_tangent_velocity = (
        relative_velocity - np.dot(relative_velocity, normal) * normal
    )
    static_limit = parameters.static_friction_coefficient * load
    dynamic_limit = parameters.dynamic_friction_coefficient * load
    contact_active = (
        pair.signed_gap_m <= parameters.contact_tolerance_m
        and load > 0.0
    )

    if not contact_active:
        zero = np.zeros(3, dtype=np.float64)
        regime = ContactRegime.FREE
        next_state = TangentialContactState.zero()
        force_on_first = zero
    else:
        old_displacement = (
            old_state.displacement_world
            - np.dot(old_state.displacement_world, normal) * normal
        )
        trial_displacement = (
            old_displacement + time_step * relative_tangent_velocity
        )
        trial_force = (
            parameters.tangential_stiffness_n_per_m * trial_displacement
            + parameters.tangential_damping_n_s_per_m
            * relative_tangent_velocity
        )
        trial_norm = float(np.linalg.norm(trial_force))

        if trial_norm <= static_limit:
            regime = ContactRegime.STICK
            force_on_first = trial_force
            next_state = TangentialContactState(trial_displacement)
        else:
            regime = ContactRegime.SLIP
            slip_speed = float(np.linalg.norm(relative_tangent_velocity))
            if slip_speed > parameters.slip_speed_epsilon_m_per_s:
                friction_direction = relative_tangent_velocity / slip_speed
            elif trial_norm > 0.0:
                friction_direction = trial_force / trial_norm
            else:
                friction_direction = np.zeros(3, dtype=np.float64)

            force_on_first = dynamic_limit * friction_direction
            elastic_force = (
                force_on_first
                - parameters.tangential_damping_n_s_per_m
                * relative_tangent_velocity
            )
            next_state = TangentialContactState(
                elastic_force / parameters.tangential_stiffness_n_per_m
            )

    force_on_second = -force_on_first
    return ShellContactFrictionResult(
        regime=regime,
        next_state=next_state,
        signed_shell_gap_m=pair.signed_gap_m,
        contact_point_world=contact_point,
        relative_tangent_velocity_world=relative_tangent_velocity,
        normal_load_n=load,
        static_friction_limit_n=static_limit,
        dynamic_friction_limit_n=dynamic_limit,
        force_on_first_world=force_on_first,
        force_on_second_world=force_on_second,
        first_shell_wrench=Wrench.from_force_at_point(
            force_on_first,
            contact_point,
            first_state.com_world,
        ),
        second_shell_wrench=Wrench.from_force_at_point(
            force_on_second,
            contact_point,
            second_state.com_world,
        ),
    )


@dataclass(frozen=True)
class InternalMagneticPreloadInteraction:
    """Runtime force pair obtained from the precomputed internal FEM force."""

    preload_force_n: float
    interaction_point_world: Vector3
    force_on_carrier_world: Vector3
    force_on_shell_world: Vector3
    carrier_wrench: Wrench
    shell_wrench: Wrench

    def __post_init__(self) -> None:
        for name in (
            "interaction_point_world",
            "force_on_carrier_world",
            "force_on_shell_world",
        ):
            object.__setattr__(self, name, as_vector3(getattr(self, name)))


def compute_internal_magnetic_preload_interaction(
    shell_state: ShellState,
    shell_geometry: ShellGeometry,
    magnet_state: MagnetState,
    magnet_config: MagnetConfig,
    preload_force_n: float,
) -> InternalMagneticPreloadInteraction:
    """Turn the internal FEM result into a conservative runtime force pair."""

    preload = float(preload_force_n)
    if not math.isfinite(preload) or preload < 0.0:
        raise ValueError("Internal preload must be finite and non-negative")

    geometry = compute_magnet_inner_shell_geometry(
        shell_state,
        shell_geometry,
        magnet_state,
        magnet_config,
    )
    # The reduced FEM value supplies the force magnitude. Its direction follows
    # the local shell radius through the magnet face: unlike the previous
    # carrier-fixed axis force, this produces the restoring moment generated by
    # an off-radial magnet without adding an artificial angular spring.
    interaction_point = geometry.face_center_world
    radial = interaction_point - shell_state.center_world
    radial /= np.linalg.norm(radial)
    force_on_carrier = preload * radial
    force_on_shell = -force_on_carrier

    return InternalMagneticPreloadInteraction(
        preload_force_n=preload,
        interaction_point_world=interaction_point,
        force_on_carrier_world=force_on_carrier,
        force_on_shell_world=force_on_shell,
        carrier_wrench=Wrench.from_force_at_point(
            force_on_carrier,
            interaction_point,
            magnet_state.carrier_com_world,
        ),
        shell_wrench=Wrench.from_force_at_point(
            force_on_shell,
            interaction_point,
            shell_state.com_world,
        ),
    )


@dataclass(frozen=True)
class TwoModuleDockingState:
    """Persistent reduced state required by one docking contact."""

    shell_contact: TangentialContactState

    @classmethod
    def initial(cls) -> "TwoModuleDockingState":
        return cls(shell_contact=TangentialContactState.zero())


@dataclass(frozen=True)
class TwoModuleDockingResult:
    """All pre-Isaac interactions for an active and a passive module."""

    external_magnetic: ExternalMagneticInteraction
    internal_preload: InternalMagneticPreloadInteraction
    shell_contact: ShellContactFrictionResult
    active_carrier_wrench: Wrench
    active_shell_wrench: Wrench
    passive_shell_wrench: Wrench
    next_state: TwoModuleDockingState

    def total_wrench_at(self, reference_point: Vector3) -> Wrench:
        """Return the closed-system wrench residual at one world point."""

        return (
            self.active_carrier_wrench.expressed_at(reference_point)
            + self.active_shell_wrench.expressed_at(reference_point)
            + self.passive_shell_wrench.expressed_at(reference_point)
        )


def compute_two_module_docking_step(
    active_shell_state: ShellState,
    active_shell_geometry: ShellGeometry,
    passive_shell_state: ShellState,
    passive_shell_geometry: ShellGeometry,
    active_magnet_state: MagnetState,
    magnet_config: MagnetConfig,
    internal_preload_force_n: float,
    time_step_s: float,
    previous_state: TwoModuleDockingState | None = None,
    friction_config: ShellContactFrictionConfig | None = None,
    resolved_contact_normal_load_n: float | None = None,
) -> TwoModuleDockingResult:
    """Evaluate every pure-physics interaction needed before Isaac coupling.

    Before a contact solver is available, the parallel magnetic force is the
    estimate of contact preload.  Once PhysX is connected, its resolved total
    normal load replaces that estimate so the magnetic load is not counted
    twice.
    """

    resolved_load = None
    if resolved_contact_normal_load_n is not None:
        resolved_load = float(resolved_contact_normal_load_n)
        if not math.isfinite(resolved_load) or resolved_load < 0.0:
            raise ValueError(
                "Resolved contact normal load must be finite and non-negative"
            )

    state = TwoModuleDockingState.initial() if previous_state is None else previous_state
    external = compute_external_magnetic_interaction(
        active_shell_state,
        active_shell_geometry,
        passive_shell_state,
        passive_shell_geometry,
        active_magnet_state,
        magnet_config,
    )
    internal = compute_internal_magnetic_preload_interaction(
        active_shell_state,
        active_shell_geometry,
        active_magnet_state,
        magnet_config,
        internal_preload_force_n,
    )
    shell_contact = update_shell_contact_friction(
        active_shell_state,
        active_shell_geometry,
        passive_shell_state,
        passive_shell_geometry,
        normal_load_n=(
            external.parallel_force_n
            if resolved_load is None
            else resolved_load
        ),
        time_step_s=time_step_s,
        previous_state=state.shell_contact,
        config=friction_config,
    )

    active_carrier_wrench = (
        internal.carrier_wrench + external.active_carrier_wrench
    )
    active_shell_wrench = (
        internal.shell_wrench + shell_contact.first_shell_wrench
    )
    passive_shell_wrench = (
        external.passive_shell_wrench
        + shell_contact.second_shell_wrench
    )

    return TwoModuleDockingResult(
        external_magnetic=external,
        internal_preload=internal,
        shell_contact=shell_contact,
        active_carrier_wrench=active_carrier_wrench,
        active_shell_wrench=active_shell_wrench,
        passive_shell_wrench=passive_shell_wrench,
        next_state=TwoModuleDockingState(
            shell_contact=shell_contact.next_state
        ),
    )
