from __future__ import annotations

from dataclasses import dataclass
import math

from smores_ep.config.geometry import SmoresGeometry


SMORES_DOF_NO_LOAD_SPEED_RAD_S = 23.0 * 2.0 * math.pi / 60.0
SMORES_EP_CHARACTERISTIC_LENGTH_M = 0.080
SMORES_MAX_LAND_SPEED_BODY_LENGTHS_S = 1.1
SMORES_EP_MAX_LAND_SPEED_M_S = (
    SMORES_EP_CHARACTERISTIC_LENGTH_M
    * SMORES_MAX_LAND_SPEED_BODY_LENGTHS_S
)
SMORES_EP_MAX_WHEEL_SPEED_RAD_S = (
    SMORES_EP_MAX_LAND_SPEED_M_S / SmoresGeometry().wheel_radius_m
)


@dataclass(frozen=True)
class SmoresActuatorConfig:
    """Runtime PhysX drive limits and gains for one SMORES-EP module."""

    wheel_max_effort_nm: float = 1.2
    tilt_max_effort_nm: float = 2.3
    pan_max_effort_nm: float = 1.4
    wheel_damping_nm_s_per_rad: float = 0.18
    tilt_stiffness_nm_per_rad: float = 4.0
    tilt_damping_nm_s_per_rad: float = 0.20
    pan_damping_nm_s_per_rad: float = 0.58
    hold_stiffness_nm_per_rad: float = 24.0
    hold_damping_nm_s_per_rad: float = 1.2
    wheel_max_speed_rad_s: float = SMORES_EP_MAX_WHEEL_SPEED_RAD_S
    internal_max_speed_rad_s: float = SMORES_DOF_NO_LOAD_SPEED_RAD_S

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0
            for value in self.__dict__.values()
        ):
            raise ValueError(
                "Actuator efforts, gains and speed must be finite and positive"
            )

    @classmethod
    def payload_overdrive(
        cls,
        effort_scale: float = 4.0,
        wheel_max_speed_rad_s: float = SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
        tilt_effort_scale: float | None = 8.0,
    ) -> "SmoresActuatorConfig":
        """Return the intentionally exaggerated multi-module lift profile.

        The defaults deliberately let one TILT hinge lift the other seven
        Snake8 modules. ``tilt_effort_scale=None`` remains available when a
        caller explicitly wants TILT to use the common locomotion scale.
        """

        if not math.isfinite(effort_scale) or effort_scale <= 0.0:
            raise ValueError("Actuator effort scale must be positive")
        resolved_tilt_scale = (
            effort_scale
            if tilt_effort_scale is None
            else float(tilt_effort_scale)
        )
        if (
            not math.isfinite(resolved_tilt_scale)
            or resolved_tilt_scale <= 0.0
        ):
            raise ValueError("Tilt actuator effort scale must be positive")
        if (
            not math.isfinite(wheel_max_speed_rad_s)
            or wheel_max_speed_rad_s <= 0.0
        ):
            raise ValueError("Actuator maximum speed must be positive")
        return cls(
            wheel_max_effort_nm=1.2 * effort_scale,
            tilt_max_effort_nm=2.3 * resolved_tilt_scale,
            pan_max_effort_nm=1.4 * effort_scale,
            # Payload overdrive is required for lifted TILT chains, not for
            # rolling contact. Keep the nominal wheel damping so velocity
            # control stays compliant across the uneven support loads of an
            # assembled snake.
            wheel_damping_nm_s_per_rad=cls().wheel_damping_nm_s_per_rad,
            # The spring must also hold a cantilever close to its commanded
            # angle.  Effort alone only prevents torque saturation: the old
            # 12* gain left a reproducible 0.019 rad static error and could
            # deadlock a coordinated gait at its posture barrier.
            tilt_stiffness_nm_per_rad=max(
                24.0, 64.0 * resolved_tilt_scale
            ),
            tilt_damping_nm_s_per_rad=max(
                1.2, 1.60 * resolved_tilt_scale
            ),
            pan_damping_nm_s_per_rad=0.58 * effort_scale,
            hold_stiffness_nm_per_rad=max(
                24.0, 64.0 * resolved_tilt_scale
            ),
            hold_damping_nm_s_per_rad=max(
                1.2, 1.60 * resolved_tilt_scale
            ),
            wheel_max_speed_rad_s=wheel_max_speed_rad_s,
            internal_max_speed_rad_s=SMORES_DOF_NO_LOAD_SPEED_RAD_S,
        )


@dataclass(frozen=True)
class SmoresMassConfig:
    """Provisional link split constrained by the measured 0.454 kg total.

    The papers provide total module mass but not a per-link mass table. These
    values are therefore explicit engineering estimates, not paper claims.
    """

    body_kg: float = 0.314
    left_wheel_kg: float = 0.035
    right_wheel_kg: float = 0.035
    tilt_carrier_kg: float = 0.030
    pan_face_kg: float = 0.040
    # The reference states that the total center of mass is close to the
    # geometric center over the wheels, with nominally light skid contact.
    # The CAD does not include component densities, so this body-link COM is
    # an explicit balancing estimate that offsets the forward TOP assembly.
    body_com_body_m: tuple[float, float, float] = (-0.0080, 0.0, 0.0)

    def __post_init__(self) -> None:
        values = (
            self.body_kg,
            self.left_wheel_kg,
            self.right_wheel_kg,
            self.tilt_carrier_kg,
            self.pan_face_kg,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Every link mass must be finite and positive")
        if not all(math.isfinite(value) for value in self.body_com_body_m):
            raise ValueError("Body center of mass must be finite")

    @property
    def total_kg(self) -> float:
        return (
            self.body_kg
            + self.left_wheel_kg
            + self.right_wheel_kg
            + self.tilt_carrier_kg
            + self.pan_face_kg
        )

    def estimated_total_com_x_m(self, geometry: object) -> float:
        wheel_x = 0.5 * (
            geometry.left_wheel_center_body_m[0]
            + geometry.right_wheel_center_body_m[0]
        )
        moment = (
            self.body_kg * self.body_com_body_m[0]
            + (self.left_wheel_kg + self.right_wheel_kg) * wheel_x
            + self.tilt_carrier_kg
            * geometry.tilt_carrier_center_body_m[0]
            + self.pan_face_kg * geometry.pan_center_body_m[0]
        )
        return moment / self.total_kg


@dataclass(frozen=True)
class SmoresContactConfig:
    wheel_static_friction: float = 1.20
    wheel_dynamic_friction: float = 1.00
    body_static_friction: float = 0.15
    body_dynamic_friction: float = 0.12
    # The reference explicitly describes the passive rear edge as a
    # low-friction skid. Keep it separate from the chassis material.
    skid_static_friction: float = 0.03
    skid_dynamic_friction: float = 0.02
    # EP-face/module-on-table friction was experimentally measured
    # at about mu=0.15.  The TOP face can become a passive sliding
    # support during articulated Snake8 locomotion, so it must not
    # behave like a high-friction brake when it contacts the terrain.
    pan_static_friction: float = 0.15
    pan_dynamic_friction: float = 0.12

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Friction coefficients must be non-negative")
