from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class IsaacMaterialConfig:
    """PhysX coefficients used by the hybrid contact model.

    When the explicit shell patch is active, the runtime temporarily sets the
    shell coefficient to zero to disable only that duplicate path.  ``max``
    combination preserves wheel, caster and ground friction from the other
    material in each pair.
    """

    shell_static_friction: float = 1.10
    shell_dynamic_friction: float = 1.00
    wheel_static_friction: float = 2.20
    wheel_dynamic_friction: float = 1.90
    wheel_contact_stiffness_n_per_m: float = 8_000.0
    wheel_contact_damping_n_s_per_m: float = 40.0
    caster_static_friction: float = 0.03
    caster_dynamic_friction: float = 0.02
    caster_contact_stiffness_n_per_m: float = 2_000.0
    caster_contact_damping_n_s_per_m: float = 15.0
    # FreeBOT Eq. (7) gives mu_2 ~= 1.34--1.43 in the observed near-docking
    # poses.  A static value of 1.25 keeps the exactly aligned docking pose
    # stable (mu_2(0) ~= 1.21), but lets the shell leave static equilibrium
    # once the carrier starts to climb.  Dynamic friction is lower, as
    # required by the Coulomb material model.
    ground_static_friction: float = 1.25
    ground_dynamic_friction: float = 1.00

    def __post_init__(self) -> None:
        values = (
            self.shell_static_friction,
            self.shell_dynamic_friction,
            self.wheel_static_friction,
            self.wheel_dynamic_friction,
            self.wheel_contact_stiffness_n_per_m,
            self.wheel_contact_damping_n_s_per_m,
            self.caster_static_friction,
            self.caster_dynamic_friction,
            self.caster_contact_stiffness_n_per_m,
            self.caster_contact_damping_n_s_per_m,
            self.ground_static_friction,
            self.ground_dynamic_friction,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Material parameters must be finite")
        if any(float(value) < 0.0 for value in values):
            raise ValueError("Material parameters must be non-negative")
        for static, dynamic in (
            (self.shell_static_friction, self.shell_dynamic_friction),
            (self.wheel_static_friction, self.wheel_dynamic_friction),
            (self.caster_static_friction, self.caster_dynamic_friction),
            (self.ground_static_friction, self.ground_dynamic_friction),
        ):
            if dynamic > static:
                raise ValueError("Dynamic friction cannot exceed static friction")


def _create_material(
    stage: Any,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    combine_mode: str,
    compliant_stiffness_n_per_m: float = 0.0,
    compliant_damping_n_s_per_m: float = 0.0,
) -> Any:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(static_friction))
    physics.CreateDynamicFrictionAttr(float(dynamic_friction))
    physics.CreateRestitutionAttr(0.0)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.GetFrictionCombineModeAttr().Set(combine_mode)
    physx.GetRestitutionCombineModeAttr().Set("min")
    if compliant_stiffness_n_per_m > 0.0:
        # Force-based implicit spring: overlap of an enlarged wheel/caster
        # envelope represents local contact compliance.  The rigid CAD mount
        # does not acquire an artificial degree of freedom.
        physx.CreateCompliantContactAccelerationSpringAttr().Set(False)
        physx.CreateCompliantContactStiffnessAttr().Set(
            float(compliant_stiffness_n_per_m)
        )
        physx.CreateCompliantContactDampingAttr().Set(
            float(compliant_damping_n_s_per_m)
        )
        physx.GetDampingCombineModeAttr().Set("max")
    return material


def assign_freebot_materials(
    stage: Any,
    module_roots: tuple[str, ...],
    config: IsaacMaterialConfig | None = None,
) -> Any:
    """Bind shell, wheel and caster materials to every collider."""

    from pxr import Usd, UsdPhysics, UsdShade

    parameters = IsaacMaterialConfig() if config is None else config
    shell = _create_material(
        stage,
        "/World/materials/freebot_shell",
        parameters.shell_static_friction,
        parameters.shell_dynamic_friction,
        "max",
    )
    wheel = _create_material(
        stage,
        "/World/materials/freebot_wheel",
        parameters.wheel_static_friction,
        parameters.wheel_dynamic_friction,
        "max",
        parameters.wheel_contact_stiffness_n_per_m,
        parameters.wheel_contact_damping_n_s_per_m,
    )
    caster = _create_material(
        stage,
        "/World/materials/freebot_caster",
        parameters.caster_static_friction,
        parameters.caster_dynamic_friction,
        "max",
        parameters.caster_contact_stiffness_n_per_m,
        parameters.caster_contact_damping_n_s_per_m,
    )

    for root in module_roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path = str(prim.GetPath())
            selected = None
            if "/shell_link" in path:
                selected = shell
            if "left_wheel_link" in path or "right_wheel_link" in path:
                selected = wheel
            if "caster_1_ball_link" in path or "caster_2_ball_link" in path:
                selected = caster
            if selected is not None:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    selected,
                    UsdShade.Tokens.strongerThanDescendants,
                )
    return _create_material(
        stage,
        "/World/materials/freebot_ground",
        parameters.ground_static_friction,
        parameters.ground_dynamic_friction,
        "max",
    )
