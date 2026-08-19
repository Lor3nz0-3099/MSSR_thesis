# FreeBOT PhysX Modeling Notes

## Correct physical interpretation

The imported CAD is not a normal differential-drive robot. The shell is part of the locomotion mechanism.

Correct topology:

- `shell_link`: dynamic spherical shell/body that rolls on the floor.
- `internal_link`: internal chassis, motors, magnet support, caster supports.
- `left_wheel_link` and `right_wheel_link`: wheels rotating with respect to `internal_link`.
- caster balls: passive balls carried by brackets/arms fixed to the central
  internal chassis. The current four-point-support experiment places them at
  zero nominal clearance from the inner shell. They remain rigid, passive and
  very low friction; no adhesive or spring constraint keeps them attached.
- `magnet_frame`: functional attachment point fixed to the internal mechanism and magnetically coupled to the inner surface of the shell.

The robot moves because the internal wheels rotate against the inner shell surface and shift/apply forces to the shell. The shell then rolls on the external floor.

The caster support geometry must be grouped with `internal_link`. The caster balls themselves may be modeled as passive spherical links attached to the internal chassis with spherical joints, but their centers and supports are determined by the CAD arms that extend from the central chassis toward the inner shell surface.

## Current working assets

The active source assets are:

- visual USD: `assets/freebot/usd_visual/freebot_visual_nearer_wheels.usd`
- physics USD: `assets/freebot/usd_physics/freebot_cad_full_nearer_wheels_rigid.usd`
- physics generator: `scripts/isaac_freebot/create_freebot_cad_full_v2.py`
- runtime test script: `scripts/isaac_freebot/run_freebot_magnetic_locomotion.py`

At stage build time the nominal loaded 32 mm tires are placed at zero analytic
clearance from the fitted inner sphere. The collision proxy adds 0.9 mm to the
radius to represent the unloaded rubber envelope and uses a force-based PhysX
compliant contact (`k=8000 N/m`, `c=40 N s/m`). Proxy overlap is therefore tire
deflection. Both caster proxies are placed at zero nominal clearance for the
four-point-support test. Joint anchors move with their rigid bodies; no
compliant wheel mount is introduced.

## Main PhysX challenge

PhysX primitive sphere colliders are solid external colliders; they do not provide a hollow inner collision surface. A dynamic concave shell mesh is also not a robust default choice.

The current model keeps the CAD visual geometry but uses authored physics links,
SDF collision for the shell and tire contact, and runtime force models for the
magnet-shell and magnet-wall effects.

The main open validation issue is the transfer of tangential force from the
drive wheels to the inner shell. If the CAD/SDF tire-shell contact remains too
weak, the next robust path is to keep CAD for visuals and author calibrated
physics-only colliders for the shell interior and wheel tires.
