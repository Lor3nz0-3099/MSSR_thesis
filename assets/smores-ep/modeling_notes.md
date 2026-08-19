# SMORES-EP modeling notes

## Evidence used

The mechanical model is based on these local references:

- `references/SMORES-EP.pdf`
- `references/Design of the SMORES system.pdf`
- `references/design and characterization of the EP-Face Connector.pdf`

`references/A novel docking system.pdf` describes a different active/passive
module architecture and is not used to define SMORES-EP joints. Likewise,
`references/assemblies.pdf` concerns abstract learned self-assembling limbs,
not the SMORES-EP hardware.

## Reference-backed mechanism

SMORES-EP has four active rotational degrees of freedom:

- LEFT wheel/face: continuous rotation;
- RIGHT wheel/face: continuous rotation;
- PAN/TOP face: continuous rotation;
- TILT carrier: limited to plus or minus 90 degrees.

LEFT, RIGHT, and TILT have parallel, coincident axes. PAN is perpendicular to
that common axis. LEFT and RIGHT are also the differential-drive wheels.

The two internal pan/tilt motors do not drive two unrelated joints directly.
Their four identical 9-tooth pinions drive 48-tooth spur gears and a crown
gear differential:

- opposite internal motor rotations produce pan;
- concordant internal motor rotations produce tilt and combine motor torque.

The kinematic control package therefore exposes pan and tilt as output
coordinates while retaining the explicit motor mixing equations:

```text
motor_a = tilt + pan
motor_b = tilt - pan
```

The output-to-motor ratio remains configurable because the papers do not give
enough information to infer the complete effective ratio from the converted
CAD alone.

## Docking interfaces

The four and only four module faces are:

- `LEFT`: carried by the left wheel link;
- `RIGHT`: carried by the right wheel link;
- `TOP`: carried by the pan link and moved by both pan and tilt;
- `BOTTOM`: the passive, non-rotating `base_chassis` face.

The EP version uses four electro-permanent magnets per face. The cited
connector paper reports:

- 80 mm characteristic module length;
- 81 mm magnet-to-magnet module length;
- 0.454 kg module mass in its strength calculation;
- mean face normal holding force of 88.4 N;
- approximately 4 mm normal and 7 mm in-plane magnetic capture;
- tilt-independent four-fold connector symmetry.

Docking forces are deliberately not implemented in the first kinematic
scenario. The generated physics USD does include the four attachment frames
so magnetic capture can be added without changing link topology.

## CAD-to-runtime mapping

Imported source:

```text
assets/smores-ep/usd_visual/smores_ep_usd_visual_v1.usd
```

The visual is metric and Z-up, but its body axes do not follow ROS. Runtime
code maps:

```text
CAD +Y -> ROS +X (forward)
CAD -X -> ROS +Y (left)
CAD +Z -> ROS +Z (up)
```

The mean wheel axis is the module-frame origin. Measured CAD centers place the
wheel centers at approximately `Y = +/-35.1 mm`, giving a track width of
70.410 mm. The TOP/pan center is approximately 33.6 mm forward of the tilt
axis.

Part groups:

- body: base chassis, side chassis, and motor housings;
- left wheel: `smores_wheel1` and outer spur gear 4;
- right wheel: `smores_wheel2` and outer spur gear 2;
- wheel-drive pinions: pinion 2 on the left and pinion 4 on the right; these
  form the outer diagonal and rotate oppositely to their 48-tooth gears by
  the ratio 48:9;
- pan/tilt differential: inner spur gears 1 and 3, driven by pinions 1 and 3
  on the other diagonal;
- tilt carrier: `chassis_up21`;
- pan link: crown gear and `smores_wheel3`.

## What is measured and what is provisional

Measured/reference-backed:

- CAD scale, axes, centers, part grouping;
- identical 62 mm driving-wheel diameter, measured from the shared CAD
  prototype's transformed vertices;
- total mass of 0.454 kg;
- four output DoFs and tilt limits;
- four docking faces;
- nominal module dimensions.

Provisional engineering parameters:

- per-link mass distribution;
- simplified convex collision topology around the measured CAD envelopes;
- friction coefficients, including the low-friction rear-skid estimate;
- drive stiffness and damping.

These provisional values are isolated in configuration modules and must not
be presented as measurements from the papers.
