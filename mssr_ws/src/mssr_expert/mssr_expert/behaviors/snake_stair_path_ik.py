"""Wheel-centre path generation and serial-chain IK for Snake8 stairs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from mssr_expert.behaviors.snake_stair_concertina_geometry import (
    ConcertinaStaircase,
)
from mssr_expert.behaviors.snake_stair_gait import SnakeStairGaitError


@dataclass(frozen=True)
class PathPoint:
    """One module centre in the stair longitudinal plane."""

    x_m: float
    z_m: float


@dataclass(frozen=True)
class WheelCenterPath:
    """Smooth, collider-aware centreline above a staircase.

    Each riser is replaced by a smooth rise followed by a smooth descent from
    a clearance apex.  At the physical top-front corner the module centre is
    one ``corner_clearance_radius_m`` away; on a tread it settles back to one
    wheel radius above the surface.  The curve is a graph z(x), so it cannot
    fold back into a vertical riser.
    """

    staircase: ConcertinaStaircase
    wheel_radius_m: float
    corner_clearance_radius_m: float
    approach_run_m: float
    landing_run_m: float
    transition_bias: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = (
            self.wheel_radius_m,
            self.corner_clearance_radius_m,
            self.approach_run_m,
            self.landing_run_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise SnakeStairGaitError(
                "Wheel-centre path dimensions must be positive and finite"
            )
        if self.corner_clearance_radius_m < self.wheel_radius_m:
            raise SnakeStairGaitError(
                "Corner clearance cannot be smaller than the wheel radius"
            )
        if self.approach_run_m + self.landing_run_m >= (
            self.staircase.tread_depth_m
        ):
            raise SnakeStairGaitError(
                "Wheel-centre smoothing consumes an entire stair tread"
            )
        object.__setattr__(
            self,
            "transition_bias",
            self._solve_transition_bias(),
        )
        self.validate_corner_clearance()

    @property
    def riser_edges_m(self) -> tuple[float, ...]:
        return tuple(
            self.staircase.first_riser_x_m
            + index * self.staircase.tread_depth_m
            for index in range(len(self.staircase.top_heights_m))
        )

    @staticmethod
    def _smootherstep(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value**3 * (value * (value * 6.0 - 15.0) + 10.0)

    @classmethod
    def _biased_smootherstep(cls, value: float, bias: float) -> float:
        """Return a C2 monotone transition with adjustable timing.

        ``bias < 1`` advances the rise while preserving zero endpoint slope
        and curvature.  Its reciprocal delays the landing descent.  This is
        used to clear a corner continuously instead of inserting the upper
        circular branch with a discontinuous hard ``max``.
        """

        forward = cls._smootherstep(value)
        reverse = cls._smootherstep(1.0 - value)
        denominator = forward + bias * reverse
        if denominator <= 0.0:
            return 1.0 if value >= 1.0 else 0.0
        return forward / denominator

    def height_m(self, x_m: float) -> float:
        """Return the desired module-centre height at world X."""

        if not math.isfinite(x_m):
            raise SnakeStairGaitError("Path query X must be finite")
        return self._height_with_bias(x_m, self.transition_bias)

    def _height_with_bias(self, x_m: float, bias: float) -> float:
        height = self.wheel_radius_m
        lower_top = 0.0
        apex_extra = (
            self.corner_clearance_radius_m - self.wheel_radius_m
        )
        for edge_x, upper_top in zip(
            self.riser_edges_m, self.staircase.top_heights_m
        ):
            rise = upper_top - lower_top
            lower_top = upper_top
            approach_phase = self._biased_smootherstep(
                (x_m - (edge_x - self.approach_run_m))
                / self.approach_run_m,
                bias,
            )
            if x_m <= edge_x:
                apex_phase = approach_phase
            else:
                apex_phase = 1.0 - self._biased_smootherstep(
                    (x_m - edge_x) / self.landing_run_m,
                    1.0 / bias,
                )
            height += rise * approach_phase + apex_extra * apex_phase
        return height

    def _solve_transition_bias(self) -> float:
        """Find the least aggressive smooth timing that clears all corners."""

        lower = 0.001
        upper = 1.0
        if not self._bias_clears_corners(lower):
            raise SnakeStairGaitError(
                "Stair geometry cannot satisfy smooth corner clearance"
            )
        if self._bias_clears_corners(upper):
            return upper
        for _ in range(36):
            middle = 0.5 * (lower + upper)
            if self._bias_clears_corners(middle):
                lower = middle
            else:
                upper = middle
        # Retain a small numerical margin between the sampled proof and the
        # continuous path evaluated by PhysX.
        return max(0.001, 0.995 * lower)

    def _bias_clears_corners(
        self,
        bias: float,
        samples_per_corner: int = 500,
    ) -> bool:
        radius = self.corner_clearance_radius_m
        for edge_x, top_z in zip(
            self.riser_edges_m,
            self.staircase.top_heights_m,
        ):
            for index in range(samples_per_corner + 1):
                x_m = edge_x - radius + (
                    2.0 * radius * index / samples_per_corner
                )
                distance = math.hypot(
                    x_m - edge_x,
                    self._height_with_bias(x_m, bias) - top_z,
                )
                if distance + 1.0e-9 < radius:
                    return False
        return True

    def support_height_m(self, x_m: float) -> float:
        """Return wheel-centre height for a tire resting on a flat tread."""

        if not math.isfinite(x_m):
            raise SnakeStairGaitError("Path query X must be finite")
        top_z = 0.0
        for edge_x, candidate_top_z in zip(
            self.riser_edges_m,
            self.staircase.top_heights_m,
        ):
            if x_m < edge_x:
                break
            top_z = candidate_top_z
        return top_z + self.wheel_radius_m

    def validate_corner_clearance(self, samples_per_corner: int = 400) -> None:
        """Prove the generated centreline clears every top-front corner."""

        radius = self.corner_clearance_radius_m
        for edge_x, top_z in zip(
            self.riser_edges_m, self.staircase.top_heights_m
        ):
            for index in range(samples_per_corner + 1):
                x_m = edge_x - radius + (
                    2.0 * radius * index / samples_per_corner
                )
                distance = math.hypot(
                    x_m - edge_x, self.height_m(x_m) - top_z
                )
                if distance + 1.0e-9 < radius:
                    raise SnakeStairGaitError(
                        "Smoothed wheel-centre curve violates corner "
                        f"clearance at riser x={edge_x:.3f} m"
                    )

    def sample_module_centers(
        self,
        *,
        head_x_m: float,
        module_count: int,
        link_length_m: float,
    ) -> tuple[PathPoint, ...]:
        """Place rigid module centres tail-to-head on the curve.

        Consecutive points are separated by the true Euclidean link length,
        not merely by arc length or by their horizontal projection.
        """

        if module_count < 2:
            raise SnakeStairGaitError("Path IK needs at least two modules")
        if not math.isfinite(link_length_m) or link_length_m <= 0.0:
            raise SnakeStairGaitError("Snake8 link length must be positive")

        reverse_points = [PathPoint(head_x_m, self.height_m(head_x_m))]
        for _ in range(module_count - 1):
            front = reverse_points[-1]
            lower = front.x_m - 2.0 * link_length_m
            while self._distance_to(front, lower) < link_length_m:
                lower -= link_length_m
            upper = front.x_m
            for _ in range(70):
                middle = 0.5 * (lower + upper)
                if self._distance_to(front, middle) > link_length_m:
                    lower = middle
                else:
                    upper = middle
            x_m = 0.5 * (lower + upper)
            reverse_points.append(PathPoint(x_m, self.height_m(x_m)))
        return tuple(reversed(reverse_points))

    def _distance_to(self, front: PathPoint, x_m: float) -> float:
        return math.hypot(
            front.x_m - x_m, front.z_m - self.height_m(x_m)
        )

    def head_x_for_tail_x(
        self,
        *,
        tail_x_m: float,
        module_count: int,
        link_length_m: float,
    ) -> float:
        """Solve the head coordinate that lands the tail at ``tail_x_m``."""

        lower = tail_x_m
        upper = tail_x_m + 2.0 * module_count * link_length_m
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            points = self.sample_module_centers(
                head_x_m=middle,
                module_count=module_count,
                link_length_m=link_length_m,
            )
            if points[0].x_m < tail_x_m:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)


def relative_tilt_ik(points: Sequence[PathPoint]) -> tuple[float, ...]:
    """Convert centre points into relative SMORES TILT coordinates.

    For link pitches ``phi``, the serial-chain convention is
    q[0] = phi[0] and q[i] = phi[i] - phi[i-1].  The eighth TILT does not
    connect another module and is therefore held at captured neutral.
    """

    if len(points) < 2:
        raise SnakeStairGaitError("Path IK needs at least two points")
    pitches = tuple(
        math.atan2(
            upper.z_m - lower.z_m,
            upper.x_m - lower.x_m,
        )
        for lower, upper in zip(points, points[1:])
    )
    if any(
        points[index + 1].x_m <= points[index].x_m
        for index in range(len(points) - 1)
    ):
        raise SnakeStairGaitError("Wheel-centre path folded backwards")
    joints = [pitches[0]]
    joints.extend(
        _wrap_pi(pitches[index] - pitches[index - 1])
        for index in range(1, len(pitches))
    )
    joints.append(0.0)
    return tuple(joints)


def reconstruct_centers(
    start: PathPoint,
    relative_tilts_rad: Sequence[float],
    link_length_m: float,
) -> tuple[PathPoint, ...]:
    """Forward-kinematics helper used to verify the inverse solution."""

    points = [start]
    pitch = 0.0
    for relative in relative_tilts_rad[:-1]:
        pitch += relative
        previous = points[-1]
        points.append(
            PathPoint(
                previous.x_m + link_length_m * math.cos(pitch),
                previous.z_m + link_length_m * math.sin(pitch),
            )
        )
    return tuple(points)


def _wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
