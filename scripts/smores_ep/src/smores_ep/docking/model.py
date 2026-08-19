from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal


Vector3 = tuple[float, float, float]
DockingAction = Literal["attach", "detach"]
FACE_NAMES = frozenset({"LEFT", "RIGHT", "TOP", "BOTTOM"})


def normalize_face_name(face_name: str) -> str:
    """Return the canonical name of one of the four SMORES-EP connectors."""

    normalized_name = face_name.upper().replace("-", "_")
    if normalized_name == "BASE_CHASSIS":
        normalized_name = "BOTTOM"
    if normalized_name not in FACE_NAMES:
        raise ValueError(f"Unknown SMORES-EP docking face: {face_name}")
    return normalized_name


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(a * b for a, b in zip(first, second))


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(first, second))  # type: ignore[return-value]


def _scale(vector: Vector3, factor: float) -> Vector3:
    return tuple(factor * value for value in vector)  # type: ignore[return-value]


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _normalized(vector: Vector3) -> Vector3:
    magnitude = _norm(vector)
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("Docking-frame axes must be finite and non-zero")
    return _scale(vector, 1.0 / magnitude)


@dataclass(frozen=True)
class DockingThresholds:
    """Geometric contact gate used before energizing an EP-Face."""

    # Some non-axial USD docking markers are construction frames rather than
    # rendered contact planes. Their separation can remain around 5-6 mm when
    # the loaded CAD/collision surfaces touch, hence this generic 6.7 mm gate.
    normal_contact_tolerance_m: float = 0.0067

    # TOP and BOTTOM markers lie close to their rendered outer planes, but the
    # CAD collision can stop a correctly parked, loaded module at about
    # 2.41 mm.  This
    # gate therefore covers marker-normal separation only.  Planar centring is
    # checked independently by the alignment controller (1.5 mm in the shared
    # execution policy), so accepting physical contact here does not relax
    # visible alignment.
    top_bottom_contact_tolerance_m: float = 0.0025

    # Parallel Snake7 docking moves the shared center module when the first
    # fixed joint settles.  The second TOP<->BOTTOM contact was measured at
    # 9.16 mm of 3-D marker offset, including 4.4 mm of vertical CAD settling,
    # while its normal gap was 0.48 mm and its normal error only 1.7 degrees.
    # Ten millimetres accepts that physical overlap but still rejects the
    # previously observed bad 11.61 mm docking pose.  The pair-specific normal
    # gate remains much tighter and must independently be satisfied.
    lateral_offset_tolerance_m: float = 0.0100
    normal_alignment_tolerance_rad: float = math.radians(8.0)
    clocking_tolerance_rad: float = math.radians(10.0)

    def __post_init__(self) -> None:
        values = (
            self.normal_contact_tolerance_m,
            self.top_bottom_contact_tolerance_m,
            self.lateral_offset_tolerance_m,
            self.normal_alignment_tolerance_rad,
            self.clocking_tolerance_rad,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Docking thresholds must be finite and non-negative")


@dataclass(frozen=True)
class DockingCommand:
    action: DockingAction
    first_module: str
    second_module: str
    first_face: str | None = None
    second_face: str | None = None

    def __post_init__(self) -> None:
        if not self.first_module or not self.second_module:
            raise ValueError("Docking module IDs cannot be empty")
        if self.first_module == self.second_module:
            raise ValueError("A module cannot dock with itself")
        if (self.first_face is None) != (self.second_face is None):
            raise ValueError("Docking commands must specify both faces or neither")
        if self.first_face is not None:
            object.__setattr__(
                self,
                "first_face",
                normalize_face_name(self.first_face),
            )
            object.__setattr__(
                self,
                "second_face",
                normalize_face_name(self.second_face or ""),
            )

    @classmethod
    def parse(cls, text: str) -> "DockingCommand":
        fields = text.split()
        if len(fields) not in {3, 5}:
            raise ValueError(
                "Expected 'attach <module_a> <module_b>' or "
                "'attach <module_a> <face_a> <module_b> <face_b>' "
                "(and the equivalent detach command)"
            )
        action = fields[0].lower()
        if action not in {"attach", "detach"}:
            raise ValueError("Docking action must be 'attach' or 'detach'")
        if len(fields) == 3:
            first, second = fields[1:]
            return cls(action, first, second)  # type: ignore[arg-type]
        first, first_face, second, second_face = fields[1:]
        return cls(  # type: ignore[arg-type]
            action,
            first,
            second,
            first_face,
            second_face,
        )

    @property
    def unordered_module_pair(self) -> frozenset[str]:
        return frozenset((self.first_module, self.second_module))

    @property
    def explicit_faces(self) -> bool:
        return self.first_face is not None

    @property
    def face_keys(self) -> frozenset[tuple[str, str]]:
        if self.first_face is None or self.second_face is None:
            return frozenset()
        return frozenset(
            (
                (self.first_module, self.first_face),
                (self.second_module, self.second_face),
            )
        )


@dataclass(frozen=True)
class DockingFace:
    module_id: str
    face_name: str
    frame_path: str
    rigid_body_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "face_name",
            normalize_face_name(self.face_name),
        )

    @property
    def key(self) -> tuple[str, str]:
        return self.module_id, self.face_name


@dataclass(frozen=True)
class DockingFacePose:
    face: DockingFace
    position_world_m: Vector3
    outward_normal_world: Vector3
    tangent_world: Vector3

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for vector in (
                self.position_world_m,
                self.outward_normal_world,
                self.tangent_world,
            )
            for value in vector
        ):
            raise ValueError("Docking face pose must be finite")
        normal = _normalized(self.outward_normal_world)
        tangent = _subtract(
            self.tangent_world,
            _scale(normal, _dot(self.tangent_world, normal)),
        )
        object.__setattr__(self, "outward_normal_world", normal)
        object.__setattr__(self, "tangent_world", _normalized(tangent))


@dataclass(frozen=True)
class DockingPairEvaluation:
    first: DockingFacePose
    second: DockingFacePose
    normal_separation_m: float
    lateral_offset_m: float
    normal_misalignment_rad: float
    clocking_residual_rad: float
    clocking_error_rad: float
    clocking_quarter_turns: int
    eligible: bool
    score: float


def evaluate_face_pair(
    first: DockingFacePose,
    second: DockingFacePose,
    thresholds: DockingThresholds | None = None,
) -> DockingPairEvaluation:
    """Measure whether two square EP-Faces are in an attachable contact."""

    limits = thresholds or DockingThresholds()
    delta = _subtract(second.position_world_m, first.position_world_m)
    axial = _dot(delta, first.outward_normal_world)
    lateral = _subtract(
        delta,
        _scale(first.outward_normal_world, axial),
    )
    normal_separation = abs(axial)
    lateral_offset = _norm(lateral)
    opposed_cosine = max(
        -1.0,
        min(
            1.0,
            -_dot(
                first.outward_normal_world,
                second.outward_normal_world,
            ),
        ),
    )
    normal_misalignment = math.acos(opposed_cosine)

    tangent_cross = _cross(first.tangent_world, second.tangent_world)
    raw_clocking = math.atan2(
        _dot(first.outward_normal_world, tangent_cross),
        _dot(first.tangent_world, second.tangent_world),
    )
    quarter_turn = 0.5 * math.pi
    clocking_quarter_turns = int(round(raw_clocking / quarter_turn)) % 4
    clocking_residual = (
        (raw_clocking + 0.25 * math.pi) % quarter_turn
        - 0.25 * math.pi
    )
    clocking_error = abs(clocking_residual)

    # LEFT and RIGHT are continuously rotating connector disks.  Their
    # clocking is therefore not a configuration constraint when mating with
    # the non-rotating BOTTOM face (SMORES-EP, Sec. III-A).  Keeping the
    # generic square-array gate here made a correctly positioned wheel back
    # away and retry solely because the target disk had rolled to an arbitrary
    # angle during the earlier assembly wave.  BOTTOM-to-BOTTOM and all other
    # pairs retain the explicit clocking gate.
    face_names = frozenset((first.face.face_name, second.face.face_name))
    top_bottom_pair = face_names == frozenset(("TOP", "BOTTOM"))
    bottom_lateral_pair = face_names in (
        frozenset(("BOTTOM", "LEFT")),
        frozenset(("BOTTOM", "RIGHT")),
    )
    clocking_eligible = (
        bottom_lateral_pair
        or clocking_error <= limits.clocking_tolerance_rad
    )
    normal_contact_tolerance_m = (
        limits.top_bottom_contact_tolerance_m
        if top_bottom_pair
        else limits.normal_contact_tolerance_m
    )

    eligible = (
        normal_separation <= normal_contact_tolerance_m
        and lateral_offset <= limits.lateral_offset_tolerance_m
        and normal_misalignment <= limits.normal_alignment_tolerance_rad
        and clocking_eligible
    )

    def normalized(value: float, limit: float) -> float:
        if limit <= 0.0:
            return 0.0 if value <= 0.0 else math.inf
        return value / limit

    score = (
        normalized(
            normal_separation,
            normal_contact_tolerance_m,
        )
        + normalized(
            lateral_offset,
            limits.lateral_offset_tolerance_m,
        )
        + normalized(
            normal_misalignment,
            limits.normal_alignment_tolerance_rad,
        )
        + (
            0.0
            if bottom_lateral_pair
            else normalized(
                clocking_error,
                limits.clocking_tolerance_rad,
            )
        )
    )
    return DockingPairEvaluation(
        first=first,
        second=second,
        normal_separation_m=normal_separation,
        lateral_offset_m=lateral_offset,
        normal_misalignment_rad=normal_misalignment,
        clocking_residual_rad=clocking_residual,
        clocking_error_rad=clocking_error,
        clocking_quarter_turns=clocking_quarter_turns,
        eligible=eligible,
        score=score,
    )


def select_best_face_pair(
    first_faces: Iterable[DockingFacePose],
    second_faces: Iterable[DockingFacePose],
    occupied_faces: Iterable[tuple[str, str]] = (),
    thresholds: DockingThresholds | None = None,
) -> DockingPairEvaluation | None:
    occupied = set(occupied_faces)
    candidates = (
        evaluate_face_pair(first, second, thresholds)
        for first in first_faces
        for second in second_faces
        if first.face.key not in occupied and second.face.key not in occupied
    )
    eligible = [candidate for candidate in candidates if candidate.eligible]
    return min(eligible, key=lambda candidate: candidate.score, default=None)
