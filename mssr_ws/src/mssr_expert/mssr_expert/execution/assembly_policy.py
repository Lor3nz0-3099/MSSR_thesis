"""Morphology-agnostic execution policy for SMORES assembly pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable


TARGET_EXECUTION_POLICY_FIELDS = frozenset(
    {
        "align_timeout_s",
        "dock_timeout_s",
        "align_retry_count",
        "dock_recovery_count",
        "contact_quality_planar_tolerance_m",
        "contact_quality_retry_count",
        "top_bottom_contact_tolerance_m",
        "contact_approach_feedback",
        "max_concurrent_alignments_per_wave",
        "snap_docking_faces_to_nominal",
    }
)


@dataclass(frozen=True)
class AssemblyExecutionPolicy:
    """Backend policy shared by self-assembly and self-reconfiguration.

    Target graphs describe topology and final posture.  They deliberately do
    not select retry, contact, or concurrency behavior: those properties must
    remain identical when the same topology is assembled from scratch or
    reached through self-reconfiguration.
    """

    align_timeout_s: float = 60.0
    dock_timeout_s: float = 10.0
    align_retry_count: int = 2
    dock_recovery_count: int = 2
    contact_quality_planar_tolerance_m: float = 0.0015
    contact_quality_retry_count: int = 2
    top_bottom_contact_tolerance_m: float = 0.004
    contact_approach_feedback: bool = True
    # Zero means that every action admitted by the paper's depth/root wave may
    # run in parallel. Safety comes from collective REACH -> ALIGN -> APPROACH
    # barriers, not from serializing otherwise independent modules.
    max_concurrent_alignments_per_wave: int = 0
    snap_docking_faces_to_nominal: bool = False

    def __post_init__(self) -> None:
        positive_finite = (
            self.align_timeout_s,
            self.dock_timeout_s,
            self.contact_quality_planar_tolerance_m,
            self.top_bottom_contact_tolerance_m,
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in positive_finite
        ):
            raise ValueError(
                "Assembly timeouts and contact tolerances must be positive "
                "and finite."
            )
        for field_name in (
            "align_retry_count",
            "dock_recovery_count",
            "contact_quality_retry_count",
            "max_concurrent_alignments_per_wave",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer."
                )

    def executor_kwargs(self) -> dict[str, Any]:
        """Return the common arguments accepted by the assembly executor."""

        return {
            "align_timeout_s": self.align_timeout_s,
            "dock_timeout_s": self.dock_timeout_s,
            "align_retry_count": self.align_retry_count,
            "dock_recovery_count": self.dock_recovery_count,
            "contact_quality_planar_tolerance_m": (
                self.contact_quality_planar_tolerance_m
            ),
            "contact_quality_retry_count": self.contact_quality_retry_count,
            "top_bottom_contact_tolerance_m": (
                self.top_bottom_contact_tolerance_m
            ),
            "contact_approach_feedback": self.contact_approach_feedback,
            "max_concurrent_alignments_per_wave": (
                self.max_concurrent_alignments_per_wave
            ),
            "snap_docking_faces_to_nominal": (
                self.snap_docking_faces_to_nominal
            ),
        }

    @classmethod
    def from_parameter_getter(
        cls,
        get_parameter: Callable[[str], Any],
    ) -> "AssemblyExecutionPolicy":
        """Build the policy from a ROS-like ``get_parameter`` callable."""

        def value(name: str) -> Any:
            return get_parameter(name).value

        return cls(
            align_timeout_s=float(value("align_timeout_s")),
            dock_timeout_s=float(value("dock_timeout_s")),
            align_retry_count=int(value("align_retry_count")),
            dock_recovery_count=int(value("dock_recovery_count")),
            contact_quality_planar_tolerance_m=float(
                value("contact_quality_planar_tolerance_m")
            ),
            contact_quality_retry_count=int(
                value("contact_quality_retry_count")
            ),
            top_bottom_contact_tolerance_m=float(
                value("top_bottom_contact_tolerance_m")
            ),
            contact_approach_feedback=bool(
                value("contact_approach_feedback")
            ),
            max_concurrent_alignments_per_wave=int(
                value("max_concurrent_alignments_per_wave")
            ),
            snap_docking_faces_to_nominal=bool(
                value("snap_docking_faces_to_nominal")
            ),
        )


DEFAULT_ASSEMBLY_EXECUTION_POLICY = AssemblyExecutionPolicy()
