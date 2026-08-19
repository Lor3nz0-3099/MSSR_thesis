from __future__ import annotations

from typing import Any

import numpy as np

from freebot_docking.physics.wrench import Wrench


def apply_wrench(body: Any, wrench: Wrench) -> None:
    """Apply a world-space wrench at its reference point to one rigid body."""

    body.apply_forces_and_torques_at_pos(
        forces=np.asarray(wrench.force, dtype=np.float64).reshape(1, 3),
        torques=np.asarray(wrench.torque, dtype=np.float64).reshape(1, 3),
        positions=np.asarray(
            wrench.reference_point,
            dtype=np.float64,
        ).reshape(1, 3),
        local_frame=False,
    )


def apply_action_reaction_pair(
    first_body: Any,
    first_wrench: Wrench,
    second_body: Any,
    second_wrench: Wrench,
    apply_second: bool = True,
) -> None:
    """Apply a previously validated pair, optionally skipping a fixed body."""

    residual = (
        first_wrench.expressed_at([0.0, 0.0, 0.0])
        + second_wrench.expressed_at([0.0, 0.0, 0.0])
    )
    if (
        np.linalg.norm(residual.force) > 1.0e-9
        or np.linalg.norm(residual.torque) > 1.0e-9
    ):
        raise RuntimeError("Action-reaction wrench pair is not conservative")

    apply_wrench(first_body, first_wrench)
    if apply_second:
        apply_wrench(second_body, second_wrench)
