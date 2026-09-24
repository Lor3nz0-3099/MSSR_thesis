"""World-X experts must not silently interpret local composite landmarks."""
import pytest

from mssr_expert.behaviors.snake_gap_gait import FlatGap, SnakeGapGaitError
from mssr_expert.behaviors.snake_stair_concertina_geometry import (
    UniformStaircase, SnakeStairGaitError,
)


def test_gap_macro_rejects_local_landmarks():
    with pytest.raises(SnakeGapGaitError, match="local"):
        FlatGap.from_course({"frame_id": "world", "gap": {
            "coordinate_frame": "stage_local", "near_edge_x_m": .5,
            "far_edge_x_m": .7}})


def test_stairs_macro_rejects_local_landmarks():
    with pytest.raises(SnakeStairGaitError, match="local"):
        UniformStaircase.from_course({"frame_id": "world", "stairs": {
            "coordinate_frame": "stage_local", "first_riser_x_m": .5,
            "riser_depth_m": .28, "top_heights_m": [.05, .1]}})
