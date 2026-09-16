from pathlib import Path
import json


def test_seed5100_cone03_is_shifted_5cm_away_from_centerline():
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "rc_car_seed5100_layout.json"
    )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    cone03 = data["cone_centers_xy_m"][2]

    assert cone03[0] == 1.2423424376307408
    assert cone03[1] == 0.23767209592830728
