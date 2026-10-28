"""Initial smoke weights use measured physical TRAIN scales only."""
from dataclasses import asdict
import json

import pytest

from drl.MARL import reward_config
from drl.MARL.reward_calibration import CalibrationDataset, SCHEMA
from drl.MARL.tests.test_reward_calibration import manifest, Frame, Terms


def dataset(domain="navigation", seed=7000, morphology="rc_car8", progress=(0., 2., 4., 6.)):
    rows = []
    for index, value in enumerate(progress):
        terms = {"progress": value, "success": float(index == len(progress)-1), "collision": 0., "time": 1.}
        metadata = {term: {"status": "measured_present", "units": "event"} for term in terms}
        metadata["collision"]["status"] = "not_instrumented"
        rows.append(dict(schema=SCHEMA, manifest=asdict(manifest(domain=domain, seed=seed, morphology=morphology)), env_index=0,
                         episode_step=index+1, previous_stamp=index*.1, next_stamp=(index+1)*.1, delta_seconds=.1,
                         team_raw_terms=terms, owner_raw_terms={"m0": terms}, term_availability=metadata,
                         application_provenance={}, success=bool(terms["success"]), terminated=index == len(progress)-1,
                         truncated=False, failure_reason=None))
    return CalibrationDataset(rows)


def test_config_uses_nonzero_abs_p95_and_leaves_binary_scale_one(tmp_path):
    config = reward_config.derive_initial_config(dataset())
    nav = config["navigation"]
    assert nav["terms"]["progress"]["scale"] == pytest.approx(5.8)
    assert nav["terms"]["progress"]["normalization"] == "nonzero_abs_p95"
    assert nav["terms"]["success"]["scale"] == 1.
    assert nav["terms"]["collision"]["weight"] == 0
    assert nav["terms"]["collision"]["scale"] is None
    assert nav["terms"]["collision"]["status"] == "excluded_unavailable"
    assert set(config) == {"navigation", "stairs", "gap", "manipulation"}
    assert config["stairs"]["status"] == "insufficient_no_physical_rows"
    target = tmp_path / "config.json"
    config.save(target)
    restored = reward_config.RewardConfig.load(target)
    assert restored.to_dict() == config.to_dict()
    assert json.loads(target.read_text())["purpose"] == "calibration-derived initial APPO smoke config"


def test_scalarization_is_separate_and_excludes_unavailable_terms():
    config = reward_config.derive_initial_config(dataset())
    raw = Frame(Terms(progress=5.8, success=1, collision=100), {"m0": Terms(progress=5.8, success=1, collision=100)})
    values = config.scalarize("navigation", raw)
    assert values.team == pytest.approx(5.99)
    assert values.owners["m0"] == pytest.approx(5.99)
    assert raw.team.progress == 5.8
    excluded = config.scalarize("navigation", raw, availability={"progress": {"status": "not_instrumented"}})
    assert excluded.team == 0.
    assert excluded.owners["m0"] == 0.
    assert excluded.excluded_terms["progress"]["status"] == "not_instrumented"
    assert excluded.excluded_terms["success"]["status"] == "not_instrumented"


def test_synthetic_and_nontrain_cannot_derive_configuration():
    data = dataset()
    data.rows[0]["manifest"]["source"] = "synthetic"
    with pytest.raises(ValueError, match="physical"):
        reward_config.derive_initial_config(data)
    data = dataset()
    data.rows[0]["manifest"]["split"] = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        reward_config.derive_initial_config(data)


def test_zero_dense_terms_are_marked_insufficient_and_presets_do_not_fit_frequency():
    data = dataset(progress=(0., 0., 0., 0.))
    config = reward_config.derive_initial_config(data)
    assert config["navigation"]["terms"]["progress"]["status"] == "insufficient_nonzero_support"
    assert config["navigation"]["terms"]["progress"]["scale"] == 1.
    safety = config.preset("stronger_safety")
    sparse = config.preset("stronger_sparse_success")
    assert safety["navigation"]["terms"]["time"]["weight"] == -.01
    assert sparse["navigation"]["terms"]["success"]["weight"] == 10.
    assert sparse["navigation"]["terms"]["success"]["scale"] == 1.
    assert config["navigation"]["terms"]["success"]["weight"] == 5.


def test_mixing_reports_actual_magnitude_and_proxy_not_gradient_claim():
    config = reward_config.derive_initial_config(dataset())
    report = config.mixing_summary(dataset())
    assert report["transition_weights"] == {"navigation": .7, "stairs": .1, "gap": .1, "manipulation": .1}
    assert report["domains"]["navigation"]["count"] == 4
    assert report["domains"]["navigation"]["reward_density"] == 1.
    assert report["domains"]["stairs"]["count"] == 0
    assert report["complete_four_domain_evidence"] is False
    assert report["gradient_risk_measure"] == "reward_magnitude_proxy_only"


def test_config_requires_uniform_checkpoint_provenance_and_finite_weights():
    data = dataset()
    data.rows[1]["manifest"]["checkpoint_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="manifest"):
        reward_config.derive_initial_config(data)
    config = reward_config.derive_initial_config(dataset()).to_dict()
    config["domains"]["navigation"]["terms"]["progress"]["scale"] = 0
    with pytest.raises(ValueError, match="scale"):
        reward_config.RewardConfig(config)
    config = reward_config.derive_initial_config(dataset()).to_dict()
    config["domains"]["navigation"]["terms"]["progress"]["weight"] = float("nan")
    with pytest.raises(ValueError, match="weight"):
        reward_config.RewardConfig(config)


def gap_fixture():
    """Unit fixtures exercise the physical-row gate, not empirical evidence."""
    from drl.MARL.gap_reward import GapOwnerRewardTerms, GapTeamRewardTerms, GapRawRewardFrame
    data = dataset(domain="gap", seed=16000, morphology="snake8")
    raw = GapRawRewardFrame(
        owners={f"m{i}": GapOwnerRewardTerms(approach_progress=3. if i == 0 else 0.,
                    fall=2. if i == 0 else 1., time=2. if i == 0 else 1.) for i in range(8)},
        team=GapTeamRewardTerms(approach_progress=6., fall=8., success=1., stagnation=1.))
    for index, row in enumerate(data.rows):
        row["team_raw_terms"] = asdict(raw.team)
        row["team_raw_terms"]["success"] = float(index == 3)
        row["team_raw_terms"]["stagnation"] = float(index == 3)
        row["owner_raw_terms"] = {owner: asdict(terms) for owner, terms in raw.owners.items()}
        row["term_availability"] = {term: {"status": "measured_present" if value else "measured_absent",
                                         "units": "event", "source": "unit_fixture"}
                                    for term, value in row["team_raw_terms"].items()}
    return reward_config.derive_initial_config(data), raw, data.rows[-1]["term_availability"]


def test_actual_gap_frame_broadcasts_only_absent_shared_terms_once():
    config, raw, availability = gap_fixture()
    result = config.scalarize("gap", raw, availability=availability)
    # Progress uses team p95=6 and weight=1/3. Owner-local fall/time win;
    # global success and stagnation each contribute exactly once per owner.
    assert result.owners["m0"] == pytest.approx(1/6 + 5 - 1 - 2 - .02)
    assert result.owners["m1"] == pytest.approx(5 - 1 - 1 - .01)
    assert result.team == pytest.approx(1/3 + 5 - 1 - 8 - .01)
    assert set(result.broadcast_terms["m0"]) == {"success", "stagnation"}
    assert "time" not in result.broadcast_terms["m0"]
    assert "success" not in asdict(raw.owners["m0"])
    assert result.excluded_terms == {}


def test_warmup_excludes_weighted_stagnation_and_preserves_unavailable_metadata():
    from copy import deepcopy
    config, raw, availability = gap_fixture()
    availability["stagnation"] = dict(status="not_instrumented", units="event",
                                      source="physical_centroid", reason="window_not_full")
    before = deepcopy(availability)
    result = config.scalarize("gap", raw, availability=availability)
    assert result.owners["m0"] == pytest.approx(1/6 + 5 - 2 - .02)
    assert result.team == pytest.approx(1/3 + 5 - 8 - .01)
    assert result.excluded_terms == {"stagnation": before["stagnation"]}
    assert availability == before
    assert "stagnation" not in result.broadcast_terms["m0"]
    assert raw.team.stagnation == 1.
    json.dumps(asdict(result), allow_nan=False)


def test_global_shared_fallback_cannot_hide_missing_owner_local_terms():
    config, raw, availability = gap_fixture()
    frame = asdict(raw)
    del frame["owners"]["m0"]["fall"]
    with pytest.raises(ValueError, match="fall.*finite and present"):
        config.scalarize("gap", frame, availability=availability)


@pytest.mark.parametrize("status", ["invalid", "bogus"])
def test_scalarization_fails_closed_on_invalid_signal_status(status):
    config, raw, availability = gap_fixture()
    availability["stagnation"]["status"] = status
    with pytest.raises(ValueError, match="stagnation.*invalid"):
        config.scalarize("gap", raw, availability=availability)


def test_existing_owner_shared_fields_win_and_missing_time_inherits_once():
    config, raw, availability = gap_fixture()
    frame = asdict(raw)
    frame["owners"]["m0"].update(success=.2, stagnation=.5)
    del frame["owners"]["m0"]["time"]
    result = config.scalarize("gap", frame, availability=availability)
    assert result.owners["m0"] == pytest.approx(1/6 + 1 - .5 - 2 - .01)
    assert result.broadcast_terms["m0"] == ("time",)


def test_invalid_signal_is_rejected_even_if_calibration_excluded_its_weight():
    config = reward_config.derive_initial_config(dataset())
    raw = Frame(Terms(progress=1.), {"m0": Terms(progress=1.)})
    assert config["navigation"]["terms"]["collision"]["weight"] == 0
    with pytest.raises(ValueError, match="collision.*invalid"):
        config.scalarize("navigation", raw, availability={"collision": {"status": "invalid"}})


def add_raw_term(data, name, values):
    for row, value in zip(data.rows, values):
        row["team_raw_terms"][name] = value
        row["owner_raw_terms"]["m0"][name] = value
        row["term_availability"][name] = dict(status="measured_present", units="dimensionless")
    return data


@pytest.mark.parametrize("domain,seed,term", [("navigation", 7000, "outside_road"),
    ("stairs", 12000, "outside_width"), ("stairs", 12000, "below_support")])
def test_continuous_normalized_safety_uses_empirical_scale(domain, seed, term):
    data = dataset(domain=domain, seed=seed, morphology="rc_car8" if domain == "navigation" else "snake8")
    config = reward_config.derive_initial_config(add_raw_term(data, term, [.1, .2, .4, .6]))
    settings = config[domain]["terms"][term]
    assert settings["normalization"] == "nonzero_abs_p95"
    assert settings["scale"] == pytest.approx(.57)
    assert settings["units"] == ["dimensionless"]
    assert settings["weight"] == -1.


@pytest.mark.parametrize("amplitude,scale,status", [(-1.703e-7, 1., "insufficient_signal_above_numerical_floor"),
                                                        (1e-3, 1e-3, "empirical")])
def test_numerical_depression_noise_is_not_amplified(amplitude, scale, status):
    data = dataset(domain="manipulation", seed=20000, morphology="mobile_manipulator8")
    config = reward_config.derive_initial_config(add_raw_term(data, "depression_progress", [amplitude]*4))
    term = config["manipulation"]["terms"]["depression_progress"]
    assert term["scale"] == pytest.approx(scale)
    assert term["status"] == status
    assert term["measured_count"] == term["nonzero_count"] == 4
    assert term["observed_nonzero_abs_p95"] == pytest.approx(abs(amplitude))
    assert config.to_dict()["provenance"]["minimum_effective_signal_amplitude"] == 1e-5
    assert "assumption" in config.to_dict()["provenance"]["numerical_floor_reason"]
    assert config.to_dict()["version"] == 2
