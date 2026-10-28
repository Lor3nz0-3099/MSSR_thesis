"""Common signal availability and physical-time stagnation contracts."""
from types import SimpleNamespace
import json

import pytest

from drl.MARL.runtime_signals import collect_runtime_signals, StagnationDetector


OWNERS = tuple(f"smores_{i:02d}" for i in range(1, 9))


def positions(x=0.0, y=0.0, z=0.0315):
    return {owner: (x, y, z) for owner in OWNERS}


def collect(**kwargs):
    return collect_runtime_signals(owner_ids=OWNERS, delta_seconds=8/240, **kwargs)


def test_missing_measurements_are_unavailable_and_not_measured_zero():
    result = collect()
    for term in ("collision", "conflict", "invalid", "redundant_action", "control_effort", "stagnation"):
        assert result.availability[term]["status"] == "not_instrumented"
        assert result.availability[term]["source"]
        assert result.availability[term]["units"]
        assert result.availability[term]["attribution"]
    assert result.signals.control_effort_by_owner == {}
    assert result.signals.collision_owner_ids == frozenset()
    assert result.diagnostics["delta_seconds"] == pytest.approx(1/30)
    assert result.availability["time"]["status"] == "measured_present"
    assert result.availability["time"]["units"] == "decision"


@pytest.mark.parametrize("as_mapping", [True, False])
def test_native_acceptance_measures_absent_conflict_and_invalid(as_mapping):
    outcome = dict(accepted=True, code="ACCEPTED", owner_ids=OWNERS)
    result = collect(proposal_outcome=outcome if as_mapping else SimpleNamespace(**outcome))
    assert result.availability["conflict"]["status"] == "measured_absent"
    assert result.availability["invalid"]["status"] == "measured_absent"
    assert not result.signals.conflict_owner_ids
    assert not result.signals.invalid_owner_ids


@pytest.mark.parametrize("code, field, other", [
    ("RESOURCE_BUSY", "conflict", "invalid"),
    ("INVALID_GOAL", "invalid", "conflict"),
])
def test_native_policy_rejections_remain_owner_attributed(code, field, other):
    result = collect(proposal_outcome=dict(accepted=False, code=code, owner_ids=OWNERS[:2]))
    assert getattr(result.signals, f"{field}_owner_ids") == frozenset(OWNERS[:2])
    assert result.availability[field]["status"] == "measured_present"
    assert result.availability[other]["status"] == "measured_absent"


def test_unknown_infrastructure_rejection_is_an_exception_not_a_policy_penalty():
    with pytest.raises(RuntimeError, match="OWNER_CRASH"):
        collect(proposal_outcome=dict(accepted=False, code="OWNER_CRASH", owner_ids=OWNERS))


def test_rejection_without_owner_attribution_fails_closed():
    with pytest.raises(ValueError, match="owner"):
        collect(proposal_outcome=dict(accepted=False, code="RESOURCE_BUSY", owner_ids=()))


def test_unknown_signal_owner_is_rejected():
    with pytest.raises(ValueError, match="owner"):
        collect(measured_collision_owner_ids=("ghost",))


def test_contact_sensor_absence_differs_from_measured_no_contact():
    absent = collect(measured_collision_owner_ids=())
    present = collect(measured_collision_owner_ids=(OWNERS[2],))
    assert absent.availability["collision"]["status"] == "measured_absent"
    assert present.availability["collision"]["status"] == "measured_present"
    assert present.signals.collision_owner_ids == frozenset({OWNERS[2]})
    assert "contact" in present.availability["collision"]["source"]


def test_redundancy_is_explicitly_attributed_and_never_inferred_from_owner_count():
    result = collect(redundant_action_count_by_owner={OWNERS[1]: 2},
                     redundant_action_source="native_duplicate_attribution")
    assert result.signals.redundant_action_count_by_owner == {OWNERS[1]: 2}
    assert result.availability["redundant_action"]["status"] == "measured_present"
    assert collect().availability["redundant_action"]["status"] == "not_instrumented"


def test_effort_preserves_declared_command_units_without_claiming_energy():
    result = collect(control_effort_by_owner={OWNERS[0]: 0.25},
                     control_effort_units="rad^2/s", control_effort_source="applied_pan_rate_squared_integral")
    assert result.signals.control_effort_by_owner == {OWNERS[0]: 0.25}
    assert result.availability["control_effort"]["units"] == "rad^2/s"
    assert result.availability["control_effort"]["source"] == "applied_pan_rate_squared_integral"
    assert result.availability["control_effort"]["status"] == "measured_present"


def test_effort_requires_explicit_units_and_source():
    with pytest.raises(ValueError, match="units|source"):
        collect(control_effort_by_owner={})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, "bad"])
def test_invalid_effort_is_marked_invalid_and_excluded_from_raw_signal(value):
    result = collect(control_effort_by_owner={OWNERS[0]: value},
                     control_effort_units="rad^2/s", control_effort_source="applied_pan_rate_squared_integral")
    assert result.availability["control_effort"]["status"] == "invalid"
    assert result.signals.control_effort_by_owner == {}
    json.dumps(dict(result.availability), allow_nan=False)
    json.dumps(dict(result.diagnostics), allow_nan=False)


@pytest.mark.parametrize("value", [-1, 0.5, True])
def test_invalid_redundancy_counts_are_not_valid_zero_measurements(value):
    result = collect(redundant_action_count_by_owner={OWNERS[0]: value},
                     redundant_action_source="native_duplicate_attribution")
    assert result.availability["redundant_action"]["status"] == "invalid"
    assert result.signals.redundant_action_count_by_owner == {}


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf")])
def test_physical_interval_must_be_finite_and_positive(dt):
    with pytest.raises(ValueError, match="delta_seconds"):
        collect_runtime_signals(owner_ids=OWNERS, delta_seconds=dt)


def test_stagnation_requires_a_full_physical_time_window_and_resets_per_episode():
    detector = StagnationDetector(window_seconds=1.0, displacement_threshold_m=0.005)
    for _ in range(3):
        assert detector.observe(positions(), positions(), delta_seconds=0.25) is None
    assert detector.observe(positions(), positions(), delta_seconds=0.25) is True
    assert detector.diagnostics["metric"] == "centroid_displacement_m"
    assert detector.diagnostics["window_seconds"] == 1
    assert detector.diagnostics["threshold_m"] == 0.005
    assert detector.diagnostics["displacement_m"] == 0
    detector.reset()
    assert detector.observe(positions(3), positions(3), delta_seconds=0.25) is None


def test_stagnation_uses_positions_and_not_discounted_reward_or_action_activity():
    detector = StagnationDetector(window_seconds=1, displacement_threshold_m=0.005)
    for step in range(4):
        result = detector.observe(positions(step*0.003), positions((step+1)*0.003), delta_seconds=0.25)
    assert result is False
    assert detector.diagnostics["displacement_m"] == pytest.approx(0.012)


def test_irregular_interval_interpolates_the_exact_window_boundary():
    detector = StagnationDetector(window_seconds=1, displacement_threshold_m=0.005)
    assert detector.observe(positions(0), positions(0.006), delta_seconds=0.6) is None
    assert detector.observe(positions(0.006), positions(0.006), delta_seconds=0.6) is True
    assert detector.diagnostics["displacement_m"] == pytest.approx(0.004)


def test_detector_rejects_discontinuous_state_without_advancing_its_window():
    detector = StagnationDetector(window_seconds=1, displacement_threshold_m=0.005)
    detector.observe(positions(), positions(), delta_seconds=0.5)
    with pytest.raises(ValueError, match="continuity"):
        detector.observe(positions(1), positions(1), delta_seconds=0.5)
    assert detector.observe(positions(), positions(), delta_seconds=0.5) is True


def test_collector_exports_stagnation_warmup_and_detection_without_synthetic_failure():
    detector = StagnationDetector(window_seconds=0.05, displacement_threshold_m=0.005)
    first = collect(stagnation_detector=detector,
                    previous_module_positions_m=positions(), next_module_positions_m=positions())
    assert first.availability["stagnation"]["status"] == "not_instrumented"
    second = collect(stagnation_detector=detector,
                     previous_module_positions_m=positions(), next_module_positions_m=positions())
    assert second.availability["stagnation"]["status"] == "measured_present"
    assert second.signals.stagnating
    assert not second.signals.invalid_owner_ids
    assert not second.signals.collision_owner_ids
    assert second.diagnostics["stagnation"]["metric"] == "centroid_displacement_m"


def test_invalid_physical_positions_mark_stagnation_invalid_without_poisoning_history():
    detector = StagnationDetector(window_seconds=0.05)
    broken = dict(positions(), smores_01=(float("nan"), 0, 0))
    result = collect(stagnation_detector=detector,
                     previous_module_positions_m=positions(), next_module_positions_m=broken)
    assert result.availability["stagnation"]["status"] == "invalid"
    assert not result.signals.stagnating
    # The failed observation has not advanced physical time in the detector.
    assert detector.observe(positions(), positions(), delta_seconds=0.03) is None
