"""Versioned initial APPO smoke scalarization, separate from raw evaluators.

Dense scales use the absolute nonzero TRAIN p95 (minimum three samples).
A conservative 1e-5 raw-unit numerical-signal floor prevents noise amplification;
smaller observed amplitudes retain measurements and use a natural-unit scale one.
This floor is an explicit initial assumption, not an empirically optimal threshold.
Sparse events preserve event magnitude without inverse-frequency weighting.
Weights encode an explicit initial preference, not a fitted optimum: each
observed progress channel +1, success +5, safety -1, effort -.05, time -.01.
All-zero/undersampled dense scales fall back to one with insufficient status.
Unavailable channels are excluded. No synthetic row can derive this config.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
import json
import math
from pathlib import Path
import statistics

from drl.MARL.reward_calibration import CalibrationDataset, MEASURED, MORPHOLOGIES, _percentile, measured_terms

CONFIG_SCHEMA = "mssr.reward_config.v1"
SAFETY = frozenset({"outside_road", "outside_width", "below_support", "fall", "connection_loss", "collision", "conflict", "invalid", "redundant_action", "stagnation"})
# These safety terms are continuous normalized distances, not binary events.
CONTINUOUS_SAFETY = frozenset({"outside_road", "outside_width", "below_support"})
EVENTS = (SAFETY - CONTINUOUS_SAFETY) | {"success", "target_entry", "goal_entry"}
# Owner-local values always win. These are the only shared semantics allowed
# to fill an absent owner field; team cost/progress aggregates never propagate.
GLOBAL_SHARED_TERMS = frozenset({"success", "time", "stagnation"})


@dataclass(frozen=True)
class ScalarizedReward:
    owners: dict[str, float]
    team: float
    excluded_terms: dict[str, dict] = field(default_factory=dict)
    broadcast_terms: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _terms(value):
    return asdict(value) if is_dataclass(value) else dict(value)


class RewardConfig(Mapping):
    """Mapping over four domains plus serializable schema/provenance."""

    def __init__(self, payload):
        if payload.get("schema") != CONFIG_SCHEMA:
            raise ValueError("unsupported RewardConfig schema")
        if set(payload.get("domains", {})) != set(MORPHOLOGIES):
            raise ValueError("RewardConfig requires all four domains")
        provenance = payload.get("provenance", {})
        if provenance.get("source") != "physical" or provenance.get("split") != "train":
            raise ValueError("RewardConfig requires physical TRAIN provenance")
        for settings in payload["domains"].values():
            for term, config in settings.get("terms", {}).items():
                weight = config.get("weight")
                if not isinstance(weight, (int, float)) or not math.isfinite(weight):
                    raise ValueError(f"invalid weight for {term}")
                scale = config.get("scale")
                if scale is not None and (not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
                    raise ValueError(f"invalid scale for {term}")
                if weight and scale is None:
                    raise ValueError(f"weighted term {term} has no measured scale")
        self._payload = deepcopy(payload)

    def __getitem__(self, domain):
        return deepcopy(self._payload["domains"][domain])

    def __iter__(self):
        return iter(self._payload["domains"])

    def __len__(self):
        return len(self._payload["domains"])

    def to_dict(self):
        return deepcopy(self._payload)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def scalarize(self, domain, raw_frame, *, availability=None):
        """Scalarize each owner once, plus a separate team diagnostic reward.

        Only absent success/time/stagnation owner fields inherit the team raw
        value. An existing owner field is never combined with its team aggregate.
        Unavailable weighted terms are excluded for this transition and retained
        as metadata in the result; this does not create measured-zero samples.
        Invalid signals and missing/nonfinite measured values fail closed.
        """
        settings = self._payload["domains"][domain]
        if settings["status"] != "derived_physical_train":
            raise ValueError(f"no physical TRAIN calibration for {domain}")
        frame = _terms(raw_frame)
        team_values = _terms(frame["team"])
        excluded, broadcasts = {}, {}
        for term, config in settings["terms"].items():
            if availability is not None:
                metadata = deepcopy(dict(availability.get(term, {})))
                status = metadata.get("status", "not_instrumented")
                if status not in MEASURED and status != "not_instrumented":
                    raise ValueError(f"term {term} has invalid availability status {status!r}")
                if config["weight"] and status == "not_instrumented":
                    metadata.setdefault("status", status)
                    if term not in availability:
                        metadata.setdefault("reason", "availability_not_supplied")
                    excluded[term] = metadata

        def scalar(terms, owner=None):
            values = _terms(terms)
            inherited = []
            result = 0.
            for term, config in settings["terms"].items():
                if config["weight"] == 0 or term in excluded:
                    continue
                if owner is not None and term not in values and term in GLOBAL_SHARED_TERMS:
                    value = team_values.get(term)
                    inherited.append(term)
                else:
                    value = values.get(term)
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"weighted term {term} must be finite and present")
                result += config["weight"] * value / config["scale"]
            if not math.isfinite(result):
                raise ValueError("nonfinite scalar reward")
            if owner is not None:
                broadcasts[owner] = tuple(sorted(inherited))
            return result

        team = scalar(team_values)
        owners = {owner: scalar(terms, owner) for owner, terms in frame["owners"].items()}
        return ScalarizedReward(team=team, owners=owners, excluded_terms=excluded, broadcast_terms=broadcasts)

    def preset(self, name):
        if name not in {"balanced", "stronger_safety", "stronger_sparse_success"}:
            raise ValueError("unknown initial preference preset")
        payload = self.to_dict()
        baseline = payload["baseline_weights"]
        for domain, settings in payload["domains"].items():
            for term, config in settings["terms"].items():
                config["weight"] = baseline[domain][term]
                if name == "stronger_safety" and term in SAFETY:
                    config["weight"] *= 2
                if name == "stronger_sparse_success" and term == "success":
                    config["weight"] *= 2
        payload["preset"] = name
        return RewardConfig(payload)

    def mixing_summary(self, dataset, *, transition_weights=None, gamma=.99):
        dataset.require_physical_train()
        weights = dict(transition_weights or {"navigation": .7, "stairs": .1, "gap": .1, "manipulation": .1})
        if set(weights) != set(MORPHOLOGIES) or any(not math.isfinite(value) or value < 0 for value in weights.values()) or not math.isclose(sum(weights.values()), 1.):
            raise ValueError("mixing requires four finite nonnegative transition weights summing to one")
        values = {domain: [] for domain in MORPHOLOGIES}
        returns = {}
        for row in dataset.rows:
            domain = row["manifest"]["domain"]
            raw = {"team": row["team_raw_terms"], "owners": row["owner_raw_terms"]}
            reward = self.scalarize(domain, raw, availability=row["term_availability"]).team
            values[domain].append(reward)
            episode = (domain, row["manifest"]["run_id"], row["manifest"]["episode_id"], row["env_index"])
            returns[episode] = returns.get(episode, 0.) + gamma**(row["episode_step"]-1) * reward
        domains = {}
        for domain, samples in values.items():
            magnitudes = [abs(value) for value in samples]
            domain_returns = [value for key, value in returns.items() if key[0] == domain]
            domains[domain] = dict(count=len(samples), mean=statistics.fmean(samples) if samples else None,
                mean_abs=statistics.fmean(magnitudes) if samples else None,
                p95_abs=_percentile(sorted(magnitudes), .95) if samples else None,
                reward_density=sum(value != 0 for value in samples)/len(samples) if samples else None,
                episode_return_mean=statistics.fmean(domain_returns) if domain_returns else None,
                weighted_magnitude_proxy=weights[domain]*statistics.fmean(magnitudes) if samples else None)
        total = sum(data["weighted_magnitude_proxy"] or 0 for data in domains.values())
        for data in domains.values():
            data["magnitude_share_proxy"] = data["weighted_magnitude_proxy"]/total if total and data["count"] else None
        return dict(transition_weights=weights, domains=domains, complete_four_domain_evidence=all(values.values()),
                    gamma=gamma, gradient_risk_measure="reward_magnitude_proxy_only")


def derive_initial_config(dataset: CalibrationDataset, *, minimum_nonzero=3, minimum_effective_signal_amplitude=1e-5):
    dataset.require_physical_train()
    if type(minimum_nonzero) is not int or minimum_nonzero < 1:
        raise ValueError("minimum_nonzero must be a positive integer")
    if (isinstance(minimum_effective_signal_amplitude, bool)
        or not isinstance(minimum_effective_signal_amplitude, (int, float))
        or not math.isfinite(minimum_effective_signal_amplitude) or minimum_effective_signal_amplitude <= 0):
        raise ValueError("minimum_effective_signal_amplitude must be finite and positive")
    domains, baseline = {}, {}
    for domain in MORPHOLOGIES:
        rows = [row for row in dataset.rows if row["manifest"]["domain"] == domain]
        settings = {"status": "derived_physical_train" if rows else "insufficient_no_physical_rows", "terms": {}}
        terms = sorted(set().union(*(set(row["team_raw_terms"]) for row in rows)))
        progress_terms = sum(term.endswith("progress") for term in terms) or 1
        for term in terms:
            values = [measured_terms(row).get(term) for row in rows]
            measured = [value for value in values if value is not None]
            nonzero = sorted(abs(value) for value in measured if value != 0)
            observed_p95 = _percentile(nonzero, .95) if nonzero else None
            if not measured:
                scale, status, normalization, weight = None, "excluded_unavailable", "excluded", 0.
            else:
                if term in EVENTS or term == "time":
                    scale, normalization = 1., "event_identity" if term in EVENTS else "decision_identity"
                    status = "measured_event" if nonzero else "insufficient_event_not_observed"
                elif len(nonzero) >= minimum_nonzero:
                    if observed_p95 < minimum_effective_signal_amplitude:
                        scale, normalization, status = 1., "unit_fallback", "insufficient_signal_above_numerical_floor"
                    else:
                        scale, normalization, status = observed_p95, "nonzero_abs_p95", "empirical"
                else:
                    scale, normalization, status = 1., "unit_fallback", "insufficient_nonzero_support"
                weight = (5. if term == "success" else -.01 if term == "time" else -.05 if term == "control_effort"
                          else -1. if term in SAFETY else 1./progress_terms if term.endswith("progress") else 1.)
            settings["terms"][term] = dict(scale=scale, weight=weight, status=status, normalization=normalization,
                measured_count=len(measured), nonzero_count=len(nonzero), observed_nonzero_abs_p95=observed_p95, missing_fraction=(len(values)-len(measured))/len(values),
                units=sorted({row["term_availability"].get(term, {}).get("units", "unknown") for row in rows}),
                reason="initial balanced preference; scale from physical TRAIN only; event frequency never inverted")
        domains[domain] = settings
        baseline[domain] = {term: config["weight"] for term, config in settings["terms"].items()}
    unique = {}
    for row in dataset.rows:
        manifest = row["manifest"]
        unique[(manifest["run_id"], manifest["episode_id"], row["env_index"])] = manifest
    return RewardConfig(dict(schema=CONFIG_SCHEMA, version=2, purpose="calibration-derived initial APPO smoke config",
        preset="balanced", domains=domains, baseline_weights=baseline,
        provenance=dict(source="physical", split="train", fit_rows=len(dataset.rows), episodes=list(unique.values()),
                        minimum_nonzero=minimum_nonzero, scale_method="absolute nonzero p95, minimum support, natural-unit fallback below signal floor",
                        minimum_effective_signal_amplitude=minimum_effective_signal_amplitude,
                        numerical_floor_reason="conservative raw-unit numerical-signal assumption; not learned optimal; preserves measured data",
                        preference="initial interpretable baseline; not thesis-optimal; compare presets on VALIDATION only")))
