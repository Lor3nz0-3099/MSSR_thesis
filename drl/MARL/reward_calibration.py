"""Pure, TRAIN-only calibration over completed DirectMARL transitions.

Rows retain raw terms and availability separately. Only measured finite values
enter statistics; owner samples never count as independent episodes. Memory is
bounded explicitly for the intended small pilot; no Isaac reads occur here.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from pathlib import Path
import statistics

from drl.MARL.direct_marl_env import DirectMARLTransition

SCHEMA = "mssr.reward_calibration.v1"
MEASURED = frozenset({"measured_present", "measured_absent"})
STATUSES = MEASURED | {"not_instrumented", "invalid"}
MORPHOLOGIES = {"navigation": "rc_car8", "stairs": "snake8", "gap": "snake8", "manipulation": "mobile_manipulator8"}


def curriculum_identity(domain, seed):
    if domain not in MORPHOLOGIES or type(seed) is not int:
        raise ValueError("calibration requires a supported domain and integer seed")
    from smores_ep.isaac.phase1_marl_curriculum import curriculum_identity as navigation
    from smores_ep.isaac.phase1_stairs_curriculum import stair_curriculum_identity as stairs
    from smores_ep.isaac.phase1_gap_curriculum import gap_curriculum_identity as gap
    from smores_ep.isaac.phase1_manipulation_curriculum import manipulation_curriculum_identity as manipulation
    return {"navigation": navigation, "stairs": stairs, "gap": gap, "manipulation": manipulation}[domain](seed, split="train")


@dataclass(frozen=True)
class EpisodeManifest:
    run_id: str
    episode_id: str
    domain: str
    morphology: str
    seed: int
    split: str
    difficulty: str
    family: str | None
    collection_policy: str
    checkpoint_path: str
    checkpoint_sha256: str
    base_commit: str
    working_tree_dirty: bool
    physics_hz: float
    decision_hz: float
    source: str = "physical"

    def __post_init__(self):
        if self.split != "train":
            raise ValueError("calibration accepts TRAIN only")
        identity = curriculum_identity(self.domain, self.seed)
        if self.difficulty != identity.difficulty:
            raise ValueError("difficulty differs from curriculum identity")
        if self.morphology != MORPHOLOGIES[self.domain]:
            raise ValueError("domain/morphology mismatch")
        if self.family is not None and hasattr(identity, "family") and self.family != identity.family:
            raise ValueError("family differs from curriculum identity")
        if self.physics_hz != 240 or not math.isfinite(self.decision_hz) or self.decision_hz <= 0:
            raise ValueError("requires physics 240 Hz and positive finite decision Hz")
        if self.source not in {"physical", "synthetic"}:
            raise ValueError("source must be physical or synthetic")
        if type(self.working_tree_dirty) is not bool:
            raise ValueError("working_tree_dirty must be explicit boolean")
        for label in ("run_id", "episode_id", "collection_policy", "checkpoint_path", "base_commit"):
            if not isinstance(getattr(self, label), str) or not getattr(self, label):
                raise ValueError(f"{label} is required")
        if len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256):
            raise ValueError("checkpoint SHA256 must be a lowercase hexadecimal digest")


def _mapping(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("raw reward terms must be dataclasses or mappings")


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _stamp(transition, previous):
    name = "previous_stamp_by_env" if previous else "next_stamp_by_env"
    stamps = _field(transition.physics_result, name, {})
    value = stamps.get(transition.env_index, stamps.get(str(transition.env_index)))
    if value is None:
        value = _field(transition.previous_state if previous else transition.next_state, "stamp")
    if not _finite(value):
        raise ValueError("completed transition requires finite captured physical stamps")
    return float(value)


class TransitionRecorder:
    """Append-only observer; never evaluates rewards or queries the simulator."""

    def __init__(self, path, manifest: EpisodeManifest):
        self.path = Path(path)
        self.manifest = manifest
        self._step = 0
        self._terminal = False
        self._last_stamp = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def observe(self, transition: DirectMARLTransition, *, availability,
                application_provenance=None, failure_reason=None, collector_truncated=False):
        if not isinstance(transition, DirectMARLTransition):
            raise TypeError("observer requires a completed DirectMARLTransition")
        if self._terminal:
            raise ValueError("cannot record after terminal episode boundary")
        if transition.episode_step != self._step + 1:
            raise ValueError("episode step must be contiguous and start at 1")
        previous, following = _stamp(transition, True), _stamp(transition, False)
        if following <= previous or (self._last_stamp is not None and not math.isclose(previous, self._last_stamp, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("physical stamps must increase with transition continuity")
        evaluation = transition.evaluation
        raw = _field(evaluation, "raw_reward")
        team = _mapping(_field(raw, "team"))
        owners = {str(owner): _mapping(terms) for owner, terms in _field(raw, "owners").items()}
        metadata = {}
        for term in set(team).union(*(set(terms) for terms in owners.values())):
            supplied = availability.get(term, {})
            data = _mapping(supplied)
            data.setdefault("status", "not_instrumented")
            data.setdefault("units", "unknown")
            data.setdefault("source", "not_connected")
            data.setdefault("attribution", "owner_and_team")
            if data["status"] not in STATUSES or not isinstance(data["units"], str) or not data["units"]:
                raise ValueError(f"invalid availability metadata for {term}")
            if any(term in terms and not _finite(terms[term]) for terms in [team, *owners.values()]):
                data["status"] = "invalid"
                for terms in [team, *owners.values()]:
                    if term in terms and not _finite(terms[term]):
                        terms[term] = None
            metadata[term] = data
        terminated = bool(_field(evaluation, "terminated", False))
        truncated = bool(_field(evaluation, "truncated", False) or collector_truncated)
        failure_reason = failure_reason or _field(evaluation, "failure_reason")
        row = dict(schema=SCHEMA, manifest=asdict(self.manifest), env_index=transition.env_index,
                   episode_step=transition.episode_step, previous_stamp=previous, next_stamp=following,
                   delta_seconds=following-previous, team_raw_terms=team, owner_raw_terms=owners,
                   term_availability=metadata, application_provenance=dict(application_provenance or {}),
                   success=bool(team.get("success", 0) and metadata.get("success", {}).get("status") in MEASURED),
                   terminated=terminated, truncated=truncated, failure_reason=failure_reason)
        encoded = json.dumps(row, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        self._step = transition.episode_step
        self._last_stamp = following
        self._terminal = terminated or truncated
        return row


def measured_terms(row, owner=None):
    terms = row["team_raw_terms"] if owner is None else row["owner_raw_terms"][owner]
    return {term: float(value) if _finite(value) and row["term_availability"].get(term, {}).get("status") in MEASURED else None
            for term, value in terms.items()}


def _percentile(values, p):
    position = (len(values)-1)*p
    lower = int(position)
    upper = min(lower+1, len(values)-1)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def term_statistics(values):
    values = list(values)
    clean = sorted(value for value in values if value is not None)
    result = {"count": len(clean), "missing_fraction": (len(values)-len(clean))/len(values) if values else 1.}
    for key in ("mean", "std", "min", "max", "median", "p05", "p25", "p75", "p95", "p99", "zero_fraction", "positive_fraction", "negative_fraction"):
        result[key] = None
    if clean:
        result.update(mean=statistics.fmean(clean), std=statistics.pstdev(clean), min=clean[0], max=clean[-1], median=statistics.median(clean),
                      zero_fraction=sum(value == 0 for value in clean)/len(clean), positive_fraction=sum(value > 0 for value in clean)/len(clean),
                      negative_fraction=sum(value < 0 for value in clean)/len(clean))
        for name, p in (("p05", .05), ("p25", .25), ("p75", .75), ("p95", .95), ("p99", .99)):
            result[name] = _percentile(clean, p)
    return result


def _ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[start]] == values[ordered[end]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = (start + end - 1)/2 + 1
        start = end
    return result


def _pearson(x, y):
    if len(x) < 2:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx, sy = sum((v-mx)**2 for v in x), sum((v-my)**2 for v in y)
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/math.sqrt(sx*sy) if sx and sy else None


def term_correlations(samples):
    terms = sorted(set().union(*(set(sample) for sample in samples)))
    result = {kind: {term: {} for term in terms} for kind in ("pearson", "spearman")}
    for first in terms:
        for second in terms:
            pairs = [(sample[first], sample[second]) for sample in samples if sample.get(first) is not None and sample.get(second) is not None]
            x, y = zip(*pairs) if pairs else ([], [])
            result["pearson"][first][second] = _pearson(x, y)
            result["spearman"][first][second] = _pearson(_ranks(x), _ranks(y))
    return result


def _statistics(samples):
    terms = sorted(set().union(*(set(sample) for sample in samples)))
    result = {}
    for term in terms:
        values = [sample.get(term) for sample in samples]
        result[term] = term_statistics(values)
        result[term]["nonzero"] = term_statistics([value for value in values if value is not None and value != 0])
    return result


class CalibrationDataset:
    def __init__(self, rows, *, max_rows=100000):
        self.rows = []
        seen = set()
        for row in rows:
            if len(self.rows) >= max_rows:
                raise ValueError("calibration pilot row limit exceeded")
            if row.get("schema") != SCHEMA:
                raise ValueError("unsupported calibration schema")
            EpisodeManifest(**row["manifest"])
            key = (row["manifest"]["run_id"], row["manifest"]["episode_id"], row["env_index"], row["episode_step"])
            if key in seen:
                raise ValueError("duplicate episode transition")
            seen.add(key)
            self.rows.append(row)
        episodes = defaultdict(list)
        for row in self.rows:
            episodes[(row["manifest"]["run_id"], row["manifest"]["episode_id"], row["env_index"])].append(row)
        for rows in episodes.values():
            rows.sort(key=lambda row: row["episode_step"])
            last = None
            for step, row in enumerate(rows, 1):
                if row["manifest"] != rows[0]["manifest"]:
                    raise ValueError("episode manifest changed within collection")
                if type(row["episode_step"]) is not int or row["episode_step"] != step:
                    raise ValueError("episode steps must be contiguous from one")
                if last and (last["terminated"] or last["truncated"]):
                    raise ValueError("row follows terminal episode boundary")
                if (not _finite(row["delta_seconds"]) or row["delta_seconds"] <= 0
                    or not _finite(row["previous_stamp"]) or not _finite(row["next_stamp"])
                    or not math.isclose(row["next_stamp"]-row["previous_stamp"], row["delta_seconds"], rel_tol=0, abs_tol=1e-9)):
                    raise ValueError("invalid physical transition duration")
                if last and not math.isclose(last["next_stamp"], row["previous_stamp"], rel_tol=0, abs_tol=1e-9):
                    raise ValueError("noncontiguous physical stamps")
                last = row

    @classmethod
    def from_jsonl(cls, path, **kwargs):
        with Path(path).open(encoding="utf-8") as handle:
            return cls((json.loads(line) for line in handle if line.strip()), **kwargs)

    def require_physical_train(self):
        if not self.rows or any(row["manifest"]["source"] != "physical" for row in self.rows):
            raise ValueError("scale fitting requires nonempty physical TRAIN rows; synthetic data are diagnostic only")
        CalibrationDataset(self.rows, max_rows=len(self.rows))
        return self

    def report(self, *, gamma=.99):
        if not _finite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        grouped, episodes = defaultdict(list), defaultdict(list)
        for row in self.rows:
            manifest = row["manifest"]
            policy = row.get("application_provenance", {}).get("collection_policy") or manifest["collection_policy"]
            grouped["/".join((manifest["domain"], manifest["difficulty"], policy))].append(row)
            episodes[(manifest["run_id"], manifest["episode_id"], row["env_index"])].append(row)
        report = {"schema": SCHEMA, "gamma": gamma, "groups": {}, "episodes": []}
        for key, rows in grouped.items():
            samples = [measured_terms(row) for row in rows]
            owners = sorted(set().union(*(set(row["owner_raw_terms"]) for row in rows)))
            report["groups"][key] = dict(team_terms=_statistics(samples), correlations=term_correlations(samples),
                owner_terms={owner: _statistics([measured_terms(row, owner) for row in rows if owner in row["owner_raw_terms"]]) for owner in owners})
        for key, rows in episodes.items():
            rows.sort(key=lambda row: row["episode_step"])
            samples = [measured_terms(row) for row in rows]
            terms = sorted(set().union(*(set(sample) for sample in samples)))
            sums, discounted = {}, {}
            for term in terms:
                values = [sample.get(term) for sample in samples]
                complete = all(value is not None for value in values)
                sums[term] = sum(values) if complete else None
                discounted[term] = sum(gamma**i * value for i, value in enumerate(values)) if complete else None
            owner_sums, owner_discounted = {}, {}
            owners = sorted(set().union(*(set(row["owner_raw_terms"]) for row in rows)))
            for owner in owners:
                owner_sums[owner], owner_discounted[owner] = {}, {}
                owner_samples = [measured_terms(row, owner) if owner in row["owner_raw_terms"] else {} for row in rows]
                for term in terms:
                    values = [sample.get(term) for sample in owner_samples]
                    complete = all(value is not None for value in values)
                    owner_sums[owner][term] = sum(values) if complete else None
                    owner_discounted[owner][term] = sum(gamma**i * value for i, value in enumerate(values)) if complete else None
            final = rows[-1]
            report["episodes"].append(dict(manifest=final["manifest"], env_index=key[2], length_steps=len(rows),
                length_seconds=sum(row["delta_seconds"] for row in rows), success=any(row["success"] for row in rows),
                failure=bool(final["terminated"] and not final["success"]), terminated=final["terminated"], truncated=final["truncated"],
                failure_reason=final["failure_reason"], raw_sum=sums, discounted_return=discounted,
                owner_raw_sum=owner_sums, owner_discounted_return=owner_discounted))
        return report


def pilot_layouts():
    """Two allocated TRAIN seeds per difficulty/domain; 24 layouts total."""
    result = []
    starts = {"navigation": 7000, "stairs": 12000, "gap": 16000, "manipulation": 20000}
    for domain, start in starts.items():
        for offset in (0, 250, 650):
            for seed in (start+offset, start+offset+1):
                identity = curriculum_identity(domain, seed)
                result.append(dict(domain=domain, seed=seed, split="train", difficulty=identity.difficulty))
    return result
