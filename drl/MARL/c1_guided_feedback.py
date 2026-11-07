"""Optional C1 graph shaping and ALIGN feedback.

Target topology affects rewards, never feasibility masks.
No changes to structural options, H1, H2 or MAPPO.
"""

from collections import Counter
from itertools import permutations
from pathlib import Path
import json
import math


PORTS = {"LEFT", "RIGHT", "TOP", "BOTTOM"}

DOMAIN_TARGETS = {
    "navigation": "smores_rc_car8.json",
    "stairs": "smores_snake8.json",
    "gap": "smores_snake8.json",
    "manipulation": "smores_mobile_manipulator8.json",
}


def load_target(domain):
    root = Path(__file__).resolve().parents[2]
    path = (
        root / "mssr_ws/src/mssr_expert/config"
        / DOMAIN_TARGETS[domain]
    )
    target = json.loads(path.read_text())
    return target.get("target", target)


def edge(a, fa, b, fb):
    if (
        a == b
        or fa not in PORTS
        or fb not in PORTS
    ):
        raise ValueError("Invalid port-labelled connection")

    return tuple(sorted((
        (str(a), str(fa)),
        (str(b), str(fb)),
    )))


class TargetGraphProgress:
    """Maximum port-preserving matching, regardless of module IDs."""

    def __init__(self, target, physical_ids):
        target = target.get("target", target)

        self.slots = tuple(
            n["node_id"] for n in target["nodes"]
        )

        self.physical_ids = tuple(physical_ids)

        if (
            len(self.slots) != 8
            or len(set(self.slots)) != 8
            or len(self.physical_ids) != 8
            or len(set(self.physical_ids)) != 8
        ):
            raise ValueError("Expected eight unique modules")

        self.target_edges = frozenset(
            edge(
                e["module_a_id"],
                e["attributes"]["face_a"],
                e["module_b_id"],
                e["attributes"]["face_b"],
            )
            for e in target["edges"]
        )

        if not self.target_edges:
            raise ValueError("Empty target topology")

        self.best_fraction = 0.0
        self._last_signature = None
        self._last_match = 0
        self._previous_match_fraction = 0.0
        self._previous_wrong_fraction = 0.0
        self.current_potential = 0.0

    def observe(self, frame):
        signature = tuple(sorted({
            edge(
                c.module_a,
                c.connector_a,
                c.module_b,
                c.connector_b,
            )
            for c in frame.connections
            if (c.joint_valid and c.module_a != c.module_b
                and c.connector_a in PORTS and c.connector_b in PORTS)
        }))

        if signature != self._last_signature:
            current = tuple(
                e for e in signature
                if all(
                    endpoint[0] in self.physical_ids
                    for endpoint in e
                )
            )

            best = 0
            maximum = min(
                len(current),
                len(self.target_edges),
            )

            if maximum:
                for assignment in permutations(self.slots):
                    mapping = dict(
                        zip(self.physical_ids, assignment)
                    )

                    matches = sum(
                        edge(
                            mapping[a], fa,
                            mapping[b], fb,
                        ) in self.target_edges
                        for ((a, fa), (b, fb)) in current
                    )

                    best = max(best, matches)

                    if best == maximum:
                        break

            self._last_signature = signature
            self._last_match = best

        fraction = (
            self._last_match / len(self.target_edges)
        )

        gain = max(
            0.0,
            fraction - self.best_fraction,
        )

        self.best_fraction = max(
            self.best_fraction,
            fraction,
        )

        return gain, fraction

    def potential(self, frame, *, incompatible_weight=1.0):
        """One global assignment supplies matched and incompatible edge counts."""
        _, fraction = self.observe(frame)
        wrong = (len(self._last_signature) - self._last_match) / len(self.target_edges)
        return fraction - incompatible_weight * wrong

    def prime(self, frame, *, incompatible_weight=1.0):
        """Install the measured initial potential without a reward transition.

        A partial snapshot is the episode's starting state, so its existing
        matches and incompatible edges belong on both sides of the first
        physical-step potential difference.
        """
        if not math.isfinite(incompatible_weight) or incompatible_weight < 0:
            raise ValueError('Invalid target potential configuration')
        self.current_potential = self.potential(frame, incompatible_weight=incompatible_weight)
        self._previous_match_fraction = self._last_match / len(self.target_edges)
        self._previous_wrong_fraction = (len(self._last_signature) - self._last_match) / len(self.target_edges)

    def shaping(self, frame, *, discount_factor, incompatible_weight=1.0, terminal=False):
        """Physical-step PBRS; the SMDP accumulator supplies only its usual gamma^j.

        Absorbing terminals have zero potential. Time-limit truncations do not:
        their captured next state is bootstrapped by the learner.
        """
        if not 0 <= discount_factor <= 1 or not math.isfinite(incompatible_weight) or incompatible_weight < 0:
            raise ValueError('Invalid target potential configuration')
        if terminal:
            matched = wrong = 0.0
        else:
            _, matched = self.observe(frame)
            wrong = (len(self._last_signature) - self._last_match) / len(self.target_edges)
        parts = {
            'reward_target_graph_progress': matched - self._previous_match_fraction,
            'reward_target_graph_discount': (discount_factor - 1.0) * matched,
            'reward_target_graph_incompatibility': -incompatible_weight * (
                discount_factor * wrong - self._previous_wrong_fraction),
        }
        self._previous_match_fraction, self._previous_wrong_fraction = matched, wrong
        self.current_potential = matched - incompatible_weight * wrong
        return parts


def _pair(metadata):
    if metadata.get("primitive") != "align_faces":
        return None

    connectors = metadata.get("connectors", ())

    if len(connectors) != 2:
        return None

    try:
        return edge(
            *connectors[0],
            *connectors[1],
        )
    except (ValueError, TypeError):
        return None


def _snapshot(frame, pair):
    modules = {
        m.module_id: m for m in frame.modules
    }

    positions = []
    angles = []

    for module_id, connector in pair:
        module = modules.get(module_id)

        if module is None:
            return None

        position = dict(
            module.connectors
        ).get(connector)

        if position is None:
            return None

        positions.append(
            tuple(float(v) for v in position)
        )

        joints = dict(module.joints)

        angles.extend((
            float(joints.get("pan.position_rad", 0)),
            float(joints.get("tilt.position_rad", 0)),
        ))

    if not all(
        math.isfinite(v)
        for p in positions
        for v in p
    ):
        return None

    if not all(map(math.isfinite, angles)):
        return None

    topology = tuple(sorted(
        edge(
            c.module_a,
            c.connector_a,
            c.module_b,
            c.connector_b,
        )
        for c in frame.connections
        if (c.joint_valid and c.module_a != c.module_b
            and c.connector_a in PORTS and c.connector_b in PORTS)
    ))

    return {
        "positions": tuple(positions),
        "angles": tuple(angles),
        "distance_m": math.dist(
            positions[0], positions[1]
        ),
        "height_difference_m": abs(
            positions[0][2] - positions[1][2]
        ),
        "topology": topology,
    }


def _unchanged(previous, current):
    if previous is None or current is None:
        return False

    if previous["topology"] != current["topology"]:
        return False

    motion = max(
        math.dist(a, b)
        for a, b in zip(
            previous["positions"],
            current["positions"],
        )
    )

    joint_motion = max(
        abs(a - b)
        for a, b in zip(
            previous["angles"],
            current["angles"],
        )
    )

    return (
        motion < 0.015
        and joint_motion < 0.05
    )


class AlignOutcomeFeedback:
    """Once-per-option penalties and read-only 3D diagnostics."""

    def __init__(self):
        self.started = set()
        self.terminal = set()
        self.active = {}
        self.failed_pairs = {}
        self.counts = Counter()
        self.last_events = []

    def step(self, frame, goal_metadata):
        failed = Counter()
        retries = Counter()
        stagnant = Counter()
        events = []

        # Nuove options realmente ammesse dal dispatcher.
        for goal_id, metadata in sorted(
            goal_metadata.items()
        ):
            if goal_id in self.started:
                continue

            self.started.add(goal_id)

            pair = _pair(metadata)

            if pair is None:
                continue

            owner = (
                metadata.get("owner_module_id")
                or pair[0][0]
            )

            snapshot = _snapshot(frame, pair)

            retry = _unchanged(
                self.failed_pairs.get(pair),
                snapshot,
            )

            if retry:
                retries[owner] += 1
                self.counts["unproductive_retry"] += 1

            self.active[goal_id] = {
                "pair": pair,
                "owner": owner,
                "start_s": frame.time_s,
                "start": snapshot,
                "best_distance_m": (
                    snapshot["distance_m"]
                    if snapshot else None
                ),
            }

            events.append({
                "kind": "align_started",
                "goal_id": goal_id,
                "owner": owner,
                "pair": pair,
                "time_s": frame.time_s,
                "unproductive_retry": retry,
                "geometry": snapshot,
            })

        # Osservazione geometrica delle options persistenti.
        for item in self.active.values():
            current = _snapshot(
                frame, item["pair"]
            )

            if (
                current is not None
                and item["best_distance_m"] is not None
            ):
                item["best_distance_m"] = min(
                    item["best_distance_m"],
                    current["distance_m"],
                )

        for event in frame.primitive_events:
            if event.goal_id in self.terminal:
                continue

            # Monitoraggio dei SET_TILT senza penalità.
            if (
                event.action == "set_tilt"
                and event.state == "succeeded"
            ):
                self.terminal.add(event.goal_id)
                self.counts["set_tilt_succeeded"] += 1
                continue

            if (
                event.action != "align_faces"
                or event.state not in (
                    "failed",
                    "succeeded",
                    "canceled",
                    "rejected",
                )
            ):
                continue

            self.terminal.add(event.goal_id)

            item = self.active.pop(
                event.goal_id, None
            )

            owner = (
                item["owner"]
                if item is not None
                else event.owner
            )

            record = {
                "kind": "align_terminal",
                "goal_id": event.goal_id,
                "owner": owner,
                "state": event.state,
                "code": event.code,
                "time_s": frame.time_s,
            }

            if item is not None:
                current = _snapshot(
                    frame, item["pair"]
                )

                start = item["start"]

                duration = max(
                    0.0,
                    frame.time_s - item["start_s"],
                )

                improvement = None

                if (
                    start is not None
                    and item["best_distance_m"] is not None
                ):
                    improvement = (
                        start["distance_m"]
                        - item["best_distance_m"]
                    )

                record.update({
                    "pair": item["pair"],
                    "duration_s": duration,
                    "initial_geometry": start,
                    "final_geometry": current,
                    "best_distance_improvement_m": improvement,
                })

            if event.state == "failed":
                failed[owner] += 1
                self.counts["align_failed"] += 1

                if item is not None:
                    current = _snapshot(
                        frame, item["pair"]
                    )

                    if current is not None:
                        self.failed_pairs[
                            item["pair"]
                        ] = current

                    if (
                        record["duration_s"] >= 15.0
                        and improvement is not None
                        and improvement < 0.02
                    ):
                        stagnant[owner] += 1
                        self.counts["align_stagnant"] += 1
                        record["stagnant_distance"] = True

            elif (
                event.state == "succeeded"
                and item is not None
            ):
                self.failed_pairs.pop(
                    item["pair"], None
                )

            # CANCELED_BY_CLIENT non è un fallimento.
            events.append(record)

        self.last_events = events

        return failed, retries, stagnant
