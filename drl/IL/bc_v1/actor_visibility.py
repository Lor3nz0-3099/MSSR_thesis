"""Immutable, catalog-independent actor projection of certified E2 geometry.

Full causal physical views remain upstream. This module neither consumes labels
nor changes robot state, catalogs, feature layouts or model architecture. The
accepted distance is E2's point/line/plane metric, not finite collider clearance.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path

from .feature_schema import (
    FeatureView, _closest_displacement, _field_payload, _module_environment_relations,
    _relation_targets, _source_parts,
)

VERSION = 'mssr.bc_v1.actor_visibility.v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class VisibilityPolicy:
    serialized: str

    @property
    def document(self):
        return json.loads(self.serialized)

    @property
    def fingerprint(self):
        return hashlib.sha256(self.serialized.encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        keys = {'schema_version', 'revision', 'radius_m', 'threshold', 'scope',
                'empirical_reference_mean_m', 'radius_selection', 'goal_policy',
                'prefer_dynamic_object_types', 'critic_only', 'local_compound_parts',
                'attribute_rules'}
        if (set(value) != keys or value['schema_version'] != VERSION or not value['revision']
                or value['threshold'] != '<=' or value['scope'] != 'team_union_wireless'
                or value['goal_policy'] != 'certified_current_assigned_physical_goal_exempt'):
            raise ValueError('Unsupported actor visibility contract')
        for name in ('radius_m', 'empirical_reference_mean_m'):
            if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or value[name] <= 0:
                raise ValueError('Invalid actor radius/provenance')
        for name in ('prefer_dynamic_object_types', 'critic_only', 'local_compound_parts'):
            values = value[name]
            if (not isinstance(values, list) or len(values) != len(set(values))
                    or any(type(v) is not str or not v for v in values)):
                raise ValueError('Invalid actor visibility field list')
        seen = set()
        for rule in value['attribute_rules']:
            if set(rule) != {'names', 'object_types', 'object_refs'}:
                raise ValueError('Invalid local attribute rule')
            for values in rule.values():
                if not isinstance(values, list) or any(type(v) is not str or not v for v in values):
                    raise ValueError('Invalid local attribute names')
            if not rule['names'] or not (rule['object_types'] or rule['object_refs']) or seen.intersection(rule['names']):
                raise ValueError('Ambiguous local attribute gating')
            seen.update(rule['names'])
        return cls(_json(value))


def load_visibility_policy(path):
    return VisibilityPolicy.from_dict(json.loads(Path(path).read_text()))


def _goal_signature(view):
    return _digest([(row.source_scope, row.source_name, row.primitive_part_ref,
                     row.geometry_kind, row.values, row.validity)
                    for row in view.environment if row.object_type == 'goal'])


@dataclass(frozen=True)
class AssignedGoalCertificate:
    """Trusted mission-provider receipt, outside all learned policy features."""
    goal_signature: str
    provider_identity: str
    source_contract_id: str
    evidence: tuple[str, ...]

    def __post_init__(self):
        if (not all(type(v) is str and v for v in (
                self.goal_signature, self.provider_identity, self.source_contract_id))
                or type(self.evidence) is not tuple or not self.evidence
                or any(type(v) is not str or not v for v in self.evidence)):
            raise ValueError('Assigned goal requires provider/contract/evidence')


def certify_assigned_goal(view, *, provider_identity, source_contract_id, evidence):
    """Called ONLY by an explicit causal mission/source-contract provider.

    Mere goal-token presence never invokes this automatically. The signature
    prevents a receipt being reused for different current goal measurements.
    """
    if not any(r.object_type == 'goal' for r in view.environment):
        raise ValueError('No current physical goal to certify')
    return AssignedGoalCertificate(_goal_signature(view), provider_identity,
                                   source_contract_id, evidence)


def project_actor_features(view, task, policy, schema, *, assigned_goal=None):
    """Full certified current FeatureView -> actor-only immutable FeatureView.

    Team union uses every valid current module, irrespective of physical edges.
    Compound parts are range-tested separately. Local attributes require a
    visible corresponding physical object/task geometry. No padding/token for
    excluded objects is retained. Relations are rebuilt with E2's ONE algorithm
    to preserve nearest-per-type semantics and exact source references.
    """
    if type(view) is not FeatureView or type(policy) is not VisibilityPolicy:
        raise TypeError('Expected typed physical view and actor visibility policy')
    config = policy.document
    if assigned_goal is not None:
        if (type(assigned_goal) is not AssignedGoalCertificate
                or assigned_goal.goal_signature != _goal_signature(view)):
            raise ValueError('Assigned goal certificate does not match current goal')
    denied = set(config['critic_only'])
    # Global support/calibration quantities are never visibility anchors.
    environment = tuple(r for r in view.environment if _source_parts(r.source_name)[1] not in denied)
    allowed = schema['relation_policy']['object_types_by_task'].get(task, ())
    targets = _relation_targets(environment, allowed)
    positions = tuple(p for row in view.modules if (p := _field_payload(row, 'relative_position')) is not None)
    metrics = {}
    for key, parts in targets.items():
        metrics[key] = tuple((target, min((result[1] for pos in positions
            if (result := _closest_displacement(pos, target)) is not None), default=math.inf))
            for target in parts)
    # Current observation takes precedence over a stale nominal fixture pose.
    dynamic = {(kind, ref) for kind, scope, ref in metrics
               if scope == 'o_dynamic' and kind in config['prefer_dynamic_object_types']}
    admitted = set()
    visible = set()
    for key, parts in metrics.items():
        kind, scope, ref = key
        if kind == 'goal':
            if assigned_goal is not None:
                admitted.update(source for target, _ in parts for source in target.sources)
                visible.add((kind, ref))
            continue
        primary = (kind, 'o_dynamic', ref) if (kind, ref) in dynamic else key
        if not any(distance <= config['radius_m'] for _, distance in metrics[primary]):
            continue
        for target, distance in parts:
            if distance <= config['radius_m']:
                admitted.update(target.sources)
                visible.add((kind, ref))
    types = {kind for kind, _ in visible}
    refs = {ref for _, ref in visible}
    rules = {name: rule for rule in config['attribute_rules'] for name in rule['names']}
    retained = []
    for row in environment:
        base = _source_parts(row.source_name)[1]
        if row.object_type == 'goal':
            if assigned_goal is not None:
                retained.append(row)
        elif row.ref in admitted:
            retained.append(row)
        elif base in config['local_compound_parts']:
            # Preserve only parts explicitly admitted above, never the full inventory.
            continue
        elif row.geometry_kind in ('point3', 'line', 'plane'):
            continue
        elif (row.object_type, row.object_ref) in visible:
            retained.append(row)
        elif base in rules:
            rule = rules[base]
            if types.intersection(rule['object_types']) or refs.intersection(rule['object_refs']):
                retained.append(row)
    retained = tuple(retained)
    relations = _module_environment_relations(view.modules, retained, task, schema)
    return replace(view, environment=retained, relations=relations)
