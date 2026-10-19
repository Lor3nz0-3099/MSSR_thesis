"""L1a pure H2 -> physical ActionChoice; no actuator, clock or source IO.

Runtime supplies causal pre-dispatch masks, ownership, execution tags and a
backend claim resolver. G2 admission is NOT executor feasibility. Unknown
ownership is unavailable. Masks also restrict dispatch to actual decision
points; FREE alone does not authorize a query. Nothing here schedules queries.

physics_frequency != policy_decision_frequency. Executors may advance every
physics step. A delegated claim stays BUSY until the executor releases it;
another FREE decision resource can proceed independently. HOLD_UNTIL_EXPIRY
is only a tag: no continuous horizon is certified or implemented in L1a.

Ties use first position in the supplied G2 domain (no physical-ID comparison).
Batch samples are isolated even when physical IDs repeat. Returned actions
are conflict-free proposals against a single snapshot; the future dispatcher
must atomically revalidate/reserve ownership. No WAIT/no-dispatch action exists.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
import math

import torch

from .bc_v1_contract import (
    ActionChoice, ConnectorRef, ModuleRef, ParameterValue, ResourceRef,
    robot_spec_fingerprint, validate_robot_spec,
)
from .normalization import denormalize_values, normalizer_fingerprint


class ExecutionSemantics(str, Enum):
    COMPLETION_DRIVEN = 'COMPLETION_DRIVEN'
    HOLD_UNTIL_EXPIRY = 'HOLD_UNTIL_EXPIRY'


class ResourceStatus(str, Enum):
    FREE = 'FREE'
    BUSY = 'BUSY'


Claim = ResourceRef | ConnectorRef


@dataclass(frozen=True)
class DispatchAvailability:
    """Per-sample ONLINE snapshot, never supervision masks or policy features.

    A shared executor claim is BUSY for a new exclusive writer too. The
    runtime adapter resolves shared-read compatibility before setting masks;
    this conservative FREE/BUSY view never overrides executor admission.
    """
    states: tuple[tuple[Claim, ResourceStatus], ...]
    resource_mask: tuple[bool, ...]
    structural_mask: tuple[bool, ...]

    def __post_init__(self):
        if (type(self.states) is not tuple or any(type(row) is not tuple or len(row) != 2
                or not isinstance(row[0], (ResourceRef, ConnectorRef))
                or not isinstance(row[1], ResourceStatus) for row in self.states)
                or len({ref for ref, _ in self.states}) != len(self.states)):
            raise ValueError('Invalid/duplicate resource ownership snapshot')
        for mask in (self.resource_mask, self.structural_mask):
            if type(mask) is not tuple or any(type(v) is not bool for v in mask):
                raise ValueError('Dispatch masks require explicit booleans')

    def free(self, claims):
        states = dict(self.states)
        return bool(claims) and all(states.get(ref) is ResourceStatus.FREE for ref in claims)


@dataclass(frozen=True)
class DecodedAction:
    sample_index: int
    candidate_index: int
    choice: ActionChoice
    claims: tuple[Claim, ...]
    execution: ExecutionSemantics
    # Deliberately unset: watchdog/recording cadence is not a certified horizon.
    duration_s: None = None


def _ints(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.ndim != 1:
        raise ValueError(f'Invalid integer routing: {name}')
    return value.detach().cpu().tolist()


def _values(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim != 1:
        raise ValueError(f'Invalid numeric prediction: {name}')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'Nonfinite prediction: {name}')
    return value.detach().cpu().tolist()


def _finite_tree(tree):
    if isinstance(tree, Mapping):
        for value in tree.values(): _finite_tree(value)
    elif isinstance(tree, torch.Tensor) and tree.is_floating_point() and not bool(torch.isfinite(tree).all()):
        raise ValueError('Nonfinite H2 prediction')


def _ptr(value, count, name):
    result = _ints(value, name)
    if not result or result[0] != 0 or result[-1] != count or any(a > b for a, b in zip(result, result[1:])):
        raise ValueError(f'Invalid pointer: {name}')
    return result


def _routes(record, index_name, value_name, *, ragged=True):
    indices = _ints(record[index_name], index_name)
    values = _values(record[value_name], value_name)
    ptr = _ptr(record['ptr'], len(values), 'prediction ptr') if ragged else list(range(len(indices)+1))
    if len(set(indices)) != len(indices) or len(ptr) != len(indices)+1 or ptr[-1] != len(values):
        raise ValueError('Duplicate/inconsistent prediction routing')
    return {i: values[ptr[k]:ptr[k+1]] for k, i in enumerate(indices)}


def _choose(options, scores):
    if not options or len(options) != len(scores):
        raise ValueError('Empty/inconsistent selection domain')
    return max(range(len(options)), key=lambda i: scores[i])


def _canonical(action):
    return replace(action, parameters=tuple(sorted(action.parameters, key=lambda p: p.name)))


def _inverse(values, keys, stats):
    if len(values) != len(keys) or any(key not in dict(stats.scalers) for key in keys):
        raise ValueError('Missing/inconsistent frozen F2 binding')
    physical = tuple(float(denormalize_values(v, key, stats)) for v, key in zip(values, keys))
    if not all(math.isfinite(v) for v in physical): raise ValueError('Nonfinite physical output')
    return physical


def _check_parameter(parameter, value, domain):
    kind = parameter.kind
    if kind == 'boolean':
        valid = type(value) is bool
    elif kind == 'category':
        valid = value in parameter.categories
    elif kind in ('scalar', 'vector'):
        values = (value,) if kind == 'scalar' else value
        valid = isinstance(values, tuple) and len(values) == (parameter.size or 1)
        valid = valid and all(type(v) in (int, float) and math.isfinite(v)
            and (parameter.limits is None or parameter.limits[0] <= v <= parameter.limits[1]) for v in values)
    elif kind in ('module_ref', 'connector_ref'):
        valid = value in domain.options and (domain.bound_ref is None or value == domain.bound_ref)
    else:
        valid = (isinstance(value, tuple) and len(set(value)) == len(value)
            and set(value) <= domain.options and (parameter.size is None or len(value) == parameter.size)
            and (domain.bound_ref is None or domain.bound_ref in value))
    if not valid: raise ValueError(f'Invalid physical argument {parameter.name}')


class _Arguments:
    def __init__(self, prediction, structural, module_refs, connector_refs, domains, stats):
        self.s, self.modules, self.connectors, self.domains, self.stats = structural, module_refs, connector_refs, domains, stats
        self.heads = prediction['arguments']
        self.rows = {name: _routes(row, 'argument', 'value' if name == 'numeric' else 'logits',
            ragged=name != 'boolean') for name, row in self.heads.items()}
        self.bindings = {(b.action_id, b.name): b for b in stats.argument_bindings}

    def members(self, record, row_id, lo, hi, required_mask, *, count=None):
        ids = _ints(record['argument'] if 'argument' in record else record['binding'], 'membership owners')
        if row_id not in ids: raise ValueError('Missing membership prediction')
        row = ids.index(row_id)
        scores = _values(record['logits'], 'membership scores')
        ptr = _ptr(record['ptr'], len(scores), 'membership ptr')
        indexes = _ints(record['domain_index'], 'membership domain')
        required = [i for i in range(lo, hi) if required_mask[i]]
        optional = [i for i in range(lo, hi) if not required_mask[i]]
        req = _ints(record['required_domain_index'], 'required membership')
        rp = _ptr(record['required_ptr'], len(req), 'required membership ptr')
        if (len(ptr) != len(ids)+1 or len(rp) != len(ids)+1 or len(indexes) != len(scores)
                or indexes[ptr[row]:ptr[row+1]] != optional or req[rp[row]:rp[row+1]] != required):
            raise ValueError('Membership routing differs from current domain')
        values = scores[ptr[row]:ptr[row+1]]
        if count is None:
            chosen = [i for i, score in zip(optional, values) if score > 0.]
        else:
            n = count-len(required)
            if not 0 <= n <= len(optional): raise ValueError('Invalid membership cardinality')
            chosen = sorted(optional, key=lambda i: -values[optional.index(i)])[:n]
        return sorted(required+chosen)

    def parameter(self, i, action, configured):
        domain = self.domains[i]; p = domain.parameter
        # Bound endpoints live in ActionChoice.connectors, not duplicate params.
        if domain.bound_ref is not None and p.kind == 'connector_ref': return None
        if not p.learnable:
            key = (action.action_id, p.name)
            if key not in configured: return None  # Preserve executor defaults; never infer teacher values.
            value = configured[key]
        else:
            family = {'scalar':'numeric', 'vector':'numeric', 'module_ref':'reference',
                'connector_ref':'reference', 'module_refs':'reference_set', 'connector_refs':'reference_set'}.get(p.kind, p.kind)
            if i not in self.rows[family]: raise ValueError(f'Missing predicted argument {p.name}')
            values = self.rows[family][i]
            if family == 'numeric':
                binding = self.bindings.get((action.action_id, p.name))
                if binding is None: raise ValueError('Missing F2 argument binding')
                result = _inverse(values, binding.keys, self.stats)
                value = result[0] if p.kind == 'scalar' else result
            elif family == 'boolean':
                if len(values) != 1: raise ValueError('Invalid boolean arity')
                value = values[0] > 0.
            elif family == 'category':
                value = p.categories[_choose(p.categories, values)]
            else:
                lo, hi = _ints(self.s['argument_domain_ptr'], 'argument domain ptr')[i:i+2]
                refs = self.connectors if p.kind in ('connector_ref', 'connector_refs') else self.modules
                indexes = _ints(self.s['argument_domain_index'], 'reference domain')
                record = self.heads[family]
                if family == 'reference':
                    ids = _ints(record['argument'], 'reference argument'); row = ids.index(i)
                    ptr = _ints(record['ptr'], 'reference ptr')
                    positions = _ints(record['domain_index'], 'reference positions')[ptr[row]:ptr[row+1]]
                    if positions != list(range(lo, hi)): raise ValueError('Invalid reference domain routing')
                    positions = [positions[_choose(positions, values)]]
                else:
                    positions = self.members(record, i, lo, hi,
                        self.s['argument_domain_required'].tolist(), count=p.size)
                if any(not 0 <= indexes[k] < len(refs) for k in positions): raise ValueError('Invalid reference index')
                result = tuple(refs[indexes[k]] for k in positions)
                value = result[0] if family == 'reference' else result
        _check_parameter(p, value, domain)
        return ParameterValue(p.name, value)


@torch.no_grad()
def decode_actions(prediction, inputs, routing, *, spec, normalizers,
                   availability: tuple[DispatchAvailability, ...],
                   execution_semantics: Mapping[str, ExecutionSemantics],
                   configured_arguments: Mapping[tuple[str, str], object],
                   claim_resolver: Callable[[ActionChoice], tuple[Claim, ...]]):
    """Decode proposals only. Does not query policy, dispatch, or update leases.

    Availability has one entry per sample, in G2 order. Its two masks have
    lengths equal to that sample's resource candidates and structural bindings.
    claim_resolver is a backend/runtime contract; missing claims fail closed.
    It must resolve ALL resources affected (including helpers and connectors),
    and may validate the backend-specific ActionChoice. No generic claim
    inference from PAN/TILT names or fixed connector/module counts occurs.

    Only current G2 routing fields are read; audit, supervision and targets
    incidentally retained by G2 routing are never consulted.
    """
    validate_robot_spec(spec)
    if any(key in inputs for key in ('supervision', 'targets', 'routing', 'audit')):
        raise ValueError('Pass causal G2 inputs only')
    if normalizers.metadata.get('robot_spec_fingerprint') != robot_spec_fingerprint(spec):
        raise ValueError('Frozen F2 RobotSpec mismatch')
    if routing['normalizer_fingerprint'] != normalizer_fingerprint(normalizers):
        raise ValueError('Frozen F2 identity mismatch')
    _finite_tree(prediction)
    samples = routing.get('samples', (routing,))
    if len(samples) != len(availability): raise ValueError('One availability snapshot required per sample')
    fields = ('module_refs', 'connector_refs', 'resource_refs', 'resource_candidates', 'structural_bindings', 'argument_domains')
    flat = {key: tuple(v for sample in samples for v in sample[key]) for key in fields}
    actions = tuple(sorted(spec.actions, key=lambda a: a.action_id))
    modes = tuple(sorted((t.type_id, r.resource_id, m.mode_id) for t in spec.module_types for r in t.resources for m in r.modes))
    for route in samples:
        if route['actions'] != actions or route['mode_catalog'] != modes:
            raise ValueError('Current routing/catalog mismatch')
    for family, key in (('module', 'module_refs'), ('connector', 'connector_refs'), ('resource', 'resource_refs')):
        if len(flat[key]) != len(inputs[family+'_x']): raise ValueError('Entity routing length mismatch')
    res, structural = inputs['resource_candidates'], inputs['structural']
    rrows, srows = flat['resource_candidates'], flat['structural_bindings']
    rp = _ptr(res['group_ptr'], len(rrows), 'resource groups')
    sp = _ptr(structural['group_ptr'], len(srows), 'structural groups')
    if len(rp) != len(flat['resource_refs'])+1 or len(sp) != len(flat['module_refs'])+1:
        raise ValueError('Invalid decision groups')
    rindex, mindex, aindex = (_ints(res[key], key) for key in ('resource', 'mode', 'action'))
    if any(len(v) != len(rrows) for v in (rindex, mindex, aindex)):
        raise ValueError('Resource candidate routing length mismatch')
    for i, row in enumerate(rrows):
        if (not 0 <= rindex[i] < len(flat['resource_refs']) or not 0 <= mindex[i] < len(modes)
                or not 0 <= aindex[i] < len(actions) or not row.action.enabled
                or row.resource_ref != flat['resource_refs'][rindex[i]] or row.mode.mode_id != modes[mindex[i]][2]
                or row.resource_ref.resource_id != modes[mindex[i]][1] or row.action != actions[aindex[i]]):
            raise ValueError('Invalid resource candidate/mode routing')
    for resource, (lo, hi) in enumerate(zip(rp, rp[1:])):
        if any(rindex[i] != resource for i in range(lo, hi)): raise ValueError('Invalid resource owner group')
    p = prediction['resource']
    scores = _values(p['mode_logits'], 'mode logits')
    score_ids = _ints(p['mode_candidate'], 'mode candidates')
    expected_ids = [i for lo, hi in zip(rp, rp[1:]) if hi-lo > 1 for i in range(lo, hi)]
    expected_ptr = [0]
    for lo, hi in zip(rp, rp[1:]): expected_ptr.append(expected_ptr[-1]+(hi-lo if hi-lo > 1 else 0))
    if (score_ids != expected_ids or len(scores) != len(score_ids)
            or _ints(p['mode_group_ptr'], 'mode groups') != expected_ptr):
        raise ValueError('Invalid mode score domains')
    score_map = dict(zip(score_ids, scores))
    numeric = _routes(p, 'payload_candidate', 'payload') if 'ptr' in p else _routes(
        dict(p, ptr=p['payload_ptr']), 'payload_candidate', 'payload')
    categorical = _routes(p['categorical'], 'candidate', 'logits')
    if (set(numeric) != {i for i, r in enumerate(rrows) if r.mode.codec != 'categorical'}
            or set(categorical) != {i for i, r in enumerate(rrows) if r.mode.codec == 'categorical'}):
        raise ValueError('Missing/unknown payload candidate')
    for i, row in enumerate(rrows):
        if len(categorical[i] if row.mode.codec == 'categorical' else numeric[i]) != (
                len(row.mode.categories) if row.mode.codec == 'categorical' else row.mode.payload_size):
            raise ValueError('Payload arity differs from catalog')
    sscores = _values(prediction['structural']['candidate_logits'], 'structural scores')
    if len(sscores) != len(srows) or _ints(prediction['structural']['group_ptr'], 'H2 structural groups') != sp:
        raise ValueError('Invalid structural score domains')
    none_scores=None
    if 'no_structural_logits' in prediction['structural']:
        none_scores=_values(prediction['structural']['no_structural_logits'],'NO_STRUCTURAL logits')
        if len(none_scores)!=len(sp)-1:raise ValueError('Invalid NO_STRUCTURAL groups')
    args = _Arguments(prediction['structural'], structural, flat['module_refs'], flat['connector_refs'], flat['argument_domains'], normalizers)
    bindings = {(b.module_type_id, b.resource_id, b.mode_id): b for b in normalizers.resource_bindings}
    out = []; resource_offset = structural_offset = module_offset = candidate_offset = connector_offset = 0
    for sample_index, (route, av) in enumerate(zip(samples, availability)):
        if len(av.resource_mask) != len(route['resource_candidates']) or len(av.structural_mask) != len(route['structural_bindings']):
            raise ValueError('Dispatch mask domain mismatch')
        reserved = set()

        def append(choice, candidate):
            tag = execution_semantics.get(choice.action_id)
            if not isinstance(tag, ExecutionSemantics): raise ValueError('Missing explicit execution semantics')
            claims = claim_resolver(choice)
            if (not isinstance(claims, tuple) or not claims or len(set(claims)) != len(claims)
                    or any(not isinstance(ref, (ResourceRef, ConnectorRef)) for ref in claims)
                    or choice.resource is not None and choice.resource not in claims):
                raise ValueError('Incomplete/invalid backend resource claims')
            if av.free(claims) and not reserved.intersection(claims):
                reserved.update(claims)
                out.append(DecodedAction(sample_index, candidate, choice, claims, tag))

        for local, ref in enumerate(route['resource_refs']):
            lo, hi = rp[resource_offset+local:resource_offset+local+2]
            options = [i for i in range(lo, hi) if av.resource_mask[i-candidate_offset]]
            if not av.free((ref,)) or not options: continue
            selected = options[_choose(options, [score_map.get(i, 0.) for i in options])]
            row = rrows[selected]; mode = row.mode
            if mode.codec == 'categorical':
                payload = (float(_choose(mode.categories, categorical[selected])),)
            else:
                binding = bindings.get(modes[mindex[selected]])
                if binding is None or binding.units != mode.units or binding.codec != mode.codec:
                    raise ValueError('Missing/incompatible F2 resource binding')
                payload = _inverse(numeric[selected], binding.keys, normalizers)
                limited = [p for p in row.action.parameters if p.limits is not None]
                if limited:
                    numeric_params = [p for p in row.action.parameters if p.kind in ('scalar', 'vector')]
                    if len(numeric_params) != 1 or (numeric_params[0].size or 1) != len(payload):
                        raise ValueError('Ambiguous resource parameter limit mapping')
                    low, high = numeric_params[0].limits
                    if any(not low <= value <= high for value in payload):
                        raise ValueError('Physical resource payload outside declared limits')
            append(ActionChoice(row.action.action_id, ref.module, resource=ref, payload=payload), selected)

        for local, ref in enumerate(route['module_refs']):
            lo, hi = sp[module_offset+local:module_offset+local+2]
            options = [i for i in range(lo, hi) if av.structural_mask[i-structural_offset]]
            if not options: continue
            selected = options[_choose(options, [sscores[i] for i in options])]
            if none_scores is not None and none_scores[module_offset+local]>=sscores[selected]:
                continue  # Policy abstention: no physical choice, claims, goal or cancellation.
            action, bound = srows[selected]
            if (not action.enabled or bound.initiator != ref
                    or int(structural['initiator'][selected]) != module_offset+local
                    or not 0 <= int(structural['action'][selected]) < len(actions)
                    or _canonical(actions[int(structural['action'][selected])]) != action):
                raise ValueError('Invalid structural binding/group')
            ports = bound.connectors
            if len(ports) not in (0, 2): raise ValueError('Invalid structural endpoint arity')
            expected_ports = tuple(connector_offset+route['connector_refs'].index(port) for port in ports) if ports else (-1, -1)
            expected_peer = module_offset+route['module_refs'].index(ports[1].module) if ports else -1
            if (int(structural['connector_a'][selected]), int(structural['connector_b'][selected])) != expected_ports or int(structural['peer'][selected]) != expected_peer:
                raise ValueError('Structural endpoints differ from current routing')
            start, end = _ints(structural['scope_ptr'], 'scope ptr')[selected:selected+2]
            required = structural['scope_required'].tolist()
            scope = list(range(start, end)) if all(required[start:end]) else args.members(
                prediction['structural']['scope'], selected, start, end, required)
            members = _ints(structural['scope_members'], 'scope members')
            participants = tuple(flat['module_refs'][members[i]] for i in scope)
            if ref not in participants or not set(participants) <= bound.scope_members:
                raise ValueError('Invalid decoded participant scope')
            parameter_indices = [i for i, owner in enumerate(_ints(structural['argument_binding'], 'argument binding')) if owner == selected]
            parameters = tuple(value for i in parameter_indices
                if (value := args.parameter(i, action, configured_arguments)) is not None)
            append(ActionChoice(action.action_id, ref, participants=participants,
                connectors=bound.connectors, parameters=parameters), selected)
        resource_offset += len(route['resource_refs']); module_offset += len(route['module_refs'])
        structural_offset += len(route['structural_bindings']); candidate_offset += len(route['resource_candidates'])
        connector_offset += len(route['connector_refs'])
    return tuple(out)
