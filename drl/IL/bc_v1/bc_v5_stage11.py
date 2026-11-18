"""Stage 11: rebuild a distinct actor-visible G2 with topology-gated H2 candidates.

The H1 inputs are copied BYTE/TENSOR-IDENTICALLY from the authenticated BC-v4
actor-visible G2; only candidate routing and the corresponding target indices
are regenerated from the independently authenticated G1 physical state.

47 exceptional *historical* align labels may restore the full legacy align
candidate domain for their exact TRAIN/VALIDATION row. No online exception.
No TEST payload is ever opened; no upstream artifact is modified.

This command is a dataset PREPARATION experiment, NOT a training readiness
certificate. Canonical/G2 integrity and native conflict masks remain required;
an independent temporal audit is optional.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from . import dataset as g1
from .action_set_targets import candidate_index, positive_indices, target_view
from .bc_v1_contract import load_robot_spec
from .bc_v5_integrity import read_report
from .bc_v5_offline_align_exceptions import read_registry
from .bc_v5_topology_candidates import enumerate_candidates_v5
from .candidates import CandidateTables, StructuralCandidate
from .feature_schema import load_feature_schema
from .sidecar import SidecarDataset, SidecarWriter, _hash, _publish_manifest, assert_exact, _read_manifest, verify_sidecar
from .tensorization import _candidate_inputs, _target_tensors

SCHEMA = 'mssr.bc_v5.topology_rebuild.v1'
SPLITS = ('train', 'validation')
PRIMITIVES = ('pan_delta', 'pan_target', 'pan_velocity', 'tilt_delta',
              'tilt_target', 'tilt_velocity', 'wheel_velocity',
              'align_faces', 'dock', 'gravity_settle', 'undock')


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _source_identity(g2, g1root, *, registry, overlay, census):
    original = json.loads((Path(g2) / 'manifest.json').read_text())
    old = _read_manifest(Path(g2), original['identity'])
    if old.get('actor_visibility') is None or old['inventory']['test']:
        raise ValueError('Stage 11 requires a TEST-free actor-visible source G2')
    if old['identity']['canonical_sha'] != registry['canonical_sha256']:
        raise ValueError('Stage 10 registry Canonical identity disagrees with G2')
    if old['manifest_fingerprint'] != registry['g2_manifest_fingerprint']:
        raise ValueError('Stage 10 registry refers to another source G2')
    audit = read_report(overlay, expected_g2_fingerprint=old['manifest_fingerprint'], ready=False)
    if not audit['complete'] or audit['expected_rows'] != {
            part: sum(e['samples'] for e in old['inventory'][part]) for part in SPLITS}:
        raise ValueError('Full Canonical/G2 Stage-5 audit required')
    census_data = json.loads(Path(census).read_text())
    if (not census_data.get('complete')
            or census_data.get('g2_manifest_fingerprint') != old['manifest_fingerprint']
            or census_data.get('canonical_sha256') != registry['canonical_sha256']
            or census_data.get('test_payload_access') != 0):
        raise ValueError('Stage-6 baseline census/source identity mismatch')
    cache = g1.read_cache_manifest(g1root)
    if cache['cache_fingerprint'] != old['identity']['g1_fingerprint']:
        raise ValueError('Source G1 is not G2 physical cache')
    for part in SPLITS:
        source_episodes = [e['episode_id'] for e in old['inventory'][part]]
        if source_episodes != cache['splits'][part]:
            raise ValueError('G1/G2 split membership mismatch')
        for entry in old['inventory'][part]:
            physical = cache['episodes'][entry['episode_id']]
            if (entry['samples'] != physical['samples']
                    or entry['transition_indices_fingerprint'] != physical['transition_indices_fingerprint']):
                raise ValueError('G1/G2 episode index mismatch')
    return old, cache, audit, census_data


def _check_exception_coverage(registry, inventory):
    episodes = {part: {e['episode_id'] for e in inventory[part]} for part in SPLITS}
    seen = set()
    for e in registry['exceptions']:
        key = (e['split'], e['episode_id'], e['transition_index'], e['initiator'])
        if e['episode_id'] not in episodes[e['split']] or key in seen:
            raise ValueError('Out-of-split or duplicate exception')
        seen.add(key)
    return seen


def _primitives(route):
    return ['resource:' + c.mode.mode_id for c in route['resource_candidates']] + [
        'structural:' + action.action_id for action, binding in route['structural_bindings']]


def row_count(route, blocked=()):
    """One and only one label per candidate; negatives include all unchosen."""
    names = _primitives(route)
    positive = set(positive_indices(route))
    if len(positive) != len(route['physical_supervision'].resource_targets) + len(route['physical_supervision'].structural_targets):
        raise ValueError('Positive target identity mismatch')
    masked = set()
    for mid, kind in blocked:
        for i, row in enumerate(route['resource_candidates']):
            if row.resource_ref.module.module_id == mid and (
                ('wheels' if row.resource_ref.resource_id == 'locomotion' else
                 'internal' if row.resource_ref.resource_id == 'internal_motion' else None) == kind
            ):
                masked.add(i)
        for i, (_action, binding) in enumerate(route['structural_bindings']):
            if binding.initiator.module_id == mid and kind == 'structural':
                masked.add(len(route['resource_candidates'])+i)
    if masked.intersection(positive):
        raise ValueError('Attempt to mask positive BC-v5 target')
    result = defaultdict(Counter)
    for i, name in enumerate(names):
        result[name]['positive' if i in positive else 'masked' if i in masked else 'negative'] += 1
    return result


def merge_counts(dst, row):
    for primitive, fields in row.items():
        dst[primitive].update(fields)


def _restore_historical_align_choices(table, original_routing, required_initiators):
    """Offline row-scoped fallback to ALL old align alternatives, never one positive.

    Reuse the audited historical G2 candidate bindings, not a new action
    synthesized from expert target values. Existing v5 align bindings win;
    additional historical bindings get appended with exact former semantics.
    A one-bucket stale link may make the *old graph* seem same-component;
    that is precisely the exception being recorded, not a live permission.
    """
    rows = list(table.structural_candidates)
    historical = [(a, b) for a, b in original_routing['structural_bindings']
                  if a.action_id == 'align_faces']
    former = {(a.action_id, b.initiator, b.connectors) for a,b in historical}
    targets = [t for t in original_routing['physical_supervision'].structural_targets
               if t.operation_id == 'align_faces' and t.initiator.module_id in required_initiators]
    for initiator in required_initiators:
        matching = [t for t in targets if t.initiator.module_id == initiator]
        if len(matching) != 1:
            raise ValueError('Exception must have one exact align positive for each initiator')
        t = matching[0]
        if ('align_faces', t.initiator, t.connectors) not in former:
            raise ValueError('Exception missing from historical BC-v4 align domain')
    for action, binding in historical:
        i = next((i for i, c in enumerate(rows)
                  if c.action.action_id == action.action_id
                  and c.endpoints == frozenset(binding.connectors)), None)
        if i is None:
            rows.append(StructuralCandidate(action, frozenset(binding.connectors), frozenset((binding,))))
        else:
            rows[i] = replace(rows[i], bindings=rows[i].bindings | frozenset((binding,)))
    return CandidateTables(table.resource_candidates, tuple(rows))


class _TopologyLRU:
    def __init__(self, spec, schema):
        self.spec, self.schema = spec, schema
        self.cache = OrderedDict()

    def get(self, graph):
        key = (
            tuple((m.ref, m.module_type_id) for m in graph.modules),
            tuple((c.ref, c.kind, c.status, c.connected_to) for c in graph.connectors),
            tuple((r.ref, tuple(co.coordinate_id for co in r.coordinates)) for r in graph.resources),
            tuple((e.endpoint_a, e.endpoint_b, e.relation_id) for e in graph.connections),
        )
        found = self.cache.get(key)
        if found is None:
            found = enumerate_candidates_v5(graph, self.spec, self.schema)
            self.cache[key] = found
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return found


def _make_row(old, physical_sample, *, split, registry, exceptional_keys, spec, schema, lru):
    """No actor feature/normalizer computation: only E2 candidate repacking."""
    assert_exact(old['routing']['audit'], physical_sample.audit)
    assert_exact(old['routing']['physical_supervision'], physical_sample.supervision)
    graph = physical_sample.inputs.policy.graph
    eid, tid = physical_sample.audit.episode_id, physical_sample.audit.transition_index
    exceptional = [key for key in exceptional_keys if key[:3] == (split, eid, tid)]
    table = lru.get(graph)
    restoration = False
    if exceptional:
        required_initiators = {key[3] for key in exceptional}
        v5_matches = {('align_faces', b.initiator, b.connectors)
                      for c in table.structural_candidates
                      if c.action.action_id == 'align_faces' for b in c.bindings}
        target_keys = {('align_faces', t.initiator, t.connectors)
                       for t in old['routing']['physical_supervision'].structural_targets
                       if t.operation_id == 'align_faces' and
                          t.initiator.module_id in required_initiators}
        if (len(target_keys) != len(required_initiators)
                or {key[1].module_id for key in target_keys} != required_initiators):
            raise ValueError('Exact Stage-10 exception positives do not match row')
        if not target_keys <= v5_matches:
            table = _restore_historical_align_choices(table, old['routing'], required_initiators)
            restoration = True

    route = dict(old['routing'])
    indices = {name: {ref: i for i, ref in enumerate(route[name+'_refs'])}
               for name in ('module', 'connector', 'resource')}
    resource, structural = _candidate_inputs(table, spec, indices, route)
    # BC-v4 actor G2 adds resource outcome heads; do NOT remove/replace them.
    # Assert all common resource candidate tensors, not just numerical shapes.
    for name, tensor in resource.items():
        assert_exact(tensor, old['inputs']['resource_candidates'][name], 'unchanged resource '+name)
    assert_exact(route['resource_candidates'], old['routing']['resource_candidates'], 'resource candidate catalog')
    inputs = {**old['inputs'], 'structural': structural}
    route['bc_v5_candidate_contract'] = SCHEMA
    # Original v4 auxiliary decision indices may refer to the old structural
    # catalog; this new G2 is exclusively for BC-v5 action_set training.
    targets = _target_tensors(route['resource_targets'], route['structural_targets'],
                              indices, route, inputs)
    fresh = {'inputs': inputs, 'supervision': targets, 'routing': route}
    # This covers ALL resource + structural modes, not align alone.
    new_positive = tuple(positive_indices(route))
    old_positive = tuple(positive_indices(old['routing']))
    if len(new_positive) != len(old_positive):
        raise ValueError('Lost or duplicated a physical positive target')
    from collections import Counter as _Counter
    if _Counter(_primitives(route)[i] for i in new_positive) != _Counter(
            _primitives(old['routing'])[i] for i in old_positive):
        raise ValueError('Positive primitive changed')
    # Verify H2's actual collate path regenerates targets correctly.
    target_view(fresh)
    for name in old['inputs']:
        if name != 'structural':
            assert_exact(fresh['inputs'][name], old['inputs'][name], 'unchanged H1/resource '+name)
    return fresh, restoration


def build(g2, g1root, registry_file, overlay_file, baseline_file, output, report_file,
          *, pilot_episodes=0, spec_file=None, schema_file=None, progress=None):
    if type(pilot_episodes) is not int or pilot_episodes < 0:
        raise ValueError('Pilot episodes must be nonnegative integer')
    g2 = Path(g2).resolve(); g1root = Path(g1root).resolve()
    output = Path(output).resolve(); report_file = Path(report_file).resolve()
    for source in (g2, g1root, Path(registry_file).resolve(), Path(overlay_file).resolve(), Path(baseline_file).resolve()):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('New G2 overlaps an existing source')
    if report_file.exists() or output.exists() or output.with_name(output.name+'.building').exists():
        raise ValueError('Refusing to overwrite an audit or a dataset')
    if not report_file.parent.exists():
        report_file.parent.mkdir(parents=True, exist_ok=True)
    registry = read_registry(registry_file)
    old_doc, physical_doc, overlay, baseline = _source_identity(
        g2, g1root, registry=registry, overlay=overlay_file, census=baseline_file)
    exceptional_keys = _check_exception_coverage(registry, old_doc['inventory'])
    base = Path(__file__).resolve().parents[3]
    spec_file = Path(spec_file) if spec_file is not None else base/'configs/bc_v1/smores_ep.robot.json'
    schema_file = Path(schema_file) if schema_file is not None else base/'configs/bc_v1/features.json'
    spec, schema = load_robot_spec(spec_file), load_feature_schema(schema_file)
    # A source mismatch must fail BEFORE writing the new G2, not when training.
    from .bc_v1_contract import robot_spec_fingerprint
    from .feature_schema import feature_schema_fingerprint
    if (robot_spec_fingerprint(spec) != old_doc['identity']['robot_spec_fingerprint'] or
            feature_schema_fingerprint(schema) != old_doc['identity']['feature_schema_fingerprint']):
        raise ValueError('Robot spec/feature schema identity mismatch')
    selected = {part: (old_doc['inventory'][part][:pilot_episodes] if pilot_episodes else
                       old_doc['inventory'][part]) for part in SPLITS}
    inventory = {'train': selected['train'], 'validation': selected['validation'], 'test': []}
    # No TEST access, including the source dataset reader.
    identity = dict(old_doc['identity'])
    code_hash = _sha_file(__file__)
    identity['tensorizer_fingerprint'] = _hash({'upstream': identity['tensorizer_fingerprint'],
        'stage11': SCHEMA, 'code': code_hash, 'exceptions': registry['fingerprint']})
    identity['storage_fingerprint'] = _hash({'upstream': identity['storage_fingerprint'],
        'stage11': SCHEMA, 'code': code_hash, 'exceptions': registry['fingerprint']})
    overlay_rows = {part: {} for part in SPLITS}
    for row in overlay['rows']:
        if row['split'] in overlay_rows:
            key = (row['episode_id'], row['transition_index'])
            if key in overlay_rows[row['split']]:
                raise ValueError('Duplicate Stage-5 overlay row')
            overlay_rows[row['split']][key] = row
    reporter = {part: defaultdict(Counter) for part in SPLITS}
    by_row_exception = defaultdict(set)
    for split, eid, tid, mid in exceptional_keys:
        by_row_exception[(split, eid, tid)].add(mid)
    restored = Counter(); completed_exceptions = set(); total = Counter()
    writer = SidecarWriter(output, identity, inventory)
    try:
        for part in SPLITS:
            index = 0
            with SidecarDataset(g2, part, expected_identity=old_doc['identity'],
                                template_cache_size=16) as old_data:
                for episode in selected[part]:
                    eid = episode['episode_id'];lru = _TopologyLRU(spec, schema)
                    samples = g1._load_episode(g1root, eid, physical_doc['episodes'][eid])
                    if len(samples) != episode['samples']:
                        raise ValueError('Incomplete source G1 episode '+eid)
                    seen = []
                    for sample in samples:
                        old = old_data[index]
                        seen.append(sample.audit.transition_index)
                        key = eid, sample.audit.transition_index
                        if key not in overlay_rows[part]:
                            raise ValueError('Stage-5 Canonical overlay row missing')
                        fresh, exception_used = _make_row(old, sample, split=part,
                            registry=registry, exceptional_keys=exceptional_keys,
                            spec=spec, schema=schema, lru=lru)
                        for mid in by_row_exception.get((part, eid, key[1]), ()):
                            completed_exceptions.add((part, eid, key[1], mid))
                        if exception_used:
                            restored[part] += 1
                        per = row_count(fresh['routing'],overlay_rows[part][key]['blocked'])
                        merge_counts(reporter[part], per)
                        writer.add(fresh, part, index)
                        index += 1; total[part] += 1
                        if progress and total[part] % 1000 == 0:
                            progress({'split': part, 'rows': total[part],
                                      'stage': 'repacking H2 candidates'})
                    if len(seen) != episode['samples'] or _hash(seen) != episode['transition_indices_fingerprint']:
                        raise ValueError('G1/G2 transition index mismatch')
                if not pilot_episodes and index != len(old_data):
                    raise ValueError('Not all actor-visible source rows visited')
        required = {key for key in exceptional_keys if any(
            key[1] == e['episode_id'] for e in selected[key[0]])}
        if completed_exceptions != required:
            raise ValueError('Exception registry rows not covered')
        for part in SPLITS:
            if not pilot_episodes:
                expected = baseline['splits'][part]['per_primitive']
                for name in expected:
                    if reporter[part][name]['positive'] != expected[name].get('positive', 0):
                        raise ValueError('Stage 6 positive count changed: '+part+'/'+name)
        manifest = writer.finalize()
    finally:
        writer.close()
    # G2 source provenance remains actor-visible while new candidates carry
    # an independent v5 content identity.
    manifest = _publish_manifest(output, {**manifest,
        'actor_visibility': old_doc['actor_visibility'],
        # These targets/candidate indices are NEW; never advertise the old
        # auxiliary v4 resource/structural decision certification.
        'tensor_sample_schema': 'mssr.bc_v5.tensor_sample.actor_visible.topology.v1',
        'bc_v5_topology': {'schema_version': SCHEMA, 'source_manifest_fingerprint': old_doc['manifest_fingerprint'],
            'offline_registry_fingerprint': registry['fingerprint'],
            'source_overlay_fingerprint': json.loads(Path(overlay_file).read_text())['fingerprint'],
            'pilot_only': bool(pilot_episodes), 'runtime_exception': False,
            'temporal_authenticated': False, 'ready_for_training': False}})
    if not pilot_episodes:
        # Deep-verify the newly published G2 instead of copying BC-v4's PASS.
        verification = verify_sidecar(output, expected_identity=identity)
        if verification['splits'] != {**dict(total), 'test': 0} or verification['duplicates']:
            raise ValueError('New actor-visible G2 failed deep verification')
        old_success_file = g2/'SUCCESS.json'
        if not old_success_file.is_file():
            raise ValueError('Original actor-visible certification SUCCESS.json missing')
        old_success = json.loads(old_success_file.read_text())
        if (old_success.get('manifest_fingerprint') != old_doc['manifest_fingerprint']
                or old_success.get('TEST_payload_access') != 0
                or old_success.get('fingerprint') != old_doc['actor_visibility']['fingerprint']):
            raise ValueError('Original actor-visible success source is not authenticated')
        success = {**old_success,
            'manifest_fingerprint': manifest['manifest_fingerprint'],
            'verification': {**old_success['verification'],
                'integrity': 'PASS', 'manifest_fingerprint': manifest['manifest_fingerprint'],
                'TEST_payload_access': 0,
                'bc_v5_topology_verification': verification,
                'support': {'note': 'BC-v5 ActionSet target preservation verified, auxiliary v4 resource decisions NOT certified'}},
            'sample_counts': dict(total),
            'candidates_preserved': False,
            'physical_labels_preserved': True,
            'resource_and_structural_decisions_preserved': False,
            'bc_v5_topology': manifest['bc_v5_topology'],
            'support': {'note': 'BC-v5 ActionSet positive membership only; legacy v4 auxiliary decisions not re-certified'}}
        (output/'SUCCESS.json').write_text(json.dumps(success, indent=2, sort_keys=True)+'\n')
    else:
        verification = {'not_run': 'pilot-only sidecar cannot certify training'}
    summary = {'schema_version': SCHEMA, 'complete': not pilot_episodes,
        'old_g2_fingerprint': old_doc['manifest_fingerprint'],
        'new_g2_fingerprint': manifest['manifest_fingerprint'],
        'new_g2': str(output), 'test_payload_access': 0,
        'ready_for_training': False, 'temporal_authenticated': False,
        'old_candidate_counts': {part: {p: dict(vals) for p, vals in baseline['splits'][part]['per_primitive'].items()}
                                 for part in SPLITS},
        'new_candidate_counts': {part: {p: dict(vals) for p, vals in sorted(reporter[part].items())}
                                 for part in SPLITS},
        'source_rows': dict(total), 'offline_exception_rows_restored': dict(restored),
        'offline_exception_instances_visited': len(completed_exceptions),
        'h1_features_unchanged': True,
        'positive_retention_pass': not pilot_episodes,
        'deep_verification': verification,
        'runtime_exception': False,
        'note': ('Offline exception rows restore the entire historic align choice domain; '
                 'this is a deliberate label-dependent train/live candidate-distribution difference. '
                 'No temporal certification or training permission is implied.')}
    summary['reduction_vs_baseline'] = {
        part: {name: {
            'old': old_counts.get('positive',0)+old_counts.get('negative',0)+old_counts.get('masked',0),
            'new': sum(reporter[part][name].values()),
            'removed': (old_counts.get('positive',0)+old_counts.get('negative',0)+old_counts.get('masked',0)) - sum(reporter[part][name].values())
        } for name, old_counts in baseline['splits'][part]['per_primitive'].items()}
        for part in SPLITS} if not pilot_episodes else {}
    summary['fingerprint'] = _hash(summary)
    report_file.write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    p.add_argument('--g2', type=Path, default=root/'artifacts/bc_v1/sidecar/g2_v4_actor_visible_r1m')
    p.add_argument('--g1', type=Path, default=root/'artifacts/bc_v1/cache/g1_v1_optimized')
    p.add_argument('--exceptions', type=Path, default=root/'artifacts/bc_v1/audits/bc_v5_stage10/offline_align_exceptions.json')
    p.add_argument('--overlay', type=Path, default=root/'artifacts/bc_v1/audits/bc_v5_stage6/integrity_full_20261112_155507.json')
    p.add_argument('--baseline-census', type=Path, default=root/'artifacts/bc_v1/audits/bc_v5_stage6/candidate_counts_20261112_155507.json')
    p.add_argument('--output', type=Path, default=root/'artifacts/bc_v1/sidecar/g2_v5_topology_v1')
    p.add_argument('--report', type=Path, default=root/'artifacts/bc_v1/audits/bc_v5_stage11/comparison.json')
    p.add_argument('--pilot-episodes', type=int, default=0, help='Use first N episodes of EACH split (non-training pilot)')
    args = p.parse_args(argv)
    result = build(args.g2,args.g1,args.exceptions,args.overlay,args.baseline_census,args.output,args.report,
                   pilot_episodes=args.pilot_episodes,progress=lambda r: print(json.dumps(r,sort_keys=True),flush=True))
    print(json.dumps({k: result[k] for k in ('new_g2', 'source_rows', 'ready_for_training',
          'test_payload_access', 'positive_retention_pass', 'offline_exception_instances_visited')},sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
