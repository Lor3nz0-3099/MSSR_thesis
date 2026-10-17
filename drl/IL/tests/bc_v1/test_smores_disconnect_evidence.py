"""Historical ordering proofs, mixed buckets and unchanged E2 domains."""
import copy
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.tests.bc_v1.test_causal_topology import graph


ROOT = Path(__file__).resolve().parents[4]
FIXTURE = Path(__file__).parent/'fixtures/causal_topology.v1.json'


def api():
    name = 'drl.IL.bc_v1.smores_disconnect_evidence'
    assert importlib.util.find_spec(name) is not None, 'Missing Task 3 historical disconnect provider'
    return importlib.import_module(name)


def policy():
    return json.loads((ROOT/'configs/bc_v1/causal_topology.json').read_text())


def refs(root, rows, bucket, lines, *, run='run-A'):
    root.mkdir(parents=True, exist_ok=True)
    source = root/'dispatch.jsonl'; log = root/'isaac.log'
    source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    log.write_text('\n'.join(lines)+'\n')
    return {'run_id': run,
            'runtime_log': {'path': 'isaac.log', 'run_id': run, 'sha256': hashlib.sha256(log.read_bytes()).hexdigest()},
            'source_files': [{'path': 'dispatch.jsonl', 'run_id': run,
                              'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}],
            'source_records': [{'path': 'dispatch.jsonl', 'record': n} for n in bucket]}


def dispatch(goal, primitive='undock', **extra):
    return dict(expert_action={'primitive_goal': {'goal_id': goal, 'primitive': primitive}}, **extra)


def accepted(goal):
    return f'[primitive] ACCEPTED {goal}: primitive goal accepted'


def detached(goal='old', left='alpha:socket-X', right='beta:socket-X'):
    return f'[primitive] SUCCEEDED {goal}: UNDOCKED detached {left} from {right}'


def simple(root, completion=None):
    return refs(root, [dispatch('old'), dispatch('later', 'align_faces')], [2],
                [accepted('old'), detached() if completion is None else completion, accepted('later')])


def batch(root, data):
    return api().HistoricalSmoresDisconnectProvider(root, policy()).for_bucket(data, graph())


def test_successful_physical_detach_admitted_and_order_proven(tmp_path):
    result = batch(tmp_path, simple(tmp_path))
    assert len(result.events) == 1
    x = result.events[0]
    assert x.endpoint_a == graph().connectors[0].ref and x.endpoint_b == graph().connectors[1].ref
    assert x.completed_order == 2 and result.boundary.before_order == 3
    assert result.audit['admitted'][0]['dispatch_record'] == 1
    assert result.audit['admitted'][0]['completion_line'] == 2
    assert result.audit['admitted'][0]['timestamp'] is None


@pytest.mark.parametrize('completion', [accepted('old'),
    '[primitive] FAILED old: UNDOCKED detached alpha:socket-X from beta:socket-X',
    '[primitive] SUCCEEDED old: command dispatched'])
def test_accepted_failed_and_nonphysical_completion_do_not_confirm_detach(tmp_path, completion):
    # Duplicate ACCEPTED is ambiguous, so use a non-duplicate admission-only log.
    if completion == accepted('old'):
        data = refs(tmp_path, [dispatch('old'), dispatch('later','align_faces')], [2],
                    [accepted('old'), accepted('later')])
    else: data = simple(tmp_path, completion)
    assert batch(tmp_path, data).events == ()


def test_command_only_without_physical_success_does_not_confirm_detach(tmp_path):
    data = refs(tmp_path, [dispatch('old'), dispatch('later','align_faces')], [2], [accepted('later')])
    assert batch(tmp_path, data).events == ()


@pytest.mark.parametrize('fault', ['other_run', 'missing_boundary', 'ambiguous_boundary', 'missing_source_record', 'unordered_bucket'])
def test_other_run_and_missing_ambiguous_boundaries_fail(tmp_path, fault):
    data = simple(tmp_path)
    if fault == 'other_run': data['runtime_log']['run_id'] = 'another-run'
    elif fault == 'missing_boundary':
        data = refs(tmp_path,[dispatch('old'),dispatch('later','align_faces')],[2],[accepted('old'),detached()])
    elif fault == 'ambiguous_boundary':
        data = refs(tmp_path,[dispatch('old'),dispatch('later','align_faces')],[2],
                    [accepted('old'),detached(),accepted('later'),accepted('later')])
    elif fault == 'missing_source_record': data['source_records'][0]['record'] = 99
    else: data['source_records'] *= 2
    with pytest.raises(ValueError): batch(tmp_path, data)


def test_future_completion_not_admitted(tmp_path):
    data = refs(tmp_path,[dispatch('old'),dispatch('later','align_faces')],[2],
                [accepted('old'),accepted('later'),detached()])
    assert batch(tmp_path,data).events == ()


def test_redock_invalidates_old_disconnect_without_adding_edge(tmp_path):
    data = refs(tmp_path,[dispatch('old'),dispatch('reattach','dock'),dispatch('later','align_faces')],[3],
                [accepted('old'),detached(),accepted('reattach'),
                 '[primitive] SUCCEEDED reattach: DOCKED attached alpha:socket-X to gamma:socket-X; contact=ok',
                 accepted('later')])
    result = batch(tmp_path,data)
    assert result.events == () and result.audit['invalidated'][0]['reason'] == 'attachment'
    from drl.IL.bc_v1.causal_topology import reconcile_topology
    assert reconcile_topology(graph(),result.events,result.boundary).graph == graph()


def test_reset_invalidates_previous_segment(tmp_path):
    data = refs(tmp_path,[dispatch('old'),dispatch('later','align_faces')],[2],
                [accepted('old'),detached(),'[primitive] RESET: new physical segment',accepted('later')])
    result = batch(tmp_path,data)
    assert result.events == () and result.audit['invalidated'][0]['reason'] == 'reset'


def test_success_requires_accepted_dispatch_before_completion(tmp_path):
    data = refs(tmp_path,[dispatch('old'),dispatch('later','align_faces')],[2],
                [detached(),accepted('later')])
    with pytest.raises(ValueError): batch(tmp_path,data)


def test_target_stage_masks_and_future_independence(tmp_path):
    rows = [dispatch('old'),dispatch('later','align_faces')]
    lines = [accepted('old'),detached(),accepted('later')]
    data = refs(tmp_path,rows,[2],lines); before = batch(tmp_path,data)
    for row in rows:
        row.update(action_t={'arbitrary':'opposite'},supervision={'mask':False},
                   stage='impossible',fsm_state='whatever',control_state={},
                   graph_t_plus_1={'invented':'future'},target_graph={'wrong':'goal'})
    rows.append(dispatch('future','dock',graph_t={'forbidden':'future'}))
    data = refs(tmp_path,rows,[2],lines+['[primitive] SUCCEEDED future: DOCKED attached garbage'])
    after = batch(tmp_path,data)
    assert before.events == after.events and before.boundary == after.boundary


def test_current_bucket_disconnect_deferred_previous_bucket_admitted(tmp_path):
    data = refs(tmp_path,[dispatch('old'),dispatch('inside'),dispatch('later','align_faces')],[2,3],
                [accepted('old'),accepted('inside'),detached(),
                 detached('inside','gamma:socket-X','delta:socket-X'),accepted('later')])
    result = batch(tmp_path,data)
    assert len(result.events)==1 and result.events[0].endpoint_a.module.module_id=='alpha'
    assert result.audit['deferred'][0]['goal_id']=='inside'
    assert result.audit['deferred'][0]['reason']=='current_bucket_dispatch'


def real_minimal(root):
    frozen = json.loads(FIXTURE.read_text())
    rows = [{} for _ in range(max(x['record'] for x in frozen['source_rows']))]
    for x in frozen['source_rows']: rows[x['record']-1] = x['value']
    lines = ['' for _ in range(max(x['line'] for x in frozen['log_lines']))]
    for x in frozen['log_lines']: lines[x['line']-1] = x['text']
    return frozen,refs(root,rows,frozen['bucket_records'],lines)


def has_pair(table, operation, a, b):
    return any(x.action.action_id==operation and x.endpoints==frozenset((a,b))
               for x in table.structural_candidates)


def test_mixed_bucket_boundary_1189_matches_approved_policy(tmp_path):
    m=api(); frozen,data=real_minimal(tmp_path)
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.bc_v1.causal_topology import reconcile_topology
    from drl.IL.bc_v1.candidates import enumerate_candidates
    from drl.IL.bc_v1.feature_schema import load_feature_schema
    spec=c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
    g=SMORESAdapter(spec).encode_state(frozen['published_graph'])
    result=m.HistoricalSmoresDisconnectProvider(tmp_path,policy()).for_bucket(data,g)
    assert len(result.events)==1 and result.events[0].endpoint_a.module.module_id=='smores_02'
    assert result.audit['admitted'][0]['dispatch_record']==227
    assert result.audit['deferred'][0]['dispatch_record']==228
    assert result.audit['deferred'][0]['goal_id'].endswith('undock-1')
    derived=reconcile_topology(g,result.events,result.boundary).graph
    assert len(g.connections)==7 and len(derived.connections)==6
    table=enumerate_candidates(derived,spec,load_feature_schema(ROOT/'configs/bc_v1/features.json'))
    ref=lambda module,port:c.ConnectorRef(c.ModuleRef(module),port)
    assert has_pair(table,'undock',ref('smores_05','BOTTOM'),ref('smores_06','LEFT'))
    assert has_pair(table,'align_faces',ref('smores_02','BOTTOM'),ref('smores_01','TOP'))
    assert derived.modules==g.modules and derived.resources==g.resources


def test_generic_synthetic_mixed_bucket_different_ids_and_connectors(tmp_path):
    m=api()
    from drl.IL.tests.bc_v1.test_candidates import graph as generic_graph, catalog, schema
    from drl.IL.bc_v1.causal_topology import reconcile_topology
    from drl.IL.bc_v1.candidates import enumerate_candidates
    g=generic_graph(5,2)
    edges=((g.connectors[0].ref,g.connectors[2].ref),(g.connectors[4].ref,g.connectors[6].ref))
    peers={a:b for a,b in edges};peers.update({b:a for a,b in edges})
    g=replace(g,connectors=tuple(replace(x,connected_to=peers[x.ref],status='connected')
              if x.ref in peers else x for x in g.connectors),
              connections=tuple(c.ConnectionState(a,b,'current_connection') for a,b in edges))
    data=refs(tmp_path,[dispatch('prior-release'),dispatch('local-release'),dispatch('new-intent','align_faces')],[2,3],
              [accepted('prior-release'),accepted('local-release'),
               detached('prior-release','unit-0:socket-0','unit-1:socket-0'),
               detached('local-release','unit-2:socket-0','unit-3:socket-0'),accepted('new-intent')])
    evidence=m.HistoricalSmoresDisconnectProvider(tmp_path,policy()).for_bucket(data,g)
    derived=reconcile_topology(g,evidence.events,evidence.boundary).graph
    assert len(g.connections)==2 and len(derived.connections)==1
    table=enumerate_candidates(derived,catalog(2),schema())
    assert has_pair(table,'undock',*edges[1])
    assert has_pair(table,'align_faces',g.connectors[0].ref,g.connectors[8].ref)
    assert evidence.audit['deferred'][0]['goal_id']=='local-release'


def test_evidence_source_hash_drift_rejected(tmp_path):
    data=simple(tmp_path);(tmp_path/'dispatch.jsonl').write_text('{}\n')
    with pytest.raises(ValueError): batch(tmp_path,data)


def test_provider_fingerprint_sensitivity(tmp_path):
    m=api(); data=simple(tmp_path); result=batch(tmp_path,data)
    hashes=result.audit['source_hashes']
    before=m.provider_fingerprint(policy(),hashes)
    assert before==m.provider_fingerprint(dict(reversed(list(policy().items()))),dict(reversed(list(hashes.items()))))
    assert before!=m.provider_fingerprint(policy(),dict(hashes,extra='a'*64))
    assert before!=m.provider_fingerprint(dict(policy(),mixed_bucket_policy='changed'),hashes)
    assert before!=m.provider_fingerprint(policy(),hashes,parser_version='changed')


def test_unapproved_boundary_policy_rejected(tmp_path):
    m=api()
    with pytest.raises(ValueError):
        m.HistoricalSmoresDisconnectProvider(tmp_path,dict(policy(),mixed_bucket_policy='latest_prefix'))


def test_repeated_terminal_detach_cannot_revive_after_reattach(tmp_path):
    data=refs(tmp_path,[dispatch('old'),dispatch('later','align_faces')],[2],
              [accepted('old'),detached(),
               '[primitive] SUCCEEDED redock: DOCKED attached alpha:socket-X to beta:socket-X',
               detached(),accepted('later')])
    with pytest.raises(ValueError,match='[Aa]mbiguous|[Rr]epeated'):
        batch(tmp_path,data)


@pytest.mark.parametrize('hole', [False,True])
def test_source_position_cannot_be_reclassified_previous_by_log_order(tmp_path,hole):
    rows=[dispatch('first','align_faces'),dispatch('future'),dispatch('last','align_faces')]
    data=refs(tmp_path,rows,[1,3] if hole else [1],
              [accepted('future'),detached('future'),accepted('first'),accepted('last')])
    if not hole: data['source_files'][0]['prefix_end_record']=2
    with pytest.raises(ValueError): batch(tmp_path,data)


def test_provider_batch_fingerprint_changes_with_membership_and_cutoff(tmp_path):
    data=simple(tmp_path)
    first=batch(tmp_path,data)
    data['source_records']=[{'path':'dispatch.jsonl','record':1},{'path':'dispatch.jsonl','record':2}]
    second=batch(tmp_path,data)
    assert len(first.events)==1 and second.events==()
    assert first.audit['provider_fingerprint']!=second.audit['provider_fingerprint']


REVIEWED_KEYS = tuple((eid, index) for eid, indices in {
    'teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8': (7842,),
    'teleop-1bea8f28c1c04323a4c3ec7e2829d07c': (1239, 3674, 7290, 8666),
    'teleop-39e2f4b4cd684e9ea72734cc21baca94': (1189, 2525),
    'teleop-3aa4f69128774d5eba1f8d851c02cf53': (2248, 4447),
    'teleop-5bb5a4712d924617aab5604b6deb20b4': (5004,),
    'teleop-69b37b8db7e141339518d4eef585e4bd': (3137, 7162),
    'teleop-a7d7e08f5fd14a59bd3af6afb53ccde1': (1161, 6863),
    'teleop-ce8293656bb444ac999f87c0520672fc': (11179,),
    'teleop-f8888f12c46249fc80ef0af8c56b4dcf': (1908, 2480, 5435),
    'teleop-fb69637be28d49ccbff572e721cfae24': (12755, 14417),
}.items() for index in indices)
SELECTED_KEY = ('teleop-39e2f4b4cd684e9ea72734cc21baca94', 1189)


def catalog_for(data, identity, *, complete=False):
    cases = {}
    for eid, index in REVIEWED_KEYS if complete else (SELECTED_KEY,):
        cases[f'{eid}::{index}'] = {
            'episode_id': eid, 'transition_index': index, 'run_id': eid,
            'runtime_log': dict(data['runtime_log'], run_id=eid),
            'source_files': [dict(x, run_id=eid) for x in data['source_files']],
            'source_records': copy.deepcopy(data['source_records'])}
    return {'schema_version': 'mssr.bc_v1.causal_repairs.v1',
            'source_identity': identity, 'cases': cases}


def routing(identity, records, *, key=SELECTED_KEY):
    return {'dataset_identity': identity, 'logical_episode_id': key[0],
            'canonical_transition_index': key[1], 'source_records': records}


def selected_fixture(tmp_path):
    identity = {'manifest': 'canonical/manifest.json', 'manifest_sha256': 'a'*64}
    data = simple(tmp_path)
    catalog = catalog_for(data, identity)
    provenance = routing(identity, [{'_loader_source_path': str(tmp_path/'dispatch.jsonl'),
                                     '_loader_source_row': 2}])
    return catalog, provenance


def test_nonselected_canonical_key_returns_none_without_source_io(tmp_path, monkeypatch):
    catalog, provenance = selected_fixture(tmp_path)
    provenance.update(logical_episode_id='healthy', canonical_transition_index=0)
    class UnreadableRecords:
        def __iter__(self):
            pytest.fail('Nonselected routing inspected source_records')
    provenance['source_records'] = UnreadableRecords()
    def forbidden(*args, **kwargs):
        pytest.fail('Nonselected routing opened causal evidence')
    monkeypatch.setattr(Path, 'open', forbidden)
    assert api().read_disconnect_source_refs(provenance, repo_root=tmp_path, source_catalog=catalog) is None


@pytest.mark.parametrize('primitive', ['align_faces', 'undock'])
def test_nonselected_align_or_undock_cannot_trigger_repair(tmp_path, monkeypatch, primitive):
    data = refs(tmp_path, [dispatch('temptation', primitive)], [1], [accepted('temptation')])
    identity = {'manifest': 'canonical/manifest.json', 'manifest_sha256': 'a'*64}
    catalog = catalog_for(data, identity)
    provenance = routing(identity, [{'_loader_source_path': str(tmp_path/'dispatch.jsonl'),
                                    '_loader_source_row': 1}], key=('healthy', 0))
    provenance.update(action_t={'operation': primitive}, supervision={'valid': True},
                      target_morphology='poison', graph_t_plus_1='poison')
    def forbidden(*args, **kwargs):
        pytest.fail('Nonselected primitive caused causal source access')
    monkeypatch.setattr(Path, 'open', forbidden)
    assert api().read_disconnect_source_refs(provenance, repo_root=tmp_path, source_catalog=catalog) is None


@pytest.mark.parametrize('fault', ['path', 'record', 'order', 'extra'])
def test_selected_key_requires_exact_reviewed_source_membership(tmp_path, fault):
    catalog, provenance = selected_fixture(tmp_path)
    if fault == 'path': provenance['source_records'][0]['_loader_source_path'] = str(tmp_path/'other.jsonl')
    elif fault == 'record': provenance['source_records'][0]['_loader_source_row'] = 1
    elif fault == 'order':
        item = catalog['cases'][f'{SELECTED_KEY[0]}::{SELECTED_KEY[1]}']
        item['source_records'] = [{'path': 'dispatch.jsonl', 'record': 1}, {'path': 'dispatch.jsonl', 'record': 2}]
        provenance['source_records'].append({'_loader_source_path': str(tmp_path/'dispatch.jsonl'), '_loader_source_row': 1})
    else: provenance['source_records'] *= 2
    with pytest.raises(ValueError, match='membership|pointer|reviewed|source records'):
        api().read_disconnect_source_refs(provenance, repo_root=tmp_path, source_catalog=catalog)


def test_selected_key_uses_reviewed_physical_evidence(tmp_path):
    catalog, provenance = selected_fixture(tmp_path)
    joined = api().read_disconnect_source_refs(provenance, repo_root=tmp_path, source_catalog=catalog)
    assert joined['source_records'] == [{'path': 'dispatch.jsonl', 'record': 2}]
    result = batch(tmp_path, joined)
    assert len(result.events) == 1
    assert result.events[0].endpoint_a == graph().connectors[0].ref
    assert result.audit['admitted'][0]['completion_line'] == 2


def test_selected_mixed_bucket_preserves_all_reviewed_records(tmp_path):
    data = refs(tmp_path, [dispatch('old'), dispatch('later', 'align_faces'),
                          {'expert_action': {'primitive_goal': None}}], [2, 3],
                [accepted('old'), detached(), accepted('later')])
    identity = {'manifest': 'canonical/manifest.json', 'manifest_sha256': 'a'*64}
    catalog = catalog_for(data, identity)
    provenance = routing(identity, [{'_loader_source_path': str(tmp_path/'dispatch.jsonl'),
                                    '_loader_source_row': n} for n in (2, 3)])
    joined = api().read_disconnect_source_refs(provenance, repo_root=tmp_path, source_catalog=catalog)
    assert joined['source_records'] == data['source_records']
    result = batch(tmp_path, joined)
    assert result.audit['source_records'] == (('dispatch.jsonl', 2), ('dispatch.jsonl', 3))
    assert len(result.events) == 1


@pytest.mark.parametrize('fault', ['foreign_dataset', 'missing_log', 'wrong_log_sha', 'wrong_source_sha',
    'row_bounds', 'run_membership', 'descriptor_run', 'case_episode', 'case_index', 'escape'])
def test_selected_sparse_evidence_fails_closed(tmp_path, fault):
    catalog, provenance = selected_fixture(tmp_path)
    item = catalog['cases'][f'{SELECTED_KEY[0]}::{SELECTED_KEY[1]}']
    if fault == 'foreign_dataset': provenance['dataset_identity'] = {'manifest_sha256': 'f'*64}
    elif fault == 'missing_log': item['runtime_log']['path'] = 'missing.log'
    elif fault == 'wrong_log_sha': item['runtime_log']['sha256'] = 'f'*64
    elif fault == 'wrong_source_sha': item['source_files'][0]['sha256'] = 'f'*64
    elif fault == 'row_bounds':
        item['source_records'][0]['record'] = 99
        provenance['source_records'][0]['_loader_source_row'] = 99
    elif fault == 'run_membership':
        item['run_id'] = 'another-run'
        item['runtime_log']['run_id'] = 'another-run'
        item['source_files'][0]['run_id'] = 'another-run'
    elif fault == 'descriptor_run': item['source_files'][0]['run_id'] = 'another-run'
    elif fault == 'case_episode': item['episode_id'] = 'other-episode'
    elif fault == 'case_index': item['transition_index'] += 1
    else: item['source_files'][0]['path'] = '../outside.jsonl'
    with pytest.raises(ValueError):
        api().read_disconnect_source_refs(provenance, repo_root=tmp_path, source_catalog=catalog)


@pytest.mark.parametrize('field,value', [('logical_episode_id', ''), ('canonical_transition_index', -1),
    ('canonical_transition_index', True), ('canonical_transition_index', None)])
def test_sparse_routing_requires_valid_canonical_identity(tmp_path, field, value):
    catalog, provenance = selected_fixture(tmp_path)
    provenance[field] = value
    with pytest.raises(ValueError):
        api().read_disconnect_source_refs(provenance, repo_root=tmp_path, source_catalog=catalog)


def test_selected_repair_ignores_targets_and_future(tmp_path):
    catalog, provenance = selected_fixture(tmp_path)
    reader = api().read_disconnect_source_refs
    before = batch(tmp_path, reader(provenance, repo_root=tmp_path, source_catalog=catalog))
    provenance.update(action_t='opposite', supervision=False, stage='poison', FSM='poison',
        control_state='poison', target_morphology='opposite', graph_t_plus_1='invented', outcome='future')
    after = batch(tmp_path, reader(provenance, repo_root=tmp_path, source_catalog=catalog))
    assert before == after


def test_repeated_provider_verification_uses_stat_guarded_offsets_without_future_payloads(tmp_path, monkeypatch):
    m = api(); data = simple(tmp_path)
    path = tmp_path/'dispatch.jsonl'
    with path.open('a') as stream: stream.write('unparseable-future-payload\n')
    data['source_files'][0]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    provider = m.HistoricalSmoresDisconnectProvider(tmp_path, policy())
    first = provider.for_bucket(data, graph())
    verified = dict(getattr(provider, '_verified_sources', {}))
    assert verified, 'Provider must retain verified hash/offset identities, not future payloads'
    original = Path.open; reads = []
    def opened(p, *args, **kwargs):
        if p == path: reads.append(p)
        return original(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opened)
    second = provider.for_bucket(data, graph())
    assert second.events == first.events
    assert dict(provider._verified_sources) == verified
    assert len(reads) == 1  # Read only the bounded prefix; do not hash the full file again.
    path.write_bytes(path.read_bytes()+b'changed\n')
    with pytest.raises(ValueError, match='integrity|changed'):
        provider.for_bucket(data, graph())
