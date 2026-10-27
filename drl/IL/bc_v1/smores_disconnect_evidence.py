"""Historical SMORES physical disconnect proofs, without label reconstruction.

for_bucket consumes explicit run-pinned files and ordered source-record pointers.
It does not discover runs, read Canonical actions, or integrate dataset prepare.
The last recorded dispatch in the bucket anchors the admitted prefix. Physical
detach completions of current-bucket dispatches are deferred from its shared view.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from .bc_v1_contract import ConnectorRef, ModuleRef, StateGraph
from .causal_topology import (ALGORITHM, MIXED_BUCKET_POLICY, ORDERING_CONTRACT,
                             VERSION as CORE_VERSION, CausalBoundary, ConfirmedDisconnect)


VERSION = 'mssr.bc_v1.smores_disconnect_evidence.v1'
PARSER_VERSION = 'smores_physical_completion_log.v1'
_EVENT = re.compile(r'\[primitive\] (ACCEPTED|SUCCEEDED|FAILED|REJECTED|CANCELED|CANCELLED) ([^\s:]+): (.*)')
_DETACH = re.compile(r'^UNDOCKED detached ([^\s:]+):([^\s:;]+) from ([^\s:]+):([^\s:;]+)(?:;.*)?$')
_ATTACH = re.compile(r'^DOCKED attached ([^\s:]+):([^\s:;]+) to ([^\s:]+):([^\s:;]+)(?:;.*)?$')
_RESET = re.compile(r'\[primitive\] RESET(?:\s|:)')


def _hash(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def provider_fingerprint(policy: Mapping, source_hashes: Mapping, *,
                         parser_version=PARSER_VERSION, boundary_identity: Mapping | None = None) -> str:
    return _hash({'schema_version': VERSION, 'parser_version': parser_version,
                  'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'boundary_policy': dict(policy), 'source_hashes': dict(source_hashes),
                  'boundary_identity': None if boundary_identity is None else dict(boundary_identity)})


@dataclass(frozen=True)
class HistoricalDecisionBoundary:
    source_records: tuple[tuple[str, int], ...]
    dispatch_ids: tuple[str, ...]


@dataclass(frozen=True)
class DisconnectEvidenceBatch:
    events: tuple[ConfirmedDisconnect, ...]
    boundary: CausalBoundary
    audit: Mapping


def _position(value):
    if type(value) is not int or value < 1:
        raise ValueError('Source record positions must be positive integers')
    return value


def _name(value):
    if type(value) is not str or not value or any(x.isspace() for x in value):
        raise ValueError('Missing or malformed dispatch/run identity')
    return value


def _endpoints(match):
    a, pa, b, pb = match.groups()
    return ConnectorRef(ModuleRef(a), pa), ConnectorRef(ModuleRef(b), pb)


REPAIRS_VERSION = 'mssr.bc_v1.causal_repairs.v1'
APPROVED_REPAIR_KEYS = frozenset((eid, index) for eid, indices in {
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


def _canonical_key(episode, index):
    episode = _name(episode)
    if type(index) is not int or index < 0:
        raise ValueError('Missing or malformed canonical transition index')
    return episode, index


def _relative_path(value):
    if type(value) is not str or not value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Reviewed evidence requires repository-relative source paths')
    if str(Path(value)) != value or value == '.':
        raise ValueError('Noncanonical reviewed evidence path')
    return value


def _reviewed_case(declaration, key):
    if not isinstance(declaration, Mapping) or set(declaration) != {
            'episode_id', 'transition_index', 'run_id', 'runtime_log', 'source_files', 'source_records'}:
        raise ValueError('Missing or unsupported reviewed case fields')
    if _canonical_key(declaration['episode_id'], declaration['transition_index']) != key:
        raise ValueError('Reviewed case episode/index mismatch')
    # All twenty certificates use their recorded teleop episode as run identity.
    run_id = _name(declaration['run_id'])
    if run_id != key[0]:
        raise ValueError('Reviewed run membership differs from recorded episode')
    def descriptor(item):
        if not isinstance(item, Mapping) or set(item) != {'path', 'sha256', 'run_id'}:
            raise ValueError('Missing or unsupported reviewed evidence descriptor')
        _relative_path(item['path'])
        if item['run_id'] != run_id:
            raise ValueError('Reviewed evidence belongs to a different run')
        if type(item['sha256']) is not str or re.fullmatch(r'[0-9a-f]{64}', item['sha256']) is None:
            raise ValueError('Malformed reviewed source hash')
    descriptor(declaration['runtime_log'])
    files = declaration['source_files']
    records = declaration['source_records']
    if not isinstance(files, list) or not files or not isinstance(records, list) or not records:
        raise ValueError('Missing reviewed source files/records')
    for item in files: descriptor(item)
    file_paths = [x['path'] for x in files]
    if len(file_paths) != len(set(file_paths)):
        raise ValueError('Duplicate reviewed source file')
    pointers = []
    for item in records:
        if not isinstance(item, Mapping) or set(item) != {'path', 'record'}:
            raise ValueError('Malformed reviewed source records')
        pointers.append((_relative_path(item['path']), _position(item['record'])))
    if len(pointers) != len(set(pointers)) or {p for p, _ in pointers} != set(file_paths):
        raise ValueError('Reviewed source membership is missing or duplicated')
    return pointers


def read_causal_repairs_catalog(path: Path, expected_source_identity: Mapping) -> Mapping:
    """Load only the complete frozen twenty-key oracle; never inspect source data."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate causal repair JSON key')
            result[key] = value
        return result
    catalog = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    if (not isinstance(catalog, Mapping) or set(catalog) != {'schema_version', 'source_identity', 'cases'}
            or catalog.get('schema_version') != REPAIRS_VERSION
            or catalog.get('source_identity') != expected_source_identity):
        raise ValueError('Certified causal repair schema/source identity mismatch')
    cases = catalog['cases']
    expected = {f'{e}::{i}' for e, i in APPROVED_REPAIR_KEYS}
    if not isinstance(cases, Mapping) or set(cases) != expected:
        raise ValueError('Causal repair cases must equal the exact 20 approved keys')
    for key in APPROVED_REPAIR_KEYS:
        _reviewed_case(cases[f'{key[0]}::{key[1]}'], key)
    return catalog


def read_disconnect_source_refs(provenance: Mapping, *, repo_root: Path,
                                source_catalog: Mapping) -> Mapping | None:
    """Route exclusively by Canonical key; nonselected keys do no causal source I/O.

    The reviewed oracle selects evidence, never targets or primitive contents.
    The historical provider remains the authority for physical completions.
    """
    if (source_catalog.get('schema_version') != REPAIRS_VERSION
            or source_catalog.get('source_identity') != provenance.get('dataset_identity')):
        raise ValueError('Causal repair catalog identity mismatch')
    key = _canonical_key(provenance.get('logical_episode_id'), provenance.get('canonical_transition_index'))
    cases = source_catalog.get('cases')
    if not isinstance(cases, Mapping): raise ValueError('Missing reviewed causal cases')
    case_id = f'{key[0]}::{key[1]}'
    if case_id not in cases:
        return None
    if key not in APPROVED_REPAIR_KEYS:
        raise ValueError('Unapproved causal repair key')
    declaration = cases[case_id]
    reviewed = _reviewed_case(declaration, key)
    root = Path(repo_root).resolve()
    def inside(value):
        path = (root/value).resolve()
        if not path.is_relative_to(root): raise ValueError('Reviewed source path escapes repository')
        return path
    current = []
    try:
        for ref in provenance.get('source_records', ()):
            path = inside(ref['_loader_source_path'])
            current.append((str(path.relative_to(root)), _position(ref['_loader_source_row'])))
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing exact reviewed source pointers') from exc
    if current != reviewed:
        raise ValueError('Current source pointer membership differs from reviewed source records')
    from .smores_adapter import _source_index
    indexed_files = {}
    for descriptor in [*declaration['source_files'], declaration['runtime_log']]:
        path = inside(descriptor['path'])
        if not path.is_file(): raise ValueError('Missing reviewed runtime/source evidence file')
        indexed = _source_index(path)
        if indexed.sha256 != descriptor['sha256']:
            raise ValueError('Reviewed source/runtime hash/integrity mismatch')
        indexed_files[descriptor['path']] = indexed
    for path, index in reviewed:
        if index > len(indexed_files[path].offsets):
            raise ValueError('Reviewed current source record out of bounds')
    return {name: declaration[name] for name in ('run_id', 'runtime_log', 'source_files', 'source_records')}


class HistoricalSmoresDisconnectProvider:
    def __init__(self, repo_root: Path, policy: Mapping):
        self.repo_root = Path(repo_root).resolve()
        self.policy = dict(policy)
        self._verified_sources = {}
        required = {'schema_version': CORE_VERSION, 'algorithm': ALGORITHM,
                    'ordering_contract': ORDERING_CONTRACT, 'mixed_bucket_policy': MIXED_BUCKET_POLICY,
                    'operations': ['confirmed_disconnect'], 'current_bucket_disconnects': 'defer',
                    'reattach': 'invalidate_disconnect_only', 'reset': 'invalidate_prior_segment'}
        if self.policy != required:
            raise ValueError('Unsupported or unapproved causal topology boundary policy')

    def _path(self, value):
        path = (self.repo_root / value).resolve()
        if not path.is_relative_to(self.repo_root):
            raise ValueError('Evidence path escapes declared repository/run root')
        return path

    def _verified_bytes(self, descriptor, run_id, hashes, *, prefix_lines=None):
        if descriptor.get('run_id') != run_id:
            raise ValueError('Evidence belongs to a different run')
        path = self._path(descriptor['path'])
        signature = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        before = signature(path.stat())
        cached = self._verified_sources.get(path)
        if cached is None:
            digest_builder, offsets = hashlib.sha256(), [0]
            with path.open('rb') as stream:
                for line in stream:
                    digest_builder.update(line); offsets.append(offsets[-1] + len(line))
            cached = (before, digest_builder.hexdigest(), tuple(offsets))
            if before != signature(path.stat()):
                raise ValueError('Historical evidence changed during verification')
            self._verified_sources[path] = cached
        recorded_signature, digest, offsets = cached
        if before != recorded_signature or descriptor.get('sha256') != digest:
            raise ValueError('Historical evidence hash/integrity mismatch')
        stop = offsets[-1] if prefix_lines is None else offsets[min(prefix_lines, len(offsets)-1)]
        with path.open('rb') as stream:
            data = stream.read(stop)
        if before != signature(path.stat()):
            raise ValueError('Historical evidence changed during prefix read')
        relative = str(path.relative_to(self.repo_root))
        if relative in hashes and hashes[relative] != digest:
            raise ValueError('Contradictory evidence identity')
        hashes[relative] = digest
        return relative, data

    def for_bucket(self, source_refs: Mapping, published_graph: StateGraph) -> DisconnectEvidenceBatch:
        if type(published_graph) is not StateGraph:
            raise ValueError('Expected typed published current graph')
        run_id = _name(source_refs['run_id'])
        bucket = tuple((str(self._path(x['path']).relative_to(self.repo_root)),
                        _position(x['record'])) for x in source_refs['source_records'])
        if not bucket or len(set(bucket)) != len(bucket):
            raise ValueError('Missing or duplicate bucket source boundary')
        by_path = {}
        for path, index in bucket:
            indices = by_path.setdefault(path, [])
            if indices and index <= indices[-1]:
                raise ValueError('Unordered bucket source records')
            indices.append(index)
        hashes, goals, record_goals, seen_files, prefixes = {}, {}, {}, set(), {}
        for descriptor in source_refs['source_files']:
            path = str(self._path(descriptor['path']).relative_to(self.repo_root))
            if path in seen_files:
                raise ValueError('Duplicate historical source file')
            seen_files.add(path)
            stop = descriptor.get('prefix_end_record', max(by_path.get(path, [0])))
            stop = _position(stop)
            if path in by_path and stop < max(by_path[path]):
                raise ValueError('Source prefix does not cover current bucket')
            if path in by_path and stop > max(by_path[path]):
                raise ValueError('Source prefix extends beyond current bucket')
            path, data = self._verified_bytes(descriptor, run_id, hashes, prefix_lines=stop)
            lines = data.splitlines()
            if stop > len(lines):
                raise ValueError('Source record boundary out of bounds')
            prefixes[path] = stop
            for index, line in enumerate(lines[:stop], 1):
                row = json.loads(line)
                goal = row.get('expert_action', {}).get('primitive_goal')
                if goal is None:
                    continue
                goal_id = _name(goal['goal_id'])
                if goal_id in goals:
                    raise ValueError('Ambiguous repeated source dispatch identity')
                if path in by_path and index >= min(by_path[path]) and (path, index) not in bucket:
                    raise ValueError('Dispatch inside current source interval lacks bucket membership')
                primitive = _name(goal['primitive'])
                goals[goal_id] = {'source_path': path, 'dispatch_record': index, 'primitive': primitive}
                record_goals[(path, index)] = goal_id
        if not set(by_path).issubset(seen_files):
            raise ValueError('Bucket source file is missing')
        current = tuple(record_goals[key] for key in bucket if key in record_goals)
        if not current:
            raise ValueError('Missing current dispatch decision boundary')
        historical = HistoricalDecisionBoundary(bucket, current)
        log_path, log_bytes = self._verified_bytes(source_refs['runtime_log'], run_id, hashes)
        log_lines = log_bytes.decode('utf-8').splitlines()
        accepted = {}
        # Scan admission identities to certify uniqueness; physical payloads after
        # the selected boundary are never interpreted or admitted.
        for line_number, line in enumerate(log_lines, 1):
            match = _EVENT.search(line)
            if match and match[1] == 'ACCEPTED' and match[2] in goals:
                accepted.setdefault(match[2], []).append(line_number)
        current_orders = []
        for goal_id in historical.dispatch_ids:
            if len(accepted.get(goal_id, ())) != 1:
                raise ValueError('Missing or ambiguous runtime decision boundary')
            current_orders.append(accepted[goal_id][0])
        if current_orders != sorted(set(current_orders)):
            raise ValueError('Source dispatch order contradicts runtime event order')
        cutoff = current_orders[-1]
        domain = _hash({'run_id': run_id, 'event_stream': log_path})
        boundary = CausalBoundary(domain, cutoff)
        pending, deferred, invalidated, unmatched = {}, [], [], []
        completed_goals = set()
        reset_order = 0
        for line_number, line in enumerate(log_lines[:cutoff-1], 1):
            if _RESET.search(line):
                reset_order = line_number
                invalidated.extend(dict(proof, reason='reset') for _, proof in pending.values())
                pending.clear()
                continue
            match = _EVENT.search(line)
            if not match or match[1] != 'SUCCEEDED':
                continue
            goal_id, detail = match[2], match[3]
            attachment = _ATTACH.match(detail)
            if attachment:
                touched = set(_endpoints(attachment))
                for old_id, (event, proof) in tuple(pending.items()):
                    if touched.intersection((event.endpoint_a, event.endpoint_b)):
                        invalidated.append(dict(proof, reason='attachment', invalidating_line=line_number))
                        del pending[old_id]
                continue
            detached = _DETACH.match(detail)
            if detail.startswith(('UNDOCKED', 'DOCKED')) and not detached:
                raise ValueError('Malformed physical completion evidence')
            if not detached:
                continue
            if goal_id not in goals:
                unmatched.append({'goal_id': goal_id, 'completion_line': line_number})
                continue
            if goal_id in completed_goals:
                raise ValueError('Ambiguous repeated physical completion')
            completed_goals.add(goal_id)
            goal = goals[goal_id]
            admissions = accepted.get(goal_id, ())
            if goal['primitive'] != 'undock' or len(admissions) != 1 or admissions[0] >= line_number:
                raise ValueError('Physical detach lacks an unambiguous preceding UNDOCK dispatch')
            a, b = _endpoints(detached)
            proof = dict(goal, goal_id=goal_id, accepted_line=admissions[0],
                         completion_line=line_number, timestamp=None,
                         endpoints=((a.module.module_id, a.connector_id), (b.module.module_id, b.connector_id)))
            if admissions[0] < reset_order:
                invalidated.append(dict(proof, reason='reset'))
                continue
            if goal_id in historical.dispatch_ids:
                deferred.append(dict(proof, reason='current_bucket_dispatch'))
                continue
            if admissions[0] >= current_orders[0]:
                raise ValueError('Previous-bucket source dispatch is not causally before current bucket')
            event = ConfirmedDisconnect(a, b, _hash({'domain': domain, 'proof': proof}), domain, line_number)
            pending[goal_id] = (event, proof)
        ordered = sorted(pending.values(), key=lambda x: x[0].completed_order)
        boundary_identity = {'run_id': run_id, 'order_domain': domain,
                             'source_prefixes': prefixes, 'source_records': bucket,
                             'dispatch_ids': current, 'before_order': cutoff}
        return DisconnectEvidenceBatch(tuple(x[0] for x in ordered), boundary,
            {'schema_version': VERSION, 'parser_version': PARSER_VERSION,
             'boundary_policy': MIXED_BUCKET_POLICY, 'boundary_dispatch_id': current[-1],
             'boundary_line': cutoff, 'source_records': bucket, 'source_hashes': hashes,
             'admitted': [x[1] for x in ordered], 'deferred': deferred,
             'invalidated': invalidated, 'unmatched_detaches': unmatched,
             'boundary_identity': boundary_identity,
             'provider_fingerprint': provider_fingerprint(self.policy, hashes,
                                                         boundary_identity=boundary_identity)})
