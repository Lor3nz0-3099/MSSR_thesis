"""Versioned episode/lineage holdout; stdlib only, offline metadata only.

Callers supply certified episode-level identities and content hashes. The core
never reads transitions, robot state, targets, policy inputs or training stats.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re

PARTS = ('train', 'validation', 'test')
DEFAULT_SEED = 20261006
ALGORITHM = 'stratified_grouped_holdout.v1'
MANIFEST_VERSION = 'mssr.bc_v1.episode_split.v1'


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('Expected finite JSON metadata') from error


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _text(value, name):
    if type(value) is not str or not value:
        raise ValueError(f'Missing/invalid {name}')
    return value


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'Invalid nonnegative integer {name}')
    return value


def _hash(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid SHA-256 content hash')
    return value


def _config(config):
    if not isinstance(config, Mapping) or set(config) != {'schema_version', 'eligible_count_field', 'strata'}:
        raise ValueError('Invalid strata config fields')
    if config['schema_version'] != 'mssr.bc_v1.strata.v1':
        raise ValueError('Unsupported strata config version')
    _text(config['eligible_count_field'], 'eligible_count_field')
    strata = config['strata']
    if not isinstance(strata, (list, tuple)) or not strata:
        raise ValueError('Expected nonempty strata')
    names = set()
    for s in strata:
        if not isinstance(s, Mapping) or set(s) != {'name', 'match', 'total', 'quotas'}:
            raise ValueError('Invalid stratum fields')
        name = _text(s['name'], 'stratum name')
        if name in names:
            raise ValueError('Duplicate stratum name')
        names.add(name)
        if not isinstance(s['match'], Mapping) or not s['match']:
            raise ValueError('Expected explicit stratum matching metadata')
        for key, value in s['match'].items():
            _text(key, 'match field')
            if type(value) not in (str, int, bool) and value is not None:
                raise ValueError('Stratum matches require scalar metadata')
        if not isinstance(s['quotas'], Mapping) or set(s['quotas']) != set(PARTS):
            raise ValueError('Expected train/validation/test quotas')
        quotas = [_count(s['quotas'][p], 'quota') for p in PARTS]
        if _count(s['total'], 'stratum total') != sum(quotas):
            raise ValueError('Stratum total disagrees with quotas')
    # Detach source mappings and normalize catalog ordering before the RNG.
    value = json.loads(_json(config))
    value['strata'].sort(key=lambda s:s['name'])
    return value


def strata_fingerprint(strata_config: Mapping) -> str:
    return _fingerprint(_config(strata_config))


def _inventory(episodes, config):
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise ValueError('Expected episode inventory, not transitions')
    records, eligible, groups, claims, hashes = {}, {}, defaultdict(list), {}, {}
    count_field = config['eligible_count_field']

    def claim(key, lineage):
        if key in claims and claims[key] != lineage:
            raise ValueError('Inconsistent lineage/shared identity or hash collision')
        claims[key] = lineage

    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError('Expected episode mapping')
        eid = _text(episode.get('episode_id'), 'episode_id')
        if eid in records:
            raise ValueError('Duplicate episode_id')
        lineage = _text(episode.get('lineage_id'), 'lineage_id')
        content = _hash(episode.get('sha256'))
        bc = _count(episode.get(count_field), count_field)
        aliases = episode.get('lineage_aliases', ())
        source_hashes = episode.get('source_hashes', {})
        if not isinstance(aliases, (list, tuple)) or not isinstance(source_hashes, Mapping):
            raise ValueError('Invalid lineage aliases/source hashes')
        claim(('content', content), lineage)
        claim(('identity', lineage), lineage)
        for alias in aliases:
            claim(('identity', _text(alias, 'lineage alias')), lineage)
        for source, digest in source_hashes.items():
            source = _text(source, 'source identity'); digest = _hash(digest)
            if source in hashes and hashes[source] != digest:
                raise ValueError('Inconsistent source hash for the same representation')
            hashes[source] = digest
            claim(('identity', source), lineage)
            claim(('content', digest), lineage)
        records[eid] = episode
        if not bc:
            continue
        matches = [s['name'] for s in config['strata']
                   if all(k in episode and type(episode[k]) is type(v) and episode[k] == v
                          for k,v in s['match'].items())]
        if len(matches) != 1:
            raise ValueError(f'Episode {eid} has missing or ambiguous stratum')
        eligible[eid] = matches[0]
        groups[lineage].append(eid)
    for lineage, members in groups.items():
        if len({eligible[e] for e in members}) != 1:
            raise ValueError(f'Lineage group {lineage} spans incompatible strata')
    actual = Counter(eligible.values())
    if any(actual[s['name']] != s['total'] for s in config['strata']):
        raise ValueError('Eligible episode counts disagree with stratum quotas')
    return records, eligible, {g:tuple(sorted(ids)) for g,ids in groups.items()}


def _allocate(groups, quotas):
    """Find exact weighted quotas without splitting a group or retrying seeds."""
    remaining = [0]*(len(groups)+1)
    for i in range(len(groups)-1, -1, -1):
        remaining[i] = remaining[i+1]+len(groups[i][1])

    @lru_cache(None)
    def solve(i, train, validation):
        test = remaining[i]-train-validation
        if min(train, validation, test) < 0:
            return None
        if i == len(groups):
            return () if train == validation == 0 else None
        size = len(groups[i][1])
        for part, capacity in enumerate((train, validation, test)):
            if size <= capacity:
                rest = solve(i+1, train-(size if part==0 else 0), validation-(size if part==1 else 0))
                if rest is not None:
                    return (part,)+rest
        return None

    result = solve(0, quotas['train'], quotas['validation'])
    if result is None:
        raise ValueError('Lineage group sizes incompatible with exact quotas')
    return result


def make_episode_split(episodes: Sequence[Mapping], strata_config: Mapping,
                       seed: int = DEFAULT_SEED) -> Mapping[str, tuple[str, ...]]:
    if type(seed) is not int:
        raise ValueError('Seed must be an integer')
    config = _config(strata_config)
    _, eligible, groups = _inventory(episodes, config)
    rng = random.Random(seed)
    result = {p:[] for p in PARTS}
    for stratum in config['strata']:
        ordered = sorted((g, ids) for g, ids in groups.items() if eligible[ids[0]] == stratum['name'])
        rng.shuffle(ordered)
        assignments = _allocate(ordered, stratum['quotas'])
        for (_, ids), part in zip(ordered, assignments):
            result[PARTS[part]].extend(ids)
    return {p:tuple(sorted(result[p])) for p in PARTS}


def validate_split(split: Mapping, episodes: Sequence[Mapping], strata_config: Mapping) -> None:
    config = _config(strata_config)
    _, eligible, groups = _inventory(episodes, config)
    if not isinstance(split, Mapping) or set(split) != set(PARTS):
        raise ValueError('Expected exactly three split partitions')
    owners = {}
    for part in PARTS:
        if not isinstance(split[part], (list, tuple)):
            raise ValueError('Expected episode IDs, not row/timestep memberships')
        for eid in split[part]:
            if type(eid) is not str or eid not in eligible or eid in owners:
                raise ValueError('Unknown/excluded or repeated episode in split')
            owners[eid] = part
        counts = Counter(eligible[e] for e in split[part])
        if any(counts[s['name']] != s['quotas'][part] for s in config['strata']):
            raise ValueError('Split does not satisfy exact stratum quotas')
    if set(owners) != set(eligible):
        raise ValueError('Split union differs from all eligible episodes')
    if any(len({owners[e] for e in ids}) != 1 for ids in groups.values()):
        raise ValueError('Lineage group split across partitions')


def build_split_manifest(split, episodes, strata_config, seed, dataset_identity):
    validate_split(split, episodes, strata_config)
    if {p:tuple(sorted(split[p])) for p in PARTS} != make_episode_split(episodes, strata_config, seed):
        raise ValueError('Membership does not match declared seed/config/source')
    if not isinstance(dataset_identity, Mapping):
        raise ValueError('Missing dataset identity')
    _text(dataset_identity.get('dataset_name'), 'dataset_name')
    _text(dataset_identity.get('schema_version'), 'dataset schema')
    _hash(dataset_identity.get('manifest_sha256'))
    config = _config(strata_config)
    records, eligible, groups = _inventory(episodes, config)
    metadata = {}
    for eid, e in sorted(records.items()):
        keep = ('lineage_id','sha256',config['eligible_count_field'],'lineage_aliases','source_hashes',
                'kind','task','teleop_profile','course','file','bytes','canonical_transitions',
                'supervised_instances','metadata_sources')
        metadata[eid] = {k:e[k] for k in keep if k in e}
        # Exact matching metadata is retained for future validation/audit.
        for s in config['strata']:
            metadata[eid].update({k:e[k] for k in s['match'] if k in e})
        metadata[eid]['stratum'] = eligible.get(eid)
        metadata[eid]['lineage_aliases'] = sorted(set(e.get('lineage_aliases', ())))
    value = {
        'schema_version':MANIFEST_VERSION, 'algorithm':ALGORITHM, 'seed':seed,
        'dataset_identity':dict(dataset_identity), 'strata_config':config,
        'config_fingerprint':strata_fingerprint(config),
        'splits':{p:sorted(split[p]) for p in PARTS},
        'excluded_episode_ids':sorted(set(records)-set(eligible)),
        'episode_metadata':metadata,
        'lineage_groups':{g:list(ids) for g,ids in sorted(groups.items())},
        'global_counts':{p:len(split[p]) for p in PARTS},
        'counts_by_stratum':{s['name']:{p:sum(eligible[e]==s['name'] for e in split[p]) for p in PARTS} for s in config['strata']},
    }
    value = json.loads(_json(value))  # Detached, entirely serializable metadata.
    value['manifest_fingerprint'] = _fingerprint(value)
    return value


def serialize_split_manifest(manifest):
    if not isinstance(manifest, Mapping) or manifest.get('schema_version') != MANIFEST_VERSION:
        raise ValueError('Unsupported split manifest version')
    payload = {k:v for k,v in manifest.items() if k != 'manifest_fingerprint'}
    if manifest.get('manifest_fingerprint') != _fingerprint(payload):
        raise ValueError('Split manifest fingerprint mismatch')
    return (_json(dict(manifest))+'\n').encode('utf-8')


def read_split_manifest(path, episodes, strata_config, dataset_identity, *,
                        expected_seed=DEFAULT_SEED, expected_fingerprint=None):
    """Validate source/config plus independently declared freeze expectations.

    Official consumers pass both pins from training config. A self-consistent
    checksum alone does not authenticate the already-approved holdout.
    """
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    serialize_split_manifest(value)
    if type(expected_seed) is not int or type(value.get('seed')) is not int or value['seed'] != expected_seed:
        raise ValueError('Frozen split seed differs from expected seed')
    if expected_fingerprint is not None and value['manifest_fingerprint'] != _hash(expected_fingerprint):
        raise ValueError('Frozen split fingerprint differs from independently pinned fingerprint')
    expected = build_split_manifest(value.get('splits'), episodes, strata_config,
                                    value.get('seed'), dataset_identity)
    if value != expected:
        raise ValueError('Frozen split incompatible with source/config/lineage metadata')
    return value


def write_split_manifest(manifest, path, *, source_root):
    """Create exclusively; existing different artifacts are never overwritten."""
    data = serialize_split_manifest(manifest)
    path, source = Path(path), Path(source_root).resolve()
    if path.resolve().is_relative_to(source):
        raise ValueError('Split artifact cannot be written inside source dataset')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError('Existing frozen split differs; overwrite forbidden')


def audit_split(split, episodes, strata_config, *, action_families=()):
    """Diagnostic counts only. No feedback into seeded membership selection."""
    validate_split(split, episodes, strata_config)
    config = _config(strata_config)
    records, eligible, _ = _inventory(episodes, config)
    families = sorted(set(action_families) | {f for e in records.values() for f in e.get('supervised_instances', {})})
    result = {}
    for part in PARTS:
        rows = [records[e] for e in split[part]]
        sizes = sorted((e[config['eligible_count_field']], e['episode_id']) for e in rows)
        result[part] = {
            'episodes':len(rows), 'bc_valid_transitions':sum(n for n,_ in sizes),
            'stratum_counts':dict(Counter(eligible[e['episode_id']] for e in rows)),
            'kind_counts':dict(Counter(e.get('kind', 'unknown') for e in rows)),
            'supervised_instances':{f:sum(_count(e.get('supervised_instances', {}).get(f, 0), 'supervision count') for e in rows) for f in families},
            'episode_support':{f:sum(e.get('supervised_instances', {}).get(f, 0)>0 for e in rows) for f in families},
            'smallest_episodes':[{'episode_id':e,'bc_valid_transitions':n} for n,e in sizes[:3]],
            'largest_episodes':[{'episode_id':e,'bc_valid_transitions':n} for n,e in sizes[-3:][::-1]],
        }
    return result
