"""BC-v5 experimental TRAIN-only primitive-balanced sampler (opt-in, not wired to BC-v4).

Preserves complete original G2 samples. Applies moderate exposure correction
by morphology, task, provenance and primitive, not only structural actions. Never draws
from validation/test, never changes action labels or H1/H2.

IMPORTANT: collect the census before choosing a production quota/exponent.
"""
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from types import MappingProxyType

from .bc_v5_census import _groups, _primitive


def _digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _tuples(x):
    return tuple(_tuples(y) for y in x) if isinstance(x, (tuple, list)) else x


@dataclass(frozen=True)
class PrimitiveTrainIndex:
    """One TRAIN index; keys (morphology, task, provenance, primitive) -> episode -> rows."""
    episode_rows: dict
    group_episode_rows: dict
    g2_manifest_fingerprint: str
    sample_count: int = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self):
        episodes = {e: tuple(rows) for e, rows in sorted(self.episode_rows.items())}
        rows = [r for group in episodes.values() for r in group]
        if not self.g2_manifest_fingerprint or not rows or sorted(rows) != list(range(len(rows))) \
                or any(type(r) is not int for r in rows):
            raise ValueError('Invalid or incomplete TRAIN-only primitive index')
        groups = {}
        for key, episode_map in sorted(self.group_episode_rows.items()):
            if not isinstance(key, tuple) or len(key) != 4 or any(not isinstance(x, str) or not x for x in key):
                raise ValueError('Invalid morphology/task/provenance/primitive key')
            sub = {}
            for eid, values in sorted(episode_map.items()):
                vals = tuple(values)
                if eid not in episodes or not vals or len(set(vals)) != len(vals) or not set(vals).issubset(episodes[eid]):
                    raise ValueError('Primitive selection escaped its TRAIN episode')
                sub[eid] = vals
            if not sub:
                raise ValueError('Empty primitive group')
            groups[key] = MappingProxyType(sub)
        if not groups:
            raise ValueError('No supervised primitives in TRAIN')
        document = {'schema_version': 'mssr.bc_v5.primitive_train_index.v1',
                    'g2_manifest_fingerprint': self.g2_manifest_fingerprint,
                    'episode_rows': episodes,
                    'group_episode_rows': [(list(k), dict(v)) for k, v in groups.items()]}
        object.__setattr__(self, 'episode_rows', MappingProxyType(episodes))
        object.__setattr__(self, 'group_episode_rows', MappingProxyType(groups))
        object.__setattr__(self, 'sample_count', len(rows))
        object.__setattr__(self, 'fingerprint', _digest(document))


def build_primitive_train_index(data):
    """Scan SidecarDataset('train') once. Caller must authenticate G2 before this."""
    if getattr(data, 'split', None) != 'train':
        raise ValueError('Primitive sampler can only index TRAIN')
    manifest = getattr(data, 'manifest', {})
    fingerprint = manifest.get('manifest_fingerprint')
    if not fingerprint:
        raise ValueError('Authenticated G2 manifest required')
    episodes, grouped = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for idx in range(len(data)):
        # Read authenticated dynamic routing without cloning full H1/H2 tensors.
        _, _, dynamic = data.read_parts(idx)
        route = dynamic['routing']
        audit, sup = route['audit'], route['physical_supervision']
        eid = audit.episode_id
        if eid not in data.episode_ids:
            raise ValueError('TRAIN row belongs to unrecognized episode')
        episodes[eid].append(idx)
        morph, task, provenance = _groups(audit)
        primitives = {_primitive(x) for x in sup.resource_targets if x.supervised}
        primitives.update(_primitive(x, True) for x in sup.structural_targets if x.supervised)
        if not primitives:
            raise ValueError('TRAIN BC sample has no valid positive primitive')
        for primitive in primitives:
            grouped[(morph, task, provenance, primitive)][eid].append(idx)
    return PrimitiveTrainIndex(dict(episodes), {k: dict(v) for k, v in grouped.items()}, fingerprint)


class BalancedPrimitiveBatchSampler:
    """Mostly uniform row sampling + softened multi-stratum oversampling.

    For balanced slots, morphology is chosen uniformly, then task uniformly,
    then provenance uniformly, then primitive with probability proportional to
    (number of eligible rows)^alpha.
    Episode and row selection are uniform inside that group. alpha=0.5 gives
    less extreme oversampling than equal-frequency per-primitive selection.
    """
    SCHEMA = 'mssr.bc_v5.primitive_sampler_state.v1'

    def __init__(self, index, *, batch_size=64, balanced_quota=16,
                 frequency_exponent=0.5, seed=20261112):
        if not isinstance(index, PrimitiveTrainIndex):
            raise TypeError('PrimitiveTrainIndex required')
        if type(batch_size) is not int or batch_size < 1 or type(balanced_quota) is not int \
                or not 0 <= balanced_quota <= batch_size or type(seed) is not int \
                or type(frequency_exponent) not in (float, int) or not math.isfinite(frequency_exponent) \
                or not 0 <= frequency_exponent <= 1:
            raise ValueError('Invalid BC-v5 sampler settings')
        self.index = index
        self.batch_size = batch_size
        self.balanced_quota = balanced_quota
        self.frequency_exponent = float(frequency_exponent)
        self.seed = seed
        self.rng = random.Random(seed)
        self.position = self.epoch = 0
        self.last_draw = {}
        hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for key, episodes in index.group_episode_rows.items():
            morph, task, provenance, primitive = key
            count = sum(len(x) for x in episodes.values())
            hierarchy[morph][task][provenance].append((key, count ** self.frequency_exponent))
        self.hierarchy = {m: {t: dict(p) for t, p in tasks.items()}
                          for m, tasks in hierarchy.items()}
        self.morphs = tuple(sorted(self.hierarchy))

    def __len__(self):
        return math.ceil(self.index.sample_count / self.batch_size)

    def _one_balanced(self):
        m = self.rng.choice(self.morphs)
        task = self.rng.choice(sorted(self.hierarchy[m]))
        provenance = self.rng.choice(sorted(self.hierarchy[m][task]))
        options = self.hierarchy[m][task][provenance]
        key = self.rng.choices([x[0] for x in options], weights=[x[1] for x in options], k=1)[0]
        episodes = self.index.group_episode_rows[key]
        eid = self.rng.choice(tuple(episodes))
        return self.rng.choice(episodes[eid])

    def __iter__(self):
        if self.position == len(self):
            self.position = 0
            self.epoch += 1
        while self.position < len(self):
            ordinary = self.batch_size - self.balanced_quota
            batch = [self.rng.randrange(self.index.sample_count) for _ in range(ordinary)]
            batch += [self._one_balanced() for _ in range(self.balanced_quota)]
            self.position += 1
            self.last_draw = {'ordinary': ordinary, 'balanced': self.balanced_quota}
            yield batch

    def state_dict(self):
        return {'schema_version': self.SCHEMA, 'index_fingerprint': self.index.fingerprint,
                'config': {'batch_size': self.batch_size, 'balanced_quota': self.balanced_quota,
                           'frequency_exponent': self.frequency_exponent, 'seed': self.seed},
                'position': self.position, 'epoch': self.epoch, 'rng_state': self.rng.getstate()}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(state.get(k) != expected[k] for k in ('schema_version', 'index_fingerprint', 'config')):
            raise ValueError('Incompatible BC-v5 sampler identity/config')
        if type(state.get('position')) is not int or not 0 <= state['position'] <= len(self) \
                or type(state.get('epoch')) is not int or state['epoch'] < 0:
            raise ValueError('Invalid sampler cursor')
        rng = random.Random()
        rng.setstate(_tuples(state['rng_state']))
        self.rng = rng
        self.position = state['position']
        self.epoch = state['epoch']
        self.last_draw = {}
