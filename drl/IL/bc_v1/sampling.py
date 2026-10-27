"""Train-only G2 row selection. No tensors, normalization or loss policy."""
from dataclasses import dataclass, field
from types import MappingProxyType
import hashlib
import json
import math
import random


@dataclass(frozen=True)
class TrainRowIndex:
    episode_rows: dict
    structural_rows: dict
    identity: dict
    sample_count: int = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self):
        if self.identity.get('split_name')!='train': raise ValueError('Sampler index must be TRAIN only')
        episodes={k:tuple(v) for k,v in sorted(self.episode_rows.items())}
        structural={k:tuple(v) for k,v in sorted(self.structural_rows.items()) if v}
        rows=[i for v in episodes.values() for i in v]
        if any(type(i) is not int for i in rows) or sorted(rows)!=list(range(len(rows))):
            raise ValueError('Train index must contain every contiguous row exactly once')
        if not rows: raise ValueError('Empty TRAIN index')
        if any(k not in episodes or len(set(v))!=len(v) or not set(v)<=set(episodes[k]) for k,v in structural.items()):
            raise ValueError('Structural rows must belong to their TRAIN episode')
        identity=dict(self.identity)
        document={'schema_version':'mssr.bc_v1.train_row_index.v1','identity':identity,
            'episode_rows':episodes,'structural_rows':structural}
        digest=hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        object.__setattr__(self,'episode_rows',MappingProxyType(episodes))
        object.__setattr__(self,'structural_rows',MappingProxyType(structural))
        object.__setattr__(self,'identity',MappingProxyType(identity))
        object.__setattr__(self,'sample_count',len(rows));object.__setattr__(self,'fingerprint',digest)


def build_train_index(cache_root, *, expected_cache_fingerprint, normalizer_fingerprint=None, tensor_layout_fingerprint=None):
    """One sequential TRAIN episode pass; no validation/test payload is opened.

    This explicit operation is intentionally not called by sampler construction.
    The compact result can be retained by the caller; no artifact is published.
    """
    from . import dataset
    manifest=dataset.read_cache_manifest(cache_root)
    if manifest['cache_fingerprint']!=expected_cache_fingerprint: raise ValueError('G1 fingerprint mismatch')
    episodes, structural, offset={}, {}, 0
    for eid in manifest['splits']['train']:
        entry=manifest['episodes'][eid]
        samples=dataset._load_episode(cache_root,eid,entry)
        if len(samples)!=entry['samples']:raise ValueError('Incomplete train index coverage')
        episodes[eid]=tuple(range(offset,offset+len(samples)))
        rare=tuple(offset+i for i,s in enumerate(samples) if any(t.supervised for t in s.supervision.structural_targets))
        if rare:structural[eid]=rare
        offset+=len(samples)
        del samples
    identity={'g1':manifest['cache_fingerprint'],'split':manifest['identity']['split_fingerprint'],'split_name':'train'}
    if normalizer_fingerprint is not None:identity['f2']=normalizer_fingerprint
    if tensor_layout_fingerprint is not None:identity['tensor_layout']=tensor_layout_fingerprint
    return TrainRowIndex(episodes,structural,identity)


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value,(tuple,list)) else value


class JointActionBatchSampler:
    """Uniform ordinary rows + episode-uniform structural rows, with replacement.

    Position advances before each yield. A restored iterator continues at the
    next batch; starting after an exhausted epoch advances the epoch explicitly.
    Use num_workers=0 initially to avoid unsaved DataLoader prefetch.
    """
    def __init__(self, index, *, batch_size=64, ordinary_quota=56, structural_quota=8, seed=20261006):
        if not isinstance(index,TrainRowIndex):raise TypeError('Expected TRAIN row index')
        if (any(type(v) is not int or v<0 for v in (batch_size,ordinary_quota,structural_quota))
                or batch_size<1 or ordinary_quota+structural_quota!=batch_size or type(seed) is not int):
            raise ValueError('Invalid sampler quotas/seed')
        self.index=index;self.batch_size=batch_size;self.ordinary_quota=ordinary_quota
        self.structural_quota=structural_quota;self.seed=seed
        self.rng=random.Random(seed);self.position=0;self.epoch=0;self.last_draw={}
        self._rare_episodes=tuple(index.structural_rows)

    def __len__(self):return math.ceil(self.index.sample_count/self.batch_size)

    def __iter__(self):
        if self.position==len(self):self.position=0;self.epoch+=1
        while self.position<len(self):
            rare=self.structural_quota if self._rare_episodes else 0
            ordinary=self.batch_size-rare
            batch=[self.rng.randrange(self.index.sample_count) for _ in range(ordinary)]
            for _ in range(rare):
                eid=self.rng.choice(self._rare_episodes)
                batch.append(self.rng.choice(self.index.structural_rows[eid]))
            self.last_draw={'ordinary':ordinary,'structural':rare,'fallback':not bool(self._rare_episodes)}
            self.position+=1
            yield batch

    def state_dict(self):
        return {'schema_version':'mssr.bc_v1.sampler_state.v1','index_fingerprint':self.index.fingerprint,
            'config':{'batch_size':self.batch_size,'ordinary_quota':self.ordinary_quota,
                      'structural_quota':self.structural_quota,'seed':self.seed},
            'position':self.position,'epoch':self.epoch,'rng_state':self.rng.getstate()}

    def load_state_dict(self,state):
        expected=self.state_dict()
        if any(state.get(k)!=expected[k] for k in ('schema_version','index_fingerprint','config')):
            raise ValueError('Incompatible sampler resume identity/config')
        if (type(state.get('position')) is not int or not 0<=state['position']<=len(self)
                or type(state.get('epoch')) is not int or state['epoch']<0):
            raise ValueError('Invalid sampler resume cursor')
        rng=random.Random();rng.setstate(_tuples(state['rng_state']))
        self.rng=rng;self.position=state['position'];self.epoch=state['epoch'];self.last_draw={}


def evaluation_batches(sample_count, *, batch_size=64, split_name):
    if split_name not in ('validation','test'):raise ValueError('Evaluation requires validation/test split')
    if type(sample_count) is not int or sample_count<0 or type(batch_size) is not int or batch_size<1:
        raise ValueError('Invalid evaluation count/batch size')
    for start in range(0,sample_count,batch_size):yield list(range(start,min(start+batch_size,sample_count)))
