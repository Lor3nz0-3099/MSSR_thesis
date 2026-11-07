"""Split-explicit snapshot curriculum; independent stage/scenario RNG per env."""
from collections import Counter
import hashlib
import json
import math
import random

STAGES=('S0','S1','S2','S3')
DEFAULT_WEIGHTS=dict(zip(STAGES,(.25,.20,.25,.30)))


class SnapshotSampler:
    def __init__(self,bank,domains_by_env,seed,*,weights=None,split='train'):
        from drl.MARL.c1_train_sampling import catalog_seeds
        if type(seed) is not int:raise ValueError('Snapshot RNG seed must be integer')
        self.bank,self.domains,self.seed,self.split=bank,tuple(domains_by_env),seed,split
        if not self.domains or split not in ('train','validation','test'):raise ValueError('Explicit supported snapshot split required')
        self.weights=dict(DEFAULT_WEIGHTS if weights is None else weights)
        if set(self.weights)!=set(STAGES) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in self.weights.values()) or not sum(self.weights.values()):
            raise ValueError('Snapshot stage weights must be finite nonnegative with positive mass')
        self.buckets={}
        for domain in set(self.domains):
            allowed=set(catalog_seeds(domain,split))
            for stage in STAGES:
                rows=[r for r in bank.records if r['stage']==stage and r['identity']['domain']==domain and r['identity']['split']==split]
                if any(r['identity']['seed'] not in allowed for r in rows):raise ValueError('Snapshot seed violates immutable split catalog')
                self.buckets[(domain,stage)]=sorted(rows,key=lambda r:(r['identity']['seed'],r['path']))
                if self.weights[stage] and not rows:raise ValueError(f'Missing {split} snapshots: {domain}/{stage}')
        self.rngs=[random.Random(int.from_bytes(hashlib.sha256(f'c1-snapshot:{seed}:{i}:{d}:{split}'.encode()).digest(),'big')) for i,d in enumerate(self.domains)]
        self.generations=[0]*len(self.domains);self.current=[None]*len(self.domains)
        self.counts=[Counter() for _ in self.domains]

    def sample(self,index):
        if type(index) is not int or not 0<=index<len(self.domains):raise IndexError('Invalid snapshot env index')
        rng=self.rngs[index];domain=self.domains[index]
        stage=rng.choices(STAGES,weights=[self.weights[s] for s in STAGES])[0]
        record=dict(rng.choice(self.buckets[(domain,stage)]))
        self.generations[index]+=1;self.counts[index][stage]+=1
        self.current[index]=record
        return record

    def manifest(self):
        return json.loads(json.dumps(dict(schema='mssr.c1.snapshot_sampler.v1',seed=self.seed,split=self.split,
            domains_by_env=self.domains,stage_weights=self.weights,bank_sha256=getattr(self.bank,'sha256',None),
            environments=[dict(env_index=i,generation=self.generations[i],current=self.current[i],
                stage_counts=dict(self.counts[i]),rng_state=r.getstate()) for i,r in enumerate(self.rngs)])))
