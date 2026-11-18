"""BC-v5 v8 TRAIN-only sampler: primitive-first, actor/expert-faithful.

Each mini-batch uses ordinary uniform rows plus priority rows drawn from
actual POSITIVE primitives. Rare means positive TRAIN count <= 1,000; when
known, one of the 16 priority slots is assigned to that category per batch.
No candidate changes and no synthetic task/primitive combinations.

Prior counts are authenticated Stage11 TRAIN aggregate counts. Bootstrap only
bounded routing rows, never performs a full G2 scan or opens TEST/VALIDATION.
Pools continue growing when the training loop reads more samples. With a prior,
all expected positive primitives MUST be discovered before starting training.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
import random

from .bc_v5_census import _groups, _primitive
from .action_set_targets import positive_indices


class AdaptivePrimitiveBatchSampler:
    SCHEMA = 'mssr.bc_v5.v8.primitive_first_sampler.v2'

    def __init__(self, data, *, batch_size=64, balanced_quota=16,
                 frequency_exponent=0.5, seed=20261021,
                 bootstrap_uniform=512, bootstrap_structural=512,
                 primitive_prior=None, rare_limit=1000):
        if getattr(data, 'split', None) != 'train':
            raise ValueError('Primitive sampler must use TRAIN only')
        if (type(batch_size) is not int or batch_size < 1
                or type(balanced_quota) is not int or not 0 <= balanced_quota <= batch_size
                or type(seed) is not int
                or type(frequency_exponent) not in (float, int)
                or not math.isfinite(frequency_exponent) or not 0 <= frequency_exponent <= 1
                or type(bootstrap_uniform) is not int or bootstrap_uniform < 0
                or type(bootstrap_structural) is not int or bootstrap_structural < 0
                or type(rare_limit) is not int or rare_limit < 1):
            raise ValueError('Invalid primitive sampler arguments')
        prior = {} if primitive_prior is None else dict(primitive_prior)
        if any(not isinstance(k, str) or not k or type(v) is not int or v < 1
               for k, v in prior.items()):
            raise ValueError('Invalid Stage11 positive primitive prior')
        self.data = data
        self.batch_size = batch_size
        self.balanced_quota = balanced_quota
        self.frequency_exponent = float(frequency_exponent)
        self.seed = seed
        self.rng = random.Random(seed)
        self.position = self.epoch = 0
        self.groups = defaultdict(set)
        self.observed = set()
        self.primitive_observations = Counter()
        self.primitive_prior = prior
        self.rare_limit = rare_limit
        self.n = len(data)
        self.last_balanced = Counter()
        self.last_uniform = 0
        self.last_balanced_total = 0
        if not self.n:
            raise ValueError('Empty TRAIN subset')

        raw_to_position = {raw: pos for pos, raw in enumerate(data.positions)}
        structural = []
        seen = 0
        for raw, present in data.underlying._data.db.execute(
                "SELECT idx,structural FROM samples WHERE split='train' ORDER BY idx"):
            seen += 1
            pos = raw_to_position.get(raw)
            if pos is not None and present:
                structural.append(pos)
        if seen != len(data.underlying):
            raise ValueError('Incomplete TRAIN SQLite metadata')
        self.structural_count = len(structural)

        # Deterministic, bounded discovery with priority for structural rows.
        unif = self.rng.sample(range(self.n), min(bootstrap_uniform, self.n))
        struct = self.rng.sample(structural, min(bootstrap_structural, len(structural)))
        initial_positions=sorted(set(unif + struct))
        print('[SAMPLER] Building bounded TRAIN bootstrap: '+str(len(initial_positions))+
              ' exact routing rows, no H1 tensor scan',flush=True)
        for number,pos in enumerate(initial_positions,start=1):
            self._observe_position(pos)
            if number%256==0 or number==len(initial_positions):
                print('[SAMPLER] bootstrap '+str(number)+'/'+str(len(initial_positions))+
                      ' | primitives found='+str(len(self._known_primitives())),flush=True)
        if prior:
            missing = set(prior) - self._known_primitives()
            if missing:
                print('[SAMPLER] Still discovering rare positives: '+
                      ', '.join(sorted(missing))+' (bounded TRAIN sample only)',flush=True)
                # Bounded fallback, not another full-dataset scan. The random
                # sequences are generated once and never include TEST rows.
                extended_struct = self.rng.sample(structural, min(4096, len(structural)))
                extended_uniform = self.rng.sample(range(self.n), min(4096, self.n))
                inspected = 0
                for pos in list(extended_struct) + list(extended_uniform):
                    if not missing:
                        break
                    if pos not in self.observed:
                        self._observe_position(pos)
                        inspected += 1
                        missing = set(prior) - self._known_primitives()
                        if inspected%256==0:
                            print('[SAMPLER] extra discovery '+str(inspected)+
                                  ' rows | missing='+str(len(missing)),flush=True)
            if missing:
                raise ValueError('Cannot guarantee complete primitive coverage from bounded '
                                 'TRAIN bootstrap; missing: ' + ', '.join(sorted(missing)))
        if not self.groups:
            raise ValueError('No supervised primitives discovered')
        self.bootstrap_sample_count = len(self.observed)
        self.bootstrap_signature = self._signature()

    def _known_primitives(self):
        return {k[3] for k, rows in self.groups.items() if rows}

    def _observe_position(self, pos):
        _tid, template, dynamic = self.data.read_parts(pos)
        route = {**template['routing'], **dynamic['routing']}
        self.observe_route(pos, route)

    def _signature(self):
        payload = {'schema':self.SCHEMA, 'batch_size':self.batch_size,
                   'balanced_quota':self.balanced_quota,
                   'frequency_exponent':self.frequency_exponent,
                   'seed':self.seed, 'rare_limit':self.rare_limit,
                   'primitive_prior':dict(sorted(self.primitive_prior.items())),
                   'groups':sorted((list(k), sorted(v)) for k,v in self.groups.items())}
        raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(raw).hexdigest()

    def observe_route(self, pos, route):
        if not 0 <= pos < self.n:
            raise ValueError('TRAIN sample index out of range')
        if pos in self.observed:
            return
        # Enumerated positive identities only: do not count historical
        # nonrepresentable align labels as positive supervised actions.
        positive = positive_indices(route)
        if not positive:
            raise ValueError('Zero-positive graph escaped readiness filter')
        audit = route['audit']
        morph, task, provenance = _groups(audit)
        physical = route['physical_supervision']
        types = {_primitive(t) for t in physical.resource_targets if t.supervised}
        types.update(_primitive(t, True) for t in physical.structural_targets if t.supervised)
        if not types:
            raise ValueError('No supervised positive primitive in TRAIN row')
        for primitive in types:
            if self.primitive_prior and primitive not in self.primitive_prior:
                raise ValueError('Primitive missing from certified Stage11 TRAIN census: '+primitive)
            self.groups[(morph, task, provenance, primitive)].add(pos)
            self.primitive_observations[primitive] += 1
        self.observed.add(pos)

    def observe_batch(self, positions, samples):
        if len(positions) != len(samples):
            raise ValueError('Sampler observe_batch lengths differ')
        for i, sample in zip(positions, samples):
            self.observe_route(i, sample['routing'])

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def _pick_primitive(self, primitives, *, rare=False):
        if not primitives:
            raise ValueError('Empty positive primitive pool')
        if rare:
            # Within the rare category, softer sqrt weighting avoids
            # massively oversampling e.g. gravity_settle's 67 positives.
            weights = [math.sqrt(self.primitive_prior.get(p,
                        max(1,self.primitive_observations[p]))) for p in primitives]
        else:
            # Primitive-first corrective exposure. All eligible primitives
            # are considered; less frequent ones get more priority.
            weights = [self.primitive_prior.get(p,
                       max(1,self.primitive_observations[p])) ** (-0.25)
                       for p in primitives]
        return self.rng.choices(primitives, weights=weights, k=1)[0]

    def _choose_from_primitive(self, primitive, used):
        # Every key is a real (morphology, task, provenance, primitive) found
        # in the TRAIN data; no incompatible task/action may be fabricated.
        choices = []
        for key, positions in self.groups.items():
            if key[3] == primitive and positions:
                choices.append((key, len(positions) ** self.frequency_exponent))
        if not choices:
            raise ValueError('No eligible TRAIN samples for '+primitive)
        selected = self.rng.choices([x[0] for x in choices],
                                    weights=[x[1] for x in choices], k=1)[0]
        pool = tuple(sorted(self.groups[selected]))
        # Avoid duplicates inside minibatch where possible.
        alternatives = [p for p in pool if p not in used]
        return self.rng.choice(alternatives if alternatives else pool)

    def __iter__(self):
        if self.position == len(self):
            self.position = 0
            self.epoch += 1
            self.last_balanced = Counter()
            self.last_balanced_total = self.last_uniform = 0
        while self.position < len(self):
            regular = self.batch_size - self.balanced_quota
            # True sample-uniform part; no alteration of empirical contexts.
            if regular <= self.n:
                batch = self.rng.sample(range(self.n), regular)
            else:
                batch = [self.rng.randrange(self.n) for _ in range(regular)]
            used = set(batch)
            primitives = sorted(self._known_primitives())
            rare = [p for p in primitives if self.primitive_prior.get(p, 10**9) <= self.rare_limit]
            regular_prims = [p for p in primitives if p not in rare]
            selected = []
            # At most one rare priority slot per batch, never an artificial
            # action and never a fabricated primitive/task pairing.
            if rare and self.balanced_quota:
                selected.append(self._pick_primitive(rare, rare=True))
            for _ in range(self.balanced_quota - len(selected)):
                selected.append(self._pick_primitive(regular_prims or primitives))
            for primitive in selected:
                pos = self._choose_from_primitive(primitive, used)
                batch.append(pos)
                used.add(pos)
                self.last_balanced[primitive] += 1
            self.last_balanced_total += len(selected)
            self.last_uniform += regular
            self.rng.shuffle(batch)
            self.position += 1
            yield batch

    def summary(self):
        return {'schema_version':self.SCHEMA,'train_rows':self.n,
                'batch_size':self.batch_size,'balanced_quota':self.balanced_quota,
                'frequency_exponent':self.frequency_exponent,
                'bootstrap_exact_rows':self.bootstrap_sample_count,
                'bootstrap_signature':self.bootstrap_signature,
                'structural_positive_rows_sqlite':self.structural_count,
                'observed_exact_rows':len(self.observed),
                'known_primitives':sorted(self._known_primitives()),
                'positive_primitive_pool_sizes':dict(sorted(self.primitive_observations.items())),
                'last_epoch_balanced_draws':dict(sorted(self.last_balanced.items())),
                'last_epoch_uniform_slots':self.last_uniform,
                'last_epoch_balanced_slots':self.last_balanced_total,
                'test_payload_access':0,
                'note':'Primitive-first; one rare priority slot per batch if prior and rare pools known; all choices are real TRAIN rows.'}
