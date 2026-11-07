"""TRAIN-only native C1 reset sampling without simulator dependencies.

The caller owns balanced domain slots. Each slot owns an independent RNG, so
resetting one environment never advances another environment's seed sequence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
import hashlib
import json
import math
import random

from drl.MARL.reward_calibration import MORPHOLOGIES, curriculum_identity

DIFFICULTIES = ('easy', 'medium', 'hard')


def catalog_seeds(domain: str, split: str = 'train') -> tuple[int, ...]:
    """Use the existing immutable split allocations, never inferred seed ranges."""
    from smores_ep.isaac.phase1_marl_curriculum import curriculum_seeds
    from smores_ep.isaac.phase1_stairs_curriculum import stair_curriculum_seeds
    from smores_ep.isaac.phase1_gap_curriculum import gap_curriculum_seeds
    from smores_ep.isaac.phase1_manipulation_curriculum import manipulation_curriculum_seeds
    allocations = dict(navigation=curriculum_seeds, stairs=stair_curriculum_seeds,
                       gap=gap_curriculum_seeds, manipulation=manipulation_curriculum_seeds)
    if domain not in allocations:
        raise ValueError(f'Unsupported C1 domain: {domain!r}')
    return tuple(allocations[domain](split))


@lru_cache(maxsize=4)
def _train_catalog(domain):
    return tuple((seed, curriculum_identity(domain, seed)) for seed in catalog_seeds(domain))


@dataclass(frozen=True)
class C1TrainSample:
    domain: str
    seed: int
    split: str
    difficulty: str
    morphology: str
    generation: int


class NativeC1TrainSampler:
    """Sample reset courses; default difficulty mass follows the full catalog.

    ``cycle=True`` is an explicitly bounded smoke mode: visit every provided
    seed once per cycle, interleaving shuffled difficulty strata. It does not
    model the production curriculum proportions and rejects custom weights.
    ``initial_seeds`` pins the first sample per environment for reproducible
    smoke geometry; later samples use the ordinary configured distribution.
    """

    def __init__(self, domains_by_env: tuple[str, ...], seed: int, seed_pools=None,
                 difficulty_weights=None, *, cycle: bool = False, initial_seeds=None):
        if type(seed) is not int:
            raise ValueError('sampler seed must be an integer')
        self.domains_by_env = tuple(domains_by_env)
        if not self.domains_by_env or any(d not in MORPHOLOGIES for d in self.domains_by_env):
            raise ValueError('nonempty supported domain slots are required')
        if cycle and difficulty_weights is not None:
            raise ValueError('smoke cycle and difficulty weights are mutually exclusive')
        if seed_pools is not None and set(seed_pools) - set(MORPHOLOGIES):
            raise ValueError('unknown seed-pool domain')
        if difficulty_weights is not None:
            if set(difficulty_weights) - set(DIFFICULTIES):
                raise ValueError('unknown difficulty weight')
            if any(not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
                   for w in difficulty_weights.values()):
                raise ValueError('difficulty weights must be finite and nonnegative')
        self.seed = seed
        self.cycle = bool(cycle)
        self._pools, self._strata, self._weights, self._counts = {}, {}, {}, {}
        for domain in dict.fromkeys((*self.domains_by_env, *(seed_pools or {}))):
            catalog = dict(_train_catalog(domain))
            pool = tuple(seed_pools.get(domain, catalog) if seed_pools is not None else catalog)
            if not pool or len(set(pool)) != len(pool):
                raise ValueError('seed pools must be nonempty and unique')
            for value in pool:
                if type(value) is not int or value not in catalog:
                    raise ValueError(f'{domain} pool seed {value!r} is outside TRAIN')
                identity = curriculum_identity(domain, value)
                if identity.split != 'train':
                    raise ValueError('sampler accepts TRAIN only')
            self._pools[domain] = pool
            self._strata[domain] = {
                difficulty: tuple(s for s in pool if catalog[s].difficulty == difficulty)
                for difficulty in DIFFICULTIES}
            counts = Counter(i.difficulty for i in catalog.values())
            self._counts[domain] = dict(counts)
            self._weights[domain] = {
                d: float(counts[d] if difficulty_weights is None else difficulty_weights.get(d, 0))
                for d in DIFFICULTIES}
            if not any(self._strata[domain][d] and self._weights[domain][d] > 0 for d in DIFFICULTIES):
                raise ValueError(f'{domain} pool has no positively weighted difficulty')
        self._rngs = [random.Random(int.from_bytes(hashlib.sha256(
            f'c1-train:{seed}:{index}:{domain}'.encode()).digest(), 'big'))
            for index, domain in enumerate(self.domains_by_env)]
        self._current = [None] * len(self.domains_by_env)
        self._generation = [0] * len(self.domains_by_env)
        self._pending = [[] for _ in self.domains_by_env]
        self.initial_seeds = None if initial_seeds is None else tuple(initial_seeds)
        if self.initial_seeds is not None:
            if len(self.initial_seeds) != len(self.domains_by_env):
                raise ValueError('one initial TRAIN seed per environment is required')
            for domain, value in zip(self.domains_by_env, self.initial_seeds):
                if type(value) is not int or value not in self._pools[domain]:
                    raise ValueError(f'{domain} initial seed {value!r} is outside its TRAIN pool')

    def sample(self, env_index: int) -> C1TrainSample:
        if type(env_index) is not int or not 0 <= env_index < len(self.domains_by_env):
            raise IndexError('environment index outside sampler slots')
        domain, rng = self.domains_by_env[env_index], self._rngs[env_index]
        previous = self._current[env_index]
        last = None if previous is None else previous.seed
        # Explicit smoke assignments consume neither RNG state nor a cycle.
        # The next draw sees this seed as its predecessor and avoids repeating.
        if self.initial_seeds is not None and self._generation[env_index] == 0:
            selected = self.initial_seeds[env_index]
        elif self.cycle:
            if not self._pending[env_index]:
                buckets = [list(self._strata[domain][d]) for d in DIFFICULTIES]
                for bucket in buckets:
                    rng.shuffle(bucket)
                order = []
                while any(buckets):
                    order.extend(bucket.pop() for bucket in buckets if bucket)
                if len(order) > 1 and order[0] == last:
                    order = order[1:] + order[:1]
                self._pending[env_index] = order
            selected = self._pending[env_index].pop(0)
        else:
            choices = {d: tuple(s for s in self._strata[domain][d] if s != last)
                       for d in DIFFICULTIES}
            available = [d for d in DIFFICULTIES if choices[d] and self._weights[domain][d] > 0]
            if not available:
                # A single positively weighted seed necessarily repeats.
                selected = last if last is not None else next(
                    s for d in DIFFICULTIES if self._weights[domain][d] > 0
                    for s in self._strata[domain][d])
            else:
                difficulty = rng.choices(available, weights=[self._weights[domain][d] for d in available])[0]
                selected = rng.choice(choices[difficulty])
        identity = curriculum_identity(domain, selected)
        self._generation[env_index] += 1
        sample = C1TrainSample(domain, selected, identity.split, identity.difficulty,
                               MORPHOLOGIES[domain], self._generation[env_index])
        self._current[env_index] = sample
        return sample

    def manifest(self) -> dict:
        """JSON snapshot includes exact pools, current courses and RNG state."""
        value = dict(schema='mssr.c1_train_sampler.v1', seed=self.seed,
                     domains_by_env=self.domains_by_env, cycle=self.cycle,
                     initial_seeds=self.initial_seeds,
                     seed_pools=self._pools, catalog_difficulty_counts=self._counts,
                     difficulty_weights=self._weights, environments=[
                         dict(env_index=i, domain=d, generation=self._generation[i],
                              initial_pending=self.initial_seeds is not None and self._generation[i] == 0,
                              current=None if self._current[i] is None else asdict(self._current[i]),
                              rng_state=self._rngs[i].getstate(), pending_cycle=self._pending[i])
                         for i, d in enumerate(self.domains_by_env)])
        return json.loads(json.dumps(value, allow_nan=False))


def _geometry_summary(course):
    """Hash numeric geometry only, excluding seed/split/difficulty metadata."""
    raw = asdict(course) if is_dataclass(course) else course
    values = {}

    def collect(value, path=''):
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                if key not in {'seed', 'ordinal', 'split', 'difficulty', 'family', 'generator_id'}:
                    collect(item, f'{path}.{key}' if path else key)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                collect(item, f'{path}[{index}]')
        elif type(value) in (int, float):
            if not math.isfinite(value):
                raise ValueError('nonfinite course geometry')
            values[path] = value

    collect(raw)
    if not values:
        raise ValueError('course generator provided no numeric geometry')
    encoded = json.dumps(values, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), numeric_geometry_values=len(values),
                # Save the actual dimensions as evidence, rather than a seed-only hash.
                geometry=values)


def dispersed_smoke_manifest(seed: int = 0) -> dict:
    """Select six exact TRAIN seeds/domain, two widely separated per stratum."""
    from drl.MARL.c1_runner import make_course
    if type(seed) is not int:
        raise ValueError('smoke seed must be an integer')
    rng = random.Random(seed)
    pools, samples = {}, {}
    for domain in MORPHOLOGIES:
        catalog = _train_catalog(domain)
        selected = []
        for difficulty in DIFFICULTIES:
            bucket = [s for s, identity in catalog if identity.difficulty == difficulty]
            # Low/high deciles spread seeds across every difficulty allocation.
            width = max(1, len(bucket) // 10)
            selected.extend((rng.choice(bucket[:width]), rng.choice(bucket[-width:])))
        pools[domain] = selected
        samples[domain] = [dict(domain=domain, seed=value, split='train',
            difficulty=curriculum_identity(domain, value).difficulty,
            morphology=MORPHOLOGIES[domain],
            geometry_summary=_geometry_summary(make_course(domain, value, 'train')))
            for value in selected]
        if len({r['geometry_summary']['sha256'] for r in samples[domain]}) != len(selected):
            raise ValueError(f'{domain} smoke courses have duplicate numeric geometry')
    return dict(schema='mssr.c1_dispersed_smoke.v1', seed=seed, seed_pools=pools, samples=samples)
