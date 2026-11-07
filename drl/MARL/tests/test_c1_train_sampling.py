"""Pure TRAIN distribution and independently reproducible reset sampling."""
from collections import Counter
import json

import pytest

from drl.MARL.c1_train_sampling import NativeC1TrainSampler, dispersed_smoke_manifest
from drl.MARL.reward_calibration import MORPHOLOGIES, curriculum_identity

DOMAINS = tuple(MORPHOLOGIES)


def test_catalog_counts_reuse_existing_curriculum_identity():
    sampler = NativeC1TrainSampler(DOMAINS, seed=41)
    manifest = sampler.manifest()
    for domain in DOMAINS:
        seeds = manifest['seed_pools'][domain]
        assert len(seeds) == len(set(seeds)) == 1000
        assert Counter(curriculum_identity(domain, seed).difficulty for seed in seeds) == {
            'easy': 250, 'medium': 400, 'hard': 350}
    assert json.loads(json.dumps(manifest)) == manifest


@pytest.mark.parametrize('domain', DOMAINS)
@pytest.mark.parametrize('split', ['validation', 'test', 'challenge'])
def test_reject_nontrain_catalog_seed(domain, split):
    from drl.MARL.c1_train_sampling import catalog_seeds
    with pytest.raises(ValueError):
        NativeC1TrainSampler((domain,), seed=1, seed_pools={domain: [catalog_seeds(domain, split)[0]]})


def test_env_stream_does_not_depend_on_other_env_resets():
    a = NativeC1TrainSampler(('navigation', 'navigation'), seed=8)
    b = NativeC1TrainSampler(('navigation', 'navigation'), seed=8)
    previous = None
    for _ in range(50):
        a.sample(1)
        a.sample(1)
        value = a.sample(0)
        assert value == b.sample(0)
        assert value.seed != previous
        previous = value.seed
        assert value.split == 'train'
        assert value.difficulty == curriculum_identity(value.domain, value.seed).difficulty
    assert a.manifest()['environments'][0] == b.manifest()['environments'][0]
    with pytest.raises(IndexError):
        a.sample(-1)


def test_smoke_pools_span_strata_and_geometry_and_cycle():
    manifest = dispersed_smoke_manifest(seed=72)
    assert manifest == dispersed_smoke_manifest(seed=72)
    assert json.loads(json.dumps(manifest)) == manifest
    sampler = NativeC1TrainSampler(DOMAINS, seed=72, seed_pools=manifest['seed_pools'], cycle=True)
    for index, domain in enumerate(DOMAINS):
        seeds = manifest['seed_pools'][domain]
        assert len(seeds) == 6
        assert set(seeds) == {record['seed'] for record in manifest['samples'][domain]}
        assert Counter(curriculum_identity(domain, seed).difficulty for seed in seeds) == {
            'easy': 2, 'medium': 2, 'hard': 2}
        assert max(seeds) - min(seeds) > 800
        assert len({r['geometry_summary']['sha256'] for r in manifest['samples'][domain]}) == 6
        assert all(r['geometry_summary']['numeric_geometry_values'] > 0 for r in manifest['samples'][domain])
        first = [sampler.sample(index) for _ in range(6)]
        assert {sample.seed for sample in first} == set(seeds)
        assert {sample.difficulty for sample in first[:3]} == {'easy', 'medium', 'hard'}
        assert sampler.sample(index).seed != first[-1].seed


def test_custom_weights_and_single_seed_pool():
    sampler = NativeC1TrainSampler(('navigation',), seed=5,
                                  difficulty_weights={'easy': 1, 'medium': 0, 'hard': 0})
    assert {sampler.sample(0).difficulty for _ in range(100)} == {'easy'}
    single = NativeC1TrainSampler(('navigation',), seed=5, seed_pools={'navigation': [7000]})
    assert [single.sample(0).seed for _ in range(3)] == [7000] * 3
    with pytest.raises(ValueError):
        NativeC1TrainSampler(('navigation',), seed=1, seed_pools={'navigation': []})
    with pytest.raises(ValueError):
        NativeC1TrainSampler(('navigation',), seed=1, difficulty_weights={'easy': -1})


def test_even_unused_supplied_pools_must_be_train_only():
    with pytest.raises(ValueError):
        NativeC1TrainSampler(('navigation',), seed=1,
                              seed_pools={'navigation': [7000], 'stairs': [13000]})


def test_default_weights_are_catalog_proportions_and_cycles_do_not_repeat():
    sampler = NativeC1TrainSampler(DOMAINS, seed=23)
    assert all(weights == {'easy': 250., 'medium': 400., 'hard': 350.}
               for weights in sampler.manifest()['difficulty_weights'].values())
    smoke = dispersed_smoke_manifest(23)
    cyclic = NativeC1TrainSampler(DOMAINS, seed=23, seed_pools=smoke['seed_pools'], cycle=True)
    for env in range(4):
        sequence = [cyclic.sample(env).seed for _ in range(60)]
        assert all(a != b for a, b in zip(sequence, sequence[1:]))
        assert all(set(sequence[start:start + 6]) == set(smoke['seed_pools'][DOMAINS[env]])
                   for start in range(0, 60, 6))


@pytest.mark.parametrize('cycle', [False, True])
def test_pinned_initial_seeds_are_once_only_and_independent(cycle):
    pools = dispersed_smoke_manifest(31)['seed_pools']
    domains = DOMAINS * 4
    initial = tuple(pools[d][(0, 2, 4, 5)[i // 4]] for i, d in enumerate(domains))
    a = NativeC1TrainSampler(domains, seed=31, seed_pools=pools, cycle=cycle, initial_seeds=initial)
    b = NativeC1TrainSampler(domains, seed=31, seed_pools=pools, cycle=cycle, initial_seeds=initial)
    before = a.manifest()
    assert before['initial_seeds'] == list(initial)
    assert all(env['initial_pending'] for env in before['environments'])
    for i, value in enumerate(initial):
        sample = a.sample(i)
        assert sample.seed == value and sample.generation == 1
        assert sample == b.sample(i)
    after = a.manifest()
    assert not any(env['initial_pending'] for env in after['environments'])
    assert before['environments'][0]['rng_state'] == after['environments'][0]['rng_state']
    for _ in range(12):
        a.sample(1)
        previous = a.manifest()['environments'][0]['current']['seed']
        sample = a.sample(0)
        assert sample == b.sample(0)
        assert sample.seed != previous
    assert json.loads(json.dumps(a.manifest())) == a.manifest()


@pytest.mark.parametrize('initial', [(8000,), (9000,), (9500,), (7001,), (), (7000, 7000), (True,)])
def test_pinned_initial_seed_validated_against_train_and_pool(initial):
    with pytest.raises(ValueError):
        NativeC1TrainSampler(('navigation',), seed=1, seed_pools={'navigation': [7000]},
                              initial_seeds=initial)
