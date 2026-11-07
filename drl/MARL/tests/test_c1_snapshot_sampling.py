from collections import Counter
from types import SimpleNamespace as NS
import pytest
from drl.MARL.c1_snapshot_sampling import SnapshotSampler,STAGES


def bank():
    return NS(sha256='fixture',records=[dict(path=f'{d}/{s}/{seed}',stage=s,identity=dict(domain=d,split='train',seed=seed))
        for d,seeds in [('navigation',(7033,7034)),('gap',(16076,16077))] for s in STAGES for seed in seeds])


def test_snapshot_rng_is_reproducible_and_sibling_resets_are_independent():
    a=SnapshotSampler(bank(),('navigation','gap'),7033);b=SnapshotSampler(bank(),('navigation','gap'),7033)
    assert [a.sample(0) for _ in range(20)]==[b.sample(0) for _ in range(20)]
    for _ in range(100):a.sample(0)
    assert [a.sample(1) for _ in range(50)]==[b.sample(1) for _ in range(50)]


def test_stage_distribution_preserves_requested_mass():
    s=SnapshotSampler(bank(),('navigation',),0)
    counts=Counter(s.sample(0)['stage'] for _ in range(20000))
    for key,p in zip(STAGES,(.25,.20,.25,.30)):assert counts[key]/20000==pytest.approx(p,abs=.015)
    assert s.manifest()['environments'][0]['stage_counts']==counts


def test_missing_positive_stage_fails_instead_of_renormalizing():
    b=bank();b.records=[r for r in b.records if r['stage']!='S3']
    with pytest.raises(ValueError,match='Missing'):SnapshotSampler(b,('navigation',),0)


def test_train_bank_cannot_silently_supply_validation_initialization():
    with pytest.raises(ValueError,match='Missing'):SnapshotSampler(bank(),('navigation',),0,split='validation')
