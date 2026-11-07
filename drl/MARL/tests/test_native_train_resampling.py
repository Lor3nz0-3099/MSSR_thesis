"""Native reset metadata isolation; physical assertions live in sentinel smokes."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from drl.MARL.c1_train_sampling import NativeC1TrainSampler
from drl.MARL.c1_runner import make_course
from drl.MARL.isaaclab_c1_tasks import NativeC1TaskHooks


def test_resample_changes_only_selected_course_and_seed(monkeypatch):
    sampler = NativeC1TrainSampler(('gap','gap'), 13, {'gap': (16000,16911)}, cycle=True)
    first = [sampler.sample(i) for i in range(2)]
    courses = tuple(make_course('gap', s.seed, 'train') for s in first)
    calls = []
    monkeypatch.setattr('drl.MARL.isaaclab_course_reset.update_native_course',
        lambda *a, **kw: calls.append((a,kw)))
    env = SimpleNamespace(train_sampler=sampler, generation=[1,1],
        domain_by_env=('gap','gap'), seed_by_env=tuple(s.seed for s in first),
        courses=courses, sim=SimpleNamespace(stage=object()),
        scene=SimpleNamespace(env_origins=torch.zeros(2,3)))
    other_rng = deepcopy(sampler.manifest()['environments'][1])
    NativeC1TaskHooks._resample_course(env,0)
    assert env.seed_by_env[0] != first[0].seed
    assert env.courses[0].seed == env.seed_by_env[0]
    assert env.courses[1] is courses[1] and env.seed_by_env[1] == first[1].seed
    assert sampler.manifest()['environments'][1] == other_rng
    assert len(calls) == 1 and calls[0][1]['env_root'] == '/World/envs/env_0'


def test_initial_reset_keeps_already_sampled_initial_geometry(monkeypatch):
    env = SimpleNamespace(train_sampler=object(), generation=[0])
    NativeC1TaskHooks._resample_course(env,0)


def test_initial_expert_barrier_observes_total_campaign_wall_deadline():
    import pytest
    env=SimpleNamespace(_wall_deadline_s=0.)
    with pytest.raises(RuntimeError,match='wall budget'):
        NativeC1TaskHooks.assemble_until_ready(env,max_steps=10)
