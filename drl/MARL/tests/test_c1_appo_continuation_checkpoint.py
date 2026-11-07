"""Offline continuation contracts: no simulator, training run, or native reset."""
from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from drl.MARL.c1_expert_snapshots import SnapshotBank, save_snapshot
from drl.MARL.c1_snapshot_sampling import SnapshotSampler
from drl.MARL.c1_critic_quality_gate import C1CriticQualityGate
from drl.MARL.c1_longtrain_checkpoint import restore_rng
from drl.MARL.tests.test_c1_expert_snapshots import payload


class TinyLearner:
    """SKRL's real nested registry/save protocol with real Torch models/Adam."""
    def __init__(self, device='cpu'):
        self.policy = torch.nn.Linear(2, 1).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=.003)
        self.checkpoint_modules = {'agent': {'policy': self.policy, 'actor_optimizer': self.optimizer}}
        self.memories = {}
        self._rollout = 0
        self._smdp_pending = None
        self._smdp_policy_version = 0

    def step(self):
        self.optimizer.zero_grad()
        self.policy(torch.tensor([[1., 2.]], device=next(self.policy.parameters()).device)).square().sum().backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({uid: {name: module.state_dict() for name, module in modules.items()}
                    for uid, modules in self.checkpoint_modules.items()}, path)


@pytest.fixture
def args(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    checkpoint = tmp_path/'initial.pt'; checkpoint.write_bytes(b'initial fixture')
    reward = tmp_path/'reward.json'; reward.write_text('{}')
    return NS(repository_root=repo, train=True, c1_staged_training=True,
        num_envs=1, rollout_steps=8, learning_epochs=2, mini_batches=1,
        curriculum='expert_snapshots', initialization_mode='bc_v4', structural_scheduler='c1',
        domains=['navigation'], global_seed=7033, device='cpu', episode_seconds=10.,
        c1_probe_config=None, alpha=.5, gamma=.99, altruism_radius_m=2.,
        checkpoint=checkpoint, reward_config=reward, snapshot_stage_weights={'S0':1.,'S1':0.,'S2':0.,'S3':0.})


@pytest.fixture
def sampler(tmp_path, args):
    records=[]
    for seed in (7033, 7034):
        value=payload()
        value['identity'].update(seed=seed, scenario_id=f'navigation:train:{seed}')
        path=tmp_path/f'state_{seed}.json'
        save_snapshot(path,value)
        records.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            identity=value['identity'],stage='S0'))
    args.snapshot_bank=tmp_path/'bank.json'
    args.snapshot_bank.write_text(json.dumps(dict(schema='mssr.c1_snapshot_bank.v1',records=records)))
    result=SnapshotSampler(SnapshotBank(args.snapshot_bank,'train'), ['navigation'],7033,
                           weights=args.snapshot_stage_weights)
    for _ in range(7):result.sample(0)
    return result


def counters(update=109):
    return dict(completed_updates=update,vector_steps=900,smdp_transition_count=400,
                smdp_event_step_count=70,smdp_policy_version=105)


def gate():
    result=C1CriticQualityGate().state_dict()
    result.update(last_completed=109, first_actor_update=5)
    return result


def provenance():
    return dict(historical_checkpoint_sha256='a'*64,previous_checkpoint_sha256='b'*64,
                discarded_pending_options=3, discarded_physical_episodes=1)


def api():
    from drl.MARL import c1_appo_continuation_checkpoint as continuation
    return continuation


def save(args,sampler,tmp_path,learner=None):
    learner=learner or TinyLearner();learner.step()
    path=api().save_continuation_checkpoint(args,tmp_path/'output',learner,sampler,
        counters=counters(),critic_gate_state=gate(),provenance=provenance())
    return learner,path


def test_snapshot_restore_preserves_exact_next_draws_and_rejects_foreign_current(sampler):
    saved=sampler.manifest()
    expected=[sampler.sample(0) for _ in range(20)]
    api().restore_snapshot_sampler(sampler,saved)
    assert sampler.manifest()==saved
    assert [sampler.sample(0) for _ in range(20)]==expected
    invalid=deepcopy(saved);invalid['environments'][0]['current']['path']='outside.json'
    before=sampler.manifest()
    with pytest.raises(ValueError,match='bank'):
        api().restore_snapshot_sampler(sampler,invalid)
    assert sampler.manifest()==before


@pytest.mark.parametrize('mutation',[
    lambda s:s.__setitem__('bank_sha256','c'*64),
    lambda s:s.__setitem__('seed',7034),
    lambda s:s.__setitem__('split','validation'),
    lambda s:s.__setitem__('stage_weights',{'S0':.5,'S1':.5,'S2':0.,'S3':0.}),
    lambda s:s['environments'][0].__setitem__('generation',-1),
    lambda s:s['environments'][0].__setitem__('stage_counts',{'S0':3}),
    lambda s:s['environments'][0].__setitem__('rng_state',[0,[],None]),
    lambda s:s['environments'][0].__setitem__('env_index',1),
])
def test_sampler_validation_is_transactional(sampler,mutation):
    saved=sampler.manifest();mutation(saved);before=sampler.manifest()
    with pytest.raises(ValueError):api().restore_snapshot_sampler(sampler,saved)
    assert sampler.manifest()==before


def test_save_full_state_rng_and_exclusive_publication(args,sampler,tmp_path):
    learner,path=save(args,sampler,tmp_path)
    state=api().read_continuation_manifest(path,expected_fingerprint=api().continuation_fingerprint(args))
    assert state['counters']==counters()
    assert state['sampler_manifest']==sampler.manifest()
    assert state['critic_gate_state']==gate()
    assert state['provenance']==provenance()
    assert state['physics_resume']=='new_episode_not_bitwise_continuation'
    assert state['checkpoint_registry']=={'agent':['actor_optimizer','policy']}
    raw=torch.load(path,map_location='cpu',weights_only=False)
    assert raw['agent']['actor_optimizer']['state'][0]['step']==1
    assert torch.equal(raw['agent']['policy']['weight'],learner.policy.weight.cpu())
    expected=(random.random(),np.random.rand(),torch.rand(2))
    random.random();np.random.rand();torch.rand(2)
    restore_rng(state['rng'])
    actual=(random.random(),np.random.rand(),torch.rand(2))
    assert actual[:2]==expected[:2]
    assert torch.equal(actual[2],expected[2])
    before=path.read_bytes()
    with pytest.raises(FileExistsError):save(args,sampler,tmp_path)
    assert path.read_bytes()==before


@pytest.mark.parametrize('mutation,match',[
    (lambda s:s.__setitem__('schema','wrong'),'schema'),
    (lambda s:s['counters'].__setitem__('completed_updates',0),'update'),
    (lambda s:s['counters'].__setitem__('smdp_policy_version',110),'version'),
    (lambda s:s['critic_gate_state'].__setitem__('last_completed',108),'gate'),
    (lambda s:s['sampler_manifest'].__setitem__('bank_sha256','c'*64),'bank'),
    (lambda s:s['checkpoint_registry'].__setitem__('agent',['policy']),'registry'),
    (lambda s:s.__setitem__('rng',{}),'RNG'),
    (lambda s:s['provenance'].__setitem__('historical_checkpoint_sha256','invalid'),'provenance'),
])
def test_manifest_rejects_corrupt_state(args,sampler,tmp_path,mutation,match):
    _,path=save(args,sampler,tmp_path)
    metadata=Path(str(path)+'.continuation.json')
    state=json.loads(metadata.read_text());mutation(state);metadata.write_text(json.dumps(state))
    with pytest.raises(ValueError,match=match):api().read_continuation_manifest(path)


def test_checkpoint_bytes_and_scientific_fingerprint_are_checked(args,sampler,tmp_path):
    _,path=save(args,sampler,tmp_path)
    changed=deepcopy(api().continuation_fingerprint(args));changed['alpha']=.9
    with pytest.raises(ValueError,match='fingerprint|Configuration'):
        api().read_continuation_manifest(path,expected_fingerprint=changed)
    with path.open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='corrupt|hash'):
        api().read_continuation_manifest(path)


def test_save_requires_finished_update_and_matching_quality_gate(args,sampler,tmp_path):
    learner=TinyLearner()
    with pytest.raises(ValueError,match='update'):
        api().save_continuation_checkpoint(args,tmp_path/'out',learner,sampler,
            counters=counters(0),critic_gate_state=gate(),provenance=provenance())
    assert not (tmp_path/'out').exists()


def test_fingerprint_covers_bank_weights_and_current_physics(args,sampler):
    first=api().continuation_fingerprint(args)
    assert first['snapshot_bank_sha256']==sampler.bank.sha256
    assert first['snapshot_stage_weights']==sampler.weights
    assert len(first['snapshot_physics']['source_sha256'])==64
    args.snapshot_stage_weights={'S0':.5,'S1':.5,'S2':0.,'S3':0.}
    assert api().continuation_fingerprint(args)!=first
