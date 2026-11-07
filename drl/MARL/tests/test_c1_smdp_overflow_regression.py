"""B2: full event rows survive pre/post closure without circular overwrite."""
import pytest
import torch

from drl.MARL.mappo import MSSRMAPPO, MultiAgent
from drl.MARL.tests.c1_smdp_fixtures import make_learner, frame, ScalarCritic


def observe_updates(learner, monkeypatch):
    batches = []
    monkeypatch.setattr(MultiAgent, 'post_interaction', lambda *a, **k: None)
    def update(**_):
        assert torch.is_grad_enabled(), 'Update must stay outside no-grad collection'
        memory = learner.memories['a']
        structured = learner._structured_rollout_sidecar._samples
        altruistic = learner._altruistic_rollout_sidecar.flatten()
        assert len(memory) == len(structured) == len(altruistic)
        states = memory.get_tensor_by_name('states').reshape(-1, 1)
        rewards = memory.get_tensor_by_name('rewards').reshape(-1, 1)
        for i, sample in enumerate(structured):
            assert sample.graph_snapshot.item() == states[i].item()
            assert altruistic[i].owner_rewards['a'].item() == rewards[i].item()
        # Exercise real target construction and coordinated sampling, not just
        # the collector's row count. Literal return expectations checked below.
        owner, mf = MSSRMAPPO.evaluate_altruistic_rollout_values(
            structured_samples=structured, altruistic_sidecar=learner._altruistic_rollout_sidecar,
            possible_agents=['a'], owner_critic=ScalarCritic(), mean_field_critic=ScalarCritic(3.))
        own_boot, mf_boot = MSSRMAPPO.evaluate_altruistic_bootstrap_values(
            next_critic_graphs=learner._current_next_critic_graph,
            altruistic_sidecar=learner._altruistic_rollout_sidecar, possible_agents=['a'],
            owner_critic=ScalarCritic(), mean_field_critic=ScalarCritic(3.), structured_samples=structured)
        dones = memory.get_tensor_by_name('terminated') | memory.get_tensor_by_name('truncated')
        next_owner, next_mf = MSSRMAPPO.evaluate_altruistic_rollout_values(
            structured_samples=structured, altruistic_sidecar=learner._altruistic_rollout_sidecar,
            possible_agents=['a'], owner_critic=ScalarCritic(), mean_field_critic=ScalarCritic(3.), next_state=True)
        targets = MSSRMAPPO.compute_altruistic_rollout_targets(
            sidecar=learner._altruistic_rollout_sidecar, possible_agents=['a'], owner_values=owner,
            mean_field_values=mf, owner_bootstrap_values=own_boot, mean_field_bootstrap_values=mf_boot,
            dones=dones, terminated=memory.get_tensor_by_name('terminated'),
            owner_next_values=next_owner, mean_field_next_values=next_mf,
            discount_factor=.9, lambda_coefficient=0., alpha=.5,
            duration_steps=torch.tensor([s.duration_steps for s in structured]).reshape(dones.shape))
        owner_batches, side_batches = MSSRMAPPO.sample_coordinated_batches(
            memories=learner.memories, sidecar=learner._structured_rollout_sidecar,
            names=['states', 'rewards'], batch_size=len(memory), mini_batches=2)
        target_batches = MSSRMAPPO.sample_owner_tensor_map_minibatches(
            tensors_by_owner=targets.owner_returns, indexes=memory.sampling_indexes, mini_batches=2)
        indexes = memory.sampling_indexes.tolist()
        assert sorted(indexes) == list(range(len(memory)))
        offset = 0
        for tensors, samples, target in zip(owner_batches['a'], side_batches, target_batches):
            for j, sample in enumerate(samples):
                assert sample.graph_snapshot.item() == tensors[0][j].item()
                assert target['a'][j].item() == targets.owner_returns['a'].reshape(-1)[indexes[offset+j]].item()
            offset += len(samples)
        batches.append(dict(states=states.clone(), rewards=rewards.clone(),
            returns=targets.owner_returns['a'].clone(), durations=[s.duration_steps for s in structured]))
        learner._last_update_metrics = {'actor_optimizer_steps': 1}
    learner.coordinated_update = update
    return batches


def post(learner):
    learner.post_interaction(timestep=100, timesteps=1000)


@pytest.mark.parametrize('terminal', [True, False])
def test_pre_post_closures_at_capacity_minus_one_survive_across_two_updates(monkeypatch, terminal):
    learner = make_learner(capacity=32)
    batches = observe_updates(learner, monkeypatch)
    with torch.no_grad():
        frame(learner, [True, True], [0., 0.], [1., 1.])
        for i in range(1, 32):
            frame(learner, [True, True], [float(i)]*2, [float(i+1)]*2)
        frame(learner, [True, True], [32., 32.], [1000., 1000.],
            terminated=[terminal]*2, truncated=[not terminal]*2)
    assert len(learner.memories['a']) == len(learner._structured_rollout_sidecar) == 66
    assert learner.memories['a'].get_tensor_by_name('observations')[0, 0].item() == 0.
    post(learner)
    assert len(batches) == 1
    assert batches[0]['states'].reshape(33, 2)[:, 0].tolist() == list(range(33))
    assert batches[0]['rewards'][-1].item() == 1000.
    assert batches[0]['returns'][-1, 0].item() == pytest.approx(1000. if terminal else 1029.7)
    for i in range(33, 66):
        with torch.no_grad():
            frame(learner, [True, True], [float(i)]*2, [float(i+1)]*2)
        post(learner)
    assert len(batches) == 2
    assert batches[1]['states'].reshape(32, 2)[:, 0].tolist() == list(range(33, 65))
    assert learner.smdp_transition_count == 130
    assert learner.smdp_event_step_count == 65
    assert learner.smdp_policy_version == 2


def test_full_buffer_never_overwrites_before_post_interaction(monkeypatch):
    learner = make_learner(capacity=2)
    batches = observe_updates(learner, monkeypatch)
    frame(learner, [True, True], [0., 0.], [1., 1.])
    frame(learner, [True, True], [1., 1.], [2., 2.])
    frame(learner, [True, True], [2., 2.], [3., 3.])
    frame(learner, [True, True], [3., 3.], [4., 4.], terminated=[True, True])
    assert len(learner._structured_rollout_sidecar) == len(learner.memories['a']) == 8
    assert learner.memories['a'].get_tensor_by_name('states')[:, 0, 0].tolist() == [0., 1., 2., 3.]
    assert not batches, 'Collection must not update the learner'
    post(learner)
    assert [b['states'][::2, 0].tolist() for b in batches] == [[0., 1., 2., 3.]]
    assert batches[-1]['returns'][-1, 0].item() == 4.
    assert learner._rollout == 0


def test_pre_and_post_closure_in_different_envs_keeps_padding_order(monkeypatch):
    learner = make_learner(capacity=2)
    batches = observe_updates(learner, monkeypatch)
    frame(learner, [True, True], [0., 10.], [1., 2.])
    frame(learner, [True, False], [1., 11.], [3., 4.])
    frame(learner, [True, False], [2., 12.], [5., 6.], terminated=[False, True])
    assert len(learner._structured_rollout_sidecar) == len(learner.memories['a']) == 6
    post(learner)
    # Last row: env0 padding, env1 initial epoch duration 3, return
    # 2 + .9*4 + .9**2*6 = 10.46 (terminal, not bootstrapped).
    assert batches[0]['durations'][-2:] == [0, 3]
    assert batches[0]['returns'][2, 1].item() == pytest.approx(10.46)
    frame(learner, [True, True], [3., 100.], [7., 8.])
    post(learner)
    frame(learner, [True, True], [4., 101.], [9., 10.])
    post(learner)
    assert len(batches) == 2
    assert learner.smdp_transition_count == 6


def test_unused_reserve_never_enters_targets_and_filled_storage_cannot_wrap():
    learner = make_learner(capacity=2)
    memory = learner.memories['a']
    frame(learner, [True, True], [0., 0.], [1., 1.])
    for i in range(1, 5):
        frame(learner, [True, True], [float(i)]*2, [float(i+1)]*2)
    assert memory.get_tensor_by_name('states').shape == (4, 2, 1)
    before = memory.get_tensor_by_name('states').clone()
    with pytest.raises(BufferError):
        frame(learner, [True, True], [5., 5.], [6., 6.])
    assert torch.equal(memory.get_tensor_by_name('states'), before)


def test_frame_overflow_never_defers_an_accepted_lag_one_decision_across_update(monkeypatch):
    learner = make_learner(capacity=32)
    monkeypatch.setattr(MultiAgent, 'post_interaction', lambda *a, **k: None)
    versions = []
    def update(**_):
        for sample in learner._structured_rollout_sidecar._samples[1::2]:
            if sample.rollout_active:
                # Env1 has only the initial behavior decision, version zero.
                assert learner.smdp_policy_version <= 1
                versions.append(learner.smdp_policy_version)
        learner._last_update_metrics = {'actor_optimizer_steps': 1}
    learner.coordinated_update = update
    for i in range(96):
        frame(learner, [True, i == 0], [float(i), 100.+i], [1., 1.], terminated=[False, i == 64])
        post(learner)
    assert versions == [1]


def test_two_closures_in_one_frame_produce_one_post_update(monkeypatch):
    learner = make_learner(capacity=1)
    batches = observe_updates(learner, monkeypatch)
    frame(learner, [True, True], [0., 0.], [1., 1.])
    post(learner)
    frame(learner, [True, True], [1., 1.], [2., 2.], terminated=[True, True])
    post(learner)
    assert len(batches) == learner.smdp_policy_version == 1
    assert batches[0]['states'].flatten().tolist() == [0., 0., 1., 1.]
