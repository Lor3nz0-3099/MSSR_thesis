from types import SimpleNamespace

import pytest
import torch

from drl.MARL.bc_v4_runtime_observation import BCV4RuntimeObservation
from drl.MARL.runtime_context import DirectMARLRuntimeContextProvider
from drl.MARL.runtime_handoff import DirectMARLHandoff


def active_handoff():
    handoff = DirectMARLHandoff()
    handoff.begin_expert('rc_car8')
    handoff.observe_expert_terminal(True)
    handoff.observe_topology('rc_car8')
    return handoff


class NativeEnv:
    num_envs = 2

    def __init__(self):
        self.stamp = 1.0
        self.staged = []

    def cached_raw_snapshot(self, index):
        return {'stamp': self.stamp, 'env': index}

    def stage_decoded(self, env_index, commands, outcome):
        self.staged.append((env_index, commands, outcome))


class ObservationBridge:
    normalizers = object()

    def __init__(self):
        self.prepared = []

    def prepare_snapshot(self, raw, *, task_context):
        index = raw['env']
        actor = {'visible': torch.tensor([index, raw['stamp']])}
        critic = {**actor, 'privileged': torch.tensor([20.0 + index])}
        routing = {'owner': index, 'stamp': raw['stamp']}
        decoder = {'inputs': actor, 'routing': [routing], 'exact_sidecar': object()}
        prepared = BCV4RuntimeObservation(SimpleNamespace(stamp=raw['stamp']), actor, routing, decoder, critic)
        self.prepared.append(prepared)
        return prepared


def make_bridge(handoffs=None):
    from drl.MARL.native_policy_bridge import build_native_policy_bridge

    env, observations, decoded = NativeEnv(), ObservationBridge(), []

    class Binding:
        def decode_marl_action(self, **kwargs):
            decoded.append(kwargs)
            return (SimpleNamespace(command={'command': 'wheel_velocity', 'module_ids': ['m0'], 'parameters': {}}),)

    result = build_native_policy_bridge(
        env, observation_bridge=observations, task_context_provider=lambda index: index,
        binding=Binding(), handoffs=handoffs or [active_handoff(), active_handoff()],
        hard_mask_builder=lambda actor, routing: torch.tensor([routing['owner'] == 1]),
        placeholder_action=lambda index, context: {'m0': torch.zeros(1)},
    )
    return result, env, observations, decoded


def test_two_envs_use_prepared_actor_and_privileged_critic_with_exact_decoder_sidecars():
    bridge, env, observations, decoded = make_bridge()
    assert isinstance(bridge.provider, DirectMARLRuntimeContextProvider)
    contexts = bridge.provider.current_action_contexts()
    assert len(contexts) == 2
    for index, context in enumerate(contexts):
        assert 'privileged' not in context.actor_graph
        assert context.critic_graph['privileged'].item() == 20 + index
        assert context.routing_snapshot['owner'] == index
        assert context.hard_mask.item() == bool(index)
    decision = SimpleNamespace(action=object(), action_set_prediction_template=object())
    result = bridge.provider.encode_env_actions([decision, decision])
    assert result['m0'].shape == (2, 1)
    assert len(env.staged) == 2
    for index, row in enumerate(decoded):
        assert row['batch'] is observations.prepared[index].decoder_batch
        assert row['routing'] == contexts[index].routing_snapshot
        assert env.staged[index][2].accepted
    env.stamp = 2.0
    critics = bridge.provider.current_critic_graphs()
    assert len(observations.prepared) == 4
    assert critics[1]['visible'][1].item() == 2
    assert bridge.provider.current_action_contexts()[1].critic_graph is critics[1]
    assert len(observations.prepared) == 4


def test_assembly_rows_are_neutral_and_reset_invalidates_same_stamp_preprocessing():
    bridge, env, observations, decoded = make_bridge([active_handoff(), DirectMARLHandoff()])
    decision = SimpleNamespace(action=object(), action_set_prediction_template=object())
    bridge.provider.encode_env_actions([decision, None])
    assert len(decoded) == len(env.staged) == 1
    bridge.provider.reset()
    bridge.provider.current_action_contexts()
    assert len(observations.prepared) == 4


def test_missing_privileged_graph_fails_closed():
    bridge, env, observations, decoded = make_bridge()
    original = observations.prepare_snapshot

    def actor_only(*args, **kwargs):
        prepared = original(*args, **kwargs)
        return BCV4RuntimeObservation(prepared.physical_state, prepared.actor_graph, prepared.routing_snapshot, prepared.decoder_batch)

    observations.prepare_snapshot = actor_only
    with pytest.raises(ValueError, match='privileged'):
        bridge.provider.current_action_contexts()


@pytest.mark.parametrize('kind', ['invalid', 'conflict'])
def test_sampled_local_rejections_keep_owner_and_reach_env_step(kind):
    from dataclasses import dataclass

    @dataclass
    class Action:
        presence: torch.Tensor

    bridge, env, observations, decoded = make_bridge([active_handoff(), DirectMARLHandoff()])
    context = bridge.provider.current_action_contexts()[0]
    candidate = SimpleNamespace(resource_ref=SimpleNamespace(
        module=SimpleNamespace(module_id='m0'), resource_id='wheels'))
    context.routing_snapshot['resource_candidates'] = [candidate, candidate]
    attempts = []

    def reject(**kwargs):
        attempts.append(kwargs)
        if kind == 'conflict':
            raise ValueError('Conflicting physical resource claims')
        raise ValueError('Physical payload outside limits')

    bridge.decision_bridge.binding.binding.decode_marl_action = reject
    decision = SimpleNamespace(action=Action(torch.ones(2)), action_set_prediction_template={})
    result = bridge.provider.encode_env_actions([decision, None])
    assert result['m0'].shape == (2, 1)
    assert len(env.staged) == 1
    index, commands, outcome = env.staged[0]
    assert index == 0 and commands == ()
    assert not outcome.accepted and outcome.owner_ids == ('m0',)
    assert outcome.code == ('INVALID_GOAL' if kind == 'invalid' else 'RESOURCE_BUSY')
    assert len(attempts) == (3 if kind == 'invalid' else 1)


def test_unexpected_decode_error_propagates_and_clears_staging():
    bridge, env, observations, decoded = make_bridge()
    cleared = []
    env.clear_staged = cleared.append

    def fail(**kwargs):
        raise ValueError('broken decoder schema')

    bridge.decision_bridge.binding.binding.decode_marl_action = fail
    decision = SimpleNamespace(action=object(), action_set_prediction_template={})
    with pytest.raises(ValueError, match='broken decoder schema'):
        bridge.provider.encode_env_actions([decision, decision])
    assert cleared == [0]
    assert not env.staged


def test_native_phase_metadata_tracks_handoffs_and_altruistic_provider_delegates():
    bridge, env, observations, decoded = make_bridge([active_handoff(), DirectMARLHandoff()])
    contexts = bridge.provider.current_action_contexts()
    assert [context.rollout_active for context in contexts] == [True, False]
    payload = object()
    env.current_altruistic_rollout_batch = lambda: payload
    assert bridge.provider.current_altruistic_rollout_batch() is payload


def test_post_step_critic_uses_terminal_state_before_auto_reset_current_state():
    bridge, env, observations, decoded = make_bridge()
    bridge.provider.current_action_contexts()
    decision = SimpleNamespace(action=object(), action_set_prediction_template={})
    bridge.provider.encode_env_actions([decision, decision])
    terminal = [observations.prepare_snapshot({'stamp': 2.0, 'env': i}, task_context=i)
                for i in range(2)]
    calls = []

    def post_step(index):
        calls.append(index)
        return terminal[index]

    bridge.provider.post_step_observation_provider = post_step
    # Auto-reset can use the same clock stamp as a previous episode.
    env.stamp = 1.0
    final_critics = bridge.provider.current_critic_graphs()
    assert final_critics[0]['visible'][1].item() == 2
    assert bridge.provider.current_critic_graphs() is final_critics
    assert calls == [0, 1]
    reset_contexts = bridge.provider.current_action_contexts()
    assert reset_contexts[0].critic_graph['visible'][1].item() == 1
    assert reset_contexts[0].critic_graph is not final_critics[0]
    assert bridge.provider.current_critic_graphs()[0] is reset_contexts[0].critic_graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA tensor placement')
def test_native_policy_transfers_actor_critic_mask_and_decoder_tensors_to_env_device():
    bridge, env, observations, decoded = make_bridge()
    env.device = torch.device('cuda:0')
    contexts = bridge.provider.current_action_contexts()
    layer = torch.nn.Linear(2, 1).to(env.device)
    for context in contexts:
        assert layer(context.actor_graph['visible'].float()).device == env.device
        assert context.critic_graph['privileged'].device == env.device
        assert context.hard_mask.device == env.device
    decision = SimpleNamespace(action=object(), action_set_prediction_template={})
    bridge.provider.encode_env_actions([decision, decision])
    for index, kwargs in enumerate(decoded):
        assert kwargs['batch']['inputs']['visible'].device == env.device
        assert kwargs['batch']['routing'] is observations.prepared[index].decoder_batch['routing']
        assert kwargs['batch']['exact_sidecar'] is observations.prepared[index].decoder_batch['exact_sidecar']
        assert observations.prepared[index].actor_graph['visible'].device.type == 'cpu'
    terminal = [observations.prepare_snapshot({'stamp': 2., 'env': i}, task_context=i) for i in range(2)]
    bridge.provider.post_step_observation_provider = terminal.__getitem__
    assert all(graph['privileged'].device == env.device for graph in bridge.provider.current_critic_graphs())


@pytest.mark.parametrize('kind',['conflict','invalid','unavailable'])
def test_sampled_structural_rejections_keep_policy_initiator_and_continue(kind):
    from dataclasses import dataclass
    @dataclass
    class Action:
        presence:torch.Tensor
    bridge,env,_,_=make_bridge([active_handoff(),DirectMARLHandoff()])
    context=bridge.provider.current_action_contexts()[0]
    context.routing_snapshot.update(resource_candidates=[],structural_bindings=[
        (None,SimpleNamespace(initiator=SimpleNamespace(module_id='m1')))])
    def reject(**kwargs):
        raise ValueError({'conflict':'Conflicting physical resource claims',
            'invalid':'Physical payload outside limits','unavailable':'Physical resource unavailable'}[kind])
    bridge.decision_bridge.binding.binding.decode_marl_action=reject
    decision=SimpleNamespace(action=Action(torch.ones(1)),action_set_prediction_template={})
    result=bridge.provider.encode_env_actions([decision,None])
    assert result['m0'].shape==(2,1)
    index,commands,outcome=env.staged[0]
    assert index==0 and commands==() and not outcome.accepted
    assert outcome.owner_ids==('m1',)
    assert outcome.code==('INVALID_GOAL' if kind=='invalid' else 'RESOURCE_BUSY')


@pytest.mark.parametrize('invalid',['duration','scope'])
def test_real_adapter_invalid_gravity_choice_becomes_policy_rejection(invalid):
    from dataclasses import dataclass
    from pathlib import Path
    from drl.IL.bc_v1 import bc_v1_contract as c
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    spec=c.load_robot_spec(Path(__file__).resolve().parents[3]/'configs/bc_v1/smores_ep.robot.json')
    adapter=SMORESAdapter(spec);m0,m1=c.ModuleRef('m0'),c.ModuleRef('m1')
    choice=c.ActionChoice('gravity_settle',m0,participants=(m0,m1) if invalid=='duration' else (m0,),
        parameters=(c.ParameterValue('duration_s',-1. if invalid=='duration' else 1.),
                    c.ParameterValue('passive_module_ids',(m0,m1))))
    @dataclass
    class Action:
        presence:torch.Tensor
    bridge,env,_,_=make_bridge([active_handoff(),DirectMARLHandoff()])
    context=bridge.provider.current_action_contexts()[0]
    context.routing_snapshot.update(resource_candidates=[],structural_bindings=[(None,SimpleNamespace(initiator=m0))])
    bridge.decision_bridge.binding.binding.decode_marl_action=lambda **kw:adapter.decode_action(choice)
    decision=SimpleNamespace(action=Action(torch.ones(1)),action_set_prediction_template={})
    bridge.provider.encode_env_actions([decision,None])
    index,commands,outcome=env.staged[0]
    assert index==0 and commands==() and not outcome.accepted
    assert outcome.code=='INVALID_GOAL' and outcome.owner_ids==('m0',)
