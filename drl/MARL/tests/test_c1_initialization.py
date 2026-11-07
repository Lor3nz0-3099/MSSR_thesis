"""Initialization authority contracts; no simulator or Expert is launched."""
from types import SimpleNamespace

import pytest
import torch


def test_modes_are_explicit_and_c1_cannot_choose_expert_or_assembled():
    from drl.MARL.c1_initialization import InitializationMode, resolve_initialization
    assert {m.value for m in InitializationMode} == {'loose_modules','assembled_state','expert_validation'}
    assert resolve_initialization('c1', None) is InitializationMode.LOOSE_MODULES
    assert resolve_initialization('catalog', None) is InitializationMode.EXPERT_VALIDATION
    for mode in ('assembled_state','expert_validation'):
        with pytest.raises(ValueError, match='loose_modules'):
            resolve_initialization('c1', mode)


def test_policy_authority_never_claims_expert_or_target_topology():
    from drl.MARL.c1_initialization import PolicyControlledHandoff
    from drl.MARL.runtime_handoff import DirectMARLHandoff, RuntimeAuthority
    handoff=PolicyControlledHandoff()
    assert isinstance(handoff,DirectMARLHandoff)  # retain certified provider ABI
    handoff.activate_policy()
    assert handoff.marl_active and handoff.authority is RuntimeAuthority.MARL
    assert handoff.target_morphology is None and handoff.last_expert_success is None
    with pytest.raises(RuntimeError,match='Expert'):
        handoff.begin_expert('snake8')
    with pytest.raises(RuntimeError,match='topology'):
        handoff.observe_topology('snake8')
    handoff.finish_episode()
    assert not handoff.marl_active
    handoff.reset();handoff.activate_policy()
    assert handoff.marl_active and handoff.target_morphology is None


def test_structural_candidates_are_not_blanket_masked_in_policy_mode():
    from drl.MARL.c1_initialization import native_action_mask
    route=dict(resource_candidates=(1,2),structural_bindings=(3,4,5))
    assert native_action_mask(route,'loose_modules',device='cpu').tolist()==[False]*5
    assert native_action_mask(route,'expert_validation',device='cpu').tolist()==[False,False,True,True,True]


def test_new_c1_cli_selects_loose_and_only_task5_smoke_is_permitted():
    from drl.MARL.isaaclab_c1_runner import build_parser,configure_run_arguments
    args=configure_run_arguments(build_parser().parse_args(['--curriculum','c1','--reward-config','unit.json']))
    assert args.initialization_mode=='loose_modules'
    assert args.curriculum=='c1'
    for flag in ('--train','--benchmark','--evaluate'):
        parsed=build_parser().parse_args(['--curriculum','c1',flag,'--num-envs','4',
            '--total-transitions','128','--output','unused','--reward-config','unit.json'])
        with pytest.raises(ValueError,match='Task 5'):
            configure_run_arguments(parsed)
    smoke=configure_run_arguments(build_parser().parse_args(['--curriculum','c1',
        '--smoke-train','--num-envs','4','--total-transitions','128',
        '--output','unused','--reward-config','unit.json']))
    assert smoke.smoke_train and smoke.initialization_mode=='loose_modules'


def test_loose_reset_and_control_never_start_expert(monkeypatch):
    from drl.MARL.c1_initialization import PolicyControlledHandoff
    from drl.MARL.isaaclab_c1_tasks import NativeC1TaskHooks
    calls=[]
    class Base:
        def _reset_idx(self,ids):calls.append('native_reset')
    class Env(NativeC1TaskHooks,Base):
        pass
    owners=tuple(f'smores_{i:02d}' for i in range(1,9))
    env=Env();env.cfg=SimpleNamespace(initialization_mode='loose_modules',possible_agents=owners)
    env.num_envs=2;env.device='cpu';env.physics_dt=1/240;env._sim_step_counter=4
    env.handoffs=[PolicyControlledHandoff(),PolicyControlledHandoff()]
    for h in env.handoffs:h.activate_policy()
    env.generation=[1,1];env.domain_by_env=('navigation','navigation');env.morphology_by_env=(None,None)
    env.seed_by_env=(7000,7001)
    env.docking_managers=[SimpleNamespace(connections=()) for _ in range(2)]
    env.scene=SimpleNamespace(env_origins=torch.zeros(2,3));env.plunger_paths={}
    env._spawn_layout={owner:(0.,0.,.032,0.) for owner in owners}
    access=SimpleNamespace(reset_pose=lambda *a:calls.append('pose'))
    env.state_readers_by_env=[{o:SimpleNamespace(_articulation=access) for o in owners} for _ in range(2)]
    backend=dict(router=SimpleNamespace(_configured={},reset_free_modules=lambda ids:None),primitive_executor=SimpleNamespace(active_goals=()))
    env.physical_expert_backends=(backend,backend)
    env._ensure_native_actuation_backends=lambda **kw:env.physical_expert_backends
    def forbidden(*a,**k):raise AssertionError('Expert called by loose policy path')
    env._ensure_physical_expert_backends=forbidden;env._start_expert_assembly=forbidden
    env._ensure_expert_planners=forbidden;env._load_target_graph=forbidden
    env._resample_course=lambda i:None
    env.native_transforms=SimpleNamespace(invalidate=lambda:None)
    env.native_dispatcher=SimpleNamespace(clear=lambda i:None)
    env.native_actuation=SimpleNamespace(step=lambda i:SimpleNamespace(proposal_outcome=None))
    env.stagnation=[SimpleNamespace(reset=lambda:None) for _ in range(2)]
    env.sim=SimpleNamespace(stage=object());env.module_roots_by_env=({o:'/e0/'+o for o in owners},{o:'/e1/'+o for o in owners})
    env.intended_connections=[frozenset(),frozenset()];env.end_effector_ids=[None,None]
    env._prepared_cache={};env.assembly_runtimes=[None,None];env.assembly_verified=[False,False]
    env.marl_steps=[0,0];env.assembly_started_wall=[None,None];env._last_evaluations=[None,None];env.staged=[None,None]
    monkeypatch.setattr('drl.MARL.native_episode_reset.clear_authored_external_wrenches',lambda *a:None)
    monkeypatch.setattr('drl.MARL.native_episode_reset.isolate_course_colliders',lambda *a:{})
    monkeypatch.setattr('drl.MARL.native_episode_reset.flush_pending_physics',lambda:calls.append('flush'))
    monkeypatch.setattr('drl.MARL.c1_native_observation.record_physics_evidence',lambda *a:calls.append('measurement'))
    env._reset_idx(torch.tensor([0]))
    assert env.handoffs[0].marl_active and env.handoffs[0].target_morphology is None
    assert env.generation==[2,1] and env.assembly_runtimes==[None,None]
    env._apply_action()  # traps any assembly fallback as well
    assert calls.count('pose')==8
    assert calls.index('flush')<calls.index('pose')


@pytest.mark.parametrize('flag',['--train','--benchmark','--evaluate','--controlled-update'])
def test_explicit_loose_mode_cannot_bypass_task5_gate(flag):
    from drl.MARL.isaaclab_c1_runner import build_parser,configure_run_arguments
    args=build_parser().parse_args(['--curriculum','catalog','--initialization-mode','loose_modules',
        flag,'--reward-config','unit.json','--num-envs','4','--total-transitions','128','--output','unused'])
    with pytest.raises(ValueError,match='Task 5|legacy Expert'):
        configure_run_arguments(args)
