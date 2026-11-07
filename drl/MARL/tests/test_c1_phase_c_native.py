from types import SimpleNamespace as NS
from dataclasses import replace
import pytest

from drl.MARL.c1_action_events import C1ActionEvents
from drl.MARL.c1_native_runtime import NativeC1RewardRuntime
from drl.MARL.tests.test_c1_native_runtime import env_fixture
from drl.MARL.tests.test_c1_structure import frame


def test_resource_metadata_preserves_actual_decoded_joint_target():
    candidate=NS(choice=None,candidate_id='resource|tilt_target|m0',candidate_index=0,
        owner_ids=('m0',),claims=(),raw_presence_logit=6.,
        decoded=NS(command=dict(command='set_tilt',module_ids=('m0',),parameters={'angle_rad':0.})))
    row=C1ActionEvents()._candidate(candidate)
    assert row['resolved_primitive']=='set_tilt'
    assert row['resolved_module_ids']==('m0',)
    assert row['resolved_parameters']=={'angle_rad':0.}


def test_native_real_decoded_metadata_reaches_owner_reward_without_optional_logging():
    env=env_fixture();env.c1_action_events=C1ActionEvents(None,masked_detail_limit=0)
    runtime=NativeC1RewardRuntime(settings_path='configs/marl/c1_discovery_phase_c_v1.json')
    metadata=dict(primitive='tilt_target',module_ids=('m0',),owner_module_id='m0',arguments=(),
        resolved_primitive='set_tilt',resolved_module_ids=('m0',),resolved_parameters={'angle_rad':0.})
    env.c1_action_events.goals[(0,1,'first')]=metadata
    runtime.evaluate(env,0,frame(edges=0))
    env.c1_action_events.goals[(0,1,'second')]=metadata
    r=runtime.evaluate(env,0,frame(1/30,edges=0))
    assert r.owner_rewards['m0']-r.owner_rewards['m1']==pytest.approx(-.1)


def test_native_sleeping_rows_do_not_count_as_policy_opportunities():
    env=env_fixture();env.runtime_context_provider=NS(_captured=(NS(rollout_active=False),)*2)
    runtime=NativeC1RewardRuntime(settings_path='configs/marl/c1_guided_phase_c_v1.json')
    for i in range(35):r=runtime.evaluate(env,0,frame(float(i),edges=0))
    assert not r.terminated and r.stagnation['decision_opportunities']==0


@pytest.mark.parametrize('mode',['guided','discovery'])
def test_train_guard_accepts_explicit_phase_c_contract_but_does_not_launch(mode,tmp_path):
    from drl.MARL.isaaclab_c1_runner import build_parser, configure_run_arguments
    args=build_parser().parse_args(['--train','--c1-staged-training','--initialization-mode','loose_modules',
        '--structural-scheduler','event','--c1-probe-config',f'configs/marl/c1_{mode}_phase_c_v1.json',
        '--episode-seconds','120','--num-envs','4','--total-transitions','32','--reward-config',
        'logs/appo_smoke/calibration_validated_train/reward_config.initial.json','--output',str(tmp_path/'new')])
    configure_run_arguments(args)
    assert args.episode_seconds==120


def test_native_selected_reset_cancels_active_option_before_teleport_preserving_sibling(monkeypatch):
    import torch
    from drl.MARL.isaaclab_c1_tasks import NativeC1TaskHooks
    from drl.MARL.c1_initialization import PolicyControlledHandoff
    calls=[]
    class Base:
        def _reset_idx(self,ids):pass
    class Env(NativeC1TaskHooks,Base):pass
    env=Env();owners=tuple(f'smores_{i:02d}' for i in range(1,9))
    env.cfg=NS(initialization_mode='loose_modules',possible_agents=owners)
    env.num_envs=2;env.device='cpu';env.physics_dt=1/240;env._sim_step_counter=240
    env.handoffs=[PolicyControlledHandoff(),PolicyControlledHandoff()]
    for h in env.handoffs:h.activate_policy()
    env.generation=[1,1];env.domain_by_env=('navigation',)*2;env.morphology_by_env=(None,)*2;env.seed_by_env=(7033,7033)
    env.docking_managers=[NS(connections=()) for _ in range(2)]
    env.scene=NS(env_origins=torch.zeros(2,3));env.plunger_paths={}
    env._spawn_layout={o:(0.,0.,.032,0.) for o in owners}
    env.state_readers_by_env=[{o:NS(_articulation=NS(reset_pose=lambda *a,i=i:calls.append(('pose',i,tuple(env.generation))))) for o in owners} for i in range(2)]
    backends=[]
    for i in range(2):
        executor=NS(active_goals=(NS(goal_id=f'option{i}'),))
        executor.cancel=lambda gid,t,i=i:calls.append(('cancel',i,tuple(env.generation)))
        backends.append(dict(router=NS(_configured={},reset_free_modules=lambda ids:None),primitive_executor=executor))
    env.physical_expert_backends=tuple(backends)
    env._ensure_native_actuation_backends=lambda **kw:env.physical_expert_backends
    def forbidden(*a,**k):raise AssertionError('Expert fallback')
    env._ensure_physical_expert_backends=env._start_expert_assembly=env._ensure_expert_planners=env._load_target_graph=forbidden
    env._resample_course=lambda i:None;env.native_transforms=NS(invalidate=lambda:None)
    env.native_dispatcher=NS(clear=lambda i:None);env.stagnation=[NS(reset=lambda:None) for _ in range(2)]
    env.sim=NS(stage=object());env.module_roots_by_env=({o:'/e0/'+o for o in owners},{o:'/e1/'+o for o in owners})
    env.intended_connections=[frozenset(),frozenset()];env.end_effector_ids=[None,None]
    env._prepared_cache={};env.assembly_runtimes=[None,None];env.assembly_verified=[False,False]
    env.marl_steps=[0,0];env.assembly_started_wall=[None,None];env._last_evaluations=[None,None];env.staged=[None,None]
    monkeypatch.setattr('drl.MARL.native_episode_reset.clear_authored_external_wrenches',lambda *a:None)
    monkeypatch.setattr('drl.MARL.native_episode_reset.isolate_course_colliders',lambda *a:{})
    monkeypatch.setattr('drl.MARL.native_episode_reset.flush_pending_physics',lambda:None)
    env._reset_idx(torch.tensor([0]))
    assert calls[0]==('cancel',0,(1,1))
    assert len([c for c in calls if c[0]=='pose'])==8
    assert all(c[1]==0 for c in calls)
    assert env.generation==[2,1]
    assert env.physical_expert_backends[1] is backends[1]
    assert backends[1]['primitive_executor'].active_goals[0].goal_id=='option1'
