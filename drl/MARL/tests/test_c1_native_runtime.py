from types import SimpleNamespace
from dataclasses import replace
import json
import torch

from drl.MARL.c1_native_runtime import NativeC1RewardRuntime
from drl.MARL.c1_runner import make_course
from drl.MARL.smores_runtime_binding import SMORESMARLProposalOutcome
from drl.MARL.c1_training_telemetry import TrainingTelemetry
from drl.MARL.tests.test_c1_structure import frame


def env_fixture():
    def forbidden(*args,**kwargs):raise AssertionError('Expert/target morphology entered C1 evaluator')
    return SimpleNamespace(num_envs=2,domain_by_env=('navigation','navigation'),
        courses=(make_course('navigation',7017,'train'),make_course('navigation',7017,'train')),
        seed_by_env=(7017,7017),generation=[1,1],max_episode_length=1800,step_dt=1/30,
        cfg=SimpleNamespace(initialization_mode='loose_modules',possible_agents=tuple(f'm{i}' for i in range(8)),reward_config_path='logs/appo_smoke/calibration_validated_train/reward_config.initial.json'),
        _outcomes=[SMORESMARLProposalOutcome(True,'EMPTY','none'),SMORESMARLProposalOutcome(True,'EMPTY','none')],
        training_telemetry=TrainingTelemetry(),_load_target_graph=forbidden,_ensure_expert_planners=forbidden)


def test_generation_replaces_only_selected_evaluator_and_logs_censored_episode(tmp_path):
    env=env_fixture();runtime=NativeC1RewardRuntime(evidence_directory=tmp_path)
    runtime.evaluate(env,0,frame(edges=0));runtime.evaluate(env,1,frame(edges=0))
    sibling=runtime.evaluators[1][1]
    env.generation[0]+=1
    runtime.evaluate(env,0,frame(1/30,edges=0))
    assert runtime.evaluators[1][1] is sibling
    assert runtime.episode_records[0]['success_failure_reason']=='reset_before_termination'
    assert runtime.episode_records[0]['generation']==1
    assert len((tmp_path/'transitions.jsonl').read_text().splitlines())==3


def test_native_partial_drop_penalties_reach_reward_and_telemetry(tmp_path):
    env=env_fixture();runtime=NativeC1RewardRuntime(evidence_directory=tmp_path)
    env._outcomes[0]=SMORESMARLProposalOutcome(True,'ACCEPTED','partial',('m0',),conflict_owner_ids=('m1',),invalid_owner_ids=('m2',))
    result=runtime.evaluate(env,0,frame(edges=0))
    assert result.team_reward==sum(result.owner_rewards.values())/8
    assert result.reward_breakdown['penalty_conflict']<0 and result.reward_breakdown['penalty_invalid']<0
    episode=next(iter(env.training_telemetry.open_episodes.values()))
    assert episode['reward_breakdown']['penalty_conflict']==result.reward_breakdown['penalty_conflict']


def test_terminal_nan_state_is_serializable_evidence_and_never_success(tmp_path):
    env=env_fixture();runtime=NativeC1RewardRuntime(evidence_directory=tmp_path)
    f=frame();f=replace(f,modules=(replace(f.modules[0],position=(float('nan'),0,.04)),*f.modules[1:]))
    result=runtime.evaluate(env,0,f)
    assert result.terminated and not result.success
    record=json.loads((tmp_path/'episodes.jsonl').read_text())
    assert record['success_failure_reason']=='nonfinite_physical_state'
    assert record['sanitized_nonfinite_fields']
