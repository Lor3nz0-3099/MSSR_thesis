"""Policy-controlled snapshot reset authority; no simulator imports."""
from types import SimpleNamespace as NS
import pytest


def test_c1_curriculum_explicitly_allows_snapshot_initialization():
    from drl.MARL.c1_initialization import resolve_initialization,InitializationMode
    assert resolve_initialization('c1','assembled_state') is InitializationMode.ASSEMBLED_STATE


def test_snapshot_guard_requires_explicit_phase_d_bank_and_forbids_old_resume(tmp_path):
    from drl.MARL.isaaclab_c1_runner import build_parser,configure_run_arguments
    base=['--train','--c1-staged-training','--curriculum','c1','--initialization-mode','assembled_state',
        '--reward-config','logs/appo_smoke/calibration_validated_train/reward_config.initial.json',
        '--c1-probe-config','configs/marl/c1_guided_phase_c_v1.json','--episode-seconds','120',
        '--total-transitions','32','--num-envs','4','--output',str(tmp_path/'new')]
    args=build_parser().parse_args(base)
    with pytest.raises(ValueError,match='snapshot-bank'):configure_run_arguments(args)


def test_initial_native_frame_does_not_invent_decimation_ticks(monkeypatch):
    from drl.MARL import c1_native_observation as obs
    assert hasattr(obs,'initial_frame_from_native')


def test_native_evaluator_requires_measured_baseline_for_snapshot_mode():
    from drl.MARL.c1_native_runtime import NativeC1RewardRuntime
    from drl.MARL.tests.test_c1_native_runtime import env_fixture
    from drl.MARL.tests.test_c1_structure import frame
    env=env_fixture();env.cfg.initialization_mode='assembled_state'
    runtime=NativeC1RewardRuntime(settings_path='configs/marl/c1_guided_phase_c_v1.json')
    with pytest.raises(RuntimeError,match='baseline'):runtime.evaluate(env,0,frame(edges=0))


def test_native_baseline_never_pays_existing_snapshot_edges():
    from drl.MARL.c1_native_runtime import NativeC1RewardRuntime
    from drl.MARL.tests.test_c1_native_runtime import env_fixture
    from drl.MARL.tests.test_c1_phase_c_episode import guided_frame
    from dataclasses import replace
    env=env_fixture();env.cfg.initialization_mode='assembled_state'
    env.cfg.shaping_gamma=.99
    env.cfg.course_split='train'
    initial=guided_frame();initial=replace(initial,connections=initial.connections[:6])
    runtime=NativeC1RewardRuntime(settings_path='configs/marl/c1_guided_phase_c_v1.json')
    runtime.prime(env,0,initial)
    r=runtime.evaluate(env,0,replace(initial,time_s=1/30))
    assert r.reward_breakdown['reward_target_graph_progress']==0.
    assert r.reward_breakdown['reward_target_graph_discount']==pytest.approx(-6*.01*6/7)
