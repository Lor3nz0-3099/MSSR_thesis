"""Manual performance driver checks. No simulator is imported or launched."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / 'docs/experiments/c1_5b_perf_20261107_petrlqrb/benchmark_native.py'


def _driver():
    assert DRIVER.is_file(), 'manual performance driver has not been implemented'
    spec = importlib.util.spec_from_file_location('offline_c1_benchmark_driver', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('variant', ['baseline', 'optimized'])
def test_validate_only_preserves_training_contract_without_creating_output(tmp_path, capsys, variant):
    driver = _driver()
    output = tmp_path / variant
    before_modules = set(sys.modules)
    assert driver.main(['--variant', variant, '--output', str(output), '--validate-only',
                        '--device', 'cuda:0']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert payload['variant'] == variant
    assert payload['fresh_bc_critic_warmup'] is True
    args = payload['native_arguments']
    reference = json.loads((driver.EXPERIMENT/'baseline_run_arguments.json').read_text())
    for name in ('train', 'c1_staged_training', 'curriculum', 'initialization_mode',
                 'train_sampler', 'snapshot_bank', 'snapshot_stage_weights', 'reward_config',
                 'c1_probe_config', 'domains', 'num_envs', 'rollout_steps', 'learning_epochs',
                 'mini_batches', 'structural_scheduler', 'global_seed', 'episode_seconds',
                 'gamma', 'alpha', 'altruism_radius_m', 'early_stopping_patience',
                 'checkpoint_every_transitions'):
        assert args[name] == reference[name], name
    assert args['train'] and args['c1_staged_training']
    assert not args['benchmark'] and not args['c1_integration_probe']
    assert args['gamma'] == .999
    assert args['num_envs'] == 8 and args['rollout_steps'] == 32
    assert args['learning_epochs'] == 2 and args['mini_batches'] == 8
    assert args['total_transitions'] == 1 and args['max_vector_steps'] == 1000
    assert args['max_wall_seconds'] == 1800. and args['global_seed'] == 3103
    assert args['initialization_mode'] == 'assembled_state'
    assert args['device'] == 'cuda:0'
    assert args['visualizer'] is None and args['visualizer_explicit'] and args['livestream'] == 0
    assert args['headless'] and '--headless' in payload['native_argv']
    assert not args['enable_cameras'] and not args['xr']
    assert payload['resolved_rendering'] == dict(headless=True, visualizers=[],
        livestream=0, cameras=False, xr=False, offscreen_render=False, viewport=False)
    assert Path(args['checkpoint']).is_absolute()
    assert payload['physics']['current'] == payload['physics']['baseline']
    assert payload['physics']['bank_verified'] is True
    assert payload['source_provenance']['snapshot_hashes_verified'] > 0
    import hashlib
    source_root = driver.BEFORE if variant == 'baseline' else driver.ROOT
    selected_hashes = payload['source_provenance']['selected_performance_source_sha256']
    assert selected_hashes['drl/MARL/mappo.py'] == hashlib.sha256(
        (source_root/'drl/MARL/mappo.py').read_bytes()).hexdigest()
    assert ('drl/MARL/c1_incremental_diagnostics.py' in selected_hashes) == (variant == 'optimized')
    assert not any(name.startswith(('isaaclab', 'isaacsim', 'omni.'))
                   for name in set(sys.modules) - before_modules)


def test_existing_output_is_rejected_even_for_validate_only(tmp_path):
    driver = _driver()
    with pytest.raises(ValueError, match='already exists'):
        driver.main(['--variant', 'baseline', '--output', str(tmp_path), '--validate-only'])


def test_materialized_baseline_does_not_leak_new_python_or_recurse_outputs(tmp_path):
    driver = _driver()
    repo, before = tmp_path/'repo', tmp_path/'saved'
    (before/'drl/MARL').mkdir(parents=True)
    (before/'configs').mkdir()
    (before/'drl/MARL/model.py').write_text('VERSION = "before"\n')
    (before/'configs/options.json').write_text('{"before": true}')
    (repo/'drl/MARL').mkdir(parents=True)
    (repo/'configs').mkdir()
    (repo/'assets').mkdir()
    (repo/'docs/experiment').mkdir(parents=True)
    (repo/'logs').mkdir()
    (repo/'drl/MARL/model.py').write_text('VERSION = "optimized"\n')
    (repo/'drl/MARL/new_optimized_helper.py').write_text('raise RuntimeError("leak")\n')
    (repo/'drl/MARL/layout.bin').write_bytes(b'asset')
    (repo/'configs/options.json').write_text('{"optimized": true}')
    destination = repo/'docs/experiment/output/_baseline_source'
    driver.materialize_baseline(repo, before, destination)
    assert (destination/'drl/MARL/model.py').read_text() == 'VERSION = "before"\n'
    assert not (destination/'drl/MARL/model.py').is_symlink()
    assert not (destination/'drl/MARL/new_optimized_helper.py').exists()
    assert (destination/'configs/options.json').read_text() == '{"before": true}'
    assert (destination/'drl/MARL/layout.bin').is_symlink()
    assert (destination/'assets').is_symlink() and (destination/'logs').is_symlink()
    assert not (destination/'docs').exists()
    assert not list(destination.rglob('__pycache__'))


def test_installed_launcher_ast_rejects_invalid_visualizer(tmp_path):
    driver = _driver()
    argv = driver.build_native_argv(tmp_path/'new', driver.ROOT, 'cuda:0')
    argv[argv.index('--visualizer') + 1] = 'unsupported_backend'
    with pytest.raises(SystemExit):
        driver._parse_native(driver.ROOT, argv)


def test_runner_configuration_rejects_assembled_state_integration_probe(tmp_path):
    driver = _driver()
    argv = driver.build_native_argv(tmp_path/'new', driver.ROOT, 'cuda:0')
    argv.append('--c1-integration-probe')
    with pytest.raises(ValueError, match='Integration probe'):
        driver._parse_native(driver.ROOT, argv)


def test_launcher_arguments_are_forwarded_and_restored_even_on_failure():
    driver = _driver()
    original = sys.argv
    with pytest.raises(RuntimeError, match='fixture stop'):
        with driver.runner_process_arguments(['--device', 'cuda:0', '--visualizer', 'kit']):
            assert sys.argv == ['isaaclab_c1_runner.py', '--device', 'cuda:0', '--visualizer', 'kit']
            raise RuntimeError('fixture stop')
    assert sys.argv is original


def test_headless_environment_overrides_ambient_cameras_and_xr_then_restores(monkeypatch):
    import os
    driver = _driver()
    for name in ('HEADLESS', 'ENABLE_CAMERAS', 'XR', 'LIVESTREAM'):
        monkeypatch.setenv(name, '1')
    before = dict(os.environ)
    with pytest.raises(RuntimeError), driver.headless_environment():
        assert os.environ['HEADLESS'] == '1'
        assert all(os.environ[name] == '0' for name in ('ENABLE_CAMERAS', 'XR', 'LIVESTREAM'))
        raise RuntimeError('restore on error')
    assert dict(os.environ) == before


def test_runtime_headless_guard_disables_render_without_changing_physical_configuration():
    from types import SimpleNamespace
    driver = _driver()
    env = SimpleNamespace(render_enabled=True, sim=SimpleNamespace(has_gui=False, is_rendering=False),
        cfg=SimpleNamespace(decimation=8, sim=SimpleNamespace(dt=1/240)))
    result = driver.configure_headless_runtime(env)
    assert result == dict(has_gui=False, is_rendering=False, render_enabled=False)
    assert not env.render_enabled
    assert env.cfg.decimation == 8 and env.cfg.sim.dt == 1/240
    env.sim.is_rendering = True
    with pytest.raises(RuntimeError, match='render'):
        driver.configure_headless_runtime(env)


def test_decision_log_preserves_order_and_closes_on_exception(tmp_path):
    driver = _driver()
    path = tmp_path/'decision_reasons.jsonl'
    with pytest.raises(RuntimeError), driver.decision_log(path) as emit:
        emit(dict(vector_frame=1, env_index=0, reason='structural_decision'))
        emit(dict(vector_frame=1, env_index=1, reason='continuous_resource_control'))
        assert len(path.read_text().splitlines()) == 2
        raise RuntimeError('intentional')
    assert [r['env_index'] for r in map(json.loads, path.read_text().splitlines())] == [0, 1]
    with pytest.raises(FileExistsError), driver.decision_log(path):
        pass


def test_driver_training_hook_records_reasons_and_rendering_with_offline_runner_stub(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from drl.MARL import isaaclab_c1_runner as runner, isaaclab_c1_training as training
    from drl.MARL.c1_training_options import NativeStructuralOptionController
    from drl.MARL.tests.test_c1_training_option_runtime import _Env
    driver = _driver()
    output = tmp_path/'hook_output'
    env = _Env()
    env.sim = SimpleNamespace(has_gui=False, is_rendering=False)
    env.render_enabled = True
    ctl = NativeStructuralOptionController(env, mode='event')
    learner = SimpleNamespace(_rollout_context_provider=SimpleNamespace(native_env=env, _structural_options=ctl))
    def original(*args, **kwargs):
        assert not env.render_enabled
        assert ctl.decision_mask() == (True, True)
        return 'offline-hook-tested'
    monkeypatch.setattr(training, 'run_native_training', original)
    # This stub replaces the application entrypoint: no SDK import or startup.
    monkeypatch.setattr(runner, 'main', lambda argv: training.run_native_training(
        SimpleNamespace(device='cpu'), env, SimpleNamespace(), learner))
    result = driver.run(dict(output=str(output), variant='optimized', source_root=str(ROOT),
                             native_argv=[], cuda_events=False))
    assert result == 'offline-hook-tested'
    records = list(map(json.loads, (output/'decision_reasons.jsonl').read_text().splitlines()))
    assert len(records) == 2 and all(r['reason'] == 'structural_decision' for r in records)
    report = json.loads((output/'performance.json').read_text())
    assert report['rendering_state'] == dict(has_gui=False, is_rendering=False, render_enabled=False)
    assert report['performance']['decision_reason_counts']['structural_decision'] == 2
    assert training.run_native_training is original


def test_real_materialized_baseline_has_original_physics_and_import_source(tmp_path):
    driver = _driver()
    destination = tmp_path/'baseline_source'
    driver.materialize_baseline(driver.ROOT, driver.BEFORE, destination)
    snapshots = driver._load(destination/'drl/MARL/c1_expert_snapshots.py', '_materialized_physics_test')
    assert snapshots.current_snapshot_physics() == {
        'version': 'c1-native-snapshot-physics-v1',
        'source_sha256': 'a009a64712a2ff7974bc3b4bc1422f7bcfc13a20d83ebef9d7beb7c2be3c89f2'}
    for name in ('c1_incremental_diagnostics.py', 'c1_performance.py', 'c1_ordered_logging.py'):
        assert not (destination/'drl/MARL'/name).exists()
    with driver._selected_source_imports(destination):
        from drl.MARL import isaaclab_c1_runner as runner
        assert Path(runner.__file__).is_relative_to(destination)
    assert not (destination/'docs').exists()
    parsed, _ = driver._parse_native(destination, driver.build_native_argv(tmp_path/'run', destination, 'cuda:0'))
    assert parsed['repository_root'] == destination
    assert parsed['snapshot_stage_weights'] == {'S0': .25, 'S1': .2, 'S2': .25, 'S3': .3}
