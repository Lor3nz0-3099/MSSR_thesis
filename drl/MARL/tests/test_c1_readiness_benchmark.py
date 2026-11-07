"""Offline readiness accounting; no Isaac imports or simulator launch."""
import json
import sys

import pytest

from drl.MARL import c1_readiness_benchmark as benchmark


DOMAINS = ('navigation', 'stairs', 'gap', 'manipulation')


def measurement(envs=4, transitions=128):
    return dict(status='pass', num_envs=envs,
                requested_total_transitions=transitions, active_marl_transitions=transitions,
                timings_s=dict(startup=10., assembly=5., reset=4., rollout=16., update=8., total=43.),
                optimizer=dict(rollout_active_transitions=32, learning_epochs=2, mini_batches=2, completed_ppo_updates=4),
                checkpoint_bytes=1000,
                domains={domain: dict(episodes=[dict(transitions=8, simulated_s=1., wall_s=2.,
                                                      termination='terminated')],
                                      censored_episodes=1, reset_assembly_s=1.) for domain in DOMAINS})


def test_exit_zero_does_not_override_failure_sentinel_or_missing_status(tmp_path):
    result_path = tmp_path / 'result.json'
    result_path.write_text(json.dumps(measurement()))
    with pytest.raises(ValueError, match='FAIL'):
        benchmark.load_run_result(result_path, returncode=0, stdout='NATIVE_C1_PRETRAIN_FAIL: bad')
    result_path.write_text(json.dumps({'num_envs': 4}))
    with pytest.raises(ValueError, match='status'):
        benchmark.load_run_result(result_path, returncode=0)


def test_comparison_requires_equal_actual_active_experience():
    assert benchmark.scaling_comparison([measurement(4), measurement(8), measurement(16)])['comparable']
    report = benchmark.scaling_comparison([measurement(4), measurement(8, 129)])
    assert not report['comparable']
    assert 'equal' in report['reason']
    assert report['rows'][0]['active_transitions_per_s'] == pytest.approx(128 / 16)
    assert report['rows'][0]['vector_decisions_per_s'] is None


def test_campaign_budget_uses_completed_episode_units_and_optimizer_steps():
    report = benchmark.estimate_campaigns(measurement(),
                                         episode_exposures={'SHORT': 1, 'RECOMMENDED': 2, 'EXTENDED': 4},
                                         checkpoint_every_transitions=10000)
    assert report['available']
    short = report['budgets']['SHORT']
    assert short['episodes_total'] == 4000
    assert short['episodes_per_domain'] == dict.fromkeys(DOMAINS, 1000)
    assert short['episodes_per_difficulty'] == dict(easy=1000, medium=1600, hard=1400)
    assert short['active_environment_transitions'] == 32000
    assert short['ppo_updates'] == 1000
    assert short['optimizer_steps_per_owner'] == 4000
    # 16/128 active rollout sec/transition + 8/128 update sec/transition;
    # reset/assembly per completed episode, initial startup+assembly.
    assert short['wall_seconds_point_estimate'] == 10015
    assert short['checkpoint_storage_bytes'] == 5000
    assert short['uncertainty']['kind'] == 'scenario_range_not_confidence_interval'
    assert report['observations']['navigation']['censored_episodes'] == 1


def test_estimate_fails_closed_without_real_completed_episodes():
    run = measurement()
    run['domains']['gap']['episodes'] = []
    report = benchmark.estimate_campaigns(run)
    assert not report['available']
    assert 'gap' in report['reason']
    assert report['budgets'] == {}


def test_horizon_endings_are_not_treated_as_natural_duration():
    run = measurement()
    run['domains']['stairs']['episodes'][0]['termination'] = 'horizon'
    report = benchmark.estimate_campaigns(run)
    assert not report['available']
    assert report['observations']['stairs']['horizon_episodes'] == 1
    assert report['observations']['stairs']['natural_completed_episodes'] == 0


def test_nonfinite_or_missing_timings_cannot_generate_estimates():
    for invalid in (None, float('nan'), 0):
        run = measurement()
        run['timings_s']['rollout'] = invalid
        assert not benchmark.estimate_campaigns(run)['available']


def test_report_writes_json_csv_and_preserves_seed_manifest(tmp_path):
    run = measurement()
    run['seed_manifest'] = {'navigation': [7002, 7549]}
    benchmark.write_reports(tmp_path, [run], metadata={'git_head': 'abc'})
    report = json.loads((tmp_path / 'benchmark.json').read_text())
    assert report['runs'][0]['seed_manifest'] == run['seed_manifest']
    assert report['metadata']['git_head'] == 'abc'
    assert 'active_marl_transitions' in (tmp_path / 'scaling.csv').read_text().splitlines()[0]


def test_driver_executes_sequential_commands_and_records_failure(tmp_path):
    command = [sys.executable, '-c', "print('NATIVE_C1_PRETRAIN_FAIL: deliberate')"]
    result = benchmark.execute_run(command, tmp_path)
    assert result['status'] == 'fail'
    assert result['returncode'] == 0
    assert 'FAIL' in result['error']
    assert (tmp_path / 'stdout.log').is_file()


def test_update_time_uses_actual_completed_updates_not_rounded_transition_count():
    run = measurement()
    run['optimizer']['completed_ppo_updates'] = 2
    report = benchmark.estimate_campaigns(run)
    assert report['budgets']['SHORT']['timing_components_s']['update'] == 4000


def test_unknown_completed_update_count_prevents_duration_claim():
    run = measurement()
    run['optimizer'].pop('completed_ppo_updates', None)
    assert not benchmark.estimate_campaigns(run)['available']


def test_report_emits_exact_manifest_and_fail_closed_campaigns(tmp_path):
    run = measurement()
    run['seed_manifest'] = {'navigation': [7002, 7549]}
    run['domains']['gap']['episodes'] = []
    benchmark.write_reports(tmp_path, [run], metadata={})
    assert json.loads((tmp_path / 'seed_manifest.json').read_text()) == [run['seed_manifest']]
    estimates = json.loads((tmp_path / 'campaign_estimates.json').read_text())
    assert not estimates[0]['available']


def test_offline_cli_never_requires_simulator(tmp_path):
    source = tmp_path / 'measured.json'
    source.write_text(json.dumps(measurement()))
    assert benchmark.main(['--input', str(source), '--output', str(tmp_path / 'report'),
                           '--repository-root', str(tmp_path)]) == 0
    assert (tmp_path / 'report' / 'benchmark.json').is_file()


def test_horizon_bounds_use_production_horizon_concurrency_and_catalog_draws():
    run = measurement(envs=8)
    run['assembly_records'] = [dict(env_index=i, domain=domain, wall_s=10.)
        for i, domain in enumerate(DOMAINS * 2)]
    for row in run['domains'].values():
        row['episodes'][0]['termination'] = 'horizon'
    report = benchmark.estimate_horizon_bounds(run, production_horizon_s=60., decision_dt_s=1/30,
        episode_scenarios={'SHORT': 100, 'RECOMMENDED': 1000, 'EXTENDED': 3000})
    assert report['available']
    short = report['budgets']['SHORT']
    assert short['active_environment_transitions_bounds'] == [400, 720000]
    assert short['episodes_total'] == 400
    assert short['episodes_per_difficulty'] == dict(easy=100., medium=160., hard=140.)
    assert short['expected_catalog_coverage_fraction_per_domain']['gap'] == pytest.approx(0.09516258196)
    # 2 initial episodes/domain already covered by initial assembly timing.
    # Parallel groups: max(98*10/2)=490; serialized:4*98*10=3920.
    assert short['reset_wall_seconds_scenario_bounds'] == [490., 3920.]
    assert 'wall_seconds_point_estimate' not in short
    assert report['production_horizon_decisions'] == 1800
    assert not benchmark.estimate_campaigns(run)['available']


def test_horizon_bounds_refuse_missing_assembly_or_rate_measurements():
    report = benchmark.estimate_horizon_bounds(measurement(), production_horizon_s=60.,
        decision_dt_s=1/30, episode_scenarios={'SHORT': 100})
    assert not report['available']
    assert 'assembly' in report['reason']


def test_vector_steps_alias_is_reported_as_vector_decisions():
    run = measurement()
    run['vector_steps'] = 32
    row = benchmark.scaling_comparison([run])['rows'][0]
    assert row['vector_decisions_per_s'] == 2.
