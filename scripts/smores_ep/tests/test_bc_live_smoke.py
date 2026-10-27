"""Bounded runner contract; no process launches, simulation, or dataset reads."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[3]


def api():
    path=ROOT/'scripts/smores_ep/run_bc_live_smoke.py'
    assert path.is_file(), 'Missing bounded BC runtime orchestration'
    spec=importlib.util.spec_from_file_location('bc_live_smoke_test',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def test_runtime_launch_uses_existing_standalone_runtime_30hz_and_no_action_teachers(tmp_path):
    command=api().runtime_command(ROOT,tmp_path)
    assert command[0]==str(ROOT/'scripts/smores_ep/run_self_assembly.sh')
    assert command[command.index('--state-publish-hz')+1]=='30'
    assert command[command.index('--rc-car-seed')+1]=='5100'
    assert '--rc-car-planar-test-course' in command
    assert not any(x in ' '.join(command).lower() for x in ('nav2','dataset','teleop'))


def test_unexpected_structural_proposal_stops_without_silently_filtering():
    a=api()
    from drl.IL.bc_v1.live_controller import RuntimeCommand
    structural=RuntimeCommand(('m1','structural'),{'command':'undock'},('connector:m1:TOP',),False)
    with pytest.raises(a.UnexpectedStructuralProposal):
        a.smoke_inference(lambda *_:(structural,),None,{})


def test_smoke_guard_preserves_legitimate_resource_targets_and_zero():
    a=api()
    from drl.IL.bc_v1.live_controller import RuntimeCommand
    target=RuntimeCommand(('m1','internal_motion'),{'command':'set_tilt'},('internal_motion:m1',),False)
    assert a.smoke_inference(lambda *_:(target,),None,{})==(target,)


def test_pre_evaluation_setup_reuses_existing_planner_and_executor():
    a=api()
    from drl.IL.tests.bc_v1.test_live_observation import snapshot,builder
    from mssr_expert.execution.parallel_assembly_executor import ParallelAssemblyExecutor
    raw=snapshot(8)
    for module,node in zip(raw['state']['modules'],raw['graph']['nodes']):
        module['robot_family']='smores_ep';node['attributes']['robot_family']='smores_ep'
    current=builder().build(raw['state'],raw['graph'])
    executor=a.make_setup_executor(ROOT,current,'setup-fixture')
    assert isinstance(executor,ParallelAssemblyExecutor)
    decision=executor.step({})
    assert not decision.done and decision.primitive_goal is not None


def test_headless_cli_budget_allows_setup_past_thirty_sim_seconds(monkeypatch,tmp_path):
    """Catch the CLI's 0 -> 7200 fallback, without starting SimulationApp."""
    from types import SimpleNamespace
    from smores_ep import self_assembly_cli
    captured=[]
    class App:
        def __init__(self,*args,**kwargs):pass
        def close(self):pass
    monkeypatch.setitem(sys.modules,'isaacsim',SimpleNamespace(SimulationApp=App))
    # Replace only the simulator boundary; execute real argument/config handling.
    monkeypatch.setitem(sys.modules,'smores_ep.scenarios.parallel_self_assembly',
        SimpleNamespace(run_parallel_self_assembly_scenario=lambda config,app:captured.append(config)))
    monkeypatch.setattr(sys,'argv',['self_assembly',*api().runtime_command(ROOT,tmp_path)[1:]])
    self_assembly_cli.main()
    config=captured[0]
    setup_seconds=45.
    assert config.steps/config.physics_hz > setup_seconds+5., (
        'Headless runtime must survive a 45 s setup and the subsequent 5 s BC window')


def test_smoke_bounds_start_only_at_bc_authority_arming():
    a=api()
    check=getattr(a,'smoke_bound_reached',None)
    assert callable(check), 'Runner must enforce its bounds relative to BC authority arming'
    # Long setup is outside the BC evaluation window, even past both BC bounds.
    assert not check(None,None,wall_now=200.,physical_stamp=80.)
    assert not check(200.,80.,wall_now=204.9,physical_stamp=84.9)
    assert check(200.,80.,wall_now=205.,physical_stamp=85.)
    assert not check(200.,80.,wall_now=319.9,physical_stamp=80.1)
    assert check(200.,80.,wall_now=320.,physical_stamp=80.1)


def test_summary_serializes_owner_busy_set_explicitly_and_deterministically(tmp_path):
    a=api();convert=getattr(a,'serializable_owner_status',None)
    assert callable(convert), 'Owner status busy set needs deterministic explicit serialization'
    status={'busy':{'locomotion:m2','internal_motion:m1'},'applied_packets':0}
    result=convert(status)
    assert result['busy']==['internal_motion:m1','locomotion:m2']
    assert isinstance(status['busy'],set)
    a.write_json(tmp_path/'summary.json',{'owner_status':result})


def test_runner_records_loaded_checkpoint_provenance_without_starting_runtime(tmp_path, monkeypatch):
    import json
    import time
    from types import SimpleNamespace
    from drl.IL.bc_v1 import live_policy
    a=api()
    provenance=dict(checkpoint_path='/synthetic/epoch13.pt',checkpoint_sha256='a'*64,
        checkpoint_epoch=13,checkpoint_global_step=15912,checkpoint_profile='bc_v1.no_structural_v2',
        checkpoint_schema='mssr.bc_v1.training_checkpoint.v1',action_heads_schema='mssr.bc_v1.action_heads.v2')
    policy=SimpleNamespace(checkpoint_provenance=provenance,forward_count=0,timings=[],last_candidate_domain={})
    monkeypatch.setattr(live_policy,'load_frozen_policy',lambda *args,**kwargs:policy)
    launches=[]
    def no_launch(*args,**kwargs):
        launches.append('blocked');raise RuntimeError('Simulation disabled by CPU-only unit test')
    monkeypatch.setattr(a.subprocess,'Popen',no_launch)
    cleanup=tmp_path/'cleanup.json';cleanup.write_text(json.dumps(dict(passed=True,wall_monotonic=time.monotonic())))
    output=tmp_path/'output';result=a.run(output,Path(provenance['checkpoint_path']),cleanup)
    saved=json.loads((output/'run_config.json').read_text())
    for key,value in provenance.items():
        assert saved[key]==value
        assert result[key]==value
    assert launches==['blocked']


def test_cli_expected_checkpoint_sha_is_forwarded_without_running_simulation(tmp_path,monkeypatch):
    a=api();captured=[]
    def fake_run(*args,**kwargs):
        captured.append((args,kwargs))
        return dict(runtime_integration_success=True,policy_behavior='fixture',failure=None,decision_count=0)
    monkeypatch.setattr(a,'run',fake_run)
    monkeypatch.setattr(sys,'argv',['run_bc_live_smoke.py','--output',str(tmp_path/'output'),
        '--checkpoint','/synthetic/model.pt','--cleanup-manifest',str(tmp_path/'cleanup.json'),
        '--expected-checkpoint-sha256','b'*64])
    assert a.main()==0
    assert captured[0][1]['expected_checkpoint_sha256']=='b'*64


def test_runner_sha_failure_occurs_before_any_process_launch(tmp_path,monkeypatch):
    import json
    import time
    from drl.IL.bc_v1 import live_policy
    a=api()
    def reject(*args,**kwargs):raise ValueError('Frozen checkpoint SHA mismatch')
    monkeypatch.setattr(live_policy,'load_frozen_policy',reject)
    monkeypatch.setattr(a.subprocess,'Popen',lambda *a,**kw:pytest.fail('No process may start'))
    cleanup=tmp_path/'cleanup.json';cleanup.write_text(json.dumps(dict(passed=True,wall_monotonic=time.monotonic())))
    with pytest.raises(ValueError,match='checkpoint SHA'):
        a.run(tmp_path/'output',tmp_path/'model.pt',cleanup,expected_checkpoint_sha256='0'*64)
