"""CPU diagnostics must be passive and distinguish attach-ready from executable."""
import json
from types import SimpleNamespace
import pytest
import torch
from smores_ep.docking.model import DockingFace,DockingFacePose,DockingThresholds


def pair(gap=.002,source='TOP',target='BOTTOM'):
    a=DockingFacePose(DockingFace('a',source,'/a','/a'),(0.,0.,0.),(1.,0.,0.),(0.,1.,0.))
    b=DockingFacePose(DockingFace('b',target,'/b','/b'),(gap,0.,0.),(-1.,0.,0.),(0.,1.,0.))
    return a,b


@pytest.mark.parametrize('face,limit',[('TOP',.0025),('LEFT',.0067)])
def test_geometry_uses_native_boundaries_without_changing_support(face,limit):
    from drl.MARL.c1_viewer_geometry import geometry_record
    for gap,ready in ((limit-1e-9,True),(limit,True),(limit+1e-9,False)):
        a,b=pair(gap,face,'BOTTOM' if face=='TOP' else 'RIGHT')
        result=geometry_record(a,b,DockingThresholds())
        assert result['native_attach_eligible'] is ready
        assert result['gap_m']==pytest.approx(gap)
        assert result['connector_distance_m']==pytest.approx(gap)
        assert result['classification']==('READY' if ready else 'NOT_READY_FOR_FINAL_ATTACH')
        assert result['admission_mask_changed'] is False


def test_geometry_records_lateral_normal_clocking_and_unknown_without_fabrication():
    from drl.MARL.c1_viewer_geometry import geometry_record,capture_dock_geometry
    a,b=pair(.006)
    result=geometry_record(a,b,DockingThresholds())
    assert not result['native_attach_eligible']  # known legal6mm approach, no mask!
    assert result['lateral_error_m']==0. and result['normal_error_rad']==0.
    assert 'relative_pose' in result and 'clocking_constrained' in result
    bad=capture_dock_geometry(SimpleNamespace(),SimpleNamespace(module_ids=('a','b'),parameters={'face_a':'TOP','face_b':'BOTTOM'}))
    assert bad['classification']=='UNKNOWN' and bad['gap_m'] is None


def test_host_profiler_nested_spans_quantiles_render_gaps_and_return_identity(tmp_path):
    from drl.MARL.c1_viewer_timing import HostTimingProbe
    clock=[0.];probe=HostTimingProbe(tmp_path,clock=lambda:clock[0])
    target=object();calls=[]
    api=SimpleNamespace(call=lambda:calls.append(1) or target)
    probe.wrap(api,'call','inner')
    probe.begin_decision(0,0.)
    with probe.span('outer'):
        clock[0]=.02;assert api.call() is target;clock[0]=.8
    probe.render_event('sim.render',.0,.01);probe.render_event('sim.render',.02,.03);probe.render_event('sim.render',.82,.83)
    probe.end_decision()
    summary=probe.finish()
    assert calls==[1]
    assert summary['stages']['outer']['inclusive_s']['p50']==pytest.approx(.8)
    assert summary['render_intervals']['sim.render']['p95_s']>.7
    probe.detach();assert api.call() is target and len(calls)==2
    assert 'cuda_synchronize_per_block' in summary['notes']


def test_probe_model_and_sampler_do_not_change_rng_or_outputs(tmp_path):
    from drl.MARL.c1_viewer_timing import HostTimingProbe
    from drl.MARL.tests.test_c1_tier1_policy import mixed,equal
    from drl.MARL import policy
    data=mixed();mask=torch.tensor([False,True,False,False,True,False])
    torch.manual_seed(77);expected=policy.sample_marl_action_bundle(**data,hard_mask=mask);rng=torch.random.get_rng_state()
    probe=HostTimingProbe(tmp_path);probe.wrap(policy,'sample_marl_action_bundle','sampling')
    torch.manual_seed(77);probe.begin_decision(0,0.)
    actual=policy.sample_marl_action_bundle(**data,hard_mask=mask)
    probe.end_decision();probe.finish();probe.detach()
    equal(actual,expected);assert torch.equal(rng,torch.random.get_rng_state())


def test_probe_stats_do_not_resample_and_record_resource_structural_counts(tmp_path):
    from drl.MARL.c1_viewer_timing import HostTimingProbe
    from drl.MARL.tests.test_c1_tier1_policy import mixed
    from drl.MARL.policy import sample_marl_action_bundle
    from drl.MARL.c1_feasibility import NativeAvailability,CandidateAvailability
    data=mixed();mask=torch.tensor([False,True,False,False,True,False])
    torch.manual_seed(3);action=sample_marl_action_bundle(**data,hard_mask=mask);rng=torch.random.get_rng_state()
    reasons=tuple(CandidateAvailability('masked_busy' if x else 'available') for x in mask)
    support=NativeAvailability(tuple(mask.tolist()),reasons)
    out=dict(data,prediction_template={'presence_logits':torch.stack([r.presence_logit for r in data['records']])})
    probe=HostTimingProbe(tmp_path);probe.begin_decision(0,0.)
    probe.observe_policy(out,action,support);probe.end_decision();probe.finish();probe.detach()
    row=json.loads((tmp_path/'policy_support.jsonl').read_text().splitlines()[0])
    assert row['available']['structural']==2 and row['available']['resource']==2
    assert row['sampled']['structural']+row['sampled']['resource']==int(action.presence.sum())
    assert torch.equal(rng,torch.random.get_rng_state())
