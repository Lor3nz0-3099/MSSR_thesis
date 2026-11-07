import json

from drl.MARL.action_projection_diagnostic import summarize,write_json


def test_unexecuted_diagnostic_has_reference_and_no_invented_after_counts(tmp_path):
    write_json(tmp_path/'reference_summary.json',{'captured_decisions':16,'accepted':0,
        'structural_proposals_by_action':{'align_faces':84,'dock':21}})
    result=summarize(tmp_path)
    assert result['status']=='awaiting_offline_diagnostic'
    assert result['before_reference']['captured_decisions']==16
    assert result['after']['native_admitted_nonempty_bundles'] is None
    assert result['after']['raw_structural_selections'] is None
    assert result['optimizer_updates']==result['expert_actions']==0


def test_diagnostic_distinguishes_projection_admission_and_observed_execution(tmp_path):
    directory=tmp_path/'navigation';directory.mkdir()
    (directory/'decisions.jsonl').write_text(json.dumps(dict(task='navigation',raw_cardinality=3,
        projected_cardinality=2,native_admitted_cardinality=1,raw_structural_selections=3,
        projected_structural=2,native_admitted_structural=1,conflict_pairs=1,
        conflict_owner_ids=['b'],invalid_owner_ids=[]))+'\n')
    (directory/'physics.jsonl').write_text(json.dumps(dict(task='navigation',primitive_statuses=[
        dict(goal_id='g',primitive='align_faces',state='running')]))+'\n')
    result=summarize(tmp_path)
    assert result['after']['raw_structural_selections']==3
    assert result['after']['projected_executable_nonempty_bundles']==1
    assert result['after']['native_admitted_structural_actions']==1
    assert result['after']['observed_executing_structural_goals']==1
    assert result['status']=='awaiting_offline_diagnostic'
