"""Unit tests for Stage-8 evidence; no RAW/G2 fixture is necessary."""
from drl.IL.bc_v1.bc_v5_align_joint_history import (
    _links, _status, _event_faces, _first_link_disappearance, _row_snapshot,
)


def _graph(effective=False, with_edge=True):
    edge = {'relation': 'current_connection', 'module_a_id': 'm1',
            'module_b_id': 'm2', 'face_a':'TOP','face_b':'RIGHT'}
    return {'stamp': 1.0, 'morphology': 'reconfiguring',
            'edges':[edge] if with_edge else [], 'nodes':[
                {'module_id': 'm1', 'control_state': {'faces':{
                    'TOP':{'physical_connected':with_edge,
                           'connected_for_executor':effective,
                           'availability_source':'executor_structural_history'}}}},
                {'module_id':'m2','control_state':{'faces':{
                    'RIGHT':{'physical_connected':with_edge,
                             'connected_for_executor':effective}}}}]}


def test_stale_edge_is_possible_but_not_proven_removed():
    links = _links(_graph(False))
    assert links['m1'] == [{'own_face':'TOP','peer':'m2','peer_face':'RIGHT'}]
    assert links['m2'] == [{'own_face':'RIGHT','peer':'m1','peer_face':'TOP'}]
    assert _status(links['m1'], _graph(False)['nodes'][0]['control_state']['faces']) == (
        'physical_edge_but_executor_override_detached')
    assert _status(links['m1'], _graph(True)['nodes'][0]['control_state']['faces']) == (
        'also_connected_for_executor')


def test_absent_evidence_is_not_inferred_detached():
    assert _status([{'own_face':'TOP','peer':'m2','peer_face':'RIGHT'}], {}) == 'executor_evidence_missing'
    assert _status([], {}) == 'no_current_connection_edge'


def test_prior_undock_faces_are_both_registered():
    cmd = {'module':'m1','primitive':'undock','parameters':{'face_a':'TOP','face_b':'RIGHT'}}
    participants = {'m1':{'participates_in':[]},'m2':{'participates_in':[
        {'primitive':'undock','initiator_module_id':'m1','own_face':'RIGHT'}]}}
    assert _event_faces(cmd, participants) == [('m1','TOP'),('m2','RIGHT')]


def test_following_physical_snapshot_can_resolve_lag():
    seq = [ {'index':10, 'stamp':1.0,'links':_links(_graph(False))},
            {'index':11, 'stamp':1.1,'links':_links(_graph(False))},
            {'index':12, 'stamp':1.2,'links':_links(_graph(False,False))}]
    res = _first_link_disappearance(seq, 0, 'm1', seq[0]['links']['m1'], 10)
    assert res['transition_index'] == 12
    assert res['fully_isolated'] is True
    assert _first_link_disappearance(seq,0,'m1',seq[0]['links']['m1'],1) is None


def test_snapshot_does_not_invent_actions():
    g = _graph(False)
    modules = {mid: {'structural': {'commanded': False, 'primitive':None}}
               for mid in ('m1','m2')}
    row = {'graph_t':g, 'action_t':{'modules':modules},
           'supervision':{'module_resource_masks':{
               mid:{'structural':False} for mid in modules}}}
    assert _row_snapshot(row, 7)['actions'] == []


def test_new_executor_only_dock_on_other_face_means_not_detached():
    link = [{'own_face':'TOP','peer':'m2','peer_face':'RIGHT'}]
    faces = {
        'TOP':{'connected_for_executor':False},
        'BOTTOM':{'connected_for_executor':True},
    }
    assert _status(link,faces) == 'also_connected_for_executor'
