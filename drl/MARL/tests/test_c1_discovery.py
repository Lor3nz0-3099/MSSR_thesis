import json
from pathlib import Path
from drl.MARL.c1_discovery import classify_records_offline
from drl.MARL.c1_structure import C1Module,C1Connection,C1Frame,canonical_signature
from drl.MARL.tests.test_c1_structure import frame

ROOT=Path(__file__).resolve().parents[3]


def record(f):return dict(task='navigation',seed=7017,canonical_topology_signature=canonical_signature(f),
    unlabelled_topology_signature=canonical_signature(f,include_ports=False),posture_summary={})


def test_known_comparison_is_offline_and_does_not_certify_unknown_posture():
    doc=json.loads((ROOT/'mssr_ws/src/mssr_expert/config/smores_snake8.json').read_text())
    modules=tuple(C1Module(n['node_id'],(0.,0.,0.)) for n in doc['nodes'])
    edges=tuple(C1Connection(str(i),e['module_a_id'],e['attributes']['face_a'],e['module_b_id'],e['attributes']['face_b'],True) for i,e in enumerate(doc['edges']))
    result=classify_records_offline([record(C1Frame(0.,modules,edges))],ROOT)[0]
    assert result['classification']=='known_topology_exact_match'
    assert 'snake8' in result['known_topologies']
    assert not result['used_for_reward']
    assert 'not certified' in result['posture_status']


def test_connected_novel_cycle_is_logged_as_different_without_changing_validity():
    result=classify_records_offline([record(frame(edges=8))],ROOT)[0]
    assert result['classification']=='different_topology'
    assert not result['used_for_reward']
