"""Optional OFFLINE topology classification. Never imported by reward/control."""
import json
from pathlib import Path

from drl.MARL.c1_structure import C1Frame,C1Module,C1Connection,canonical_signature


def classify_records_offline(records,repository_root,baseline_records=()):
    """Compare logged actual graphs with known topology metadata post-hoc.

    Expert target graphs do not certify a physical posture. That uncertainty is
    explicit. A supplied physical baseline can prove a different joint-position
    multiset; equal multisets alone are not claimed as full pose isomorphism.
    """
    root=Path(repository_root);known=[]
    for name in ('rc_car8','snake8','mobile_manipulator8'):
        doc=json.loads((root/'mssr_ws/src/mssr_expert/config'/f'smores_{name}.json').read_text())
        modules=tuple(C1Module(n['node_id'],(0.,0.,0.)) for n in doc['nodes'])
        edges=tuple(C1Connection(str(i),e['module_a_id'],e['attributes']['face_a'],e['module_b_id'],e['attributes']['face_b'],True) for i,e in enumerate(doc['edges']))
        frame=C1Frame(0.,modules,edges)
        known.append((name,canonical_signature(frame),canonical_signature(frame,include_ports=False)))
    def joint_multiset(record):
        return sorted(tuple(sorted((k,round(float(v),3)) for k,v in posture.get('joints',())
            if k in ('pan.position_rad','tilt.position_rad'))) for posture in record.get('posture_summary',{}).values())
    rows=[]
    for record in records:
        exact=[name for name,ports,_ in known if ports==record.get('canonical_topology_signature')]
        same=[name for name,_,shape in known if shape==record.get('unlabelled_topology_signature')]
        label='known_topology_exact_match' if exact else ('same_known_topology_different_connector_pattern' if same else 'different_topology')
        posture='not certified by topology-only baseline'
        for baseline in baseline_records:
            if baseline.get('canonical_topology_signature')==record.get('canonical_topology_signature'):
                if joint_multiset(baseline)!=joint_multiset(record):label='same_known_topology_different_posture';posture='different physical joint-position multiset'
                else:posture='joint multiset matches; full spatial posture not certified'
        rows.append(dict(env_index=record.get('env_index'),generation=record.get('generation'),task=record['task'],seed=record['seed'],
            classification=label,known_topologies=exact or same,posture_status=posture,
            used_for_reward=False))
    return rows
