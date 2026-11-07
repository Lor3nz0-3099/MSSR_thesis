from dataclasses import replace
import math
import pytest

from drl.MARL.c1_structure import C1Module,C1Connection,C1Frame,PhysicalBounds,assess_structure,canonical_signature


def frame(t=0.,edges=7,**kwargs):
    modules=[]
    for i in range(8):
        p=(.1*math.cos(i*math.pi/4),.1*math.sin(i*math.pi/4),.04)
        joints=tuple((f'{name}.{field}',0.) for name in ('left_wheel','right_wheel','pan','tilt') for field in ('position_rad','velocity_rad_s'))
        ports=tuple((name,(p[0]+dx,p[1]+dy,p[2]+dz)) for name,dx,dy,dz in (
            ('LEFT',-.04,0,0),('RIGHT',.04,0,0),('TOP',0,0,.04),('BOTTOM',0,0,-.04)))
        modules.append(C1Module(f'm{i}',p,joints=joints,connectors=ports))
    connections=tuple(C1Connection(f'j{i}',f'm{i}','RIGHT',f'm{(i+1)%8}','LEFT',True) for i in range(edges))
    return C1Frame(t,tuple(modules),connections,**kwargs)


def assess(value):return assess_structure(value,PhysicalBounds((-2,2,-2,2),-.16,1.,3.,100.,.132,6.))


def test_disconnected_eight_modules_is_not_valid():
    result=assess(frame(edges=6))
    assert not result.instantaneous_valid
    assert result.component_sizes==(7,1)


def test_connected_stable_component_and_novel_cycle_are_valid():
    assert assess(frame()).instantaneous_valid
    ring=assess(frame(edges=8))
    assert ring.instantaneous_valid and ring.valid_connections==8
    assert ring.component_sizes==(8,)


@pytest.mark.parametrize('fault',['duplicate_connector','bad_joint','missing_module','nan','fast'])
def test_invalid_physical_assemblies_fail(fault):
    f=frame()
    if fault=='duplicate_connector':f=replace(f,connections=(*f.connections,C1Connection('jx','m0','RIGHT','m7','RIGHT',True)))
    if fault=='bad_joint':f=replace(f,connections=(replace(f.connections[0],joint_valid=False),*f.connections[1:]))
    if fault=='missing_module':f=replace(f,modules=f.modules[:-1])
    if fault=='nan':f=replace(f,modules=(replace(f.modules[0],position=(math.nan,0,.04)),*f.modules[1:]))
    if fault=='fast':f=replace(f,peak_linear_speed=.5)
    result=assess(f)
    assert not result.instantaneous_valid
    if fault in ('nan','missing_module'):assert not result.safe


def test_one_lost_module_or_explosion_is_terminal_safety_failure():
    f=frame()
    assert not assess(replace(f,modules=(replace(f.modules[0],position=(0,0,-1.)),*f.modules[1:]))).safe
    assert not assess(replace(f,peak_linear_speed=50.)).safe


def test_canonical_signature_is_exact_and_invariant_to_module_ids():
    f=frame(edges=8)
    mapping={f'm{i}':f'other{(i*3)%8}' for i in range(8)}
    changed=replace(f,modules=tuple(replace(m,module_id=mapping[m.module_id]) for m in reversed(f.modules)),
        connections=tuple(replace(c,module_a=mapping[c.module_a],module_b=mapping[c.module_b]) for c in reversed(f.connections)))
    assert canonical_signature(f)==canonical_signature(changed)
    assert canonical_signature(f)!=canonical_signature(frame(edges=7))


@pytest.mark.parametrize('fault',['missing_joints','missing_ports','tilt_position','joint_speed','transient_invalid'])
def test_missing_or_impossible_joint_evidence_cannot_pass_validity(fault):
    f=frame();m=f.modules[0]
    if fault=='missing_joints':m=replace(m,joints=())
    if fault=='missing_ports':m=replace(m,connectors=())
    if fault in ('tilt_position','joint_speed'):
        key='tilt.position_rad' if fault=='tilt_position' else 'tilt.velocity_rad_s'
        m=replace(m,joints=tuple((k,1e6 if k==key else v) for k,v in m.joints))
    f=replace(f,modules=(m,*f.modules[1:]))
    if fault=='transient_invalid':f=replace(f,interval_valid=False)
    assert not assess(f).instantaneous_valid


@pytest.mark.parametrize('depression',[-.01,.02])
def test_impossible_button_stroke_is_a_physical_failure(depression):
    result=assess(replace(frame(),button_depression_m=depression))
    assert not result.safe and result.reason=='invalid_button_stroke_state'
