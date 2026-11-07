from types import SimpleNamespace as NS
from dataclasses import replace
from drl.MARL.tests.test_c1_stagnation import monitor
from drl.MARL.tests.test_c1_structure import frame
from drl.MARL.c1_structure import C1PrimitiveEvent
from drl.MARL.c1_redundancy import RedundancyMonitor


def test_global_stagnation_requires_opportunities_after_a_recovery_grant():
    m=monitor()
    for i in range(24):m.step(frame(i,edges=0),structural_score=0,decision_opportunity=True)
    f=replace(frame(24,edges=0),primitive_events=(C1PrimitiveEvent('g','align_faces','failed','NO_PATH','m0'),))
    m.step(f,structural_score=0)
    for i in range(25,35):r=m.step(frame(i,edges=0),structural_score=0)
    assert not r.terminate and r.decision_opportunities==0
    for i in range(35,38):r=m.step(frame(i,edges=0),structural_score=0,decision_opportunity=True)
    assert r.terminate


def test_sparse_requests_with_held_ticks_accumulate_without_counting_refreshes():
    settings=NS(values={'redundancy':dict(enabled=True,continuous_window_s=10.,cooldown_s=10.)})
    m=RedundancyMonitor(tuple(f'm{i}' for i in range(8)),settings)
    cmd=NS(linear_x_m_s=.1,angular_z_rad_s=0.,pan_velocity_rad_s=0.)
    events=[]
    for i in range(361):
        payload=dict(request_id=i//15,command=cmd) if i%15==0 else cmd
        _,d=m.step(frame(i/30,edges=0),{},continuous_commands={'m0':payload})
        events+=d['events']
    assert len(events)==1 and events[0]['reason']=='sustained_ineffectual_requests'


def test_periodic_pan_revolutions_are_not_new_physical_progress():
    import math
    m=monitor();base=frame(edges=0)
    goal=dict(primitive='align_faces',module_ids=('m0','m1'),connectors=(('m0','TOP'),('m1','BOTTOM')))
    for i in range(70):
        module=base.modules[0]
        joints=dict(module.joints);joints['pan.position_rad']=i*2*math.pi
        f=replace(base,time_s=float(i),modules=(replace(module,joints=tuple(joints.items())),*base.modules[1:]))
        r=m.step(f,structural_score=0.,decision_opportunity=True,active_goals={'g':goal})
    assert r.terminate
