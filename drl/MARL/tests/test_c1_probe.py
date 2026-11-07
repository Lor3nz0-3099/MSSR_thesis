from dataclasses import replace
import pytest
from drl.MARL.c1_config import C1Settings
from drl.MARL.c1_probe import ProbeGeometry,probe_metrics
from drl.MARL.c1_runner import make_course
from drl.MARL.tests.test_c1_structure import frame


def move(f,dx=0.,dz=0.,dy=0.):
    return replace(f,time_s=f.time_s+1.,modules=tuple(replace(m,
        position=(m.position[0]+dx,m.position[1]+dy,m.position[2]+dz),
        connectors=tuple((port,(xyz[0]+dx,xyz[1]+dy,xyz[2]+dz)) for port,xyz in m.connectors)) for m in f.modules))


def test_navigation_net_path_progress_not_noise_or_high_speed():
    cfg=C1Settings.load();geo=ProbeGeometry('navigation',make_course('navigation',7017,'train'),cfg)
    anchor=frame()
    assert probe_metrics(geo,anchor,move(anchor,dx=.25)).passed_geometry
    assert not probe_metrics(geo,anchor,move(anchor,dx=.001)).passed_geometry
    assert not probe_metrics(geo,anchor,replace(move(anchor,dx=.25),peak_linear_speed=2.)).safe


def test_stairs_engagement_requires_real_vertical_and_longitudinal_progress():
    cfg=C1Settings.load();geo=ProbeGeometry('stairs',make_course('stairs',12005,'train'),cfg)
    anchor=move(frame(),dx=.65)
    assert probe_metrics(geo,anchor,move(anchor,dx=.1,dz=.02)).passed_geometry
    assert not probe_metrics(geo,anchor,move(anchor,dx=.1,dz=.001)).passed_geometry
    assert not probe_metrics(geo,anchor,move(anchor,dx=.1,dz=-.3)).safe


def test_gap_engagement_progress_is_not_falling_progress():
    geo=ProbeGeometry('gap',make_course('gap',16022,'train'),C1Settings.load())
    anchor=move(frame(),dx=.59)
    after=move(anchor,dx=.04)
    assert probe_metrics(geo,anchor,after).passed_geometry
    assert not probe_metrics(geo,anchor,move(anchor,dx=.001)).passed_geometry
    assert not probe_metrics(geo,anchor,move(anchor,dx=.08,dz=-.2)).safe
    assert not probe_metrics(geo,anchor,move(anchor,dx=.04,dz=-.01)).safe


def test_manipulation_uses_any_connector_controlled_approach_not_expert_role():
    course=make_course('manipulation',20004,'train');geo=ProbeGeometry('manipulation',course,C1Settings.load())
    normal=course.press_direction_world_xy;center=course.button_center_xyz_m
    def near(distance):
        point=(center[0]-normal[0]*(.02+distance),center[1]-normal[1]*(.02+distance),center[2])
        f=frame();return replace(f,modules=(replace(f.modules[0],connectors=(('LEFT',point),)),*f.modules[1:]))
    anchor=near(.09);after=near(.04)
    assert probe_metrics(geo,anchor,after).passed_geometry
    assert not probe_metrics(geo,anchor,near(.089)).passed_geometry
    assert not probe_metrics(geo,anchor,replace(after,peak_linear_speed=2.)).safe


def test_geometry_normalization_scales_with_task_not_topology():
    cfg=C1Settings.load();course=make_course('gap',16022,'train')
    a=ProbeGeometry('gap',course,cfg)
    b=ProbeGeometry('gap',replace(course,gap_width_m=course.gap_width_m*1.2),cfg)
    assert b.thresholds['progress_m']==pytest.approx(1.2*a.thresholds['progress_m'])


@pytest.mark.parametrize('task,seed,offset,dx,dz',[('stairs',12005,.65,.1,.02),('gap',16022,.59,.04,0.)])
def test_progress_beside_obstacle_is_not_functional_engagement(task,seed,offset,dx,dz):
    geo=ProbeGeometry(task,make_course(task,seed,'train'),C1Settings.load())
    anchor=move(frame(),dx=offset,dy=1.)
    assert not probe_metrics(geo,anchor,move(anchor,dx=dx,dz=dz)).passed_geometry
