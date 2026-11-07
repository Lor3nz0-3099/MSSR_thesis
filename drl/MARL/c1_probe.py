"""Short task-specific physical metrics, with no desired robot or action script."""
from dataclasses import dataclass
import math

from drl.MARL.c1_config import physical_bounds
from drl.MARL.c1_structure import norm
from drl.MARL.phase1_reward import _project_to_centerline,_route_geometry
from drl.MARL.stairs_reward import _local_support_floor_height
from drl.MARL.gap_reward import _bank_top_z_m,_is_supported_height,_inside_landing_xy
from smores_ep.config.geometry import SmoresGeometry
from smores_ep.docking.model import DockingThresholds
from smores_ep.isaac.obstacle_course import BUTTON_PLUNGER_TANGENT_M,BUTTON_PLUNGER_HEIGHT_M,BUTTON_PLUNGER_DEPTH_M,BUTTON_PLUNGER_STROKE_M


@dataclass(frozen=True)
class ProbeGeometry:
    task: str
    course: object
    settings: object

    @property
    def thresholds(self):
        c,s=self.course,self.settings
        if self.task=='navigation':return dict(progress_m=max(s.jitter_m,s.navigation_progress_fraction*_route_geometry(c)[1]))
        if self.task=='stairs':return dict(progress_m=max(s.jitter_m,s.stairs_tread_progress_fraction*c.tread_depths_m[0]),
            vertical_m=max(s.vertical_jitter_wheel_radius_fraction*SmoresGeometry().wheel_radius_m,s.stairs_rise_progress_fraction*abs(c.step_rises_m[0])))
        if self.task=='gap':return dict(progress_m=max(s.jitter_m,s.gap_progress_fraction*c.gap_width_m),engagement_x_m=c.near_edge_x_m+s.gap_progress_fraction*c.gap_width_m)
        if self.task=='manipulation':return dict(progress_m=max(s.jitter_m,s.manipulation_approach_face_fraction*BUTTON_PLUNGER_TANGENT_M),
            proximity_m=s.manipulation_proximity_face_fraction*BUTTON_PLUNGER_TANGENT_M,preload_m=s.manipulation_preload_stroke_fraction*BUTTON_PLUNGER_STROKE_M)
        raise ValueError('Unknown C1 probe task')


@dataclass(frozen=True)
class ProbeMeasurement:
    progress_fraction: float
    passed_geometry: bool
    safe: bool
    metric: dict
    thresholds: dict


def _button_distance(geo,frame):
    c=geo.course;n=c.press_direction_world_xy;t=(-n[1],n[0])
    front=(c.button_center_xyz_m[0]-n[0]*BUTTON_PLUNGER_DEPTH_M/2,
           c.button_center_xyz_m[1]-n[1]*BUTTON_PLUNGER_DEPTH_M/2,c.button_center_xyz_m[2])
    best=math.inf
    for module in frame.modules:
        for _port,point in module.connectors:
            d=tuple(point[k]-front[k] for k in range(3));normal=-(d[0]*n[0]+d[1]*n[1])
            if normal < -DockingThresholds().normal_contact_tolerance_m:continue
            lateral=max(0.,abs(d[0]*t[0]+d[1]*t[1])-BUTTON_PLUNGER_TANGENT_M/2)
            vertical=max(0.,abs(d[2])-BUTTON_PLUNGER_HEIGHT_M/2)
            best=min(best,math.sqrt(max(0.,normal)**2+lateral*lateral+vertical*vertical))
    return best


def probe_metrics(geo,anchor,current):
    thresholds=geo.thresholds;c=geo.course;s=geo.settings
    bounds=physical_bounds(c,s)
    safe=(not current.pathological_collision_owner_ids and current.evidence_complete
        and max(current.peak_linear_speed,*(norm(m.linear_velocity) for m in current.modules))<=bounds.stable_linear_speed
        and max(current.peak_angular_speed,*(norm(m.angular_velocity) for m in current.modules))<=bounds.stable_angular_speed)
    before={m.module_id:m.position for m in anchor.modules};after={m.module_id:m.position for m in current.modules}
    if set(before)!=set(after) or len(after)!=8:return ProbeMeasurement(0.,False,False,{'reason':'missing_module'},thresholds)
    mean=lambda numbers:sum(numbers)/8
    dx=mean(after[k][0]-before[k][0] for k in after)
    if geo.task=='navigation':
        delta=mean(_project_to_centerline(c,after[k][:2]).s_m-_project_to_centerline(c,before[k][:2]).s_m for k in after)
        fraction=delta/thresholds['progress_m'];passed=delta>=thresholds['progress_m']
        metric=dict(net_path_progress_m=delta)
    elif geo.task=='stairs':
        sign=1. if c.goal_height_m>c.start_height_m else -1.
        engaged=[k for k in after if after[k][0]>=c.first_transition_x_m+s.stairs_tread_progress_fraction*c.tread_depths_m[0]
            and after[k][0]<=c.first_transition_x_m+sum(c.tread_depths_m) and abs(after[k][1])<=c.width_m/2]
        vertical=max((sign*(after[k][2]-before[k][2]) for k in engaged),default=0.)
        radius=SmoresGeometry().wheel_radius_m
        supported=all(v[2]>=_local_support_floor_height(c,v[0],wheel_radius_m=radius)-radius for v in after.values())
        safe=safe and supported
        fraction=min(dx/thresholds['progress_m'],vertical/thresholds['vertical_m'])
        passed=bool(engaged) and dx>=thresholds['progress_m'] and vertical>=thresholds['vertical_m']
        metric=dict(net_ascent_axis_progress_m=dx,signed_vertical_progress_m=vertical,engaged_modules=engaged,vertical_direction=sign)
    elif geo.task=='gap':
        top=_bank_top_z_m(c);radius=SmoresGeometry().wheel_radius_m
        height_safe=all(_is_supported_height(v[2],support_center_z_m=top,wheel_radius_m=radius,tolerance_m=s.jitter_m) for v in after.values())
        bank_support=any((c.approach_start_x_m<=v[0]<=c.near_edge_x_m and abs(v[1])<=c.bank_width_m/2)
            or _inside_landing_xy(c,x_m=v[0],y_m=v[1]) for v in after.values())
        centroid_drop=mean(before[k][2]-after[k][2] for k in after)
        safe=safe and height_safe and bank_support and centroid_drop<=s.vertical_jitter_wheel_radius_fraction*radius
        low_y=max(-c.bank_width_m/2,c.landing_center_y_m-c.bank_width_m/2)
        high_y=min(c.bank_width_m/2,c.landing_center_y_m+c.bank_width_m/2)
        leading=max((v[0] for v in after.values() if low_y<=v[1]<=high_y),default=c.approach_start_x_m)
        approach_anchor=any(c.approach_start_x_m<=v[0]<=c.near_edge_x_m and abs(v[1])<=c.bank_width_m/2 for v in before.values())
        fraction=dx/thresholds['progress_m'];passed=approach_anchor and dx>=thresholds['progress_m'] and leading>=thresholds['engagement_x_m']
        metric=dict(net_crossing_progress_m=dx,leading_x_m=leading,bank_supported=bank_support,height_safe=height_safe,centroid_drop_m=centroid_drop,approach_anchor=approach_anchor)
    else:
        initial=_button_distance(geo,anchor);distance=_button_distance(geo,current)
        delta=initial-distance if math.isfinite(initial) and math.isfinite(distance) else 0.
        preload=max(0.,(current.button_depression_m or 0.)-(anchor.button_depression_m or 0.))
        fraction=max(delta/thresholds['progress_m'],preload/thresholds['preload_m'])
        passed=math.isfinite(distance) and distance<=thresholds['proximity_m'] and (delta>=thresholds['progress_m'] or preload>=thresholds['preload_m'])
        metric=dict(distance_reduction_m=delta,button_face_distance_m=distance if math.isfinite(distance) else None,preload_delta_m=preload)
    return ProbeMeasurement(max(0.,min(1.,fraction)),bool(passed and safe),bool(safe),metric,thresholds)
