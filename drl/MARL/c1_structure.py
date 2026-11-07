"""Generic measured physical assembly validity. No desired graph or policy code."""
from dataclasses import dataclass
import hashlib
import itertools
import json
import math


@dataclass(frozen=True)
class C1Module:
    module_id: str
    position: tuple[float,float,float]
    orientation: tuple = (0.,0.,0.,1.)
    linear_velocity: tuple = (0.,0.,0.)
    angular_velocity: tuple = (0.,0.,0.)
    joints: tuple = ()
    connectors: tuple = ()  # (physical connector ID, current world XYZ)
    joint_velocity_limits: tuple = ()
    joint_position_limits: tuple = ()


@dataclass(frozen=True)
class C1Connection:
    joint_id: str
    module_a: str
    connector_a: str
    module_b: str
    connector_b: str
    joint_valid: bool
    anchor_error_m: float = 0.
    anchor_error_rad: float = 0.

    @property
    def endpoints(self):return ((self.module_a,self.connector_a),(self.module_b,self.connector_b))


@dataclass(frozen=True)
class C1PrimitiveEvent:
    goal_id: str
    action: str
    state: str
    code: str
    owner: str


@dataclass(frozen=True)
class C1Frame:
    time_s: float
    modules: tuple[C1Module,...]
    connections: tuple[C1Connection,...]
    peak_linear_speed: float = 0.
    peak_angular_speed: float = 0.
    topology_changed: bool = False
    evidence_complete: bool = True
    pathological_collision_owner_ids: tuple = ()
    primitive_events: tuple[C1PrimitiveEvent,...] = ()
    button_depression_m: float | None = None
    connection_events: tuple = ()
    interval_valid: bool = True


@dataclass(frozen=True)
class PhysicalBounds:
    xy: tuple[float,float,float,float]
    z_min: float
    z_max: float
    explosion_linear_speed: float
    explosion_angular_speed: float
    stable_linear_speed: float
    stable_angular_speed: float
    expected_owner_ids: tuple = ()


@dataclass(frozen=True)
class StructureAssessment:
    instantaneous_valid: bool
    safe: bool
    reason: str
    component_sizes: tuple
    valid_connections: int
    connection_signature: tuple
    peak_linear_speed: float
    peak_angular_speed: float


def norm(v):return math.sqrt(sum(float(x)**2 for x in v))


def module_joint_status(module):
    """(complete, physically sane positions, controlled joint speeds, max speed)."""
    from smores_ep.config.geometry import SmoresGeometry
    from smores_ep.config.physics import SMORES_EP_MAX_WHEEL_SPEED_RAD_S,SMORES_DOF_SIMULATION_MAX_SPEED_RAD_S
    values=dict(module.joints)
    names=('left_wheel','right_wheel','pan','tilt')
    if len(values)!=len(module.joints) or any(f'{n}.{field}' not in values for n in names for field in ('position_rad','velocity_rad_s')):
        return False,False,False,0.
    limits={n:SMORES_EP_MAX_WHEEL_SPEED_RAD_S if 'wheel' in n else SMORES_DOF_SIMULATION_MAX_SPEED_RAD_S for n in names}
    limits.update(dict(module.joint_velocity_limits))
    if any(not math.isfinite(float(v)) or v<=0 for v in limits.values()):return False,False,False,0.
    geometry=SmoresGeometry()
    position_limits={'tilt':(geometry.tilt_min_rad,geometry.tilt_max_rad)}
    position_limits.update({name:(low,high) for name,low,high in module.joint_position_limits})
    # 1mrad is a numerical position tolerance (~0.057deg), far below the
    # existing 8/10deg physical alignment gates; it does not expand a target.
    sane=all(math.isfinite(low) and math.isfinite(high) and low<=high
        and low-1e-3<=values[f'{name}.position_rad']<=high+1e-3 for name,(low,high) in position_limits.items())
    speeds=[abs(values[f'{name}.velocity_rad_s']) for name in names]
    controlled=all(speed<=1.5*limits[name] for name,speed in zip(names,speeds))
    return True,sane,controlled,max(speeds)


def assess_structure(frame: C1Frame,bounds: PhysicalBounds):
    """Current validity; temporal stability is checked by the episode evaluator.

    Cycles are allowed. Physical joints are measured by the native adapter;
    proximity/contact-candidate edges are never counted as connections.
    """
    ids=[m.module_id for m in frame.modules]
    signature=tuple(sorted((c.joint_id,*sorted(c.endpoints)) for c in frame.connections))
    def result(valid,safe,reason,components=(),edges=0,linear=0.,angular=0.):
        return StructureAssessment(valid,safe,reason,components,edges,signature,linear,angular)
    if len(ids)!=8 or len(set(ids))!=8 or bounds.expected_owner_ids and set(ids)!=set(bounds.expected_owner_ids):
        return result(False,False,'missing_or_duplicate_module')
    values=[frame.time_s,frame.peak_linear_speed,frame.peak_angular_speed]
    for m in frame.modules:
        if len(m.position)!=3 or len(m.orientation)!=4 or len(m.linear_velocity)!=3 or len(m.angular_velocity)!=3:
            return result(False,False,'invalid_physical_state_shape')
        values.extend((*m.position,*m.orientation,*m.linear_velocity,*m.angular_velocity))
        values.extend(value for _name,value in m.joints)
        for _port,point in m.connectors:values.extend(point)
    values.extend(v for c in frame.connections for v in (c.anchor_error_m,c.anchor_error_rad))
    if frame.button_depression_m is not None:values.append(frame.button_depression_m)
    if not all(math.isfinite(float(v)) for v in values):return result(False,False,'nonfinite_physical_state')
    if frame.button_depression_m is not None:
        from smores_ep.isaac.obstacle_course import BUTTON_PLUNGER_STROKE_M
        from drl.MARL.manipulation_reward import BUTTON_DEPRESSION_NUMERICAL_TOLERANCE_M
        if not -BUTTON_DEPRESSION_NUMERICAL_TOLERANCE_M<=frame.button_depression_m<=BUTTON_PLUNGER_STROKE_M+BUTTON_DEPRESSION_NUMERICAL_TOLERANCE_M:
            return result(False,False,'invalid_button_stroke_state')
    if not frame.evidence_complete:return result(False,False,'missing_physical_evidence')
    joint_dynamic=False
    for module in frame.modules:
        ports=[name for name,_position in module.connectors]
        if len(ports)!=4 or set(ports)!={'LEFT','RIGHT','TOP','BOTTOM'}:
            return result(False,False,'missing_or_duplicate_connector_evidence')
        complete,sane,controlled,speed=module_joint_status(module)
        if not complete:return result(False,False,'missing_joint_evidence')
        if not sane:return result(False,False,'invalid_joint_state')
        if speed>bounds.explosion_angular_speed:return result(False,False,'physical_explosion')
        joint_dynamic=joint_dynamic or not controlled
    if any(abs(norm(m.orientation)-1)>1e-3 for m in frame.modules):return result(False,False,'invalid_orientation')
    if any(not(bounds.xy[0]<=m.position[0]<=bounds.xy[1] and bounds.xy[2]<=m.position[1]<=bounds.xy[3]
                   and bounds.z_min<=m.position[2]<=bounds.z_max) for m in frame.modules):
        return result(False,False,'module_outside_recoverable_workspace')
    linear=max(frame.peak_linear_speed,*(norm(m.linear_velocity) for m in frame.modules))
    angular=max(frame.peak_angular_speed,*(norm(m.angular_velocity) for m in frame.modules))
    if linear>bounds.explosion_linear_speed or angular>bounds.explosion_angular_speed:
        return result(False,False,'physical_explosion',linear=linear,angular=angular)
    adjacency={mid:set() for mid in ids};used=set();joint_ids=set();valid_edges=0;invalid=False
    for c in frame.connections:
        ports=c.endpoints
        good=(c.module_a in adjacency and c.module_b in adjacency and c.module_a!=c.module_b
            and all(p in ('LEFT','RIGHT','TOP','BOTTOM') for p in (c.connector_a,c.connector_b))
            and c.joint_valid and c.joint_id not in joint_ids and not used.intersection(ports))
        if good:
            adjacency[c.module_a].add(c.module_b);adjacency[c.module_b].add(c.module_a)
            valid_edges+=1
        else:invalid=True
        used.update(ports);joint_ids.add(c.joint_id)
    remaining=set(ids);sizes=[]
    while remaining:
        component={min(remaining)};pending=list(component)
        while pending:
            for peer in adjacency[pending.pop()]-component:
                component.add(peer);pending.append(peer)
        sizes.append(len(component));remaining-=component
    sizes=tuple(sorted(sizes,reverse=True))
    if invalid:reason='invalid_docking_constraint_or_connector_ownership'
    elif not frame.interval_valid:reason='transient_physical_invalidity'
    elif joint_dynamic or linear>bounds.stable_linear_speed or angular>bounds.stable_angular_speed:reason='excessive_dynamics'
    elif frame.pathological_collision_owner_ids:reason='pathological_collision'
    elif sizes!=(8,) or valid_edges<7:reason='disconnected_assembly'
    else:reason='valid'
    return result(reason=='valid',True,reason,sizes,valid_edges,linear,angular)


def canonical_signature(frame: C1Frame, *, include_ports=True):
    """Exact ID-independent graph canonicalization, bounded to eight modules.

    Used for discovery logging only. No known topology is consulted. Degree/
    port partitions reduce the <=8! label permutations; posture is separate.
    """
    ids=tuple(m.module_id for m in frame.modules)
    if len(ids)!=8 or len(set(ids))!=8:return None
    if not frame.connections:
        return hashlib.sha256(json.dumps([len(ids),[]],separators=(',',':')).encode()).hexdigest()
    incident={mid:[] for mid in ids}
    for c in frame.connections:
        if c.module_a in incident:incident[c.module_a].append(c.connector_a if include_ports else '*')
        if c.module_b in incident:incident[c.module_b].append(c.connector_b if include_ports else '*')
    groups={}
    for mid in ids:groups.setdefault(tuple(sorted(incident[mid])),[]).append(mid)
    ordered=[groups[key] for key in sorted(groups)]
    best=None
    for assignment in itertools.product(*(itertools.permutations(group) for group in ordered)):
        labels={mid:i for i,mid in enumerate(mid for group in assignment for mid in group)}
        edges=[]
        for c in frame.connections:
            if c.module_a not in labels or c.module_b not in labels:continue
            ends=((labels[c.module_a],c.connector_a if include_ports else '*'),
                  (labels[c.module_b],c.connector_b if include_ports else '*'))
            edges.append(tuple(sorted(ends)))
        encoded=json.dumps([len(ids),sorted(edges)],separators=(',',':'))
        if best is None or encoded<best:best=encoded
    return hashlib.sha256((best or '[]').encode()).hexdigest()
