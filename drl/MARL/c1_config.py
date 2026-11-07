"""C1 scientific settings, geometry-derived limits and calibrated event costs."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.config.physics import (SMORES_EP_CHARACTERISTIC_LENGTH_M as BODY_LENGTH,
    SMORES_EP_MAX_LAND_SPEED_M_S as LAND_SPEED,SMORES_DOF_SIMULATION_MAX_SPEED_RAD_S as INTERNAL_SPEED)
from smores_ep.docking.model import DockingThresholds
from drl.MARL.c1_structure import PhysicalBounds

ROOT=Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class C1Settings:
    values: dict
    path: str
    sha256: str

    def __getattr__(self,name):
        try:return self.values[name]
        except KeyError:raise AttributeError(name) from None

    @classmethod
    def load(cls,path=None):
        path=Path(path or ROOT/'configs/marl/c1_assembly_probe.json')
        raw=path.read_bytes();values=json.loads(raw)
        if values.get('schema')!='mssr.c1.assembly_probe.v1':raise ValueError('Unsupported C1 settings schema')
        for name,value in values.items():
            if isinstance(value,(int,float)) and (not math.isfinite(value) or value<=0):raise ValueError(f'Invalid C1 setting {name}')
            if name.endswith('_fraction') and not 0<value<=1:raise ValueError(f'Invalid C1 fraction {name}')
        if values['probe_min_duration_s']>=values['probe_max_duration_s']:raise ValueError('Invalid probe duration bounds')
        if any(not math.isfinite(v) for v in values['weights'].values()):raise ValueError('Nonfinite C1 reward coefficient')
        mode=values.get('episode_mode')
        if mode not in (None,'guided','discovery'):raise ValueError('Invalid C1 episode_mode')
        shaping=values.get('target_graph_shaping_mode','off')
        if shaping not in ('off','temporary','potential'):raise ValueError('Invalid target shaping mode')
        if mode=='guided':
            if shaping!='potential':raise ValueError('Guided phase C requires potential shaping')
            if any(values['weights'].get(k,0)!=0 for k in ('reward_structural_progress','reward_valid_docking')):
                raise ValueError('Guided reward must not pay generic connectivity bonuses')
        if mode=='discovery' and (shaping!='off' or any(v for k,v in values['weights'].items() if k.startswith('reward_target_'))):
            raise ValueError('Discovery must disable target reward')
        if shaping=='potential':
            terms=[values['weights'].get(k) for k in ('reward_target_graph_progress','reward_target_graph_discount','reward_target_graph_incompatibility')]
            if None in terms or len(set(terms))!=1 or terms[0]<=0:
                raise ValueError('Potential reward components require one shared positive scale')
        for section in ('stagnation','redundancy'):
            options=values.get(section,{})
            if not isinstance(options,dict):raise ValueError(f'Invalid {section} settings')
            if 'enabled' in options and type(options['enabled']) is not bool:
                raise ValueError(f'Invalid {section} enabled flag')
            for key,value in options.items():
                numeric=key.endswith(('_s','_m','_rad')) or key in ('orientation_quantization','progress_epsilon')
                if numeric and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0):
                    raise ValueError(f'Invalid {section} {key}')
        s=values.get('stagnation',{})
        if s.get('enabled'):
            if not 0<s.get('strategic_window_s',20)<=s.get('global_window_s',25):raise ValueError('Invalid stagnation windows')
            if type(s.get('min_decision_opportunities',3)) is not int or s.get('min_decision_opportunities',3)<1:
                raise ValueError('Invalid recovery opportunity count')
        return cls(values,str(path.resolve()),hashlib.sha256(raw).hexdigest())

    @property
    def jitter_m(self):return self.jitter_contact_multiplier*DockingThresholds().lateral_offset_tolerance_m

    @property
    def controlled_speed(self):return self.controlled_linear_speed_multiplier*LAND_SPEED


def physical_bounds(course,settings,owner_ids=()):
    boxes=[b for b in course.support_boxes if b.collidable]
    margin=settings.workspace_margin_body_lengths*BODY_LENGTH
    # Rotated box bounding envelope uses all four physical XY corners.
    xs=[];ys=[];heights=[];bottoms=[]
    for box in boxes:
        yaw=math.radians(float(box.yaw_deg));c,s=abs(math.cos(yaw)),abs(math.sin(yaw))
        hx=(c*box.size_xyz_m[0]+s*box.size_xyz_m[1])/2
        hy=(s*box.size_xyz_m[0]+c*box.size_xyz_m[1])/2
        xs.extend((box.center_xyz_m[0]-hx,box.center_xyz_m[0]+hx))
        ys.extend((box.center_xyz_m[1]-hy,box.center_xyz_m[1]+hy))
        pitch=math.radians(float(box.pitch_deg))
        half_z=(abs(math.sin(pitch))*box.size_xyz_m[0]+abs(math.cos(pitch))*box.size_xyz_m[2])/2
        heights.append(box.center_xyz_m[2]+half_z);bottoms.append(box.center_xyz_m[2]-half_z)
    if not boxes:raise ValueError('C1 requires measured support geometry')
    zmin=min(bottoms)-settings.lost_height_margin_body_lengths*BODY_LENGTH
    zmax=max(heights)+(8+2)*BODY_LENGTH
    diagnostic_v=LAND_SPEED+math.sqrt(2*9.81*(zmax-zmin))
    return PhysicalBounds((min(xs)-margin,max(xs)+margin,min(ys)-margin,max(ys)+margin),zmin,zmax,
        diagnostic_v,INTERNAL_SPEED+diagnostic_v/SmoresGeometry().wheel_radius_m,
        settings.controlled_speed,INTERNAL_SPEED+LAND_SPEED/BODY_LENGTH,tuple(owner_ids))


def inherited_event_weights(domain,path=None):
    from drl.MARL.reward_config import RewardConfig
    config=RewardConfig.load(path or ROOT/'logs/appo_smoke/calibration_validated_train/reward_config.initial.json')
    terms=config[domain]['terms']
    return {f'penalty_{name}':float(terms[name]['weight'])/float(terms[name]['scale']) for name in ('conflict','invalid','time')}
