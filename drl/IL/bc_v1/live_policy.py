"""SMORES live inference composition of frozen BC-v1 components, never a trainer.

No source dataset, split sample, route controller or simulator is opened here.
The generic temporal controller remains in live_controller without torch/hardware.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch

from . import bc_v1_contract as c, normalization as f2
from .action_decoder import decode_actions, DispatchAvailability, ExecutionSemantics, ResourceStatus
from .action_heads import load_action_head_config
from .dataset import _hash
from .feature_schema import load_feature_schema, feature_schema_fingerprint
from .live_controller import RuntimeCommand
from .live_observation import encode_smores_snapshot,prepare_live_observation
from .model import BCGNN,load_model_config
from .smores_adapter import SMORESAdapter
from .source_frame_bridges import BridgeResolution,SourceFrameBridgeBatch
from .task_context import build_task_view
from .tensorization import load_tensor_layout
from .training import CHECKPOINT_SCHEMA,move_batch_to_device

K1_SHA='35737c025e7fa53492cb12e59ccf4c4f3f6210a6d6f0882c245bf1847975d06d'


def navigation_context(course,task_config):
    """Expose actual static course geometry and physical final goal, never route."""
    physical=course.get('physical_track',{})
    goal=physical.get('goal_xyyaw')
    if not isinstance(goal,(list,tuple)) or len(goal)!=3:
        raise ValueError('Missing explicit physical goal; no nominal route fallback')
    safe={'frame_id':'world','current_goal':{'xyyaw':list(goal)}}
    for key in ('cone_centers_xy_m','cone_radius_m','cone_height_m','corridor_width_m','platform_bounds_xy_m'):
        if key in physical:safe[key]=physical[key]
    if 'collision_boxes' in course:
        safe['collision_boxes']=[{k:box[k] for k in ('center_xyz_m','size_xyz_m','yaw_deg','pitch_deg') if k in box}
            for box in course['collision_boxes']]
    context,observation=build_task_view({'task_type':'navigation'},{'course':safe},task_config)
    if observation.features:raise ValueError('Unexpected navigation dynamic context')
    return context


def smores_claims(choice):
    """Logical backend claims only. Final authority remains the native owner."""
    if choice.resource is not None:return (choice.resource,)
    params={p.name:p.value for p in choice.parameters}
    if choice.action_id=='gravity_settle':
        members=params.get('passive_module_ids',choice.participants)
        return tuple(c.ResourceRef(m,r) for m in members for r in ('locomotion','internal_motion'))
    if len(choice.connectors)!=2:raise ValueError('Unsupported structural endpoints')
    first,second=choice.connectors
    claims=list(choice.connectors)
    if choice.action_id in ('align_faces','dock'):
        claims.extend(c.ResourceRef(p.module,'locomotion') for p in (first,second))
    if choice.action_id=='align_faces':
        claims.extend(c.ResourceRef(p.module,'internal_motion') for p in (first,second) if p.connector_id=='TOP')
    return tuple(dict.fromkeys(claims))


def decode_runtime_action(adapter,choice):
    # Scope is a set in G2/L1a; endpoint order is already explicit in connectors.
    # Preserve that exact choice while adapting its serialization convention.
    if choice.connectors:
        ordered=tuple(port.module for port in choice.connectors)
        if set(choice.participants)!=set(ordered):raise ValueError('Contradictory structural participants')
        choice=replace(choice,participants=ordered)
    return adapter.decode_action(choice)


def claim_key(ref):
    if isinstance(ref,c.ResourceRef):return f'{ref.resource_id}:{ref.module.module_id}'
    return f'connector:{ref.module.module_id}:{ref.connector_id}'


def _finite_tree(value):
    if isinstance(value,dict):
        for v in value.values():_finite_tree(v)
    elif isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite model output')


class LivePolicy:
    def __init__(self,model,settings,graph_builder,context_provider,*,device='cpu',event_sink=None):
        self.model=model.eval();self.settings=settings;self.builder=graph_builder
        self.adapter=SMORESAdapter(settings['spec']);self.context_provider=context_provider
        self.device=device;self.event_sink=event_sink;self.forward_count=0;self.last_observation=None
        self.last_decoded=();self.timings=[];self.last_candidate_domain={}

    def _event(self,event,**fields):
        if self.event_sink is not None:self.event_sink(dict(event=event,wall_monotonic=time.monotonic(),**fields))

    @torch.inference_mode()
    def __call__(self,snapshot,feedback):
        started=time.monotonic()
        state=encode_smores_snapshot(snapshot.payload,adapter=self.adapter,graph_builder=self.builder)
        if state.stamp!=snapshot.stamp:raise ValueError('Physical stamp changed at L1b boundary')
        view=prepare_live_observation(state,context=self.context_provider(snapshot.payload),
            observation=c.ObservationBundle(),bridges=SourceFrameBridgeBatch((BridgeResolution(
                'world','world','current-live',None,True,'current publisher WORLD contract',None,()),),
                (('provider_id','smores-current-world'),('revision','1'))),**self.settings)
        self.last_observation=view
        batch=move_batch_to_device(view.batch,self.device)
        self._event('live_preprocessed',stamp=snapshot.stamp)
        preprocessed=time.monotonic()
        route=batch['routing']['samples'][0];busy=set(feedback.get('busy',()))
        states=tuple((r,ResourceStatus.BUSY if claim_key(r) in busy else ResourceStatus.FREE)
            for r in (*route['resource_refs'],*route['connector_refs']))
        availability=DispatchAvailability(states,
            tuple(claim_key(r.resource_ref) not in busy for r in route['resource_candidates']),
            (True,)*len(route['structural_bindings']))
        self.last_candidate_domain=dict(resources=len(route['resource_candidates']),structural=len(route['structural_bindings']),
            structural_actions=sorted({a.action_id for a,_ in route['structural_bindings']}))
        self.forward_count+=1;prediction=self.model(batch['inputs']);_finite_tree(prediction)
        if str(self.device).startswith('cuda'):torch.cuda.synchronize()
        forwarded=time.monotonic();self._event('model_forward',stamp=snapshot.stamp)
        semantics={action.action_id:(ExecutionSemantics.HOLD_UNTIL_EXPIRY if action.kind=='continuous'
            else ExecutionSemantics.COMPLETION_DRIVEN) for action in self.settings['spec'].actions}
        decoded=decode_actions(prediction,batch['inputs'],batch['routing'],spec=self.settings['spec'],
            normalizers=self.settings['normalizers'],availability=(availability,),execution_semantics=semantics,
            configured_arguments={},claim_resolver=smores_claims)
        self.last_decoded=decoded;commands=[]
        for row in decoded:
            payload=decode_runtime_action(self.adapter,row.choice)
            resource=row.choice.resource
            key=(row.choice.initiator.module_id,resource.resource_id if resource is not None else 'structural')
            commands.append(RuntimeCommand(key,payload,tuple(claim_key(r) for r in row.claims),
                row.execution is ExecutionSemantics.HOLD_UNTIL_EXPIRY))
        end=time.monotonic()
        self.timings.append(dict(stamp=snapshot.stamp,preprocessing_s=preprocessed-started,
            model_s=forwarded-preprocessed,decoder_s=end-forwarded,total_s=end-started))
        self._event('actions_decoded',actions=[x.payload for x in commands],
            candidate_domain=self.last_candidate_domain,**self.timings[-1])
        return tuple(commands)


def load_frozen_policy(root,checkpoint,graph_builder,*,device=None,event_sink=None):
    root,checkpoint=Path(root),Path(checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=K1_SHA:raise ValueError('Frozen checkpoint SHA mismatch')
    # Training's loader requires an optimizer; inference intentionally loads only
    # the same BCGNN state dict/schema and never instantiates a training optimizer.
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['schema_version']!=CHECKPOINT_SCHEMA or payload['step']!=6120 or payload['summary']['epoch']!=5:
        raise ValueError('Frozen K1 checkpoint metadata mismatch')
    cfg=root/'configs/bc_v1';identity=payload['identity']
    spec=c.load_robot_spec(cfg/'smores_ep.robot.json');schema=load_feature_schema(cfg/'features.json')
    normalization=f2.load_normalization_policy(cfg/'normalization.json',schema)
    path=root/'artifacts/bc_v1/normalization/f2_v1_optimized/normalizers.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=identity['normalizer_sha']:raise ValueError('Frozen F2 SHA mismatch')
    stats=f2.read_normalizers(path,expected_metadata=json.loads(path.read_text())['metadata'])
    layout=load_tensor_layout(cfg/'tensor_layout.v1.json')
    h1=load_model_config(cfg/'gnn_v1.json');h2=load_action_head_config(cfg/'action_heads_v1.json')
    for key,actual in [('robot_spec_fingerprint',c.robot_spec_fingerprint(spec)),
            ('feature_schema_fingerprint',feature_schema_fingerprint(schema)),('layout_fingerprint',layout['fingerprint']),
            ('f2_fingerprint',f2.normalizer_fingerprint(stats)),('h1_config_fingerprint',_hash(h1)),('h2_config_fingerprint',_hash(h2))]:
        if identity[key]!=actual:raise ValueError('Frozen identity mismatch: '+key)
    device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=BCGNN(spec,schema,h1,action_head_config=h2).to(device)
    model.load_state_dict(payload['model'],strict=True);model.eval();model.requires_grad_(False)
    settings=dict(spec=spec,feature_schema=schema,normalization_policy=normalization,normalizers=stats,tensor_layout=layout)
    task_config=json.loads((cfg/'task_context.json').read_text())
    return LivePolicy(model,settings,graph_builder,lambda raw:navigation_context(raw['state']['course'],task_config),
        device=device,event_sink=event_sink)
