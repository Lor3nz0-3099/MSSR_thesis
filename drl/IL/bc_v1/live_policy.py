"""SMORES live inference composition of frozen BC-v1 components, never a trainer.

No source dataset, split sample, route controller or simulator is opened here.
The generic temporal controller remains in live_controller without torch/hardware.
"""
from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import torch

from . import bc_v1_contract as c, normalization as f2
from .action_decoder import decode_actions, DispatchAvailability, ExecutionSemantics, ResourceStatus
from .action_heads import load_action_head_config
from .dataset import _hash
from .feature_schema import load_feature_schema, feature_schema_fingerprint
from .live_controller import RuntimeCommand,RuntimeDecision
from .live_observation import encode_smores_snapshot,prepare_live_observation
from .model import BCGNN,load_model_config
from .smores_adapter import SMORESAdapter
from .source_frame_bridges import BridgeResolution,SourceFrameBridgeBatch
from .task_context import build_task_view
from .tensorization import load_tensor_layout
from .training import CHECKPOINT_SCHEMA,move_batch_to_device

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
    def __init__(self,model,settings,graph_builder,context_provider,*,device='cpu',event_sink=None,
                 checkpoint_provenance=None,decode_policy=None):
        self.model=model.eval();self.settings=settings;self.builder=graph_builder
        self.adapter=SMORESAdapter(settings['spec']);self.context_provider=context_provider
        self.device=device;self.event_sink=event_sink;self.forward_count=0;self.last_observation=None
        self.last_decoded=();self.timings=[];self.last_candidate_domain={}
        # Provenance stays outside settings (which enter physical preprocessing).
        self.checkpoint_provenance=dict(checkpoint_provenance or {})
        self.decode_policy=decode_policy
        if decode_policy is not None:
            contract=getattr(getattr(model,'action_heads',None),'resource_contract',None)
            if contract is None:raise ValueError('Decode policy requires resource v3 contract')
            decode_policy.validate_contract(contract)
            self.checkpoint_provenance.update(decode_policy_fingerprint=decode_policy.fingerprint,
                decode_policy=decode_policy.document)

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
        contract=getattr(getattr(self.model,'action_heads',None),'resource_contract',None)
        if contract is not None:
            from .resource_decisions import outcome_inputs
            route=view.batch['routing']['samples'][0]
            view.batch['inputs']['resource_candidates'].update(outcome_inputs({'routing':route},contract))
            route.update(resource_contract_fingerprint=contract.fingerprint,resource_contract=contract.document)
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
        withdrawals=[]
        decoded=decode_actions(prediction,batch['inputs'],batch['routing'],spec=self.settings['spec'],
            normalizers=self.settings['normalizers'],availability=(availability,),execution_semantics=semantics,
            configured_arguments={},claim_resolver=smores_claims,resource_contract=contract,decode_policy=self.decode_policy,
            inactive_sink=lambda sample,ref:withdrawals.append((ref.module.module_id,ref.resource_id)))
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
            decode_policy_fingerprint=self.decode_policy.fingerprint if self.decode_policy is not None else None,
            candidate_domain=self.last_candidate_domain,**self.timings[-1])
        return RuntimeDecision(tuple(commands),tuple(withdrawals)) if contract is not None else tuple(commands)


def _checkpoint_metadata(payload):
    if not isinstance(payload,dict) or payload.get('schema_version')!=CHECKPOINT_SCHEMA:
        raise ValueError('Unsupported checkpoint schema')
    if any(not isinstance(payload.get(k),dict) for k in ('identity','config','summary','model')):
        raise ValueError('Missing checkpoint metadata/identity/model contract')
    epoch=payload['summary'].get('epoch');step=payload.get('step')
    if type(epoch) is not int or epoch<0 or type(step) is not int or step<0:
        raise ValueError('Invalid checkpoint epoch/global step metadata')
    if 'global_step' in payload['summary'] and (
            type(payload['summary']['global_step']) is not int or payload['summary']['global_step']!=step):
        raise ValueError('Contradictory checkpoint global step metadata')
    identity=payload['identity']
    required=('robot_spec_fingerprint','feature_schema_fingerprint','layout_fingerprint',
        'normalizer_sha','f2_fingerprint','h1_config_fingerprint','h2_config_fingerprint',
        'training_config_fingerprint')
    for key in required:
        if type(identity.get(key)) is not str or not identity[key]:
            raise ValueError('Missing checkpoint identity: '+key)
    if identity['training_config_fingerprint']!=_hash(payload['config']):
        raise ValueError('Frozen identity mismatch: training_config_fingerprint')
    return epoch,step


def _head_contract(cfg,identity):
    """Explicit supported head contracts, selected by content identity, never path.

    Legacy v1 predates structural_decision_schema and has no abstention scorer.
    This isolated compatibility rule also accepts future v1 weights with the
    same contract; it does not pin K1's SHA, epoch or training hyperparameters.
    """
    profiles=(('action_heads_v1.json','bc_v1.legacy_v1',None),
        ('action_heads_no_structural_v2.json','bc_v1.no_structural_v2','mssr.bc_v1.structural_decisions.v2'),
        ('action_heads_resource_v3.json','bc_v1.resource_v3','mssr.bc_v1.structural_decisions.v2'))
    for filename,profile,decision_schema in profiles:
        h2=load_action_head_config(cfg/filename)
        if _hash(h2)==identity['h2_config_fingerprint']:
            if identity.get('structural_decision_schema')!=decision_schema:
                raise ValueError('Incompatible structural_decision_schema for '+profile)
            if profile=='bc_v1.resource_v3':
                from .resource_decisions import ResourceDecisionContract,VERSION
                spec=c.load_robot_spec(cfg/'smores_ep.robot.json')
                contract=ResourceDecisionContract.from_dict(h2['resource_contract'],spec)
                if (identity.get('resource_decision_schema')!=VERSION or identity.get('resource_contract_fingerprint')!=contract.fingerprint):
                    raise ValueError('Incompatible resource decision contract identity')
            elif identity.get('resource_contract_fingerprint') or identity.get('resource_decision_schema'):
                raise ValueError('Legacy profile cannot carry resource v3 identity')
            return h2,profile
    raise ValueError('Unsupported checkpoint head contract: h2_config_fingerprint')


def load_frozen_policy(root,checkpoint,graph_builder,*,device=None,event_sink=None,
                       expected_checkpoint_sha256=None):
    """Load a trusted local BC-v1 checkpoint; optional SHA pins its provenance.

    Authenticate and deserialize the same bytes, then validate on CPU before any
    device transfer. This never opens dataset payloads or builds an optimizer.
    """
    root,checkpoint=Path(root),Path(checkpoint).resolve()
    blob=checkpoint.read_bytes();digest=hashlib.sha256(blob).hexdigest()
    if expected_checkpoint_sha256 is not None:
        if (not isinstance(expected_checkpoint_sha256,str) or len(expected_checkpoint_sha256)!=64
                or any(c not in '0123456789abcdef' for c in expected_checkpoint_sha256.lower())):
            raise ValueError('Invalid expected checkpoint SHA256')
        if digest!=expected_checkpoint_sha256.lower():raise ValueError('Frozen checkpoint SHA mismatch')
    try:payload=torch.load(BytesIO(blob),map_location='cpu',weights_only=False)
    except Exception as exc:raise ValueError('Invalid checkpoint format') from exc
    epoch,step=_checkpoint_metadata(payload)
    cfg=root/'configs/bc_v1';identity=payload['identity']
    spec=c.load_robot_spec(cfg/'smores_ep.robot.json');schema=load_feature_schema(cfg/'features.json')
    normalization=f2.load_normalization_policy(cfg/'normalization.json',schema)
    path=root/'artifacts/bc_v1/normalization/f2_v1_optimized/normalizers.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=identity['normalizer_sha']:
        raise ValueError('Frozen identity mismatch: normalizer_sha')
    metadata=json.loads(path.read_text())['metadata']
    if metadata.get('policy_fingerprint')!=f2.policy_fingerprint(normalization,schema):
        raise ValueError('Frozen identity mismatch: policy_fingerprint')
    stats=f2.read_normalizers(path,expected_metadata=metadata)
    layout=load_tensor_layout(cfg/'tensor_layout.v1.json')
    h1=load_model_config(cfg/'gnn_v1.json');h2,profile=_head_contract(cfg,identity)
    for key,actual in [('robot_spec_fingerprint',c.robot_spec_fingerprint(spec)),
            ('feature_schema_fingerprint',feature_schema_fingerprint(schema)),('layout_fingerprint',layout['fingerprint']),
            ('f2_fingerprint',f2.normalizer_fingerprint(stats)),('h1_config_fingerprint',_hash(h1)),('h2_config_fingerprint',_hash(h2))]:
        if identity[key]!=actual:raise ValueError('Frozen identity mismatch: '+key)
    model=BCGNN(spec,schema,h1,action_head_config=h2)
    expected=model.state_dict()
    for key,value in payload['model'].items():
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite/invalid checkpoint state_dict tensor: '+str(key))
        if key in expected and value.dtype!=expected[key].dtype:
            raise ValueError('Incompatible checkpoint state_dict dtype: '+key)
    try:model.load_state_dict(payload['model'],strict=True)
    except RuntimeError as exc:raise ValueError('Incompatible checkpoint state_dict: '+str(exc)) from exc
    device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=model.to(device);model.eval();model.requires_grad_(False)
    settings=dict(spec=spec,feature_schema=schema,normalization_policy=normalization,normalizers=stats,tensor_layout=layout)
    task_config=json.loads((cfg/'task_context.json').read_text())
    provenance=dict(checkpoint_path=str(checkpoint),checkpoint_sha256=digest,
        checkpoint_epoch=epoch,checkpoint_global_step=step,checkpoint_profile=profile,
        checkpoint_schema=payload['schema_version'],model_schema=h1['schema_version'],
        action_heads_schema=h2['schema_version'],resource_decision_schema=identity.get('resource_decision_schema'),
        resource_contract_fingerprint=identity.get('resource_contract_fingerprint'),structural_decision_schema=identity.get('structural_decision_schema'),
        checkpoint_training_config_schema=payload['config'].get('schema_version'),
        checkpoint_identity=dict(identity),state_dict_strict=True)
    decode_policy=None
    if profile=='bc_v1.resource_v3':
        from .resource_decisions import load_decode_policy
        decode_policy=load_decode_policy(cfg/'resource_decode_policy_v1.json',model.action_heads.resource_contract,spec)
    return LivePolicy(model,settings,graph_builder,lambda raw:navigation_context(raw['state']['course'],task_config),
        device=device,event_sink=event_sink,checkpoint_provenance=provenance,decode_policy=decode_policy)
