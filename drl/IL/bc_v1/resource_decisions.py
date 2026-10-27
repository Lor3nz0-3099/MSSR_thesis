"""Versioned per-resource actor vocabulary, independent of robot hardware.

INACTIVE is no physical action. FIXED carries catalog physical values, never F2.
The physical RobotSpec and its parameterized candidate codec remain unchanged.
Compatibility enumerates the Cartesian product per module: O(product(domains)).
Small catalogs only; isolated here for a later solver replacement if needed.
"""
from dataclasses import dataclass
from itertools import product
import hashlib
import json
import math
from pathlib import Path
from .bc_v1_contract import robot_spec_fingerprint

VERSION='mssr.bc_v1.resource_decisions.v1'
INACTIVE,FIXED,PARAMETERIZED=0,1,2


@dataclass(frozen=True)
class Outcome:
    module_type_id: str
    resource_id: str
    outcome_id: str
    kind: int
    mode_id: str | None = None
    payload: tuple = ()


@dataclass(frozen=True)
class ResourceDecisionContract:
    serialized: str
    outcomes: tuple[Outcome,...]
    tag_rules: tuple
    forbidden_pairs: tuple

    @property
    def document(self):return json.loads(self.serialized)
    @property
    def fingerprint(self):return hashlib.sha256(self.serialized.encode()).hexdigest()

    @classmethod
    def from_dict(cls,doc,spec):
        if (set(doc)!={'schema_version','revision','robot_spec_fingerprint','fixed','tags','forbidden_tag_pairs'}
                or doc['schema_version']!=VERSION or not isinstance(doc['revision'],str) or not doc['revision']
                or doc['robot_spec_fingerprint']!=robot_spec_fingerprint(spec)):
            raise ValueError('Incompatible resource decision contract')
        modes={(t.type_id,r.resource_id,m.mode_id):m for t in spec.module_types for r in t.resources for m in r.modes}
        rows=[]
        for t in sorted(spec.module_types,key=lambda t:t.type_id):
            for r in sorted(t.resources,key=lambda r:r.resource_id):
                rows.append(Outcome(t.type_id,r.resource_id,'INACTIVE',INACTIVE))
                for f in doc['fixed']:
                    if set(f)!={'module_type_id','resource_id','outcome_id','mode_id','payload','evidence'}:raise ValueError('Invalid fixed outcome')
                    if (f['module_type_id'],f['resource_id'])!=(t.type_id,r.resource_id):continue
                    mode=modes.get((t.type_id,r.resource_id,f['mode_id']))
                    if (mode is None or mode.codec=='categorical' or not f['evidence'] or not f['outcome_id']
                            or len(f['payload'])!=mode.payload_size or any(type(v) not in (float,int) or not math.isfinite(v) for v in f['payload'])):
                        raise ValueError('Invalid fixed physical payload/evidence')
                    rows.append(Outcome(t.type_id,r.resource_id,f['outcome_id'],FIXED,f['mode_id'],tuple(float(v) for v in f['payload'])))
                rows.extend(Outcome(t.type_id,r.resource_id,m.mode_id,PARAMETERIZED,m.mode_id) for m in sorted(r.modes,key=lambda m:m.mode_id))
        if len({(r.module_type_id,r.resource_id,r.outcome_id) for r in rows})!=len(rows):raise ValueError('Duplicate outcomes')
        if len([r for r in rows if r.kind==FIXED])!=len(doc['fixed']):raise ValueError('Unknown fixed resource')
        tags=set()
        for row in doc['tags']:
            if (set(row)!={'module_type_id','resource_id','mode_id','tag','predicate','epsilon'}
                    or (row['module_type_id'],row['resource_id'],row['mode_id']) not in modes
                    or row['predicate'] not in ('always','any_nonzero') or not row['tag']
                    or type(row['epsilon']) not in (int,float) or not math.isfinite(row['epsilon']) or row['epsilon']<0):
                raise ValueError('Invalid compatibility tag')
            tags.add(row['tag'])
        for pair in doc['forbidden_tag_pairs']:
            if not isinstance(pair,list) or len(pair)!=2 or any(t not in tags for t in pair):raise ValueError('Invalid forbidden tags')
        rules=tuple(tuple((r['tag'],r['predicate'],r['epsilon']) for r in doc['tags']
            if (o.module_type_id,o.resource_id,o.mode_id)==(r['module_type_id'],r['resource_id'],r['mode_id'])) for o in rows)
        return cls(json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False),tuple(rows),rules,
            tuple(tuple(pair) for pair in doc['forbidden_tag_pairs']))

    def compatible(self,assignment):
        """Assignment entries are (semantic Outcome, physical payload)."""
        tags=set()
        for outcome,payload in assignment:
            if outcome.kind==INACTIVE:continue
            for tag,predicate,epsilon in self.tag_rules[self.outcomes.index(outcome)]:
                if predicate=='always' or any(abs(v)>epsilon for v in payload):tags.add(tag)
        return not any(set(pair)<=tags for pair in self.forbidden_pairs)


def load_resource_contract(path,spec):return ResourceDecisionContract.from_dict(json.loads(Path(path).read_text()),spec)


@dataclass(frozen=True)
class ResourceDecodePolicy:
    """Explicit decode-only rules, bound to an unchanged tensor contract.

    The archived contract still authenticates vocabulary, labels and routing.
    This separately identified policy reuses its generic compatibility compiler;
    it cannot replace fixed outcomes, physical modes or supervision semantics.
    Omission at the decoder preserves historical contract rules for replay.
    """
    serialized: str
    resource_contract_fingerprint: str
    compatibility: ResourceDecisionContract

    @property
    def document(self):return json.loads(self.serialized)
    @property
    def fingerprint(self):return hashlib.sha256(self.serialized.encode()).hexdigest()

    @classmethod
    def from_dict(cls,doc,contract,spec):
        if (set(doc)!={'schema_version','revision','resource_contract_fingerprint','tags','forbidden_tag_pairs'}
                or doc['schema_version']!='mssr.bc_v1.resource_decode_policy.v1'
                or not isinstance(doc['revision'],str) or not doc['revision']
                or doc['resource_contract_fingerprint']!=contract.fingerprint):
            raise ValueError('Incompatible resource decode policy/contract identity')
        compiled=ResourceDecisionContract.from_dict({**contract.document,
            'tags':doc['tags'],'forbidden_tag_pairs':doc['forbidden_tag_pairs']},spec)
        if compiled.outcomes!=contract.outcomes:raise ValueError('Decode policy changed outcome contract')
        return cls(json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False),contract.fingerprint,compiled)

    def validate_contract(self,contract):
        if contract.fingerprint!=self.resource_contract_fingerprint:
            raise ValueError('Decode policy resource contract fingerprint mismatch')

    def compatible(self,assignment):return self.compatibility.compatible(assignment)


def load_decode_policy(path,contract,spec):
    return ResourceDecodePolicy.from_dict(json.loads(Path(path).read_text()),contract,spec)


def select_joint(domains,scores,compatible):
    if any(not d for d in domains):raise ValueError('Empty feasible resource domain')
    best=None;value=-math.inf
    for choice in product(*domains):
        score=sum(float(scores[i]) for i in choice)
        if not math.isfinite(score):raise ValueError('Nonfinite resource outcome scores')
        if compatible(choice) and score>value:best,value=choice,score
    if best is None:raise ValueError('No compatible resource assignment')
    return best


def outcome_inputs(sample,contract):
    """Pure current-catalog routing: independent of labels, provenance and IDs."""
    import torch
    route=sample['routing'];rows=route['resource_candidates'];refs=route['resource_refs']
    catalog=[];physical=[];owners=[];kinds=[];ptr=[0]
    for r,ref in enumerate(refs):
        candidates={row.mode.mode_id:i for i,row in enumerate(rows) if row.resource_ref==ref}
        tid=route['module_types'][route['module_refs'].index(ref.module)]
        for i,o in enumerate(contract.outcomes):
            if (o.module_type_id,o.resource_id)!=(tid,ref.resource_id):continue
            if o.kind!=INACTIVE and o.mode_id not in candidates:continue
            catalog.append(i);physical.append(candidates.get(o.mode_id,-1));owners.append(r);kinds.append(o.kind)
        ptr.append(len(catalog))
    if any(a==b for a,b in zip(ptr,ptr[1:])):raise ValueError('Resource without outcome domain')
    return {k:torch.tensor(v,dtype=torch.int64) for k,v in dict(outcome_ptr=ptr,outcome_resource=owners,
        outcome_catalog=catalog,outcome_candidate=physical,outcome_kind=kinds).items()}


def add_resource_decisions(sample,contract,*,authoritative_resources=frozenset()):
    """Only explicit authority certifies absent negatives. Existing masked rows
    override that authority: mask-false is never turned into INACTIVE.
    Payload targets contain ONLY supervised parameterized decisions.
    """
    import torch
    x=outcome_inputs(sample,contract);old=sample['supervision']['resource'];route=sample['routing']
    physical={(t.resource,t.mode_id):t for t in route['physical_supervision'].resource_targets}
    existing={int(r):i for i,r in enumerate(old['resource'].tolist())}
    target=[];supervised=[];keep=[]
    for r,ref in enumerate(route['resource_refs']):
        row=existing.get(r);chosen=-1;valid=False
        if row is not None and bool(old['supervised'][row]):
            q=int(old['candidate'][row]);mode=route['resource_candidates'][q].mode.mode_id
            source=physical.get((ref,mode))
            if source is None:raise ValueError('Missing exact physical source target')
            options=[i for i in range(int(x['outcome_ptr'][r]),int(x['outcome_ptr'][r+1]))
                if contract.outcomes[int(x['outcome_catalog'][i])].mode_id==mode]
            fixed=[i for i in options if int(x['outcome_kind'][i])==FIXED and contract.outcomes[int(x['outcome_catalog'][i])].payload==source.payload]
            chosen=fixed[0] if fixed else next(i for i in options if int(x['outcome_kind'][i])==PARAMETERIZED)
            valid=True
            if not fixed:keep.append(row)
        elif row is None and r in authoritative_resources:
            chosen=int(x['outcome_ptr'][r]);valid=True
            if int(x['outcome_kind'][chosen])!=INACTIVE:raise ValueError('Missing INACTIVE outcome')
        target.append(chosen);supervised.append(valid)
    y={};parts=[];ptr=[0]
    for row in keep:
        a,b=old['payload_ptr'][row:row+2].tolist();parts.extend(range(a,b));ptr.append(len(parts))
    for key,value in old.items():
        if key=='payload_ptr':y[key]=torch.tensor(ptr,dtype=torch.int64)
        elif key in ('payload','payload_valid','payload_category'):y[key]=value[parts].clone()
        else:y[key]=value[keep].clone()
    y.update(decision_group=torch.arange(len(target)),decision_candidate=torch.tensor(target,dtype=torch.int64),
        decision_supervised=torch.tensor(supervised,dtype=torch.bool))
    return {**sample,'inputs':{**sample['inputs'],'resource_candidates':{**sample['inputs']['resource_candidates'],**x}},
        'supervision':{**sample['supervision'],'resource':y},'routing':{**route,
        'resource_decision_schema':VERSION,'resource_contract_fingerprint':contract.fingerprint,'resource_contract':contract.document}}


def validate_outcome_inputs(rows,contract,count):
    import torch
    required=('outcome_ptr','outcome_resource','outcome_catalog','outcome_candidate','outcome_kind')
    if any(k not in rows or rows[k].dtype!=torch.int64 or rows[k].ndim!=1 for k in required):raise ValueError('Missing/invalid v3 outcome routing')
    modes=sorted((o.module_type_id,o.resource_id,o.mode_id) for o in contract.outcomes if o.kind==PARAMETERIZED)
    ptr=rows['outcome_ptr'];n=len(rows['outcome_kind'])
    if len(ptr)!=count+1 or ptr[0]!=0 or ptr[-1]!=n or (ptr[1:]<=ptr[:-1]).any():raise ValueError('Invalid outcome groups')
    if any(len(rows[k])!=n for k in required[1:]):raise ValueError('Outcome row mismatch')
    catalog=rows['outcome_catalog'];q=rows['outcome_candidate'];kind=rows['outcome_kind']
    owner=torch.repeat_interleave(torch.arange(count,device=ptr.device),ptr[1:]-ptr[:-1])
    if ((catalog<0)|(catalog>=len(contract.outcomes))|(rows['outcome_resource']!=owner)).any():raise ValueError('Invalid outcome catalog/owner')
    if (owner*len(contract.outcomes)+catalog).unique().numel()!=n:raise ValueError('Invalid outcome catalog/owner')
    expected_kind=torch.tensor([o.kind for o in contract.outcomes],device=ptr.device,dtype=torch.int64)[catalog]
    inactive=kind==INACTIVE
    if ((kind!=expected_kind)|(inactive&(q!=-1))|(~inactive&((q<0)|(q>=len(rows['resource']))))).any():raise ValueError('Invalid physical outcome routing')
    physical=q[~inactive]
    if (rows['resource'][physical]!=owner[~inactive]).any():raise ValueError('Cross-resource outcome')
    mode_index={m:i for i,m in enumerate(modes)}
    expected_mode=torch.tensor([mode_index.get((o.module_type_id,o.resource_id,o.mode_id),-1)
        for o in contract.outcomes],device=ptr.device,dtype=torch.int64)
    if (rows['mode'][physical]!=expected_mode[catalog[~inactive]]).any():raise ValueError('Outcome physical mode mismatch')
    numbers=ptr.new_zeros(count).scatter_add_(0,owner,inactive.long())
    if (numbers!=1).any():raise ValueError('Exactly one INACTIVE required')


def supervised_decisions(prediction,batch):
    import torch
    p=prediction['resource'];x=batch['inputs']['resource_candidates'];y=batch['supervision']['resource']
    n=len(x['outcome_ptr'])-1
    validate_decision_targets(batch)
    if any(y[k].shape!=(n,) for k in ('decision_group','decision_candidate','decision_supervised')):raise ValueError('Resource decision shape mismatch')
    if not torch.equal(y['decision_group'],torch.arange(n,device=y['decision_group'].device)):raise ValueError('Resource decisions must cover each resource once')
    if p['outcome_logits'].shape!=x['outcome_kind'].shape:raise ValueError('Outcome logit shape mismatch')
    for r,valid in enumerate(y['decision_supervised'].tolist()):
        q=int(y['decision_candidate'][r]);a,b=x['outcome_ptr'][r:r+2].tolist()
        if not valid:
            if q!=-1:raise ValueError('Unknown resource decision must have sentinel target')
            continue
        if not a<=q<b:raise ValueError('Target outside outcome group')
        yield r,p['outcome_logits'][a:b],q-a


class ResourceDecisionMetrics:
    """Streaming integer confusion sufficient statistics, no payload retention."""
    def __init__(self):self.rows={};self.pre=0;self.post=0;self.joint_support=0
    def add(self,resource,target,predicted,kind,target_name):
        for key in ('ALL','resource:'+resource,'mode:'+target_name):
            c=self.rows.setdefault(key,dict(support=0,inactive=0,false=0,missed=0,exact=0,fixed=0,fixed_exact=0))
            c['support']+=1;c['inactive']+=int(kind==INACTIVE);c['false']+=int(kind==INACTIVE and predicted!=target)
            c['missed']+=int(kind!=INACTIVE and predicted is None);c['exact']+=int(predicted==target)
            c['fixed']+=int(kind==FIXED);c['fixed_exact']+=int(kind==FIXED and predicted==target)
    def compute(self):
        def row(c):
            ratio=lambda a,b:a/b if b else None
            inactive=c['inactive'];active=c['support']-inactive;tn=inactive-c['false']
            return {'support':c['support'],'inactive':{'support':inactive,'accuracy':ratio(tn,inactive),
                'precision':ratio(tn,tn+c['missed']),'recall':ratio(tn,inactive)},
                'activation_recall':ratio(active-c['missed'],active),'false_activation_rate':ratio(c['false'],inactive),
                'missed_activation_rate':ratio(c['missed'],active),'outcome_exact_accuracy':ratio(c['exact'],c['support']),
                'fixed':{'support':c['fixed'],'accuracy':ratio(c['fixed_exact'],c['fixed'])},'counts':dict(c)}
        result=row(self.rows.get('ALL',dict(support=0,inactive=0,false=0,missed=0,exact=0,fixed=0,fixed_exact=0)))
        result['by_resource']={k[9:]:row(v) for k,v in self.rows.items() if k.startswith('resource:')}
        result['by_mode']={k[5:]:row(v) for k,v in self.rows.items() if k.startswith('mode:')}
        result['compatibility']={'support':self.joint_support,'pre_violations':self.pre,'post_violations':self.post}
        return result


def validate_decision_targets(batch):
    x=batch['inputs']['resource_candidates'];y=batch['supervision']['resource'];n=len(x['outcome_ptr'])-1
    if any(k not in y or y[k].shape!=(n,) for k in ('decision_group','decision_candidate','decision_supervised')):raise ValueError('Missing resource decision targets')
    expected={}
    for r,valid in enumerate(y['decision_supervised'].tolist()):
        q=int(y['decision_candidate'][r]);a,b=x['outcome_ptr'][r:r+2].tolist()
        if not valid:
            if q!=-1:raise ValueError('Unknown resource target must be sentinel')
            continue
        if not a<=q<b:raise ValueError('Outcome target outside resource domain')
        if int(x['outcome_kind'][q])==PARAMETERIZED:expected[r]=int(x['outcome_candidate'][q])
    actual={}
    for r,q,valid in zip(y['resource'].tolist(),y['candidate'].tolist(),y['supervised'].tolist()):
        if not valid:raise ValueError('v3 payload rows must be supervised parameterized decisions')
        if r in actual:raise ValueError('Duplicate resource payload decision')
        actual[r]=q
    if expected!=actual:raise ValueError('Payload supervision inconsistent with resource outcomes')
