"""I2 offline expert-action diagnostics. Never simulator/task-success metrics.

State contains scalar sufficient statistics only: Python float64 sums and exact
integer counts. No predictions, targets, autograd graphs or sample IDs retained.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
import json
import math

import torch

from .normalization import denormalize_values,normalizer_fingerprint
from .bc_v1_contract import validate_robot_spec,robot_spec_fingerprint
from .losses import _bounds,_routes,_argument_rows,_validate_membership_owners,_validate_supervision
from .tensorization import PARAMETER_KINDS


class MeanAccumulator:
    def __init__(self):self.reset()

    def reset(self):self.sum=0.;self.support=0

    def add(self,values):
        if isinstance(values,torch.Tensor):values=values.detach().flatten().cpu().tolist()
        if isinstance(values,(int,float,bool)):values=[values]
        values=[float(v) for v in values]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite metric observation')
        self.sum=math.fsum((self.sum,*values));self.support+=len(values)

    def compute(self):
        return {'value':self.sum/self.support if self.support else None,'support':self.support,'sum':self.sum}


class ErrorAccumulator:
    def __init__(self):self.reset()

    def reset(self):self.sum_abs=0.;self.sum_squared=0.;self.support=0

    @torch.no_grad()
    def add(self,prediction,target,valid=None):
        if prediction.shape!=target.shape:raise ValueError('Metric numeric arity mismatch')
        valid=torch.ones_like(target,dtype=torch.bool) if valid is None else valid
        if valid.dtype!=torch.bool or valid.shape!=target.shape:raise ValueError('Invalid metric mask')
        p=prediction.detach()[valid].double();t=target.detach()[valid].double()
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():raise ValueError('Nonfinite supervised metric value')
        errors=p-t
        if not torch.isfinite(errors).all() or not torch.isfinite(errors.square()).all():
            raise ValueError('Metric error outside finite float64 accumulator range')
        self.sum_abs=math.fsum((self.sum_abs,float(errors.abs().sum())))
        self.sum_squared=math.fsum((self.sum_squared,float(errors.square().sum())))
        self.support+=errors.numel()

    def compute(self):
        return {'mae':{'value':self.sum_abs/self.support if self.support else None,
                'support':self.support,'sum':self.sum_abs},
            'rmse':{'value':math.sqrt(self.sum_squared/self.support) if self.support else None,
                'support':self.support,'sum_squared_error':self.sum_squared}}


def physical_values(normalized,keys,normalizers):
    """Authoritative F2 inverse, including identity/no-support and bounds."""
    if len(normalized)!=len(keys):raise ValueError('Metric inverse arity mismatch')
    return tuple(denormalize_values(float(v),k,normalizers) for v,k in zip(normalized,keys))


class ContinuousAccumulator:
    def __init__(self,normalizers,keys,units,tolerances):
        if len(keys)!=len(units):raise ValueError('Metric unit/scaler arity mismatch')
        self.normalizers=normalizers;self.keys=keys;self.units=units;self.tolerances=tolerances
        self.normalized=ErrorAccumulator();self.physical={u:ErrorAccumulator() for u in sorted(set(units))}
        self.within={u:MeanAccumulator() for u in self.physical};self.decisions=0

    @torch.no_grad()
    def add(self,prediction,target,valid):
        if len(prediction)!=len(self.keys):raise ValueError('Target signature arity mismatch')
        self.normalized.add(prediction,target,valid)
        if valid.any():self.decisions+=1
        for j in torch.nonzero(valid,as_tuple=False).flatten().tolist():
            p,t=physical_values((prediction[j],target[j]),(self.keys[j],self.keys[j]),self.normalizers)
            if not math.isfinite(p) or not math.isfinite(t):raise ValueError('Nonfinite physical metric value')
            unit=self.units[j];self.physical[unit].add(torch.tensor([p],dtype=torch.float64),torch.tensor([t],dtype=torch.float64))
            if unit in self.tolerances:self.within[unit].add(abs(p-t)<=self.tolerances[unit])

    def compute(self):
        result=self.normalized.compute();physical={}
        for unit,accumulator in self.physical.items():
            row=accumulator.compute()
            for metric in row.values():metric['unit']=unit
            row['within_tolerance']={**self.within[unit].compute(),'unit':unit,'tolerance':self.tolerances.get(unit)}
            physical[unit]=row
        return {'normalized_mae':result['mae'],'normalized_rmse':result['rmse'],
            'decision_support':self.decisions,'physical_by_unit':physical}


class ConfusionAccumulator:
    def __init__(self,classes):
        self.classes=tuple(classes);self.counts=[[0]*len(classes) for _ in classes]

    def add(self,target,predicted):
        if not 0<=target<len(self.classes) or not 0<=predicted<len(self.classes):raise ValueError('Metric category outside domain')
        self.counts[target][predicted]+=1

    def compute(self):
        rows=[sum(row) for row in self.counts];total=sum(rows);recall=[];f1=[]
        for i,n in enumerate(rows):
            tp=self.counts[i][i];fp=sum(row[i] for row in self.counts)-tp;fn=n-tp
            recall.append({'value':tp/n if n else None,'support':n,'sum':tp})
            if n:f1.append(2*tp/(2*tp+fp+fn));
        active=[r['value'] for r in recall if r['support']]
        return {'classes':list(self.classes),'counts':[r[:] for r in self.counts],'class_support':rows,
            'accuracy':{'value':sum(self.counts[i][i] for i in range(len(rows)))/total if total else None,'support':total},
            'macro_f1':{'value':math.fsum(f1)/len(f1) if f1 else None,'support':total,'classes_with_support':len(f1)},
            'balanced_accuracy':{'value':math.fsum(active)/len(active) if active else None,'support':total,'classes_with_support':len(active)},
            'recall':recall}


def _maxima(logits):
    if not len(logits) or not torch.isfinite(logits).all():raise ValueError('Empty/nonfinite supervised metric logits')
    return torch.nonzero(logits==logits.max(),as_tuple=False).flatten()


def _membership_observation(records,row,indices,values,valid,required,domain):
    start,end=_bounds(records['ptr'],row);options=records['domain_index'][start:end].tolist();routes=_routes(records['domain_index'][start:end])
    truth={};lo,hi=domain
    for i in torch.nonzero(valid,as_tuple=False).flatten().tolist():
        d=int(indices[i])
        if not lo<=d<hi:raise ValueError('Metric membership outside decision domain')
        if bool(required[d]):continue
        if d in truth or d not in routes:raise ValueError('Duplicate/missing optional metric member')
        value=float(values[i])
        if value not in (0.,1.):raise ValueError('Invalid supervised membership value')
        if not torch.isfinite(records['logits'][start+routes[d]]):raise ValueError('Nonfinite supervised membership prediction')
        truth[d]=bool(value)
    # Full-set diagnostics need the full OPTIONAL target, not an observed subset.
    complete=set(truth)==set(options)
    if not complete:return {'complete':False,'eligible':False,'correct':False,'jaccard':None}
    predicted={d for j,d in enumerate(options) if float(records['logits'][start+j])>0}
    actual={d for d,value in truth.items() if value};union=predicted|actual
    return {'complete':True,'eligible':bool(options),'correct':predicted==actual,
        'jaccard':len(predicted&actual)/len(union) if union else 1.}


def _key(parts):return json.dumps(parts,separators=(',',':'))


def load_metrics_config(path):
    config=json.loads(Path(path).read_text());_validate_config(config)
    return config


def _validate_config(config):
    if (not isinstance(config,Mapping) or set(config)!={'schema_version','tolerances','breakdowns'}
            or config['schema_version']!='mssr.bc_v1.metrics.v1' or not isinstance(config['tolerances'],Mapping)
            or any(not isinstance(k,str) or not k or type(v) not in (float,int) or not math.isfinite(v) or v<0
                for k,v in config['tolerances'].items()) or not isinstance(config['breakdowns'],list)
            or any(k not in ('task','morphology','provenance','task_stratum') for k in config['breakdowns'])
            or len(set(config['breakdowns']))!=len(config['breakdowns'])):
        raise ValueError('Unsupported I2 metric configuration')


class _Store:
    def __init__(self,owner):
        from .structural_decisions import StructuralDecisionMetrics
        self.decisions=StructuralDecisionMetrics(tuple(a.action_id for a in owner.actions if a.scope!='resource'))
        self.has_decisions=False
        self.resource_error=ErrorAccumulator();self.resource_decisions=0
        self.payloads={_key(k):ContinuousAccumulator(owner.normalizers,owner.resource_bindings[k].keys,
            owner.resource_bindings[k].units,owner.tolerances) for k,m in zip(owner.mode_catalog,owner.modes) if m.codec!='categorical'}
        self.mode_accuracy=MeanAccumulator()
        self.mode_domains={k:ConfusionAccumulator(v) for k,v in owner.mode_domains.items()}
        self.payload_category=MeanAccumulator()
        self.structural={k:MeanAccumulator() for k in ('candidate_top1','operation_accuracy','peer_accuracy',
            'connector_a_accuracy','connector_b_accuracy','ordered_connector_pair_accuracy')}
        self.operation_recall={a.action_id:MeanAccumulator() for a in owner.actions if a.scope!='resource'}
        self.arguments={k:MeanAccumulator() for k in ('category_accuracy','reference_top1','boolean_accuracy','reference_set_exact','reference_set_jaccard')}
        self.scope={k:MeanAccumulator() for k in ('exact_match','jaccard')}
        self.numeric_error=ErrorAccumulator();self.numeric_decisions=0
        self.numeric={k:ContinuousAccumulator(owner.normalizers,keys,units,owner.tolerances) for k,(keys,units) in owner.numeric_signatures.items()}
        self.tuple_exact=MeanAccumulator();self.tuple_incomplete=0

    def compute(self):
        error=self.resource_error.compute()
        structural={k:v.compute() for k,v in self.structural.items()}
        structural['binding_exact_match']=dict(structural['candidate_top1'])
        recalls={k:v.compute() for k,v in self.operation_recall.items()};active=[v['value'] for v in recalls.values() if v['support']]
        structural['operation_recall']=recalls
        if self.has_decisions:structural['decisions']=self.decisions.compute()
        structural['operation_macro_recall']={'value':math.fsum(active)/len(active) if active else None,
            'support':sum(v['support'] for v in recalls.values()),'classes_with_support':len(active)}
        structural['scope']={k:v.compute() for k,v in self.scope.items()}
        structural['discrete_tuple_exact']={**self.tuple_exact.compute(),'excluded_incomplete':self.tuple_incomplete}
        numeric=self.numeric_error.compute()
        arguments={k:self.arguments[k].compute() for k in ('category_accuracy','reference_top1','boolean_accuracy')}
        arguments['reference_set']={'exact_match':self.arguments['reference_set_exact'].compute(),'jaccard':self.arguments['reference_set_jaccard'].compute()}
        arguments['numeric']={'normalized_mae':numeric['mae'],'normalized_rmse':numeric['rmse'],
            'argument_support':self.numeric_decisions,'by_parameter':{k:v.compute() for k,v in sorted(self.numeric.items())}}
        return {'resource':{'payload':{'normalized_mae':error['mae'],'normalized_rmse':error['rmse'],
            'decision_support':self.resource_decisions,'by_signature':{k:v.compute() for k,v in sorted(self.payloads.items())}},
            'mode':{'accuracy':self.mode_accuracy.compute(),'domains':{k:v.compute() for k,v in sorted(self.mode_domains.items())}},
            'payload_category_accuracy':self.payload_category.compute()},
            'structural':structural,'arguments':arguments}


class BCMetrics:
    """Streaming offline metrics; candidates and teacher-conditioned payloads.

    Numeric errors aggregate COMPONENTS; decision counts are explicit. Physical
    units never share an error average. No inverse fitting, gradient or model IO.
    """
    def __init__(self,spec,normalizers,config=None):
        config={'schema_version':'mssr.bc_v1.metrics.v1','tolerances':{},'breakdowns':[]} if config is None else config
        validate_robot_spec(spec);_validate_config(config)
        if normalizers.metadata.get('robot_spec_fingerprint')!=robot_spec_fingerprint(spec):
            raise ValueError('Metric normalizers/RobotSpec mismatch')
        self.normalizers=normalizers;self.tolerances=dict(config['tolerances']);self.dimensions=tuple(config['breakdowns'])
        self.normalizer_fingerprint=normalizer_fingerprint(normalizers)
        mode_specs={(t.type_id,r.resource_id,m.mode_id):m for t in spec.module_types for r in t.resources for m in r.modes}
        self.mode_catalog=tuple(sorted(mode_specs));self.modes=tuple(mode_specs[k] for k in self.mode_catalog)
        self.resource_bindings={(b.module_type_id,b.resource_id,b.mode_id):b for b in normalizers.resource_bindings}
        self.mode_domains={_key(sorted((t.type_id,r.resource_id,m.mode_id) for m in r.modes)):
            tuple(sorted((t.type_id,r.resource_id,m.mode_id) for m in r.modes))
            for t in spec.module_types for r in t.resources if len(r.modes)>1}
        for k,m in zip(self.mode_catalog,self.modes):
            if k not in self.resource_bindings or self.resource_bindings[k].units!=m.units:
                raise ValueError('Missing/incompatible metric target signature')
        self.actions=tuple(sorted(spec.actions,key=lambda a:a.action_id))
        self.parameter_catalog=tuple((a.action_id,p.name) for a in self.actions for p in sorted(a.parameters,key=lambda p:p.name))
        self.parameters=tuple(p for a in self.actions for p in sorted(a.parameters,key=lambda p:p.name))
        bindings={(b.action_id,b.name):b for b in normalizers.argument_bindings};self.numeric_signatures={}
        scopes={a.action_id:a.scope for a in self.actions}
        for k,p in zip(self.parameter_catalog,self.parameters):
            if scopes[k[0]]!='resource' and p.learnable and p.kind in ('scalar','vector'):
                if k not in bindings or len(bindings[k].keys)!=(p.size or 1):raise ValueError('Missing numeric argument scaler signature')
                self.numeric_signatures[_key(k)]=(bindings[k].keys,(p.units or '1',)*(p.size or 1))
        self.reset()

    def reset(self):
        self._base=_Store(self);self._breakdowns={d:{} for d in self.dimensions};self._labels=None

    def compute(self):
        result=self._base.compute()
        result['breakdowns']={d:{k:v.compute() for k,v in sorted(groups.items())} for d,groups in self._breakdowns.items()}
        return result

    def _stores(self,graph):
        stores=[self._base]
        if self._labels is not None:
            for dimension in self.dimensions:
                label=self._labels[graph][dimension];groups=self._breakdowns[dimension]
                if label not in groups:groups[label]=_Store(self)
                stores.append(groups[label])
        return stores

    @contextmanager
    def audit_context(self,batch,audit=None):
        """Only post-forward fanout labels, never supervision eligibility.

        Task is current G2 ontology, task_stratum is explicitly a separate audit
        field. No phase/FSM heuristics. UNKNOWN labels retain the same support.
        Only small label strings survive this context, no batch tensors retained.
        """
        count=len(batch['inputs']['graph_ptr'])-1;routing=batch.get('routing',{})
        sources=routing.get('samples',(routing,))
        if audit is not None:
            audits=list(audit) if isinstance(audit,(list,tuple)) else [audit]
            if len(audits)!=count:raise ValueError('Audit metadata/graph count mismatch')
        else:audits=[sources[g].get('audit') if g<len(sources) else None for g in range(count)]
        labels=[]
        for g,metadata in enumerate(audits):
            route=sources[g] if g<len(sources) else {};vocabulary=route.get('task_vocabulary',())
            task='UNKNOWN'
            if vocabulary and bool(batch['inputs']['task_valid'][g]):
                index=int(batch['inputs']['task_one_hot'][g].argmax())
                if index<len(vocabulary):task=vocabulary[index]
            row={'task':task}
            for dimension,field in (('morphology','morphology_stratum'),('provenance','provenance'),('task_stratum','task_stratum')):
                value=metadata.get(field) if isinstance(metadata,Mapping) else getattr(metadata,field,None)
                row[dimension]=value if isinstance(value,str) and value else 'UNKNOWN'
            labels.append(row)
        previous=self._labels;self._labels=labels
        try:yield self
        finally:self._labels=previous

    @staticmethod
    def _graph(batch,module):
        ptr=batch['inputs']['graph_ptr']
        if not 0<=module<int(ptr[-1]):raise ValueError('Metric module route outside graph')
        return next(i for i in range(len(ptr)-1) if int(ptr[i])<=module<int(ptr[i+1]))

    @torch.no_grad()
    def update_resource_payload(self,prediction,batch):
        p=prediction['resource'];y=batch['supervision']['resource'];x=batch['inputs']['resource_candidates']
        routes=_routes(p['payload_candidate']);categories=_routes(p['categorical']['candidate']);seen=set()
        for row,valid in enumerate(y['supervised'].tolist()):
            if not valid:continue
            resource=int(y['resource'][row]);q=int(y['candidate'][row]);start,end=_bounds(x['group_ptr'],resource)
            if not start<=q<end or resource in seen:raise ValueError('Invalid/duplicate resource metric target')
            seen.add(resource)
            if q not in routes:
                if self.modes[int(x['mode'][q])].codec=='categorical':
                    if q not in categories:raise ValueError('Missing categorical resource metric prediction')
                    a,b=_bounds(y['payload_ptr'],row)
                    if b-a!=1:raise ValueError('Categorical metric payload arity mismatch')
                    if bool(y['payload_valid'][a]):
                        lo,hi=_bounds(p['categorical']['ptr'],categories[q]);target=int(y['payload_category'][a])
                        if not 0<=target<hi-lo:raise ValueError('Categorical payload metric target outside domain')
                        predicted=int(_maxima(p['categorical']['logits'][lo:hi])[0])
                        graph=self._graph(batch,int(batch['inputs']['resource_to_module'][resource]))
                        for store in self._stores(graph):store.payload_category.add(predicted==target)
                    continue
                raise ValueError('Missing numeric resource prediction')
            a,b=_bounds(y['payload_ptr'],row);lo,hi=_bounds(p['payload_ptr'],routes[q]);mask=y['payload_valid'][a:b]
            mode=int(x['mode'][q]);signature=_key(self.mode_catalog[mode])
            graph=self._graph(batch,int(batch['inputs']['resource_to_module'][resource]))
            for store in self._stores(graph):
                store.payloads[signature].add(p['payload'][lo:hi],y['payload'][a:b],mask)
                store.resource_error.add(p['payload'][lo:hi],y['payload'][a:b],mask)
                store.resource_decisions+=int(bool(mask.any()))

    @torch.no_grad()
    def update_resource_mode(self,prediction,batch):
        p=prediction['resource'];y=batch['supervision']['resource'];x=batch['inputs']['resource_candidates'];seen=set()
        routes=_routes(p['mode_candidate'])
        for row,valid in enumerate(y['supervised'].tolist()):
            if not valid:continue
            resource=int(y['resource'][row]);q=int(y['candidate'][row]);start,end=_bounds(x['group_ptr'],resource)
            if not start<=q<end or resource in seen:raise ValueError('Invalid/duplicate resource mode target')
            seen.add(resource)
            if end-start==1:continue
            lo,hi=_bounds(p['mode_group_ptr'],resource)
            if q not in routes or not lo<=routes[q]<hi or hi-lo!=end-start:raise ValueError('Misaligned mode metric group')
            candidates=p['mode_candidate'][lo:hi];domain=sorted(x['mode'][start:end].tolist())
            maxima=_maxima(p['mode_logits'][lo:hi])
            # Category order is a static semantic vocabulary, never a physical ID.
            predicted=min(int(x['mode'][candidates[i]]) for i in maxima.tolist());target=int(x['mode'][q])
            classes=tuple(self.mode_catalog[i] for i in domain);key=_key(classes)
            graph=self._graph(batch,int(batch['inputs']['resource_to_module'][resource]))
            for store in self._stores(graph):
                store.mode_accuracy.add(predicted==target)
                if key not in store.mode_domains:store.mode_domains[key]=ConfusionAccumulator(classes)
                store.mode_domains[key].add(domain.index(target),domain.index(predicted))

    def _structural_decisions(self,prediction,batch):
        p=prediction['structural'];y=batch['supervision']['structural'];seen=set()
        for row,valid in enumerate(y['supervised'].tolist()):
            if not valid:continue
            group=int(y['group'][row]);q=int(y['candidate'][row]);start,end=_bounds(p['group_ptr'],group)
            if not start<=q<end or group in seen:raise ValueError('Invalid/duplicate structural metric decision')
            seen.add(group);maxima=_maxima(p['candidate_logits'][start:end])+start
            if 'no_structural_logits' in p and p['no_structural_logits'][group]>=p['candidate_logits'][maxima[0]]:
                maxima=maxima[:0]
            yield row,q,maxima,self._graph(batch,group)

    @torch.no_grad()
    def update_structural(self,prediction,batch):
        """Set-valued exact ties: existential agreement, no physical ID tiebreak.

        Each attribute gets credit if a maximizer agrees. Ordered pair agreement
        requires both endpoints of ONE maximizer. This is offline diagnostics,
        not a deterministic action decoder or operation-confusion assignment.
        """
        x=batch['inputs']['structural']
        if 'no_structural_logits' in prediction['structural']:
            from .structural_decisions import joint_decisions
            p=prediction['structural']
            for group,logits,target in joint_decisions(prediction,batch):
                chosen=int(logits.argmax());lo=int(p['group_ptr'][group])
                truth=None if target==0 else self.actions[int(x['action'][lo+target-1])].action_id
                selected=None if chosen==0 else self.actions[int(x['action'][lo+chosen-1])].action_id
                for store in self._stores(self._graph(batch,group)):
                    store.has_decisions=True;store.decisions.add(truth,selected,chosen==target)
        for row,q,maxima,graph in self._structural_decisions(prediction,batch):
            operation=self.actions[int(x['action'][q])].action_id
            correct=bool((x['action'][maxima]==x['action'][q]).any())
            for store in self._stores(graph):
                store.structural['candidate_top1'].add(bool((maxima==q).any()))
                store.structural['operation_accuracy'].add(correct);store.operation_recall[operation].add(correct)
                if int(x['peer'][q])>=0:
                    store.structural['peer_accuracy'].add(bool((x['peer'][maxima]==x['peer'][q]).any()))
                if int(x['connector_a'][q])>=0:
                    a=x['connector_a'][maxima]==x['connector_a'][q];b=x['connector_b'][maxima]==x['connector_b'][q]
                    store.structural['connector_a_accuracy'].add(bool(a.any()))
                    store.structural['connector_b_accuracy'].add(bool(b.any()))
                    store.structural['ordered_connector_pair_accuracy'].add(bool((a&b).any()))

    def _argument_observations(self,prediction,batch):
        _validate_membership_owners(batch)
        x=batch['inputs']['structural'];y=batch['supervision']['structural'];heads=prediction['structural']['arguments']
        routes={k:_routes(v['argument']) for k,v in heads.items()}
        for row,u in _argument_rows(batch,PARAMETER_KINDS):
            parameter_index=int(x['argument_parameter'][u])
            if not 0<=parameter_index<len(self.parameters):raise ValueError('Unknown metric parameter')
            p=self.parameters[parameter_index];kind=p.kind;binding=int(x['argument_binding'][u])
            if not p.learnable or int(x['argument_kind'][u])!=PARAMETER_KINDS.index(kind):raise ValueError('Invalid learned metric argument')
            if kind in ('module_ref','connector_ref') and int(x['argument_bound_ref'][u])!=-1:
                a,b=_bounds(y['reference_ptr'],row)
                if b-a!=1 or int(y['reference_index'][a])!=int(x['argument_bound_ref'][u]):raise ValueError('Incorrect fixed reference target')
                continue
            family={'scalar':'numeric','vector':'numeric','module_ref':'reference','connector_ref':'reference',
                'module_refs':'reference_set','connector_refs':'reference_set'}.get(kind,kind)
            records=heads[family]
            if u not in routes[family]:raise ValueError('Missing supervised argument metric prediction')
            r=routes[family][u];obs={'binding':binding,'argument':u,'family':family,
                'graph':self._graph(batch,int(x['initiator'][binding])),'complete':True,'eligible':True}
            if family=='boolean':
                a,b=_bounds(y['argument_value_ptr'],row)
                if b-a!=1:raise ValueError('Boolean metric arity mismatch')
                if not bool(y['argument_value_valid'][a]):obs.update(complete=False,eligible=False,correct=False)
                else:
                    value=float(y['argument_value'][a]);logit=float(records['logits'][r])
                    if value not in (0.,1.) or not math.isfinite(logit):raise ValueError('Invalid supervised boolean metric')
                    obs['correct']=(logit>0)==bool(value)
            else:
                start,end=_bounds(records['ptr'],r)
                if family=='numeric':
                    a,b=_bounds(y['argument_value_ptr'],row)
                    obs.update(prediction=records['value'][start:end],target=y['argument_value'][a:b],
                        valid=y['argument_value_valid'][a:b],signature=_key(self.parameter_catalog[parameter_index]))
                elif family=='category':
                    target=int(y['argument_category'][row]);maxima=_maxima(records['logits'][start:end])
                    if not 0<=target<end-start:raise ValueError('Metric category outside parameter domain')
                    obs['correct']=int(maxima[0])==target
                elif family=='reference':
                    a,b=_bounds(y['reference_ptr'],row)
                    if b-a!=1:raise ValueError('Metric single reference arity mismatch')
                    options=x['argument_domain_index'][records['domain_index'][start:end]].tolist();target=int(y['reference_index'][a])
                    if target not in options:raise ValueError('Metric reference outside current domain')
                    maxima=_maxima(records['logits'][start:end]);obs['correct']=options.index(target) in maxima.tolist()
                else:
                    lo,hi=_bounds(x['argument_domain_ptr'],u);a,b=_bounds(y['reference_ptr'],row)
                    if not set(y['reference_index'][a:b].tolist())<=set(x['argument_domain_index'][lo:hi].tolist()):raise ValueError('Metric set target outside allowed domain')
                    select=(y['membership_domain']>=lo)&(y['membership_domain']<hi)
                    obs.update(_membership_observation(records,r,y['membership_domain'][select],y['membership_value'][select],
                        y['membership_valid'][select],x['argument_domain_required'],(lo,hi)))
            yield obs

    def _record_arguments(self,observations):
        for obs in observations:
            for store in self._stores(obs['graph']):
                family=obs['family']
                if family=='numeric':
                    store.numeric[obs['signature']].add(obs['prediction'],obs['target'],obs['valid'])
                    store.numeric_error.add(obs['prediction'],obs['target'],obs['valid']);store.numeric_decisions+=int(bool(obs['valid'].any()))
                elif obs['eligible']:
                    name={'category':'category_accuracy','reference':'reference_top1','boolean':'boolean_accuracy','reference_set':'reference_set_exact'}[family]
                    store.arguments[name].add(obs['correct'])
                    if family=='reference_set':store.arguments['reference_set_jaccard'].add(obs['jaccard'])

    @torch.no_grad()
    def update_arguments(self,prediction,batch):self._record_arguments(self._argument_observations(prediction,batch))

    def _scope_observations(self,prediction,batch):
        x=batch['inputs']['structural'];y=batch['supervision']['structural'];records=prediction['structural']['scope'];routes=_routes(records['binding'])
        for row,valid in enumerate(y['supervised'].tolist()):
            if not valid:continue
            q=int(y['candidate'][row]);lo,hi=_bounds(x['scope_ptr'],q)
            obs={'binding':q,'graph':self._graph(batch,int(x['initiator'][q])),'complete':True,'eligible':False,'correct':True,'jaccard':1.}
            if not x['scope_required'][lo:hi].all():
                if q not in routes:raise ValueError('Missing variable scope metric prediction')
                a,b=_bounds(y['scope_membership_ptr'],row)
                obs.update(_membership_observation(records,routes[q],y['scope_membership_index'][a:b],y['scope_membership_value'][a:b],
                    y['scope_membership_valid'][a:b],x['scope_required'],(lo,hi)))
            yield obs

    def _record_scope(self,observations):
        for obs in observations:
            if obs['eligible']:
                for store in self._stores(obs['graph']):
                    store.scope['exact_match'].add(obs['correct']);store.scope['jaccard'].add(obs['jaccard'])

    @torch.no_grad()
    def update_scope(self,prediction,batch):self._record_scope(self._scope_observations(prediction,batch))

    def _record_tuple(self,prediction,batch,arguments,scopes):
        x=batch['inputs']['structural'];heads=prediction['structural']['arguments']
        discrete={(o['binding'],o['argument']):o for o in arguments if o['family']!='numeric'}
        scope={o['binding']:o for o in scopes}
        expected={}
        for family in ('category','reference','reference_set','boolean'):
            for u in heads[family]['argument'].tolist():expected.setdefault(int(x['argument_binding'][u]),set()).add(u)
        for row,q,maxima,graph in self._structural_decisions(prediction,batch):
            required=expected.get(q,set())
            complete=all((q,u) in discrete and discrete[q,u]['complete'] for u in required) and scope[q]['complete']
            for store in self._stores(graph):
                if not complete:store.tuple_incomplete+=1;continue
                correct=bool((maxima==q).any()) and all(discrete[q,u]['correct'] for u in required) and scope[q]['correct']
                store.tuple_exact.add(correct)

    @torch.no_grad()
    def update_tuple(self,prediction,batch):
        """Full supervised discrete expert-binding agreement, no numeric equality.

        Missing/partially masked learned discrete choices exclude tuple support
        (reported as incomplete), rather than fabricating failure or correctness.
        Component argument metrics remain teacher-conditioned diagnostics.
        """
        self._record_tuple(prediction,batch,list(self._argument_observations(prediction,batch)),list(self._scope_observations(prediction,batch)))

    @torch.no_grad()
    def update(self,prediction,batch,audit=None):
        """Consume one batch; keep sufficient statistics, never the tensors.

        I1's read-only ABI/routing validators are reused, not its loss functions
        or values. Detached labels may originally require gradients; metrics do
        not mutate or retain them. Audit fanout occurs after mask-based routing.
        """
        if batch.get('routing',{}).get('normalizer_fingerprint')!=self.normalizer_fingerprint:
            raise ValueError('Metric batch/normalizer identity mismatch')
        view={'inputs':batch['inputs'],'supervision':{family:{k:v.detach() for k,v in values.items()}
            for family,values in batch['supervision'].items()}}
        _validate_supervision(view)
        with self.audit_context(batch,audit):
            self.update_resource_payload(prediction,view);self.update_resource_mode(prediction,view)
            self.update_structural(prediction,view)
            arguments=list(self._argument_observations(prediction,view));scopes=list(self._scope_observations(prediction,view))
            self._record_arguments(arguments);self._record_scope(scopes)
            self._record_tuple(prediction,view,arguments,scopes)
