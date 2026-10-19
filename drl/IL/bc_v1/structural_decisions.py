"""Policy-level categorical abstention. No physical action, IDs or robot semantics."""
import torch

VERSION='mssr.bc_v1.structural_decisions.v2'
NO_STRUCTURAL=-1


def _positive_targets(y,ptr):
    count=len(ptr)-1;target=ptr.new_full((count,),NO_STRUCTURAL);seen=set()
    if len({len(y[k]) for k in ('group','candidate','supervised')})!=1:raise ValueError('Invalid positive row counts')
    for group,candidate,valid in zip(y['group'].tolist(),y['candidate'].tolist(),y['supervised'].tolist()):
        if not valid:continue
        if group in seen:raise ValueError('Duplicate positive structural initiator')
        if not 0<=group<count:raise ValueError('Invalid positive group')
        lo,hi=ptr[group:group+2].tolist()
        if not lo<=candidate<hi:raise ValueError('Positive outside initiator domain')
        seen.add(group);target[group]=candidate
    return target


def add_no_structural_supervision(sample):
    """Preserve positive/argument rows; add one decision for EVERY current module."""
    y=sample['supervision']['structural'];s=sample['inputs']['structural'];count=len(s['group_ptr'])-1
    if count!=len(sample['inputs']['module_x']):raise ValueError('Structural group/module mismatch')
    if any(k.startswith('decision_') for k in y):raise ValueError('Decision supervision already exists')
    target=_positive_targets(y,s['group_ptr'])
    decision=dict(y,decision_group=torch.arange(count,dtype=torch.int64,device=target.device),
        decision_candidate=target,decision_supervised=torch.ones(count,dtype=torch.bool,device=target.device))
    return dict(sample,supervision=dict(sample['supervision'],structural=decision))


def joint_decisions(prediction,batch):
    """Validate dense targets; yield (group, joint logits, local CE target)."""
    p=prediction['structural'];y=batch['supervision']['structural'];ptr=p['group_ptr'];none=p['no_structural_logits']
    count=len(ptr)-1
    if none.shape!=(count,):raise ValueError('Invalid NO_STRUCTURAL logit groups')
    for key,dtype in [('decision_group',torch.int64),('decision_candidate',torch.int64),('decision_supervised',torch.bool)]:
        if key not in y or y[key].shape!=(count,) or y[key].dtype!=dtype:raise ValueError('Missing/invalid dense decision supervision')
    if not torch.equal(y['decision_group'],torch.arange(count,device=ptr.device)):raise ValueError('Decision groups must cover every initiator once')
    if not torch.equal(y['decision_candidate'],_positive_targets(y,ptr)):raise ValueError('Dense decision mapping changed preserved positive targets')
    for g,(candidate,valid) in enumerate(zip(y['decision_candidate'].tolist(),y['decision_supervised'].tolist())):
        if not valid:continue
        lo,hi=ptr[g:g+2].tolist()
        if candidate!=NO_STRUCTURAL and not lo<=candidate<hi:raise ValueError('Structural target outside joint domain')
        logits=torch.cat((none[g:g+1],p['candidate_logits'][lo:hi]))
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite joint structural logits')
        yield g,logits,0 if candidate==NO_STRUCTURAL else candidate-lo+1


class StructuralDecisionMetrics:
    """Streaming confusion counts. NONE ties win, matching deployment selection."""
    def __init__(self,operations):
        self.counts={'negative':0,'positive':0,'false':0,'missed':0,'exact_positive':0,'correct_operation':0}
        self.ops={op:{'support':0,'activated':0,'operation_correct':0,'exact':0} for op in operations}

    def add(self,target,prediction,exact):
        c=self.counts
        if target is None:
            c['negative']+=1;c['false']+=int(prediction is not None)
        else:
            if target not in self.ops:raise ValueError('Unknown positive operation')
            c['positive']+=1;c['missed']+=int(prediction is None);c['exact_positive']+=int(exact)
            c['correct_operation']+=int(target==prediction)
            row=self.ops[target];row['support']+=1;row['activated']+=int(prediction is not None)
            row['operation_correct']+=int(target==prediction);row['exact']+=int(exact)

    def compute(self):
        c=self.counts;tn=c['negative']-c['false'];tp=c['positive']-c['missed']
        def ratio(a,b):return a/b if b else None
        return {'no_structural':{'support':c['negative'],'accuracy':ratio(tn,c['negative']),
            'precision':ratio(tn,tn+c['missed']),'recall':ratio(tn,c['negative'])},
            'positive':{'support':c['positive'],'accuracy':ratio(c['exact_positive'],c['positive']),
            'recall':ratio(tp,c['positive']),'operation_accuracy':ratio(c['correct_operation'],c['positive'])},
            'false_activation_rate':ratio(c['false'],c['negative']),
            'missed_activation_rate':ratio(c['missed'],c['positive']),'counts':dict(c),
            'by_operation':{op:{'support':r['support'],'accuracy':ratio(r['exact'],r['support']),
                'operation_recall':ratio(r['operation_correct'],r['support']),
                'activation_recall':ratio(r['activated'],r['support'])} for op,r in self.ops.items()}}
