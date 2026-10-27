"""Generic exact joint resource selection, physical payloads outside the actor.

Factorized scores are maximized jointly per module over causal feasible
outcomes. INACTIVE has no claims or physical payload. Busy/unknown resources
are skipped (including withdrawal), never treated as an available writer.
"""
from .resource_decisions import INACTIVE,FIXED,validate_outcome_inputs,select_joint
from .bc_v1_contract import ActionChoice


def resolve_resource_outcomes(prediction,inputs,routing,*,contract,normalizers,availability=None,claim_resolver=None,decode_policy=None):
    from .action_decoder import _routes,_inverse
    rows=inputs['resource_candidates'];samples=routing.get('samples',(routing,))
    refs=tuple(r for route in samples for r in route['resource_refs'])
    candidates=tuple(r for route in samples for r in route['resource_candidates'])
    validate_outcome_inputs(rows,contract,len(refs))
    if any(route.get('resource_contract_fingerprint')!=contract.fingerprint for route in samples):raise ValueError('Resource contract fingerprint mismatch')
    if decode_policy is not None:decode_policy.validate_contract(contract)
    compatibility=contract if decode_policy is None else decode_policy
    p=prediction['resource'];scores=p['outcome_logits'].detach().cpu().tolist()
    if len(scores)!=len(rows['outcome_kind']):raise ValueError('Outcome score domain mismatch')
    values=_routes(dict(p,ptr=p['payload_ptr']),'payload_candidate','payload');cats=_routes(p['categorical'],'candidate','logits')
    bindings={(b.module_type_id,b.resource_id,b.mode_id):b for b in normalizers.resource_bindings}
    payloads={};claims_by_outcome={};domains={};ro=qo=0
    for si,route in enumerate(samples):
        av=availability[si] if availability is not None else None
        for local,ref in enumerate(route['resource_refs']):
            r=ro+local
            if av is not None and not av.free((ref,)):continue
            a,b=rows['outcome_ptr'][r:r+2].tolist();domain=[]
            for i in range(a,b):
                o=contract.outcomes[int(rows['outcome_catalog'][i])];q=int(rows['outcome_candidate'][i])
                if o.kind==INACTIVE:payloads[i]=();claims_by_outcome[i]=();domain.append(i);continue
                if av is not None and not av.resource_mask[q-qo]:continue
                row=candidates[q]
                if o.kind==FIXED:payload=o.payload
                elif row.mode.codec=='categorical':payload=(float(max(range(len(cats[q])),key=lambda k:cats[q][k])),)
                else:
                    binding=bindings.get((o.module_type_id,o.resource_id,o.mode_id))
                    if binding is None or binding.units!=row.mode.units or binding.codec!=row.mode.codec:raise ValueError('Missing/incompatible F2 physical binding')
                    payload=_inverse(values[q],binding.keys,normalizers)
                numeric=[p for p in row.action.parameters if p.kind in ('scalar','vector')]
                limited=[p for p in numeric if p.limits is not None]
                if limited and (len(numeric)!=1 or (numeric[0].size or 1)!=len(payload)):raise ValueError('Ambiguous physical limits')
                if limited and any(not limited[0].limits[0]<=v<=limited[0].limits[1] for v in payload):continue
                claims=()
                if av is not None and claim_resolver is not None:
                    claims=claim_resolver(ActionChoice(row.action.action_id,ref.module,resource=ref,payload=payload))
                    if not av.free(claims):continue
                payloads[i]=payload;claims_by_outcome[i]=claims;domain.append(i)
            if not domain:raise ValueError('No feasible resource outcome')
            domains[r]=domain
        ro+=len(route['resource_refs']);qo+=len(route['resource_candidates'])
    # Decompose independent modules, merging only when backend claims couple them.
    # Exact enumeration is exponential in the largest coupled component.
    parent={r:r for r in domains}
    def find(r):
        while parent[r]!=r:r=parent[r]
        return r
    def join(a,b):parent[find(a)]=find(b)
    seen_modules={};seen_claims={}
    for r,domain in domains.items():
        module=int(inputs['resource_to_module'][r])
        if module in seen_modules:join(r,seen_modules[module])
        seen_modules[module]=r
        for i in domain:
            for claim in claims_by_outcome[i]:
                if claim in seen_claims:join(r,seen_claims[claim])
                seen_claims[claim]=r
    module_groups={}
    for r in domains:module_groups.setdefault(find(r),[]).append(r)
    selected={};pre=post=0
    def compatible(choice):
        claimed=set()
        for i in choice:
            claims=claims_by_outcome[i]
            if claimed.intersection(claims):return False
            claimed.update(claims)
        by_module={}
        for i in choice:
            module=int(inputs['resource_to_module'][int(rows['outcome_resource'][i])])
            by_module.setdefault(module,[]).append((contract.outcomes[int(rows['outcome_catalog'][i])],payloads[i]))
        return all(compatibility.compatible(rows) for rows in by_module.values())
    for resources in module_groups.values():
        ds=[domains[r] for r in resources]
        independent=tuple(max(d,key=lambda i:scores[i]) for d in ds)
        pre+=int(not compatible(independent));joint=select_joint(ds,scores,compatible);post+=int(not compatible(joint))
        for r,i in zip(resources,joint):selected[r]=(i,int(rows['outcome_candidate'][i]),payloads[i])
    return selected,{'support':len(module_groups),'pre_violations':pre,'post_violations':post}
