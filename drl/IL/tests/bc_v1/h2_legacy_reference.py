"""Test-only pre-P1 validator oracle; never imported by production."""
import torch
from drl.IL.bc_v1.model import _indices, _pointer, _owners
from drl.IL.bc_v1.tensorization import PARAMETER_KINDS


def legacy_validate_arguments(self,inputs):
        s=inputs['structural'];u=len(s['argument_binding']);q=len(s['initiator'])
        _indices(s['argument_binding'],q,'argument binding')
        _indices(s['argument_parameter'],len(self.parameter_catalog),'argument parameter')
        _pointer(s['argument_domain_ptr'],len(s['argument_domain_index']),'argument domain')
        if len(s['argument_domain_ptr'])!=u+1:raise ValueError('Argument/domain count mismatch')
        for key in ('argument_parameter','argument_kind','argument_bound_ref'):
            if s[key].dtype!=torch.int64 or s[key].shape!=(u,):raise ValueError('Invalid argument routing shape')
        if s['argument_domain_required'].dtype!=torch.bool or s['argument_domain_required'].shape!=s['argument_domain_index'].shape:
            raise ValueError('Invalid required argument domain mask')
        module_graph=_owners(inputs['graph_ptr'])
        for i,parameter_index in enumerate(s['argument_parameter'].tolist()):
            p=self.parameters_catalog[parameter_index];binding=int(s['argument_binding'][i])
            expected_action=self.action_index[self.parameter_catalog[parameter_index][0]]
            if int(s['action'][binding])!=expected_action or int(s['argument_kind'][i])!=PARAMETER_KINDS.index(p.kind):
                raise ValueError('Argument/action/type mismatch')
            start,end=s['argument_domain_ptr'][i:i+2].tolist();refs=s['argument_domain_index'][start:end]
            if p.kind in ('module_ref','module_refs','connector_ref','connector_refs'):
                connector=p.kind.startswith('connector');limit=len(inputs['connector_x']) if connector else len(inputs['module_x'])
                _indices(refs,limit,'argument option')
                parents=inputs['connector_to_module'][refs] if connector else refs
                if (module_graph[parents]!=module_graph[s['initiator'][binding]]).any():raise ValueError('Cross-graph argument option')
                bound=int(s['argument_bound_ref'][i])
                if bound!=-1 and bound not in refs.tolist():raise ValueError('Argument bound reference absent from domain')
            elif end!=start or int(s['argument_bound_ref'][i])!=-1:
                raise ValueError('Nonreference argument has reference domain')
