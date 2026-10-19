"""H2 catalog-driven raw predictions. No targets, normalization fit or decoding.

Only H1 latents and current G2 tensor routing enter neural computation. Static
catalog indices are semantic categories; physical IDs are never accessed.
Regression outputs live in the already-normalized target ABI, without clamps.

Prediction ABI (float32 scores/values, int64 routing):
* resource.mode_candidate indexes inputs.resource_candidates; mode_group_ptr
  has one segment per resource, empty for singleton domains. payload_candidate
  and payload_ptr route flat numeric values by candidate and ABI component.
  categorical.candidate/ptr similarly route catalog-ordered category logits.
* structural.candidate_logits align exactly with inputs.structural bindings;
  group_ptr retains one decision group per (sample, initiator).
* arguments.*.argument indexes the G2 structural argument table. numeric uses
  flat value/ptr (scalar or vector); category uses logits/ptr in ParameterSpec
  category order. boolean has one logit per argument. reference/reference_set
  use logits/ptr/domain_index, where domain_index indexes G2's flat
  argument_domain_index table, not physical entities directly.
* reference_set required_ptr/required_domain_index retain fixed members without
  logits. Scope uses binding/ptr/domain_index into G2 scope_members and its own
  required routing. Fully fixed scopes and bound connector args are unpredicted.
No output contains labels. Catalog identities remain available on ActionHeads;
physical decoding uses external current G2 routing, never neural ID embeddings.
"""
from pathlib import Path
import json

import torch
from torch import nn
from torch.nn import functional as F

from .bc_v1_contract import validate_robot_spec
from .model import _owners,_indices,_pointer,_mean
from .tensorization import PARAMETER_KINDS,FAMILIES


def load_action_head_config(path):
    config=json.loads(Path(path).read_text());_validate_config(config)
    return config


def _validate_config(config):
    if (set(config)!={'schema_version','hidden_dim','activation','dropout'}
            or config['schema_version'] not in ('mssr.bc_v1.action_heads.v1','mssr.bc_v1.action_heads.v2')
            or type(config['hidden_dim']) is not int or config['hidden_dim']!=128
            or config['activation']!='silu' or type(config['dropout']) is not float or config['dropout']!=.1):
        raise ValueError('Unsupported H2 baseline configuration')


def _mlp(source,target,config):
    # Raw final output: no LayerNorm(1), probability transform or clamping.
    return nn.Sequential(nn.Linear(source,config['hidden_dim']),nn.SiLU(),
        nn.Dropout(config['dropout']),nn.Linear(config['hidden_dim'],target))


def _ints(values,like):
    return torch.tensor(values,dtype=torch.int64,device=like.device)


class ActionHeads(nn.Module):
    def __init__(self,spec,latent_config,head_config):
        super().__init__();validate_robot_spec(spec);_validate_config(head_config)
        self.config=dict(head_config);cfg=self.config;h=cfg['hidden_dim'];self.spec=spec
        self.mode_catalog=tuple(sorted((t.type_id,r.resource_id,m.mode_id) for t in spec.module_types for r in t.resources for m in r.modes))
        mode_specs={(t.type_id,r.resource_id,m.mode_id):m for t in spec.module_types for r in t.resources for m in r.modes}
        self.modes=tuple(mode_specs[k] for k in self.mode_catalog)
        self.payload_components=tuple((i,j) for i,m in enumerate(self.modes) if m.codec!='categorical' for j in range(m.payload_size))
        self.component_index={component:i for i,component in enumerate(self.payload_components)}
        self.resource_projection=nn.Linear(latent_config['hidden_dim'],h)
        self.module_projection=nn.Linear(latent_config['hidden_dim'],h)
        self.graph_projection=nn.Linear(latent_config['hidden_dim'],h)
        self.context_projection=nn.Linear(latent_config['context_latent_dim'],h)
        self.observation_projection=nn.Linear(latent_config['observation_latent_dim'],h)
        self.mode_projection=nn.Linear(len(self.modes),h,bias=False) if self.modes else None
        self.component_projection=nn.Linear(len(self.payload_components),h,bias=False) if self.payload_components else None
        multimode=any(len(r.modes)>1 for t in spec.module_types for r in t.resources)
        self.mode_scorer=_mlp(h,1,cfg) if multimode else None
        self.payload_decoder=_mlp(h,1,cfg) if self.payload_components else None
        self.resource_categories=tuple((i,j) for i,m in enumerate(self.modes)
            if m.codec=='categorical' for j in range(len(m.categories)))
        self.resource_category_index={key:i for i,key in enumerate(self.resource_categories)}
        self.resource_category_projection=nn.Linear(len(self.resource_categories),h,bias=False) if self.resource_categories else None
        self.resource_category_scorer=_mlp(h,1,cfg) if self.resource_categories else None
        self.actions=tuple(sorted(spec.actions,key=lambda a:a.action_id))
        self.operation_projection=nn.Linear(len(self.actions),h,bias=False) if self.actions else None
        self.initiator_projection=nn.Linear(latent_config['hidden_dim'],h)
        self.peer_projection=nn.Linear(latent_config['hidden_dim'],h)
        self.connector_a_projection=nn.Linear(latent_config['hidden_dim'],h)
        self.connector_b_projection=nn.Linear(latent_config['hidden_dim'],h)
        self.scope_projection=nn.Linear(latent_config['hidden_dim'],h)
        self.pair_projection=nn.Linear(1,h,bias=False)
        self.structural_scorer=_mlp(h,1,cfg)
        self.no_structural_scorer=(_mlp(h,1,cfg)
            if cfg['schema_version']=='mssr.bc_v1.action_heads.v2' else None)
        self.parameter_catalog=tuple((a.action_id,p.name) for a in self.actions for p in sorted(a.parameters,key=lambda p:p.name))
        self.parameters_catalog=tuple(p for a in self.actions for p in sorted(a.parameters,key=lambda p:p.name))
        self.action_index={a.action_id:i for i,a in enumerate(self.actions)}
        self.learnable_parameters=tuple(i for i,p in enumerate(self.parameters_catalog)
            if p.learnable and self.actions[self.action_index[self.parameter_catalog[i][0]]].scope!='resource')
        self.learnable_index={i:j for j,i in enumerate(self.learnable_parameters)}
        self.parameter_projection=nn.Linear(len(self.learnable_parameters),h,bias=False) if self.learnable_parameters else None
        self.numeric_components=tuple((i,j) for i in self.learnable_parameters
            if self.parameters_catalog[i].kind in ('scalar','vector') for j in range(self.parameters_catalog[i].size or 1))
        self.numeric_index={key:i for i,key in enumerate(self.numeric_components)}
        self.numeric_projection=nn.Linear(len(self.numeric_components),h,bias=False) if self.numeric_components else None
        self.numeric_decoder=_mlp(h,1,cfg) if self.numeric_components else None
        self.reference_option_projection=nn.Linear(latent_config['hidden_dim'],h)
        self.reference_kind_projection=nn.Linear(2,h,bias=False)
        has_sets=any(self.parameters_catalog[i].kind in ('module_refs','connector_refs') for i in self.learnable_parameters)
        self.reference_set_scorer=_mlp(h,1,cfg) if has_sets else None
        # Frozen G2 canonicalizes parameters by name, then binds the first two
        # connector_ref parameters to the ordered connector endpoints.
        bound_keys={(a.action_id,p.name) for a in self.actions if a.scope=='connector_pair'
            for p in tuple(p for p in sorted(a.parameters,key=lambda p:p.name) if p.kind=='connector_ref')[:2]}
        has_references=any(self.parameters_catalog[i].kind in ('module_ref','connector_ref')
            and self.parameter_catalog[i] not in bound_keys for i in self.learnable_parameters)
        self.reference_scorer=_mlp(h,1,cfg) if has_references else None
        self.boolean_scorer=_mlp(h,1,cfg) if any(self.parameters_catalog[i].kind=='boolean' for i in self.learnable_parameters) else None
        self.argument_categories=tuple((i,j) for i in self.learnable_parameters
            if self.parameters_catalog[i].kind=='category' for j in range(len(self.parameters_catalog[i].categories)))
        self.argument_category_index={key:i for i,key in enumerate(self.argument_categories)}
        self.argument_category_projection=nn.Linear(len(self.argument_categories),h,bias=False) if self.argument_categories else None
        self.argument_category_scorer=_mlp(h,1,cfg) if self.argument_categories else None
        has_scope=any(a.scope=='modules' for a in self.actions)
        self.scope_option_projection=nn.Linear(latent_config['hidden_dim'],h) if has_scope else None
        self.scope_scorer=_mlp(h,1,cfg) if has_scope else None
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:nn.init.zeros_(layer.bias)

    @staticmethod
    def _validate_input_boundary(inputs):
        # Same closed G2 input root as H1. The heads use only current routing;
        # neither a full tensorized batch nor a widened policy boundary is legal.
        allowed={'graph_ptr','module_batch','connector_to_module','resource_to_module','coordinate_to_resource',
            'coordinate_slot','connection_connectors','task_one_hot','task_valid','environment','environment_ptr',
            'object_ptr','relation_to_module','relation_geometry_index','relation_geometry_ptr',
            'resource_candidates','structural'}
        allowed.update(f+'_'+suffix for f in FAMILIES for suffix in ('x','validity','ptr'))
        if set(inputs)-allowed or not {'graph_ptr','resource_candidates','structural'}<=set(inputs):
            raise ValueError('Pass only G2 causal inputs, without supervision/audit')

    def forward(self,latents,inputs):
        return {'resource':self.forward_resource(latents,inputs),
            'structural':self.forward_structural(latents,inputs)}

    def _resource_conditions(self,latents,inputs):
        module_graph=_owners(inputs['graph_ptr']);parent=inputs['resource_to_module'];graph=module_graph[parent]
        return (self.resource_projection(latents['resource_h'])+self.module_projection(latents['module_h'][parent])
            +self.graph_projection(latents['graph_h'][graph])+self.context_projection(latents['context_h'][graph])
            +self.observation_projection(latents['observation_h'][graph]))

    def forward_resource(self,latents,inputs):
        """One CE domain per multi-mode resource; no CE output for singletons.

        payload_candidate/payload_ptr map numeric mode-conditioned outputs back
        to existing resource-candidate rows. No expert mode selects a decoder.
        """
        self._validate_input_boundary(inputs)
        rows=inputs['resource_candidates'];r=len(inputs['resource_x']);q=len(rows['resource'])
        _pointer(rows['group_ptr'],q,'resource candidate groups')
        if len(rows['group_ptr'])!=r+1:raise ValueError('Invalid resource decision groups')
        _indices(rows['resource'],r,'resource candidate');_indices(rows['mode'],len(self.modes),'resource mode')
        if rows['mode'].shape!=(q,):raise ValueError('Resource mode shape mismatch')
        owners=_owners(rows['group_ptr'])
        if not torch.equal(owners,rows['resource']):raise ValueError('Resource candidate/group mismatch')
        sizes=[self.modes[i].payload_size for i in rows['mode'].tolist()]
        expected=[0]
        for width in sizes:expected.append(expected[-1]+width)
        if not torch.equal(rows['payload_ptr'],_ints(expected,rows['mode'])):raise ValueError('Catalog payload arity mismatch')
        base=self._resource_conditions(latents,inputs)
        conditioned=base[rows['resource']]
        if q:conditioned=conditioned+self.mode_projection(F.one_hot(rows['mode'],len(self.modes)).float())
        counts=rows['group_ptr'][1:]-rows['group_ptr'][:-1]
        multi=counts>1;chosen=torch.nonzero(multi[rows['resource']],as_tuple=False).flatten()
        mode_ptr=torch.cat((counts.new_zeros(1),torch.cumsum(torch.where(multi,counts,0),0)))
        if len(chosen) and self.mode_scorer is None:raise ValueError('Catalog does not permit a multi-mode resource')
        mode_logits=self.mode_scorer(conditioned[chosen]).flatten() if len(chosen) else base.new_empty(0)
        payload_rows=[i for i,m in enumerate(rows['mode'].tolist()) if self.modes[m].codec!='categorical']
        query=[];components=[];payload_ptr=[0]
        for candidate in payload_rows:
            mode=int(rows['mode'][candidate]);width=self.modes[mode].payload_size
            query.extend([candidate]*width);components.extend(self.component_index[(mode,j)] for j in range(width))
            payload_ptr.append(len(query))
        query=_ints(query,rows['mode']);components=_ints(components,rows['mode'])
        payload=base.new_empty(0)
        if len(query):
            role=self.component_projection(F.one_hot(components,len(self.payload_components)).float())
            payload=self.payload_decoder(conditioned[query]+role).flatten()
        category_rows=[i for i,m in enumerate(rows['mode'].tolist()) if self.modes[m].codec=='categorical']
        queries=[];options=[];category_ptr=[0]
        for candidate in category_rows:
            mode=int(rows['mode'][candidate]);width=len(self.modes[mode].categories)
            queries.extend([candidate]*width);options.extend(self.resource_category_index[(mode,j)] for j in range(width))
            category_ptr.append(len(queries))
        category_logits=base.new_empty(0)
        if queries:
            role=self.resource_category_projection(F.one_hot(_ints(options,rows['mode']),len(self.resource_categories)).float())
            category_logits=self.resource_category_scorer(conditioned[_ints(queries,rows['mode'])]+role).flatten()
        return {'mode_logits':mode_logits,'mode_group_ptr':mode_ptr,'mode_candidate':chosen,
            'payload':payload,'payload_candidate':_ints(payload_rows,rows['mode']),
            'payload_ptr':_ints(payload_ptr,rows['mode']),
            'categorical':{'candidate':_ints(category_rows,rows['mode']),
                'ptr':_ints(category_ptr,rows['mode']),'logits':category_logits}}

    def _structural_conditions(self,latents,inputs):
        s=inputs['structural'];q=len(s['initiator']);m=len(inputs['module_x']);c=len(inputs['connector_x'])
        _pointer(s['group_ptr'],q,'structural groups')
        if len(s['group_ptr'])!=m+1:raise ValueError('Invalid initiator decision groups')
        _indices(s['initiator'],m,'initiator');_indices(s['action'],len(self.actions),'operation')
        for key in ('initiator','action','peer','connector_a','connector_b'):
            if s[key].shape!=(q,):raise ValueError('Invalid structural binding shape')
        if not torch.equal(_owners(s['group_ptr']),s['initiator']):raise ValueError('Initiator/group mismatch')
        paired=s['connector_a']>=0
        if (not torch.equal(paired,s['connector_b']>=0) or not torch.equal(paired,s['peer']>=0)
                or (s['connector_a'][~paired]!=-1).any() or (s['connector_b'][~paired]!=-1).any() or (s['peer'][~paired]!=-1).any()):
            raise ValueError('Inconsistent endpoint applicability')
        _indices(s['peer'][paired],m,'peer')
        _indices(s['connector_a'][paired],c,'source connector');_indices(s['connector_b'][paired],c,'peer connector')
        module_graph=_owners(inputs['graph_ptr']);graph=module_graph[s['initiator']]
        own=inputs['connector_to_module']
        if (not torch.equal(own[s['connector_a'][paired]],s['initiator'][paired])
                or not torch.equal(own[s['connector_b'][paired]],s['peer'][paired])):
            raise ValueError('Ordered endpoint ownership mismatch')
        if (graph[paired]!=module_graph[s['peer'][paired]]).any():raise ValueError('Cross-graph peer')
        _pointer(s['scope_ptr'],len(s['scope_members']),'structural scope')
        if len(s['scope_ptr'])!=q+1:raise ValueError('Scope/binding mismatch')
        _indices(s['scope_members'],m,'scope member');scope_owner=_owners(s['scope_ptr'])
        if (module_graph[s['scope_members']]!=graph[scope_owner]).any():raise ValueError('Cross-graph scope')
        if s['scope_required'].dtype!=torch.bool or s['scope_required'].shape!=s['scope_members'].shape:
            raise ValueError('Invalid required scope mask')
        for i,action_index in enumerate(s['action'].tolist()):
            action=self.actions[action_index]
            if bool(paired[i])!=(action.scope=='connector_pair') or action.scope=='resource':
                raise ValueError('Operation/binding applicability mismatch')
        scope=_mean(latents['module_h'][s['scope_members']],scope_owner,q)
        conditions=(self.initiator_projection(latents['module_h'][s['initiator']])+self.scope_projection(scope)
            +self.graph_projection(latents['graph_h'][graph])+self.context_projection(latents['context_h'][graph])
            +self.observation_projection(latents['observation_h'][graph])+self.pair_projection(paired[:,None].float()))
        if q:conditions=conditions+self.operation_projection(F.one_hot(s['action'],len(self.actions)).float())
        selection=torch.nonzero(paired,as_tuple=False).flatten()
        # Endpoint-free operations never index -1 or create dummy connectors.
        pair=(self.peer_projection(latents['module_h'][s['peer'][selection]])
            +self.connector_a_projection(latents['connector_h'][s['connector_a'][selection]])
            +self.connector_b_projection(latents['connector_h'][s['connector_b'][selection]]))
        return conditions.index_add(0,selection,pair)

    def forward_structural(self,latents,inputs):
        self._validate_input_boundary(inputs)
        conditions=self._structural_conditions(latents,inputs)
        result={'candidate_logits':self.structural_scorer(conditions).flatten(),
            'group_ptr':inputs['structural']['group_ptr'].clone(),
            'arguments':self._arguments(latents,inputs,conditions),
            'scope':self._scope(latents,inputs,conditions)}
        if self.no_structural_scorer is not None:
            graph=_owners(inputs['graph_ptr'])
            none=(self.initiator_projection(latents['module_h'])
                +self.graph_projection(latents['graph_h'][graph])
                +self.context_projection(latents['context_h'][graph])
                +self.observation_projection(latents['observation_h'][graph]))
            result['no_structural_logits']=self.no_structural_scorer(none).flatten()
        return result

    def _scope(self,latents,inputs,conditions):
        """Independent participant membership, never an argument-set alias.

        Fully determined scopes need no prediction. Required members for a
        variable scope are routing only; logits cover its optional members.
        domain_index indexes the existing flat structural scope_members table.
        """
        # Structural conditions have checked current pointers, masks, member
        # ranges and graph ownership. Derive routing afresh; fixed/empty scopes
        # need no Python visit. Keep per-variable-scope neural call shapes and
        # order unchanged, including training dropout RNG consumption.
        s=inputs['structural'];q=len(s['initiator']);owner=_owners(s['scope_ptr'])
        counts=s['initiator'].new_zeros(q)
        counts.scatter_add_(0,owner,(~s['scope_required']).long())
        variable=counts>0;bindings=torch.nonzero(variable,as_tuple=False).flatten()
        optional=torch.nonzero(~s['scope_required'],as_tuple=False).flatten()
        required=torch.nonzero(s['scope_required']&variable[owner],as_tuple=False).flatten()
        ptr=torch.cat((counts.new_zeros(1),counts[bindings].cumsum(0)))
        required_counts=s['scope_ptr'][bindings+1]-s['scope_ptr'][bindings]-counts[bindings]
        required_ptr=torch.cat((counts.new_zeros(1),required_counts.cumsum(0)))
        if len(bindings) and self.scope_scorer is None:
            raise ValueError('Catalog has no variable participant scope')
        scores=[]
        for binding,start,end in zip(bindings.tolist(),ptr[:-1].tolist(),ptr[1:].tolist()):
            chosen=optional[start:end]
            members=s['scope_members'][chosen]
            scores.append(self.scope_scorer(conditions[binding,None]+
                self.scope_option_projection(latents['module_h'][members])).flatten())
        return {'binding':bindings,'logits':torch.cat(scores) if scores else conditions.new_empty(0),
            'ptr':ptr,'domain_index':optional,'required_ptr':required_ptr,'required_domain_index':required}

    def _validate_arguments(self,inputs):
        self._validate_arguments_bulk(inputs)

    def _validate_arguments_bulk(self,inputs):
        """Validate current routing in bulk, including nonlearnable arguments.

        Catalog-sized lookups describe the already validated RobotSpec. Every
        batch-dependent predicate is recomputed on each call: no trusted/stale
        cache and no supervision-dependent shortcut. Scalar reads are bounded
        by the number of checks, never by argument/domain cardinality.
        """
        s=inputs['structural'];u=len(s['argument_binding']);q=len(s['initiator'])
        _indices(s['argument_binding'],q,'argument binding')
        _indices(s['argument_parameter'],len(self.parameter_catalog),'argument parameter')
        _pointer(s['argument_domain_ptr'],len(s['argument_domain_index']),'argument domain')
        if len(s['argument_domain_ptr'])!=u+1:raise ValueError('Argument/domain count mismatch')
        for key in ('argument_binding','argument_parameter','argument_kind','argument_bound_ref'):
            if s[key].dtype!=torch.int64 or s[key].shape!=(u,):raise ValueError('Invalid argument routing shape')
        refs=s['argument_domain_index']
        if refs.dtype!=torch.int64 or refs.ndim!=1:raise ValueError('Invalid argument domain indices')
        if s['argument_domain_required'].dtype!=torch.bool or s['argument_domain_required'].shape!=s['argument_domain_index'].shape:
            raise ValueError('Invalid required argument domain mask')
        kinds=_ints([PARAMETER_KINDS.index(p.kind) for p in self.parameters_catalog],s['argument_kind'])
        actions=_ints([self.action_index[a] for a,_ in self.parameter_catalog],s['argument_parameter'])
        parameter=s['argument_parameter'];binding=s['argument_binding'];kind=s['argument_kind'];bound=s['argument_bound_ref']
        if ((s['action'][binding]!=actions[parameter])|(kind!=kinds[parameter])).any():
            raise ValueError('Argument/action/type mismatch')
        reference=kind>=PARAMETER_KINDS.index('module_ref')
        sizes=s['argument_domain_ptr'][1:]-s['argument_domain_ptr'][:-1]
        if ((~reference)&((sizes!=0)|(bound!=-1))).any():
            raise ValueError('Nonreference argument has reference domain')
        owner=_owners(s['argument_domain_ptr'])
        connector=(kind[owner]==PARAMETER_KINDS.index('connector_ref'))|(kind[owner]==PARAMETER_KINDS.index('connector_refs'))
        limit=torch.where(connector,len(inputs['connector_x']),len(inputs['module_x']))
        if ((refs<0)|(refs>=limit)).any():raise ValueError('Invalid routing argument option')
        parents=refs.clone()
        parents[connector]=inputs['connector_to_module'][refs[connector]]
        module_graph=_owners(inputs['graph_ptr'])
        if (module_graph[parents]!=module_graph[s['initiator'][binding[owner]]]).any():
            raise ValueError('Cross-graph argument option')
        matches=bound.new_zeros(u)
        matches.scatter_add_(0,owner,(refs==bound[owner]).long())
        if (reference&(bound!=-1)&(matches==0)).any():
            raise ValueError('Argument bound reference absent from domain')

    def _parameter_conditions(self,conditions,s,arguments):
        if not isinstance(arguments,torch.Tensor):arguments=_ints(arguments,s['argument_binding'])
        # Catalog lookup only; binding/parameter rows always come from the
        # current, validated batch. No per-argument CUDA -> Python conversion.
        lookup=_ints([self.learnable_index.get(i,-1) for i in range(len(self.parameter_catalog))],arguments)
        params=lookup[s['argument_parameter'][arguments]]
        result=conditions[s['argument_binding'][arguments]]
        if len(arguments):result=result+self.parameter_projection(F.one_hot(params,len(self.learnable_parameters)).float())
        return arguments,result

    def _select_argument_rows(self,s):
        """Current row order, generic kinds/learnable flags; no target access.

        Called after full dynamic argument validation, including fixed fields.
        Bound single references remain routing-only, without new predictions.
        """
        learnable=_ints([p.learnable for p in self.parameters_catalog],s['argument_parameter']).bool()
        learnable=learnable[s['argument_parameter']];kind=s['argument_kind']
        masks={'numeric':(kind==0)|(kind==1),'reference_set':(kind==6)|(kind==7),
            'reference':((kind==4)|(kind==5))&(s['argument_bound_ref']==-1),
            'boolean':kind==2,'category':kind==3}
        return {key:torch.nonzero(learnable&mask,as_tuple=False).flatten() for key,mask in masks.items()}

    def _category_routes(self,s,arguments):
        """Expand catalog categories in legacy row/category order in bulk."""
        parameter=s['argument_parameter'][arguments]
        widths=_ints([len(p.categories) for p in self.parameters_catalog],arguments)[parameter]
        ptr=torch.cat((arguments.new_zeros(1),widths.cumsum(0)))
        query=_owners(ptr)
        first=_ints([self.argument_category_index.get((i,0),0)
            for i in range(len(self.parameter_catalog))],arguments)
        options=first[parameter[query]]+torch.arange(len(query),device=query.device)-ptr[query]
        return query,options,ptr

    def _reference_logits(self,latents,s,argument,condition,domain_indices,scorer):
        refs=s['argument_domain_index'][domain_indices];connector=int(s['argument_kind'][argument]) in (5,7)
        option=latents['connector_h' if connector else 'module_h'][refs]
        kind=option.new_zeros((len(option),2));kind[:,int(connector)]=1
        return scorer(condition[None]+self.reference_option_projection(option)+self.reference_kind_projection(kind)).flatten()

    def _arguments(self,latents,inputs,conditions):
        self._validate_arguments(inputs);s=inputs['structural']
        selected=self._select_argument_rows(s)
        arguments,base=self._parameter_conditions(conditions,s,selected['numeric'])
        query=[];components=[];ptr=[0]
        for row,i in enumerate(arguments.tolist()):
            parameter=int(s['argument_parameter'][i]);width=self.parameters_catalog[parameter].size or 1
            query.extend([row]*width);components.extend(self.numeric_index[(parameter,j)] for j in range(width));ptr.append(len(query))
        value=conditions.new_empty(0)
        if query:
            role=self.numeric_projection(F.one_hot(_ints(components,arguments),len(self.numeric_components)).float())
            value=self.numeric_decoder(base[_ints(query,arguments)]+role).flatten()
        numeric={'argument':arguments,'value':value,'ptr':_ints(ptr,arguments)}
        arguments,base=self._parameter_conditions(conditions,s,selected['reference_set'])
        optional=[];required=[];ptr=[0];required_ptr=[0];scores=[]
        for row,i in enumerate(arguments.tolist()):
            start,end=s['argument_domain_ptr'][i:i+2].tolist();domain=torch.arange(start,end,device=arguments.device)
            mask=s['argument_domain_required'][start:end];chosen=domain[~mask]
            optional.extend(chosen.tolist());required.extend(domain[mask].tolist());ptr.append(len(optional));required_ptr.append(len(required))
            scores.append(self._reference_logits(latents,s,i,base[row],chosen,self.reference_set_scorer))
        members={'argument':arguments,'logits':torch.cat(scores) if scores else conditions.new_empty(0),
            'ptr':_ints(ptr,arguments),'domain_index':_ints(optional,arguments),
            'required_ptr':_ints(required_ptr,arguments),'required_domain_index':_ints(required,arguments)}
        arguments,base=self._parameter_conditions(conditions,s,selected['reference'])
        domain_indices=[];ptr=[0];scores=[]
        for row,i in enumerate(arguments.tolist()):
            start,end=s['argument_domain_ptr'][i:i+2].tolist();domain=torch.arange(start,end,device=arguments.device)
            domain_indices.extend(domain.tolist());ptr.append(len(domain_indices))
            if self.reference_scorer is None:raise ValueError('A catalog-bound reference cannot become an unbound prediction')
            scores.append(self._reference_logits(latents,s,i,base[row],domain,self.reference_scorer))
        references={'argument':arguments,'logits':torch.cat(scores) if scores else conditions.new_empty(0),
            'ptr':_ints(ptr,arguments),'domain_index':_ints(domain_indices,arguments)}
        arguments,base=self._parameter_conditions(conditions,s,selected['boolean'])
        booleans={'argument':arguments,'logits':self.boolean_scorer(base).flatten() if len(arguments) else conditions.new_empty(0)}
        arguments,base=self._parameter_conditions(conditions,s,selected['category'])
        queries,options,ptr=self._category_routes(s,arguments)
        logits=conditions.new_empty(0)
        if len(queries):
            role=self.argument_category_projection(F.one_hot(options,len(self.argument_categories)).float())
            logits=self.argument_category_scorer(base[queries]+role).flatten()
        categories={'argument':arguments,'ptr':ptr,'logits':logits}
        return {'numeric':numeric,'reference_set':members,'reference':references,'boolean':booleans,'category':categories}
