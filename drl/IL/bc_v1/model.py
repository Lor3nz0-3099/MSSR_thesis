"""H1 instantaneous backbone over G2 inputs only; no labels or hardware imports.

Widths and closed semantic vocabularies come from RobotSpec, FeatureSchema and
the versioned G2 layout. Instance refs, candidates and audit are never encoded.
Geometry tokens retain their semantic slot, primitive kind, scope and object
ownership. Variable-arity primitives use shared component encoding with fixed
component-role codes (array positions, never object/module identifiers).
"""
from pathlib import Path
import json
import math

import torch
from torch import nn

from .bc_v1_contract import validate_robot_spec
from .feature_schema import validate_feature_schema
from .tensorization import FAMILIES, _width, load_tensor_layout


def load_model_config(path):
    config=json.loads(Path(path).read_text())
    _validate_config(config)
    return config


def _validate_config(config):
    dimensions=('hidden_dim','coordinate_latent_dim','edge_latent_dim','relation_latent_dim',
        'environment_latent_dim','context_latent_dim','observation_latent_dim','task_projection_dim')
    fixed={'schema_version':'mssr.bc_v1.gnn.v1','message_passing_layers':3,'activation':'silu',
        'aggregation':'mean','graph_pooling':['mean','max'],'use_residual':True,'normalization':'layernorm'}
    if set(config)!=set(dimensions)|set(fixed)|{'tensor_layout','dropout'}:
        raise ValueError('Unknown/missing H1 config keys')
    if any(type(config[k]) is not int or config[k]<1 for k in dimensions):
        raise ValueError('Invalid latent dimensions')
    if any(type(config[k]) is not type(v) or config[k]!=v for k,v in fixed.items()):
        raise ValueError('Unsupported H1 architecture')
    if type(config['dropout']) not in (float,int) or not 0<=config['dropout']<1:
        raise ValueError('Invalid dropout')
    if type(config['tensor_layout']) is not str or not config['tensor_layout']:
        raise ValueError('Missing tensor layout')


def _mlp(source,target,config):
    return nn.Sequential(nn.Linear(source,config['hidden_dim']),nn.SiLU(),
        nn.Dropout(config['dropout']),nn.Linear(config['hidden_dim'],target),nn.LayerNorm(target))


def _mean(values,owner,count):
    result=values.new_zeros((count,values.shape[1]))
    result.index_add_(0,owner,values)
    numbers=values.new_zeros(count)
    numbers.index_add_(0,owner,values.new_ones(len(values)))
    return result/numbers.clamp_min(1)[:,None]


def _owners(ptr):
    return torch.repeat_interleave(torch.arange(len(ptr)-1,device=ptr.device),ptr[1:]-ptr[:-1])


def _indices(value,size,name):
    if value.dtype!=torch.int64 or ((value<0)|(value>=size)).any():
        raise ValueError('Invalid routing '+name)


def _pointer(value,total,name,*,nonempty=False):
    if (value.dtype!=torch.int64 or value.ndim!=1 or len(value)<1 or int(value[0])!=0
            or int(value[-1])!=total or (value[1:]<value[:-1]).any()
            or nonempty and (len(value)<2 or (value[1:]==value[:-1]).any())):
        raise ValueError('Invalid pointer '+name)


def _packed(x,validity,width,name):
    if (x.dtype!=torch.float32 or x.ndim!=2 or x.shape[1]!=2*width
            or validity.dtype!=torch.bool or validity.shape!=(len(x),width)
            or not torch.isfinite(x).all() or not torch.equal(x[:,width:],validity.float())
            or (x[:,:width][~validity]!=0).any()):
        raise ValueError('Invalid G2 features '+name)


class MessageBlock(nn.Module):
    """Two message families, separate means, residual module update.

    Edge features ALWAYS describe the published ordered A,B pair in A's body
    frame. Both directions consume this canonical descriptor. The two appended
    role channels explicitly identify whether the receiver is A or B. They do
    not claim that the geometry is a sender->receiver inverse pose.
    """
    def __init__(self,config):
        super().__init__();h=config['hidden_dim']
        self.physical_message=_mlp(2*h+config['edge_latent_dim']+2,h,config)
        self.environment_message=_mlp(h+config['environment_latent_dim']+config['relation_latent_dim'],h,config)
        self.update=_mlp(3*h+config['context_latent_dim']+config['observation_latent_dim']+2,h,config)
        self.norm=nn.LayerNorm(h)

    def forward(self,h,inputs,local):
        ends=inputs['connector_to_module'][inputs['connection_connectors']]
        sender=torch.cat((ends[:,0],ends[:,1]));receiver=torch.cat((ends[:,1],ends[:,0]))
        roles=h.new_tensor([[0.,1.],[1.,0.]]).repeat_interleave(len(ends),0)
        physical=self.physical_message(torch.cat((h[receiver],h[sender],local['connection_h'].repeat(2,1),roles),1))
        physical=_mean(physical,receiver,len(h))
        relation_owner=inputs['relation_to_module']
        geometry=_mean(local['environment_h'][inputs['relation_geometry_index']],
            _owners(inputs['relation_geometry_ptr']),len(relation_owner))
        environment=self.environment_message(torch.cat((h[relation_owner],geometry,local['relation_h']),1))
        environment=_mean(environment,relation_owner,len(h))
        graph=_owners(inputs['graph_ptr'])
        update=self.update(torch.cat((h,physical,environment,local['context_h'][graph],local['observation_h'][graph],
            local['context_present'][graph,None].float(),local['observation_present'][graph,None].float()),1))
        return self.norm(h+update)


class BCGNN(nn.Module):
    """Shared, stateless entity encoders; feed-forward physical backbone."""
    def __init__(self,spec,feature_schema,model_config,*,action_head_config=None):
        super().__init__()
        validate_robot_spec(spec);validate_feature_schema(feature_schema);_validate_config(model_config)
        self.config=dict(model_config)
        cfg=self.config;self.schema=feature_schema
        root=Path(__file__).resolve().parents[3]
        self.layout=load_tensor_layout(root/cfg['tensor_layout'])
        self.widths={family:_width(spec,feature_schema,family) for family in FAMILIES}
        self.slots=self.layout['semantic_slots']
        self.kinds=tuple(sorted(feature_schema['geometry_kinds']))
        self.object_width=len(feature_schema['entities']['environment']['fields'][1]['categories'])
        dimensions={family:cfg[{'coordinate':'coordinate_latent_dim','connection':'edge_latent_dim',
            'relation':'relation_latent_dim'}.get(family,'hidden_dim')] for family in FAMILIES}
        self.encoders=nn.ModuleDict({f:_mlp(2*self.widths[f],dimensions[f],cfg) for f in FAMILIES})
        ed=cfg['environment_latent_dim']
        self.geometry_encoders=nn.ModuleDict({kind:_mlp(2*entry['dimension'],ed,cfg)
            for kind,entry in feature_schema['geometry_kinds'].items() if type(entry['dimension']) is int})
        self.component_encoder=_mlp(6,ed,cfg)
        self.environment_encoder=_mlp(ed+2*self.object_width+len(self.slots)+len(self.kinds)+2,ed,cfg)
        self.object_encoder=_mlp(ed,ed,cfg)
        self.task_projection=nn.Linear(len(self.layout['task_vocabulary']),cfg['task_projection_dim'])
        self.context_encoder=_mlp(cfg['task_projection_dim']+ed+2,cfg['context_latent_dim'],cfg)
        self.observation_encoder=_mlp(ed+1,cfg['observation_latent_dim'],cfg)
        self.resource_fusion=_mlp(cfg['hidden_dim']+cfg['coordinate_latent_dim'],cfg['hidden_dim'],cfg)
        self.module_fusion=_mlp(3*cfg['hidden_dim']+cfg['context_latent_dim']+cfg['observation_latent_dim']+2,cfg['hidden_dim'],cfg)
        self.message_blocks=nn.ModuleList(MessageBlock(cfg) for _ in range(cfg['message_passing_layers']))
        self.graph_projection=nn.Linear(2*cfg['hidden_dim'],cfg['hidden_dim'])
        self.graph_norm=nn.LayerNorm(cfg['hidden_dim'])
        contextual_width=3*cfg['hidden_dim']+cfg['context_latent_dim']+cfg['observation_latent_dim']+2
        self.connector_context=_mlp(contextual_width,cfg['hidden_dim'],cfg)
        self.resource_context=_mlp(contextual_width,cfg['hidden_dim'],cfg)
        self._initialize_weights()
        # Optional H2 composition leaves the standalone H1 initialization/API
        # intact. Lazy import avoids a module cycle with shared routing helpers.
        self.action_heads=None
        if action_head_config is not None:
            from .action_heads import ActionHeads
            self.action_heads=ActionHeads(spec,self.config,action_head_config)

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)

    def _validate_inputs(self,x):
        allowed={'graph_ptr','module_batch','connector_to_module','resource_to_module','coordinate_to_resource',
            'coordinate_slot','connection_connectors','task_one_hot','task_valid','environment','environment_ptr',
            'object_ptr','relation_to_module','relation_geometry_index','relation_geometry_ptr',
            'resource_candidates','structural'}
        allowed.update(f+'_'+suffix for f in FAMILIES for suffix in ('x','validity','ptr'))
        if set(x)-allowed or not {'graph_ptr','environment'}<=set(x):
            raise ValueError('Pass only G2 causal inputs, without supervision/audit')
        for family in FAMILIES:_packed(x[family+'_x'],x[family+'_validity'],self.widths[family],family)
        _pointer(x['graph_ptr'],len(x['module_x']),'graph_ptr',nonempty=True)
        b=len(x['graph_ptr'])-1
        if (x['task_one_hot'].dtype!=torch.float32 or x['task_one_hot'].shape!=(b,len(self.layout['task_vocabulary']))
                or not torch.isfinite(x['task_one_hot']).all() or x['task_valid'].dtype!=torch.bool
                or x['task_valid'].shape!=(b,)):
            raise ValueError('Invalid task layout')
        for name,parent,child in [('connector_to_module','module','connector'),('resource_to_module','module','resource'),
                ('coordinate_to_resource','resource','coordinate'),('relation_to_module','module','relation')]:
            _indices(x[name],len(x[parent+'_x']),name)
            if x[name].shape!=(len(x[child+'_x']),):raise ValueError('Ownership shape mismatch')
        _indices(x['connection_connectors'],len(x['connector_x']),'connection_connectors')
        if x['connection_connectors'].shape!=(len(x['connection_x']),2):raise ValueError('Invalid endpoint shape')
        module_graph=_owners(x['graph_ptr'])
        ends=x['connector_to_module'][x['connection_connectors']]
        if len(ends) and (module_graph[ends[:,0]]!=module_graph[ends[:,1]]).any():raise ValueError('Cross-graph physical edge')
        # Global range checks are insufficient: a child must also belong to
        # the graph declared by its G2 collection boundary. Single G2 samples
        # have only graph_ptr; collate publishes the additional family pointers.
        parent_graph={
            'connector':module_graph[x['connector_to_module']],
            'resource':module_graph[x['resource_to_module']],
            'coordinate':module_graph[x['resource_to_module'][x['coordinate_to_resource']]],
            'connection':module_graph[ends[:,0]],
            'relation':module_graph[x['relation_to_module']]}
        for family,actual in parent_graph.items():
            key=family+'_ptr'
            if key not in x:
                if b>1:raise ValueError('Missing batched family pointer '+key)
                continue
            _pointer(x[key],len(x[family+'_x']),key)
            if len(x[key])!=b+1:raise ValueError('Invalid batch pointer '+key)
            if not torch.equal(_owners(x[key]),actual):raise ValueError('Cross-graph '+family+' ownership')
        if 'module_batch' in x and not torch.equal(x['module_batch'],module_graph):
            raise ValueError('Inconsistent module graph routing')
        _pointer(x['environment_ptr'],int(x['environment_ptr'][-1]),'environment_ptr')
        _pointer(x['object_ptr'],int(x['object_ptr'][-1]),'object_ptr')
        if len(x['environment_ptr'])!=b+1 or len(x['object_ptr'])!=b+1:raise ValueError('Invalid environment batch boundaries')
        _pointer(x['relation_geometry_ptr'],len(x['relation_geometry_index']),'relation_geometry_ptr')
        if len(x['relation_geometry_ptr'])!=len(x['relation_x'])+1:raise ValueError('Invalid relation geometry pointers')
        _indices(x['relation_geometry_index'],int(x['environment_ptr'][-1]),'relation_geometry_index')
        relation_owner=_owners(x['relation_geometry_ptr'])
        if (_owners(x['environment_ptr'])[x['relation_geometry_index']]!=module_graph[x['relation_to_module'][relation_owner]]).any():
            raise ValueError('Cross-graph environment relation')

    def _environment(self,x):
        """Semantic encoders independent of module features and instance names."""
        reference=x['module_x'];cfg=self.config;ed=cfg['environment_latent_dim']
        t=int(x['environment_ptr'][-1]);o=int(x['object_ptr'][-1]);b=len(x['graph_ptr'])-1
        env=reference.new_zeros((t,ed));valid=torch.zeros(t,dtype=torch.bool,device=reference.device)
        objects=torch.zeros(t,dtype=torch.int64,device=reference.device)
        scope=torch.zeros(t,dtype=torch.int64,device=reference.device)
        seen=[];token_graph=_owners(x['environment_ptr']);object_graph=_owners(x['object_ptr'])
        for key,row in x['environment'].items():
            parts=key.split(':')
            if len(parts)!=4 or parts[0] not in ('c_static','o_dynamic') or parts[1] not in self.slots or parts[2] not in self.kinds:
                raise ValueError('Unknown G2 semantic geometry layout')
            source,slot,kind,arity=parts;d=int(arity)
            declared=self.schema['geometry_kinds'][kind]['dimension']
            if d<1 or type(declared) is int and d!=declared:raise ValueError('Invalid primitive arity')
            if set(row)!={'x','validity','token_index','object_index','graph_index'}:raise ValueError('Invalid environment record keys')
            packed=row['x'];width=d+self.object_width;_packed(packed,row['validity'],width,key)
            for name,size in [('token_index',t),('object_index',o),('graph_index',b)]:
                _indices(row[name],size,name)
                if row[name].shape!=(len(packed),):raise ValueError('Environment routing shape mismatch')
            if (token_graph[row['token_index']]!=row['graph_index']).any() or (object_graph[row['object_index']]!=row['graph_index']).any():
                raise ValueError('Cross-graph environment ownership')
            geometry=torch.cat((packed[:,:d],packed[:,width:width+d]),1)
            if kind in self.geometry_encoders:geom=self.geometry_encoders[kind](geometry)
            else:
                positions=torch.arange(d,device=packed.device,dtype=packed.dtype)
                role=torch.stack((positions.sin(),positions.cos(),(positions/10).sin(),(positions/10).cos()),1)
                components=torch.cat((packed[:,:d,None],packed[:,width:width+d,None],role.expand(len(packed),d,4)),2)
                geom=self.component_encoder(components).mean(1)
            semantics=packed.new_zeros((len(packed),len(self.slots)+len(self.kinds)+2))
            semantics[:,self.slots.index(slot)]=1
            semantics[:,len(self.slots)+self.kinds.index(kind)]=1
            semantics[:,len(self.slots)+len(self.kinds)+(source=='o_dynamic')]=1
            categories=torch.cat((packed[:,d:width],packed[:,width+d:]),1)
            token_valid=row['validity'].any(1)
            latent=self.environment_encoder(torch.cat((geom,categories,semantics),1))*token_valid[:,None]
            env=env.index_copy(0,row['token_index'],latent)
            valid=valid.index_copy(0,row['token_index'],token_valid)
            objects=objects.index_copy(0,row['token_index'],row['object_index'])
            scope=scope.index_fill(0,row['token_index'],int(source=='o_dynamic'))
            seen.extend(row['token_index'].tolist())
        if sorted(seen)!=list(range(t)):raise ValueError('Duplicate/missing environment token')
        object_h=_mean(env[valid],objects[valid],o)
        static_count=reference.new_zeros(o);dynamic_count=reference.new_zeros(o)
        static_count.index_add_(0,objects[(scope==0)&valid],reference.new_ones(int(((scope==0)&valid).sum())))
        dynamic_count.index_add_(0,objects[(scope==1)&valid],reference.new_ones(int(((scope==1)&valid).sum())))
        if ((static_count>0)&(dynamic_count>0)).any():raise ValueError('Mixed source scopes in one object')
        # The nonlinear object encoder preserves attribute pairing. Without it,
        # two consecutive means collapse equal-sized objects into a token bag.
        object_h=self.object_encoder(object_h)*((static_count>0)|(dynamic_count>0))[:,None]
        pools=[];presences=[]
        for count in (static_count,dynamic_count):
            chosen=count>0;pools.append(_mean(object_h[chosen],object_graph[chosen],b))
            numbers=reference.new_zeros(b);numbers.index_add_(0,object_graph[chosen],reference.new_ones(int(chosen.sum())))
            presences.append(numbers>0)
        return env,valid,pools,presences

    def encode_local(self,inputs):
        self._validate_inputs(inputs)
        result={('module_local_h' if f=='module' else f+'_local_h' if f in ('connector','resource') else f+'_h'):
            encoder(inputs[f+'_x']) for f,encoder in self.encoders.items()}
        coordinates=_mean(result['coordinate_h'],inputs['coordinate_to_resource'],len(inputs['resource_x']))
        result['resource_local_h']=self.resource_fusion(torch.cat((result['resource_local_h'],coordinates),1))
        env,valid,(static,dynamic),(static_present,dynamic_present)=self._environment(inputs)
        task=self.task_projection(inputs['task_one_hot'])*inputs['task_valid'][:,None]
        context_present=static_present|inputs['task_valid']
        context=self.context_encoder(torch.cat((task,static,inputs['task_valid'][:,None].float(),static_present[:,None].float()),1))
        observation=self.observation_encoder(torch.cat((dynamic,dynamic_present[:,None].float()),1))
        result.update(environment_h=env,environment_valid=valid,task_h=task,
            context_h=context*context_present[:,None],observation_h=observation*dynamic_present[:,None],
            context_present=context_present,observation_present=dynamic_present)
        return result

    def initialize_modules(self,inputs,local):
        """Separate invariant child means, conditioned before physical messages."""
        m=len(inputs['module_x']);owner=_owners(inputs['graph_ptr'])
        resources=_mean(local['resource_local_h'],inputs['resource_to_module'],m)
        connectors=_mean(local['connector_local_h'],inputs['connector_to_module'],m)
        return self.module_fusion(torch.cat((local['module_local_h'],resources,connectors,
            local['context_h'][owner],local['observation_h'][owner],
            local['context_present'][owner,None].float(),local['observation_present'][owner,None].float()),1))

    def propagate(self,inputs,local,module_h):
        for block in self.message_blocks:module_h=block(module_h,inputs,local)
        return module_h

    def contextualize(self,inputs,local,module_h):
        """Global pooling occurs only after the physical blocks.

        Module states are never updated from this pool. Contextual connector/
        resource outputs may consume it, as required by the latent contract.
        """
        owner=_owners(inputs['graph_ptr']);b=len(inputs['graph_ptr'])-1
        average=_mean(module_h,owner,b)
        maximum=module_h.new_full((b,module_h.shape[1]),-math.inf)
        maximum.scatter_reduce_(0,owner[:,None].expand_as(module_h),module_h,reduce='amax',include_self=True)
        graph=self.graph_norm(self.graph_projection(torch.cat((average,maximum),1)))
        result={k:v for k,v in local.items() if not k.endswith('_local_h')}
        result.update(module_h=module_h,graph_h=graph)
        for family,encoder in [('connector',self.connector_context),('resource',self.resource_context)]:
            parent=inputs[family+'_to_module'];g=owner[parent]
            result[family+'_h']=encoder(torch.cat((local[family+'_local_h'],module_h[parent],graph[g],
                local['context_h'][g],local['observation_h'][g],
                local['context_present'][g,None].float(),local['observation_present'][g,None].float()),1))
        return result

    def encode(self,inputs):
        """G2 inputs -> float32 latents plus bool presence/validity; no action.

        module/connector/resource/graph: H; coordinate: coordinate_latent_dim;
        connection/relation/environment: their configured latent dimensions;
        task/context/observation: per graph. No routing/audit argument exists.
        """
        local=self.encode_local(inputs)
        initial=self.initialize_modules(inputs,local)
        return self.contextualize(inputs,local,self.propagate(inputs,local,initial))

    def forward_heads(self,latents,inputs):
        """H1 latents + current G2 inputs -> H2 raw, explicitly routed tensors."""
        if self.action_heads is None:raise ValueError('H2 action heads are not configured')
        return self.action_heads(latents,inputs)

    def forward(self,inputs):
        return self.forward_heads(self.encode(inputs),inputs)
