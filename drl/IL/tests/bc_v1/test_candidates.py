"""E2 Task5: observable candidate behavior, independent of labels and hardware.

Literal synthetic counts catch missing/extra choices; invariance tests compare
physical routing sets after explicit renaming, not implementation helpers.
"""

from dataclasses import FrozenInstanceError, replace
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c

ROOT = Path(__file__).resolve().parents[4]


def api():
    name = 'drl.IL.bc_v1.candidates'
    assert importlib.util.find_spec(name) is not None, 'Missing Task5 candidate enumeration'
    return importlib.import_module(name)


def schema():
    from drl.IL.bc_v1.feature_schema import load_feature_schema
    return load_feature_schema(ROOT/'configs/bc_v1/features.json')


def catalog(ports=1):
    connectors = tuple(c.ConnectorSpec(f'socket-{i}', 'fixed_port', capabilities=('couple',)) for i in range(ports))
    resource = c.ResourceSpec('rotor', (
        c.CoordinateSpec('hinge', 'rad', (-2., 2.)),
        c.CoordinateSpec('roller', 'rad', periodic=True)), (
        c.ResourceModeSpec('pace', 2, ('rad/s', 'rad/s')),
        c.ResourceModeSpec('aim', 1, ('rad',), kind='discrete')), ('motor',))
    arguments = (c.ParameterSpec('endpoint_a', 'connector_ref', module_type_id='kit'),
                 c.ParameterSpec('endpoint_b', 'connector_ref', module_type_id='kit'),
                 c.ParameterSpec('turn', 'category', categories=('zero', 'quarter')))
    actions = (
        c.ActionSpec('pace', 'continuous', 'resource', (
            c.ParameterSpec('speeds', 'vector', 'rad/s', size=2, module_type_id='kit', resource_id='rotor'),), ('motor',)),
        c.ActionSpec('aim', 'discrete', 'resource', (
            c.ParameterSpec('angle', 'scalar', 'rad', limits=(-2., 2.), module_type_id='kit', resource_id='rotor'),), ('motor',)),
        c.ActionSpec('align_faces', 'prepare_connection', 'connector_pair', arguments, ('couple',)),
        c.ActionSpec('dock', 'connect', 'connector_pair', arguments, ('couple',)),
        c.ActionSpec('undock', 'disconnect', 'connector_pair', arguments, ('couple',)),
        c.ActionSpec('gravity_settle', 'skill', 'modules', (
            c.ParameterSpec('duration', 'scalar', 's'),
            c.ParameterSpec('members', 'module_refs', module_type_id='kit'),
            c.ParameterSpec('clear', 'boolean', learnable=False))))
    return c.RobotSpec('mssr.bc_v1.robot_spec.v1', 'generic', (c.ModuleTypeSpec('kit', connectors, (resource,)),), actions)


def graph(modules=2, ports=1):
    refs = tuple(c.ModuleRef(f'unit-{i}') for i in range(modules))
    return c.StateGraph(
        tuple(c.ModuleState(r, 'kit', (float(i), 0., 0.), (0., 0., 0., 1.), pose_valid=True) for i,r in enumerate(refs)),
        tuple(c.ConnectorState(c.ConnectorRef(r, f'socket-{j}'), 'fixed_port', status='available') for r in refs for j in range(ports)),
        tuple(c.ResourceState(c.ResourceRef(r, 'rotor'), (c.CoordinateState('hinge', .2), c.CoordinateState('roller'))) for r in refs))


def enumerate_(g=None, spec=None):
    return api().enumerate_candidates(graph() if g is None else g, catalog() if spec is None else spec, schema())


def structural(tables, name):
    return tuple(x for x in tables.structural_candidates if x.action.action_id == name)


def attach(g, a=0, b=1, relation='current_connection', pointers=True):
    left, right = g.connectors[a].ref, g.connectors[b].ref
    connectors = tuple(replace(x, status='connected', connected_to=right if x.ref==left else left)
                       if pointers and x.ref in (left,right) else x for x in g.connectors)
    return replace(g, connectors=connectors, connections=(c.ConnectionState(left, right, relation),))


def test_candidate_api_has_no_supervision_or_future_argument():
    fn = api().enumerate_candidates
    assert tuple(inspect.signature(fn).parameters) == ('graph', 'spec', 'schema')
    assert all(p.kind not in (p.VAR_KEYWORD,p.VAR_POSITIONAL) for p in inspect.signature(fn).parameters.values())
    for key in ('supervision', 'labels', 'future_graph', 'target_morphology'):
        with pytest.raises(TypeError): fn(graph(), catalog(), schema(), **{key: object()})


@pytest.mark.parametrize('field', ['expert_action','supervision_target','supervision_mask',
    'module_resource_masks','target_morphology','final_graph','G_t_plus_1','future_state',
    'reward','success','outcome','behavioral_stage','executor_fsm','nav2_path','route','waypoints',
    'centerline','training_split'])
def test_candidates_same_when_forbidden_metadata_changes(field):
    g=graph(); before=enumerate_(g)
    object.__setattr__(g,field,{'forbidden':12345})
    assert enumerate_(g)==before


def test_resource_candidates_come_only_from_current_resources_and_catalog_actions():
    g=replace(graph(),resources=graph().resources[:1]); table=enumerate_(g)
    assert len(table.resource_candidates)==2
    assert {x.mode.mode_id for x in table.resource_candidates}=={'pace','aim'}
    assert all(x.resource_ref==c.ResourceRef(c.ModuleRef('unit-0'),'rotor') for x in table.resource_candidates)


@pytest.mark.parametrize('n,ports,want_resources,want_pairs',[(0,0,0,0),(1,7,2,0),(2,1,4,1),(3,2,6,12),(5,3,10,90),(12,1,24,66)])
def test_resource_and_connector_counts_are_variable(n,ports,want_resources,want_pairs):
    # Empty current graph still uses a valid hardware capability catalog.
    table=enumerate_(graph(n,ports),catalog(ports or 1))
    assert len(table.resource_candidates)==want_resources
    assert len(structural(table,'align_faces'))==want_pairs
    assert len(structural(table,'dock'))==want_pairs
    assert len(structural(table,'gravity_settle'))==(1 if n else 0)


def test_resource_mode_keeps_kind_codec_units_limits_and_current_coordinate_routing():
    table=enumerate_(); item=next(x for x in table.resource_candidates if x.mode.mode_id=='aim')
    assert item.mode.kind=='discrete' and item.mode.payload_size==1 and item.mode.units==('rad',)
    assert item.action.parameters[0].limits==(-2.,2.)
    assert item.coordinates[0].coordinate_id=='hinge' and item.coordinates[0].limits==(-2.,2.)
    assert item.coordinates[1].periodic
    assert item.coordinate_refs==frozenset((item.resource_ref,name) for name in ('hinge','roller'))
    assert item.capabilities==('motor',)


def test_resource_coordinates_are_observables_not_invented_mode_argument_binding():
    g=replace(graph(),resources=(replace(graph().resources[0],coordinates=()),))
    table=enumerate_(g)
    assert len(table.resource_candidates)==2
    assert all(not x.coordinate_refs and len(x.coordinates)==2 for x in table.resource_candidates)


def test_resource_mode_requires_matching_enabled_action_and_capabilities():
    spec=catalog(); actions=tuple(replace(a,enabled=False) if a.action_id=='pace' else a for a in spec.actions)
    assert {x.mode.mode_id for x in enumerate_(spec=replace(spec,actions=actions)).resource_candidates}=={'aim'}
    assert not enumerate_(spec=replace(spec,actions=tuple(a for a in spec.actions if a.scope!='resource'))).resource_candidates


def test_resource_capability_is_not_borrowed_from_another_current_resource():
    spec=catalog(); owner=spec.module_types[0]
    bare=replace(owner.resources[0],capabilities=())
    other=c.ResourceSpec('helper',capabilities=('motor',))
    spec=replace(spec,module_types=(replace(owner,resources=(bare,other)),))
    assert not enumerate_(spec=spec).resource_candidates


def test_resource_modes_are_generic_even_when_resource_and_coordinate_names_change():
    spec=catalog(); owner=spec.module_types[0]; resource=owner.resources[0]
    updated=replace(resource,resource_id='arbitrary_actuator',coordinates=(
        replace(resource.coordinates[0],coordinate_id='qA'),replace(resource.coordinates[1],coordinate_id='qB')))
    actions=tuple(replace(a,parameters=tuple(replace(p,resource_id='arbitrary_actuator') for p in a.parameters))
                  if a.scope=='resource' else a for a in spec.actions)
    spec=replace(spec,module_types=(replace(owner,resources=(updated,)),),actions=actions)
    g=graph(); g=replace(g,resources=tuple(replace(r,ref=replace(r.ref,resource_id='arbitrary_actuator'),
        coordinates=(c.CoordinateState('qA'),c.CoordinateState('qB'))) for r in g.resources))
    table=enumerate_(g,spec)
    assert len(table.resource_candidates)==4
    assert {name for item in table.resource_candidates for _,name in item.coordinate_refs}=={'qA','qB'}


def test_categorical_resource_modes_preserve_spec_declared_choices():
    spec=catalog(); owner=spec.module_types[0]; res=owner.resources[0]
    mode=c.ResourceModeSpec('latch',1,('1',),kind='discrete',codec='categorical',categories=('open','closed'))
    res=replace(res,modes=(mode,))
    action=c.ActionSpec('latch','discrete','resource',(c.ParameterSpec('selection','category',categories=('open','closed'),module_type_id='kit',resource_id='rotor'),))
    table=enumerate_(spec=replace(spec,module_types=(replace(owner,resources=(res,)),),actions=(action,)))
    assert len(table.resource_candidates)==2
    assert all(x.mode.categories==('open','closed') for x in table.resource_candidates)


def test_align_and_dock_use_compatible_free_endpoints_without_reversed_duplicate_rows():
    table=enumerate_()
    for name in ('align_faces','dock'):
        row,=structural(table,name)
        assert row.endpoints==frozenset(x.ref for x in graph().connectors)
        assert len(row.bindings)==2  # distinct initiator decisions inside one physical pair
        assert {x.initiator for x in row.bindings}==frozenset(m.ref for m in graph().modules)
        for binding in row.bindings:
            assert binding.connectors[0].module==binding.initiator
            assert binding.arguments[0].bound_ref==binding.connectors[0]
            assert binding.arguments[1].bound_ref==binding.connectors[1]
            assert binding.arguments[2].parameter.categories==('zero','quarter')


def test_align_faces_is_not_pruned_by_distance_or_invalid_pose():
    g=graph(); changed=replace(g,modules=(replace(g.modules[0],pose_valid=False),replace(g.modules[1],position=(1e8,1e8,1e8))))
    assert enumerate_(g)==enumerate_(changed)


def test_candidate_action_catalog_has_no_internal_executor_subactions():
    spec=catalog(); extras=tuple(c.ActionSpec(name,'skill','modules') for name in ('REACH','ALIGN','CLOCKING','APPROACH','WAIT','no_dispatch'))
    table=enumerate_(spec=replace(spec,actions=spec.actions+extras))
    assert {x.action.action_id for x in table.structural_candidates}=={'align_faces','dock','gravity_settle'}


def test_missing_connector_capability_prevents_pair_admission():
    spec=catalog(2); owner=spec.module_types[0]
    spec=replace(spec,module_types=(replace(owner,connectors=(owner.connectors[0],replace(owner.connectors[1],capabilities=()))),))
    table=enumerate_(graph(2,2),spec)
    row,=structural(table,'align_faces')
    assert all(x.connector_id=='socket-0' for x in row.endpoints)


def test_pair_reference_constraints_allow_only_declared_endpoint_roles():
    spec=catalog(2); owner=spec.module_types[0]
    changed=[]
    for a in spec.actions:
        if a.scope=='connector_pair':
            a=replace(a,parameters=(replace(a.parameters[0],connector_id='socket-0'),replace(a.parameters[1],connector_id='socket-1'),*a.parameters[2:]))
        changed.append(a)
    table=enumerate_(graph(2,2),replace(spec,actions=tuple(changed)))
    rows=structural(table,'dock')
    assert len(rows)==2 and all(len(x.bindings)==1 for x in rows)
    assert all(next(iter(x.bindings)).connectors[0].connector_id=='socket-0' for x in rows)
    assert all(next(iter(x.bindings)).connectors[1].connector_id=='socket-1' for x in rows)


def test_self_and_same_module_structural_pairs_are_not_generated():
    rows=structural(enumerate_(graph(2,3),catalog(3)),'align_faces')
    assert len(rows)==9
    assert all(len(x.endpoints)==2 and len({e.module for e in x.endpoints})==2 for x in rows)


@pytest.mark.parametrize('pointers',[False,True])
def test_current_connection_is_undock_only_even_with_missing_connector_pointers(pointers):
    table=enumerate_(attach(graph(),pointers=pointers))
    assert not structural(table,'align_faces') and not structural(table,'dock')
    row,=structural(table,'undock')
    assert row.endpoints==frozenset(x.ref for x in graph().connectors)
    assert all(x.applicability=='available' for x in row.bindings)


def test_connected_to_physical_link_is_undock_candidate_without_duplicate_graph_edge():
    g=attach(graph()); table=enumerate_(replace(g,connections=()))
    assert len(structural(table,'undock'))==1 and not structural(table,'dock')


def test_undock_never_includes_disconnected_pairs():
    assert structural(enumerate_(),'undock')==()


@pytest.mark.parametrize('relation',['contact','rolling_contact'])
def test_unattached_contact_does_not_masquerade_as_physical_connection(relation):
    table=enumerate_(attach(graph(),relation=relation,pointers=False))
    assert len(structural(table,'dock'))==1 and not structural(table,'undock')


def test_connected_endpoint_cannot_dock_to_another_free_peer():
    g=attach(graph(3),0,1); table=enumerate_(g)
    assert not structural(table,'dock') and len(structural(table,'undock'))==1


def test_status_connected_without_known_peer_does_not_invent_undock_pair():
    g=graph(); g=replace(g,connectors=(replace(g.connectors[0],status='connected'),g.connectors[1]))
    table=enumerate_(g)
    assert not structural(table,'dock') and not structural(table,'undock')


def test_unknown_connector_status_is_explicitly_not_certified_runtime_feasible():
    g=graph(); g=replace(g,connectors=tuple(replace(x,status='unknown') for x in g.connectors))
    row,=structural(enumerate_(g),'dock')
    assert {x.applicability for x in row.bindings}=={'unknown'}


def test_gravity_settle_has_no_endpoints_and_keeps_selectable_scope_and_arguments():
    row,=structural(enumerate_(graph(3),catalog()),'gravity_settle')
    assert row.endpoints==frozenset() and len(row.bindings)==3
    current=frozenset(m.ref for m in graph(3).modules)
    for b in row.bindings:
        assert not b.connectors and b.scope_members==current and b.required_scope_members==frozenset((b.initiator,))
        domain=next(x for x in b.arguments if x.parameter.kind=='module_refs')
        assert domain.options==current
        by_name={d.parameter.name:d.parameter for d in b.arguments}
        assert by_name['duration'].units=='s' and by_name['clear'].learnable is False


def test_gravity_settle_module_scope_and_disabled_setting_are_spec_driven():
    spec=catalog(); actions=tuple(replace(a,scope='module',parameters=a.parameters[:1]) if a.action_id=='gravity_settle' else a for a in spec.actions)
    row,=structural(enumerate_(spec=replace(spec,actions=actions)),'gravity_settle')
    assert all(b.scope_members==frozenset((b.initiator,)) for b in row.bindings)
    disabled=tuple(replace(a,enabled=False) if a.action_id=='gravity_settle' else a for a in spec.actions)
    assert not structural(enumerate_(spec=replace(spec,actions=disabled)),'gravity_settle')


def test_gravity_scope_reference_set_size_and_module_type_constraints_are_preserved():
    spec=catalog(); actions=tuple(replace(a,parameters=tuple(replace(p,size=3) if p.kind=='module_refs' else p for p in a.parameters)) if a.action_id=='gravity_settle' else a for a in spec.actions)
    assert not structural(enumerate_(spec=replace(spec,actions=actions)),'gravity_settle')
    row,=structural(enumerate_(graph(3),replace(spec,actions=actions)),'gravity_settle')
    assert all(next(d for d in b.arguments if d.parameter.kind=='module_refs').parameter.size==3 for b in row.bindings)


def test_joint_decision_can_represent_multiple_align_and_dock_pairs():
    table=enumerate_(graph(4),catalog())
    desired={frozenset((c.ConnectorRef(c.ModuleRef('unit-0'),'socket-0'),c.ConnectorRef(c.ModuleRef('unit-1'),'socket-0'))),
             frozenset((c.ConnectorRef(c.ModuleRef('unit-2'),'socket-0'),c.ConnectorRef(c.ModuleRef('unit-3'),'socket-0')))}
    for name in ('align_faces','dock'):
        assert desired <= {x.endpoints for x in structural(table,name)}


def renamed(g,spec):
    modules={m.ref:c.ModuleRef('renamed-'+str(i*7+31)) for i,m in enumerate(g.modules)}
    names={p.connector_id:'port-renamed-'+str(i*13+19) for i,p in enumerate(spec.module_types[0].connectors)}
    def port(r):return c.ConnectorRef(modules[r.module],names[r.connector_id])
    owner=spec.module_types[0]; updated=replace(spec,module_types=(replace(owner,connectors=tuple(replace(p,connector_id=names[p.connector_id]) for p in owner.connectors)),))
    changed=replace(g,modules=tuple(replace(m,ref=modules[m.ref]) for m in g.modules),
        connectors=tuple(replace(p,ref=port(p.ref),connected_to=None if p.connected_to is None else port(p.connected_to)) for p in g.connectors),
        resources=tuple(replace(r,ref=replace(r.ref,module=modules[r.ref.module])) for r in g.resources),
        connections=tuple(replace(e,endpoint_a=port(e.endpoint_a),endpoint_b=port(e.endpoint_b)) for e in g.connections))
    return changed,updated,{v:k for k,v in modules.items()},{v:k for k,v in names.items()}


def routing_semantics(tables, module_map=None, connector_map=None):
    module_map=module_map or {};connector_map=connector_map or {}
    def mid(r):return module_map.get(r,r).module_id
    def port(r):return (mid(r.module),connector_map.get(r.connector_id,r.connector_id))
    resources=frozenset((x.action.action_id,mid(x.resource_ref.module),x.resource_ref.resource_id,x.mode.mode_id,
        frozenset((mid(r.module),r.resource_id,name) for r,name in x.coordinate_refs)) for x in tables.resource_candidates)
    structural_set=frozenset((x.action.action_id,frozenset(port(r) for r in x.endpoints),
        frozenset((mid(b.initiator),tuple(port(r) for r in b.connectors),frozenset(mid(r) for r in b.scope_members)) for b in x.bindings)) for x in tables.structural_candidates)
    return resources,structural_set


def test_candidate_set_is_invariant_to_module_connector_rename_and_input_reorder():
    g=attach(graph(3,2),0,2);spec=catalog(2);before=enumerate_(g,spec)
    changed,new_spec,mod_map,port_map=renamed(g,spec)
    assert routing_semantics(enumerate_(changed,new_spec),mod_map,port_map)==routing_semantics(before)
    reverse=replace(g,modules=g.modules[::-1],connectors=g.connectors[::-1],resources=tuple(replace(r,coordinates=r.coordinates[::-1]) for r in g.resources[::-1]),connections=tuple(replace(e,endpoint_a=e.endpoint_b,endpoint_b=e.endpoint_a) for e in g.connections[::-1]))
    other=enumerate_(reverse,spec)
    assert frozenset(other.resource_candidates)==frozenset(before.resource_candidates)
    assert frozenset(other.structural_candidates)==frozenset(before.structural_candidates)


def test_surface_connectors_are_not_treated_as_four_fixed_faces():
    spec=catalog();owner=spec.module_types[0]
    port=replace(owner.connectors[0],kind='surface',geometry=(c.NumericFeature('radius_m',(.05,)),))
    g=graph();g=replace(g,connectors=tuple(replace(x,kind='surface') for x in g.connectors))
    table=enumerate_(g,replace(spec,module_types=(replace(owner,connectors=(port,)),)))
    assert len(structural(table,'align_faces'))==1


def test_candidate_values_are_immutable_and_contain_no_learned_id_vector():
    table=enumerate_();r=table.resource_candidates[0];s=table.structural_candidates[0]
    with pytest.raises(FrozenInstanceError):r.resource_ref=None
    with pytest.raises(FrozenInstanceError):s.endpoints=frozenset()
    assert isinstance(table.resource_candidates,tuple) and isinstance(s.bindings,frozenset)
    assert not hasattr(r,'numeric') and not hasattr(s,'numeric')


@pytest.mark.parametrize('fault',['duplicate_module','dangling_connector','unknown_resource','unknown_coordinate','unknown_connector','kind_mismatch','dangling_edge','self_edge','contradictory_peer'])
def test_invalid_catalog_routing_or_topology_is_rejected(fault):
    g=graph()
    if fault=='duplicate_module':g=replace(g,modules=g.modules+g.modules[:1])
    elif fault=='dangling_connector':g=replace(g,connectors=(replace(g.connectors[0],ref=c.ConnectorRef(c.ModuleRef('absent'),'socket-0')),))
    elif fault=='unknown_resource':g=replace(g,resources=(replace(g.resources[0],ref=replace(g.resources[0].ref,resource_id='absent')),))
    elif fault=='unknown_coordinate':g=replace(g,resources=(replace(g.resources[0],coordinates=(c.CoordinateState('absent'),)),))
    elif fault=='unknown_connector':g=replace(g,connectors=(replace(g.connectors[0],ref=replace(g.connectors[0].ref,connector_id='absent')),))
    elif fault=='kind_mismatch':g=replace(g,connectors=(replace(g.connectors[0],kind='surface'),g.connectors[1]))
    elif fault=='dangling_edge':g=replace(g,connections=(c.ConnectionState(g.connectors[0].ref,c.ConnectorRef(c.ModuleRef('absent'),'socket-0'),'current_connection'),))
    elif fault=='self_edge':g=replace(g,connections=(c.ConnectionState(g.connectors[0].ref,g.connectors[0].ref,'current_connection'),))
    elif fault=='contradictory_peer':g=attach(graph(3),0,1);g=replace(g,connectors=(replace(g.connectors[0],connected_to=g.connectors[2].ref),*g.connectors[1:]))
    with pytest.raises(ValueError):enumerate_(g)


def test_candidate_import_isolation_during_actual_enumeration():
    script='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'rclpy','isaacsim','isaaclab','omni','torch','numpy','smores_ep','freebot_docking','mssr_expert'} or fullname in {'drl.IL.bc_v1.smores_adapter','drl.IL.bc_v1.task_context','drl.IL.bc_v1.features','drl.IL.bc_v1.policy_inputs','drl.IL.preprocessing'}:
   raise ImportError('candidate isolation: '+fullname)
sys.meta_path.insert(0,Block())
from drl.IL.bc_v1.bc_v1_contract import *
from drl.IL.bc_v1.feature_schema import load_feature_schema
from drl.IL.bc_v1.candidates import enumerate_candidates
s=RobotSpec('mssr.bc_v1.robot_spec.v1','generic',(ModuleTypeSpec('node'),),(ActionSpec('gravity_settle','skill','modules'),))
g=StateGraph((ModuleState(ModuleRef('node-any'),'node',(0.,0.,0.),(0.,0.,0.,1.)),))
t=enumerate_candidates(g,s,load_feature_schema('configs/bc_v1/features.json'))
assert len(t.structural_candidates)==1 and not t.structural_candidates[0].endpoints
assert len(t.structural_candidates[0].bindings)==1
'''
    # Import path existence is asserted before launching, giving valid RED.
    api()
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.fixture(scope='module')
def current_real_rows(tmp_path_factory):
    from drl.IL.tests.bc_v1.test_policy_inputs import real_policy_rows
    yield from real_policy_rows.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize('case',['assembly','gap','stairs','button','navigation','terminal-goal'])
def test_six_real_current_samples_enumerate_without_labels(current_real_rows,case):
    row,g,_,_,_=current_real_rows[case]
    spec=c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
    table=enumerate_(g,spec)
    assert len(table.resource_candidates)==56
    assert {x.action.action_id for x in table.resource_candidates}=={'wheel_velocity','pan_velocity','tilt_velocity','pan_target','tilt_target','pan_delta','tilt_delta'}
    assert len(structural(table,'gravity_settle'))==1
    assert len(structural(table,'gravity_settle')[0].bindings)==8
    assert {x.resource_ref for x in table.resource_candidates}=={r.ref for r in g.resources}
    assert all({e.module for e in x.endpoints} <= {m.ref for m in g.modules} for x in table.structural_candidates)
    if case=='assembly':
        assert not structural(table,'undock') and len(structural(table,'align_faces'))==448
        assert len(structural(table,'dock'))==448
    else:assert len(structural(table,'undock'))==7


@pytest.mark.parametrize('operation',['align_faces','dock','undock','gravity_settle'])
def test_supported_b1_structural_targets_have_label_independent_bindings(operation):
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.tests.bc_v1.fixtures import smores_graph,smores_label
    spec=c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json');adapter=SMORESAdapter(spec)
    raw=smores_graph()
    if operation=='undock':raw['edges']=[{'module_a_id':'moving','face_a':'TOP','module_b_id':'peer','face_b':'BOTTOM','relation':'current_connection'}]
    g=adapter.encode_state(raw);table=enumerate_(g,spec)  # FIRST, no labels
    action,masks=smores_label(operation)
    target,=adapter.encode_labels(action,masks,()).structural_targets
    rows=structural(table,operation)
    matches=[b for x in rows if x.endpoints==frozenset(target.connectors) for b in x.bindings
             if b.initiator==target.initiator and (b.connectors==target.connectors if target.connectors else set(target.participants)<=b.scope_members)]
    assert len(matches)==1
    for parameter in target.parameters:
        domain=next(x for x in matches[0].arguments if x.parameter.name==parameter.name)
        if isinstance(parameter.value,c.ModuleRef):assert parameter.value in domain.options
        elif isinstance(parameter.value,tuple) and parameter.value and isinstance(parameter.value[0],c.ModuleRef):assert set(parameter.value)<=domain.options


def test_parameter_catalog_reorder_preserves_directional_bindings_and_named_arguments():
    spec=catalog(2)
    actions=tuple(replace(a,parameters=(replace(a.parameters[0],connector_id='socket-0'),
        replace(a.parameters[1],connector_id='socket-1'),*a.parameters[2:]))
        if a.scope=='connector_pair' else a for a in spec.actions)
    original=replace(spec,actions=actions)
    reordered=replace(original,actions=tuple(replace(a,parameters=a.parameters[::-1]) for a in original.actions[::-1]))
    assert c.robot_spec_fingerprint(original)==c.robot_spec_fingerprint(reordered)
    before=enumerate_(graph(2,2),original);after=enumerate_(graph(2,2),reordered)
    assert routing_semantics(before)==routing_semantics(after)
    assert frozenset(before.structural_candidates)==frozenset(after.structural_candidates)
    assert frozenset(before.resource_candidates)==frozenset(after.resource_candidates)
    for row in structural(after,'dock'):
        for binding in row.bindings:
            by_name={d.parameter.name:d.bound_ref for d in binding.arguments}
            assert by_name['endpoint_a']==binding.connectors[0]
            assert by_name['endpoint_b']==binding.connectors[1]


def assert_supported_target(table,target):
    """Independent audit AFTER enumeration, including argument domains/limits."""
    matches=[b for x in structural(table,target.operation_id) if x.endpoints==frozenset(target.connectors)
        for b in x.bindings if b.initiator==target.initiator and
        (b.connectors==target.connectors if target.connectors else set(target.participants)<=b.scope_members)]
    assert len(matches)==1
    b=matches[0]
    assert b.required_scope_members<=set(target.participants)<=b.scope_members
    for value in target.parameters:
        domain=next(d for d in b.arguments if d.parameter.name==value.name);p=domain.parameter
        if p.kind in ('module_ref','connector_ref'):
            assert value.value in domain.options
            if domain.bound_ref is not None:assert value.value==domain.bound_ref
        elif p.kind in ('module_refs','connector_refs'):
            assert set(value.value)<=domain.options
            if p.size is not None:assert len(value.value)==p.size
        elif p.kind=='category':assert value.value in p.categories
        elif p.kind=='boolean':assert type(value.value) is bool
        else:
            components=value.value if p.kind=='vector' else (value.value,)
            if p.kind=='vector':assert len(components)==p.size
            assert all(type(x) in (int,float) for x in components)
            if p.limits is not None:assert all(p.limits[0]<=x<=p.limits[1] for x in components)


@pytest.mark.parametrize('case',['assembly','gap','stairs','button','navigation','terminal-goal'])
def test_real_b1_targets_fit_precomputed_current_candidates_and_argument_domains(current_real_rows,case):
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter,LabelEvidenceRequired,read_label_evidence
    row,g,_,_,_=current_real_rows[case]
    spec=c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json');adapter=SMORESAdapter(spec)
    table=enumerate_(g,spec)  # before any label decoding
    try:bundle=adapter.encode_labels(row['action_t'],row['supervision'],())
    except LabelEvidenceRequired:bundle=adapter.encode_labels(row['action_t'],row['supervision'],read_label_evidence(row,ROOT))
    assert len(bundle.structural_targets)==(2 if case in ('assembly','navigation') else 1)
    for target in bundle.structural_targets:assert_supported_target(table,target)


def test_first_real_assembly_dock_is_representable_on_current_pre_dock_graph():
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter,LabelEvidenceRequired,read_label_evidence
    path=ROOT/'datasets/expert_v1_canonical_v6/episodes/expert__assembly__seed-006217.jsonl'
    if not path.is_file():pytest.skip('Canonical v6 unavailable')
    def signature():
        s=path.stat();return s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    before=signature()
    with path.open() as stream:
        for i,line in enumerate(stream):
            if i==469:row=json.loads(line);break
        else:pytest.fail('Missing audited assembly DOCK row469')
    spec=c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json');adapter=SMORESAdapter(spec)
    g=adapter.encode_state(row['graph_t']);table=enumerate_(g,spec)  # no labels
    assert not g.connections
    try:bundle=adapter.encode_labels(row['action_t'],row['supervision'],())
    except LabelEvidenceRequired:bundle=adapter.encode_labels(row['action_t'],row['supervision'],read_label_evidence(row,ROOT))
    target,=bundle.structural_targets
    assert target.operation_id=='dock' and target.initiator.module_id=='smores_03'
    assert_supported_target(table,target)
    assert signature()==before


def test_reference_domains_are_not_recomputed_for_every_connector_pair(monkeypatch):
    """A current topology should not rebuild identical reference domains per pair."""
    module = api()
    original = module._reference_options
    calls = []

    def counted(parameter, owners, ports):
        calls.append((parameter.name, parameter.kind))
        return original(parameter, owners, ports)

    monkeypatch.setattr(module, '_reference_options', counted)

    table = module.enumerate_candidates(graph(8, 4), catalog(4), schema())

    # Semantic contract is unchanged: all physical inter-module pairs remain.
    assert len(structural(table, 'align_faces')) == 448
    assert len(structural(table, 'dock')) == 448

    # Domains depend on action + current topology, not on every individual pair.
    assert len(calls) <= 64, (
        f'identical current reference domains were recomputed {len(calls)} times'
    )
