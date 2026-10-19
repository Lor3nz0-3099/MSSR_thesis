"""E1 physical transforms: hand-derived geometry, no learned feature schema."""

import copy
from dataclasses import FrozenInstanceError, asdict, replace
import importlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.policy_inputs import select_policy_inputs
from drl.IL.tests.bc_v1 import fixtures as f
from drl.IL.tests.bc_v1.test_policy_inputs import real_policy_rows, _REAL_CASES

ROOT = Path(__file__).resolve().parents[4]
I = (1., 0., 0., 0., 1., 0., 0., 0., 1.)
RZ = (0., -1., 0., 1., 0., 0., 0., 0., 1.)
RX = (1., 0., 0., 0., 0., -1., 0., 1., 0.)


def api():
    name = "drl.IL.bc_v1.features"
    assert importlib.util.find_spec(name) is not None, "Missing E1 causal physical transforms"
    return importlib.import_module(name)


def inputs(task="button", static=(), dynamic=(), graph=None):
    return select_policy_inputs(graph if graph is not None else f.state_graph(2, 1),
                                c.ObservationBundle(dynamic), c.TaskContext(task, static))


def cfg(rotation=I, **kwargs):
    return api().PhysicalFrameConfig(rotation_source_from_axes=rotation, **kwargs)


def frame(rotation=I):
    return api().physical_frame(f.state_graph(3, 0), cfg(rotation))


def quantity(view, name, scope="c_static"):
    return next(q for q in view.quantities if q.source_name == name and q.source_scope == scope)


@pytest.mark.parametrize("n", [1, 2, 5, 8, 12])
def test_frame_arithmetic_centroid_variable_n(n):
    result = api().physical_frame(f.state_graph(n, 0), cfg())
    assert result.valid
    assert result.origin == ((n-1)/2, 0., 0.)
    assert result.rotation_source_from_axes == I


def test_frame_excludes_invalid_poses_without_inventing_origin():
    graph = f.state_graph(3, 0)
    invalid = replace(graph.modules[1], position=(float("nan"), 999., 999.), pose_valid=False)
    result = api().physical_frame(replace(graph, modules=(graph.modules[0], invalid, graph.modules[2])), cfg())
    assert result.origin == (1., 0., 0.)
    empty = api().physical_frame(replace(graph, modules=(invalid,)), cfg())
    assert not empty.valid and empty.origin is None
    assert api().transform_point((5., 0., 0.), empty) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_frame_valid_nonfinite_position_rejected(bad):
    graph = f.state_graph(1, 0)
    graph = replace(graph, modules=(replace(graph.modules[0], position=(bad, 0., 0.)),))
    with pytest.raises(ValueError):
        api().physical_frame(graph, cfg())


def test_frame_reordering_centroid_and_entity_geometry():
    original = inputs(graph=f.state_graph(5, 3))
    shuffled = replace(original, graph=replace(original.graph, modules=original.graph.modules[::-1]))
    a, b = (api().transform_inputs(x, cfg(RZ)) for x in (original, shuffled))
    assert a.frame == b.frame
    assert {m.ref: m for m in a.modules} == {m.ref: m for m in b.modules}


def renamed(graph, module_ids=True, connector_ids=True):
    mapping = {m.ref: c.ModuleRef(f"renamed-{len(graph.modules)-i}") for i, m in enumerate(graph.modules)}
    def mr(ref): return mapping[ref] if module_ids else ref
    def cr(ref): return c.ConnectorRef(mr(ref.module), 'renamed-'+ref.connector_id if connector_ids else ref.connector_id)
    return c.StateGraph(
        tuple(replace(m, ref=mr(m.ref)) for m in graph.modules),
        tuple(replace(p, ref=cr(p.ref), connected_to=None if p.connected_to is None else cr(p.connected_to)) for p in graph.connectors),
        tuple(replace(r, ref=c.ResourceRef(mr(r.ref.module), r.ref.resource_id)) for r in graph.resources),
        tuple(replace(e, endpoint_a=cr(e.endpoint_a), endpoint_b=cr(e.endpoint_b)) for e in graph.connections))


@pytest.mark.parametrize("module_ids,connector_ids", [(True, False), (False, True), (True, True)])
def test_frame_id_renaming_preserves_physical_numbers(module_ids, connector_ids):
    original = inputs(graph=f.state_graph(5, 7, surface=True))
    a = api().transform_inputs(original, cfg(RZ))
    b = api().transform_inputs(replace(original, graph=renamed(original.graph, module_ids, connector_ids)), cfg(RZ))
    assert a.frame == b.frame
    for x, y in zip(a.modules, b.modules):
        assert x.position == y.position and x.orientation_matrix == y.orientation_matrix
        assert x.linear_velocity == y.linear_velocity and x.angular_velocity == y.angular_velocity
    assert [(q.kind, q.values) for q in a.quantities] == [(q.kind, q.values) for q in b.quantities]


@pytest.mark.parametrize("edges", [(), ((0, 1),), ((0, 1), (1, 2), (2, 3))])
def test_frame_disconnected_zero_edge_and_transient_use_all_modules(edges):
    graph = f.state_graph(4, 1)
    graph = replace(graph, connections=tuple(c.ConnectionState(graph.connectors[a].ref,
                    graph.connectors[b].ref, "current_connection") for a, b in edges))
    result = api().transform_inputs(inputs(graph=graph), cfg())
    assert result.frame.origin == (1.5, 0., 0.)
    assert result.modules[-1].position == (1.5, 0., 0.)


def test_frame_common_translation_invariance():
    static = (c.NumericFeature("world.button.center_xyz_m", (4., 2., 3.)),)
    original = inputs(static=static)
    shifted = replace(original, graph=replace(original.graph, modules=tuple(
        replace(m, position=tuple(v+d for v,d in zip(m.position, (10.,-7.,5.)))) for m in original.graph.modules)),
        context=replace(original.context, static_geometry=(replace(static[0], values=(14.,-5.,8.)),)))
    a,b = [api().transform_inputs(x, cfg(RZ)) for x in (original, shifted)]
    assert [m.position for m in a.modules] == [m.position for m in b.modules]
    assert a.quantities == b.quantities


def test_frame_point_vector_orientation_twist_use_one_physical_rotation():
    fr = frame(RZ)  # centroid (1,0,0), axes yaw +90 degrees
    assert api().transform_point((3., 4., 5.), fr) == (4., -2., 5.)
    assert api().relative_position((3., 4., 5.), fr) == (4., -2., 5.)
    assert api().transform_vector((3., 4., 5.), fr) == (4., -3., 5.)
    assert api().transform_twist((3.,4.,5.), (6.,7.,8.), fr) == ((4.,-3.,5.), (7.,-6.,8.))
    np.testing.assert_allclose(api().transform_orientation((0.,0.,0.,1.), fr), (0.,1.,0.,-1.,0.,0.,0.,0.,1.), atol=1e-15)


def test_frame_modules_pose_and_current_twist_are_derived_with_g_scope():
    graph = f.state_graph(3, 0)
    graph = replace(graph, modules=tuple(replace(m, linear_velocity=(2.,3.,4.), angular_velocity=(5.,6.,7.)) for m in graph.modules))
    view = api().transform_inputs(inputs(graph=graph), cfg(RZ))
    assert view.modules[0].source_scope == "g_current"
    assert view.modules[0].position == (0., 1., 0.)
    assert view.modules[0].orientation_matrix == (0.,1.,0.,-1.,0.,0.,0.,0.,1.)
    assert view.modules[0].linear_velocity == (3.,-2.,4.)
    assert view.modules[0].angular_velocity == (6.,-5.,7.)


def test_frame_twist_missing_and_measured_not_centroid_subtracted():
    g = f.state_graph(2, 0)
    g = replace(g, modules=(replace(g.modules[0], linear_velocity=(9.,0.,0.)), g.modules[1]))
    view = api().transform_inputs(inputs(graph=g), cfg())
    assert view.modules[0].linear_velocity == (9.,0.,0.)
    assert view.modules[0].angular_velocity is None and view.modules[1].linear_velocity is None


@pytest.mark.parametrize("q", [(0.,0.,0.,2.), (0.,0.,0.,-2.), (0.,0.,0.,1e300), (0.,0.,0.,1e-300)])
def test_frame_quaternion_scale_and_sign_equivalent(q):
    assert api().quaternion_to_matrix(q) == I
    assert api().quaternion_to_r6(q) == (1.,0.,0.,0.,1.,0.)
    assert api().matrix_to_r6(RZ) == (0.,1.,0.,-1.,0.,0.)
    np.testing.assert_allclose(api().transform_orientation(q, frame(RZ)), (0.,1.,0.,-1.,0.,0.,0.,0.,1.), atol=1e-15)


@pytest.mark.parametrize("q", [(0.,0.,0.,0.), (float('nan'),0.,0.,1.), (0.,float('inf'),0.,1.)])
def test_frame_valid_quaternion_zero_or_nonfinite_rejected(q):
    with pytest.raises(ValueError): api().quaternion_to_matrix(q)
    graph = f.state_graph(1, 0)
    if all(math.isfinite(x) for x in q):
        graph = replace(graph, modules=(replace(graph.modules[0], orientation_xyzw=q),))
        with pytest.raises(ValueError): api().transform_inputs(inputs(graph=graph), cfg())


def test_frame_q_minus_q_nontrivial_orientation_equivalence():
    q=(.2,.3,.4,.5)
    assert api().quaternion_to_matrix(q) == api().quaternion_to_matrix(tuple(-v for v in q))
    assert api().transform_orientation(q, frame(RX)) == api().transform_orientation(tuple(-v for v in q), frame(RX))


@pytest.mark.parametrize("rotation", [(1.,)*9, (1.,0.,0.,0.,1.,0.,0.,0.,-1.), (float('nan'),)+I[1:]])
def test_frame_axes_require_finite_proper_rotation(rotation):
    with pytest.raises(ValueError): api().physical_frame(f.state_graph(2,0), cfg(rotation))


def test_frame_configured_axes_rotate_with_scene_when_explicit():
    original=inputs(static=(c.NumericFeature('world.button.center_xyz_m',(3.,2.,1.)),))
    # All source poses and geometry physically rotated +90deg about world Z.
    moved=replace(original, graph=replace(original.graph, modules=tuple(replace(m,
        position=(0.,m.position[0],0.), orientation_xyzw=(0.,0.,math.sqrt(.5),math.sqrt(.5))) for m in original.graph.modules)),
        context=replace(original.context, static_geometry=(c.NumericFeature('world.button.center_xyz_m',(-2.,3.,1.)),)))
    a,b=api().transform_inputs(original,cfg()),api().transform_inputs(moved,cfg(RZ))
    np.testing.assert_allclose(quantity(a,'world.button.center_xyz_m').values, quantity(b,'world.button.center_xyz_m').values,atol=1e-14)
    np.testing.assert_allclose(a.modules[0].orientation_matrix,b.modules[0].orientation_matrix,atol=1e-14)


def test_frame_connector_and_edge_local_semantics_preserved_once():
    graph=f.state_graph(2,1)
    ports=tuple(replace(p,features=(c.NumericFeature('position_body_m',(.1,.2,.3)),
                        c.NumericFeature('normal_body',(1.,0.,0.)),c.NumericFeature('tangent_body',(0.,1.,0.)))) for p in graph.connectors)
    edge=c.ConnectionState(ports[0].ref,ports[1].ref,'current_connection',features=(
        c.NumericFeature('relative_position_m',(.3,.4,.5)),c.NumericFeature('relative_orientation_xyzw',(0.,0.,0.,1.)),
        c.NumericFeature('clocking_quarter_turns',(1.,))))
    graph=replace(graph,connectors=ports,connections=(edge,))
    view=api().transform_inputs(inputs(graph=graph),cfg(RX))
    assert view.connectors == ports and view.connections == (edge,) and view.resources == graph.resources
    assert all(q.source_scope=='g_current' for q in view.local_geometry)
    assert {'module_body','endpoint_a_body','connection_intrinsic'} <= {q.output_frame for q in view.local_geometry}
    assert next(q for q in view.local_geometry if q.source_name=='position_body_m').values==(.1,.2,.3)


def test_frame_resource_coordinate_measurements_have_explicit_g_scope_and_local_semantics():
    graph=f.state_graph(2,0)
    graph=replace(graph,resources=(replace(graph.resources[0],coordinates=(
        c.CoordinateState('angle',position=.3,velocity=.4),
        c.CoordinateState('travel',position=None,velocity=None))),))
    view=api().transform_inputs(inputs(graph=graph),cfg(RZ))
    coords=[q for q in view.local_geometry if q.output_frame=='resource_local']
    assert len(coords)==4
    assert all(q.source_scope=='g_current' and q.entity_ref==graph.resources[0].ref for q in coords)
    assert [(q.coordinate_id,q.source_name,q.values,q.valid) for q in coords]==[
        ('angle','position',(.3,),True),('angle','velocity',(.4,),True),
        ('travel','position',None,False),('travel','velocity',None,False)]


def test_frame_nonfinite_computed_distance_rejected():
    with pytest.raises(ValueError):
        api().euclidean_distance((1.79e308,1.79e308,1.79e308))


def test_frame_static_dynamic_button_same_transform_distinct_ownership():
    original=inputs(static=(c.NumericFeature('world.button.center_xyz_m',(4.,1.,2.)),),
                    dynamic=(c.NumericFeature('world.button.current_center_xyz_m',(3.9,1.,2.)),c.NumericFeature('button.depression_m',(.1,))))
    view=api().transform_inputs(original,cfg(RZ))
    nominal=quantity(view,'world.button.center_xyz_m')
    current=quantity(view,'world.button.current_center_xyz_m','o_dynamic')
    assert nominal.values==(1.,-3.5,2.) and nominal.output_frame=='robot_centroid'
    np.testing.assert_allclose(current.values,(1.,-3.4,2.),atol=1e-15)
    assert nominal.source_scope=='c_static' and current.source_scope=='o_dynamic'
    assert quantity(view,'button.depression_m','o_dynamic').values==(.1,)
    assert original.context.static_geometry[0].values==(4.,1.,2.)


def test_frame_all_spatial_types_represented_without_guessing_z_or_full_attitude():
    static=(c.NumericFeature('world.cones.0.center_xy_m',(3.,4.)),
        c.NumericFeature('world.goal.xyyaw',(3.,4.,0.)),
        c.NumericFeature('world.environment.floor_height_m',(2.,)),
        c.NumericFeature('world.environment.platform_bounds_xy_m',(1.,3.,2.,4.)),
        c.NumericFeature('world.obstacles.0.center_xyz_m',(3.,4.,5.)),
        c.NumericFeature('world.obstacles.0.size_xyz_m',(1.,2.,3.)),
        c.NumericFeature('world.obstacles.0.yaw_deg',(90.,)),
        c.NumericFeature('world.obstacles.0.pitch_deg',(15.,)))
    view=api().transform_inputs(inputs('navigation',static),cfg(RX))
    assert len(view.quantities)==len(static)
    line=quantity(view,static[0].name)
    assert line.kind=='line' and line.values==(2.5,0.,-4.,0.,1.,0.)
    plane=quantity(view,static[2].name)
    assert plane.kind=='planes' and plane.values==(0.,1.,0.,2.)
    bounds=quantity(view,static[3].name)
    assert bounds.kind=='planes' and len(bounds.values)==16
    assert quantity(view,static[5].name).kind=='dimensions'
    assert quantity(view,static[5].name).values==(1.,2.,3.)
    # Scalar angle alone does not certify the other Euler components.
    assert quantity(view,static[6].name).kind=='angle_constraint'
    assert quantity(view,static[7].name).kind=='angle_constraint'
    assert quantity(view,static[1].name).kind=='planar_pose'


def test_frame_partial_plane_constraints_transformed_without_missing_axes_fabrication():
    static=(c.NumericFeature('world.gap.near_edge_x_m',(5.,)),
            c.NumericFeature('world.gap.crossing_direction_world_xy',(1.,0.)))
    view=api().transform_inputs(inputs('gap',static),cfg(RZ))
    assert quantity(view,static[0].name).values==(0.,-1.,0.,4.5)
    assert quantity(view,static[1].name).values==(0.,-1.,0.)


def test_frame_heights_are_planes_and_dimensions_are_intrinsic():
    static=(c.NumericFeature('world.stairs.top_heights_world_m',(.1,.2)),
            c.NumericFeature('stairs.tread_depth_m',(.3,)))
    view=api().transform_inputs(inputs('stairs',static),cfg(RX))
    assert quantity(view,static[0].name).values==(0.,1.,0.,.1,0.,1.,0.,.2)
    assert quantity(view,static[1].name).values==(.3,)
    assert quantity(view,static[1].name).output_frame=='intrinsic'


def test_frame_certified_map_and_task_local_transform_required():
    static=(c.NumericFeature('map.button.center_xyz_m',(2.,3.,4.)),)
    original=inputs(static=static)
    absent=api().transform_inputs(original,cfg())
    assert quantity(absent,static[0].name).values is None
    bridge=api().SourceFrameTransform('map',(10.,0.,0.),RZ)
    present=api().transform_inputs(original,cfg(transforms=(bridge,)))
    assert quantity(present,static[0].name).values==(6.5,2.,4.)
    local=inputs('gap',(c.NumericFeature('stage_local.gap.near_edge_x_m',(2.,)),))
    bridge=api().SourceFrameTransform('stage_local',(10.,0.,0.),RZ)
    assert quantity(api().transform_inputs(local,cfg(transforms=(bridge,))),local.context.static_geometry[0].name).values==(0.,1.,0.,2.)


@pytest.mark.parametrize('source_frame',['map','body_local'])
def test_frame_intrinsic_box_dimensions_need_no_origin_or_frame_bridge(source_frame):
    name=source_frame+'.obstacles.0.size_xyz_m'
    graph=f.state_graph(1,0)
    graph=replace(graph,modules=(replace(graph.modules[0],pose_valid=False),))
    view=api().transform_inputs(inputs(static=(c.NumericFeature(name,(1.,2.,3.)),),graph=graph),cfg())
    q=quantity(view,name)
    assert q.valid and q.values==(1.,2.,3.) and q.output_frame=='intrinsic'


@pytest.mark.parametrize('name', ['unframed.button.center_xyz_m','body_local.button.center_xyz_m'])
def test_frame_unknown_root_body_origin_or_invalid_source_is_missing(name):
    src=c.NumericFeature(name,(3.,4.,5.),valid=name.startswith('body_local'))
    q=quantity(api().transform_inputs(inputs(static=(src,)),cfg()),name)
    assert not q.valid and q.values is None


def test_frame_missing_pose_and_invalid_source_remain_explicit():
    graph=f.state_graph(2,0)
    graph=replace(graph,modules=tuple(replace(m,pose_valid=False) for m in graph.modules))
    static=(c.NumericFeature('world.button.center_xyz_m',(5.,0.,0.)),)
    result=api().transform_inputs(inputs(static=static,graph=graph),cfg())
    assert not result.frame.valid and result.frame.origin is None
    assert all(m.position is None and m.orientation_matrix is None for m in result.modules)
    assert not quantity(result,static[0].name).valid
    invalid=inputs(static=(replace(static[0],valid=False),))
    assert quantity(api().transform_inputs(invalid,cfg()),static[0].name).values is None


def test_frame_generic_euclidean_distance_not_behavior_specific():
    assert api().euclidean_distance((3.,4.,12.))==13.
    assert api().euclidean_distance((3.,4.))==5.
    assert api().euclidean_distance((1.,2.,3.),(1.,2.,3.))==0.
    assert api().euclidean_distance(None) is None
    with pytest.raises(ValueError): api().euclidean_distance((float('nan'),0.,0.))


def test_frame_view_and_source_are_immutable_without_bypassing_d2():
    original=inputs(static=(c.NumericFeature('world.button.center_xyz_m',(5.,0.,0.)),))
    before=asdict(original)
    result=api().transform_inputs(original,cfg())
    assert isinstance(result,api().FrameGeometryView) and not isinstance(result,c.PolicyInputs)
    assert asdict(original)==before
    with pytest.raises(FrozenInstanceError): result.frame.origin=(99.,0.,0.)
    with pytest.raises(FrozenInstanceError): result.modules[0].position=(99.,0.,0.)
    with pytest.raises(FrozenInstanceError): result.quantities[0].values=(99.,0.,0.)
    with pytest.raises(TypeError): api().transform_inputs(asdict(original),cfg())
    # Defeat frozen dataclass using low-level mutation: E1 must revalidate D2.
    poisoned=copy.deepcopy(original)
    object.__setattr__(poisoned.context,'static_geometry',(c.NumericFeature('world.future_goal_xyz_m',(9.,9.,9.)),))
    with pytest.raises(ValueError): api().transform_inputs(poisoned,cfg())


@pytest.mark.parametrize('poison', ['root_module_id','morphology','action_t','supervision','graph_t_plus_1','stage_name'])
def test_frame_raw_root_morphology_label_future_stage_cannot_change_outputs(poison):
    from drl.IL.tests.bc_v1.test_policy_inputs import raw_view_fixture, from_raw
    raw,metadata=raw_view_fixture()
    before=copy.deepcopy(raw)
    original=from_raw(raw,metadata)
    changed=copy.deepcopy(raw)
    changed[poison]={'forbidden':'OPPOSITE'}
    changed['graph_t'][poison]={'forbidden':'OPPOSITE'}
    a,b=[api().transform_inputs(x,cfg()) for x in (original,from_raw(changed,metadata))]
    assert a==b and raw==before


def test_frame_import_isolated_from_robot_ros_isaac_torch_and_source_readers():
    api()
    script='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','rclpy','isaacsim','omni','smores_ep'} or fullname in {'drl.IL.bc_v1.smores_adapter','drl.IL.bc_v1.task_context','drl.IL.preprocessing'}:
   raise ImportError('E1 isolation: '+fullname)
sys.meta_path.insert(0,Block())
from drl.IL.bc_v1.features import physical_frame,PhysicalFrameConfig
from drl.IL.bc_v1.bc_v1_contract import StateGraph
assert physical_frame(StateGraph(),PhysicalFrameConfig()).origin is None
'''
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('case,episode,index,task,geometry',_REAL_CASES)
def test_frame_six_targeted_real_v6_rows_include_g_c_o(real_policy_rows,case,episode,index,task,geometry):
    row,graph,observation,context,selected=real_policy_rows[case]
    view=api().transform_inputs(selected,cfg())
    assert view.frame.valid and len(view.modules)==8
    assert all(m.source_scope=='g_current' and m.position is not None and m.orientation_matrix is not None for m in view.modules)
    assert len(view.quantities)==len(context.static_geometry)+len(observation.features)
    assert view.connectors==graph.connectors and view.connections==graph.connections
    names={'assembly':'world.button.center_xyz_m','gap':'world.gap.near_edge_center_world_xyz_m',
           'stairs':'world.stairs.first_riser_center_world_xyz_m','button':'world.button.center_xyz_m',
           'navigation':'world.environment_cones.0.center_xyz_m','terminal-goal':'world.goal.center_xyz_m'}
    distances={'assembly':1.2014036501375136,'gap':.7439739228855462,'stairs':.46756993743665487,
               'button':.6859854754972594,'navigation':1.1976058860247616,'terminal-goal':1.6246730999085794}
    q=quantity(view,names[case]); assert q.valid
    assert api().euclidean_distance(q.values)==pytest.approx(distances[case],abs=1e-10)
    if case=='assembly': assert not graph.connections
    if task=='button':
        current=quantity(view,'world.button.current_center_xyz_m','o_dynamic')
        assert current.valid and math.isfinite(api().euclidean_distance(current.values))
    assert selected.graph==graph and selected.context==context and selected.observation==observation
