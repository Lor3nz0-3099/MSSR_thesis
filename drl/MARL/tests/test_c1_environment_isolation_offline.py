"""Real USD collision authoring for overlapping rows; never load/step PhysX."""
import ast
from pathlib import Path
from types import SimpleNamespace

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[3]
SDK = Path('/home/lorenzo/IsaacLab/source/isaaclab/isaaclab')


def sdk_function(path, function_name, class_name=None, **namespace):
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == function_name)
    function.decorator_list = []
    namespace.update(Usd=Usd, UsdGeom=UsdGeom, Sdf=Sdf)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function_name]


def overlapping_scene():
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    roots = tuple(f'/World/envs/env_{i}' for i in range(8))
    for i, root in enumerate(roots):
        xf = UsdGeom.Xform.Define(stage, root)
        xf.AddTranslateOp().Set(Gf.Vec3d((i % 4)*1.5, (i // 4)*1.5, 0))
        for relative, scale in (('C1Course/Support', (3., 3., .05)),
                                ('C1Course/ButtonPlunger', (.1, .1, .1)),
                                ('SMORES_0/body/Collider', (.05, .05, .05))):
            cube = UsdGeom.Cube.Define(stage, root+'/'+relative)
            cube.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cloner = sdk_function(SDK/'cloner/cloner_utils.py', 'filter_collisions')
    scene = SimpleNamespace(stage=stage, _global_prim_paths=[], env_prim_paths=list(roots),
                            physics_scene_path='/physicsScene')
    filter_scene = sdk_function(SDK/'scene/interactive_scene.py', 'filter_collisions',
        'InteractiveScene', cloner=SimpleNamespace(filter_collisions=cloner))
    filter_scene(scene, global_prim_paths=[])
    return stage, roots


def test_eight_overlapping_courses_have_separate_collision_groups_and_own_contacts():
    stage, roots = overlapping_scene()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    a = bbox.ComputeWorldBound(stage.GetPrimAtPath(roots[0]+'/C1Course/Support')).ComputeAlignedRange()
    b = bbox.ComputeWorldBound(stage.GetPrimAtPath(roots[1]+'/C1Course/Support')).ComputeAlignedRange()
    assert all(max(a.GetMin()[k], b.GetMin()[k]) < min(a.GetMax()[k], b.GetMax()[k]) for k in range(3))
    assert stage.GetPrimAtPath('/physicsScene').GetAttribute('physxScene:invertCollisionGroupFilter').Get()
    groups = [UsdPhysics.CollisionGroup.Get(stage, f'/World/collisions/group{i}') for i in range(8)]
    for i, group in enumerate(groups):
        query = group.GetCollidersCollectionAPI().ComputeMembershipQuery()
        assert group.GetFilteredGroupsRel().GetTargets() == [group.GetPath()]
        for j, root in enumerate(roots):
            for path in ('C1Course/Support', 'C1Course/ButtonPlunger', 'SMORES_0/body/Collider'):
                assert query.IsPathIncluded(Sdf.Path(root+'/'+path)) == (i == j)
                assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath(root+'/'+path)).GetCollisionEnabledAttr().Get()
    assert not stage.GetPrimAtPath('/World/collisions/global_group')


def test_selected_course_regeneration_filters_all_foreign_shapes_in_eight_rows():
    from drl.MARL.native_episode_reset import isolate_course_colliders
    stage, roots = overlapping_scene()
    for owner in roots:
        isolate_course_colliders(stage, owner, roots)
    sibling_before = stage.GetPrimAtPath(roots[1]+'/C1Course/Support').GetPrimStack()[0].GetAsText()
    stage.RemovePrim(roots[0]+'/C1Course/Support')
    new = UsdGeom.Cube.Define(stage, roots[0]+'/C1Course/RegeneratedSupport').GetPrim()
    UsdPhysics.CollisionAPI.Apply(new)
    counts = isolate_course_colliders(stage, roots[0], roots)
    assert counts == dict(colliders=2, foreign_colliders=21)
    targets = set(UsdPhysics.FilteredPairsAPI(new).GetFilteredPairsRel().GetTargets())
    for root in roots[1:]:
        for relative in ('C1Course/Support', 'C1Course/ButtonPlunger', 'SMORES_0/body/Collider'):
            assert Sdf.Path(root+'/'+relative) in targets
    assert not any(path.HasPrefix(Sdf.Path(roots[0])) for path in targets)
    assert stage.GetPrimAtPath(roots[1]+'/C1Course/Support').GetPrimStack()[0].GetAsText() == sibling_before
    assert UsdPhysics.CollisionAPI(new).GetCollisionEnabledAttr().Get()


def test_native_authoring_and_resets_use_the_audited_isolation_paths():
    # Protect the real integration order; do not infer safety from spacing alone.
    source = (ROOT/'drl/MARL/isaaclab_c1_tasks.py').read_text()
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == 'filter_collisions'
               and any(k.arg == 'global_prim_paths' and isinstance(k.value, ast.List) and not k.value.elts
                       for k in n.keywords) for n in calls)
    reset = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_reset_idx')
    named = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr: n.lineno
             for n in ast.walk(reset) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert named['_resample_course'] < named['isolate_course_colliders'] < named['flush_pending_physics']
    env = ast.parse((ROOT/'drl/MARL/isaaclab_c1_env.py').read_text())
    spacing = next(n for n in env.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'C1_ENV_SPACING_M' for t in n.targets))
    assert ast.literal_eval(spacing.value) == 1.5
