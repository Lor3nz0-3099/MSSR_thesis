"""Real USD stage reset geometry tests; no simulation/application launch."""
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from drl.MARL.c1_runner import make_course
from drl.MARL.isaaclab_course_reset import update_native_course

ROOT = '/World/envs/env_0'
COURSE = ROOT + '/C1Course'


def stage_fixture():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, ROOT)
    UsdShade.Material.Define(stage, '/World/materials/dynamic_ground')
    sibling = UsdGeom.Cube.Define(stage, '/World/envs/env_1/C1Course/Sentinel')
    sibling.AddTranslateOp().Set(Gf.Vec3d(11, 12, 13))
    UsdPhysics.CollisionAPI.Apply(sibling.GetPrim())
    return stage


@pytest.mark.parametrize('domain,seeds', [
    ('navigation', (7000, 7450, 7970)), ('stairs', (12000, 12450, 12970)),
    ('gap', (16000, 16450, 16970)), ('manipulation', (20000, 20450, 20970))])
def test_three_geometry_generations_preserve_sibling_and_root(domain, seeds):
    stage = stage_fixture()
    sibling = stage.GetPrimAtPath('/World/envs/env_1/C1Course/Sentinel')
    sibling_state = str(sibling.GetAttribute('xformOp:translate').Get())
    root = None
    previous_geometry = None
    for seed in seeds:
        course = make_course(domain, seed, 'train')
        path = update_native_course(stage, ROOT, domain, course, (30., 40., 0.))
        current_root = stage.GetPrimAtPath(COURSE)
        if root is not None:
            assert current_root == root
        root = current_root
        if seed == seeds[0]:
            root.CreateAttribute('test:retained', Sdf.ValueTypeNames.Int).Set(23)
        assert root.GetAttribute('test:retained').Get() == 23
        assert sibling.IsValid()
        assert str(sibling.GetAttribute('xformOp:translate').Get()) == sibling_state
        boxes = {box.name: box for box in course.support_boxes}
        children = {p.GetName() for p in root.GetChildren()}
        assert children == set(boxes) | ({'Cones'} if domain == 'navigation' else set()) | (
            {'ButtonPrismaticJoint'} if domain == 'manipulation' else set())
        geometry = []
        for name, box in boxes.items():
            prim = stage.GetPrimAtPath(COURSE + '/' + name)
            assert tuple(prim.GetAttribute('xformOp:translate').Get()) == pytest.approx(box.center_xyz_m)
            assert tuple(prim.GetAttribute('xformOp:scale').Get()) == pytest.approx(box.size_xyz_m)
            order = prim.GetAttribute('xformOpOrder').Get()
            assert len(order) == len(set(order))
            assert prim.HasAPI(UsdPhysics.CollisionAPI) == box.collidable
            geometry.append((name, box.center_xyz_m, box.size_xyz_m))
        if previous_geometry is not None:
            assert geometry != previous_geometry
        previous_geometry = geometry
        if domain == 'navigation':
            assert len(stage.GetPrimAtPath(COURSE + '/Cones').GetChildren()) == len(course.cones)
        if domain == 'manipulation':
            assert path == COURSE + '/' + course.button_plunger.name
        else:
            assert path is None
        # Explicit obsolete static geometry must disappear on next update.
        stale = UsdGeom.Cube.Define(stage, COURSE + '/ObsoleteCourseBox')
        UsdPhysics.CollisionAPI.Apply(stale.GetPrim())
        stale.GetPrim().CreateAttribute('mssr:obstacleSemantic', Sdf.ValueTypeNames.String).Set('road')


def test_manipulation_keeps_dynamic_body_and_joint_and_updates_world_anchor():
    stage = stage_fixture()
    first = make_course('manipulation', 20000, 'train')
    path = update_native_course(stage, ROOT, 'manipulation', first, (10., 20., 3.))
    body = stage.GetPrimAtPath(path)
    joint_prim = stage.GetPrimAtPath(COURSE + '/ButtonPrismaticJoint')
    body.CreateAttribute('test:retained', Sdf.ValueTypeNames.Int).Set(41)
    joint_prim.CreateAttribute('test:retained', Sdf.ValueTypeNames.Int).Set(42)
    dimensions = tuple(body.GetAttribute('xformOp:scale').Get())
    initial_anchor = tuple(UsdPhysics.PrismaticJoint(joint_prim).GetLocalPos0Attr().Get())
    for seed in (20450, 20970):
        course = make_course('manipulation', seed, 'train')
        assert update_native_course(stage, ROOT, 'manipulation', course, (10., 20., 3.)) == path
        assert body.IsValid() and joint_prim.IsValid()
        assert body.GetAttribute('test:retained').Get() == 41
        assert joint_prim.GetAttribute('test:retained').Get() == 42
        assert body == stage.GetPrimAtPath(path)
        assert joint_prim == stage.GetPrimAtPath(COURSE + '/ButtonPrismaticJoint')
        assert tuple(body.GetAttribute('xformOp:scale').Get()) == dimensions
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert bodies == [body]
        joint = UsdPhysics.PrismaticJoint(joint_prim)
        anchor = tuple(joint.GetLocalPos0Attr().Get())
        assert anchor == pytest.approx(tuple(a + b for a, b in zip(course.button_center_xyz_m, (10., 20., 3.))))
        assert anchor != initial_anchor
        assert list(joint.GetBody1Rel().GetTargets()) == [body.GetPath()]


def test_global_ground_and_foreign_dynamic_body_are_protected():
    stage = stage_fixture()
    ground = UsdGeom.Cube.Define(stage, '/World/Ground').GetPrim()
    with pytest.raises(ValueError, match='Ground'):
        update_native_course(stage, ROOT, 'stairs', make_course('stairs', 12000, 'train'), (0, 0, 0))
    assert ground.IsValid()
    stage.RemovePrim('/World/Ground')
    update_native_course(stage, ROOT, 'stairs', make_course('stairs', 12000, 'train'), (0, 0, 0))
    body = UsdGeom.Cube.Define(stage, COURSE + '/ForeignDynamic').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    with pytest.raises(ValueError, match='dynamic'):
        update_native_course(stage, ROOT, 'stairs', make_course('stairs', 12450, 'train'), (0, 0, 0))
    assert body.IsValid()


def test_environment_transform_is_retained_without_double_offset():
    stage = stage_fixture()
    origin = (30., 40., 2.)
    env = UsdGeom.Xform(stage.GetPrimAtPath(ROOT))
    env.AddTranslateOp().Set(Gf.Vec3d(*origin))
    for seed in (16000, 16450, 16970):
        course = make_course('gap', seed, 'train')
        update_native_course(stage, ROOT, 'gap', course, origin)
        cache = UsdGeom.XformCache()
        for box in course.support_boxes:
            prim = stage.GetPrimAtPath(COURSE + '/' + box.name)
            center = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            assert tuple(center) == pytest.approx(tuple(a + b for a, b in zip(box.center_xyz_m, origin)))


def test_cannot_change_fixed_domain_slot():
    stage = stage_fixture()
    update_native_course(stage, ROOT, 'gap', make_course('gap', 16000, 'train'), (0, 0, 0))
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='fixed'):
        update_native_course(stage, ROOT, 'stairs', make_course('stairs', 12000, 'train'), (0, 0, 0))
    assert stage.GetRootLayer().ExportToString() == before
