"""Native adapter regressions with CPU Warp tensors and real USD matrices.

No simulation scene or physical stepping is needed for these boundary tests.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from drl.MARL.isaaclab_articulation import NativeArticulationAccess, NativeTransformSource


ROOT = "/World/env_0/module_0"


class TensorView:
    """Tensor API boundary double; array conversion/indexing uses real Warp."""

    def __init__(self, root=ROOT, count=1):
        self.count = count
        self.prim_paths = [root]
        self.shared_metatype = SimpleNamespace(dof_names=("left", "right", "pan"), link_names=("body",))
        self.arrays = {}
        self.writes = []
        self.link_reads = 0
        for field in ('dof_stiffnesses','dof_dampings','dof_position_targets',
                      'dof_velocity_targets','dof_actuation_forces'):
            self.put(field,[[0.,0.,0.]])

    def put(self, name, values):
        self.arrays[name] = wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device="cpu")

    def __getattr__(self, name):
        if name.startswith("get_"):
            return lambda: self.arrays[name[4:]]
        if name.startswith("set_"):
            def store(values, indices):
                np.testing.assert_array_equal(indices.numpy(), [0])
                assert indices.dtype == wp.uint32
                self.writes.append(name[4:])
                self.put(name[4:], values.numpy())
            return store
        raise AttributeError(name)

    def get_link_transforms(self):
        self.link_reads += 1
        return self.arrays["link_transforms"]


def make_access(view):
    simulation = SimpleNamespace(create_articulation_view=lambda path: view,
        update_articulations_kinematic=lambda: view.writes.append('kinematic_update'))
    return NativeArticulationAccess(simulation, ROOT)


def test_articulation_access_requires_exact_root_and_one_match():
    view = TensorView()
    access = make_access(view)
    assert access.view is view
    assert access.dof_paths == [[ROOT + "/joints/" + name for name in ("left", "right", "pan")]]
    for count in (0, 2):
        with pytest.raises(RuntimeError, match="exactly one"):
            make_access(TensorView(count=count))
    with pytest.raises(RuntimeError, match="root"):
        make_access(TensorView(root="/World/env_1/module_0"))


@pytest.mark.parametrize("setter,field", [
    ("set_dof_max_efforts", "dof_max_forces"),
    ("set_dof_max_velocities", "dof_max_velocities"),
    ("set_dof_position_targets", "dof_position_targets"),
    ("set_dof_velocity_targets", "dof_velocity_targets"),
])
def test_indexed_setters_preserve_unselected_dofs(setter, field):
    view = TensorView()
    view.put(field, [[1., 2., 3.]])
    access = make_access(view)
    getattr(access, setter)([8., 9.], dof_indices=[2, 0])
    np.testing.assert_array_equal(view.arrays[field].numpy(), [[9., 2., 8.]])
    assert view.writes == [field]


def test_indexed_gain_setter_preserves_other_dofs():
    view = TensorView()
    view.put("dof_stiffnesses", [[1., 2., 3.]])
    view.put("dof_dampings", [[4., 5., 6.]])
    make_access(view).set_dof_gains(stiffnesses=[7.], dampings=[8.], dof_indices=[1])
    np.testing.assert_array_equal(view.arrays["dof_stiffnesses"].numpy(), [[1., 7., 3.]])
    np.testing.assert_array_equal(view.arrays["dof_dampings"].numpy(), [[4., 8., 6.]])


def test_reset_pose_writes_xyzw_yaw_and_zeros_root_and_joint_motion():
    view = TensorView()
    view.put("root_transforms", [[0., 0., 0., 0., 0., 0., 1.]])
    view.put("root_velocities", [[1.] * 6])
    view.put("dof_positions", [[1., 2., 3.]])
    view.put("dof_velocities", [[4., 5., 6.]])
    access = make_access(view)
    access.reset_pose((2., 3., 4.), 90.)
    np.testing.assert_allclose(view.arrays["root_transforms"].numpy(),
        [[2., 3., 4., 0., 0., np.sqrt(.5), np.sqrt(.5)]], atol=1e-7)
    for field in ("root_velocities", "dof_positions", "dof_velocities"):
        assert np.count_nonzero(view.arrays[field].numpy()) == 0
    assert view.writes == ["dof_stiffnesses", "dof_dampings", "dof_position_targets",
        "dof_velocity_targets", "dof_actuation_forces", "root_transforms", "root_velocities",
        "dof_positions", "dof_velocities", "kinematic_update"]
    assert access.reset_pose.__qualname__ == "NativeArticulationAccess.reset_pose"


def test_transform_source_uses_live_pose_rotates_connector_and_caches_per_tick():
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, ROOT + "/body")
    body.AddTranslateOp().Set(Gf.Vec3d(100., 200., 300.))
    connector = UsdGeom.Xform.Define(stage, ROOT + "/body/connector")
    connector.AddTranslateOp().Set(Gf.Vec3d(1., 0., 0.))
    static = UsdGeom.Xform.Define(stage, "/World/static")
    static.AddTranslateOp().Set(Gf.Vec3d(5., 6., 7.))
    view = TensorView()
    view.put("link_transforms", [[[10., 20., 30., 0., 0., np.sqrt(.5), np.sqrt(.5)]]])
    tick = [0]
    source = NativeTransformSource(stage, {ROOT: make_access(view)}, lambda: tick[0])
    live = source(str(connector.GetPath()))
    np.testing.assert_allclose(live.ExtractTranslation(), [10., 21., 30.], atol=1e-6)
    np.testing.assert_allclose(live.TransformDir(Gf.Vec3d(1., 0., 0.)), [0., 1., 0.], atol=1e-6)
    np.testing.assert_allclose(source(str(body.GetPath())).ExtractTranslation(), [10., 20., 30.])
    assert view.link_reads == 1
    view.put("link_transforms", [[[40., 50., 60., 0., 0., 0., 1.]]])
    np.testing.assert_allclose(source(str(connector.GetPath())).ExtractTranslation(), [10., 21., 30.], atol=1e-6)
    tick[0] += 1
    np.testing.assert_allclose(source(str(connector.GetPath())).ExtractTranslation(), [41., 50., 60.])
    assert view.link_reads == 2
    view.put("link_transforms", [[[70., 80., 90., 0., 0., 0., 1.]]])
    source.invalidate()
    np.testing.assert_allclose(source(str(connector.GetPath())).ExtractTranslation(), [71., 80., 90.])
    assert view.link_reads == 3
    np.testing.assert_allclose(source(str(static.GetPath())).ExtractTranslation(), [5., 6., 7.])


def test_reset_rigid_body_restores_xyzw_pose_and_clears_motion():
    from drl.MARL.isaaclab_articulation import reset_rigid_body

    view = TensorView()
    view.put("transforms", [[9., 8., 7., 0., 0., 0., 1.]])
    view.put("velocities", [[1., 2., 3., 4., 5., 6.]])
    reset_rigid_body(view, (1., 2., 3.), -90.)
    np.testing.assert_allclose(view.arrays["transforms"].numpy(),
        [[1., 2., 3., 0., 0., -np.sqrt(.5), np.sqrt(.5)]], atol=1e-7)
    np.testing.assert_array_equal(view.arrays["velocities"].numpy(), np.zeros((1, 6)))
    assert view.writes == ["transforms", "velocities"]


def test_transform_source_reads_live_rigid_view_and_child_offset():
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    button = UsdGeom.Xform.Define(stage, "/World/button")
    button.AddTranslateOp().Set(Gf.Vec3d(100., 0., 0.))
    tip = UsdGeom.Xform.Define(stage, "/World/button/tip")
    tip.AddTranslateOp().Set(Gf.Vec3d(0., 0., .25))
    view = TensorView(root="/World/button")
    view.put("transforms", [[2., 3., 4., 0., 0., 0., 1.]])
    UsdGeom.Xform.Define(stage, ROOT + "/body")
    articulation = TensorView()
    articulation.put("link_transforms", [[[8., 9., 10., 0., 0., 0., 1.]]])
    source = NativeTransformSource(stage, {ROOT: make_access(articulation)}, lambda: 0)
    source.rigid_views = {"/World/button": view}
    np.testing.assert_allclose(source(str(tip.GetPath())).ExtractTranslation(), [2., 3., 4.25])
    np.testing.assert_allclose(source(ROOT + "/body").ExtractTranslation(), [8., 9., 10.])
    view.put("transforms", [[5., 6., 7., 0., 0., 0., 1.]])
    source.invalidate()
    np.testing.assert_allclose(source(str(tip.GetPath())).ExtractTranslation(), [5., 6., 7.25])


def test_reset_pose_refreshes_stale_child_links_before_return_without_physics_step():
    """GPU tensor setters can leave link transforms at the previous assembly."""
    view = TensorView()
    view.put('root_transforms', [[10., 20., 30., 0., 0., 0., 1.]])
    view.put('root_velocities', [[1.] * 6])
    view.put('dof_positions', [[1., 2., 3.]])
    view.put('dof_velocities', [[4., 5., 6.]])
    view.put('link_transforms', [[[10.1, 20., 30., 0., 0., 0., 1.]]])
    old_child = view.get_link_transforms().numpy().copy()
    class StagedSimulation:
        updates = 0
        def create_articulation_view(self, path): return view
        def update_articulations_kinematic(self):
            # The synchronization must observe the completed reset, not an
            # intermediate root teleport with old folded joint coordinates.
            assert view.writes[-4:] == ['root_transforms', 'root_velocities', 'dof_positions', 'dof_velocities']
            np.testing.assert_array_equal(view.arrays['dof_positions'].numpy(), [[0., 0., 0.]])
            np.testing.assert_array_equal(view.get_link_transforms().numpy(), old_child)
            root = view.get_root_transforms().numpy().copy()
            root[0, 0] += .1
            view.put('link_transforms', [root])
            self.updates += 1
    simulation = StagedSimulation()
    access = NativeArticulationAccess(simulation, ROOT)
    access.reset_pose((2., 3., 4.), 0.)
    np.testing.assert_allclose(view.get_link_transforms().numpy()[0, 0, :3], [2.1, 3., 4.], atol=1e-6)
    assert simulation.updates == 1
    for field in ('root_velocities', 'dof_positions', 'dof_velocities'):
        assert np.count_nonzero(view.arrays[field].numpy()) == 0


def test_reset_pose_requires_the_native_kinematic_update_contract():
    view = TensorView()
    view.put('root_transforms', [[0., 0., 0., 0., 0., 0., 1.]])
    view.put('dof_positions', [[1., 2., 3.]])
    access = NativeArticulationAccess(SimpleNamespace(create_articulation_view=lambda path: view), ROOT)
    with pytest.raises(AttributeError, match='update_articulations_kinematic'):
        access.reset_pose((2., 3., 4.), 0.)


def test_reset_pose_clears_drive_and_effort_targets_before_teleport():
    view=TensorView()
    for key,values in dict(root_transforms=[[0.,0.,0.,0.,0.,0.,1.]],root_velocities=[[8.]*6],
        dof_positions=[[1.,2.,3.]],dof_velocities=[[4.,5.,6.]],
        dof_position_targets=[[.2,.3,.4]],dof_velocity_targets=[[5.,6.,7.]],
        dof_actuation_forces=[[.5,.6,.7]],dof_stiffnesses=[[10.,20.,30.]],
        dof_dampings=[[1.,2.,3.]]).items():view.put(key,values)
    make_access(view).reset_pose((1.,2.,.032),0.)
    for key in ('dof_position_targets','dof_velocity_targets','dof_actuation_forces','dof_stiffnesses'):
        assert np.count_nonzero(view.arrays[key].numpy())==0,key
    np.testing.assert_allclose(view.arrays['dof_dampings'].numpy(),[[.02]*3])
    assert view.writes.index('dof_actuation_forces')<view.writes.index('root_transforms')


@pytest.mark.skipif(not wp.is_cuda_available(),reason='CUDA needed to distinguish property and state devices')
def test_reset_controls_follow_each_property_native_device():
    class DeviceView(TensorView):
        def __getattr__(self,name):
            operation=super().__getattr__(name)
            if name in ('set_dof_stiffnesses','set_dof_dampings'):
                def checked(values,indices):
                    assert values.device.is_cpu and indices.device.is_cpu
                    return operation(values,indices)
                return checked
            return operation
    view=DeviceView()
    for key in ('dof_stiffnesses','dof_dampings','dof_position_targets','dof_velocity_targets','dof_actuation_forces'):
        view.put(key,[[1.,2.,3.]])
    view.put('root_transforms',[[0.,0.,0.,0.,0.,0.,1.]])
    view.arrays['dof_positions']=wp.zeros((1,3),dtype=wp.float32,device='cuda:0')
    view.put('dof_velocities',[[1.,2.,3.]])
    make_access(view).reset_pose((0.,0.,.032),0.)
