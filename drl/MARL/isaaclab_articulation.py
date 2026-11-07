"""Small robot-drive adapter over Isaac Lab's existing PhysX tensor view.

No SimulationManager compatibility monkeypatch, stepping or env vectorizer.
The legacy SMORES reader/controller keeps its tested sign/gain conventions.
"""
import numpy as np


class _ReadonlyArray:
    """Owned immutable host snapshot, with device metadata for existing readers."""
    def __init__(self, native):
        self._native = native
        self._values = native.numpy().copy()
        self._values.flags.writeable = False

    def numpy(self):
        return self._values

    def __getattr__(self, name):
        return getattr(self._native, name)


class ReadonlyNativeView:
    """Viewer-only actual-state readback cache; commanded targets stay live.

    The clock must identify *completed* physical steps plus reset generation.
    State setters invalidate every field, including derived link kinematics.
    """
    FIELDS = frozenset(('dof_positions', 'dof_velocities', 'root_transforms',
                        'root_velocities', 'link_transforms', 'link_velocities'))

    def __init__(self, view, clock):
        self._view, self._clock = view, clock
        self._stamp = object()
        self._snapshots = {}

    def invalidate(self):
        self._snapshots.clear()
        self._stamp = object()

    def __getattr__(self, name):
        native = getattr(self._view, name)
        if name.startswith('get_') and name[4:] in self.FIELDS:
            def read():
                stamp = self._clock()
                if stamp != self._stamp:
                    self._snapshots.clear()
                    self._stamp = stamp
                if name not in self._snapshots:
                    self._snapshots[name] = _ReadonlyArray(native())
                return self._snapshots[name]
            return read
        if name.startswith('set_') and name[4:] in self.FIELDS:
            def write(*args, **kwargs):
                self.invalidate()
                try:
                    return native(*args, **kwargs)
                finally:
                    self.invalidate()
            return write
        return native


class NativeArticulationAccess:
    def __init__(self, simulation_view, module_root, *, state_clock=None):
        self.simulation_view = simulation_view
        self.view = simulation_view.create_articulation_view(module_root)
        if self.view.count != 1:
            raise RuntimeError(f'Expected exactly one native articulation: {module_root}')
        if tuple(self.view.prim_paths) != (module_root,):
            raise RuntimeError(
                f'Native articulation root mismatch: expected {module_root}, '
                f'got {self.view.prim_paths}'
            )
        self.dof_paths = [[f'{module_root}/joints/{name}' for name in self.view.shared_metatype.dof_names]]
        if state_clock is not None:
            self.view = ReadonlyNativeView(self.view, state_clock)

    def _get(self, name):
        # Existing SMORES code expects a .numpy() accessor, already provided
        # by Warp arrays. Data comes from the current PhysX scene, not USD.
        return getattr(self.view, 'get_'+name)()

    def get_dof_positions(self): return self._get('dof_positions')
    def get_dof_velocities(self): return self._get('dof_velocities')
    def get_dof_position_targets(self): return self._get('dof_position_targets')
    def get_dof_velocity_targets(self): return self._get('dof_velocity_targets')

    def _set(self, name, values, dof_indices):
        import warp as wp
        old = self._get(name)
        data = old.numpy().copy()
        data[0, dof_indices] = np.asarray(values, dtype=np.float32)
        tensor = wp.array(data, dtype=wp.float32, device=old.device)
        indices = wp.array([0], dtype=wp.uint32, device=old.device)
        getattr(self.view, 'set_'+name)(tensor, indices)

    def set_dof_gains(self, *, stiffnesses, dampings, dof_indices):
        self._set('dof_stiffnesses', stiffnesses, dof_indices)
        self._set('dof_dampings', dampings, dof_indices)

    def set_dof_max_efforts(self, values, *, dof_indices): self._set('dof_max_forces', values, dof_indices)
    def set_dof_max_velocities(self, values, *, dof_indices): self._set('dof_max_velocities', values, dof_indices)
    def set_dof_position_targets(self, values, *, dof_indices): self._set('dof_position_targets', values, dof_indices)
    def set_dof_velocity_targets(self, values, *, dof_indices): self._set('dof_velocity_targets', values, dof_indices)

    def capture_state(self):
        """Copy all generalized state and controller provenance from native tensors."""
        from drl.MARL.c1_expert_snapshots import CONTROL_FIELDS
        pose = self._get('root_transforms').numpy()[0].tolist()
        state = dict(root_path=str(self.view.prim_paths[0]), position_xyz_m=pose[:3],
            quaternion_xyzw=pose[3:], root_velocity=self._get('root_velocities').numpy()[0].tolist(),
            dof_names=list(self.view.shared_metatype.dof_names),
            q=self._get('dof_positions').numpy()[0].tolist(),
            qd=self._get('dof_velocities').numpy()[0].tolist(),
            control={field:self._get(field).numpy()[0].tolist() for field in CONTROL_FIELDS})
        self.validate_state(state)
        return state

    def validate_state(self, state):
        """Preflight state before any native write, including exact DOF ordering."""
        from drl.MARL.c1_expert_snapshots import CONTROL_FIELDS, _vector
        if state.get('root_path') != str(self.view.prim_paths[0]):
            raise ValueError('native articulation root mismatch')
        names = list(self.view.shared_metatype.dof_names)
        if state.get('dof_names') != names:
            raise ValueError('native articulation DOF names/order mismatch')
        _vector(state.get('position_xyz_m'),3,'position')
        _vector(state.get('quaternion_xyzw'),4,'rotation',quaternion=True)
        _vector(state.get('root_velocity'),6,'velocity')
        for key in ('q','qd'):
            _vector(state.get(key),len(names),key)
        control = state.get('control')
        if not isinstance(control,dict) or set(control) != set(CONTROL_FIELDS):
            raise ValueError('native articulation control fields mismatch')
        for field in CONTROL_FIELDS:
            _vector(control[field],len(names),field)

    def restore_state(self, state, *, restore_control=True):
        """Restore only this articulation; update kinematics without physical stepping.

        ``restore_control=False`` is the policy handover path: keep measured
        generalized state, retire Expert forces and targets, and use passive
        gains. Captured controller state remains available as audit provenance.
        """
        import warp as wp
        from drl.MARL.c1_expert_snapshots import CONTROL_FIELDS
        self.validate_state(state)
        names = state['dof_names']
        control = dict(state['control']) if restore_control else {
            'dof_stiffnesses':[0.]*len(names), 'dof_dampings':[.02]*len(names),
            'dof_position_targets':state['q'], 'dof_velocity_targets':[0.]*len(names),
            'dof_actuation_forces':[0.]*len(names)}
        values = [(field,control[field]) for field in CONTROL_FIELDS]
        values.extend((('root_transforms',state['position_xyz_m']+state['quaternion_xyzw']),
                       ('root_velocities',state['root_velocity']),
                       ('dof_positions',state['q']),('dof_velocities',state['qd'])))
        # Prepare every tensor before the first mutation.
        writes = []
        for field,values in values:
            sample = self._get(field)
            data = wp.array([values],dtype=wp.float32,device=sample.device)
            indices = wp.array([0],dtype=wp.uint32,device=sample.device)
            writes.append((field,data,indices))
        for field,data,indices in writes:
            getattr(self.view,'set_'+field)(data,indices)
        self.simulation_view.update_articulations_kinematic()
        if isinstance(self.view, ReadonlyNativeView):
            self.view.invalidate()

    def reset_pose(self, position, yaw_deg):
        """Teleport one module and clear root and joint motion (PhysX xyzw)."""
        import math
        import warp as wp

        sample = self.view.get_root_transforms()
        half = math.radians(yaw_deg) / 2
        pose = wp.array(
            [[*position, 0., 0., math.sin(half), math.cos(half)]],
            dtype=wp.float32,
            device=sample.device,
        )
        indices = wp.array([0], dtype=wp.uint32, device=sample.device)
        positions = self.view.get_dof_positions()
        zeros = wp.zeros(positions.shape, dtype=wp.float32, device=positions.device)
        # Retire the previous controller before changing its generalized state.
        # These writes address this one articulation, never a global force reset.
        for field,value in (('dof_stiffnesses',0.),('dof_dampings',.02),
                            ('dof_position_targets',0.),('dof_velocity_targets',0.),
                            ('dof_actuation_forces',0.)):
            native=self._get(field)
            data=wp.full(native.shape,value,dtype=wp.float32,device=native.device)
            native_indices=wp.array([0],dtype=wp.uint32,device=native.device)
            getattr(self.view,'set_'+field)(data,native_indices)
        self.view.set_root_transforms(pose, indices)
        self.view.set_root_velocities(
            wp.zeros((1, 6), dtype=wp.float32, device=sample.device), indices
        )
        self.view.set_dof_positions(zeros, indices)
        self.view.set_dof_velocities(zeros, indices)
        # GPU PhysX does not refresh child-link transforms when generalized
        # coordinates are written. Expert reads those links immediately after
        # reset; synchronize kinematics without stepping the physical scene.
        self.simulation_view.update_articulations_kinematic()
        if isinstance(self.view, ReadonlyNativeView):
            self.view.invalidate()


class NativeTransformSource:
    """Live PhysX link transforms plus immutable connector-to-link geometry."""
    def __init__(self, stage, accesses, clock):
        self.stage, self.accesses, self.clock = stage, accesses, clock
        self.stamp = None
        self.transforms = {}
        self.rigid_views = {}

    def invalidate(self):
        self.stamp = None

    def __call__(self, path):
        from pxr import Gf, UsdGeom
        stamp = self.clock()
        if stamp != self.stamp:
            transforms = {}
            for root, access in self.accesses.items():
                poses = access.view.get_link_transforms().numpy()[0]
                for name, pose in zip(access.view.shared_metatype.link_names, poses, strict=True):
                    matrix = Gf.Matrix4d(1.)
                    matrix.SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float,pose[3:6]))))
                    matrix.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])))
                    transforms[root+'/'+name] = matrix
            for rigid_path, view in self.rigid_views.items():
                pose = view.get_transforms().numpy()[0]
                matrix = Gf.Matrix4d(1.)
                matrix.SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float,pose[3:6]))))
                matrix.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])))
                transforms[rigid_path] = matrix
            self.transforms, self.stamp = transforms, stamp
        parent = path
        while parent:
            if parent in self.transforms:
                relative = UsdGeom.XformCache().ComputeRelativeTransform(
                    self.stage.GetPrimAtPath(path), self.stage.GetPrimAtPath(parent))[0]
                return relative * self.transforms[parent]
            parent = parent.rsplit('/',1)[0]
        return UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))


def reset_rigid_body(view, position, yaw_deg):
    """Restore one native rigid body's world pose and clear its velocities."""
    import math
    import warp as wp

    sample = view.get_transforms()
    half = math.radians(yaw_deg) / 2
    pose = wp.array(
        [[*position, 0., 0., math.sin(half), math.cos(half)]],
        dtype=wp.float32,
        device=sample.device,
    )
    indices = wp.array([0], dtype=wp.uint32, device=sample.device)
    view.set_transforms(pose, indices)
    view.set_velocities(
        wp.zeros((1, 6), dtype=wp.float32, device=sample.device), indices
    )
