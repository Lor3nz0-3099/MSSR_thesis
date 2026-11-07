"""Versioned C1 native snapshots and CPU-only bank validation.

No simulator imports, stepping, Expert execution or synthetic bank generation.
Capture records observed native state; the caller supplies its measured physical
certificate. Hashes detect corruption; they do not authenticate provenance.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'mssr.c1_expert_snapshot.v1'
BANK_SCHEMA = 'mssr.c1_snapshot_bank.v1'
PHYSICS_VERSION = 'c1-native-snapshot-physics-v1'
STAGE_TARGET_COUNTS = {'S0': 0, 'S1': 2, 'S2': 4, 'S3': 6}
CONTROL_FIELDS = ('dof_stiffnesses', 'dof_dampings', 'dof_position_targets',
                  'dof_velocity_targets', 'dof_actuation_forces')
PORTS = frozenset(('LEFT', 'RIGHT', 'TOP', 'BOTTOM'))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def content_hash(payload):
    value = deepcopy(payload)
    value.pop('content_sha256', None)
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _relative(value):
    if (not isinstance(value, str) or not value or value.startswith('/') or
        '\\' in value or any(part in ('', '.', '..') for part in value.split('/')) or
        any(c in value for c in '*?[]{}:')):
        raise ValueError('unsafe relative root/body path')
    return value


def _vector(value, count, label, *, quaternion=False):
    if (not isinstance(value, (list, tuple)) or len(value) != count or
        any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError(f'invalid finite {label}')
    if quaternion and abs(sum(v*v for v in value)-1) > 1e-4:
        raise ValueError(f'non-unit quaternion {label}')


def _identity(identity):
    if not isinstance(identity, dict):
        raise ValueError('invalid identity')
    if any(not isinstance(identity.get(k), str) or not identity[k] for k in
           ('scenario_id', 'domain', 'morphology', 'split')):
        raise ValueError('incomplete identity')
    morphologies = dict(navigation='rc_car8', stairs='snake8', gap='snake8',
                       manipulation='mobile_manipulator8')
    if (identity['domain'] not in morphologies or
        identity['morphology'] != morphologies[identity['domain']] or
        identity['split'] not in ('train', 'validation', 'test') or
        type(identity.get('seed')) is not int or not _digest(identity.get('geometry_sha256'))):
        raise ValueError('incompatible identity')


def validate_connections(edges, owners):
    """Validate topology and exact local frames without importing USD."""
    if not isinstance(edges, list):
        raise ValueError('connections must be a list')
    occupied, pairs = set(), set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError('invalid connection')
        keys = []
        for side in ('first', 'second'):
            owner, port = edge.get(side+'_owner'), edge.get(side+'_port')
            if owner not in owners or port not in PORTS:
                raise ValueError('unknown owner/port')
            _relative(edge.get(side+'_body_relative_path'))
            keys.append((owner, port))
        pair = frozenset(k[0] for k in keys)
        if len(pair) != 2 or pair in pairs or any(k in occupied for k in keys):
            raise ValueError('duplicate connection or occupied port')
        occupied.update(keys); pairs.add(pair)
        for index in (0, 1):
            _vector(edge.get(f'local_pos{index}'), 3, 'joint local position')
            _vector(edge.get(f'local_rot{index}_xyzw'), 4, 'joint local rotation', quaternion=True)
        if edge.get('collision_enabled') is not False or edge.get('exclude_from_articulation') is not True:
            raise ValueError('incompatible joint physics')


def validate_snapshot(payload, *, expected_identity=None, expected_physics=None):
    """Return an owned validated payload; reject even correctly rehashed bad state."""
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('incompatible snapshot schema')
    _canonical(payload)  # rejects NaN/Infinity anywhere, including extra metadata
    if not _digest(payload.get('content_sha256')) or content_hash(payload) != payload['content_sha256']:
        raise ValueError('snapshot content hash mismatch')
    _identity(payload.get('identity'))
    physics = payload.get('physics')
    if (not isinstance(physics, dict) or not isinstance(physics.get('version'), str) or
        not physics['version'] or not _digest(physics.get('source_sha256'))):
        raise ValueError('invalid physics fingerprint')
    for label, expected, actual in (('identity', expected_identity, payload['identity']),
                                     ('physics', expected_physics, physics)):
        if expected is not None and any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{label} mismatch')
    owners = payload.get('owners')
    if (not isinstance(owners, list) or len(owners) != 8 or
        any(not isinstance(o, str) or not o for o in owners) or len(set(owners)) != 8):
        raise ValueError('exactly eight named owners required')
    modules = payload.get('modules')
    if not isinstance(modules, dict) or set(modules) != set(owners):
        raise ValueError('module owner mismatch')
    roots = set()
    for module in modules.values():
        root = _relative(module.get('root_relative_path'))
        if root in roots:
            raise ValueError('duplicate root path')
        roots.add(root)
        _vector(module.get('position_xyz_m'), 3, 'root position')
        _vector(module.get('quaternion_xyzw'), 4, 'root rotation', quaternion=True)
        _vector(module.get('root_velocity'), 6, 'root velocity')
        names = module.get('dof_names')
        if (not isinstance(names, list) or not names or
            any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
            raise ValueError('invalid named DOFs')
        for field in ('q', 'qd'):
            _vector(module.get(field), len(names), field)
        control = module.get('control')
        if not isinstance(control, dict) or set(control) != set(CONTROL_FIELDS):
            raise ValueError('incomplete control state')
        for field in CONTROL_FIELDS:
            _vector(control[field], len(names), field)
    validate_connections(payload.get('connections'), owners)
    count = payload.get('actual_target_edges')
    certificate = payload.get('physical_certificate')
    stage = payload.get('stage')
    # The Expert may advance multiple links in one synchronized wave. Stage
    # buckets therefore denote a range, not a fabricated exact link count.
    bucket = (('S0' if count == 0 else 'S1' if 1 <= count <= 2 else
               'S2' if 3 <= count <= 4 else 'S3' if 5 <= count <= 6 else None)
              if type(count) is int else None)
    if (payload.get('stage') not in STAGE_TARGET_COUNTS or type(count) is not int or
        not 0 <= count <= 6 or stage != bucket or count != len(payload['connections']) or
        not isinstance(certificate, dict) or certificate.get('expert_generated') is not True or
        certificate.get('target_compatible') is not True or
        type(certificate.get('target_edges')) is not int or certificate['target_edges'] != count or
        not isinstance(certificate.get('source'), str) or not certificate['source']):
        raise ValueError('invalid physical target certificate/stage/count')
    scenario = payload.get('scenario_state', {})
    if not isinstance(scenario, dict):
        raise ValueError('invalid scenario_state')
    if payload['identity']['domain'] == 'manipulation':
        plunger = scenario.get('plunger')
        if not isinstance(plunger, dict):
            raise ValueError('manipulation snapshot requires plunger state')
        _relative(plunger.get('root_relative_path'))
        _vector(plunger.get('position_xyz_m'), 3, 'plunger position')
        _vector(plunger.get('quaternion_xyzw'), 4, 'plunger rotation', quaternion=True)
        _vector(plunger.get('root_velocity'), 6, 'plunger velocity')
    elif scenario:
        raise ValueError('unexpected scenario state in non-manipulation snapshot')
    return deepcopy(payload)


def load_snapshot(path, *, expected_identity=None, expected_physics=None):
    return validate_snapshot(json.loads(Path(path).read_text()),
        expected_identity=expected_identity, expected_physics=expected_physics)


def save_snapshot(path, payload):
    """Seal and exclusively create a snapshot; never overwrite evidence."""
    value = deepcopy(payload)
    value['content_sha256'] = content_hash(value)
    value = validate_snapshot(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return value


class SnapshotBank:
    """Validate every file and split before exposing the selected split records.

    Manifest records contain ``path`` relative to the manifest, file-byte
    ``sha256``, full ``identity`` and ``stage``. Symlink escapes are rejected.
    Geometry and scenario seeds cannot occur in more than one split.
    """
    def __init__(self, manifest_path, split, expected_physics=None):
        if split not in ('train', 'validation', 'test'):
            raise ValueError('invalid bank split')
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.expected_physics = deepcopy(expected_physics)
        manifest_bytes = self.manifest_path.read_bytes()
        self.sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        if manifest.get('schema') != BANK_SCHEMA or not isinstance(manifest.get('records'), list):
            raise ValueError('incompatible bank schema')
        selected, seen_paths, split_keys = [], set(), {}
        for record in manifest['records']:
            path = self._path(record.get('path'))
            if path in seen_paths:
                raise ValueError('duplicate bank path')
            seen_paths.add(path)
            value = self._load(record)
            identity = value['identity']
            for key in (('geometry', identity['geometry_sha256']),
                        ('seed', identity['domain'], identity['seed']),
                        ('scenario', identity['scenario_id'])):
                previous = split_keys.setdefault(key, identity['split'])
                if previous != identity['split']:
                    raise ValueError('bank split leakage')
            if identity['split'] == split:
                selected.append(deepcopy(record))
        self.records = tuple(selected)

    def _path(self, relative):
        _relative(relative)
        path = (self.manifest_path.parent / relative).resolve()
        if not path.is_relative_to(self.manifest_path.parent):
            raise ValueError('bank path containment failure')
        return path

    def _load(self, record):
        raw = self._path(record.get('path')).read_bytes()
        if not _digest(record.get('sha256')) or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('bank file hash mismatch')
        _identity(record.get('identity'))
        value = validate_snapshot(json.loads(raw), expected_identity=record['identity'],
                                  expected_physics=self.expected_physics)
        if value['identity'] != record['identity'] or value['stage'] != record.get('stage'):
            raise ValueError('bank identity/stage mismatch')
        return value

    def load(self, record):
        if record not in self.records:
            raise ValueError('record is outside selected bank split')
        value = self._load(record)
        if value['identity']['split'] != self.split:
            raise ValueError('bank selected split mismatch')
        return value


def current_snapshot_physics():
    """Hash the physics/assets/Expert implementation, excluding runner/evaluator."""
    root = Path(__file__).resolve().parents[2]
    paths = [root/'drl/MARL'/name for name in (
        'c1_physics_profile.py', 'isaaclab_articulation.py', 'isaaclab_expert_runtime.py')]
    package = root/'scripts/smores_ep/src/smores_ep'
    paths.extend(package/'isaac'/name for name in (
        'docking.py', 'primitive_executor.py', 'dynamic_stage.py', 'physics_asset.py',
        'multi_module_stage.py', 'command_router.py'))
    paths.append(root/'robots/state_readers.py')
    paths.extend(p for directory in (package/'config', package/'primitives', package/'docking')
                 for p in directory.rglob('*.py'))
    paths.extend(p for p in (root/'assets/smores-ep').rglob('*')
                 if p.is_file() and p.suffix.lower() in ('.usd', '.usda', '.usdc'))
    expert = root/'mssr_ws/src/mssr_expert'
    paths.extend(expert.rglob('*.py'))
    paths.extend(expert.rglob('*.json'))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():
            raise FileNotFoundError(f'physics fingerprint source missing: {path}')
        digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes()+b'\0')
    return dict(version=PHYSICS_VERSION, source_sha256=digest.hexdigest())


def current_snapshot_identity(env, index):
    from drl.MARL.c1_train_sampling import _geometry_summary
    from drl.MARL.reward_calibration import MORPHOLOGIES
    domain, seed, split = env.domain_by_env[index], env.seed_by_env[index], env.cfg.course_split
    return dict(scenario_id=f'{domain}:{split}:{seed}', domain=domain, morphology=MORPHOLOGIES[domain],
                seed=int(seed), split=split, geometry_sha256=_geometry_summary(env.courses[index])['sha256'])


def capture_native_snapshot(env, index, stage, *, identity=None, physics=None, physical_certificate):
    """Read one current environment without stepping or changing controllers."""
    identity = current_snapshot_identity(env, index) if identity is None else identity
    physics = current_snapshot_physics() if physics is None else physics
    origin = [float(v) for v in env.scene.env_origins[index].cpu().tolist()]
    env_root = f'/World/envs/env_{index}/'
    modules = {}
    for owner, reader in env.state_readers_by_env[index].items():
        state = reader._articulation.capture_state()
        root = state.pop('root_path')
        if not root.startswith(env_root):
            raise ValueError('native module root outside selected environment')
        state['root_relative_path'] = root[len(env_root):]
        state['position_xyz_m'] = [v-o for v,o in zip(state['position_xyz_m'],origin,strict=True)]
        modules[owner] = state
    connections = env.docking_managers[index].export_connections()
    scenario_state = {}
    if identity['domain'] == 'manipulation':
        path = env.plunger_paths[index]
        view = env.native_transforms.rigid_views[path]
        pose = view.get_transforms().numpy()[0].tolist()
        scenario_state['plunger'] = dict(
            root_relative_path=path[len(env_root):] if path.startswith(env_root) else '',
            position_xyz_m=[v-o for v,o in zip(pose[:3], origin, strict=True)],
            quaternion_xyzw=pose[3:],
            root_velocity=view.get_velocities().numpy()[0].tolist())
    value = dict(schema=SCHEMA, stage=stage, identity=deepcopy(identity), physics=deepcopy(physics),
                 owners=list(env.cfg.possible_agents), modules=modules, connections=connections,
                 actual_target_edges=len(connections), scenario_state=scenario_state,
                 physical_certificate=deepcopy(physical_certificate))
    value['content_sha256'] = content_hash(value)
    return validate_snapshot(value)


def restore_native_snapshot(env, index, payload, *, expected_identity=None, expected_physics=None,
                            restore_connections=True):
    """Restore isolated roots after caller cancels options and detaches joints.

    Captured drive plans are provenance only. Restoration sets measured position
    targets, passive gains, zero drive velocities/forces. A fresh router must be
    initialized while detached before reconstructing joints.
    """
    value = validate_snapshot(payload,
        expected_identity=current_snapshot_identity(env,index) if expected_identity is None else expected_identity,
        expected_physics=current_snapshot_physics() if expected_physics is None else expected_physics)
    manager = env.docking_managers[index]
    if manager.connections:
        raise ValueError('snapshot restoration requires detached connections')
    backend = getattr(env, 'physical_expert_backends', ())[index]
    if backend['primitive_executor'].active_goals:
        raise ValueError('snapshot restoration requires canceled Expert plans')
    owners = tuple(env.cfg.possible_agents)
    if set(owners) != set(value['owners']):
        raise ValueError('native owner mismatch')
    origin = [float(v) for v in env.scene.env_origins[index].cpu().tolist()]
    states = {}
    for owner in owners:
        state = deepcopy(value['modules'][owner])
        root = env.module_roots_by_env[index][owner]
        if root != f'/World/envs/env_{index}/'+state.pop('root_relative_path'):
            raise ValueError('native root path mismatch')
        state['root_path'] = root
        state['position_xyz_m'] = [v+o for v,o in zip(state['position_xyz_m'],origin,strict=True)]
        access = env.state_readers_by_env[index][owner]._articulation
        access.validate_state(state)
        states[owner] = state
    manager.validate_connection_states(value['connections'])
    plunger_state = value.get('scenario_state', {}).get('plunger')
    plunger_view = None
    if plunger_state is not None:
        expected = f"/World/envs/env_{index}/" + plunger_state['root_relative_path']
        if getattr(env, 'plunger_paths', {}).get(index) != expected:
            raise ValueError('native plunger root mismatch')
        plunger_view = env.native_transforms.rigid_views[expected]
    for owner,state in states.items():
        env.state_readers_by_env[index][owner]._articulation.restore_state(state,restore_control=False)
    if plunger_view is not None:
        import warp as wp
        pose = [v+o for v,o in zip(plunger_state['position_xyz_m'],origin,strict=True)]
        pose.extend(plunger_state['quaternion_xyzw'])
        p = plunger_view.get_transforms()
        v = plunger_view.get_velocities()
        indices = wp.array([0],dtype=wp.uint32,device=p.device)
        plunger_view.set_transforms(wp.array([pose],dtype=wp.float32,device=p.device),indices)
        plunger_view.set_velocities(wp.array([plunger_state['root_velocity']],dtype=wp.float32,device=v.device),indices)
    env.native_transforms.invalidate()
    if restore_connections:
        manager.restore_connections(value['connections'])
    return value
