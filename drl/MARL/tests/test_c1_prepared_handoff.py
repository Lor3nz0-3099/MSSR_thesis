from types import SimpleNamespace

from drl.IL.bc_v1.bc_v1_contract import ConnectorRef, ModuleRef, ResourceRef
from drl.MARL.c1_feasibility import CandidateAvailability, NativeAvailability


def _relation(a='m5', fa='TOP', b='m1', fb='BOTTOM', *, docked=False):
    return SimpleNamespace(source_module=a, source_face=fa, target_module=b, target_face=fb, docked=docked)


def _goal(a, fa, b, fb):
    return SimpleNamespace(
        primitive=SimpleNamespace(value='align_faces'),
        module_ids=(a, b),
        parameters={'face_a': fa, 'face_b': fb},
    )


def _binding(modules, action, a, fa, b, fb):
    ma, mb = modules[a], modules[b]
    return (
        SimpleNamespace(action_id=action),
        SimpleNamespace(
            initiator=ma,
            connectors=(ConnectorRef(ma, fa), ConnectorRef(mb, fb)),
            required_scope_members=frozenset((ma, mb)),
            scope_members=frozenset((ma, mb)),
        ),
    )


def _route():
    modules = {name: ModuleRef(name) for name in ('m1', 'm4', 'm5', 'm6', 'm7', 'm8')}
    resources = [SimpleNamespace(resource_ref=ResourceRef(modules['m5'], 'locomotion')),
                 SimpleNamespace(resource_ref=ResourceRef(modules['m8'], 'locomotion'))]
    structural = [
        _binding(modules, 'align_faces', 'm4', 'BOTTOM', 'm5', 'LEFT'),
        _binding(modules, 'align_faces', 'm5', 'TOP', 'm1', 'BOTTOM'),
        _binding(modules, 'dock', 'm5', 'TOP', 'm1', 'BOTTOM'),
        _binding(modules, 'align_faces', 'm8', 'BOTTOM', 'm6', 'TOP'),
    ]
    return dict(module_refs=tuple(modules.values()), resource_candidates=resources,
                structural_bindings=structural, argument_domains=())


def _available(route):
    count = len(route['resource_candidates']) + len(route['structural_bindings'])
    return NativeAvailability((False,) * count, (CandidateAvailability('available'),) * count)


class _Executor:
    def __init__(self, *, active=(), ready=True):
        self.active_goals = tuple(active)
        self.ready = ready
    def prepared_relation_snapshot(self):
        return (_relation(),)
    def dock_readiness_for_pairs(self, pairs):
        return tuple(self.ready for _ in pairs)


def test_prepared_component_waits_for_existing_align_and_blocks_new_work():
    from drl.MARL.c1_feasibility import apply_prepared_handoff
    route = _route()
    executor = _Executor(active=(_goal('m7', 'BOTTOM', 'm1', 'TOP'),), ready=True)
    result = apply_prepared_handoff(_available(route), route, executor)
    # resource m5 blocked, independent m8 survives
    assert result.candidate_mask[:2] == (True, False)
    # m4->m5 blocked; exact re-align blocked; matching dock waits; independent survives
    assert result.candidate_mask[2:] == (True, True, True, False)
    assert result.reasons[4].code == 'masked_handoff_wait'
    assert result.reasons[2].code == 'masked_prepared_conflict'


def test_idle_prepared_component_exposes_matching_dock_but_not_new_align():
    from drl.MARL.c1_feasibility import apply_prepared_handoff
    route = _route()
    result = apply_prepared_handoff(_available(route), route, _Executor(ready=True))
    assert result.candidate_mask[2:] == (True, True, False, False)
    assert result.reasons[4].code == 'available'


def test_idle_prepared_relation_that_lost_proximity_allows_only_exact_realign():
    from drl.MARL.c1_feasibility import apply_prepared_handoff
    route = _route()
    result = apply_prepared_handoff(_available(route), route, _Executor(ready=False))
    assert result.candidate_mask[2:] == (True, False, True, False)
    assert result.reasons[3].code == 'available'
    assert result.reasons[4].code == 'masked_handoff_not_ready'


def test_executor_keeps_docked_relation_until_whole_prepared_component_finishes():
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
    from smores_ep.primitives.model import PrimitiveGoal, PrimitiveState
    from scripts.smores_ep.tests.test_primitives import _FakeDocking, _FakeStateReader
    ids = ('m1', 'm5', 'm7')
    docking = _FakeDocking(); docking.module_ids = ids
    executor = IsaacPrimitiveExecutor(stage=object(), module_roots={m: '/' + m for m in ids},
        states={m: _FakeStateReader() for m in ids}, docking=docking)

    def goal(gid, primitive, a, fa, b, fb):
        return PrimitiveGoal.from_dict(dict(goal_id=gid, primitive=primitive, module_ids=[a, b],
            parameters=dict(face_a=fa, face_b=fb)))
    def finish(g, code):
        assert executor.submit(g, 0.).state is PrimitiveState.ACCEPTED
        executor._finish(executor._active[g.goal_id], PrimitiveState.SUCCEEDED, .1, code, code)

    finish(goal('a1', 'align_faces', 'm5', 'TOP', 'm1', 'BOTTOM'), 'FACES_IN_CONTACT')
    finish(goal('a2', 'align_faces', 'm7', 'BOTTOM', 'm1', 'TOP'), 'FACES_IN_CONTACT')
    rows = executor.prepared_relation_snapshot()
    assert len(rows) == 2 and not any(r.docked for r in rows)

    finish(goal('d1', 'dock', 'm5', 'TOP', 'm1', 'BOTTOM'), 'DOCKED')
    rows = executor.prepared_relation_snapshot()
    assert len(rows) == 2 and sum(r.docked for r in rows) == 1

    finish(goal('d2', 'dock', 'm7', 'BOTTOM', 'm1', 'TOP'), 'DOCKED')
    assert executor.prepared_relation_snapshot() == ()


def test_gui_display_uses_teleop_performance_resolution():
    from drl.MARL.c1_gui_viewer import configure_viewer_display
    values={'/isaaclab/render/active_viewport': True}
    class Settings:
        def get(self, name): return values.get(name)
        def set_bool(self, name, value): values[name] = value
        def set_int(self, name, value): values[name] = value
    result=configure_viewer_display(SimpleNamespace(config={'headless':False}), Settings(), requested_gui=True)
    assert result['window_width'] == 960 and result['window_height'] == 540
    assert values['/app/window/width'] == 960 and values['/app/window/height'] == 540
    assert values['/app/renderer/resolution/width'] == 960
    assert values['/app/renderer/resolution/height'] == 540


def test_prepared_handoff_blocks_work_on_topology_connected_sibling_branch():
    """Prepared reservations follow the rigid component, not only prepared edges."""
    from drl.MARL.c1_feasibility import apply_prepared_handoff

    modules = {
        name: ModuleRef(name)
        for name in ("m1", "m3", "m4", "m5", "m6", "m7", "m8")
    }
    structural = [
        _binding(modules, "align_faces", "m4", "BOTTOM", "m5", "LEFT"),
        _binding(modules, "dock", "m4", "BOTTOM", "m5", "LEFT"),
        _binding(modules, "align_faces", "m3", "BOTTOM", "m7", "TOP"),
        _binding(modules, "align_faces", "m8", "BOTTOM", "m6", "TOP"),
    ]
    route = dict(
        module_refs=tuple(modules.values()),
        resource_candidates=(),
        structural_bindings=structural,
        argument_domains=(),
    )

    class Executor:
        active_goals = ()

        def prepared_relation_snapshot(self):
            return (_relation("m4", "BOTTOM", "m5", "LEFT"),)

        def dock_readiness_for_pairs(self, pairs):
            return tuple(True for _ in pairs)

        def connected_component_module_ids(self, module_ids):
            seeds = {str(module_id) for module_id in module_ids}
            rigid = {"m1", "m5", "m7"}
            if seeds.intersection(rigid):
                seeds.update(rigid)
            return tuple(sorted(seeds))

    result = apply_prepared_handoff(_available(route), route, Executor())

    # Exact prepared ALIGN remains blocked because the pair is already ready;
    # its matching DOCK stays exposed.  Work on m7 is blocked because m7 is
    # rigidly connected to m5 through m1.  Independent m8->m6 remains free.
    assert result.candidate_mask == (True, False, True, False)
    assert result.reasons[1].code == "available"
    assert result.reasons[2].code == "masked_prepared_conflict"
    assert result.reasons[3].code == "available"
