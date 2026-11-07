from types import SimpleNamespace
import torch

from drl.IL.bc_v1.bc_v1_contract import ModuleRef, ConnectorRef
from smores_ep.docking.model import DockingFace, DockingFacePose, DockingThresholds


def _pose(module, face, x, normal):
    return DockingFacePose(
        DockingFace(module, face, f'/{module}/{face}', f'/{module}/body'),
        (x, 0.0, 0.0), normal, (0.0, 1.0, 0.0),
    )


class _Docking:
    module_ids = ('module_a', 'module_b')
    thresholds = DockingThresholds()
    def __init__(self, gap):
        self.gap = gap
        self.calls = []
    def face_poses_for(self, module):
        self.calls.append(module)
        if module == 'module_a':
            return (_pose(module, 'TOP', 0.0, (1.0, 0.0, 0.0)),)
        return (_pose(module, 'BOTTOM', self.gap, (-1.0, 0.0, 0.0)),)


def _executor(gap):
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
    docking = _Docking(gap)
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={'module_a': '/a', 'module_b': '/b'},
        states={'module_a': object(), 'module_b': object()},
        docking=docking,
    )
    return executor, docking


def _route(action_id='dock'):
    a, b = ModuleRef('module_a'), ModuleRef('module_b')
    binding = SimpleNamespace(
        initiator=a,
        connectors=(ConnectorRef(a, 'TOP'), ConnectorRef(b, 'BOTTOM')),
        required_scope_members=frozenset((a, b)),
        scope_members=frozenset((a, b)),
    )
    return dict(module_refs=(a, b), resource_candidates=(),
                structural_bindings=[(SimpleNamespace(action_id=action_id), binding)],
                argument_domains=())


def test_executor_dock_readiness_uses_exact_docking_gate_and_batches_face_reads():
    executor, docking = _executor(0.0020)
    pairs = (('module_a', 'TOP', 'module_b', 'BOTTOM'),) * 2
    assert executor.dock_readiness_for_pairs(pairs) == (True, True)
    assert docking.calls == ['module_a', 'module_b']
    executor_far, _ = _executor(0.020)
    assert executor_far.dock_readiness_for_pairs(pairs[:1]) == (False,)


def test_c1_masks_nonready_dock_but_does_not_mask_align():
    from drl.MARL.c1_feasibility import capture_native_availability
    executor, _ = _executor(0.020)
    availability = capture_native_availability(
        routing=_route('dock'), graph={}, hard_mask=torch.zeros(1, dtype=torch.bool), executor=executor)
    assert availability.candidate_mask == (True,)
    assert availability.reasons[0].code == 'masked_not_ready'
    align = capture_native_availability(
        routing=_route('align_faces'), graph={}, hard_mask=torch.zeros(1, dtype=torch.bool), executor=executor)
    assert align.candidate_mask == (False,)


def test_dock_readiness_is_fresh_even_when_busy_cache_hits():
    from drl.MARL.c1_feasibility_cache import NativeAvailabilityCache
    executor, docking = _executor(0.020)
    cache = NativeAvailabilityCache()
    kwargs = dict(routing=_route('dock'), graph={}, hard_mask=torch.zeros(1, dtype=torch.bool),
                  executor=executor, generation=1)
    first = cache.capture(**kwargs)
    assert first.candidate_mask == (True,)
    docking.gap = 0.0020
    second = cache.capture(**kwargs)
    assert second.candidate_mask == (False,)
    assert cache.metrics()['mask_cache_hits'] == 1


def test_action_diagnostics_count_not_ready_separately(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.c1_feasibility import capture_native_availability
    from drl.MARL.policy import MARLCandidateRecord
    executor, _ = _executor(0.020)
    availability = capture_native_availability(
        routing=_route('dock'), graph={}, hard_mask=torch.zeros(1, dtype=torch.bool), executor=executor)
    record = MARLCandidateRecord(0, 'structural',
        'structural|dock|module_a|module_a:TOP>module_b:BOTTOM',
        'module_a', ('module_b',), torch.tensor(0.0), 4.5)
    events = C1ActionEvents(tmp_path, masked_detail_limit=1)
    rows = events.mask_events(records=(record,), availability=availability,
        action=SimpleNamespace(presence=torch.zeros(1)),
        identity=dict(env_index=0, generation=1, task='navigation', seed=1, timestamp_s=0.0))
    assert events.summary()['masked_not_ready_count'] == 1
    assert rows[0]['availability_state'] == 'masked_not_ready'
    assert rows[0]['drop_reason'] == 'dock_not_geometrically_ready'


def test_executor_dock_readiness_accepts_proximal_aligned_contact_closure():
    pair = (('module_a', 'TOP', 'module_b', 'BOTTOM'),)
    executor, _ = _executor(0.010)
    assert executor.dock_readiness_for_pairs(pair) == (True,)
    too_far, _ = _executor(0.013)
    assert too_far.dock_readiness_for_pairs(pair) == (False,)
