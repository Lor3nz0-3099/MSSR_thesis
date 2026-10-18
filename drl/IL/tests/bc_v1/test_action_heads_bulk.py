"""P1: preserve the H2 contract without argument-row scalar extraction."""
import copy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.tests.bc_v1.g2_fixtures import tensorize
from drl.IL.tests.bc_v1.test_action_heads import enriched_fixture, heads_for
from drl.IL.tests.bc_v1.test_model import model_for
from drl.IL.tests.bc_v1.h2_legacy_reference import legacy_validate_arguments


@pytest.fixture(autouse=True, scope='module')
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(copies=2):
    sample, kw = enriched_fixture()
    batch = collate_policy_samples([tensorize(sample, kw)] * copies)
    return batch['inputs'], heads_for(kw), kw


class ScalarReads(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_bulk_validator_scalar_reads_are_bounded_not_per_argument_row(device):
    # Reintroducing int/bool(tensor[row]) must break this independently of
    # machine speed. CPU exercises the same control flow without requiring GPU.
    counts = []
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    for copies in (1, 8):
        x, heads, _ = example(copies)
        if device == 'cuda':
            from drl.IL.bc_v1.training import move_batch_to_device
            x = move_batch_to_device({'inputs': x, 'supervision': {}}, device)['inputs']
        assert hasattr(heads, '_validate_arguments_bulk'), 'Missing bulk H2 validator'
        with ScalarReads() as reads:
            heads._validate_arguments(x)
        counts.append(reads.count)
    assert max(counts) <= 32
    assert counts[0] == counts[1]


def test_characterization_all_argument_kinds_and_fixed_routing():
    x, heads, kw = example()
    legacy_validate_arguments(heads, x)
    heads._validate_arguments(x)
    s = x['structural']
    assert set(s['argument_kind'].tolist()) == set(range(8))
    out = heads.forward_structural(model_for(kw).encode(x), x)
    assert torch.equal(out['group_ptr'], s['group_ptr'])
    assert out['candidate_logits'].numel() == len(s['initiator'])
    assert all(v['argument'].numel() for v in out['arguments'].values())
    predicted = {i for row in out['arguments'].values() for i in row['argument'].tolist()}
    for i, pid in enumerate(s['argument_parameter'].tolist()):
        if not heads.parameters_catalog[pid].learnable or s['argument_bound_ref'][i] != -1:
            assert i not in predicted
    # Endpoint-free skill candidates and multiple independent initiators persist.
    assert (s['peer'] == -1).any()
    assert s['initiator'].unique().numel() == 6
    for records in (out['scope'], out['arguments']['reference_set']):
        assert set(records['domain_index'].tolist()).isdisjoint(records['required_domain_index'].tolist())


def test_characterization_empty_reference_domain_and_required_members():
    x, heads, _ = example(1)
    s = x['structural']
    row = int(torch.nonzero(s['argument_kind'] == 6)[0])
    a, b = s['argument_domain_ptr'][row:row+2].tolist()
    s['argument_domain_index'] = torch.cat((s['argument_domain_index'][:a], s['argument_domain_index'][b:]))
    s['argument_domain_required'] = torch.cat((s['argument_domain_required'][:a], s['argument_domain_required'][b:]))
    s['argument_domain_ptr'][row+1:] -= b-a
    legacy_validate_arguments(heads, x)
    heads._validate_arguments(x)  # Empty unbound set domain is legal.
    s['argument_domain_required'][:] = True
    legacy_validate_arguments(heads, x)
    heads._validate_arguments(x)


def test_bulk_validation_rechecks_mutated_nonlearnable_routing():
    # A previous successful call cannot authorize a later malformed batch,
    # including executor-owned fields that produce no trainable output.
    x, heads, _ = example(1)
    heads._validate_arguments(x)
    s = x['structural']
    pid = next(i for i, (_, name) in enumerate(heads.parameter_catalog) if name == 'ignored')
    row = int(torch.nonzero(s['argument_parameter'] == pid)[0])
    s['argument_kind'][row] = 3
    with pytest.raises(ValueError):
        legacy_validate_arguments(heads, x)
    with pytest.raises(ValueError):
        heads._validate_arguments(x)


def malformed(x, name):
    x = copy.deepcopy(x)
    s = x['structural']
    if name == 'nonmonotonic_pointer':
        s['argument_domain_ptr'][2] = -1
    elif name == 'pointer_out_of_range':
        s['argument_domain_ptr'][-1] += 1
    elif name == 'index_out_of_range':
        s['argument_domain_index'][0] = 99999
    elif name == 'wrong_owner':
        s['argument_domain_index'][0] = len(x['connector_x']) // 2
    elif name == 'nonreference_domain':
        u = int(torch.nonzero(s['argument_kind'] == 0)[0])
        pos = int(s['argument_domain_ptr'][u])
        for key, value in [('argument_domain_index', 0), ('argument_domain_required', False)]:
            s[key] = torch.cat((s[key][:pos], s[key].new_tensor([value]), s[key][pos:]))
        s['argument_domain_ptr'][u+1:] += 1
    elif name == 'bound_outside_domain':
        u = int(torch.nonzero(s['argument_bound_ref'] >= 0)[0])
        s['argument_bound_ref'][u] = 99999
    elif name == 'invalid_category_kind':
        u = int(torch.nonzero(s['argument_kind'] == 3)[0])
        s['argument_kind'][u] = 99
    elif name == 'invalid_binding':
        s['argument_binding'][0] = len(s['initiator'])
    elif name == 'wrong_action_binding':
        s['argument_parameter'][0] = int(s['argument_parameter'][-1])
    elif name == 'invalid_required_mask':
        s['argument_domain_required'] = s['argument_domain_required'].long()
    elif name == 'invalid_scope_member':
        s['scope_members'][0] = len(x['module_x'])
    elif name == 'cross_graph_scope':
        s['scope_members'][0] = len(x['module_x']) // 2
    elif name == 'shape_mismatch':
        s['argument_kind'] = s['argument_kind'][:, None]
    elif name == 'dtype_mismatch':
        s['argument_binding'] = s['argument_binding'].float()
    elif name == 'missing_routing':
        del s['argument_bound_ref']
    else:
        raise AssertionError(name)
    return x


INVALID_CASES = ('nonmonotonic_pointer', 'pointer_out_of_range', 'index_out_of_range',
    'wrong_owner', 'nonreference_domain', 'bound_outside_domain', 'invalid_category_kind',
    'invalid_binding', 'wrong_action_binding', 'invalid_required_mask', 'invalid_scope_member',
    'cross_graph_scope', 'shape_mismatch', 'dtype_mismatch', 'missing_routing')


@pytest.mark.parametrize('name', INVALID_CASES)
def test_characterization_invalid_inputs_still_fail_closed(name):
    x, heads, kw = example()
    latent = model_for(kw).encode(x)
    broken = malformed(x, name)
    # Legacy scope checks are in structural conditions, not argument validation.
    if 'scope' not in name:
        with pytest.raises((ValueError, KeyError, IndexError)):
            legacy_validate_arguments(heads, broken)
    with pytest.raises((ValueError, KeyError, IndexError)):
        heads.forward_structural(latent, broken)
