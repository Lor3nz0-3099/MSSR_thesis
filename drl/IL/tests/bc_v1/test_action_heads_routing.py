"""P1b scope/routing semantics and bounded host scalar work."""
import copy
import types

import pytest
import torch

from drl.IL.bc_v1.training import move_batch_to_device
from drl.IL.tests.bc_v1.test_action_heads_bulk import example, ScalarReads
from drl.IL.tests.bc_v1.test_action_heads import reroute_structural
from drl.IL.tests.bc_v1.test_model import model_for, equal_trees
from drl.IL.tests.bc_v1.h2_scope_routing_reference import PreP1bRouting


@pytest.fixture(autouse=True, scope='module')
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def reference(heads):
    result = copy.deepcopy(heads)
    for name in ('_scope', '_parameter_conditions', '_arguments'):
        setattr(result, name, types.MethodType(getattr(PreP1bRouting, name), result))
    return result


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_scope_host_scalar_reads_do_not_scale_with_candidates(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    counts = []
    for copies in (1, 8):
        x, heads, _ = example(copies)
        heads.to(device)
        x = move_batch_to_device({'inputs': x, 'supervision': {}}, device)['inputs']
        latent = {'module_h': torch.zeros((len(x['module_x']), 128), device=device)}
        conditions = torch.zeros((len(x['structural']['initiator']), 128), device=device)
        with ScalarReads() as calls:
            heads._scope(latent, x, conditions)
        counts.append(calls.count)
    assert max(counts) <= 16, counts
    assert counts[0] == counts[1], counts


@pytest.mark.parametrize('kind', ['mixed', 'all_required', 'empty', 'duplicate', 'permuted'])
def test_scope_characterization_matches_p1_without_domain_changes(kind):
    x, heads, kw = example()
    s = x['structural']
    if kind == 'all_required':
        s['scope_required'][:] = True
    elif kind == 'empty':
        s['scope_members'] = s['scope_members'][:0]
        s['scope_required'] = s['scope_required'][:0]
        s['scope_ptr'].zero_()
    elif kind == 'duplicate':
        position = int(torch.nonzero(~s['scope_required'])[0])
        owner = int(torch.searchsorted(s['scope_ptr'][1:], torch.tensor(position), right=True))
        for key in ('scope_members', 'scope_required'):
            s[key] = torch.cat((s[key][:position], s[key][position:position+1], s[key][position:]))
        s['scope_ptr'][owner+1:] += 1
    elif kind == 'permuted':
        order = []
        for a, b in zip(s['group_ptr'][:-1].tolist(), s['group_ptr'][1:].tolist()):
            order.extend(range(b-1, a-1, -1))
        x = reroute_structural(x, order)
    h = model_for(kw).encode(x)
    expected = reference(heads).forward_structural(h, x)
    actual = heads.forward_structural(h, x)
    equal_trees(expected, actual)
    assert torch.equal(actual['group_ptr'], x['structural']['group_ptr'])
    if kind in ('all_required', 'empty'):
        assert actual['scope']['binding'].numel() == actual['scope']['logits'].numel() == 0
        assert actual['scope']['ptr'].tolist() == [0]


@pytest.mark.parametrize('kind', ['pointer', 'range', 'owner', 'mask_shape', 'mask_dtype', 'missing'])
def test_scope_invalid_inputs_rejected_before_and_after(kind):
    x, heads, kw = example()
    h = model_for(kw).encode(x)
    s = x['structural']
    if kind == 'pointer':
        s['scope_ptr'][2] = -1
    elif kind == 'range':
        s['scope_members'][0] = len(x['module_x'])
    elif kind == 'owner':
        s['scope_members'][0] = len(x['module_x']) // 2
    elif kind == 'mask_shape':
        s['scope_required'] = s['scope_required'][:, None]
    elif kind == 'mask_dtype':
        s['scope_required'] = s['scope_required'].long()
    elif kind == 'missing':
        del s['scope_ptr']
    for candidate in (reference(heads), heads):
        with pytest.raises((ValueError, KeyError, IndexError)):
            candidate.forward_structural(h, x)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_argument_selection_and_category_routes_have_bounded_scalar_reads(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    counts = []
    for copies in (1, 8):
        x, heads, _ = example(copies)
        heads.to(device)
        x = move_batch_to_device({'inputs': x, 'supervision': {}}, device)['inputs']
        assert hasattr(heads, '_select_argument_rows'), 'Missing bulk argument selection'
        assert hasattr(heads, '_category_routes'), 'Missing bulk category routing'
        conditions = torch.zeros((len(x['structural']['initiator']), 128), device=device)
        with ScalarReads() as calls:
            selected = heads._select_argument_rows(x['structural'])
            arguments, _ = heads._parameter_conditions(conditions, x['structural'], selected['category'])
            query, options, ptr = heads._category_routes(x['structural'], arguments)
        counts.append(calls.count)
        assert query.numel() == options.numel() == 105 * copies
        assert ptr.numel() == 51 * copies + 1
    assert max(counts) <= 16, counts
    assert counts[0] == counts[1], counts


@pytest.mark.parametrize('kind', ['mixed', 'required_set', 'empty_set', 'empty_reference'])
def test_argument_routing_characterization_keeps_all_kinds_and_domains(kind):
    x, heads, kw = example(1)
    s = x['structural']
    if kind != 'mixed':
        target_kind = 4 if kind == 'empty_reference' else 6
        row = int(torch.nonzero(s['argument_kind'] == target_kind)[0])
        a, b = s['argument_domain_ptr'][row:row+2].tolist()
        if kind == 'required_set':
            s['argument_domain_required'][a] = True
        else:
            for key in ('argument_domain_index', 'argument_domain_required'):
                s[key] = torch.cat((s[key][:a], s[key][b:]))
            s['argument_domain_ptr'][row+1:] -= b-a
    h = model_for(kw).encode(x)
    before = reference(heads).forward_structural(h, x)
    after = heads.forward_structural(h, x)
    equal_trees(before, after)
    records = after['arguments']
    assert records['numeric']['value'].numel() == 9
    assert records['category']['logits'].numel() == 105
    assert records['boolean']['logits'].numel() == 3
    assert records['reference']['argument'].numel() == 6
    assert records['reference_set']['argument'].numel() == 6
