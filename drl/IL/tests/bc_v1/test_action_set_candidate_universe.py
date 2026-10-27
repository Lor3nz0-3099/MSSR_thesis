"""Certification: targets label the full physical universe; they never select it."""
from dataclasses import replace
import json

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.action_set_targets import collate_action_set_samples, candidate_index
from drl.IL.bc_v1.action_set_head import ActionSetPolicy
from drl.IL.bc_v1.action_set_loss import compute_action_set_loss
from drl.IL.bc_v1.candidates import enumerate_candidates
from drl.IL.bc_v1.sidecar import SidecarDataset
from drl.IL.bc_v1.sidecar_build import load_sources
from drl.IL.bc_v1.train_action_set import measure_batch
from drl.IL.tests.bc_v1.g2_fixtures import ROOT, fixture, tensorize
from drl.IL.tests.bc_v1.test_action_set_head import policy_for
from drl.IL.tests.bc_v1.test_model import equal_trees


def test_candidate_universe_is_identical_for_one_vs_two_different_expert_actions():
    sample, kw = fixture(n=3)
    resources = sample.inputs.policy.graph.resources
    a = c.SupervisionBundle((c.ResourceTarget(resources[0].ref, 'aim', (0.,)),))
    b = c.SupervisionBundle((c.ResourceTarget(resources[1].ref, 'pace', (2., 3.)),
                             c.ResourceTarget(resources[2].ref, 'aim', (1.,))))
    rows = []
    for targets in (a, b):
        table = enumerate_candidates(sample.inputs.policy.graph, kw['spec'], kw['feature_schema'])
        rows.append(tensorize(replace(sample, inputs=replace(sample.inputs, candidates=table), supervision=targets), kw))
    assert candidate_index(rows[0]['routing']) == candidate_index(rows[1]['routing'])
    for key in ('resource_candidates', 'structural_bindings', 'argument_domains'):
        assert rows[0]['routing'][key] == rows[1]['routing'][key]
    equal_trees(rows[0]['inputs'], rows[1]['inputs'])
    batches = [collate_action_set_samples([row]) for row in rows]
    equal_trees(batches[0]['inputs'], batches[1]['inputs'])
    labels = [batch['supervision']['action_set']['presence'] for batch in batches]
    assert [int(y.sum()) for y in labels] == [1, 2]
    assert not (labels[0] & labels[1]).any() and len(labels[0]) == 57
    policy = policy_for(kw)
    with torch.no_grad():
        equal_trees(policy(batches[0]['inputs']), policy(batches[1]['inputs']))


@pytest.fixture(scope='module')
def real_cases():
    source = ROOT/'artifacts/bc_v1/sidecar/g2_v3_resource_contract'
    if not (source/'manifest.json').is_file(): pytest.skip('External TRAIN G2 fixture unavailable')
    manifest = json.loads((source/'manifest.json').read_text())
    kw = load_sources()['kwargs']
    read = lambda path: json.loads(path.read_text())
    torch.manual_seed(20261021)
    model = ActionSetPolicy(kw['spec'], kw['feature_schema'], read(ROOT/'configs/bc_v1/gnn_v1.json'),
                            read(ROOT/'configs/bc_v1/action_set_head_v1.json')).eval()
    with SidecarDataset(source, 'train', expected_identity=manifest['identity']) as data:
        result = {}
        for index in (0, 3063, 9260):
            row = data[index]; batch = collate_action_set_samples([row])
            with torch.no_grad(): prediction = model(batch['inputs'])
            result[index] = row, batch, prediction
    return result


@pytest.mark.parametrize('index,positive', [(0, 1), (3063, 6), (9260, 1)])
def test_real_train_h2_scores_every_positive_and_negative_candidate(real_cases, index, positive):
    row, batch, prediction = real_cases[index]
    x = batch['inputs']; y = batch['supervision']['action_set']['presence']
    total = len(x['resource_candidates']['resource']) + len(x['structural']['initiator'])
    assert int(y.sum()) == positive and total > 100 * positive
    assert len(prediction['presence_logits']) == total == len(y)
    assert int((~y).sum()) == total - positive
    loss, families = compute_action_set_loss(prediction, batch)
    assert torch.isfinite(loss) and families['presence']['support'] == 1
    expected = torch.nn.functional.softplus(-prediction['presence_logits'][y]).mean()
    expected += torch.nn.functional.softplus(prediction['presence_logits'][~y]).mean()
    torch.testing.assert_close(families['presence']['mean'], expected)
    if index == 3063:
        audit = row['routing']['audit']
        assert audit.episode_id == 'expert:rc_car:seed-005100' and audit.transition_index == 198
    if index == 0:
        assert not row['routing']['physical_supervision'].resource_targets
    if index == 9260:
        assert not row['routing']['physical_supervision'].structural_targets


def test_full_real_loss_penalizes_one_extra_or_missing_action_with_unchanged_targets(real_cases):
    _, batch, prediction = real_cases[3063]
    labels = batch['supervision']['action_set']['presence']; before = labels.clone()
    correct = {**prediction, 'presence_logits': torch.where(labels, 8., -8.)}
    extra = {**correct, 'presence_logits': correct['presence_logits'].clone()}
    missing = {**correct, 'presence_logits': correct['presence_logits'].clone()}
    extra['presence_logits'][torch.nonzero(~labels)[0, 0]] = 8.
    missing['presence_logits'][torch.nonzero(labels)[0, 0]] = -8.
    losses = [float(compute_action_set_loss(p, batch)[0]) for p in (correct, extra, missing)]
    metrics = [measure_batch(p, batch).compute() for p in (correct, extra, missing)]
    assert losses[1] > losses[0] and losses[2] > losses[0]
    assert metrics[0]['false_positive_actions'] == metrics[0]['false_negative_actions'] == 0
    assert metrics[1]['false_positive_actions'] == 1 and metrics[1]['false_negative_actions'] == 0
    assert metrics[2]['false_negative_actions'] == 1 and metrics[2]['false_positive_actions'] == 0
    assert torch.equal(labels, before)
