"""B3: inspect gradients at real optimizer step boundaries, on tiny leaves."""
import pytest
import torch

from drl.MARL.mappo import MSSRMAPPO
from drl.MARL.tests.c1_smdp_fixtures import make_learner, frame


def optimizers_with_gradients(device, gradients):
    parameters, optimizers, observed = [], [], []
    for vector in gradients:
        # Two param groups ensure the norm covers the whole optimizer.
        pair = [torch.nn.Parameter(torch.zeros((), device=device)) for _ in vector]
        optimizer = torch.optim.SGD([{'params': [p]} for p in pair], lr=.1)
        values = []
        def capture(opt, args, kwargs, leaves=pair, values=values):
            values.append(torch.stack([p.grad for p in leaves]).detach().clone())
        optimizer.register_step_pre_hook(capture)
        parameters.append(pair)
        optimizers.append(optimizer)
        observed.append(values)
    losses = [sum(p*g for p, g in zip(pair, vector)) for pair, vector in zip(parameters, gradients)]
    return parameters, optimizers, observed, losses


DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('limit', [.5, .25, 0.])
def test_all_three_optimizers_receive_configured_clipped_gradients(device, limit):
    leaves, opts, observed, losses = optimizers_with_gradients(device, [(3., 4.), (0., 10.), (6., 8.)])
    metrics = MSSRMAPPO.coordinated_altruistic_optimizer_step(
        owner_policy_losses={'a': losses[0]}, owner_value_loss=losses[1], mean_field_value_loss=losses[2],
        actor_optimizer=opts[0], owner_critic_optimizer=opts[1], mean_field_critic_optimizer=opts[2],
        grad_norm_clip=limit)
    for prefix, before, gradients, params in zip(
            ['actor', 'owner_critic', 'mean_field_critic'], [5., 10., 10.], observed, leaves):
        expected = min(before, limit) if limit > 0 else before
        assert len(gradients) == 1
        assert gradients[0].norm().item() == pytest.approx(expected, abs=1e-6)
        assert metrics[f'{prefix}_grad_norm'] == pytest.approx(before)
        assert metrics[f'{prefix}_grad_norm_post_clip'] == pytest.approx(expected, abs=1e-6)
        torch.testing.assert_close(torch.stack([p.detach() for p in params]), -.1*gradients[0])


@pytest.mark.parametrize('device', DEVICES)
def test_freeze_skips_actor_backward_and_step_while_both_critics_are_clipped(device):
    leaves, opts, observed, losses = optimizers_with_gradients(device, [(3., 4.), (0., 10.), (6., 8.)])
    for p in leaves[0]:
        p.grad = torch.ones_like(p)
    metrics = MSSRMAPPO.coordinated_altruistic_optimizer_step(
        owner_policy_losses={'a': losses[0]}, owner_value_loss=losses[1], mean_field_value_loss=losses[2],
        actor_optimizer=opts[0], owner_critic_optimizer=opts[1], mean_field_critic_optimizer=opts[2],
        grad_norm_clip=.5, actor_updates_enabled=False)
    assert not observed[0]
    assert all(p.grad is None and p.item() == 0. for p in leaves[0])
    assert metrics['actor_grad_norm'] == metrics['actor_grad_norm_post_clip'] == 0.
    assert metrics['actor_optimizer_steps'] == 0
    assert observed[1][0].norm().item() == pytest.approx(.5, abs=1e-6)
    assert observed[2][0].norm().item() == pytest.approx(.5, abs=1e-6)


def test_below_limit_gradients_and_loss_values_are_unchanged():
    _, opts, observed, losses = optimizers_with_gradients('cpu', [(.03, .04), (0., .1), (.06, .08)])
    values = [loss.item() for loss in losses]
    metrics = MSSRMAPPO.coordinated_altruistic_optimizer_step(
        owner_policy_losses={'a': losses[0]}, owner_value_loss=losses[1], mean_field_value_loss=losses[2],
        actor_optimizer=opts[0], owner_critic_optimizer=opts[1], mean_field_critic_optimizer=opts[2],
        grad_norm_clip=.5)
    for actual, expected in zip(observed, [(.03, .04), (0., .1), (.06, .08)]):
        torch.testing.assert_close(actual[0], torch.tensor(expected))
    assert [metrics[k].item() for k in ['policy_loss', 'owner_value_loss', 'mean_field_value_loss']] == values


@pytest.mark.parametrize('configured', [.123, {'a': .123}])
def test_actual_learner_update_forwards_its_configured_limit(configured):
    learner = make_learner()
    learner.cfg.grad_norm_clip = configured
    for name, value in [('learning_epochs', 1), ('mini_batches', 1), ('ratio_clip', .2),
                        ('value_clip', .2), ('entropy_loss_scale', .001), ('value_loss_scale', 1.)]:
        setattr(learner.cfg, name, {'a': value})
    learner.actor_optimizer = None
    frame(learner, [True, True], [0., 0.], [1., 1.])
    frame(learner, [True, True], [1., 1.], [2., 2.])
    frame(learner, [True, True], [2., 2.], [3., 3.])
    learner.reevaluate_structured_minibatch = lambda **kw: (
        {'a': torch.zeros(len(kw['structured_batch']), 1)},
        {'a': torch.zeros(len(kw['structured_batch']), 1)}, None)
    class Reached(Exception):
        pass
    def step(**kwargs):
        assert kwargs.get('grad_norm_clip') == .123
        raise Reached
    learner.coordinated_altruistic_optimizer_step = step
    with pytest.raises(Reached):
        learner.coordinated_altruistic_update(timestep=2, timesteps=10)
