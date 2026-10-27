"""K1 feed-forward BC orchestration. No preparation, tuning or resume path."""
from dataclasses import dataclass
import math
import statistics
import time

import torch

from .batching import collate_policy_samples
from .losses import compute_bc_loss
from .sampling import evaluation_batches
from .sidecar import SidecarDataset
from .training import family_statistics, move_batch_to_device, train_step


def steps_per_epoch(samples, batch_size):
    if type(samples) is not int or type(batch_size) is not int or min(samples, batch_size) < 1:
        raise ValueError('Positive sample and batch counts required')
    return math.ceil(samples / batch_size)


def open_dataset(root, split, identity, *, factory=SidecarDataset):
    """Reject TEST before even constructing a loader or opening a database."""
    if split not in ('train', 'validation'):
        raise ValueError('K1 forbids TEST: only train/validation splits may be opened')
    return factory(root, split, expected_identity=identity)


def validation_batches(count, batch_size=64):
    return (tuple(indices) for indices in evaluation_batches(count, batch_size=batch_size, split_name='validation'))


class FamilyAccumulator:
    """Float64 sufficient statistics; support first, lambda weighting last."""
    def __init__(self, weights):
        self.weights = dict(weights)
        self.sums = {k: 0. for k in weights}
        self.supports = {k: 0 for k in weights}

    def add(self, families):
        if families.keys() != self.weights.keys():
            raise ValueError('Loss-family identity mismatch')
        for name, stats in families.items():
            if any(not math.isfinite(float(v)) for v in stats.values()):
                raise ValueError('Nonfinite family loss: '+name)
            support = stats['support']
            if type(support) is not int or support < 0:
                raise ValueError('Invalid loss support')
            self.sums[name] = math.fsum((self.sums[name], stats['sum']))
            self.supports[name] += support

    def compute(self):
        families = {}
        for name, weight in self.weights.items():
            support = self.supports[name]
            mean = self.sums[name] / support if support else 0.
            families[name] = {'sum': self.sums[name], 'support': support, 'mean': mean, 'weighted': weight * mean}
        total = math.fsum(s['weighted'] for s in families.values())
        if not math.isfinite(total):
            raise ValueError('Nonfinite aggregate loss')
        return {'total_loss': total, 'families': families}


@dataclass
class SelectionState:
    patience: int = 15
    min_delta: float = 1e-4
    best_epoch: int = -1
    best_val: float = math.inf
    counter: int = 0
    initial_val: float = math.inf
    last_epoch: int = -1

    def observe(self, epoch, val_total):
        if not math.isfinite(val_total):
            raise ValueError('Nonfinite VAL total')
        if epoch != self.last_epoch + 1:
            raise ValueError('Validation epochs must be consecutive from epoch zero')
        self.last_epoch = epoch
        if epoch == 0:
            self.initial_val = val_total
        improved = val_total < self.best_val - self.min_delta
        if improved:
            self.best_epoch, self.best_val, self.counter = epoch, val_total, 0
        else:
            self.counter += 1
        return improved

    @property
    def should_stop(self):
        return self.counter >= self.patience

    @property
    def trained_improvement(self):
        return self.best_epoch > 0 and self.best_val < self.initial_val - self.min_delta


def support_snapshot(value):
    """All I1/I2 supports, class counts and incomplete exclusions, recursively."""
    result = {}
    def visit(item, path):
        if isinstance(item, dict):
            for key, child in item.items():
                current = path + '/' + key
                if key == 'support' or key.endswith('_support') or key in ('class_support', 'classes_with_support', 'excluded_incomplete'):
                    result[current] = child
                else:
                    visit(child, current)
    visit(value, '')
    return result


def check_supports(expected, result):
    actual = support_snapshot(result)
    if expected != actual:
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
        raise ValueError('VAL support drift: '+repr(changed[:10]))


def ensure_finite_parameters(model):
    with torch.no_grad():
        flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        if not bool(torch.isfinite(flat).all()):
            raise ValueError('Nonfinite model parameters')


def _detached_cpu(value):
    """One bulk transfer of raw predictions for scalar-heavy offline consumers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return {k: _detached_cpu(v) for k, v in value.items()}


@torch.inference_mode()
def evaluate_validation(model, data, loss_config, metrics, *, device, batch_size=64,
                        collate=collate_policy_samples, max_batches=None,
                        epoch=None, log_interval=50, progress=None, loss_fn=None):
    if data.split != 'validation':
        raise ValueError('K1 evaluation requires validation; TEST forbidden')
    previous = model.training
    accumulator = FamilyAccumulator(loss_config['weights'])
    metrics.reset()
    start = time.perf_counter()
    samples = batches = 0
    total_batches = math.ceil(len(data)/batch_size)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    try:
        model.eval()
        for indices in validation_batches(len(data), batch_size):
            if max_batches is not None and batches >= max_batches:
                break  # Isolated smoke only. Official runner never supplies this.
            reader = getattr(data, 'get_for_collate', data.__getitem__)
            host_batch = collate([reader(i) for i in indices])
            batch = move_batch_to_device(host_batch, device)
            prediction = model(batch['inputs'])
            # I1/I2 use many small routed decisions and Python scalar checks.
            # On CUDA those checks serialize thousands of transfers per batch.
            # Keep the GPU forward unchanged; consume its float32 outputs once
            # on CPU with the original CPU targets and identical I1/I2 functions.
            if torch.device(device).type != 'cpu':
                prediction = _detached_cpu(prediction)
                batch = host_batch
            total, families = (loss_fn or compute_bc_loss)(prediction, batch, loss_config)
            if not bool(torch.isfinite(total)):
                raise ValueError('Nonfinite VAL loss')
            accumulator.add(family_statistics(families))
            metrics.update(prediction, batch)
            samples += len(indices)
            batches += 1
            if (epoch is not None or progress is not None) and (batches == 1 or batches % log_interval == 0 or batches == total_batches):
                elapsed = time.perf_counter()-start
                rate = samples/elapsed
                event = {'epoch': epoch, 'batch': batches, 'total_batches': total_batches,
                    'samples': samples, 'loss': accumulator.compute()['total_loss'],
                    'samples_per_second': rate,
                    'eta_seconds': max(0, total_batches-batches)*elapsed/batches}
                if progress is not None:
                    progress(event)
                print(f'VAL epoch {epoch} batch {batches}/{total_batches} loss {event["loss"]:.6f} '
                      f'samples/s {rate:.2f} ETA {event["eta_seconds"]/60:.1f} min', flush=True)
        return {**accumulator.compute(), 'metrics': metrics.compute(), 'samples': samples,
                'batches': batches, 'wall_seconds': time.perf_counter()-start}
    finally:
        model.train(previous)


def train_epoch(model, data, sampler, optimizer, loss_config, *, device, epoch,
                global_step=0, gradient_clip=1., log_interval=50, max_steps=None, progress=None,
                loss_fn=None, collate=None):
    if data.split != 'train':
        raise ValueError('K1 optimizer steps require TRAIN')
    accumulator = FamilyAccumulator(loss_config['weights'])
    start = time.perf_counter()
    durations, loader_times, norms, optimization_losses = [], [], [], []
    first_audit = None
    model.train()
    for step, indices in enumerate(sampler, 1):
        t = time.perf_counter()
        reader = getattr(data, 'get_for_collate', data.__getitem__)
        batch = move_batch_to_device((collate or collate_policy_samples)([reader(i) for i in indices]), device)
        loader_times.append(time.perf_counter()-t)
        t = time.perf_counter()
        result = train_step(model, batch, optimizer, loss_config, max_grad_norm=gradient_clip,
                            audit=global_step == 0 and step == 1, loss_fn=loss_fn)
        ensure_finite_parameters(model)
        if not result['optimizer_step_completed']:
            raise ValueError('Optimizer step incomplete')
        accumulator.add(result['families'])
        if not math.isfinite(result['total_loss']):
            raise ValueError('Nonfinite training optimization loss')
        optimization_losses.append(result['total_loss'])
        norms.append(result['grad_norm_before_clip'])
        if not math.isfinite(norms[-1]):
            raise ValueError('Nonfinite gradient norm')
        durations.append(time.perf_counter()-t)
        if step == 1 and global_step == 0:
            first_audit = result
        if progress and (step == 1 or step % log_interval == 0 or step == len(sampler)):
            progress(global_step+step, result['total_loss'])
        if step == 1 or step % log_interval == 0 or step == len(sampler):
            elapsed = time.perf_counter()-start
            print(f'epoch {epoch} step {step}/{len(sampler)} loss {result["total_loss"]:.6f} '
                  f'grad_norm {norms[-1]:.4f} lr {optimizer.param_groups[0]["lr"]:g} '
                  f'samples/s {step*sampler.batch_size/elapsed:.2f} ETA {(len(sampler)-step)*elapsed/step/60:.1f} min', flush=True)
        if max_steps is not None and step >= max_steps:
            break  # Isolated smoke only.
    elapsed = time.perf_counter()-start
    aggregate = accumulator.compute()
    return {**aggregate, 'global_support_total_loss': aggregate['total_loss'],
            'total_loss': statistics.mean(optimization_losses), 'steps': len(durations), 'global_step': global_step+len(durations),
            'wall_seconds': elapsed, 'mean_step_seconds': statistics.mean(durations),
            'median_step_seconds': statistics.median(durations), 'sidecar_collate_seconds': sum(loader_times),
            'samples_per_second': len(durations)*sampler.batch_size/elapsed,
            'mean_grad_norm_before_clip': statistics.mean(norms), 'max_grad_norm_before_clip': max(norms),
            'learning_rates': [g['lr'] for g in optimizer.param_groups], 'first_step_audit': first_audit}
