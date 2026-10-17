"""Minimal J training plumbing. No dataset, robot backend, scheduler or trainer."""
import os
import math
from pathlib import Path
import random
import tempfile
from contextlib import contextmanager

import numpy as np
import torch

from .losses import compute_bc_loss, LOSS_FUNCTIONS

CHECKPOINT_SCHEMA = 'mssr.bc_v1.training_checkpoint.v1'
HEADS = {
    'resource_mode': 'mode_scorer', 'resource_payload': 'payload_decoder',
    'structural_choice': 'structural_scorer', 'arg_category': 'argument_category_scorer',
    'arg_numeric': 'numeric_decoder', 'arg_reference': 'reference_scorer',
    'arg_reference_set': 'reference_set_scorer', 'arg_boolean': 'boolean_scorer',
    'scope': 'scope_scorer',
}


class TrainingStepError(ValueError):
    """Fail-fast step error with JSON-safe current-step diagnostic evidence."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        def safe(value):
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [safe(v) for v in value]
            return str(value) if isinstance(value, float) and not math.isfinite(value) else value
        self.diagnostics = safe(diagnostics)


def build_optimizer(model, *, lr=3e-4, weight_decay=1e-4):
    """AdamW: each trainable parameter once; biases and LayerNorm never decay."""
    exempt = {id(p) for module in model.modules() for name, p in module.named_parameters(recurse=False)
              if name == 'bias' or isinstance(module, torch.nn.LayerNorm)}
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if id(p) in exempt else decay).append(p)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                              {'params': no_decay, 'weight_decay': 0.}], lr=lr)


def move_batch_to_device(batch, device):
    """Move numerical inputs/targets; leave routing/audit outside model inputs."""
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        raise TypeError('Only numerical tensor trees may enter inputs/supervision')
    return {**batch, 'inputs': move(batch['inputs']), 'supervision': move(batch['supervision'])}


def family_statistics(families):
    return {k: {name: (int(value.detach()) if name == 'support' else float(value.detach()))
                for name, value in v.items()} for k, v in families.items()}


def _gradient_summary(named):
    values = [(name, p.grad) for name, p in named if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for _, g in values)
    norms = [(name, float(g.detach().norm())) for name, g in values]
    name, norm = max(norms, key=lambda x: x[1], default=(None, 0.))
    return {'finite': finite, 'nonzero': norm > 0., 'parameter': name, 'norm': norm}


def _step_failure(stage, error, model, result, prediction=None, batch=None):
    diagnostics = {**result, 'stage': stage, 'error': repr(error),
        'gradients': _gradient_summary(model.named_parameters()),
        'nonfinite_gradient_parameters': [n for n, p in model.named_parameters()
            if p.grad is not None and not bool(torch.isfinite(p.grad).all())]}
    if stage == 'loss' and prediction is not None:
        # Failure-only diagnosis reuses I1 family functions; no alternate loss.
        diagnostics['families'] = {}
        for name, function in LOSS_FUNCTIONS.items():
            try:
                diagnostics['families'][name] = family_statistics({name: function(prediction, batch)})[name]
            except Exception as exc:
                diagnostics['families'][name] = {'error': repr(exc)}
    return TrainingStepError(str(error), diagnostics)


def train_step(model, batch, optimizer, loss_config, *, max_grad_norm=1., audit=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        prediction = model(batch['inputs'])
    except (ValueError, RuntimeError) as exc:
        raise _step_failure('forward', exc, model, {}) from exc
    try:
        total, families = compute_bc_loss(prediction, batch, loss_config)
    except (ValueError, RuntimeError) as exc:
        raise _step_failure('loss', exc, model, {}, prediction, batch) from exc
    result = {'total_loss': float(total.detach()), 'families': family_statistics(families),
              'clip_threshold': max_grad_norm, 'optimizer_step_completed': False}
    if not bool(torch.isfinite(total)):
        raise _step_failure('loss', ValueError('Nonfinite loss'), model, result, prediction, batch)
    try:
        total.backward()
    except (ValueError, RuntimeError) as exc:
        raise _step_failure('backward', exc, model, result) from exc
    if audit:
        named = list(model.named_parameters())
        gradients = {'h1': _gradient_summary([(n, p) for n, p in named if not n.startswith('action_heads.')])}
        for family, stats in result['families'].items():
            if stats['support']:
                prefix = 'action_heads.'+HEADS[family]+'.'
                gradients[family] = _gradient_summary([(n, p) for n, p in named if n.startswith(prefix)])
        result['gradients'] = gradients
        if not all(s['finite'] and s['nonzero'] for s in gradients.values()):
            raise _step_failure('gradient', ValueError('Nonfinite or zero supported gradients'), model, result)
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    except RuntimeError as exc:
        raise _step_failure('gradient', exc, model, result) from exc
    result['grad_norm_before_clip'] = float(norm)
    if not bool(torch.isfinite(norm)) or float(norm) == 0.:
        raise _step_failure('gradient', ValueError('Nonfinite or zero global gradient'), model, result)
    before = {n: p.detach().clone() for n, p in model.named_parameters()} if audit else None
    optimizer.step()
    result['optimizer_step_completed'] = True
    if audit:
        deltas = {n: float((p.detach()-before[n]).abs().max()) for n, p in model.named_parameters()}
        def largest(items):
            n, value = max(items, key=lambda x: x[1], default=(None, 0.))
            return {'parameter': n, 'value': value}
        result['parameter_delta'] = {
            'largest': largest(deltas.items()),
            'h1': largest((n, v) for n, v in deltas.items() if not n.startswith('action_heads.')),
            'h2': largest((n, v) for n, v in deltas.items() if n.startswith('action_heads.'))}
        if any(result['parameter_delta'][part]['value'] == 0. for part in ('h1', 'h2')):
            raise ValueError(f'Optimizer did not change H1/H2 parameters: {result}')
    return result


def save_checkpoint(path, model, optimizer, *, step, identity, config, selection, seed, summary=None):
    """Atomic local checkpoint; an interrupted save never publishes a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        rng['cuda'] = torch.cuda.get_rng_state_all()
    payload = {'schema_version': CHECKPOINT_SCHEMA, 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'step': step, 'identity': dict(identity),
               'config': config, 'selection': selection, 'seed': seed, 'rng_state': rng,
               'summary': summary}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, model, optimizer, *, expected_identity):
    """Trusted local checkpoint only. Validate identity before loading any state."""
    # State loaders place parameters/moments on the destination device while
    # retaining CPU step counters for noncapturable AdamW and CPU RNG state.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema_version') != CHECKPOINT_SCHEMA:
        raise ValueError('Incompatible checkpoint schema')
    if payload.get('identity') != expected_identity:
        raise ValueError('Incompatible checkpoint identity')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    return payload


@contextmanager
def deterministic_verification():
    """Temporary strict deterministic algorithms, exclusively for verification."""
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _compare_checkpoint_outputs(before, after):
    """Compare the complete raw H2 tree, including routing tensors, strictly."""
    result = {'output_tensors': 0, 'bitwise_equal_count': 0, 'max_output_absolute_difference': 0.}
    def compare(a, b):
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError('Checkpoint output keys differ')
            for key in a:
                compare(a[key], b[key])
            return
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Checkpoint output shape/dtype differs')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Nonfinite checkpoint output')
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        result['output_tensors'] += 1
        result['bitwise_equal_count'] += int(torch.equal(a, b))
        if a.numel():
            result['max_output_absolute_difference'] = max(result['max_output_absolute_difference'],
                                                            float((a.double()-b.double()).abs().max()))
    compare(before, after)
    return {**result, 'shape_matches': True, 'dtype_matches': True, 'all_finite': True,
            'assert_close': 'PASS', 'rtol': 1e-6, 'atol': 1e-7}


def verify_checkpoint_equivalence(reference, reloaded, batch, loss_config, metrics=None):
    """Compare fresh deterministic evals; never optimize or alter saved state.

    Both model modes and the caller's deterministic settings survive failure.
    Supervision/routing are used by I1/I2 only, never by either model forward.
    """
    previous_modes = (reference.training, reloaded.training)
    try:
        with deterministic_verification(), torch.no_grad():
            reference.eval()
            reloaded.eval()
            before = reference(batch['inputs'])
            loss_before, _ = compute_bc_loss(before, batch, loss_config)
            after = reloaded(batch['inputs'])
            result = _compare_checkpoint_outputs(before, after)
            loss_after, _ = compute_bc_loss(after, batch, loss_config)
            torch.testing.assert_close(loss_before, loss_after, rtol=1e-6, atol=1e-7)
            delta = abs(float(loss_after)-float(loss_before))
            result.update(deterministic_checkpoint_roundtrip=True,
                reference_loss=float(loss_before), loss_after_reload=float(loss_after),
                loss_absolute_difference=delta,
                loss_relative_difference=delta/abs(float(loss_before)) if float(loss_before) else (0. if not delta else None))
            if metrics is not None:
                metrics.reset()
                metrics.update(before, batch)
                result['reference_metrics'] = metrics.compute()
            return result
    finally:
        reference.train(previous_modes[0])
        reloaded.train(previous_modes[1])
