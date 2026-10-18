"""J fixed-TRAIN diagnostic only: python -m drl.IL.bc_v1.small_overfit.

No epochs, sampler, validation, HPO, upstream preparation or production trainer.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import resource
import tempfile
import time

import numpy as np
import torch

from .action_heads import load_action_head_config
from .batching import collate_policy_samples
from .dataset import _hash, file_hash
from .losses import compute_bc_loss, load_loss_config
from .metrics import BCMetrics, load_metrics_config
from .model import BCGNN, load_model_config
from .sidecar import SidecarDataset
from .sidecar_build import ROOT, load_sources
from .training import (CHECKPOINT_SCHEMA, build_optimizer, family_statistics,
                       load_checkpoint, move_batch_to_device, save_checkpoint, train_step,
                       verify_checkpoint_equivalence)

DEFAULT_CONFIG = ROOT/'configs/bc_v1/training_smoke_v1.json'
KNOWN_KEYS = (('expert:rc_car:seed-005100', 0), ('expert:gap:seed-004303', 299),
              ('expert:assembly:seed-006307', 0), ('expert:assembly:seed-006307', 444))


def select_micro_set(data, *, required_keys=KNOWN_KEYS):
    """Stable TRAIN-only diagnostic selection; targets never become inputs."""
    if data.split != 'train':
        raise ValueError('J selection requires TRAIN')
    keys = list(data.sample_keys())  # Logical keys only, not dataset payloads.
    available = set(keys)
    selected, selected_keys = [], set()
    for key in map(tuple, required_keys):
        if key in available and key not in selected_keys:
            selected.append(data.get(*key))
            selected_keys.add(key)
    if len(selected) > 8:
        raise ValueError('Too many required J samples')
    used_episodes = {key[0] for key in selected_keys}
    # Prefer one resource-bearing row per additional episode. Stable fallback
    # permits multiple rows only if episode diversity cannot complete the eight.
    for diverse in (True, False):
        for key in keys:
            if len(selected) == 8:
                return selected
            if key in selected_keys or (diverse and key[0] in used_episodes):
                continue
            sample = data.get(*key)
            y = sample['supervision']
            if not bool(y['resource']['supervised'].any()) or bool(y['structural']['supervised'].any()):
                continue
            selected.append(sample)
            selected_keys.add(key)
            used_episodes.add(key[0])
    if len(selected) != 8:
        raise ValueError('Cannot complete eight unique TRAIN samples')
    return selected


def per_sample_support(prediction, batch, loss_config, count):
    """Reuse I1 itself for audit counts, including singleton/parent mask rules."""
    x, y = batch['inputs'], batch['supervision']
    resource_graph = x['module_batch'][x['resource_to_module'][y['resource']['resource']]]
    structural_graph = x['module_batch'][y['structural']['group']]
    result = []
    for i in range(count):
        supervision = {
            'resource': {**y['resource'], 'supervised': y['resource']['supervised'] & (resource_graph == i)},
            'structural': {**y['structural'], 'supervised': y['structural']['supervised'] & (structural_graph == i)}}
        _, families = compute_bc_loss(prediction, {**batch, 'supervision': supervision}, loss_config)
        result.append({k: int(v['support']) for k, v in families.items()})
    return result


def selected_metrics(metrics):
    return {'resource_mode': metrics['resource']['mode']['accuracy'],
            'structural_top1': metrics['structural']['candidate_top1'],
            'resource_payload_nrmse': metrics['resource']['payload']['normalized_rmse'],
            'arg_numeric_nrmse': metrics['arguments']['numeric']['normalized_rmse'],
            'full_tuple': metrics['structural']['discrete_tuple_exact']}


def success_gates(initial, final):
    old, new = initial['selected_metrics'], final['selected_metrics']
    gates = {'loss': final['total_loss'] <= .25*initial['total_loss']}
    for key in ('resource_mode', 'structural_top1'):
        a, b = old[key], new[key]
        gates[key] = b['support'] == a['support'] and (
            not a['support'] or (b['value'] >= .90 and b['value'] > a['value']))
    gates['continuous'] = any(old[key]['support'] > 0 and new[key]['support'] == old[key]['support']
        and new[key]['value'] <= .5*old[key]['value'] for key in ('resource_payload_nrmse', 'arg_numeric_nrmse'))
    return {**gates, 'passed': all(gates.values())}


@torch.no_grad()
def evaluate(model, batch, loss_config, metrics, step):
    model.eval()
    prediction = model(batch['inputs'])
    total, families = compute_bc_loss(prediction, batch, loss_config)
    if not bool(torch.isfinite(total)):
        raise ValueError('Nonfinite diagnostic eval loss')
    metrics.reset()
    metrics.update(prediction, batch)
    values = metrics.compute()
    return {'step': step, 'total_loss': float(total), 'families': family_statistics(families),
            'selected_metrics': selected_metrics(values), 'metrics': values}, prediction


def _write_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(config_path=DEFAULT_CONFIG):
    config = json.loads(Path(config_path).read_text())
    # J is a fixed diagnostic, not an implicit hyperparameter-search interface.
    baseline = {'schema_version': 'mssr.bc_v1.training_smoke.v1', 'seed': 20261006,
        'device': 'cuda:0', 'dtype': 'float32', 'amp': False, 'workers': 0, 'optimizer': 'AdamW',
        'learning_rate': 3e-4, 'weight_decay': 1e-4, 'gradient_clip': 1., 'scheduler': None,
        'batch_size': 8, 'max_steps': 1500, 'eval_interval': 50}
    if any(config.get(k) != v for k, v in baseline.items()):
        raise ValueError('J configuration differs from approved fixed baseline')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 4070 Laptop GPU':
        raise ValueError('J requires cuda:0 on the RTX 4070 Laptop; CPU fallback forbidden')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    sources = load_sources()
    cfg = {name: loader(ROOT/('configs/bc_v1/'+path)) for name, loader, path in (
        ('h1', load_model_config, 'gnn_v1.json'), ('h2', load_action_head_config, 'action_heads_v1.json'),
        ('i1', load_loss_config, 'loss_v1.json'), ('i2', load_metrics_config, 'metrics_v1.json'))}
    audit, checkpoint_dir = (ROOT/config[k] for k in ('audit_dir', 'checkpoint_dir'))
    for directory, allowed in ((audit, ROOT/'artifacts/bc_v1/audits'),
                               (checkpoint_dir, ROOT/'artifacts/bc_v1/checkpoints')):
        if not directory.resolve().is_relative_to(allowed.resolve()) or directory.resolve() == allowed.resolve():
            raise ValueError('J output must be separate from immutable upstream artifacts')
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('J output already exists; refusing an implicit rerun/overwrite')
    with SidecarDataset(sources['paths']['output'], 'train', expected_identity=sources['identity']) as data:
        rows = select_micro_set(data, required_keys=config['required_keys'])
        identity = {**sources['identity'], 'sidecar_manifest_fingerprint': data.manifest['manifest_fingerprint'],
                    **{name+'_config_fingerprint': _hash(value) for name, value in cfg.items()},
                    'training_config_fingerprint': _hash(config)}
    # Exactly one collate; SQLite is closed before any optimizer step.
    batch = move_batch_to_device(collate_policy_samples(rows), torch.device('cuda:0'))
    kw = sources['kwargs']
    def new_model():
        return BCGNN(kw['spec'], kw['feature_schema'], cfg['h1'], action_head_config=cfg['h2']).to('cuda:0')
    model = new_model()
    optimizer = build_optimizer(model, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    metrics = BCMetrics(kw['spec'], kw['normalizers'], cfg['i2'])
    initial, prediction = evaluate(model, batch, cfg['i1'], metrics, 0)
    supports = per_sample_support(prediction, batch, cfg['i1'], len(rows))
    selection = []
    for row, support in zip(rows, supports):
        routing = row['routing']
        a = routing['audit']
        selection.append({'logical_episode_id': a.episode_id, 'canonical_transition_index': a.transition_index,
            'task': routing['task_vocabulary'][int(row['inputs']['task_one_hot'].argmax())],
            'morphology': a.morphology_stratum, 'provenance': a.provenance, 'split': 'train',
            'supported_loss_families': {k: v for k, v in support.items() if v},
            'resource_modes': [t.mode_id for t in routing['resource_targets'] if t.supervised],
            'structural_operations': [t.operation_id for t in routing['structural_targets'] if t.supervised]})
    if (len({(s['logical_episode_id'], s['canonical_transition_index']) for s in selection}) != 8
            or sum(bool(s['resource_modes']) for s in selection) < 2
            or sum(bool(s['structural_operations']) for s in selection) < 2
            or not initial['selected_metrics']['resource_payload_nrmse']['support']):
        raise ValueError('Incomplete diagnostic TRAIN selection')
    for directory in (audit, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(audit/'selection.json', {'identity': identity, 'config': config, 'samples': selection,
        'selection_policy': 'known TRAIN keys; then first resource-only row per additional TRAIN episode in stable logical order'})
    _write_json(audit/'initial_metrics.json', initial)
    del prediction, rows
    environment = {'device': 'cuda:0', 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'initial_vram_bytes': torch.cuda.memory_allocated(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark, 'float32_matmul_precision': torch.get_float32_matmul_precision()}
    grouping = [{'weight_decay': g['weight_decay'], 'tensors': len(g['params']),
                 'parameters': sum(p.numel() for p in g['params'])} for g in optimizer.param_groups]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    first, final, step, max_norm = None, initial, 0, 0.
    with (audit/'history.jsonl').open('x') as history:
        def record(value):
            compact = {k: value[k] for k in ('step', 'total_loss', 'families', 'selected_metrics')}
            compact['grad_norm_before_clip'] = max_norm if value['step'] else None
            history.write(json.dumps(compact, sort_keys=True, allow_nan=False)+'\n')
            history.flush()
        record(initial)
        try:
            for step in range(1, config['max_steps']+1):
                result = train_step(model, batch, optimizer, cfg['i1'], max_grad_norm=config['gradient_clip'], audit=step == 1)
                max_norm = max(max_norm, result['grad_norm_before_clip'])
                if step == 1:
                    first = result
                if step % config['eval_interval'] == 0:
                    final, _ = evaluate(model, batch, cfg['i1'], metrics, step)
                    record(final)
                    if success_gates(initial, final)['passed']:
                        break
        except (ValueError, RuntimeError) as exc:
            diagnostic = {'ready': False, 'step': step, 'error': repr(exc), 'first_step': first,
                'current_step_diagnostics': getattr(exc, 'diagnostics', None),
                'last_eval': final, 'nonfinite_gradient_parameters': [n for n, p in model.named_parameters()
                    if p.grad is not None and not bool(torch.isfinite(p.grad).all())]}
            _write_json(audit/'summary.json', diagnostic)
            raise
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    final, output = evaluate(model, batch, cfg['i1'], metrics, step)
    _write_json(audit/'final_metrics.json', final)
    del output
    gates = success_gates(initial, final)
    summary = {'ready': False, 'gates': gates, 'identity': identity, 'config': config, 'first_step': first,
        'initial_loss': initial['total_loss'], 'final_loss': final['total_loss'], 'steps': step,
        'optimizer_groups': grouping, 'max_grad_norm_before_clip': max_norm, 'environment': environment,
        'performance': {'training_wall_s': elapsed, 'mean_step_s': elapsed/step,
            'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'peak_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    checkpoint = checkpoint_dir/'checkpoint.pt'
    save_checkpoint(checkpoint, model, optimizer, step=step, identity=identity, config=config,
                    selection=selection, seed=config['seed'], summary=summary)
    saved_optimizer = optimizer.state_dict()
    # Check serialization independently of the deterministic forward comparison.
    saved_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    restored = new_model()
    fresh_optimizer = build_optimizer(restored, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    payload = load_checkpoint(checkpoint, restored, fresh_optimizer, expected_identity=identity)
    for key, value in restored.state_dict().items():
        if not torch.equal(value.cpu(), saved_state[key]):
            raise ValueError('Checkpoint state not exactly restored')
    verification = verify_checkpoint_equivalence(model, restored, batch, cfg['i1'], metrics)
    # State equality includes all AdamW moments/step, not merely group metadata.
    def equal_state(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
        return torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    if not equal_state(saved_optimizer, fresh_optimizer.state_dict()):
        raise ValueError('Checkpoint optimizer not exactly restored')
    summary['checkpoint_reload'] = {**verification, 'new_instance': True,
                                     'model_state_exact': True, 'optimizer_state_exact': True}
    del model, optimizer
    gc.collect()
    summary['ready'] = gates['passed']  # Full-suite gate is reported separately after this run.
    _write_json(checkpoint_dir/'manifest.json', {'schema_version': CHECKPOINT_SCHEMA, 'file': checkpoint.name,
        'sha256': file_hash(checkpoint), 'bytes': checkpoint.stat().st_size, 'step': payload['step'], 'identity': identity,
        'selection_fingerprint': _hash(selection), 'selection_sha256': file_hash(audit/'selection.json'),
        'training_config_fingerprint': _hash(config),
        'verification': {k: v for k, v in verification.items() if k != 'reference_metrics'}})
    _write_json(audit/'summary.json', summary)
    print(json.dumps({'gates': gates, 'steps': step, 'initial_loss': initial['total_loss'],
                      'final_loss': final['total_loss'], 'checkpoint_reload': summary['checkpoint_reload']}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run(args.config)
    return 0 if summary['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
