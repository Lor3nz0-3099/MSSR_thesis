"""One fixed K1 baseline. Run with -u -m drl.IL.bc_v1.train_baseline.

--smoke is an isolated 2 TRAIN / 2 VAL preflight; it publishes no run/checkpoint.
No TEST, preparation, HPO, resume or automatic restart entry point exists.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback

# Required by the existing J temporary deterministic checkpoint verification.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from .action_heads import load_action_head_config
from .batching import collate_policy_samples
from .dataset import _hash, file_hash
from .losses import compute_bc_loss, load_loss_config
from .metrics import BCMetrics, load_metrics_config
from .model import BCGNN, load_model_config
from .sampling import JointActionBatchSampler
from .sidecar_build import ROOT, load_sources
from .small_overfit import _write_json, selected_metrics
from .trainer import (SelectionState, check_supports, ensure_finite_parameters, evaluate_validation,
                      open_dataset, steps_per_epoch, support_snapshot, train_epoch)
from .training import (_compare_checkpoint_outputs, build_optimizer, deterministic_verification,
                       load_checkpoint, move_batch_to_device, save_checkpoint)

DEFAULT_CONFIG = ROOT/'configs/bc_v1/training_baseline_v1.json'
SIDECAR_PIN = '7bb9f77df7eefe7c3b0246cea10383ec484274093514b59294ea7d752458052a'
J_PIN = '84d88d4332c65bd7f7c57f024668d4eb021647272519884505d1df2768e65e47'
BASELINE = {
    'schema_version': 'mssr.bc_v1.training_baseline.v1', 'seed': 20261006,
    'device': 'cuda:0', 'dtype': 'float32', 'amp': False, 'compile': False,
    'deterministic_algorithms': False, 'workers': 0, 'cpu_threads': 1,
    'optimizer': 'AdamW', 'learning_rate': 3e-4, 'betas': [.9, .999], 'eps': 1e-8,
    'weight_decay': 1e-4, 'bias_weight_decay': 0., 'layernorm_weight_decay': 0.,
    'gradient_clip': 1., 'scheduler': None, 'gradient_accumulation': 1,
    'batch_size': 64, 'general_quota': 56, 'structural_quota': 8, 'steps_per_epoch': 1224,
    'max_epochs': 100, 'early_stopping_patience': 15, 'min_delta': 1e-4,
    'validation_cadence': 1, 'logging_cadence': 50,
    'selection_metric': 'validation.global_support_normalized_i1_total', 'dropout': .1,
    'metrics_breakdowns': ['task', 'morphology', 'provenance', 'task_stratum'],
    'audit_dir': 'artifacts/bc_v1/audits/k1_baseline_v1',
    'checkpoint_dir': 'artifacts/bc_v1/checkpoints/k1_baseline_v1',
    'run_dir': 'artifacts/bc_v1/runs/k1_baseline_v1'}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    if cfg != BASELINE:
        raise ValueError('K1 configuration differs from approved fixed baseline')
    return cfg


class RunStatus:
    STATES = {'STARTING', 'RUNNING', 'EARLY_STOPPED', 'MAX_EPOCHS_REACHED', 'FAILED', 'COMPLETED'}

    def __init__(self, path, *, device):
        self.path = Path(path)
        self.value = {'pid': os.getpid(), 'epoch': 0, 'global_step': 0, 'best_epoch': None,
                      'best_val_loss': None, 'patience_counter': 0, 'last_train_loss': None,
                      'last_val_loss': None, 'cuda_device': device, 'stop_reason': None}

    def update(self, state, **fields):
        if state not in self.STATES:
            raise ValueError('Invalid K1 run state')
        self.value.update(fields, run_state=state, last_update_timestamp=timestamp())
        _write_json(self.path, self.value)


def memory_stats():
    return {'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved': torch.cuda.max_memory_reserved(),
            'peak_rss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def _cpu_tree(value):
    return {k: _cpu_tree(v) for k, v in value.items()} if isinstance(value, dict) else value.detach().cpu().clone()


@torch.no_grad()
def capture_reference(model, val_data, device):
    batch = move_batch_to_device(collate_policy_samples([val_data[i] for i in range(min(64, len(val_data)))]), device)
    previous = model.training
    try:
        model.eval()
        with deterministic_verification():
            return _cpu_tree(model(batch['inputs']))
    finally:
        model.train(previous)


def fit(model, optimizer, *, train_data, val_data, sampler, cfg, components, audit, checkpoints,
        identity_check, status=None):
    """Epoch-zero reference, then consecutive full epochs; one continuous sampler."""
    selection = SelectionState(cfg['early_stopping_patience'], cfg['min_delta'])
    global_step = 0
    expected_supports = None
    reference = initial = best = None
    run_start = time.perf_counter()
    with (audit/'history.jsonl').open('x') as history:
        for epoch in range(cfg['max_epochs']+1):
            identity_check()
            train = None
            if status:
                status.update('RUNNING', epoch=epoch, global_step=global_step,
                              phase='TRAINING' if epoch else 'EPOCH0_VALIDATION')
            if epoch:
                train = train_epoch(model, train_data, sampler, optimizer, components['i1'],
                    device=cfg['device'], epoch=epoch, global_step=global_step,
                    gradient_clip=cfg['gradient_clip'], log_interval=cfg['logging_cadence'],
                    progress=(lambda step, loss: status.update('RUNNING', global_step=step,
                              last_train_loss=loss, phase='TRAINING')) if status else None)
                if train['steps'] != cfg['steps_per_epoch']:
                    raise ValueError('Incomplete training epoch')
                if epoch == 1 and not train['first_step_audit']['optimizer_step_completed']:
                    raise ValueError('Epoch-one optimizer safety gate failed')
                global_step = train['global_step']
            if status:
                status.update('RUNNING', global_step=global_step, phase='VALIDATION' if epoch else 'EPOCH0_VALIDATION')
            val = evaluate_validation(model, val_data, components['i1'], components['metrics'],
                                      device=cfg['device'], batch_size=cfg['batch_size'])
            if val['samples'] != 22762 or val['batches'] != 356:
                raise ValueError('Incomplete full VAL coverage')
            if expected_supports is None:
                initial = val
                expected_supports = support_snapshot(val)
                _write_json(audit/'epoch0_val.json', {'epoch': 0, **val, 'supports': expected_supports})
            else:
                check_supports(expected_supports, val)
            identity_check()
            improved = selection.observe(epoch, val['total_loss'])
            summary = {'epoch': epoch, 'global_step': global_step, 'best_epoch': selection.best_epoch,
                       'best_val_loss': selection.best_val, 'patience_counter': selection.counter}
            if improved:
                reference = capture_reference(model, val_data, cfg['device'])
                save_checkpoint(checkpoints/'best_checkpoint.pt', model, optimizer, step=global_step,
                    identity=components['identity'], config=cfg, selection={'metric': cfg['selection_metric']},
                    seed=cfg['seed'], summary=summary)
                best = {'epoch': epoch, **val}
                _write_json(audit/'best_val_metrics.json', best)
            save_checkpoint(checkpoints/'last_checkpoint.pt', model, optimizer, step=global_step,
                identity=components['identity'], config=cfg, selection={'metric': cfg['selection_metric']},
                seed=cfg['seed'], summary=summary)
            _write_json(checkpoints/'manifest.json', {'schema_version': 'mssr.bc_v1.k1_checkpoints.v1',
                **summary, 'identity': components['identity'], 'best_file': 'best_checkpoint.pt',
                'last_file': 'last_checkpoint.pt', 'atomic_writes': True})
            selected = selected_metrics(val['metrics']) if val['metrics'] else {}
            health = memory_stats()
            row = {**summary, 'timestamp': timestamp(), 'train': train, 'validation': val,
                'train_total_loss': train['total_loss'] if train else None,
                'train_family_losses': train['families'] if train else {},
                'val_total_loss': val['total_loss'], 'val_family_losses': val['families'],
                'selected_metrics': selected, 'best_improved': improved,
                'resource_mode_accuracy': selected.get('resource_mode'),
                'structural_top1': selected.get('structural_top1'),
                'continuous_nRMSE': selected.get('resource_payload_nrmse'),
                'full_discrete_tuple': selected.get('full_tuple'),
                'train_wall_s': train['wall_seconds'] if train else 0., 'val_wall_s': val['wall_seconds'],
                'mean_step_s': train['mean_step_seconds'] if train else None,
                'samples_per_s': train['samples_per_second'] if train else None, **health}
            history.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            history.flush()
            os.fsync(history.fileno())
            # One compact line per epoch, independently searchable by external tools.
            print('K1_EPOCH_SUMMARY '+json.dumps({**summary, 'train_loss': row['train_total_loss'],
                'val_loss': val['total_loss'], 'resource_acc': row['resource_mode_accuracy'],
                'structural_top1': row['structural_top1'], 'continuous_nRMSE': row['continuous_nRMSE'],
                'full_tuple': row['full_discrete_tuple'], 'epoch_wall': row['train_wall_s']+row['val_wall_s'],
                'rare_ops': val['metrics'].get('structural', {}).get('operation_recall', {})}, sort_keys=True), flush=True)
            if status:
                status.update('RUNNING', **summary, last_train_loss=row['train_total_loss'],
                              last_val_loss=val['total_loss'], phase='EPOCH_COMPLETE')
            if selection.should_stop:
                break
    return {'selection': selection, 'epoch': epoch, 'global_step': global_step,
            'initial': initial, 'best': best, 'final': val, 'reference': reference,
            'wall_seconds': time.perf_counter()-run_start}


def frozen_snapshot(sources):
    """No payload scan: immutable manifests/config hashes + file stat identities."""
    roots = [ROOT/'datasets/expert_v1_canonical_v6', sources['paths']['g1'],
             sources['paths']['normalizers'].parent, sources['paths']['output'],
             ROOT/'artifacts/bc_v1/splits/f1_v1', ROOT/'artifacts/bc_v1/checkpoints/j_small_overfit_v1',
             ROOT/'artifacts/bc_v1/audits/p1_h2_optimization_v1',
             ROOT/'artifacts/bc_v1/audits/p1b_h2_final_optimization_v1']
    stats = {}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                st = path.stat()
                stats[str(path.relative_to(ROOT))] = [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    files = [p for base in ('drl/IL/bc_v1', 'configs/bc_v1') for p in (ROOT/base).glob('*') if p.suffix in ('.py', '.json')]
    files += [sources['paths']['canonical_manifest'], sources['paths']['split'],
              sources['paths']['normalizers'], sources['paths']['g1']/'manifest.json',
              sources['paths']['output']/'manifest.json',
              ROOT/'artifacts/bc_v1/checkpoints/j_small_overfit_v1/checkpoint.pt']
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in files}
    if hashes['artifacts/bc_v1/checkpoints/j_small_overfit_v1/checkpoint.pt'] != J_PIN:
        raise ValueError('Frozen J checkpoint mismatch')
    return {'sha256': hashes, 'file_stats': stats}


def assert_metric_consistency(expected, actual, path='validation'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError('Reload metric keys mismatch: '+path)
        for key in expected:
            assert_metric_consistency(expected[key], actual[key], path+'/'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError('Reload metric shape mismatch: '+path)
        for i, (a, b) in enumerate(zip(expected, actual)):
            assert_metric_consistency(a, b, path+'/'+str(i))
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError('Reload metric numeric mismatch: '+path)
    elif expected != actual:
        raise ValueError('Reload metric mismatch: '+path)


def initialize(cfg):
    if (not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 4070 Laptop GPU'
            or torch.__version__ != '2.14.1+cu130' or torch.version.cuda != '13.0'):
        raise ValueError('K1 requires torch 2.14.1+cu130 / CUDA 13.0 / cuda:0 RTX 4070 Laptop; no CPU fallback')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    # Branch is audit provenance; scientific identities remain checked below.
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(False)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('K1 requires float32')
    sources = load_sources()
    config = {key: loader(ROOT/('configs/bc_v1/'+filename)) for key, loader, filename in (
        ('h1', load_model_config, 'gnn_v1.json'), ('h2', load_action_head_config, 'action_heads_v1.json'),
        ('i1', load_loss_config, 'loss_v1.json'), ('i2', load_metrics_config, 'metrics_v1.json'))}
    if config['h1']['dropout'] != cfg['dropout']:
        raise ValueError('Frozen dropout mismatch')
    config['i2'] = {**config['i2'], 'breakdowns': cfg['metrics_breakdowns']}
    train_data = open_dataset(sources['paths']['output'], 'train', sources['identity'])
    try:
        val_data = open_dataset(sources['paths']['output'], 'validation', sources['identity'])
        if (len(train_data) != 78302 or len(train_data.episode_ids) != 49
                or len(val_data) != 22762 or len(val_data.episode_ids) != 12
                or train_data.manifest['manifest_fingerprint'] != SIDECAR_PIN
                or train_data.manifest['complete'] is not True):
            raise ValueError('Frozen dataset/sidecar counts or fingerprint mismatch')
        identity = {**sources['identity'], 'sidecar_manifest_fingerprint': SIDECAR_PIN,
                    **{key+'_config_fingerprint': _hash(value) for key, value in config.items()},
                    'training_config_fingerprint': _hash(cfg)}
        index = train_data.train_index()
        if not index.structural_rows:
            raise ValueError('TRAIN lacks guaranteed structural samples')
        sampler = JointActionBatchSampler(index, batch_size=64, ordinary_quota=56, structural_quota=8, seed=cfg['seed'])
        if len(sampler) != steps_per_epoch(len(train_data), 64) or len(sampler) != cfg['steps_per_epoch']:
            raise ValueError('Incorrect epoch length')
        kw = sources['kwargs']
        def new_model():
            return BCGNN(kw['spec'], kw['feature_schema'], config['h1'], action_head_config=config['h2']).to('cuda:0')
        model = new_model()
        optimizer = build_optimizer(model, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        if any(g['betas'] != tuple(cfg['betas']) or g['eps'] != cfg['eps'] for g in optimizer.param_groups):
            raise ValueError('AdamW baseline mismatch')
        config.update(identity=identity, metrics=BCMetrics(kw['spec'], kw['normalizers'], config['i2']))
        return sources, config, model, optimizer, sampler, train_data, val_data, new_model, branch
    except BaseException:
        train_data.close()
        if 'val_data' in locals(): val_data.close()
        raise


def run(config_path=DEFAULT_CONFIG, *, smoke=False):
    cfg = load_config(config_path)
    status = None
    train_data = val_data = None
    audit = ROOT/cfg['audit_dir']
    checkpoints = ROOT/cfg['checkpoint_dir']
    run_dir = ROOT/cfg['run_dir']
    # Never overwrite scientific output or silently resume an interrupted run.
    if not smoke:
        for directory in (audit, checkpoints):
            if directory.exists() and any(directory.iterdir()):
                raise ValueError('K1 output already exists; rerun/resume forbidden: '+str(directory))
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir/'status.json').exists():
            raise ValueError('K1 runtime status already exists; refusing another run')
        status = RunStatus(run_dir/'status.json', device=cfg['device'])
        status.update('STARTING', phase='PREFLIGHT')
        (run_dir/'training.pid').write_text(str(os.getpid())+'\n')
    start = time.perf_counter()
    try:
        sources, components, model, optimizer, sampler, train_data, val_data, new_model, branch = initialize(cfg)
        frozen = frozen_snapshot(sources)
        def check():
            if not torch.cuda.is_available():
                raise ValueError('CUDA unavailable')
            if frozen_snapshot(sources) != frozen:
                raise ValueError('Frozen artifact / production configuration identity mismatch')
        parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
        free = shutil.disk_usage(ROOT).free
        if free < max(2*1024**3, parameter_bytes*24):
            raise ValueError('Insufficient free disk for best/last checkpoints and audits')
        environment = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'device': 'cuda:0', 'gpu': torch.cuda.get_device_name(0), 'branch': branch,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cpu_threads': torch.get_num_threads(), 'disk_free_bytes': free,
            'estimated_checkpoint_bytes': parameter_bytes*3, 'audit_budget_bytes': 100*1024**2}
        print('K1_STARTUP '+json.dumps({'pid': os.getpid(), **environment, 'smoke': smoke,
            'test_accessed': False, 'fresh_model': True, 'j_weights_used': False}), flush=True)
        torch.cuda.reset_peak_memory_stats()
        if smoke:
            train = train_epoch(model, train_data, sampler, optimizer, components['i1'], device='cuda:0',
                epoch=1, max_steps=2, log_interval=50)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            val = evaluate_validation(model, val_data, components['i1'], components['metrics'],
                                      device='cuda:0', max_batches=2)
            unchanged = all(torch.equal(before[n], p) for n, p in model.named_parameters())
            if not unchanged or val['samples'] != 128 or train['steps'] != 2:
                raise ValueError('Smoke parameter/coverage gate failed')
            check()
            result = {'smoke_passed': True, 'train': train, 'validation': val, 'validation_parameters_unchanged': unchanged,
                      'test_accessed': False, 'fresh_model_discarded': True, **environment, **memory_stats()}
            print('K1_SMOKE_RESULT '+json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
            return result
        for directory in (audit, checkpoints): directory.mkdir(parents=True, exist_ok=False)
        _write_json(audit/'config_resolved.json', {'training': cfg, **{k: components[k] for k in ('h1', 'h2', 'i1', 'i2')}})
        _write_json(audit/'run_manifest.json', {'schema_version': 'mssr.bc_v1.k1_run.v1', 'timestamp': timestamp(),
            'pid': os.getpid(), 'identity': components['identity'], 'environment': environment,
            'frozen_snapshot': frozen, 'seed': cfg['seed'], 'test_accessed': False,
            'train': {'episodes': 49, 'samples': 78302}, 'validation': {'episodes': 12, 'samples': 22762},
            'optimization_distribution': '56 general + 8 structural guaranteed; replacement; episode then row uniform',
            'validation_distribution': 'natural sequential; no replacement; no dropped tail',
            'epoch_convention': '1224 optimizer steps, not exact sample coverage', 'fresh_initialization': True})
        status.update('RUNNING', phase='EPOCH0_VALIDATION', cuda_available=True, gpu=environment['gpu'])
        state = fit(model, optimizer, train_data=train_data, val_data=val_data, sampler=sampler, cfg=cfg,
                    components=components, audit=audit, checkpoints=checkpoints, identity_check=check, status=status)
        selection = state['selection']
        stopped = selection.should_stop
        status.update('EARLY_STOPPED' if stopped else 'MAX_EPOCHS_REACHED', phase='BEST_CHECKPOINT_VERIFICATION',
                      stop_reason='patience exhausted' if stopped else 'maximum epochs reached')
        # Fresh model instance; do not restore RNG or attempt exact continuation.
        del model, optimizer
        reloaded = new_model()
        restored_optimizer = build_optimizer(reloaded, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        payload = load_checkpoint(checkpoints/'best_checkpoint.pt', reloaded, restored_optimizer,
                                  expected_identity=components['identity'])
        if payload['summary']['epoch'] != selection.best_epoch:
            raise ValueError('Best checkpoint epoch mismatch')
        output = capture_reference(reloaded, val_data, cfg['device'])
        equivalence = _compare_checkpoint_outputs(state['reference'], output)
        if (equivalence['output_tensors'] != 35 or equivalence['bitwise_equal_count'] != 35
                or equivalence['max_output_absolute_difference'] != 0.):
            raise ValueError('Deterministic best checkpoint output roundtrip failed')
        recomputed = evaluate_validation(reloaded, val_data, components['i1'], components['metrics'], device='cuda:0')
        check_supports(support_snapshot(state['best']), {'epoch': selection.best_epoch, **recomputed})
        for key in ('total_loss', 'families', 'metrics', 'samples', 'batches'):
            assert_metric_consistency(state['best'][key], recomputed[key], key)
        check()
        result = {'schema_version': 'mssr.bc_v1.k1_summary.v1', 'ready': selection.trained_improvement,
            'stop_epoch': state['epoch'], 'global_step': state['global_step'], 'selection': asdict(selection),
            'stopped_early': stopped, 'initial': state['initial'], 'best': state['best'], 'final': state['final'],
            'reloaded_validation': recomputed, 'checkpoint_verification': {**equivalence,
                'identity_validation': 'PASS', 'i2_consistency': 'PASS',
                'i1_absolute_difference': abs(recomputed['total_loss']-state['best']['total_loss'])},
            'best_checkpoint_sha256': file_hash(checkpoints/'best_checkpoint.pt'),
            'last_checkpoint_sha256': file_hash(checkpoints/'last_checkpoint.pt'),
            'test_accessed': False, 'frozen_artifacts_unchanged': True,
            'wall_seconds': time.perf_counter()-start, **memory_stats(),
            'interpretation': 'Offline BC validation metrics do not establish closed-loop Isaac task success.'}
        _write_json(audit/'final_summary.json', result)
        status.update('COMPLETED', phase='COMPLETE', ready=result['ready'])
        print('K1 FULL BC BASELINE READY FOR REVIEW: '+('YES' if result['ready'] else 'NO'), flush=True)
        return result
    except BaseException as exc:
        failure = {'ready': False, 'timestamp': timestamp(), 'error': repr(exc), 'traceback': traceback.format_exc(),
                   'step_diagnostics': getattr(exc, 'diagnostics', None)}
        # Independent best-effort reports: a disk/diagnostic failure must never
        # suppress the original exception or prevent another status write.
        print(failure['traceback'], file=sys.stderr, flush=True)
        if not smoke:
            reports = [lambda: status.update('FAILED', stop_reason=repr(exc), phase='FAILED'),
                       lambda: _write_json(run_dir/'failure.json', failure)]
            if audit.is_dir():
                reports.append(lambda: _write_json(audit/'final_summary.json', failure))
            for report in reports:
                try:
                    report()
                except Exception as reporting_error:
                    print('K1_DIAGNOSTIC_WRITE_FAILED '+repr(reporting_error), file=sys.stderr, flush=True)
        raise
    finally:
        if train_data is not None: train_data.close()
        if val_data is not None: val_data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, smoke=args.smoke)


if __name__ == '__main__':
    main()
