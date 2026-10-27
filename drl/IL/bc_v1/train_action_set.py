"""Explicit future scratch ActionSet runner; default is static DRY_RUN only."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from .action_set import ActionBundle, ActionInstance
from .action_set_head import load_action_set_head_config
from .action_set_loss import compute_action_set_loss, _candidate_graph
from .action_set_metrics import ActionSetMetrics
from .action_set_targets import collate_action_set_samples, target_bundle
from .action_set_training import (ROOT, _source, _model_identity, build_scratch_actor,
    load_action_set_training_config, open_dataset, save_action_set_checkpoint, selection_state)
from .bc_v1_contract import robot_spec_fingerprint, load_robot_spec
from .feature_schema import feature_schema_fingerprint, load_feature_schema
from .dataset import file_hash
from .losses import _argument_rows
from .model import load_model_config
from .normalization import (normalizer_fingerprint, read_normalizers,
    load_normalization_policy, policy_fingerprint)
from .sampling import JointActionBatchSampler
from .sidecar import _read_manifest, _hash
from .sidecar_build import load_sources
from .trainer import FamilyAccumulator, ensure_finite_parameters, validation_batches
from .training import family_statistics, move_batch_to_device

DEFAULT_CONFIG = ROOT/'configs/bc_v1/training_action_set_v1.json'
DEFAULT_G2 = ROOT/'artifacts/bc_v1/sidecar/g2_v3_resource_contract'
# Proven v3 operating settings; these do not change the scientific model/config.
CACHE_BUDGETS = {'train_dynamic': 2*1024**3, 'train_template': 4*1024**3,
                 'validation_dynamic': 256*1024**2, 'validation_template': 512*1024**2}
RUNTIME = {'cpu_threads': 1, 'cpu_interop_threads': 1, 'numeric_mode': 'FP32 eager',
           'tf32': False, 'amp': False, 'compile': False, 'prefetch': False,
           'sidecar_template_cache_entries': 0}
FAMILIES = ('presence', 'resource_payload', 'arg_category', 'arg_numeric', 'arg_reference',
            'arg_boolean', 'arg_reference_set', 'scope')


def train_batch(model, batch, optimizer, max_grad_norm=1.):
    """One ordinary finite FP32 forward/backward/clip/optimizer step."""
    if type(max_grad_norm) not in (int, float) or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError('Positive finite gradient clip required')
    model.train(); optimizer.zero_grad(set_to_none=True)
    prediction = model(batch['inputs'])
    total, families = compute_action_set_loss(prediction, batch)
    if not torch.isfinite(total): raise ValueError('Nonfinite ActionSet loss')
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step(); ensure_finite_parameters(model)
    return {'total_loss': float(total.detach()), 'families': family_statistics(families),
            'grad_norm': float(norm), 'prediction': prediction}


@torch.no_grad()
def measure_batch(prediction, batch, metrics=None, threshold=0.):
    """Measure raw membership before physical feasibility or claim validation."""
    if type(threshold) not in (float, int) or not math.isfinite(threshold):
        raise ValueError('Finite presence threshold required')
    metrics = ActionSetMetrics() if metrics is None else metrics
    graph, count = _candidate_graph(batch['inputs'])
    logits = prediction['presence_logits'].detach().cpu()
    if (logits.ndim != 1 or not logits.is_floating_point() or logits.shape != graph.shape
            or not torch.isfinite(logits).all()
            or not torch.equal(prediction['candidate_graph'].detach().cpu(), graph)):
        raise ValueError('Invalid raw ActionSet metric prediction')
    routes = batch['routing'].get('samples', (batch['routing'],))
    if len(routes) != count: raise ValueError('Metric routing graph count mismatch')
    selected = logits >= threshold
    resource_count = len(batch['inputs']['resource_candidates']['resource'])
    roffset = soffset = 0
    for route in routes:
        actions = []
        for row, candidate in enumerate(route['resource_candidates']):
            if selected[roffset+row]:
                actions.append(ActionInstance(candidate.action.action_id, candidate.resource_ref.module,
                                               resource=candidate.resource_ref))
        for row, (action, binding) in enumerate(route['structural_bindings']):
            if selected[resource_count+soffset+row]:
                actions.append(ActionInstance(action.action_id, binding.initiator, connectors=binding.connectors))
        metrics.update(target_bundle({'routing': route}), ActionBundle(tuple(actions)), numeric=False)
        roffset += len(route['resource_candidates']); soffset += len(route['structural_bindings'])
    p = prediction['parameters']; y = batch['supervision']
    numeric = p['resource']; routes = {int(q): row for row, q in enumerate(numeric['payload_candidate'].tolist())}
    for row, candidate in enumerate(y['resource']['candidate'].tolist()):
        if candidate not in routes or not selected[candidate] or not y['resource']['supervised'][row]: continue
        a, b = y['resource']['payload_ptr'][row:row+2].tolist()
        lo, hi = numeric['payload_ptr'][routes[candidate]:routes[candidate]+2].tolist()
        valid = y['resource']['payload_valid'][a:b]
        metrics.update_numeric('payload', numeric['payload'][lo:hi][valid], y['resource']['payload'][a:b][valid], '1', normalized=True)
    numeric = p['structural']['arguments']['numeric']
    routes = {int(q): row for row, q in enumerate(numeric['argument'].tolist())}
    for row, argument in _argument_rows(batch, ('scalar', 'vector')):
        binding = int(batch['inputs']['structural']['argument_binding'][argument])
        if not selected[resource_count+binding]: continue
        a, b = y['structural']['argument_value_ptr'][row:row+2].tolist()
        lo, hi = numeric['ptr'][routes[argument]:routes[argument]+2].tolist()
        valid = y['structural']['argument_value_valid'][a:b]
        metrics.update_numeric('arguments', numeric['value'][lo:hi][valid], y['structural']['argument_value'][a:b][valid], '1', normalized=True)
    return metrics


def _output_path(output, g2):
    output, g2 = Path(output).resolve(), Path(g2).resolve()
    if output == g2 or output in g2.parents or g2 in output.parents:
        raise ValueError('Output and frozen G2 source paths overlap')
    if output.exists(): raise ValueError('Output exists; no overwrite or resume')
    return output


def _actor_sources(g2, manifest, normalizer_dir):
    """Authenticate explicit actor F2 metadata, without legacy data or samples."""
    from .actor_visibility import VisibilityPolicy
    if normalizer_dir is None:
        raise ValueError('Actor-visible source requires explicit --normalizers directory')
    directory = Path(normalizer_dir).resolve()
    try:
        actor = manifest['actor_visibility']
        policy = VisibilityPolicy.from_dict(actor['policy'])
        context = json.loads((directory/'fit_context.json').read_text())
        success = json.loads((g2/'SUCCESS.json').read_text())
        sha = file_hash(directory/'normalizers.json')
        counts = {s: sum(e['samples'] for e in manifest['inventory'][s]) for s in ('train', 'validation')}
        if (actor['schema_version'] != 'mssr.bc_v1.actor_inputs.v1'
                or actor['fingerprint'] != policy.fingerprint
                or context['actor_visibility'] != policy.document
                or context['actor_input_schema'] != actor['schema_version']
                or context['actor_visibility_fingerprint'] != policy.fingerprint
                or context['TRAIN_only'] is not True or context['TEST_payload_access'] != 0
                or success['manifest_fingerprint'] != manifest['manifest_fingerprint']
                or success['fingerprint'] != policy.fingerprint
                or success['TEST_payload_access'] != 0
                or success['normalizer_context'] != context
                or success['sample_counts'] != counts or actor['sample_counts'] != counts
                or success['verification']['integrity'] != 'PASS'
                or success['verification']['manifest_fingerprint'] != manifest['manifest_fingerprint']
                or success['verification']['TEST_payload_access'] != 0
                or context['normalizer_sha256'] != sha
                or manifest['identity']['normalizer_sha'] != sha):
            raise ValueError('Actor-visible SUCCESS/F2/visibility identity mismatch')
        spec = load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
        schema = load_feature_schema(ROOT/'configs/bc_v1/features.json')
        normalization = load_normalization_policy(ROOT/'configs/bc_v1/normalization.json', schema)
        train = sorted(manifest['inventory']['train'], key=lambda e: e['episode_id'])
        expected = dict(split_fingerprint=manifest['identity']['split_fingerprint'],
            dataset_manifest_sha256=manifest['identity']['canonical_sha'],
            train_membership_fingerprint=_hash([e['episode_id'] for e in train]),
            train_episode_ids=[e['episode_id'] for e in train],
            sample_counts=[[e['episode_id'], e['samples']] for e in train],
            robot_spec_fingerprint=robot_spec_fingerprint(spec),
            feature_schema_fingerprint=feature_schema_fingerprint(schema),
            policy_fingerprint=policy_fingerprint(normalization, schema), complete_train_coverage=True,
            actor_input_schema=actor['schema_version'], actor_visibility_fingerprint=policy.fingerprint)
        stats = read_normalizers(directory/'normalizers.json', expected_metadata=expected)
        if stats.metadata != context['expected_metadata'] or normalizer_fingerprint(stats) != context['normalizer_fingerprint']:
            raise ValueError('Actor-visible normalizer metadata mismatch')
        return {'kwargs': dict(spec=spec, feature_schema=schema, normalizers=stats),
                'identity': {'normalizer_sha': sha}, 'normalizer_directory': str(directory),
                'actor_visibility': dict(schema_version=actor['schema_version'],
                    fingerprint=policy.fingerprint, policy=policy.document)}
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('Missing or malformed actor-visible source certification') from exc


def preflight(g2, config, *, normalizer_dir=None):
    """Authenticate metadata and actual F2 before any dataset/model factory."""
    g2 = Path(g2).resolve(); manifest = json.loads((g2/'manifest.json').read_text())
    _source(manifest)  # TEST is rejected from metadata before dataset access.
    manifest = _read_manifest(g2, manifest['identity'])
    if 'actor_visibility' in manifest:
        sources = _actor_sources(g2, manifest, normalizer_dir)
    else:
        if normalizer_dir is not None:
            raise ValueError('Explicit actor normalizers require an actor-visible source')
        sources = load_sources()
    kw = sources['kwargs']
    h1 = load_model_config(ROOT/'configs/bc_v1/gnn_v1.json')
    h2 = load_action_set_head_config(ROOT/'configs/bc_v1/action_set_head_v1.json')
    if h1['dropout'] != .1 or h2['dropout'] != .1:
        raise ValueError('Production ActionSet training requires H1/H2 dropout .1')
    expected = {'f2_fingerprint': normalizer_fingerprint(kw['normalizers']),
        'normalizer_sha': sources['identity']['normalizer_sha'],
        'robot_spec_fingerprint': robot_spec_fingerprint(kw['spec']),
        'feature_schema_fingerprint': feature_schema_fingerprint(kw['feature_schema'])}
    if any(manifest['identity'].get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen preprocessing identity mismatch')
    identity = _model_identity(kw['spec'], kw['feature_schema'], h1, h2, manifest['identity'], manifest['manifest_fingerprint'], config)
    rare = config['batch_size']//4
    report = {'mode': 'DRY_RUN', 'model_contract': 'action_set_v1', 'source': str(g2),
        'source_manifest_fingerprint': manifest['manifest_fingerprint'], 'TEST_payload_access': 0,
        'initialization': {'mode': 'scratch', 'pretrained_checkpoint': None, 'resume_checkpoint': None,
            'fresh_model': True, 'fresh_optimizer': True, 'fresh_selection_state': True},
        'splits': {part: {'enabled': part != 'test', 'samples': sum(e['samples'] for e in entries)}
            for part, entries in manifest['inventory'].items()},
        'dropout': {'h1': h1['dropout'], 'h2': h2['dropout']},
        'config': config, 'runtime': {**RUNTIME, 'cache_budgets_bytes': dict(CACHE_BUDGETS)},
        'sampler_quotas': {'ordinary': config['batch_size']-rare, 'structural': rare},
        'identity': identity}
    if 'actor_visibility' in sources:
        report.update(actor_visibility=sources['actor_visibility'],
                      normalizer_directory=sources['normalizer_directory'])
    return report, kw, h1, h2, manifest


@torch.no_grad()
def evaluate(model, data, device, *, batch_size=64, threshold=0.):
    if not len(data): raise ValueError('Empty validation split')
    model.eval(); metrics = ActionSetMetrics(); aggregate = FamilyAccumulator({name: 1. for name in FAMILIES})
    for indices in validation_batches(len(data), batch_size):
        batch = move_batch_to_device(collate_action_set_samples([data.get_for_collate(i) for i in indices]), device)
        prediction = model(batch['inputs']); _, stats = compute_action_set_loss(prediction, batch)
        aggregate.add(family_statistics(stats)); measure_batch(prediction, batch, metrics, threshold)
    return {**aggregate.compute(), 'metrics': metrics.compute()}


def _write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(output, g2, config, prepared):
    """Explicit --run only: TRAIN/VAL caches and epoch-zero selection baseline."""
    output = _output_path(output, g2)
    report, kw, h1, h2, manifest = prepared
    torch.set_num_threads(RUNTIME['cpu_threads'])
    if torch.get_num_interop_threads() != RUNTIME['cpu_interop_threads']:
        torch.set_num_interop_threads(RUNTIME['cpu_interop_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, optimizer, identity = build_scratch_actor(kw['spec'], kw['feature_schema'], h1, h2, manifest, config, device=device)
    state = selection_state(config); step = 0
    with open_dataset(g2, 'train', manifest['identity'],
            dynamic_budget_bytes=CACHE_BUDGETS['train_dynamic'],
            template_budget_bytes=CACHE_BUDGETS['train_template']) as train, \
         open_dataset(g2, 'validation', manifest['identity'],
            dynamic_budget_bytes=CACHE_BUDGETS['validation_dynamic'],
            template_budget_bytes=CACHE_BUDGETS['validation_template']) as validation:
        if not len(validation): raise ValueError('Empty validation split')
        sampler = JointActionBatchSampler(train.train_index(), batch_size=config['batch_size'],
            ordinary_quota=report['sampler_quotas']['ordinary'], structural_quota=report['sampler_quotas']['structural'], seed=config['seed'])
        output.mkdir(parents=True, exist_ok=False)
        _write(output/'config.json', config); _write(output/'identity.json', identity)
        _write(output/'preflight.json', {**report, 'mode': 'RUN', 'device': device})
        with (output/'history.jsonl').open('x') as history:
            for epoch in range(config['max_epochs']+1):
                _write(output/'status.json', {'phase': 'TRAIN' if epoch else 'VAL', 'epoch': epoch, 'global_step': step})
                training = FamilyAccumulator({name: 1. for name in FAMILIES})
                if epoch:
                    for indices in sampler:
                        batch = move_batch_to_device(collate_action_set_samples([train.get_for_collate(i) for i in indices]), device)
                        result = train_batch(model, batch, optimizer); training.add(result['families']); step += 1
                _write(output/'status.json', {'phase': 'VAL', 'epoch': epoch, 'global_step': step})
                val = evaluate(model, validation, device, batch_size=config['batch_size'], threshold=config['threshold_logit'])
                improved = state.observe(epoch, val['total_loss'])
                row = {'epoch': epoch, 'global_step': step, 'train': training.compute() if epoch else None,
                       'epoch_kind': 'TRAINED_EPOCH' if epoch else 'INITIAL_RANDOM_VAL',
                       'trained_epochs': epoch,
                       'validation': val, 'selection': asdict(state),
                       'initialization': report['initialization'], 'runtime': report['runtime'],
                       'dropout': report['dropout']}
                if 'actor_visibility' in report:
                    row['actor_visibility'] = report['actor_visibility']
                history.write(json.dumps(row, allow_nan=False)+'\n'); history.flush()
                save_action_set_checkpoint(output/'last.pt', model, optimizer, identity, config, row, step=step)
                if improved: save_action_set_checkpoint(output/'best.pt', model, optimizer, identity, config, row, step=step)
                if state.should_stop: break
        _write(output/'status.json', {'phase': 'COMPLETE', 'global_step': step, 'selection': asdict(state), 'TEST_payload_access': 0})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--g2', type=Path, default=DEFAULT_G2)
    parser.add_argument('--normalizers', type=Path, help='Explicit matching actor-visible F2 output directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Explicitly execute future TRAIN/VAL scratch training')
    args = parser.parse_args(argv)
    output = _output_path(args.output, args.g2)
    config = load_action_set_training_config(args.config)
    prepared = preflight(args.g2, config, normalizer_dir=args.normalizers)
    if args.run:
        print('BC-v4 FULL TRAINING\nINITIALIZATION: FROM SCRATCH\nPRETRAINED CHECKPOINT: NONE\n'
              'RESUME CHECKPOINT: NONE\nTRAIN: enabled\nVAL: enabled\nTEST: disabled', flush=True)
        print(json.dumps({**prepared[0], 'mode': 'RUN_REQUESTED'}, sort_keys=True, allow_nan=False), flush=True)
        return run(output, args.g2, config, prepared)
    print(json.dumps(prepared[0], sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
