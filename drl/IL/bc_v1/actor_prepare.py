"""Manual-only fresh actor F2 + TRAIN/VAL G2; never opens TEST payloads.

The source G1 stays full/physical and immutable; frozen G2-v3 supplies reviewed
resource authority/decisions. Action architecture is unchanged. Default CLI is
metadata preflight only. Full rebuild requires --run and absent destinations.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .actor_visibility import (VERSION, certify_assigned_goal, load_visibility_policy,
                               project_actor_features)
from . import dataset as g1
from .bc_v1_contract import load_robot_spec, robot_spec_fingerprint
from .feature_schema import load_feature_schema, feature_schema_fingerprint
from .normalization import (NormalizationSample, fit_normalizers, load_normalization_policy,
    normalizer_fingerprint, read_normalizers, write_normalizers, policy_fingerprint,
    serialize_normalizers)
from .resource_decisions import (add_resource_decisions, load_resource_contract,
    validate_decision_targets, validate_outcome_inputs)
from .resource_prepare import support_counts
from .sidecar import SidecarDataset, SidecarWriter, _hash, _publish_manifest, assert_exact
from .split import serialize_split_manifest
from .structural_decisions import add_no_structural_supervision
from .tensorization import load_tensor_layout, tensorize_policy_sample

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'configs/bc_v1/actor_prepare_r1m.json'
SPLITS = ('train', 'validation')
INPUT_SCHEMA = 'mssr.bc_v1.actor_inputs.v1'
TENSOR_SCHEMA = 'mssr.bc_v1.tensor_sample.actor_visible.v1'


def file_sha(path):
    return g1.file_hash(path)


def _write(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


class FrozenD1GoalProvider:
    """Explicit certificate for goals produced by the pinned reviewed D1 path.

    D1 emits goal.* only for current_goal or the CURRENT mission task of type
    goal, never a nominal route/task_frame/action. This source contract is
    checked against D1 code + task config, immutable cache identity and per-
    episode cache receipt. Caller must load samples with G1's verified reader.
    Arbitrary externally supplied FeatureViews receive no automatic exemption.
    """
    def __init__(self, manifest, task_config_path):
        self.manifest = manifest
        expected = file_sha(Path(__file__).with_name('task_context.py'))
        if (manifest.get('complete') is not True or manifest.get('normalized') is not False
                or manifest['identity']['code'].get('task_context') != expected
                or manifest['identity'].get('task_config_fingerprint') != _hash(json.loads(Path(task_config_path).read_text()))):
            raise ValueError('Frozen D1 goal source contract/code/config mismatch')
        self.identity = _hash({'provider': 'frozen_verified_D1_current_goal.v1',
            'g1': manifest['cache_fingerprint'], 'D1': expected,
            'task_config': manifest['identity']['task_config_fingerprint']})

    def __call__(self, sample):
        goals = [r for r in sample.inputs.features.environment if r.object_type == 'goal']
        if not goals:
            return None
        entry = self.manifest['episodes'].get(sample.audit.episode_id)
        if entry is None or not entry.get('cache_sha256') or sample.inputs.policy.context.task_id != 'navigation':
            raise ValueError('Uncertified current goal source sample')
        return certify_assigned_goal(sample.inputs.features, provider_identity=self.identity,
            source_contract_id='mssr.bc_v1.d1.current_physical_goal.v1', evidence=(
                'verified_G1_episode_sha256:'+entry['cache_sha256'],
                'current_sample:'+sample.audit.episode_id+':'+str(sample.audit.transition_index),
                'D1_only_current_goal_or_current_declared_physical_mission_goal'))


def _project(sample, visibility, schema, goal_provider):
    receipt = goal_provider(sample) if goal_provider is not None else None
    return project_actor_features(sample.inputs.features, sample.inputs.policy.context.task_id,
                                  visibility, schema, assigned_goal=receipt)


def fit_actor_normalizers(episode_loaders, split_manifest, *, spec, schema,
                          normalization_policy, visibility, goal_provider=None):
    """Same F2 mathematics, fitted ONLY on complete projected F1 TRAIN streams."""
    def stream(eid):
        for sample in episode_loaders[eid]():
            if sample.audit.episode_id != eid:
                raise ValueError('Actor normalization episode mismatch')
            yield NormalizationSample(eid, sample.audit.transition_index,
                _project(sample, visibility, schema, goal_provider), sample.supervision)
    train = split_manifest['splits']['train']
    stats = fit_normalizers({eid: lambda eid=eid: stream(eid) for eid in train},
        split_manifest, spec, schema, normalization_policy,
        expected_split_fingerprint=split_manifest['manifest_fingerprint'])
    if not stats.metadata['complete_train_coverage']:
        raise ValueError('Incomplete actor TRAIN normalization coverage')
    metadata = {**stats.metadata, 'actor_input_schema': INPUT_SCHEMA,
                'actor_visibility_fingerprint': visibility.fingerprint,
                'goal_provider_identity': getattr(goal_provider, 'identity', 'explicit_or_absent')}
    return replace(stats, metadata_items=tuple(sorted(metadata.items())))


def publish_actor_normalizers(stats, output_dir, *, source_root, visibility):
    output_dir, source_root = Path(output_dir).resolve(), Path(source_root).resolve()
    if output_dir.exists():
        raise ValueError('Actor F2 output collision: directory exists')
    if output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
        raise ValueError('Actor F2 output overlaps source')
    if stats.metadata.get('actor_visibility_fingerprint') != visibility.fingerprint:
        raise ValueError('Actor normalizer visibility mismatch')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'-', dir=output_dir.parent))
    reserved = published = False
    try:
        write_normalizers(stats, staging/'normalizers.json', source_root=source_root)
        context = {'schema_version': 'mssr.bc_v1.actor_normalization_fit.v1',
            'actor_input_schema': INPUT_SCHEMA, 'actor_visibility_fingerprint': visibility.fingerprint,
            'actor_visibility': visibility.document, 'normalizer_fingerprint': normalizer_fingerprint(stats),
            'normalizer_sha256': file_sha(staging/'normalizers.json'), 'TRAIN_only': True,
            'TEST_payload_access': 0, 'expected_metadata': stats.metadata}
        read_normalizers(staging/'normalizers.json', expected_metadata=stats.metadata)
        _write(staging/'fit_context.json', context)
        output_dir.mkdir()
        reserved = True
        os.rename(staging, output_dir)
        published = True
        return context
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if reserved and not published and output_dir.is_dir() and not any(output_dir.iterdir()):
            output_dir.rmdir()


def tensorize_actor_sample(sample, *, spec, feature_schema, normalization_policy,
                            normalizers, tensor_layout, visibility, assigned_goal=None):
    if normalizers.metadata.get('actor_visibility_fingerprint') != visibility.fingerprint:
        raise ValueError('Normalizer incompatible with actor visibility')
    view = project_actor_features(sample.inputs.features, sample.inputs.policy.context.task_id,
        visibility, feature_schema, assigned_goal=assigned_goal)
    projected = replace(sample, inputs=replace(sample.inputs, features=view))
    result = tensorize_policy_sample(projected, spec=spec, feature_schema=feature_schema,
        normalization_policy=normalization_policy, normalizers=normalizers, tensor_layout=tensor_layout)
    result['routing'].update(actor_input_schema=INPUT_SCHEMA,
        actor_visibility_fingerprint=visibility.fingerprint)
    return result


def _preserve_decisions(fresh, old, contract):
    """Re-encode physical labels with new F2; retain certified v3 decisions."""
    assert_exact(fresh['routing']['audit'], old['routing']['audit'])
    assert_exact(fresh['routing']['physical_supervision'], old['routing']['physical_supervision'])
    fresh = add_no_structural_supervision(fresh)
    authority = set(old['supervision']['resource']['decision_group'][
        old['supervision']['resource']['decision_supervised']].tolist())
    fresh = add_resource_decisions(fresh, contract, authoritative_resources=authority)
    for key in ('structural', 'resource_candidates'):
        assert_exact(fresh['inputs'][key], old['inputs'][key])
    for family in ('resource', 'structural'):
        for key in ('decision_group', 'decision_candidate', 'decision_supervised'):
            assert_exact(fresh['supervision'][family][key], old['supervision'][family][key])
    if 'resource_label_authority' in old['routing']:
        fresh['routing']['resource_label_authority'] = old['routing']['resource_label_authority']
    validate_outcome_inputs(fresh['inputs']['resource_candidates'], contract, len(fresh['inputs']['resource_x']))
    validate_decision_targets(fresh)
    return fresh


def build_actor_sidecar(source_root, output, *, sample_loader, split_manifest,
                        visibility, contract, normalizers, spec, schema,
                        normalization_policy, tensor_layout, goal_provider=None,
                        normalizer_sha=None, progress=None):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output.exists() or output.with_name(output.name+'.building').exists():
        raise ValueError('Actor sidecar output collision: exists')
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError('Actor sidecar output overlaps source')
    doc = json.loads((source_root/'manifest.json').read_text())
    if doc.get('manifest_fingerprint') != _hash({k:v for k,v in doc.items() if k!='manifest_fingerprint'}):
        raise ValueError('Frozen v3 manifest checksum mismatch')
    if (doc.get('complete') is not True or doc['inventory']['test']
            or doc['identity'].get('resource_contract_fingerprint') != contract.fingerprint):
        raise ValueError('Expected complete sealed frozen resource v3 source')
    if normalizers.metadata.get('actor_visibility_fingerprint') != visibility.fingerprint:
        raise ValueError('Actor F2 visibility mismatch')
    if (normalizers.metadata.get('split_fingerprint') != split_manifest['manifest_fingerprint']
            or doc['identity']['split_fingerprint'] != split_manifest['manifest_fingerprint']):
        raise ValueError('Actor normalizer/source split differs from F1')
    if normalizers.metadata.get('goal_provider_identity') != getattr(goal_provider, 'identity', 'explicit_or_absent'):
        raise ValueError('Actor normalizer goal provider identity mismatch')
    for part in SPLITS:
        if [e['episode_id'] for e in doc['inventory'][part]] != split_manifest['splits'][part]:
            raise ValueError('Actor source membership differs from F1')
    semantics = {'actor_input_schema': INPUT_SCHEMA, 'visibility': visibility.fingerprint,
        'code': {n:file_sha(Path(__file__).with_name(n+'.py')) for n in
                 ('actor_visibility', 'actor_prepare', 'normalization', 'tensorization')}}
    identity = {**doc['identity'], 'f2_fingerprint': normalizer_fingerprint(normalizers),
        'normalizer_sha': normalizer_sha or hashlib.sha256(serialize_normalizers(normalizers)).hexdigest(),
        'tensorizer_fingerprint': _hash({'upstream':doc['identity']['tensorizer_fingerprint'],**semantics}),
        'storage_fingerprint': _hash({'upstream':doc['identity']['storage_fingerprint'],**semantics})}
    writer = SidecarWriter(output, identity, doc['inventory'])
    supports = {part:Counter() for part in SPLITS}
    counts = {}
    try:
        for part in SPLITS:
            index = 0
            with SidecarDataset(source_root, part, expected_identity=doc['identity']) as olddata:
                for episode in doc['inventory'][part]:
                    samples = sample_loader(episode['episode_id'])
                    seen = []
                    for sample in samples:
                        if sample.audit.episode_id != episode['episode_id']:
                            raise ValueError('G1 episode mismatch')
                        seen.append(sample.audit.transition_index)
                        old = olddata[index]
                        goal = goal_provider(sample) if goal_provider is not None else None
                        new = tensorize_actor_sample(sample, spec=spec, feature_schema=schema,
                            normalization_policy=normalization_policy, normalizers=normalizers,
                            tensor_layout=tensor_layout, visibility=visibility, assigned_goal=goal)
                        new = _preserve_decisions(new, old, contract)
                        supports[part].update(support_counts(new, contract))
                        writer.add(new, part, index)
                        index += 1
                        if progress and index % 1000 == 0:
                            progress({'split':part, 'samples':index})
                    if len(seen) != episode['samples'] or _hash(seen) != episode['transition_indices_fingerprint']:
                        raise ValueError('Incomplete/foreign G1 transition coverage')
                if index != len(olddata):
                    raise ValueError('Actor source sample coverage mismatch')
            counts[part] = index
        result = writer.finalize()
    finally:
        writer.close()
    audit = {'schema_version':INPUT_SCHEMA, 'fingerprint':visibility.fingerprint,
        'policy':visibility.document, 'source_manifest_fingerprint':doc['manifest_fingerprint'],
        'goal_provider_identity':getattr(goal_provider,'identity','explicit_or_absent'),
        'sample_counts':counts, 'support':{s:dict(c) for s,c in supports.items()},
        'candidates_preserved':True, 'physical_labels_preserved':True,
        'resource_and_structural_decisions_preserved':True, 'TEST_payload_access':0}
    result = _publish_manifest(output, {**result, 'actor_visibility':audit,
        'tensor_sample_schema':TENSOR_SCHEMA,
        'resource_decision_schema':doc.get('resource_decision_schema'),
        'structural_decision_schema':doc.get('structural_decision_schema')})
    return {'manifest_fingerprint':result['manifest_fingerprint'], **audit}


def verify_actor_sidecar(root, visibility, contract):
    root = Path(root)
    doc = json.loads((root/'manifest.json').read_text())
    if (doc.get('tensor_sample_schema') != TENSOR_SCHEMA or doc['inventory']['test']
            or doc['actor_visibility']['fingerprint'] != visibility.fingerprint):
        raise ValueError('Incompatible actor sidecar contract')
    counts = {}
    support = {s:Counter() for s in SPLITS}
    for part in SPLITS:
        with SidecarDataset(root, part, expected_identity=doc['identity']) as data:
            if data.db.execute('PRAGMA integrity_check').fetchone() != ('ok',):
                raise ValueError('Actor SQLite integrity failure')
            if data.db.execute("SELECT count(*) FROM samples WHERE split='test'").fetchone()[0]:
                raise ValueError('Forbidden TEST payload')
            for i in range(len(data)):
                row = data[i]
                if row['routing'].get('actor_visibility_fingerprint') != visibility.fingerprint:
                    raise ValueError('Actor sample visibility identity mismatch')
                if any(any(name in key for name in ('task_frame.', 'environment.floor_height',
                        'environment.upper_deck', 'environment.platform_bounds', 'environment.clear_landing',
                        'stairs.step_count', 'button.plunger_', 'cones.corridor_width'))
                        for key in row['inputs']['environment']):
                    raise ValueError('Critic-only token in actor sidecar')
                validate_outcome_inputs(row['inputs']['resource_candidates'],contract,len(row['inputs']['resource_x']))
                validate_decision_targets(row)
                support[part].update(support_counts(row,contract))
            counts[part] = len(data)
    if counts != doc['actor_visibility']['sample_counts'] or {s:dict(c) for s,c in support.items()} != doc['actor_visibility']['support']:
        raise ValueError('Actor support/count integrity mismatch')
    return {'integrity':'PASS', 'sample_counts':counts, 'support':{s:dict(c) for s,c in support.items()},
            'manifest_fingerprint':doc['manifest_fingerprint'], 'TEST_payload_access':0}


def preflight(config=CONFIG):
    cfg = json.loads(Path(config).read_text())
    if cfg.get('schema_version') != 'mssr.bc_v1.actor_prepare.v1':
        raise ValueError('Unsupported actor rebuild config')
    def path(name):
        result = (ROOT/cfg[name]).resolve()
        if not result.is_relative_to(ROOT):
            raise ValueError('Actor path escapes repository')
        return result
    source, cache = path('source_sidecar'), path('physical_cache')
    out, f2out = path('sidecar_output'), path('normalizer_output')
    for output in (out,f2out):
        if output.exists() or output.with_name(output.name+'.building').exists():
            raise ValueError('Actor output exists; no overwrite/resume')
        if not output.is_relative_to(ROOT/'artifacts/bc_v1') or any(
                output.is_relative_to(src) or src.is_relative_to(output) for src in (source,cache)):
            raise ValueError('Unsafe actor output path')
    if out.is_relative_to(f2out) or f2out.is_relative_to(out):
        raise ValueError('Actor F2/sidecar output overlap')
    for name,digest in cfg['file_pins'].items():
        if file_sha(ROOT/name) != digest:
            raise ValueError('Actor rebuild input/code pin mismatch: '+name)
    visibility = load_visibility_policy(path('visibility_policy'))
    if visibility.document['radius_m'] != 1.0 or visibility.fingerprint != cfg['visibility_fingerprint']:
        raise ValueError('Expected frozen R=1m actor contract')
    source_doc = json.loads((source/'manifest.json').read_text())
    manifest = g1.read_cache_manifest(cache, expected_identity=cfg['physical_identity'])
    split = json.loads(path('split').read_text())
    serialize_split_manifest(split)
    if (source_doc['manifest_fingerprint'] != cfg['source_manifest_fingerprint']
            or source_doc['identity'] != cfg['source_identity']
            or manifest['cache_fingerprint'] != source_doc['identity']['g1_fingerprint']
            or split['manifest_fingerprint'] != source_doc['identity']['split_fingerprint']
            or manifest['splits'] != split['splits'] or source_doc['inventory']['test']):
        raise ValueError('Frozen actor upstream identity/membership mismatch')
    for part in SPLITS:
        if [e['episode_id'] for e in source_doc['inventory'][part]] != split['splits'][part]:
            raise ValueError('Frozen actor F1 membership mismatch')
    spec = load_robot_spec(path('robot_spec'))
    schema = load_feature_schema(path('feature_schema'))
    normal_policy = load_normalization_policy(path('normalization_policy'),schema)
    contract = load_resource_contract(path('resource_contract'),spec)
    layout = load_tensor_layout(path('tensor_layout'))
    for name,actual in [('robot_spec_fingerprint',robot_spec_fingerprint(spec)),
                        ('feature_schema_fingerprint',feature_schema_fingerprint(schema)),
                        ('resource_contract_fingerprint',contract.fingerprint),
                        ('layout_fingerprint',layout['fingerprint'])]:
        if source_doc['identity'].get(name) != actual:
            raise ValueError('Actor catalog/layout identity mismatch: '+name)
    goals = FrozenD1GoalProvider(manifest,path('task_config'))
    return dict(config=cfg,source=source,cache=cache,sidecar_output=out,
        normalizer_output=f2out,visibility=visibility,source_doc=source_doc,manifest=manifest,
        split=split,spec=spec,schema=schema,normalization_policy=normal_policy,
        contract=contract,tensor_layout=layout,goal_provider=goals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args(argv)
    runtime = preflight(args.config)
    if not args.run:
        print(json.dumps({'mode':'DRY_RUN','radius_m':1.0,
            'visibility_fingerprint':runtime['visibility'].fingerprint,
            'normalizer_output':str(runtime['normalizer_output']),
            'sidecar_output':str(runtime['sidecar_output']),
            'expected_counts':runtime['config']['expected_counts'], 'TEST_payload_access':0},indent=2))
        return
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest, split = runtime['manifest'],runtime['split']
    allowed = {eid for part in SPLITS for eid in split['splits'][part]}
    def samples(eid):
        if eid not in allowed:
            raise ValueError('Forbidden non-TRAIN/VAL payload access')
        return g1._load_episode(runtime['cache'],eid,manifest['episodes'][eid])
    print(json.dumps({'phase':'TRAIN-only actor F2 fit','visibility':runtime['visibility'].document}),flush=True)
    stats = fit_actor_normalizers({eid:lambda eid=eid:samples(eid) for eid in split['splits']['train']},split,
        **{k:runtime[k] for k in ('spec','schema','normalization_policy','visibility','goal_provider')})
    context = publish_actor_normalizers(stats,runtime['normalizer_output'],
        source_root=runtime['cache'],visibility=runtime['visibility'])
    print(json.dumps({'phase':'TRAIN+VAL actor sidecar build','f2_fingerprint':context['normalizer_fingerprint']}),flush=True)
    result = build_actor_sidecar(runtime['source'],runtime['sidecar_output'],sample_loader=samples,
        split_manifest=split,normalizers=stats,normalizer_sha=context['normalizer_sha256'],
        **{k:runtime[k] for k in ('spec','schema','normalization_policy','visibility','goal_provider','contract','tensor_layout')},
        progress=lambda row:print(json.dumps(row),flush=True))
    if result['sample_counts'] != runtime['config']['expected_counts']:
        raise ValueError('Actor full rebuild counts mismatch')
    verification = verify_actor_sidecar(runtime['sidecar_output'],runtime['visibility'],runtime['contract'])
    # Re-check pinned sources without rerunning a preflight that requires absent outputs.
    for name,digest in runtime['config']['file_pins'].items():
        if file_sha(ROOT/name) != digest:
            raise ValueError('Actor source/config changed during build: '+name)
    result.update(verification=verification,normalizer_context=context,TEST_payload_access=0)
    _write(runtime['sidecar_output']/'SUCCESS.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
