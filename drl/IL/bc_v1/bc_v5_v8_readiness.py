"""BC-v5 v8 Stage 6+7+8 readiness: targeted 17-row audit, no dataset scan.

Never writes Canonical/G1/F2/G2. The one and only new artifact is a compact
provenance certificate that can be read by the BC-v5 trainer. Do not issue
certificate unless all targeted checks and H1/H2 metadata preflight pass.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .action_set_targets import positive_indices
from .bc_v5_integrity import read_report
from .sidecar import SidecarDataset, _read_manifest, _hash, assert_exact

VERSION='mssr.bc_v5.v8.stage678_readiness.v1'
EXPECTED={'train':82528,'validation':23751}
EXCEPTIONS={'train':13,'validation':4}
ZERO={'train':10,'validation':2}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _stage11_report(report, target_fpr, diagnostic_sha):
    doc=_read(report)
    digest=doc.pop('fingerprint',None)
    require(_hash(doc)==digest,'Stage11 report fingerprint invalid')
    require(doc['complete'] is True and doc['test_payload_access']==0 and
            doc['new_g2_fingerprint']==target_fpr and
            doc['source_rows']==EXPECTED and
            doc['masked_align_positive_instances']==EXCEPTIONS and
            doc['zero_positive_graphs_after_mask']==ZERO and
            doc['masked_align_eligible_candidate_instances']=={'train':0,'validation':0} and
            doc['source_diagnostic_sha256']==diagnostic_sha and
            doc['positive_retention_except_mask_pass'] is True and
            doc['h1_features_unchanged'] is True and
            doc['legacy_align_restoration'] is False,
            'Stage11 report is not certified for this v8 source')
    return digest


def certify(root, *, g2_path, stage11_report, diagnostic_path, overlay_path,
            normalizers, training_config):
    """Read manifests + 17 exception samples from old and new G2, never all data."""
    from . import train_action_set as base
    from . import bc_v5_binary
    from . import bc_v5_train

    root=Path(root).resolve()
    g2_path=Path(g2_path).resolve()
    original=Path(root/'artifacts/bc_v1/sidecar/g2_v8_actor_visible_r1m').resolve()
    require(g2_path.is_relative_to(root/'artifacts/bc_v1/sidecar') and
            original.is_relative_to(root/'artifacts/bc_v1/sidecar'), 'Unexpected G2 locations')
    data=_read(g2_path/'manifest.json')
    manifest=_read_manifest(g2_path,data['identity'])
    topo=manifest.get('bc_v5_topology')
    require(manifest['complete'] and manifest['inventory']['test']==[] and
            isinstance(topo,dict) and
            topo['source_manifest_fingerprint']=='5b0deb04b26c19ce0f5e0290d984335e91b2ecb7ccafb32bdea4c2aea84ec275' and
            topo['test_payload_access']==0 and topo['offline_legacy_restoration'] is False and
            topo['runtime_exception'] is False, 'Stage11 provenance invalid')
    require(all(sum(e['samples'] for e in manifest['inventory'][s])==EXPECTED[s]
                for s in EXPECTED),'Stage11 split counts invalid')

    source=_read(original/'manifest.json')
    old=_read_manifest(original,source['identity'])
    require(old['manifest_fingerprint']==topo['source_manifest_fingerprint'] and
            old['identity']['canonical_sha']==manifest['identity']['canonical_sha'],
            'Original actor G2 mismatch')
    require(_sha(root/'datasets/expert_canonical_v8_unified/manifest.json') ==
            old['identity']['canonical_sha'], 'Canonical v8 manifest unexpectedly changed')
    require(manifest['identity']['normalizer_sha']==old['identity']['normalizer_sha'] and
            manifest['identity']['f2_fingerprint']==old['identity']['f2_fingerprint'],
            'F2 changed in Stage11')
    diag_sha=_sha(diagnostic_path)
    require(topo['masked_align_registry_sha256']==diag_sha,'Diagnostic SHA differs from Stage11')
    report_digest=_stage11_report(stage11_report,manifest['manifest_fingerprint'],diag_sha)
    stage5=_read(overlay_path)
    require(stage5.get('fingerprint')==topo['source_overlay_fingerprint'],
            'Stage11 does not refer to this Stage5 overlay')
    overlay=read_report(overlay_path,expected_g2_fingerprint=old['manifest_fingerprint'])
    require(overlay['complete'] is True and overlay['ready_for_training'] is True and
            overlay['test_payload_access']==0 and overlay['expected_rows']==EXPECTED,
            'Stage5 source overlay incomplete')
    # Only walk 17 diagnostic exception identities; no scan of G2 payloads.
    diag=_read(diagnostic_path)
    require(diag['missing_align_positive_count']==17 and
            len(diag['missing_align_positives'])==17 and
            diag['test_payload_access']==0,
            'Unexpected 17 align diagnostics')
    needed={(r['split'],r['episode_id'],r['transition_index'])
            for r in diag['missing_align_positives']}
    relevant_overlay={ (r['split'],r['episode_id'],r['transition_index']):r
                       for r in overlay['rows']
                       if (r['split'],r['episode_id'],r['transition_index']) in needed }
    require(len(relevant_overlay)==len(needed),'Stage5 overlay does not cover align exceptions')
    zero={s:[] for s in EXPECTED}
    masked=Counter()
    visible_candidate_masks=Counter()
    seen=set()
    for split in ('train','validation'):
        with SidecarDataset(original,split,expected_identity=old['identity']) as before, \
             SidecarDataset(g2_path,split,expected_identity=manifest['identity']) as after:
            for row in diag['missing_align_positives']:
                if row['split']!=split:continue
                eid,tid,mid=row['episode_id'],row['transition_index'],row['initiator']
                ident=(split,eid,tid,mid)
                require(ident not in seen,'Duplicate exceptional align identity')
                seen.add(ident);masked[split]+=1
                old_sample=before.get(eid,tid)
                sample=after.get(eid,tid)
                old_route=old_sample['routing'];route=sample['routing']
                assert_exact(old_route['audit'],route['audit'])
                require(mid in route['bc_v5_masked_align_initiators'] and
                        route['bc_v5_masked_align_count']==1,
                        'Missing loss-only align source marker')
                old_pos=positive_indices(old_route)
                pos=positive_indices(route)
                require(len(old_pos)-len(pos)==1,'Unexpected physical positive loss')
                before_targets=old_route['physical_supervision']
                after_targets=route['physical_supervision']
                require(len(before_targets.structural_targets)==len(after_targets.structural_targets)+1 and
                        before_targets.resource_targets==after_targets.resource_targets,
                        'Unexpected physical supervision mutation')
                removed=[t for t in before_targets.structural_targets
                         if t.operation_id=='align_faces' and t.initiator.module_id==mid]
                require(len(removed)==1 and removed[0] not in after_targets.structural_targets,
                        'Historical nonrepresentable label still in trainable targets')
                for field in old_sample['inputs']:
                    if field!='structural':
                        assert_exact(old_sample['inputs'][field],sample['inputs'][field],
                                     'Stage11 changed H1/resource input '+field)
                # The new candidate admission is preserved even for a masked
                # initiator; verify that any eligible align candidate is invalid
                # for the BINARY SUPERVISION only, not removed from catalog.
                entry=relevant_overlay[split,eid,tid]
                mask=bc_v5_binary.candidate_validity(route,entry)
                start=len(route['resource_candidates'])
                for i,(action,binding) in enumerate(route['structural_bindings']):
                    if action.action_id=='align_faces' and binding.initiator.module_id==mid:
                        require(not bool(mask[start+i]),'Ambiguous align candidate contributes to loss')
                        visible_candidate_masks[split]+=1
                if not pos:
                    zero[split].append({'episode_id':eid,'transition_index':tid})
    require(dict(masked)==EXCEPTIONS and
            {s:len(rows) for s,rows in zero.items()}==ZERO and
            dict(visible_candidate_masks) in ({}, {'train':0,'validation':0}),
            'Mismatch with certified Stage11 17 masks / 12 zero-positive graphs')
    # Authenticate actor-visible F2 and H1/H2 metadata; no model or GPU needed.
    config=base.load_action_set_training_config(training_config)
    base.preflight(g2_path,config,normalizer_dir=normalizers)
    require(hasattr(bc_v5_train,'FilteredTrainingDataset'), 'Trainer unavailable')
    return {
        'schema_version':VERSION,
        'complete':True,
        'train_ready_contract':True,
        'test_payload_access':0,
        'stage11_g2':str(g2_path),
        'stage11_g2_manifest_fingerprint':manifest['manifest_fingerprint'],
        'source_actor_g2_fingerprint':old['manifest_fingerprint'],
        'stage11_report_fingerprint':report_digest,
        'stage5_overlay_fingerprint':stage5['fingerprint'],
        'align_exception_diagnostic_sha256':diag_sha,
        'masked_align_positive_instances':dict(masked),
        'zero_positive_graphs':{s:sorted(rows,key=lambda x:(x['episode_id'],x['transition_index']))
                               for s,rows in zero.items()},
        'supervision_policy':'closed_world + per-resource conflicts + masked align initiators',
        'sampler_policy':'TRAIN-only adaptive exact-positive primitive pools; metadata+sample bootstrap',
        'h1_h2':'metadata preflight PASS; GPU forward/backward still pending',
        'validation_sampler':'sequential unmodified',
        'note':'Stage11 G2 remains unchanged; 12 zero-positive rows excluded from supervised batches only.'
    }


def load_readiness(path,manifest,overlay_file):
    doc=_read(path)
    digest=doc.pop('fingerprint',None)
    if not digest or _hash(doc)!=digest:
        raise ValueError('Invalid Stage678 readiness fingerprint')
    topo=manifest.get('bc_v5_topology')
    if (doc.get('schema_version')!=VERSION or not doc.get('complete')
            or not doc.get('train_ready_contract') or doc.get('test_payload_access')!=0
            or not isinstance(topo,dict) or
            doc.get('stage11_g2_manifest_fingerprint')!=manifest['manifest_fingerprint'] or
            doc.get('source_actor_g2_fingerprint')!=topo['source_manifest_fingerprint'] or
            doc.get('stage5_overlay_fingerprint')!=topo['source_overlay_fingerprint'] or
            doc.get('stage5_overlay_fingerprint')!=_read(overlay_file).get('fingerprint')):
        raise ValueError('Stage678 readiness identity mismatch')
    for s, expected in ZERO.items():
        if len(doc['zero_positive_graphs'][s])!=expected:
            raise ValueError('Stage678 zero-positive subset mismatch')
    return doc,digest
