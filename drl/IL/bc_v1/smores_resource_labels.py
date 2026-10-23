"""Offline SMORES authority policy v1. No metadata enters learned inputs.

Only complete, valid NAV2 morphology-controller snapshots with no native goal
certify sparse absence. The producer returns the ENTIRE sparse behavior command
mapping each tick (morphology_library + parallel_self_assembly sparse command
semantics); omitted modules are shape-held/towable, not implicitly zeroed.
Other behaviors/lifecycle/ownership remain UNKNOWN. This deliberately narrower
policy never promotes a Canonical loss-mask false or an expert target conflict.
"""
from pathlib import Path
import json
import math
import hashlib
from .resource_decisions import add_resource_decisions
from .__main__ import _read_indexed_source_records
from .smores_adapter import _source_index

AUTHORITY_POLICY='smores.complete_nav2_snapshot.v1'


def complete_navigation_snapshot(record):
    action=record.get('expert_action') or {};commands=action.get('locomotion')
    return (record.get('active_primitive')=='nav2_planar_route'
        and record.get('action_valid') is True
        and record.get('supervision',{}).get('valid_for_behavior_cloning') is True
        and record.get('debug',{}).get('behavior')=='nav2_planar_route'
        and action.get('primitive_goal','missing') is None and action.get('magnetic')==[]
        and isinstance(commands,dict) and bool(commands)
        and all(isinstance(v,dict) and {'vx','vy','yaw_rate','pan_rate_rad_s'}<=v.keys()
            and all(type(v[k]) in (int,float) and math.isfinite(v[k]) for k in ('vx','vy','yaw_rate','pan_rate_rad_s'))
            for v in commands.values()))


class SmoresResourceLabelProvider:
    def __init__(self,root,contract,*,allowed_episodes=None,source_certificates=None):
        self.root=Path(root).resolve();self.contract=contract;self.allowed=allowed_episodes
        self.canonical=self.root/'datasets/expert_v1_canonical_v6'
        manifest=json.loads((self.canonical/'manifest.json').read_text())
        self.episodes={e['episode_id']:e for e in manifest['episodes']}
        config=self.root/'configs/bc_v1/resource_prepare_v3.json'
        prepared=json.loads(config.read_text()) if config.is_file() else {}
        self.source_certificates=dict(prepared.get('authority_sources',{}) if source_certificates is None else source_certificates)
        self.identity=hashlib.sha256(json.dumps({'policy':AUTHORITY_POLICY,'source_certificates':self.source_certificates},sort_keys=True).encode()).hexdigest()
        self.source_hashes={};self.TEST_payload_access=0

    def __call__(self,sample):
        audit=sample['routing']['audit'];eid=audit.episode_id
        if self.allowed is not None and eid not in self.allowed:raise ValueError('Episode outside TRAIN/VAL authority domain')
        e=self.episodes[eid];path=(self.canonical/e['file']).resolve()
        if not path.is_relative_to(self.canonical):raise ValueError('Canonical path escape')
        idx=_source_index(path)
        if idx.sha256!=e['sha256']:raise ValueError('Canonical source hash mismatch')
        row=next(_read_indexed_source_records(path,(audit.transition_index+1,)))[1]
        p=row['provenance']
        if p['logical_episode_id']!=eid or p['canonical_transition_index']!=audit.transition_index:raise ValueError('Canonical source row mismatch')
        sources=[];evidence=[];certified=True
        for ref in p['source_records']:
            source=Path(ref['_loader_source_path']);source=(self.root/source).resolve()
            if not source.is_relative_to(self.root):raise ValueError('Source path escapes repository')
            if not source.is_file():sources=[];break
            value=next(_read_indexed_source_records(source,(ref['_loader_source_row'],)))[1]
            stamp=(value.get('graph_t') or {}).get('stamp',value.get('stamp'))
            if stamp is None or not math.isclose(stamp,p['physical_stamp'],rel_tol=0.,abs_tol=1e-9):raise ValueError('Source stamp mismatch')
            digest=_source_index(source).sha256;relative=str(source.relative_to(self.root))
            expected=self.source_certificates.get(relative)
            if expected is not None and digest!=expected:raise ValueError('Source authority hash mismatch: '+relative)
            certified=certified and expected is not None
            self.source_hashes[relative]=digest
            sources.append(value);evidence.append(dict(path=str(source.relative_to(self.root)),row=ref['_loader_source_row'],sha256=digest))
        modules=row['action_t']['modules'];conflicts=row['supervision'].get('conflicts',[])
        complete=certified and bool(sources) and all(complete_navigation_snapshot(v) for v in sources) and not conflicts
        complete=complete and all(not m.get('participates_in') and not m.get('structural',{}).get('commanded') for m in modules.values())
        authority=set()
        if complete:
            # All commands must agree with the authoritative full snapshots;
            # any masked existing resource is left unknown by generic mapping.
            for i,ref in enumerate(sample['routing']['resource_refs']):
                mid=ref.module.module_id;part=modules.get(mid)
                if part is None:continue
                key={'locomotion':'wheels','internal_motion':'internal'}.get(ref.resource_id)
                if key is None:continue
                if not part[key]['commanded'] and all(mid not in v['expert_action']['locomotion'] for v in sources):authority.add(i)
        new=add_resource_decisions(sample,self.contract,authoritative_resources=authority)
        new['routing']['resource_label_authority']={'policy':AUTHORITY_POLICY,'complete':complete,'evidence':evidence,
            'certified_absent_resources':sorted(authority)}
        return new
