"""Passive native action evidence. No shadow samples, policy control or rewards."""
from collections import defaultdict
from dataclasses import asdict, is_dataclass
import json
from heapq import nsmallest
from itertools import chain
from time import perf_counter
from pathlib import Path

import torch

COUNTERS = (
    'masked_busy_count', 'masked_unavailable_count', 'masked_not_ready_count', 'masked_optional_membership_count',
    'intra_step_conflict_drop_count', 'executor_busy_rejection_count',
    'executor_other_rejection_count', 'structural_sampled_count',
    'structural_projected_count', 'structural_executed_count', 'structural_started_count',
    'structural_completed_count', 'structural_failed_count', 'resource_sampled_count',
    'primitive_started_count', 'primitive_completed_count', 'primitive_failed_count',
)


def serial(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return serial(asdict(value))
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f'Unserializable action diagnostic {type(value)}')


def event_identity(env, index, timestamp_s=None):
    return dict(env_index=index, generation=env.generation[index], task=env.domain_by_env[index],
        seed=env.seed_by_env[index], timestamp_s=float(timestamp_s if timestamp_s is not None
        else env._sim_step_counter * env.physics_dt),
        vector_step=int(getattr(env, 'common_step_counter', 0)))


def ensure_action_events(env):
    recorder = getattr(env, 'c1_action_events', None)
    if recorder is None:
        from drl.MARL.bc_v4_runtime_inference import load_bc_v4_runtime_inference_config
        config = load_bc_v4_runtime_inference_config(Path(__file__).resolve().parents[2])
        recorder = C1ActionEvents(getattr(env.cfg, 'c1_evidence_directory', None),
            threshold_logit=float(config['deterministic_presence_threshold_logit']))
        env.c1_action_events = recorder
    return recorder


class C1ActionEvents:
    def __init__(self, directory=None, *, masked_detail_limit=16, threshold_logit=4.5):
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.masked_detail_limit = int(masked_detail_limit)
        if self.masked_detail_limit < 0:
            raise ValueError('Negative masked detail limit')
        self.threshold = float(threshold_logit)
        self.totals = defaultdict(int)
        self.by_task = defaultdict(lambda: defaultdict(int))
        self.row_transitions = 0
        self.reward_sum = 0.
        self.valid_connection_max = 0
        self.successes = 0
        self.counts_by_row = {}
        self.pending = {}
        self.goals = {}
        self.terminal = set()
        self.mask_detail_payload_build_count = 0
        self.mask_detail_descriptors_considered = 0
        self.mask_event_logging_s = 0.
        self.feasibility_cache_metrics = {}

    def _count(self, name, identity, count=1):
        self.totals[name] += int(count)
        self.by_task[identity['task']][name] += int(count)
        key = (identity['env_index'], identity['generation'])
        self.counts_by_row.setdefault(key, defaultdict(int))[name] += int(count)

    def _emit(self, row):
        row = serial({'schema': 'mssr.c1.action_event.v1', **row})
        if self.directory:
            with (self.directory / 'action_events.jsonl').open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
        return row

    def _score(self, raw):
        calibrated = float(raw) - self.threshold
        return dict(raw_presence_logit=float(raw), calibrated_presence_logit=calibrated,
            calibrated_presence_probability=float(torch.sigmoid(torch.tensor(calibrated, dtype=torch.float64))),
            threshold_logit=self.threshold, above_presence_boundary=calibrated >= 0.)

    def _candidate(self, candidate):
        from drl.IL.bc_v1.live_policy import claim_key
        choice = candidate.choice
        parts = candidate.candidate_id.split('|')
        primitive = choice.action_id if choice is not None else (parts[1] if len(parts) > 1 else 'unknown')
        ports = choice.connectors if choice is not None else ()
        modules = (tuple(p.module.module_id for p in ports) if ports else
                   tuple(m.module_id for m in choice.participants) if choice is not None and choice.participants
                   else candidate.owner_ids)
        # Resource candidates have no structural choice. Preserve the actual
        # decoded target for passive reward attribution, including clamping.
        command = getattr(getattr(candidate, 'decoded', None), 'command', {})
        resolved = (dict(resolved_primitive=command['command'],
            resolved_module_ids=tuple(command.get('module_ids', modules)),
            resolved_parameters=dict(command.get('parameters', {})))
            if isinstance(command, dict) and 'command' in command else {})
        return dict(candidate_id=candidate.candidate_id, candidate_index=candidate.candidate_index,
            owner_module_id=candidate.owner_ids[0] if candidate.owner_ids else None,
            primitive=primitive, module_ids=modules,
            faces=tuple(p.connector_id for p in ports),
            connectors=tuple((p.module.module_id, p.connector_id) for p in ports),
            arguments=tuple(choice.parameters) if choice is not None else (),
            resource_claims=tuple(claim_key(c) for c in candidate.claims),
            **resolved,
            **self._score(candidate.raw_presence_logit))

    def mask_events(self, *, records, availability, action, identity, presence_logits=None):
        if availability is None:
            return []
        started = perf_counter()
        try:
            masked_count = 0
            for code in ('masked_busy', 'masked_unavailable', 'masked_not_ready'):
                count = sum(availability.candidate_mask[r.candidate_index] and
                    availability.reasons[r.candidate_index].code == code for r in records)
                self._count(code + '_count', identity, count)
                masked_count += count
            self._count('masked_optional_membership_count', identity,
                sum(len(b) for _, b in availability.scope_blocked) +
                sum(len(b) for _, _, b in availability.reference_set_blocked))
            self.mask_detail_descriptors_considered += masked_count + len(availability.optional_factors)
            if not self.masked_detail_limit:
                return []
            # One bulk host transfer, not a GPU scalar read and probability
            # allocation for every masked candidate. Scores are never cached.
            if presence_logits is None:
                values = torch.stack([r.presence_logit for r in records]).detach().cpu().tolist() if records else []
                raw = {r.candidate_index: v for r, v in zip(records, values)}
            else:
                raw = presence_logits.detach().cpu().tolist()
            by_index = {r.candidate_index: r for r in records}
            descriptors = chain(
                ((r, None) for r in records if availability.candidate_mask[r.candidate_index]),
                ((by_index[f.candidate_index], f) for f in availability.optional_factors))
            def rank(item):
                record, factor = item
                return (-raw[record.candidate_index], record.candidate_id,
                    'candidate' if factor is None else 'optional_membership',
                    '' if factor is None else factor.factor, -1 if factor is None else factor.domain_index)
            chosen = nsmallest(self.masked_detail_limit, descriptors, key=rank)
            rows = []
            for record, factor in chosen:
                parts = record.candidate_id.split('|')
                if factor is None:
                    reason = availability.reasons[record.candidate_index]
                    ports = tuple(p.rsplit(':', 1) for p in parts[-1].split('>')) if (
                        record.candidate_kind == 'structural' and parts[-1] != 'none') else ()
                    row = dict(identity, candidate_id=record.candidate_id,
                        candidate_index=record.candidate_index, owner_module_id=record.owner_agent_id,
                        primitive=parts[1] if record.candidate_kind == 'structural' else (parts[3] if len(parts) > 3 else parts[0]),
                        module_ids=(record.owner_agent_id, *record.participant_agent_ids),
                        connectors=ports, faces=tuple(p[-1] for p in ports), sampled=False,
                        **self._score(raw[record.candidate_index]),
                        resource_claims=tuple(k for k, _ in reason.resource_modes), native_resource_modes=reason.resource_modes,
                        availability_state=reason.code, outcome=reason.code,
                        drop_reason=('native_active_claims' if reason.code == 'masked_busy' else
                            'dock_not_geometrically_ready' if reason.code == 'masked_not_ready' else
                            'base_hard_mask'),
                        conflicts_with=reason.conflicts_with, mask_granularity='candidate',
                        detail_selection='deterministic highest raw logits; bounded; no shadow sample')
                else:
                    row = dict(identity, candidate_id=record.candidate_id,
                        candidate_index=record.candidate_index, owner_module_id=record.owner_agent_id,
                        primitive=parts[1], module_ids=(factor.module_id,), connectors=(), faces=(), sampled=False,
                        candidate_sampled=bool(action.presence[record.candidate_index]),
                        candidate_available_at_presence=not availability.candidate_mask[record.candidate_index],
                        **self._score(raw[record.candidate_index]), factor=factor.factor,
                        argument_index=factor.argument_index, domain_index=factor.domain_index,
                        resource_claims=tuple(k for k, _ in factor.resource_modes), native_resource_modes=factor.resource_modes,
                        availability_state='masked_busy', outcome='masked_busy', drop_reason='native_active_claims',
                        conflicts_with=factor.conflicts_with, mask_granularity='optional_membership',
                        detail_selection='deterministic highest raw logits; bounded; no shadow sample')
                self.mask_detail_payload_build_count += 1
                rows.append(self._emit(row))
            return rows
        finally:
            self.mask_event_logging_s += perf_counter() - started

    def projection_events(self, projection, *, identity, availability=None, native_busy_causes=None):
        by_id = {c.candidate_id: c for c in projection.sampled_candidates}
        drops = {d.candidate_id: d for d in projection.dropped}
        rank = {c.candidate_id: i+1 for i, c in enumerate(sorted(projection.sampled_candidates,
            key=lambda c: (-c.raw_presence_logit, c.candidate_id)))}
        rows = []
        for candidate in projection.sampled_candidates:
            structural = candidate.candidate_id.startswith('structural|')
            self._count('structural_sampled_count' if structural else 'resource_sampled_count', identity)
            drop = drops.get(candidate.candidate_id)
            peers = []
            if drop is None:
                outcome, reason = 'selected_executable', None
                if structural:
                    self._count('structural_projected_count', identity)
            elif drop.code == 'RESOURCE_BUSY' and drop.conflicting_candidate_ids:
                outcome, reason = 'dropped_intra_step_conflict', drop.reason
                self._count('intra_step_conflict_drop_count', identity)
                for peer_id in drop.conflicting_candidate_ids:
                    peer = by_id[peer_id]
                    shared = set(candidate.claims).intersection(peer.claims)
                    from drl.IL.bc_v1.live_policy import claim_key
                    peers.append(dict(self._candidate(peer), shared_claims=tuple(sorted(claim_key(c) for c in shared)),
                        rule='reserved.intersection(claims)'))
            elif drop.code == 'RESOURCE_BUSY':
                outcome, reason = 'executor_rejected_busy', drop.reason
                self._count('executor_busy_rejection_count', identity)
                if availability is not None:
                    peers = availability.reasons[candidate.candidate_index].conflicts_with
                if native_busy_causes is not None:
                    peers = native_busy_causes.get(candidate.candidate_id, peers)
            else:
                outcome, reason = 'dropped_invalid', drop.reason
                if candidate.invalid_reason is None:
                    self._count('executor_other_rejection_count', identity)
            rows.append(self._emit(dict(identity, **self._candidate(candidate), sampled=True,
                availability_state='available_at_sampling', outcome=outcome, drop_reason=reason,
                rejection_code=drop.code if drop else None, stage='projection_or_physical_preflight',
                conflicts_with=peers, projection_score=candidate.raw_presence_logit,
                projection_rank=rank[candidate.candidate_id])))
        return rows

    def record_decision(self, env, index, decision, context, projection):
        from drl.MARL.policy import bind_marl_candidates
        identity = event_identity(env, index)
        key = (index, identity['generation'])
        # Bound lifecycle evidence to current generations; never alter executor.
        for old in [k for k in self.goals if k[0] == index and k[1] != key[1]]:
            del self.goals[old]
        self.terminal = {k for k in self.terminal if k[0] != index or k[1] == key[1]}
        self.counts_by_row[key] = defaultdict(int)
        records = bind_marl_candidates(routing=context.routing_snapshot,
            presence_logits=decision.action_set_prediction_template['presence_logits'],
            presence_threshold_logit=self.threshold)
        self.mask_events(records=records, availability=context.availability_snapshot,
            action=decision.action, identity=identity,
            presence_logits=decision.action_set_prediction_template['presence_logits'])
        busy_causes = {}
        executor = env.native_dispatcher.executor_provider(index)
        for drop in projection.dropped:
            if drop.code != 'RESOURCE_BUSY' or drop.conflicting_candidate_ids:
                continue
            if context.availability_snapshot is not None:
                modes = context.availability_snapshot.reasons[drop.candidate_index].resource_modes
                conflicts = executor.resource_claim_conflicts(dict(modes))
                goal_ids = {g for owners in conflicts.values() for g in owners}
                busy_causes[drop.candidate_id] = tuple(r for r in executor.active_resource_snapshot()
                    if r.goal_id in goal_ids)
        self.projection_events(projection, identity=identity, availability=context.availability_snapshot,
            native_busy_causes=busy_causes)
        self.pending[key] = {c.candidate_id: self._candidate(c) for c in projection.sampled_candidates}

    def register_goal(self, goal_id, metadata, identity):
        key = (identity['env_index'], identity['generation'], goal_id)
        if key in self.goals:
            return
        self.goals[key] = dict(metadata)
        self._count('primitive_started_count', identity)
        if metadata.get('candidate_id', '').startswith('structural|'):
            self._count('structural_started_count', identity)
        self._emit(dict(identity, **metadata, goal_id=goal_id, outcome='primitive_started',
            drop_reason=None, conflicts_with=[], sampled=True))

    def primitive_events(self, events, *, identity):
        for event in events:
            key = (identity['env_index'], identity['generation'], event.goal_id)
            if event.state not in ('succeeded', 'failed', 'canceled', 'rejected') or key in self.terminal:
                continue
            self.terminal.add(key)
            metadata = self.goals.get(key, dict(primitive=event.action, module_ids=(event.owner,),
                candidate_id=None, faces=(), connectors=(), resource_claims=(), attribution_missing=True))
            succeeded = event.state == 'succeeded'
            self._count('primitive_completed_count' if succeeded else 'primitive_failed_count', identity)
            if (metadata.get('candidate_id') or '').startswith('structural|'):
                self._count('structural_completed_count' if succeeded else 'structural_failed_count', identity)
            self._emit(dict(identity, **metadata, goal_id=event.goal_id,
                outcome='primitive_completed' if succeeded else 'primitive_failed',
                primitive_state=event.state, rejection_code=event.code,
                drop_reason=None if succeeded else event.code, conflicts_with=[]))

    def complete_step(self, *, outcome, primitive_events, identity):
        key = (identity['env_index'], identity['generation'])
        metadata = self.pending.pop(key, {})
        for admission in getattr(outcome, 'admission_events', ()):
            row = metadata.get(admission['candidate_id'], dict(candidate_id=admission['candidate_id']))
            if admission['accepted']:
                if admission.get('goal_id'):
                    self.register_goal(admission['goal_id'], row, identity)
                if admission['candidate_id'].startswith('structural|'):
                    self._count('structural_executed_count', identity)
            else:
                busy = admission['code'] == 'RESOURCE_BUSY'
                self._count('executor_busy_rejection_count' if busy else 'executor_other_rejection_count', identity)
                self._emit(dict(identity, **row, sampled=True,
                    outcome='executor_rejected_busy' if busy else 'executor_rejected_other',
                    drop_reason=admission['message'], rejection_code=admission['code'],
                    conflicts_with=admission.get('active_claims', ()), stage='native_submit'))
        self.primitive_events(primitive_events, identity=identity)
        counts = self.counts_by_row.pop(key, {})
        return {name: counts.get(name, 0) for name in COUNTERS}

    def summary(self):
        return dict({name: self.totals[name] for name in COUNTERS},
            row_transitions=self.row_transitions,
            mean_reward=self.reward_sum / self.row_transitions if self.row_transitions else None,
            valid_connection_max=self.valid_connection_max, C1_success_count=self.successes,
            by_task={task: dict(values) for task, values in self.by_task.items()},
            masked_detail_limit=self.masked_detail_limit,
            threshold_logit=self.threshold,
            mask_detail_payload_build_count=self.mask_detail_payload_build_count,
            mask_detail_descriptors_considered=self.mask_detail_descriptors_considered,
            mask_event_logging_s=self.mask_event_logging_s,
            feasibility_cache=self.feasibility_cache_metrics,
            **self.feasibility_cache_metrics.get('totals', {}))

    def record_transition_result(self, *, team_reward, valid_connections, success):
        self.row_transitions += 1
        self.reward_sum += float(team_reward)
        self.valid_connection_max = max(self.valid_connection_max, int(valid_connections))
        self.successes += int(bool(success))
        if self.directory:
            path = self.directory / 'action_summary.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.summary(), indent=2, allow_nan=False) + '\n')
            temporary.replace(path)
