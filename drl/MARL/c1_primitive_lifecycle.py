"""Passive, diagnostic-only native lifecycle evidence; never submits a goal.

Wrappers delegate each call exactly once. No random numbers, physics reads,
policy control, mutation of claims, or speculative admission checks.
"""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path


def value(item):
    return getattr(item, 'value', item)


class PrimitiveLifecycleRecorder:
    def __init__(self, directory, tasks):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tasks = tuple(tasks)
        self.decision_step = 0
        self.goals = {}
        self.blocked = {}
        self.pending = {}
        self.counters = {i: Counter() for i in range(len(tasks))}
        self.valid_connections = {i: 0 for i in range(len(tasks))}
        self.valid_seen = []
        self.errors = []

    def emit(self, name, row):
        from drl.MARL.c1_native_runtime import sanitized_record
        with (self.directory / f'{name}.jsonl').open('a') as handle:
            handle.write(json.dumps(sanitized_record(row), allow_nan=False) + '\n')

    def attach_executor(self, executor, index, generation):
        if getattr(executor, '_c1_lifecycle_recorder', None) is self:
            return
        executor._c1_lifecycle_recorder = self
        original_submit, original_finish, original_step = executor.submit, executor._finish, executor.step
        executor._c1_lifecycle_originals = (original_submit, original_finish, original_step)
        rows = {}

        def submit(goal, now_s):
            status = original_submit(goal, now_s)
            if value(status.state) != 'accepted':
                return status
            epoch = generation() if callable(generation) else generation
            key = (index, epoch, goal.goal_id)
            acquired = next(item for item in executor.active_resource_snapshot() if item.goal_id == goal.goal_id)
            claims = dict(acquired.resource_modes)
            row = dict(env_index=index, generation=epoch, task=self.tasks[index], decision_step=self.decision_step,
                goal_id=goal.goal_id, primitive=value(goal.primitive), module_ids=list(goal.module_ids),
                source_module=goal.module_ids[0], source_face=goal.parameters.get('face_a'),
                target_module=goal.module_ids[1] if len(goal.module_ids) > 1 else None,
                target_face=goal.parameters.get('face_b'), parameters=dict(goal.parameters),
                start_timestamp_s=float(status.stamp_s), terminal_timestamp_s=None, terminal_state=None,
                terminal_code=None, duration_s=None, timeout_s=goal.timeout_s, claims=claims,
                acquire_claim_version=executor.active_claim_version, claims_released=False,
                mask_after_release=None, censored=False, phase_event_count=0, last_status=None)
            rows[goal.goal_id] = row
            self.goals[key] = row
            self.counters[index]['claim_acquire_goal_count'] += 1
            self.counters[index]['claim_acquire_resource_count'] += len(claims)
            self.emit('primitive_lifecycle', dict(row, event='START', timestamp_s=status.stamp_s))
            return status

        def finish(runtime, state, now_s, code, message):
            row = rows.get(runtime.goal.goal_id)
            execution_start = runtime.execution_started_at_s
            status = original_finish(runtime, state, now_s, code, message)
            if row is not None and row['terminal_state'] is None:
                active = executor.active_resource_snapshot()
                released = all(item.goal_id != row['goal_id'] for item in active)
                label = ('COMPLETE' if value(state) == 'succeeded' else 'TIMEOUT' if code == 'TIMEOUT'
                         else 'CANCELLED' if value(state) == 'canceled' else 'FAILED')
                row.update(terminal_timestamp_s=float(status.stamp_s), terminal_state=label,
                    native_terminal_state=value(state), terminal_code=code, terminal_message=message,
                    terminal_decision_step=self.decision_step,
                    duration_s=float(status.stamp_s) - row['start_timestamp_s'],
                    execution_started_at_s=execution_start,
                    execution_duration_s=None if execution_start is None else float(status.stamp_s)-execution_start,
                    release_claim_version=executor.active_claim_version, claims_released=released,
                    remaining_claim_holders=[asdict(item) for item in active])
                self.counters[index]['claim_release_goal_count'] += int(released)
                self.counters[index]['claim_release_resource_count'] += len(row['claims']) if released else 0
                key = (index, row['generation'], row['goal_id'])
                self.pending[key] = dict(self.blocked.get(key, {}))
                self.emit('primitive_lifecycle', dict(row, event=label, timestamp_s=status.stamp_s))
                if not released:
                    self.errors.append(dict(env_index=index, goal_id=row['goal_id'], error='native_claim_release_missing'))
            return status

        def step(now_s):
            result = original_step(now_s)
            for status in result.statuses:
                row = rows.get(status.goal_id)
                if row is None:
                    continue
                last = row['last_status']
                # Retain the latest native progress/feedback even when phase detail is bounded.
                row['last_status'] = status.to_dict()
                if last is None or last['phase'] != status.phase:
                    row['phase_event_count'] += 1
                    if row['phase_event_count'] <= 64:
                        self.emit('primitive_phases', dict(env_index=index, generation=row['generation'],
                            decision_step=self.decision_step, **status.to_dict()))
            return result

        executor.submit, executor._finish, executor.step = submit, finish, step

    def detach_executor(self, executor):
        if getattr(executor, '_c1_lifecycle_recorder', None) is self:
            executor.submit, executor._finish, executor.step = executor._c1_lifecycle_originals
            del executor._c1_lifecycle_originals
            del executor._c1_lifecycle_recorder

    def attach_environment(self, env):
        original = env._ensure_native_actuation_backends

        def attach():
            for i, backend in enumerate(env.physical_expert_backends):
                if backend is not None:
                    self.attach_executor(backend['primitive_executor'], i, lambda i=i: env.generation[i])

        def ensure(*args, **kwargs):
            result = original(*args, **kwargs)
            attach()
            return result

        env._ensure_native_actuation_backends = ensure
        self._attached_environment = (env, original)
        attach()

    def detach_environment(self):
        if hasattr(self, '_attached_environment'):
            env, original = self._attached_environment
            env._ensure_native_actuation_backends = original
            for backend in env.physical_expert_backends:
                if backend is not None:
                    self.detach_executor(backend['primitive_executor'])
            del self._attached_environment

    def observe_context(self, index, generation, records, availability, stamp, *, count_masks=True):
        if availability is None:
            raise ValueError('Diagnostic requires native temporal feasibility support')
        current = {record.candidate_id: availability.reasons[record.candidate_index] for record in records}
        # A cached availability is an immutable snapshot, not a new admission.
        for key in [key for key in self.blocked if key[0] == index]:
            del self.blocked[key]
        for record in records:
            reason = availability.reasons[record.candidate_index]
            if count_masks and record.candidate_kind == 'structural' and reason.code == 'masked_busy':
                self.counters[index]['structural_busy_masked_count'] += 1
            for holder in reason.conflicts_with:
                key = (index, generation, holder.goal_id)
                self.blocked.setdefault(key, {})[record.candidate_id] = reason.resource_modes
        for key, blocked in list(self.pending.items()):
            if key[0] != index:
                continue
            row = self.goals[key]
            if key[1] != generation:
                finding = dict(state='reset_epoch_changed', reopened_count=0, timestamp_s=stamp)
            else:
                reopened = [cid for cid in blocked if cid in current and current[cid].code == 'available']
                busy = [cid for cid in blocked if cid in current and current[cid].code == 'masked_busy']
                missing = [cid for cid in blocked if cid not in current]
                finding = dict(state='reopened' if reopened else 'not_observed_busy_before_release' if not blocked
                    else 'still_busy' if busy else 'catalog_or_other_mask_changed', timestamp_s=stamp,
                    blocked_candidate_count=len(blocked), reopened_count=len(reopened), still_busy_count=len(busy),
                    missing_from_catalog_count=len(missing), reopened_examples=reopened[:16],
                    still_busy_examples=busy[:16], missing_examples=missing[:16])
            previous = row['mask_after_release']
            changed = previous is None or any(previous.get(k) != v for k, v in finding.items() if k != 'timestamp_s')
            if changed:
                row['mask_after_release'] = finding
                self.emit('mask_reopen', dict(env_index=index, generation=key[1], goal_id=key[2],
                    primitive=row['primitive'], release_timestamp_s=row['terminal_timestamp_s'],
                    release_claim_version=row['release_claim_version'], **finding))
            if finding['state'] == 'reopened':
                self.counters[index]['mask_reopen_after_release_count'] += 1
            if finding['state'] in ('reopened', 'reset_epoch_changed', 'not_observed_busy_before_release'):
                del self.pending[key]

    def observe_transition(self, row, frame=None):
        index = row['env_index']
        self.counters[index].update(row['action_counts'])
        self.counters[index]['transitions'] += 1
        self.valid_connections[index] = max(self.valid_connections[index], row['assessment']['valid_connections'])
        if frame is not None:
            if frame.pathological_collision_owner_ids:
                self.counters[index]['collision_observation_count'] += 1
            if not frame.evidence_complete:
                self.counters[index]['incomplete_physical_evidence_count'] += 1
            for joint in frame.connections:
                if joint.joint_valid:
                    pair = sorted(joint.endpoints)
                    identity = (index, row['generation'], joint.joint_id)
                    previous = next((v for v in self.valid_seen if (v['env_index'], v['generation'], v['joint_id']) == identity), None)
                    if previous is None:
                        entry = dict(env_index=index, generation=row['generation'], joint_id=joint.joint_id,
                            timestamp_s=frame.time_s, last_observed_timestamp_s=frame.time_s, pair=pair)
                        self.valid_seen.append(entry)
                        self.emit('valid_connections', entry)
                    else:
                        previous['last_observed_timestamp_s'] = frame.time_s
        if not row['assessment']['safe'] or (row.get('terminated') and not row.get('success')):
            self.counters[index]['physical_error_observation_count'] += 1
            self.emit('physical_errors', dict(env_index=index, generation=row['generation'],
                decision_step=self.decision_step, timestamp_s=row['timestamp_s'], reason=row.get('reason'),
                assessment=row['assessment'], collision_owner_ids=list(frame.pathological_collision_owner_ids) if frame else None))

    def censor_open(self, stamp, reason):
        for row in self.goals.values():
            if row['terminal_state'] is None:
                row.update(censored=True, observation_end_timestamp_s=float(stamp), censor_reason=reason)
                self.emit('primitive_lifecycle', dict(row, event='OBSERVATION_CUTOFF', timestamp_s=float(stamp)))

    @staticmethod
    def pair(row):
        return sorted(((row['source_module'], row['source_face']), (row['target_module'], row['target_face'])))

    def summary(self):
        from drl.MARL.c1_action_events import COUNTERS
        goals = list(self.goals.values())
        by_env = {}
        for i, task in enumerate(self.tasks):
            selected = [g for g in goals if g['env_index'] == i]
            names = (*COUNTERS, 'claim_acquire_goal_count', 'claim_acquire_resource_count',
                'claim_release_goal_count', 'claim_release_resource_count',
                'mask_reopen_after_release_count', 'structural_busy_masked_count',
                'collision_observation_count', 'incomplete_physical_evidence_count',
                'physical_error_observation_count', 'transitions')
            metrics = dict({name: self.counters[i][name] for name in names}, task=task,
                valid_connections_max=self.valid_connections[i])
            metrics.update(structural_proposed_count=metrics['structural_sampled_count'])
            for primitive in ('align_faces', 'dock', 'undock'):
                rows = [g for g in selected if g['primitive'] == primitive]
                metrics[primitive] = {state: sum(g['terminal_state'] == state for g in rows)
                    for state in ('COMPLETE', 'FAILED', 'TIMEOUT', 'CANCELLED')}
                metrics[primitive].update(START=len(rows), OPEN=sum(g['terminal_state'] is None for g in rows))
            durations = [g['duration_s'] for g in selected if g['primitive'] == 'align_faces' and g['terminal_state'] == 'COMPLETE']
            metrics['align_complete_duration_s'] = dict(min=min(durations), mean=sum(durations)/len(durations), max=max(durations)) if durations else dict(min=None, mean=None, max=None)
            by_env[str(i)] = metrics
        aligns = [g for g in goals if g['primitive'] == 'align_faces']
        completed = [g for g in aligns if g['terminal_state'] == 'COMPLETE']
        chains = []
        for align in completed:
            reopening = align['mask_after_release'] or {}
            if not align['claims_released'] or reopening.get('state') != 'reopened':
                continue
            for dock in goals:
                if (dock['primitive'] == 'dock' and dock['env_index'] == align['env_index']
                        and dock['generation'] == align['generation'] and self.pair(dock) == self.pair(align)
                        and dock['start_timestamp_s'] >= reopening['timestamp_s']):
                    valid = [v for v in self.valid_seen if v['env_index'] == dock['env_index'] and v['generation'] == dock['generation']
                        and v['pair'] == self.pair(dock) and dock['terminal_timestamp_s'] is not None and v['last_observed_timestamp_s'] >= dock['terminal_timestamp_s']]
                    chains.append(dict(env_index=dock['env_index'], generation=dock['generation'], align_goal=align['goal_id'],
                        dock_goal=dock['goal_id'], dock_completed=dock['terminal_state'] == 'COMPLETE', valid_joint=valid))
        terminal = any(g['terminal_state'] is not None for g in aligns)
        completed_verified = bool(completed) and all(g['claims_released'] and (g['mask_after_release'] or {}).get('state') == 'reopened' for g in completed)
        return dict(by_env=by_env, goals=goals, valid_connections=self.valid_seen, errors=self.errors,
            success_criteria=dict(align_real_terminal=terminal, completed_align_release_reopen_verified=completed_verified,
                minimum=terminal and (not completed or completed_verified), strong=bool(chains),
                ideal=any(c['dock_completed'] and c['valid_joint'] for c in chains)), chains=chains,
            interpretation=('Observed native terminal; inspect release/reopen and chain flags.' if terminal else
                'No real align terminal observed: inspect proposed/projected/executed, resets, active goal phases and cutoff.'))
