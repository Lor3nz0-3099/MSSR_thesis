"""Sparse viewer observer. Native executor calls are delegated exactly once.

Keep native status objects at 240Hz; serialize only real events or slow status.
The scientific evaluator, policy, mask and primitive code are untouched.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time
from drl.MARL.c1_primitive_lifecycle import PrimitiveLifecycleRecorder, value
from drl.MARL.c1_native_runtime import sanitized_record


class CompactEventWriter:
    def __init__(self, path):
        self.handle = Path(path).open('x', buffering=65536)

    def append(self, row):
        self.handle.write(json.dumps(sanitized_record(row), separators=(',', ':'), allow_nan=False) + '\n')

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()


class ViewerEvents(PrimitiveLifecycleRecorder):
    def __init__(self, directory, task, *, classifier, console=None, status_seconds=2., low_latency=False):
        super().__init__(directory, (task,))
        self.classifier = classifier
        self.console = console if console is not None else sys.stdout
        self.writer = CompactEventWriter(self.directory/'events.jsonl')
        self.status_seconds = status_seconds
        self.low_latency = bool(low_latency)
        self.next_status = 0.
        self.latest_connections = ()
        self.observed_generation = None
        self.latest_mask = {}
        self.align_diagnostics = {}
        self.event_counts = Counter()

    @staticmethod
    def candidate_id(row):
        source, target = row.get('source_module'), row.get('target_module')
        return (f"structural|{row['primitive']}|{source}|"
                f"{source}:{row.get('source_face')}>{target}:{row.get('target_face')}")

    def _write(self, label, row):
        event = {'schema':'mssr.c1.viewer_event.v1', **row, 'event':label}
        self.writer.append(event)
        self.event_counts[label] += 1
        parts = [label, f"sim={event.get('timestamp_s', 0.):.6f}s"]
        if event.get('source_module'):
            parts.append(f"{event['source_module']}:{event.get('source_face')} -> "
                         f"{event.get('target_module')}:{event.get('target_face')}")
        if event.get('failure_code'):
            parts += [str(event['failure_code']), f"candidate={event.get('candidate_id')}"]
        diagnostic = event.get('relation_diagnostic')
        if diagnostic:
            parts.append(diagnostic['classification'])
        print(' | '.join(parts), file=self.console, flush=True)
        if not self.low_latency:
            self.writer.flush()  # Durable diagnostics; GUI low-latency mode buffers until close.

    def emit(self, name, row):
        # The parent observer's phase stream is deliberately unused here.
        row = dict(row)
        latest = row.pop('last_status', None)
        if latest is not None:
            row['phase'] = latest.phase
        if name == 'primitive_lifecycle':
            primitive = row['primitive']
            prefix = {'align_faces':'ALIGN', 'dock':'DOCK', 'undock':'UNDOCK'}.get(primitive, primitive.upper())
            event = row['event']
            if row.get('terminal_code') in ('TIMEOUT', 'CONTACT_TIMEOUT') and event == 'FAILED':
                event = 'TIMEOUT'
                key = (row['env_index'], row['generation'], row['goal_id'])
                self.goals[key]['terminal_state'] = 'TIMEOUT'
                row['terminal_state'] = 'TIMEOUT'
            row['failure_code'] = row.get('terminal_code') if event in ('FAILED','TIMEOUT','CANCELLED') else None
            row['candidate_id'] = self.candidate_id(row)
            diagnostic = self.align_diagnostics.get((row['env_index'], row['generation'], row['goal_id']))
            if diagnostic is not None:
                row['relation_diagnostic'] = diagnostic
            self._write(f'{prefix} {event}', row)
            if event == 'START':
                self._write('CLAIMS ACQUIRED', row)
            elif row.get('claims_released'):
                self._write('CLAIMS RELEASED', row)
            if primitive == 'align_faces' and event in ('COMPLETE','FAILED','TIMEOUT','CANCELLED'):
                self.classifier.record_attempt(((row['source_module'],row['source_face']),
                    (row['target_module'],row['target_face'])), row['generation'], row['failure_code'], row['timestamp_s'])
        elif name == 'mask_reopen':
            if row.get('state') == 'reopened':
                self._write('MASK REOPEN', row)
        elif name == 'valid_connections':
            self._write('VALID CONNECTION', row)
        else:
            self._write(name.upper().replace('_',' '), row)

    def attach_executor(self, executor, index, generation):
        if getattr(executor, '_c1_lifecycle_recorder', None) is self:
            return
        super().attach_executor(executor, index, generation)
        delegated_submit = executor.submit
        original_step = executor._c1_lifecycle_originals[2]

        def submit(goal, now_s):
            epoch = generation() if callable(generation) else generation
            primitive = value(goal.primitive)
            if primitive == 'align_faces':
                pair = ((goal.module_ids[0],goal.parameters.get('face_a')),
                        (goal.module_ids[1],goal.parameters.get('face_b')))
                self.align_diagnostics[index,epoch,goal.goal_id] = self.classifier.classify(
                    pair, self.latest_connections, epoch)
            status = delegated_submit(goal, now_s)
            if value(status.state) == 'rejected':
                row = dict(env_index=index,generation=epoch,goal_id=goal.goal_id,primitive=primitive,
                    source_module=goal.module_ids[0],source_face=goal.parameters.get('face_a'),
                    target_module=goal.module_ids[1] if len(goal.module_ids)>1 else None,
                    target_face=goal.parameters.get('face_b'),timestamp_s=status.stamp_s,
                    failure_code=status.code,message=status.message)
                row['candidate_id'] = self.candidate_id(row)
                self._write(f"{'ALIGN' if primitive=='align_faces' else primitive.upper()} REJECTED", row)
            return status

        def step(now_s):
            result = original_step(now_s)
            epoch = generation() if callable(generation) else generation
            for status in result.statuses:
                row = self.goals.get((index,epoch,status.goal_id))
                if row is not None:
                    row['last_status'] = status
            return result

        executor.submit, executor.step = submit, step

    def observe_context(self, index, generation, records, availability, stamp, **kwargs):
        if self.observed_generation is not None and generation != self.observed_generation:
            # The validated loose reset removes every joint. A terminal frame
            # from the previous episode must not classify the next episode.
            self.latest_connections = ()
            self._write('RESET', dict(env_index=index,generation=generation,timestamp_s=stamp,
                                     previous_generation=self.observed_generation))
        self.observed_generation = generation
        super().observe_context(index,generation,records,availability,stamp,**kwargs)
        counts = Counter(r.code for r in availability.reasons)
        self.latest_mask = dict(available=counts['available'],busy=counts['masked_busy'],
                                unavailable=counts['masked_unavailable'],
                                not_ready=counts['masked_not_ready'] + counts['masked_handoff_not_ready'],
                                handoff_wait=counts['masked_handoff_wait'],
                                prepared_conflict=counts['masked_prepared_conflict'])

    def observe_transition(self, row, frame=None):
        if frame is not None:
            self.latest_connections = frame.connections
        super().observe_transition(row, frame)

    def slow_status(self, stamp, *, now=None, force=False):
        now = time.monotonic() if now is None else now
        if not force and now < self.next_status:
            return None
        self.next_status = now + self.status_seconds
        active = []
        for row in self.goals.values():
            if row['terminal_state'] is None:
                status = row['last_status']
                active.append(dict(primitive=row['primitive'],candidate_id=self.candidate_id(row),
                    source=f"{row['source_module']}:{row['source_face']}",
                    target=f"{row['target_module']}:{row['target_face']}",
                    phase=status.phase if status is not None else 'accepted',
                    age_s=float(stamp)-row['start_timestamp_s']))
        result = dict(timestamp_s=float(stamp),connections=sum(c.joint_valid for c in self.latest_connections),
                      active=active,mask=self.latest_mask,events=dict(self.event_counts))
        if not self.low_latency:
            temporary=self.directory/'progress.json.tmp'
            temporary.write_text(json.dumps(sanitized_record(result),separators=(',',':'),allow_nan=False)+'\n')
            temporary.replace(self.directory/'progress.json')
        lines=[f"sim={stamp:.3f}s connections={result['connections']}/7"]
        for a in active:
            lines.append(f"ACTIVE {a['primitive']} {a['source']} -> {a['target']} phase={a['phase']} age={a['age_s']:.2f}s")
        lines.append('mask: '+' '.join(f'{k}={v}' for k,v in self.latest_mask.items()))
        print('\n'.join(lines),file=self.console,flush=True)
        if not self.low_latency:
            self.writer.flush()
        return result

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
