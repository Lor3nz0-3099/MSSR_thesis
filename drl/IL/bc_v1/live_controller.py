"""Small wall-clock BC loop. No robot, simulator, torch or teacher dependency.

Inference is the injected frozen L1b/H1/H2/L1a path. The transport performs
final owner-side admission; publishing a proposal is not an admission ACK.
Only tick's control thread owns leases. At most one inference is in flight.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time
from collections.abc import Mapping


def require_finite(value):
    if isinstance(value, Mapping):
        for item in value.values(): require_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value: require_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite physical command')


@dataclass(frozen=True)
class PhysicalSnapshot:
    epoch: str
    stamp: float
    payload: Mapping

    def __post_init__(self):
        if not self.epoch or type(self.stamp) not in (int, float) or not math.isfinite(self.stamp):
            raise ValueError('Invalid physical snapshot identity')


@dataclass(frozen=True)
class RuntimeCommand:
    key: tuple[str, str]
    payload: Mapping
    claims: tuple[str, ...]
    continuous: bool

    def __post_init__(self):
        if len(self.key) != 2 or not all(self.key) or not self.claims:
            raise ValueError('Missing runtime command resource/claims')
        require_finite(self.payload)


@dataclass(frozen=True)
class RuntimeDecision:
    """BC-only withdrawal keys, separate from physical/native commands."""
    commands: tuple[RuntimeCommand,...]
    withdrawals: tuple[tuple[str,str],...] = ()

    def __iter__(self):
        # Existing proposal safety wrappers inspect physical commands; returning
        # this object unchanged retains explicit withdrawals for the controller.
        return iter(self.commands)

    def __post_init__(self):
        if (not isinstance(self.commands,tuple) or any(not isinstance(c,RuntimeCommand) for c in self.commands)
                or not isinstance(self.withdrawals,tuple) or any(not isinstance(k,tuple) or len(k)!=2
                    or any(not isinstance(v,str) or not v for v in k) for k in self.withdrawals)
                or len(set(self.withdrawals))!=len(self.withdrawals)
                or set(self.withdrawals).intersection(c.key for c in self.commands)):
            raise ValueError('Invalid/conflicting resource withdrawal')


@dataclass(frozen=True)
class HeldCommand:
    command: RuntimeCommand
    generation: str
    source_stamp: float
    decision_deadline: float | None = None


class LiveController:
    def __init__(self, session, epoch, infer, transport, *, clock=time.monotonic, event_sink=None):
        if not session or not epoch: raise ValueError('Missing session/epoch')
        self.session, self.epoch = session, epoch
        self.infer, self.transport, self.clock = infer, transport, clock
        self.events = []; self.event_sink = event_sink
        self.armed = False; self.failed = False; self.held = {}
        self.latest = None; self.last_progress = None; self.last_inference_stamp = None
        self.inflight = None; self._flight = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='bc-inference')
        self._next_policy = 0.; self._next_refresh = 0.; self._index = 0
        self._previous_wall = None

    def _event(self, event, **fields):
        row = dict(event=event, wall_monotonic=self.clock(), **fields)
        self.events.append(row)
        if self.event_sink is not None: self.event_sink(row)

    def fail(self, reason):
        if self.failed: return
        self.failed = True; self.armed = False; self.held.clear()
        self._event('failed', reason=str(reason))
        try: self.transport.clear(str(reason))
        except Exception as error: self._event('clear_transport_error', reason=str(error))

    def _proof(self):
        return dict(session=self.session, epoch=self.epoch,
            state_stamp=self.latest.stamp, state_deadline=self.last_progress+1.)

    def _publish(self, goals=()):
        self.transport.publish(self.held, goals, self._proof())
        self._next_refresh = self.clock()+.1
        self._event('holder_refresh' if self.held else 'state_lease_refresh', generations=sorted({v.generation for v in self.held.values()}),
            held_count=len(self.held), goal_count=len(goals))

    def tick(self, snapshot=None):
        now = self.clock()
        if not math.isfinite(now) or self._previous_wall is not None and now < self._previous_wall:
            self.fail('monotonic clock regression'); return
        self._previous_wall = now
        if self.failed: return
        # Expire before accepting a later packet: first smoke never auto-rearms.
        if self.last_progress is not None and now-self.last_progress >= 1.:
            self.fail('state stale >= 1.0 s'); return
        try:
            if snapshot is not None:
                if snapshot.epoch != self.epoch:
                    self.fail('session/epoch mismatch'); return
                if self.latest is not None and snapshot.stamp < self.latest.stamp:
                    self.fail('physical stamp regression'); return
                if self.latest is None:
                    self.latest = snapshot; self.last_progress = now
                    self._event('startup_baseline', stamp=snapshot.stamp)
                elif snapshot.stamp > self.latest.stamp:
                    self.latest = snapshot; self.last_progress = now
                    if not self.armed:
                        self.armed = True; self._next_refresh=now+.1
                        self._event('authority_armed', stamp=snapshot.stamp)
                else:
                    self._event('duplicate_stamp_skip', stamp=snapshot.stamp)
            feedback = self.transport.poll()
            dirty = False
            for key, generation in feedback.get('invalidated', ()):
                key = tuple(key)
                if key in self.held and self.held[key].generation == generation:
                    self.held.pop(key); dirty = True
                    self._event('holder_revoked', key=key, generation=generation)
            if feedback.get('failure'):
                self.fail(feedback['failure']); return
            if self.inflight is not None and self.inflight.done():
                future = self.inflight; self.inflight = None
                generation, source, started = self._flight
                commands = future.result()
                if now-started >= 1. or source.epoch != self.epoch:
                    self.fail('stale inference result'); return
                goals = []; busy = set(feedback.get('busy', ()))
                if isinstance(commands,RuntimeDecision):
                    for key in commands.withdrawals:
                        old=self.held.get(key)
                        if old is not None and old.command.continuous and not busy.intersection(old.command.claims):
                            self.held.pop(key);dirty=True
                            self._event('bc_resource_withdrawn',key=key,generation=old.generation)
                    commands=commands.commands
                for command in commands:
                    if not isinstance(command, RuntimeCommand): raise ValueError('Invalid runtime proposal')
                    require_finite(command.payload)
                    if busy.intersection(command.claims):
                        self._event('proposal_rejected', generation=generation, key=command.key, reason='BUSY')
                        continue
                    lease = HeldCommand(command,generation,source.stamp,started+1.)
                    if command.continuous:
                        self.held[command.key] = lease
                    else:
                        # Native completion never reinstates a retired continuous.
                        for key, old in tuple(self.held.items()):
                            if set(old.command.claims).intersection(command.claims): self.held.pop(key)
                        goals.append(lease)
                if self.held or goals or dirty: self._publish(tuple(goals)); dirty=False
                self._event('decision_completed', generation=generation, stamp=source.stamp,
                    inference_path_s=now-started, proposals=len(commands))
                self._next_policy=max(self._next_policy,now+.1)
                return
            if dirty or self.armed and now >= self._next_refresh:
                self._publish()
            if not self.armed or now < self._next_policy: return
            self._next_policy = now+.1  # no queued ticks or catch-up bursts
            if self.inflight is not None:
                self._event('inference_busy_skip'); return
            if self.last_inference_stamp is not None and self.latest.stamp <= self.last_inference_stamp: return
            self.last_inference_stamp = self.latest.stamp
            self._index += 1; generation=f'bc:{self.session}:{self._index}'
            source=self.latest; self._flight=(generation,source,now)
            self.inflight=self._worker.submit(self.infer,source,feedback)
            self._event('inference_started',generation=generation,stamp=source.stamp)
        except Exception as error:
            self.fail(error)

    def close(self):
        if self.held: self.transport.clear('controller closed'); self.held.clear()
        self._worker.shutdown(wait=True, cancel_futures=True)
