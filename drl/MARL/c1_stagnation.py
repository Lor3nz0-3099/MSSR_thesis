"""Passive physical-frame stagnation evidence; never commands an executor."""
from dataclasses import dataclass
import math

from drl.MARL.c1_guided_feedback import _pair, _snapshot


@dataclass(frozen=True)
class StagnationStatus:
    option_stagnant_ids: tuple
    strategic: bool
    global_stagnant: bool
    terminate: bool
    reason: str | None
    no_useful_progress_s: float
    decision_opportunities: int
    protected_option_ids: tuple
    recovery_until_s: float
    handoff_relations: tuple = ()


class EpisodeStagnationMonitor:
    def __init__(self, settings):
        self.cfg = settings.values.get('stagnation', {})
        self.enabled = bool(self.cfg.get('enabled', False))
        self.started = None
        self.last_useful = None
        self.best_structural = -math.inf
        self.best_functional = -math.inf
        self.opportunities = 0
        self.relations = {}
        self.terminal_seen = set()
        self.recovery_until = -math.inf
        self.last_recovery_grant = -math.inf
        self.active_goal_metadata = {}
        self.preparation_seen = set()

    def step(self, frame, *, structural_score, functional_progress=0.,
             decision_opportunity=False, active_goals=None, action_goals=None,
             prepared_relations=()):
        now = float(frame.time_s)
        epsilon = float(self.cfg.get('progress_epsilon',1e-6))
        if self.started is None:
            self.started = self.last_useful = now
        if not all(math.isfinite(v) for v in (now, structural_score, functional_progress)):
            raise ValueError('Stagnation evidence must be finite')
        useful = (structural_score > self.best_structural + epsilon
                  or functional_progress > self.best_functional + epsilon)
        if useful:
            self.best_structural = max(self.best_structural, structural_score)
            self.best_functional = max(self.best_functional, functional_progress)
            self.last_useful = now
            self.opportunities = 0
            self.preparation_seen.clear()
        self.opportunities += int(bool(decision_opportunity))
        window = float(self.cfg.get('global_window_s', 25.))
        recovery = float(self.cfg.get('recovery_grace_s', 5.))
        for event in frame.primitive_events:
            if event.state not in ('failed', 'canceled', 'rejected') or event.goal_id in self.terminal_seen:
                continue
            self.terminal_seen.add(event.goal_id)
            # Repeated immediate failures cannot renew grace every frame.
            if now - self.last_recovery_grant >= window:
                self.recovery_until = now + recovery
                self.last_recovery_grant = now
                self.opportunities = 0
        # Physics can finish an ALIGN after this frame's policy decision.
        # Its successor DOCK has not had a chance yet. Observe both terminal
        # evidence and native prepared state, without directing the scheduler.
        ready = set()
        for relation in prepared_relations or ():
            pair = _pair(dict(primitive='align_faces', connectors=relation))
            if pair is not None:
                ready.add(pair)
        for event in frame.primitive_events:
            if event.state != 'succeeded' or event.goal_id in self.terminal_seen:
                continue
            self.terminal_seen.add(event.goal_id)
            if event.action != 'align_faces' or event.code != 'FACES_IN_CONTACT':
                continue
            metadata = (action_goals or {}).get(event.goal_id,
                self.active_goal_metadata.get(event.goal_id, {}))
            pair = _pair(metadata)
            if pair is not None:
                ready.add(pair)
        handoffs = tuple(sorted(ready - self.preparation_seen))
        if handoffs:
            self.preparation_seen.update(handoffs)
            self.recovery_until = max(self.recovery_until, now + recovery)
            # This frame's decision preceded completion/readiness. Count only
            # subsequent opportunities, even if the actor ran this frame.
            self.opportunities = 0
        self.active_goal_metadata = dict(active_goals or {})
        stagnant, protected = [], []
        for goal_id, metadata in (active_goals or {}).items():
            pair = _pair(metadata)
            key = (metadata.get('primitive'), pair or tuple(sorted(metadata.get('module_ids', ()))))
            item = self.relations.setdefault(key, dict(start=now, last_progress=now,
                                                       best_distance=math.inf, poses=set()))
            snapshot = _snapshot(frame, pair) if pair else None
            if snapshot and snapshot['distance_m'] < item['best_distance'] - float(self.cfg.get('option_distance_improvement_m',.001)):
                item['best_distance'] = snapshot['distance_m']
                item['last_progress'] = now
            owners = set(metadata.get('module_ids', ())) or {p[0] for p in pair or ()}
            position_q=float(self.cfg.get('position_quantization_m',.015))
            orientation_q=float(self.cfg.get('orientation_quantization',.025))
            joint_q=float(self.cfg.get('joint_quantization_rad',.05))
            pose = tuple((m.module_id, *(round(v/position_q) for v in m.position),
                          *(round(v/orientation_q) for v in m.orientation),
                          *(round((math.remainder(v,2*math.pi) if n=='pan.position_rad' else v)/joint_q)
                            for n, v in m.joints if n in ('pan.position_rad', 'tilt.position_rad')))
                         for m in sorted(frame.modules, key=lambda m: m.module_id) if m.module_id in owners)
            if pose and pose not in item['poses']:
                item['poses'].add(pose)
                item['last_progress'] = now
            elapsed = now - item['last_progress']
            if elapsed >= float(self.cfg.get('option_window_s', 10.)):
                stagnant.append(goal_id)
            # First-attempt preparation may orient or stage away from contact.
            # New IDs for the same relation cannot renew this bounded grace.
            if (now - item['start'] < float(self.cfg.get('option_grace_s', 32.))
                    or elapsed < recovery):
                protected.append(goal_id)
        idle = now - self.last_useful
        strategic = idle >= float(self.cfg.get('strategic_window_s', 20.))
        global_stagnant = idle >= window
        terminate = (self.enabled and global_stagnant and not protected
                     and now >= self.recovery_until
                     and self.opportunities >= int(self.cfg.get('min_decision_opportunities', 3)))
        return StagnationStatus(tuple(stagnant), self.enabled and strategic,
            self.enabled and global_stagnant, bool(terminate), 'stagnation' if terminate else None,
            idle, self.opportunities, tuple(protected), max(now, self.recovery_until), handoffs)
