"""Optional offline/native probes. Normal runner execution never imports this.

Host intervals are inclusive enqueue time. CUDA events measure the current Torch
stream only; neither metric claims exclusive GPU/PhysX time. Report flush is the
only synchronization introduced by optional CUDA events.
"""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import time

import torch


_ABSENT = object()
DECISION_REASONS = ('continuous_resource_control', 'prepared_handoff',
                    'idle_fallback', 'structural_decision')


def decision_reason(*, decide, mode, continuous_control, force_decision, state):
    """Label the branch already taken; never ask the stateful clock again.

    structural_decision means an initial/event/wave-fill wake-up (or fixed
    clock), not proof that the sampled ActionSet contains a structural action.
    """
    if not decide:
        return None
    if mode == 'fixed':
        return 'structural_decision'
    if continuous_control:
        return 'continuous_resource_control'
    if force_decision:
        return 'structural_decision'
    if not state.active_goal_ids and any(row and not bool(row[-1]) for row in state.prepared_relations):
        return 'prepared_handoff'
    return 'idle_fallback'


class PerformanceObserver:
    def __init__(self, device, *, cuda_events=False, max_event_pairs=20000, decision_sink=None):
        self.device = torch.device(device)
        self.cuda_events = bool(cuda_events and self.device.type == "cuda")
        self.max_event_pairs = max_event_pairs
        self.calls = defaultdict(int)
        self.rows = defaultdict(int)
        self.host_s = defaultdict(float)
        self.events = []
        self.event_pairs_dropped = 0
        self.explicit_cuda_synchronizations = 0
        self._undo = []
        self._phase = "rollout"
        self._cuda_ms = None
        self._original_synchronize = None
        self.decision_sink = decision_sink
        self.decision_reason_counts = dict.fromkeys(DECISION_REASONS, 0)

    def _attach_decision_diagnostics(self, provider):
        controller = getattr(provider, '_structural_options', None)
        clocks = getattr(controller, '_clocks', ())
        if not clocks:
            return
        env = provider.native_env
        for scheduler_type in {type(clock.scheduler) for clock in clocks}:
            original = scheduler_type.should_decide
            @wraps(original)
            def observed(scheduler, state, _original=original):
                force = scheduler._force_decision
                continuous = scheduler._continuous_control
                idle = scheduler._idle_frames
                # Exactly one original call: should_decide consumes force/idle.
                result = _original(scheduler, state)
                index = next((i for i, clock in enumerate(controller._clocks)
                              if clock.scheduler is scheduler), None)
                if index is not None:
                    reason = decision_reason(decide=result, mode=scheduler.mode,
                        continuous_control=continuous, force_decision=force, state=state)
                    if reason is not None:
                        self.decision_reason_counts[reason] += 1
                    if self.decision_sink is not None:
                        frame = int(getattr(env, 'common_step_counter', 0))
                        dt = float(getattr(env, 'step_dt', 1/30))
                        steps = getattr(env, 'marl_steps', None)
                        handoffs = getattr(env, 'handoffs', ())
                        self.decision_sink(dict(env_index=index, generation=controller._clocks[index].generation,
                            vector_frame=frame+1, decision_time_s=(steps[index] if steps is not None else frame)*dt,
                            decide=bool(result), reason=reason, mode=scheduler.mode,
                            policy_allowed=bool(result and (not handoffs or handoffs[index].marl_active)),
                            continuous_control=bool(continuous), force_decision_before=bool(force),
                            idle_frames_before=int(idle), active_goal_ids=state.active_goal_ids,
                            prepared_relations=state.prepared_relations, connection_count=state.connection_count))
                return result
            self._set(scheduler_type, 'should_decide', observed)

    def __enter__(self):
        self._original_synchronize = torch.cuda.synchronize
        original = self._original_synchronize
        def synchronize(*args, **kwargs):
            self.explicit_cuda_synchronizations += 1
            return original(*args, **kwargs)
        self._set(torch.cuda, "synchronize", synchronize)
        return self

    def __exit__(self, *_):
        for obj, name, old in reversed(self._undo):
            if old is _ABSENT:
                delattr(obj, name)
            else:
                setattr(obj, name, old)
        self._undo.clear()

    def _set(self, obj, name, value):
        self._undo.append((obj, name, vars(obj).get(name, _ABSENT)))
        setattr(obj, name, value)

    @contextmanager
    def scope(self, name, *, phase=None):
        old_phase = self._phase
        if phase is not None:
            self._phase = phase
        self.calls[name] += 1
        start = time.perf_counter()
        pair = None
        if self.cuda_events:
            if len(self.events) < self.max_event_pairs:
                with torch.cuda.device(self.device):
                    pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    pair[0].record()
                self.events.append((name, pair))
            else:
                self.event_pairs_dropped += 1
        try:
            yield
        finally:
            if pair is not None:
                with torch.cuda.device(self.device):
                    pair[1].record()
            self.host_s[name] += time.perf_counter() - start
            self._phase = old_phase

    def wrap(self, obj, method, name, *, phase=None, phase_prefix=False):
        original = getattr(obj, method, None)
        if not callable(original):
            return
        @wraps(original)
        def observed(*args, **kwargs):
            label = self._phase + "/" + name if phase_prefix else name
            with self.scope(label, phase=phase):
                return original(*args, **kwargs)
        self._set(obj, method, observed)

    def attach_learner(self, learner):
        self.wrap(learner, "act", "rollout_actor")
        self.wrap(learner, "record_transition", "rollout_record")
        update_method = ("coordinated_update" if callable(getattr(learner, "coordinated_update", None))
                         else "_update")
        self.wrap(learner, update_method, "ppo", phase="ppo")
        policy = getattr(getattr(learner, "shared_actor", None), "policy", None)
        if policy is not None:
            # Encode is invoked directly, so a Module.forward hook would miss H1.
            self.wrap(policy.backbone, "encode", "h1_encode")
            self.wrap(policy.head, "forward", "h2_forward")
        for attr, label in (("altruistic_owner_critic", "owner_critic_graph"),
                            ("altruistic_mean_field_critic", "mean_field_critic_graph")):
            critic = getattr(learner, attr, None)
            if critic is not None:
                self.wrap(critic, "forward_all_owners", label, phase_prefix=True)
        provider = getattr(learner, "_rollout_context_provider", None)
        if provider is not None:
            self._attach_decision_diagnostics(provider)
            self.wrap(provider, "current_action_contexts", "preparation_pre", phase="preparation_pre")
            self.wrap(provider, "current_critic_graphs", "preparation_post", phase="preparation_post")
            bridge = getattr(getattr(provider, "native_env", None), "observation_bridge", None)
            if bridge is not None:
                self.wrap(bridge, "prepare_snapshot", "observation_snapshot", phase_prefix=True)
                original_prepare = bridge.prepare_snapshot
                def prepared(*args, **kwargs):
                    self.rows[self._phase + "/actor_views"] += int(kwargs.get("include_actor", True))
                    self.rows[self._phase + "/critic_views"] += int(kwargs.get("include_privileged", True))
                    return original_prepare(*args, **kwargs)
                self._set(bridge, "prepare_snapshot", prepared)
            native_env = getattr(provider, "native_env", None)
            prepare_env = getattr(native_env, "prepare_env", None)
            if callable(prepare_env):
                @wraps(prepare_env)
                def prepared_row(index, *args, **kwargs):
                    post = kwargs.get("post_step", args[0] if args else False)
                    override = kwargs.get("include_candidates_override", args[1] if len(args) > 1 else None)
                    handoffs = getattr(native_env, "handoffs", ())
                    marl_active = bool(handoffs[index].marl_active) if handoffs else True
                    active = marl_active and (override is None or bool(override))
                    label = "post" if post else "active" if active else "sleeping"
                    with self.scope("preparation_row_" + label):
                        return prepare_env(index, *args, **kwargs)
                self._set(native_env, "prepare_env", prepared_row)
        evaluate = getattr(learner, "evaluate_altruistic_critic_minibatch", None)
        if callable(evaluate):
            @wraps(evaluate)
            def evaluated(*args, **kwargs):
                samples = kwargs["structured_batch"]
                active = sum(bool(getattr(sample, "rollout_active", True)) for sample in samples)
                padding = len(samples) - active
                self.rows["ppo_critic_valid"] += active
                self.rows["ppo_critic_padding"] += padding
                self.rows["ppo_critic_padding_skipped"] += padding if kwargs.get("skip_inactive", False) else 0
                return evaluate(*args, **kwargs)
            self._set(learner, "evaluate_altruistic_critic_minibatch", evaluated)
        return self

    def report(self):
        if self._cuda_ms is None:
            self._cuda_ms = defaultdict(float)
            if self.events:
                # Can also be called after __exit__, in which case count here.
                self._original_synchronize(self.device)
                self.explicit_cuda_synchronizations += 1
                for name, (start, end) in self.events:
                    self._cuda_ms[name] += start.elapsed_time(end)
        return dict(calls=dict(self.calls), rows=dict(self.rows), inclusive_host_s=dict(self.host_s),
                    decision_reason_counts=dict(self.decision_reason_counts),
                    cuda_stream_ms=dict(self._cuda_ms), cuda_event_pairs=len(self.events),
                    cuda_event_pairs_dropped=self.event_pairs_dropped,
                    explicit_cuda_synchronizations=self.explicit_cuda_synchronizations,
                    implicit_cuda_synchronizations="not counted; tensor .item/.cpu may synchronize",
                    notes="Inclusive host and current Torch stream only; includes nested intervals; excludes PhysX streams. CUDA timings are partial if event pairs were dropped. Report once after recording.")
