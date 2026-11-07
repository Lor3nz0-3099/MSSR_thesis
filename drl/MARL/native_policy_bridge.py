"""In-process BC-v4/ActionSet adapter for a native Isaac Lab environment.

Isaac Lab owns vectorization, resets and stepping. This adapter only prepares
structured policy sidecars and stages decoded proposals before ``env.step``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import torch

from drl.IL.bc_v1.training import move_batch_to_device
from drl.MARL.phase1_bc_v4_observation import SynchronizedBCV4ObservationSource
from drl.MARL.runtime_context import DirectMARLRuntimeContextProvider, _CapturedRuntimeContext
from drl.MARL.smores_runtime_binding import (
    SMORESMARLActionRejected,
    SMORESMARLDecisionBridge,
    SMORESMARLProposalOutcome,
)


@dataclass(frozen=True)
class _NativeCapturedRuntimeContext(_CapturedRuntimeContext):
    rollout_active: bool = True
    availability_snapshot: object = None


class NativePreparedRuntimeContextProvider(DirectMARLRuntimeContextProvider):
    """Select the two certified tensor views, retaining the generic lifecycle."""

    def __init__(self, *, observation_source, native_env=None,
                 post_step_observation_provider=None, actor_only=False,
                 structural_scheduler='fixed', **kwargs):
        self.actor_only = bool(actor_only)
        self.observation_source = observation_source
        self.native_env = native_env
        self.post_step_observation_provider = post_step_observation_provider
        self._prepared = ()
        self._availability_caches = {}
        self._sleeping_actor_omission = False
        self._preparation_metrics = dict.fromkeys((
            'active_rows', 'sleeping_rows', 'sleeping_actor_omitted_rows',
            'actor_views', 'critic_views', 'post_actor_views', 'post_critic_views',
            'actor_transfer_tensors', 'critic_transfer_tensors',
        ), 0)
        self._structural_options = None
        if native_env is not None:
            from drl.MARL.c1_training_options import NativeStructuralOptionController
            self._structural_options = NativeStructuralOptionController(
                native_env, mode=structural_scheduler
            )
        super().__init__(**kwargs)
        if observation_source.num_envs != self.num_envs:
            raise ValueError('observation source and handoff batch sizes differ')

    def reset(self):
        super().reset()
        self._prepared = ()
        self._post_step_critics = None
        self.observation_source.reset()

    def _capture(self):
        decision_mask = (self._structural_options.decision_mask()
                         if self._structural_options is not None
                         else tuple(True for _ in range(self.num_envs)))
        prepare_with_candidates = getattr(
            self.observation_source, 'prepare_env_with_candidates', None
        )
        prepare_privileged = getattr(self.observation_source, 'prepare_env_privileged', None)
        rollout_active = tuple(bool(handoff.marl_active and decision_mask[i])
                               for i, handoff in enumerate(self._handoffs))
        omit_actor = tuple(self._sleeping_actor_omission and not active
                           and not self.actor_only and callable(prepare_privileged)
                           for active in rollout_active)
        prepared = tuple(
            self._prepare_on_device(
                prepare_privileged(i, include_action_candidates=decision_mask[i]) if omit_actor[i] else
                prepare_with_candidates(i, include_action_candidates=decision_mask[i])
                if callable(prepare_with_candidates)
                else self.observation_source.prepare_env(i)
            )
            for i in range(self.num_envs)
        )
        captured = []
        for index, item in enumerate(prepared):
            if item.privileged_graph is None and not self.actor_only:
                raise ValueError('native policy requires a prepared privileged critic graph')
            actor = deepcopy(item.actor_graph)
            routing = deepcopy(item.routing_snapshot)
            if actor is None and not omit_actor[index]:
                raise ValueError('native action context requires a prepared actor graph')
            hard_mask = (torch.zeros(0, dtype=torch.bool) if actor is None
                         else self._hard_mask_builder(actor, routing))
            device = getattr(self.native_env, 'device', None)
            if device is not None:
                hard_mask = hard_mask.to(device=device)
            availability = None
            dispatcher = getattr(self.native_env, 'native_dispatcher', None)
            active = rollout_active[index]
            self._preparation_metrics['active_rows' if active else 'sleeping_rows'] += 1
            self._preparation_metrics['sleeping_actor_omitted_rows'] += int(actor is None)
            if (active and
                    getattr(getattr(self.native_env, 'cfg', None), 'initialization_mode', None)
                    == 'loose_modules' and dispatcher is not None):
                from drl.MARL.c1_feasibility_cache import NativeAvailabilityCache
                cache = self._availability_caches.get(index)
                if cache is None:
                    cache = self._availability_caches[index] = NativeAvailabilityCache()
                generations = getattr(self.native_env, 'generation', [None] * self.num_envs)
                availability = cache.capture(routing=routing, graph=actor,
                    hard_mask=hard_mask, executor=dispatcher.executor_provider(index),
                    generation=generations[index])
                hard_mask = availability.mask_tensor(device=hard_mask.device)
            captured.append(_NativeCapturedRuntimeContext(
                actor_graph=actor,
                critic_graph=None if self.actor_only else deepcopy(item.privileged_graph),
                routing_snapshot=routing,
                hard_mask=hard_mask,
                rollout_active=active,
                availability_snapshot=availability,
            ))
        self._prepared = prepared
        self._post_step_critics = None
        self._captured = tuple(captured)
        return self._captured

    def encode_env_actions(self, decisions):
        result = super().encode_env_actions(decisions)
        if self._structural_options is not None:
            from drl.MARL.c1_training_options import decision_selects_resource
            captured = self._ensure_current_capture()
            for env_index, (decision, snapshot) in enumerate(zip(decisions, captured, strict=True)):
                context = self._public_context(snapshot)
                self._structural_options.set_continuous_control(
                    env_index,
                    bool(context.rollout_active and decision_selects_resource(decision, context)),
                )
        return result

    def structural_option_metrics(self):
        if self._structural_options is None:
            return None
        return self._structural_options.metrics()

    def set_sleeping_actor_omission(self, enabled):
        """Opt in when the learner excludes inactive actor/replay inputs.

        The setting applies to subsequent captures. Sources without an explicit
        privileged-only capability retain their original complete contexts.
        """
        self._sleeping_actor_omission = bool(enabled)

    def preparation_metrics(self):
        """Passive lifetime view counts and actual cross-device tensor copies.

        A legacy post-step source can still build an unused actor view; account
        for it separately even though its transfer is skipped. No device sync
        or tensor value read is used by these counters.
        """
        return dict(self._preparation_metrics)

    def feasibility_metrics(self):
        """Passive per-env accounting; reset epochs never share BUSY support."""
        from drl.MARL.c1_feasibility_cache import COUNTERS
        by_env = {}
        dispatcher = getattr(self.native_env, 'native_dispatcher', None)
        generations = getattr(self.native_env, 'generation', [None] * self.num_envs)
        for index, cache in sorted(self._availability_caches.items()):
            by_env[str(index)] = cache.metrics(
                executor=dispatcher.executor_provider(index), generation=generations[index])
        totals = {name: sum(row[name] for row in by_env.values())
                  for name in (*COUNTERS, 'feasibility_masking_s')}
        return dict(totals=totals, by_env=by_env)

    def _prepare_on_device(self, prepared, *, critic_only=False, post_step=False):
        prefix = 'post_' if post_step else ''
        self._preparation_metrics[prefix + 'actor_views'] += int(prepared.actor_graph is not None)
        self._preparation_metrics[prefix + 'critic_views'] += int(prepared.privileged_graph is not None)
        device = getattr(self.native_env, 'device', None)
        if device is None:
            return prepared
        # Reuse the certified BC tensor-tree transfer; routing references and
        # decoder sidecars stay untouched. Decoder tensors must follow the
        # prediction device too (its candidate-routing equality is tensorized).
        actor = (prepared.actor_graph if critic_only else
                 self._move_graph(prepared.actor_graph, device, 'actor'))
        critic = self._move_graph(prepared.privileged_graph, device, 'critic')
        return replace(prepared, actor_graph=actor, privileged_graph=critic,
                       decoder_batch=(prepared.decoder_batch if critic_only or prepared.decoder_batch is None
                                      else {**prepared.decoder_batch, 'inputs': actor}))

    def _move_graph(self, graph, device, view):
        if graph is None:
            return None
        target = torch.device(device)
        if target.type == 'cuda' and target.index is None:
            target = torch.device('cuda', torch.cuda.current_device())

        def count_transfers(value):
            if torch.is_tensor(value):
                return int(value.device != target)
            if isinstance(value, dict):
                return sum(count_transfers(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(count_transfers(item) for item in value)
            return 0

        self._preparation_metrics[view + '_transfer_tensors'] += count_transfers(graph)
        return move_batch_to_device({'inputs': graph, 'supervision': {}}, device)['inputs']

    @staticmethod
    def _public_context(captured):
        return replace(DirectMARLRuntimeContextProvider._public_context(captured),
                       rollout_active=captured.rollout_active,
                       availability_snapshot=captured.availability_snapshot)

    def current_critic_graphs(self):
        if self.actor_only:
            raise RuntimeError('actor-only viewer has no critic context; training requires the default provider')
        if self.post_step_observation_provider is None:
            return super().current_critic_graphs()
        if self._awaiting_post_step:
            prepared = tuple(self._prepare_on_device(self.post_step_observation_provider(i),
                                                     critic_only=True, post_step=True)
                             for i in range(self.num_envs))
            if any(item.privileged_graph is None for item in prepared):
                raise ValueError('native post-step state requires a privileged critic graph')
            self._post_step_critics = [deepcopy(item.privileged_graph) for item in prepared]
            # Native auto-reset may have advanced the current state already.
            # Preserve final s_(t+1) here; the following act prepares reset s_0.
            self._captured = None
            self._prepared = ()
            self._awaiting_post_step = False
            self.observation_source.reset()
        if self._post_step_critics is not None:
            return self._post_step_critics
        return super().current_critic_graphs()

    def current_altruistic_rollout_batch(self):
        return self.native_env.current_altruistic_rollout_batch()

    def decoder_batch(self, env_index, context):
        del context
        self._ensure_current_capture()
        return self._prepared[env_index].decoder_batch


class NativeDecodedDispatcher:
    """Stage complete proposals; physical admission/application belongs to env."""

    def __init__(self, env):
        self.env = env

    def __call__(self, *, env_index, decoded_actions, context):
        del context
        commands = tuple(decoded_actions)
        owners = tuple(sorted({owner for item in commands for owner in item.command['module_ids']}))
        outcome = SMORESMARLProposalOutcome(True, 'ACCEPTED', 'proposal staged', owners)
        projection_provider = getattr(self,'projection_provider',None)
        if projection_provider is not None:
            from drl.MARL.projected_runtime import projection_outcome
            outcome = projection_outcome(projection_provider(env_index))
        result = self.env.stage_decoded(env_index, commands, outcome)
        if isinstance(result, SMORESMARLProposalOutcome) and not result.accepted:
            raise SMORESMARLActionRejected(result)

    def clear(self, env_index=None):
        clear = getattr(self.env, 'clear_staged', None)
        if callable(clear):
            clear(env_index)


class _NativeProposalBinding:
    """Preserve sampled invalid/conflicting proposals as attributed outcomes."""

    def __init__(self, binding, env):
        self.binding, self.env = binding, env

    def decode_marl_action(self, **kwargs):
        try:
            return self.binding.decode_marl_action(**kwargs)
        except ValueError as error:
            message = str(error)
            invalid_payload_messages=('Physical payload outside limits',
                'Invalid gravity_settle passive scope/duration',
                'Contradictory gravity_settle action scope')
            if not any(token in message for token in (
                *invalid_payload_messages, 'Conflicting physical resource claims',
                'Physical resource unavailable',
            )):
                raise
            action, routing = kwargs['action'], kwargs['routing']
            selected=[]
            resources=routing['resource_candidates']
            for index,candidate in enumerate(resources):
                if action.presence[index]:
                    selected.append((index,candidate.resource_ref.module.module_id))
            for local,(_spec,binding) in enumerate(routing.get('structural_bindings',())):
                index=len(resources)+local
                if action.presence[index]:selected.append((index,binding.initiator.module_id))
            if 'claims' in message or 'unavailable' in message:
                from drl.MARL.c1_runner import sampled_resource_conflict_owners
                owners = set(sampled_resource_conflict_owners(action, routing))
                # A rejected structural ActionSet is attributed at proposal
                # level to its selected decision owners, not an invented repair.
                if not owners or any(i>=len(resources) for i,_ in selected):
                    owners.update(owner for _,owner in selected)
                code = 'RESOURCE_BUSY'
            else:
                owners = set()
                code = 'INVALID_GOAL'
                for index, owner in selected:
                    presence = torch.zeros_like(action.presence)
                    presence[index] = 1
                    isolated=dict(presence=presence)
                    for name in ('resource_numeric','resource_categorical','structural_numeric',
                                 'structural_boolean','structural_categorical','structural_reference',
                                 'structural_reference_set','structural_scope'):
                        if hasattr(action,name):
                            mapping=getattr(action,name)
                            isolated[name]={index:mapping[index]} if index in mapping else {}
                    try:
                        self.binding.decode_marl_action(**{**kwargs, 'action': replace(action, **isolated)})
                    except ValueError as candidate_error:
                        if not any(token in str(candidate_error) for token in invalid_payload_messages):
                            raise
                        owners.add(owner)
            if not owners:
                raise RuntimeError('sampled rejection has no attributable selected candidate') from error
            outcome = SMORESMARLProposalOutcome(False, code, message, tuple(sorted(owners)))
            self.env.stage_decoded(self._env_index, (), outcome)
            raise SMORESMARLActionRejected(outcome) from error


@dataclass(frozen=True)
class NativePolicyBridge:
    provider: NativePreparedRuntimeContextProvider
    observation_source: object
    decision_bridge: SMORESMARLDecisionBridge
    dispatcher: NativeDecodedDispatcher


def build_native_policy_bridge(
    env, *, observation_bridge, task_context_provider, binding, handoffs,
    hard_mask_builder, placeholder_action, observation_provider=None,
    assigned_goal_provider=None, availability_provider=None,
    configured_arguments_provider=None, observation_source=None,
    post_step_observation_provider=None,
    project_actions=False,
    actor_only=False,
    bulk_host_reads=True,
    structural_scheduler='fixed',
):
    """Build sidecars consumed by existing MSSRMAPPO and SkrlVecEnvWrapper.

    ``env.cached_raw_snapshot(i)`` must return the atomic current native frame.
    ``env.stage_decoded(i, commands, outcome)`` must stage without stepping.
    An injected observation source may reuse the environment's prepared cache.
    """
    if len(handoffs) != env.num_envs:
        raise ValueError('native environment and handoff batch sizes differ')
    source = observation_source or SynchronizedBCV4ObservationSource(
        frame_synchronizer=env, observation_bridge=observation_bridge,
        task_context_provider=task_context_provider,
        observation_provider=observation_provider,
        assigned_goal_provider=assigned_goal_provider,
    )
    dispatcher = NativeDecodedDispatcher(env)
    proposal_binding = _NativeProposalBinding(binding, env)
    if project_actions:
        from drl.MARL.projected_runtime import ProjectedNativeProposalBinding
        proposal_binding = ProjectedNativeProposalBinding(binding,env,bulk_host_reads=bulk_host_reads)
        dispatcher.projection_provider = lambda i:proposal_binding.last_projection_by_env[i]
    decision_bridge = SMORESMARLDecisionBridge(
        binding=proposal_binding, normalizers=observation_bridge.normalizers,
        dispatcher=dispatcher, placeholder_action=placeholder_action,
        availability_provider=availability_provider,
        configured_arguments_provider=configured_arguments_provider,
    )

    def encode(decision, env_index, context):
        # Decisions are encoded sequentially by the existing provider.
        proposal_binding._env_index = env_index
        carrier = decision_bridge(decision, env_index, context)
        if (project_actions and getattr(getattr(env, 'cfg', None), 'initialization_mode', None)
                == 'loose_modules'):
            from drl.MARL.c1_action_events import ensure_action_events
            ensure_action_events(env).record_decision(env, env_index, decision, context,
                proposal_binding.last_projection_by_env[env_index])
        return carrier

    provider = NativePreparedRuntimeContextProvider(
        observation_source=source,
        native_env=env,
        actor_only=actor_only,
        structural_scheduler=structural_scheduler,
        post_step_observation_provider=post_step_observation_provider,
        graph_source=lambda: (), actor_visibility=lambda graph: graph,
        routing_builder=lambda graph: None, hard_mask_builder=hard_mask_builder,
        handoffs=handoffs, action_encoder=encode, neutral_action=placeholder_action,
    )
    decision_bridge.decoder_batch_builder = provider.decoder_batch
    return NativePolicyBridge(provider, source, decision_bridge, dispatcher)
