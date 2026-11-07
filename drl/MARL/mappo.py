from dataclasses import dataclass, replace
from copy import deepcopy
from collections.abc import Mapping
from typing import Any
from time import perf_counter

import torch

from drl.MARL.altruistic_rollout import (
    AltruisticRolloutBatchPayload,
    AltruisticRolloutSample,
    AltruisticRolloutSidecar,
    build_altruistic_rollout_batch,
)
from drl.MARL.c1_smdp import (
    DiscountedOptionAccumulator,
    actor_policy_lag_is_acceptable,
)
from skrl.multi_agents.torch.base import MultiAgent
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
from skrl.multi_agents.torch.mappo.mappo import compute_gae
from skrl.utils import ScopedTimer

from drl.MARL.policy import (
    compute_owner_action_entropies,
    compute_owner_action_log_probs,
    sample_marl_action_bundle,
)


class SMDPEventMemory(RandomMemory):
    """Keep a physical frame's pre/post events in one PPO update.

    The requested rollout size remains the update threshold. Two reserve slots
    cover a frame that starts at that threshold and closes both pre and post.
    Tensor consumers see only recorded rows, never unused reserve storage.
    """

    def __init__(self, *, memory_size, num_envs=1, device=None, **kwargs):
        super().__init__(memory_size=memory_size + 2, num_envs=num_envs, device=device, **kwargs)

    def add_samples(self, **tensors):
        if self.filled:
            raise BufferError("SMDP event memory requires post_interaction before more frames")
        super().add_samples(**tensors)

    def get_tensor_by_name(self, name):
        valid_rows = len(self) // self.num_envs
        return super().get_tensor_by_name(name)[:valid_rows]

    def set_tensor_by_name(self, name, tensor):
        with torch.no_grad():
            self.get_tensor_by_name(name).copy_(tensor)


@dataclass(frozen=True)
class StructuredRolloutSample:
    """Structured transition data that cannot live in fixed-size skrl tensors."""

    action: Any
    hard_mask: Any
    owner_active_mask: dict[str, bool]
    routing_snapshot: Any
    graph_snapshot: Any
    critic_graph_snapshot: Any = None
    next_critic_graph_snapshot: Any = None
    rollout_active: bool = True
    availability_snapshot: Any = None
    # Number of 30 Hz control steps represented by this decision row.
    # Zero denotes vector padding for an env with no closed option.
    duration_steps: int = 1


@dataclass(frozen=True)
class OnlineRolloutContext:
    """Actor/critic context for one vectorized environment."""

    actor_graph: Any
    critic_graph: Any
    routing_snapshot: Any
    hard_mask: Any
    rollout_active: bool = True
    availability_snapshot: Any = None


@dataclass(frozen=True)
class OnlineRolloutDecision:
    """One complete structured MAPPO interaction decision."""

    action: Any
    owner_log_probs: dict[str, torch.Tensor]
    global_value: torch.Tensor
    structured_sample: StructuredRolloutSample
    action_set_prediction_template: Any = None


@dataclass
class _SMDPPendingTransition:
    decision: OnlineRolloutDecision
    observations: dict[str, torch.Tensor]
    states: dict[str, torch.Tensor]
    altruistic_sample: AltruisticRolloutSample
    rewards: DiscountedOptionAccumulator
    policy_version: int


class StructuredRolloutSidecar:
    """Structured rollout storage aligned with skrl Memory.tensors_view indexes."""

    def __init__(self):
        self._samples = []
        self._diagnostic_generation = 0

    def __len__(self) -> int:
        return len(self._samples)

    def append_step(self, samples) -> None:
        self._samples.extend(samples)

    def sample_by_index(self, indexes):
        return [
            self._samples[int(index)]
            for index in indexes
        ]

    def sample_minibatches(self, *, indexes, mini_batches: int):
        indexes = (
            indexes.tolist()
            if hasattr(indexes, "tolist")
            else list(indexes)
        )

        quotient, remainder = divmod(len(indexes), mini_batches)

        batches = []
        start = 0
        for batch_index in range(mini_batches):
            size = quotient + (1 if batch_index < remainder else 0)
            stop = start + size
            batches.append(self.sample_by_index(indexes[start:stop]))
            start = stop

        return batches

    def clear(self) -> None:
        self._samples.clear()
        self._diagnostic_generation += 1


def _detach_clone_tree(value):
    """Detach and clone a nested tensor tree for interaction-time use."""
    if torch.is_tensor(value):
        return value.detach().clone()

    if isinstance(value, dict):
        return {
            key: _detach_clone_tree(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _detach_clone_tree(item)
            for item in value
        )

    if isinstance(value, list):
        return [
            _detach_clone_tree(item)
            for item in value
        ]

    return value


class MSSRMAPPO(MAPPO):
    """MAPPO variant for the MSSR shared-policy architecture."""

    def __init__(
        self,
        *,
        possible_agents,
        models,
        memories=None,
        observation_spaces=None,
        state_spaces=None,
        action_spaces=None,
        device=None,
        cfg={},
        deterministic_update_mode: bool = False,
        skip_expert_policy_inference: bool = False,
    ):
        """Initialize MAPPO without creating one optimizer per agent."""
        cfg = MAPPO_CFG(**cfg) if isinstance(cfg, dict) else cfg

        MultiAgent.__init__(
            self,
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=cfg,
        )

        self._structured_rollout_sidecar = StructuredRolloutSidecar()
        self.deterministic_update_mode = deterministic_update_mode
        self.skip_expert_policy_inference = skip_expert_policy_inference
        self._current_next_critic_graph = None
        self._smdp_options_enabled = False
        self._smdp_pending = None
        self._smdp_transition_count = 0
        self._smdp_event_step_count = 0
        self._smdp_event_rows_last_interaction = 0
        self._smdp_policy_version = 0
        self._smdp_max_policy_lag = 0

        self.policies = {
            uid: self.models[uid].get("policy")
            for uid in self.possible_agents
        }
        self.values = {
            uid: self.models[uid].get("value")
            for uid in self.possible_agents
        }

        first_uid = self.possible_agents[0]
        self.shared_actor = self.policies[first_uid]
        self.shared_critic = self.values[first_uid]

        # skrl saves each UID registry. Shared models belong to the first UID
        # once, rather than eight duplicate aliases of the same state.
        self.checkpoint_modules[first_uid]["policy"] = self.shared_actor
        self.checkpoint_modules[first_uid]["value"] = self.shared_critic

        self._device_type = torch.device(self.device).type
        self.scaler = torch.amp.GradScaler(
            device=self._device_type,
            enabled=self.cfg.mixed_precision,
        )

        actor_lr, critic_lr = self.cfg.learning_rate[first_uid]

        (
            self.actor_optimizer,
            self.critic_optimizer,
        ) = self.build_shared_optimizers(
            actor=self.shared_actor,
            critic=self.shared_critic,
            actor_learning_rate=actor_lr,
            critic_learning_rate=critic_lr,
        )

        self.optimizers = {
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
        }
        self.schedulers = {}
        self.checkpoint_modules[first_uid]["actor_optimizer"] = self.actor_optimizer
        self.checkpoint_modules[first_uid]["critic_optimizer"] = self.critic_optimizer
        self._last_update_metrics = {}

        self._observation_preprocessor = {}
        self._state_preprocessor = {}
        self._value_preprocessor = {}

        for uid in self.possible_agents:
            if self.cfg.observation_preprocessor[uid]:
                self._observation_preprocessor[uid] = (
                    self.cfg.observation_preprocessor[uid](
                        **self.cfg.observation_preprocessor_kwargs[uid]
                    )
                )
                self.checkpoint_modules[uid]["observation_preprocessor"] = (
                    self._observation_preprocessor[uid]
                )
            else:
                self._observation_preprocessor[uid] = self._empty_preprocessor

            if self.cfg.state_preprocessor[uid]:
                self._state_preprocessor[uid] = self.cfg.state_preprocessor[uid](
                    **self.cfg.state_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["state_preprocessor"] = (
                    self._state_preprocessor[uid]
                )
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor

            if self.cfg.value_preprocessor[uid]:
                self._value_preprocessor[uid] = self.cfg.value_preprocessor[uid](
                    **self.cfg.value_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["value_preprocessor"] = (
                    self._value_preprocessor[uid]
                )
            else:
                self._value_preprocessor[uid] = self._empty_preprocessor

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Trigger updates from physical rows or compact SMDP event rows."""
        if self.training:
            if bool(getattr(self, "_smdp_options_enabled", False)):
                advance = int(self._smdp_event_rows_last_interaction)
                self._smdp_event_rows_last_interaction = 0
                self._rollout += advance
                update_due = self._rollout >= self.cfg.rollouts
            else:
                self._rollout += 1
                update_due = not self._rollout % self.cfg.rollouts

            if update_due and timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    if getattr(self, 'deterministic_update_mode', False):
                        for name in ('shared_actor', 'shared_critic',
                                     'altruistic_owner_critic', 'altruistic_mean_field_critic'):
                            model = getattr(self, name, None)
                            if model is not None:
                                model.eval()
                    self.coordinated_update(timestep=timestep, timesteps=timesteps)

                    if bool(getattr(self, "_smdp_options_enabled", False)):
                        metrics = getattr(
                            self,
                            "last_update_metrics",
                            {},
                        )
                        if callable(metrics):
                            metrics = metrics()

                        if int(
                            metrics.get(
                                "actor_optimizer_steps",
                                0,
                            )
                        ) > 0:
                            self._smdp_policy_version = (
                                int(
                                    getattr(
                                        self,
                                        "_smdp_policy_version",
                                        0,
                                    )
                                )
                                + 1
                            )

                    self._structured_rollout_sidecar.clear()
                    altruistic_sidecar = getattr(self, "_altruistic_rollout_sidecar", None)
                    if altruistic_sidecar is not None:
                        altruistic_sidecar.clear()
                    self._current_next_critic_graph = None
                    if bool(getattr(self, "_smdp_options_enabled", False)):
                        # Consume all events from this physical frame together:
                        # deferring one past an actor update would increase lag.
                        self._rollout = 0
                        for memory in getattr(self, "memories", {}).values():
                            memory.reset()
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        MultiAgent.post_interaction(self, timestep=timestep, timesteps=timesteps)

    def record_structured_rollout_step(self, samples) -> None:
        self._structured_rollout_sidecar.append_step(samples)

    def build_online_rollout_decision(
        self,
        *,
        actor_graph,
        critic_graph,
        routing_snapshot,
        hard_mask,
        availability_snapshot=None,
        actor_output=None,
    ) -> OnlineRolloutDecision:
        """Build one complete structured interaction decision.

        The actor receives only ``actor_graph``. The centralized critic
        receives only ``critic_graph``. Both snapshots are retained for
        exact PPO reevaluation.
        """
        with torch.no_grad():
            if actor_output is None:
                actor_output = self.forward_shared_actor(
                    actor=self.shared_actor,
                    graph=actor_graph,
                    possible_agents=self.possible_agents,
                    routing=routing_snapshot,
                )

            required_keys = (
                "records",
                "resource_payloads",
                "structural_payloads",
                "resource_log_std",
                "structural_log_std",
            )

            missing = [
                key
                for key in required_keys
                if key not in actor_output
            ]

            if missing:
                raise KeyError(
                    "Shared actor output is missing required keys: "
                    + ", ".join(missing)
                )

            records = actor_output["records"]

            from drl.MARL.c1_feasibility import apply_availability
            actor_output = apply_availability(actor_output, availability_snapshot)

            if torch.is_tensor(hard_mask):
                normalized_hard_mask = hard_mask.to(
                    dtype=torch.bool,
                )
            else:
                device = (
                    records[0].presence_logit.device
                    if records
                    else self.device
                )

                normalized_hard_mask = torch.as_tensor(
                    hard_mask,
                    dtype=torch.bool,
                    device=device,
                )

            if normalized_hard_mask.numel() != len(records):
                raise ValueError(
                    "hard_mask length must match candidate records"
                )

            if availability_snapshot is not None:
                normalized_hard_mask = normalized_hard_mask | availability_snapshot.mask_tensor(
                    device=normalized_hard_mask.device)
            # The exact rollout support must survive later runtime changes.
            normalized_hard_mask = normalized_hard_mask.detach().clone()

            action = sample_marl_action_bundle(
                records=records,
                resource_payloads=actor_output[
                    "resource_payloads"
                ],
                structural_payloads=actor_output[
                    "structural_payloads"
                ],
                hard_mask=normalized_hard_mask,
                resource_log_std=actor_output[
                    "resource_log_std"
                ],
                structural_log_std=actor_output[
                    "structural_log_std"
                ],
                deterministic=bool(getattr(self,'deterministic_policy_actions',False)),
            )

            owner_log_probs = compute_owner_action_log_probs(
                records=records,
                resource_payloads=actor_output[
                    "resource_payloads"
                ],
                structural_payloads=actor_output[
                    "structural_payloads"
                ],
                action=action,
                hard_mask=normalized_hard_mask,
                possible_agents=self.possible_agents,
                resource_log_std=actor_output[
                    "resource_log_std"
                ],
                structural_log_std=actor_output[
                    "structural_log_std"
                ],
            )

            owner_log_probs = {
                uid: value.detach()
                for uid, value in owner_log_probs.items()
            }

            global_value, _ = self.shared_critic.act(
                {
                    "states": critic_graph,
                },
                role="value",
            )

            global_value = global_value.detach()

        owner_active_mask = {
            uid: False
            for uid in self.possible_agents
        }

        for record in records:
            index = int(
                record.candidate_index
            )

            if (
                index < 0
                or index >= normalized_hard_mask.numel()
            ):
                raise ValueError(
                    "candidate_index is outside hard_mask"
                )

            if not bool(
                normalized_hard_mask[index].item()
            ):
                if (
                    record.owner_agent_id
                    in owner_active_mask
                ):
                    owner_active_mask[
                        record.owner_agent_id
                    ] = True

        sample = StructuredRolloutSample(
            action=action,
            hard_mask=normalized_hard_mask.detach(),
            owner_active_mask=owner_active_mask,
            routing_snapshot=routing_snapshot,
            graph_snapshot=actor_graph,
            critic_graph_snapshot=critic_graph,
            availability_snapshot=availability_snapshot,
        )

        return OnlineRolloutDecision(
            action=action,
            owner_log_probs=owner_log_probs,
            global_value=global_value,
            structured_sample=sample,
            action_set_prediction_template=(
                _detach_clone_tree(
                    actor_output[
                        "prediction_template"
                    ]
                )
                if actor_output.get(
                    "prediction_template"
                ) is not None
                else None
            ),
        )

    def record_online_rollout_batch(
        self,
        *,
        decisions,
        observations,
        states,
        env_actions,
        team_rewards,
        next_observations,
        next_states,
        terminated,
        truncated,
        next_critic_graphs,
        altruistic_samples=None,
    ) -> None:
        """Store one vectorized structured MARL transition.

        Fixed-shape tensors required by skrl live in Memory.
        The true structured MARL action and graph snapshots live in
        StructuredRolloutSidecar.

        ``actions`` in skrl Memory is deliberately a zero placeholder:
        coordinated_update never uses it for policy reevaluation.
        """
        decisions = list(decisions)
        next_critic_graphs = list(
            next_critic_graphs
        )

        batch_size = len(decisions)

        if batch_size <= 0:
            raise ValueError(
                "decision batch must not be empty"
            )

        if (
            len(next_critic_graphs)
            != batch_size
        ):
            raise ValueError(
                "decision batch and next critic graph batch "
                "must have the same size"
            )

        def validate_batch_tensor(
            *,
            name,
            tensor,
        ):
            if not torch.is_tensor(tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor"
                )

            if (
                tensor.ndim != 2
                or tensor.shape[0]
                != batch_size
            ):
                raise ValueError(
                    f"decision batch size {batch_size} "
                    f"does not match {name} shape "
                    f"{tuple(tensor.shape)}"
                )

        validate_batch_tensor(
            name="team_rewards",
            tensor=team_rewards,
        )
        validate_batch_tensor(
            name="terminated",
            tensor=terminated,
        )
        validate_batch_tensor(
            name="truncated",
            tensor=truncated,
        )

        if (
            team_rewards.shape[1] != 1
            or terminated.shape[1] != 1
            or truncated.shape[1] != 1
        ):
            raise ValueError(
                "team reward and termination tensors "
                "must have shape (batch, 1)"
            )

        owner_log_probs = {}

        for uid in self.possible_agents:
            for name, mapping in (
                (
                    "observations",
                    observations,
                ),
                (
                    "states",
                    states,
                ),
                (
                    "env_actions",
                    env_actions,
                ),
                (
                    "next_observations",
                    next_observations,
                ),
                (
                    "next_states",
                    next_states,
                ),
            ):
                if uid not in mapping:
                    raise KeyError(
                        f"{name} is missing owner {uid}"
                    )

                validate_batch_tensor(
                    name=f"{name}[{uid}]",
                    tensor=mapping[uid],
                )

            owner_samples = []

            for decision in decisions:
                if (
                    uid
                    not in decision.owner_log_probs
                ):
                    raise ValueError(
                        f"missing owner log-prob for {uid}"
                    )

                value = decision.owner_log_probs[
                    uid
                ]

                if (
                    not torch.is_tensor(value)
                    or value.numel() != 1
                ):
                    raise ValueError(
                        f"owner log-prob for {uid} "
                        "must be scalar"
                    )

                owner_samples.append(
                    value.detach().reshape(1)
                )

            owner_log_probs[uid] = (
                torch.stack(
                    owner_samples,
                    dim=0,
                )
            )

        value_samples = []

        for decision in decisions:
            value = decision.global_value

            if (
                not torch.is_tensor(value)
                or value.numel() != 1
            ):
                raise ValueError(
                    "global value for each decision "
                    "must be scalar"
                )

            value_samples.append(
                value.detach().reshape(1)
            )

        global_values = torch.stack(
            value_samples,
            dim=0,
        )

        # All validation is complete before touching Memory.
        for uid in self.possible_agents:
            self.memories[uid].add_samples(
                observations=observations[
                    uid
                ].detach(),
                states=states[
                    uid
                ].detach(),
                actions=torch.zeros_like(
                    env_actions[uid]
                ),
                rewards=team_rewards.detach(),
                terminated=terminated.detach(),
                truncated=truncated.detach(),
                log_prob=owner_log_probs[
                    uid
                ],
                values=global_values,
            )

        self._structured_rollout_sidecar.append_step(
            [
                replace(decision.structured_sample,
                        next_critic_graph_snapshot=deepcopy(next_graph))
                for decision, next_graph in zip(decisions, next_critic_graphs, strict=True)
            ]
        )

        if altruistic_samples is None:
            self._record_altruistic_rollout_step(expected_num_envs=len(decisions))
        else:
            samples = tuple(altruistic_samples)
            if len(samples) != len(decisions):
                raise ValueError("altruistic SMDP batch must match decision batch")
            self._altruistic_rollout_sidecar.append_step(samples)

        self._current_next_observations = (
            next_observations
        )
        self._current_next_states = (
            next_states
        )
        self._current_next_critic_graph = (
            next_critic_graphs
        )

    def set_altruistic_rollout_provider(
        self,
        provider,
        *,
        radius_m: float,
    ) -> None:
        """Enable the separate APPO local-reward rollout channel."""

        if provider is None:
            self._altruistic_rollout_provider = None
            if hasattr(
                self,
                "_altruistic_rollout_sidecar",
            ):
                self._altruistic_rollout_sidecar.clear()
            return

        method = getattr(
            provider,
            "current_altruistic_rollout_batch",
            None,
        )

        if not callable(
            method
        ):
            raise TypeError(
                "altruistic rollout provider must expose "
                "current_altruistic_rollout_batch()"
            )

        if (
            isinstance(
                radius_m,
                bool,
            )
            or not isinstance(
                radius_m,
                (
                    int,
                    float,
                ),
            )
            or not float(
                radius_m
            ) > 0.0
        ):
            raise ValueError(
                "altruism radius_m must be positive"
            )

        self._altruistic_rollout_provider = (
            provider
        )

        self._altruism_radius_m = float(
            radius_m
        )

        self._altruistic_rollout_sidecar = (
            AltruisticRolloutSidecar()
        )

    def _build_altruistic_rollout_step(
        self,
        *,
        expected_num_envs: int,
    ):
        provider = getattr(
            self,
            "_altruistic_rollout_provider",
            None,
        )

        if provider is None:
            return None

        payload = (
            provider
            .current_altruistic_rollout_batch()
        )

        if not isinstance(
            payload,
            AltruisticRolloutBatchPayload,
        ):
            raise TypeError(
                "current_altruistic_rollout_batch() must return "
                "AltruisticRolloutBatchPayload"
            )

        if len(
            payload.module_ids_by_row
        ) != expected_num_envs:
            raise ValueError(
                "altruistic rollout environment count "
                "does not match online rollout"
            )

        samples = (
            build_altruistic_rollout_batch(
                possible_agents=(
                    self.possible_agents
                ),
                module_ids_by_row=(
                    payload.module_ids_by_row
                ),
                owner_to_module_id=(
                    payload.owner_to_module_id
                ),
                module_positions_by_id=(
                    payload.module_positions_by_id
                ),
                owner_rewards=(
                    payload.owner_rewards
                ),
                radius_m=(
                    self._altruism_radius_m
                ),
                next_module_ids_by_row=(
                    payload.next_module_ids_by_row
                ),
                next_owner_to_module_id=(
                    payload.next_owner_to_module_id
                ),
            )
        )

        return samples

    def _record_altruistic_rollout_step(
        self,
        *,
        expected_num_envs: int,
    ):
        samples = self._build_altruistic_rollout_step(
            expected_num_envs=expected_num_envs
        )
        if samples is None:
            return None
        sidecar = getattr(self, "_altruistic_rollout_sidecar", None)
        if sidecar is None:
            raise RuntimeError("altruistic rollout sidecar is not initialized")
        sidecar.append_step(samples)
        return samples

    def configure_fixed_alpha_appo(
        self,
        *,
        owner_critic,
        mean_field_critic,
        alpha,
        critic_learning_rate=None,
    ) -> None:
        """Enable the fixed-alpha altruistic APPO objective."""

        actor_params = {
            id(
                parameter
            )
            for parameter
            in self.shared_actor.parameters()
        }

        owner_params = {
            id(
                parameter
            )
            for parameter
            in owner_critic.parameters()
        }

        mf_params = {
            id(
                parameter
            )
            for parameter
            in mean_field_critic.parameters()
        }

        global_params = {
            id(
                parameter
            )
            for parameter
            in self.shared_critic.parameters()
        }

        if (
            actor_params
            & owner_params
            or actor_params
            & mf_params
            or owner_params
            & mf_params
            or global_params
            & owner_params
            or global_params
            & mf_params
        ):
            raise ValueError(
                "actor, global critic, owner critic and "
                "mean-field critic must have disjoint parameters"
            )

        first_uid = next(
            iter(
                self.possible_agents
            )
        )

        if critic_learning_rate is None:
            critic_learning_rate = (
                self
                .cfg
                .learning_rate[
                    first_uid
                ][
                    1
                ]
            )

        self.altruistic_owner_critic = (
            owner_critic
        )

        self.altruistic_mean_field_critic = (
            mean_field_critic
        )

        self.altruistic_alpha = alpha

        self.owner_critic_optimizer = (
            torch.optim.Adam(
                owner_critic.parameters(),
                lr=float(
                    critic_learning_rate
                ),
            )
        )

        self.mean_field_critic_optimizer = (
            torch.optim.Adam(
                mean_field_critic.parameters(),
                lr=float(
                    critic_learning_rate
                ),
            )
        )

        self.optimizers[
            "owner_critic"
        ] = (
            self.owner_critic_optimizer
        )

        self.optimizers[
            "mean_field_critic"
        ] = (
            self.mean_field_critic_optimizer
        )

        registry = self.checkpoint_modules[first_uid]
        registry.update(owner_critic=owner_critic, mean_field_critic=mean_field_critic,
                        owner_critic_optimizer=self.owner_critic_optimizer,
                        mean_field_critic_optimizer=self.mean_field_critic_optimizer)
        self._training_advantage_mode = (
            "altruistic"
        )

    def use_team_mappo(
        self,
    ) -> None:
        """Select the original global/team MAPPO objective."""

        self._training_advantage_mode = (
            "team"
        )

    def configure_smdp_options(self, *, enabled: bool = True) -> None:
        """Enable compact SMDP collection for temporally extended C1 options."""
        self._smdp_options_enabled = bool(enabled)
        if enabled and any(not isinstance(memory, SMDPEventMemory)
                           for memory in getattr(self, "memories", {}).values()):
            raise ValueError("SMDP collection requires SMDPEventMemory frame reserve")
        self._smdp_pending = None
        self._smdp_event_rows_last_interaction = 0
        self._smdp_policy_version = 0
        self._smdp_max_policy_lag = 0
        provider = getattr(self, '_rollout_context_provider', None)
        configure_sleeping = getattr(provider, 'set_sleeping_actor_omission', None)
        if callable(configure_sleeping):
            configure_sleeping(bool(enabled and self.skip_expert_policy_inference))

    @property
    def smdp_transition_count(self) -> int:
        return int(self._smdp_transition_count)

    @property
    def smdp_event_step_count(self) -> int:
        return int(self._smdp_event_step_count)

    @property
    def smdp_policy_version(self) -> int:
        return int(
            getattr(
                self,
                "_smdp_policy_version",
                0,
            )
        )

    @property
    def smdp_max_policy_lag(self) -> int:
        return int(
            getattr(
                self,
                "_smdp_max_policy_lag",
                0,
            )
        )

    def set_rollout_context_provider(
        self,
        provider,
    ) -> None:
        """Attach the runtime provider used by the standard skrl loop."""
        self._rollout_context_provider = provider
        self._current_online_decisions = None

    @staticmethod
    def _require_team_tensor(
        *,
        mapping,
        possible_agents,
        label,
    ):
        """Return a synchronized team tensor or fail explicitly."""
        if not possible_agents:
            raise ValueError(
                "possible_agents must not be empty"
            )

        first_uid = possible_agents[0]

        if first_uid not in mapping:
            raise KeyError(
                f"{label} is missing owner {first_uid}"
            )

        first = mapping[
            first_uid
        ]

        if not torch.is_tensor(
            first
        ):
            raise TypeError(
                f"{label} must contain tensors"
            )

        for uid in possible_agents[1:]:
            if uid not in mapping:
                raise KeyError(
                    f"{label} is missing owner {uid}"
                )

            other = mapping[
                uid
            ]

            if (
                not torch.is_tensor(other)
                or other.shape != first.shape
                or not torch.equal(
                    other,
                    first,
                )
            ):
                raise ValueError(
                    f"{label} must be identical for the whole team"
                )

        return first

    def act(
        self,
        observations,
        states,
        *,
        timestep: int,
        timesteps: int,
    ):
        """Standard skrl act() backed by structured MSSR graphs."""
        provider = getattr(
            self,
            "_rollout_context_provider",
            None,
        )

        if provider is None:
            return MAPPO.act(
                self,
                observations,
                states,
                timestep=timestep,
                timesteps=timesteps,
            )

        contexts = list(
            provider.current_action_contexts()
        )

        if not contexts:
            raise ValueError(
                "rollout context provider returned no environments"
            )

        if any(
            not isinstance(
                context,
                OnlineRolloutContext,
            )
            for context in contexts
        ):
            raise TypeError(
                "current_action_contexts must return "
                "OnlineRolloutContext objects"
            )

        batch_size = len(
            contexts
        )

        for uid in self.possible_agents:
            if uid not in observations:
                raise KeyError(
                    f"observations is missing owner {uid}"
                )

            value = observations[
                uid
            ]

            if (
                not torch.is_tensor(
                    value
                )
                or value.ndim < 1
                or value.shape[0]
                != batch_size
            ):
                raise ValueError(
                    "rollout context batch does not match "
                    f"observations[{uid}]"
                )

        policy_indexes = [
            index
            for index, context in enumerate(contexts)
            if not (
                getattr(self, "skip_expert_policy_inference", False)
                and not context.rollout_active
            )
        ]
        with torch.no_grad():
            policy_outputs = self.forward_shared_actor_batch(
                actor=self.shared_actor,
                graphs=[contexts[index].actor_graph for index in policy_indexes],
                routings=[contexts[index].routing_snapshot for index in policy_indexes],
                possible_agents=self.possible_agents,
            )
        actor_output_by_index = dict(zip(policy_indexes, policy_outputs, strict=True))

        decisions = []
        for index, context in enumerate(contexts):
            if getattr(self, "skip_expert_policy_inference", False) and not context.rollout_active:
                # This native row belongs to Expert control. Retain its real
                # graph/state identity and neutral storage ABI without sampling
                # a policy that cannot act or contribute an actor loss.
                zero = torch.zeros(1, device=self.device)
                sample = StructuredRolloutSample(
                    action=None,
                    hard_mask=torch.as_tensor(context.hard_mask, dtype=torch.bool,
                                              device=self.device).detach(),
                    owner_active_mask={uid: False for uid in self.possible_agents},
                    routing_snapshot=context.routing_snapshot,
                    graph_snapshot=context.actor_graph,
                    critic_graph_snapshot=context.critic_graph,
                    rollout_active=False,
                )
                decision = OnlineRolloutDecision(action=None,
                    owner_log_probs={uid: zero.clone() for uid in self.possible_agents},
                    global_value=zero.reshape(1, 1), structured_sample=sample)
            else:
                decision = self.build_online_rollout_decision(
                    actor_graph=context.actor_graph,
                    critic_graph=context.critic_graph,
                    routing_snapshot=context.routing_snapshot,
                    hard_mask=context.hard_mask,
                    actor_output=actor_output_by_index[index],
                    **({'availability_snapshot': context.availability_snapshot}
                       if context.availability_snapshot is not None else {}),
                )
            decisions.append(decision)

        # Assembly occupies native vector rows but contributes no policy
        # training data. Keep its actual sample/value for storage alignment.
        decisions = [
            replace(decision, structured_sample=replace(
                decision.structured_sample,
                rollout_active=context.rollout_active,
                owner_active_mask=(decision.structured_sample.owner_active_mask
                    if context.rollout_active else {
                        uid: False for uid in self.possible_agents
                    }),
            ))
            for decision, context in zip(decisions, contexts)
        ]

        env_actions = (
            provider.encode_env_actions(
                decisions
            )
        )

        if not isinstance(
            env_actions,
            dict,
        ):
            raise TypeError(
                "encode_env_actions must return a dict"
            )

        for uid in self.possible_agents:
            if uid not in env_actions:
                raise KeyError(
                    f"encoded actions are missing owner {uid}"
                )

            action = env_actions[
                uid
            ]

            if (
                not torch.is_tensor(
                    action
                )
                or action.ndim != 2
                or action.shape[0]
                != batch_size
            ):
                raise ValueError(
                    f"encoded actions[{uid}] must have "
                    "shape (num_envs, action_size)"
                )

        self._current_online_decisions = (
            decisions
        )

        outputs = {}

        for uid in self.possible_agents:
            outputs[
                uid
            ] = {
                "log_prob": torch.stack(
                    [
                        decision.owner_log_probs[
                            uid
                        ].reshape(1)
                        for decision in decisions
                    ],
                    dim=0,
                ),
                # Keeps stochastic-evaluation trainer surfaces
                # compatible without changing the environment ABI.
                "mean_actions": env_actions[
                    uid
                ],
            }

        return (
            env_actions,
            outputs,
        )

    @staticmethod
    def _smdp_owner_row(mapping, env_index: int):
        return {
            uid: value[env_index].detach().clone()
            for uid, value in mapping.items()
        }

    def _smdp_start_pending(self, *, env_index, decision, observations, states,
                            altruistic_sample, discount):
        if self._smdp_pending is None:
            self._smdp_pending = [None for _ in range(len(self._current_online_decisions))]
        self._smdp_pending[env_index] = _SMDPPendingTransition(
            decision=decision,
            observations=self._smdp_owner_row(
                observations,
                env_index,
            ),
            states=self._smdp_owner_row(
                states,
                env_index,
            ),
            altruistic_sample=altruistic_sample,
            rewards=DiscountedOptionAccumulator(
                discount=discount
            ),
            policy_version=int(
                getattr(
                    self,
                    "_smdp_policy_version",
                    0,
                )
            ),
        )

    def _smdp_padding_row(self, *, env_index, decision, observations, states,
                          next_observations, next_states, next_graph,
                          altruistic_sample):
        base = next(iter(observations.values()))
        zero = torch.zeros((), dtype=base.dtype, device=base.device)
        structured = replace(
            decision.structured_sample,
            action=None,
            owner_active_mask={uid: False for uid in self.possible_agents},
            rollout_active=False,
            duration_steps=0,
            # record_online_rollout_batch owns the one detached next-graph
            # snapshot. This temporary event row is consumed synchronously.
            next_critic_graph_snapshot=None,
        )
        padded_decision = replace(
            decision,
            action=None,
            owner_log_probs={uid: zero.reshape(1) for uid in self.possible_agents},
            global_value=zero.reshape(1, 1),
            structured_sample=structured,
        )
        padded_altruistic = replace(
            altruistic_sample,
            owner_rewards={uid: torch.zeros_like(value)
                           for uid, value in altruistic_sample.owner_rewards.items()},
            mean_field_rewards={uid: torch.zeros_like(value)
                                for uid, value in altruistic_sample.mean_field_rewards.items()},
            next_owner_row_index=dict(altruistic_sample.owner_row_index),
        )
        return dict(
            decision=padded_decision,
            observations=self._smdp_owner_row(observations, env_index),
            states=self._smdp_owner_row(states, env_index),
            next_observations=self._smdp_owner_row(next_observations, env_index),
            next_states=self._smdp_owner_row(next_states, env_index),
            team_reward=zero,
            terminated=False,
            truncated=False,
            next_graph=next_graph,
            altruistic=padded_altruistic,
        )

    def _smdp_close_pending(self, *, env_index, next_observations, next_states,
                            next_graph, next_owner_row_index, terminated, truncated):
        pending = self._smdp_pending[env_index]
        if pending is None or pending.rewards.duration_steps <= 0:
            return None

        current_version = int(
            getattr(
                self,
                "_smdp_policy_version",
                0,
            )
        )
        policy_lag = (
            current_version
            - int(pending.policy_version)
        )

        if not actor_policy_lag_is_acceptable(
            start_version=int(pending.policy_version),
            current_version=current_version,
            max_lag=1,
        ):
            # Preserve the physical option, but never train PPO
            # on a structural decision sampled by a stale actor.
            self._smdp_stale_option_drop_count = (
                int(getattr(self, "_smdp_stale_option_drop_count", 0))
                + 1
            )
            print(
                "[C1 SMDP] drop stale option "
                f"env={env_index} "
                f"start_version={pending.policy_version} "
                f"current_version={current_version} "
                f"lag={policy_lag}",
                flush=True,
            )
            return None

        self._smdp_max_policy_lag = max(
            int(
                getattr(
                    self,
                    "_smdp_max_policy_lag",
                    0,
                )
            ),
            policy_lag,
        )
        structured = replace(
            pending.decision.structured_sample,
            duration_steps=int(pending.rewards.duration_steps),
            next_critic_graph_snapshot=None,
            rollout_active=True,
        )
        decision = replace(pending.decision, structured_sample=structured)
        altruistic = replace(
            pending.altruistic_sample,
            owner_rewards={uid: value.detach().clone()
                           for uid, value in pending.rewards.owner_rewards.items()},
            mean_field_rewards={uid: value.detach().clone()
                                for uid, value in pending.rewards.mean_field_rewards.items()},
            next_owner_row_index=dict(next_owner_row_index),
        )
        return dict(
            decision=decision,
            observations=pending.observations,
            states=pending.states,
            next_observations=next_observations,
            next_states=next_states,
            team_reward=pending.rewards.team_reward.detach().clone(),
            terminated=bool(terminated),
            truncated=bool(truncated),
            next_graph=next_graph,
            altruistic=altruistic,
        )

    def _smdp_store_event_step(self, *, rows, decisions, observations, states,
                               next_observations, next_states, next_graphs,
                               altruistic_samples, env_actions):
        if not any(row is not None for row in rows):
            return False
        materialized = []
        for env_index, row in enumerate(rows):
            if row is None:
                row = self._smdp_padding_row(
                    env_index=env_index, decision=decisions[env_index],
                    observations=observations, states=states,
                    next_observations=next_observations, next_states=next_states,
                    next_graph=next_graphs[env_index],
                    altruistic_sample=altruistic_samples[env_index],
                )
            materialized.append(row)

        def stack_owner(field):
            return {
                uid: torch.stack([row[field][uid] for row in materialized], dim=0)
                for uid in self.possible_agents
            }

        device = next(iter(observations.values())).device
        batch = len(materialized)
        self.record_online_rollout_batch(
            decisions=[row["decision"] for row in materialized],
            observations=stack_owner("observations"),
            states=stack_owner("states"),
            env_actions={uid: torch.zeros((batch, env_actions[uid].shape[1]),
                                          dtype=env_actions[uid].dtype,
                                          device=env_actions[uid].device)
                         for uid in self.possible_agents},
            team_rewards=torch.stack([row["team_reward"] for row in materialized]).reshape(batch, 1),
            next_observations=stack_owner("next_observations"),
            next_states=stack_owner("next_states"),
            terminated=torch.tensor([row["terminated"] for row in materialized],
                                    dtype=torch.bool, device=device).reshape(batch, 1),
            truncated=torch.tensor([row["truncated"] for row in materialized],
                                   dtype=torch.bool, device=device).reshape(batch, 1),
            next_critic_graphs=[row["next_graph"] for row in materialized],
            altruistic_samples=[row["altruistic"] for row in materialized],
        )
        active = sum(row["decision"].structured_sample.duration_steps > 0
                     for row in materialized)
        self._smdp_transition_count += int(active)
        self._smdp_event_step_count += 1
        self._smdp_event_rows_last_interaction += 1
        return True

    def _record_smdp_transition(self, *, observations, states, actions,
                                next_observations, next_states, terminated, truncated,
                                decisions, next_critic_graphs, team_rewards):
        batch = len(decisions)
        if self._smdp_pending is None or len(self._smdp_pending) != batch:
            self._smdp_pending = [None for _ in range(batch)]

        altruistic_samples = self._build_altruistic_rollout_step(expected_num_envs=batch)
        if altruistic_samples is None:
            raise RuntimeError("SMDP APPO collection requires altruistic rollout rewards")

        first_uid = self.possible_agents[0]
        discount_cfg = getattr(self.cfg, "discount_factor", {})
        discount = float(discount_cfg[first_uid] if isinstance(discount_cfg, dict)
                         else discount_cfg)

        # A fresh decision epoch closes the previous option at the pre-step state.
        pre_rows = [None for _ in range(batch)]
        for env_index, decision in enumerate(decisions):
            if decision.structured_sample.rollout_active and self._smdp_pending[env_index] is not None:
                sample = altruistic_samples[env_index]
                pre_rows[env_index] = self._smdp_close_pending(
                    env_index=env_index,
                    next_observations=self._smdp_owner_row(observations, env_index),
                    next_states=self._smdp_owner_row(states, env_index),
                    next_graph=decision.structured_sample.critic_graph_snapshot,
                    next_owner_row_index=sample.owner_row_index,
                    terminated=False, truncated=False,
                )
                self._smdp_pending[env_index] = None
        self._smdp_store_event_step(
            rows=pre_rows, decisions=decisions, observations=observations, states=states,
            next_observations=observations, next_states=states,
            next_graphs=[d.structured_sample.critic_graph_snapshot for d in decisions],
            altruistic_samples=altruistic_samples, env_actions=actions,
        )

        # Start newly sampled decisions after the previous epoch is closed.
        for env_index, decision in enumerate(decisions):
            if decision.structured_sample.rollout_active:
                self._smdp_start_pending(
                    env_index=env_index, decision=decision, observations=observations,
                    states=states, altruistic_sample=altruistic_samples[env_index],
                    discount=discount,
                )

        # This physical frame belongs to the currently open decision epoch.
        for env_index, pending in enumerate(self._smdp_pending):
            if pending is None:
                continue
            sample = altruistic_samples[env_index]
            pending.rewards.add(
                team_reward=team_rewards[env_index, 0],
                owner_rewards=sample.owner_rewards,
                mean_field_rewards=sample.mean_field_rewards,
            )

        # Episode boundaries close on the captured post-step state.
        post_rows = [None for _ in range(batch)]
        for env_index, pending in enumerate(self._smdp_pending):
            if pending is None:
                continue
            is_terminated = bool(terminated[env_index, 0].item())
            is_truncated = bool(truncated[env_index, 0].item())
            if not (is_terminated or is_truncated):
                continue
            sample = altruistic_samples[env_index]
            next_rows = sample.next_owner_row_index or sample.owner_row_index
            post_rows[env_index] = self._smdp_close_pending(
                env_index=env_index,
                next_observations=self._smdp_owner_row(next_observations, env_index),
                next_states=self._smdp_owner_row(next_states, env_index),
                next_graph=next_critic_graphs[env_index],
                next_owner_row_index=next_rows,
                terminated=is_terminated, truncated=is_truncated,
            )
            self._smdp_pending[env_index] = None
        self._smdp_store_event_step(
            rows=post_rows, decisions=decisions, observations=observations, states=states,
            next_observations=next_observations, next_states=next_states,
            next_graphs=next_critic_graphs, altruistic_samples=altruistic_samples,
            env_actions=actions,
        )

    def record_transition(
        self,
        *,
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        terminated,
        truncated,
        infos,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Standard skrl transition hook for structured MSSR MAPPO."""
        provider = getattr(
            self,
            "_rollout_context_provider",
            None,
        )

        decisions = getattr(
            self,
            "_current_online_decisions",
            None,
        )

        if (
            provider is None
            or decisions is None
        ):
            MAPPO.record_transition(
                self,
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
            )
            return

        # Preserve skrl episode/reward tracking without invoking its
        # fixed-action MAPPO storage path.
        MultiAgent.record_transition(
            self,
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        team_rewards = self._require_team_tensor(
            mapping=rewards,
            possible_agents=self.possible_agents,
            label="team reward",
        ).clone()

        team_terminated = self._require_team_tensor(
            mapping=terminated,
            possible_agents=self.possible_agents,
            label="team terminated",
        )

        team_truncated = self._require_team_tensor(
            mapping=truncated,
            possible_agents=self.possible_agents,
            label="team truncated",
        )

        next_critic_graphs = list(
            provider.current_critic_graphs()
        )

        if (
            len(next_critic_graphs)
            != len(decisions)
        ):
            raise ValueError(
                "next privileged graph batch does not match "
                "the online decision batch"
            )

        rewards_shaper = getattr(
            self.cfg,
            "rewards_shaper",
            None,
        )

        if rewards_shaper is not None:
            team_rewards = rewards_shaper(
                team_rewards,
                timestep,
                timesteps,
            )

        first_uid = self.possible_agents[
            0
        ]

        time_limit_cfg = getattr(
            self.cfg,
            "time_limit_bootstrap",
            {},
        )

        time_limit_bootstrap = (
            time_limit_cfg[
                first_uid
            ]
            if isinstance(
                time_limit_cfg,
                dict,
            )
            else bool(
                time_limit_cfg
            )
        )

        if (
            time_limit_bootstrap
            and team_truncated.any()
        ):
            with torch.no_grad():
                next_values, _ = (
                    self.shared_critic.act(
                        {
                            "states": next_critic_graphs,
                        },
                        role="value",
                    )
                )

                value_preprocessor = getattr(
                    self,
                    "_value_preprocessor",
                    {},
                )

                if (
                    isinstance(
                        value_preprocessor,
                        dict,
                    )
                    and first_uid
                    in value_preprocessor
                ):
                    next_values = (
                        value_preprocessor[
                            first_uid
                        ](
                            next_values,
                            inverse=True,
                        )
                    )

            discount_cfg = getattr(
                self.cfg,
                "discount_factor",
                {},
            )

            discount = (
                discount_cfg[
                    first_uid
                ]
                if isinstance(
                    discount_cfg,
                    dict,
                )
                else float(
                    discount_cfg
                )
            )

            team_rewards = (
                team_rewards
                + discount
                * next_values
                * team_truncated
            )

        if bool(getattr(self, "_smdp_options_enabled", False)):
            self._record_smdp_transition(
                observations=observations, states=states, actions=actions,
                next_observations=next_observations, next_states=next_states,
                terminated=team_terminated, truncated=team_truncated,
                decisions=decisions, next_critic_graphs=next_critic_graphs,
                team_rewards=team_rewards,
            )
            self._current_online_decisions = None
            return

        self.record_online_rollout_batch(
            decisions=decisions,
            observations=observations,
            states=states,
            env_actions=actions,
            team_rewards=team_rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=team_terminated,
            truncated=team_truncated,
            next_critic_graphs=next_critic_graphs,
        )

        # Only clear after successful storage.
        self._current_online_decisions = None

    def set_current_next_critic_graph(
        self,
        graph,
    ) -> None:
        """Store the privileged next-state graph used for GAE bootstrap."""
        self._current_next_critic_graph = graph

    def evaluate_structured_critic_batch(
        self,
        *,
        structured_batch,
        owner_tensor_batch,
    ):
        """Evaluate the global critic on privileged rollout snapshots.

        Structured MARL rollouts use critic_graph_snapshot. The legacy
        tensor-state path remains temporarily available for existing
        compatibility tests and pre-runtime integration.
        """
        critic_graphs = [
            sample.critic_graph_snapshot
            for sample in structured_batch
        ]

        if (
            critic_graphs
            and all(
                graph is not None
                for graph in critic_graphs
            )
        ):
            predicted_values, _ = self.shared_critic.act(
                {
                    "states": critic_graphs,
                },
                role="value",
            )

            return predicted_values

        if owner_tensor_batch is None:
            raise ValueError(
                "Structured critic minibatch has neither privileged "
                "graphs nor legacy tensor states"
            )

        first_uid = next(
            iter(self.possible_agents)
        )

        sampled_states = (
            owner_tensor_batch[first_uid][0]
        )

        critic_inputs = {
            "states": self._state_preprocessor[
                first_uid
            ](
                sampled_states
            )
        }

        predicted_values, _ = self.shared_critic.act(
            critic_inputs,
            role="value",
        )

        return predicted_values

    @staticmethod
    def build_shared_model_registry(
        *,
        possible_agents: list[str],
        actor: Any,
        critic: Any,
    ) -> dict[str, dict[str, Any]]:
        """Register one shared actor and one shared critic for N agents."""
        return {
            uid: {
                "policy": actor,
                "value": critic,
            }
            for uid in possible_agents
        }

    @staticmethod
    def forward_shared_actor(
        *,
        actor: Any,
        graph: Any,
        possible_agents: list[str],
        routing: Any = None,
    ) -> dict[str, Any]:
        """Run one global actor forward and return MARL distribution views."""
        if routing is None:
            return actor.forward_all_owners(
                graph=graph,
                possible_agents=possible_agents,
            )

        try:
            return actor.forward_all_owners(
                graph=graph,
                routing=routing,
                possible_agents=possible_agents,
            )
        except TypeError as error:
            # Compatibility for existing minimal test actors that predate
            # the explicit routing argument. Production BC-v4 wrappers
            # consume routing explicitly.
            if "unexpected keyword argument 'routing'" not in str(
                error
            ):
                raise

            return actor.forward_all_owners(
                graph=graph,
                possible_agents=possible_agents,
            )

    @staticmethod
    def forward_shared_actor_batch(
        *,
        actor: Any,
        graphs: list[Any],
        routings: list[Any],
        possible_agents: list[str],
    ) -> list[dict[str, Any]]:
        """Prefer one batched actor call, with a serial compatibility fallback."""
        graphs = list(graphs)
        routings = list(routings)
        if len(graphs) != len(routings):
            raise ValueError(
                "graph and routing batches must have the same size"
            )
        if not graphs:
            return []

        batched = getattr(
            actor,
            "forward_all_owners_batch",
            None,
        )
        if callable(batched):
            kwargs = {
                "graphs": graphs,
                "routings": routings,
                "possible_agents": possible_agents,
            }
            if getattr(actor, "prefer_lazy_payloads", False):
                kwargs["lazy_payloads"] = True
            return list(batched(**kwargs))

        return [
            MSSRMAPPO.forward_shared_actor(
                actor=actor,
                graph=graph,
                routing=routing,
                possible_agents=possible_agents,
            )
            for graph, routing in zip(
                graphs, routings, strict=True
            )
        ]

    @staticmethod
    def build_shared_optimizers(
        *,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        actor_learning_rate: float,
        critic_learning_rate: float,
    ) -> tuple[torch.optim.Adam, torch.optim.Adam]:
        """Build one optimizer for the shared actor and one for the global critic."""
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=actor_learning_rate,
        )
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=critic_learning_rate,
        )
        return actor_optimizer, critic_optimizer

    def compute_global_bootstrap_value(self) -> torch.Tensor:
        """Evaluate the global critic on the final rollout state."""
        first_uid = next(
            iter(self.possible_agents)
        )

        with torch.no_grad(), torch.autocast(
            device_type=self._device_type,
            enabled=self.cfg.mixed_precision,
        ):
            critic_graph = getattr(
                self,
                "_current_next_critic_graph",
                None,
            )

            if critic_graph is not None:
                last_values, _ = self.shared_critic.act(
                    {
                        "states": critic_graph,
                    },
                    role="value",
                )
            else:
                # Compatibility fallback for the old fixed-tensor path.
                inputs = {
                    "observations": self._observation_preprocessor[
                        first_uid
                    ](
                        self._current_next_observations[
                            first_uid
                        ]
                    ),
                    "states": self._state_preprocessor[
                        first_uid
                    ](
                        self._current_next_states[
                            first_uid
                        ]
                    ),
                }

                last_values, _ = self.shared_critic.act(
                    inputs,
                    role="value",
                )

            return self._value_preprocessor[
                first_uid
            ](
                last_values,
                inverse=True,
            )

    @staticmethod
    def compute_global_value_loss(
        *,
        predicted_values: torch.Tensor,
        sampled_values: torch.Tensor,
        sampled_returns: torch.Tensor,
        value_clip: float,
        value_loss_scale: float,
    ) -> torch.Tensor:
        """Compute the single centralized critic value loss."""
        if value_clip > 0:
            predicted_values = sampled_values + torch.clamp(
                predicted_values - sampled_values,
                min=-value_clip,
                max=value_clip,
            )

        return value_loss_scale * torch.nn.functional.mse_loss(
            sampled_returns,
            predicted_values,
        )

    @staticmethod
    def build_owner_activity_masks(
        *,
        structured_batch,
        possible_agents,
        device=None,
    ) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
        """Build per-timestep owner masks and identify active owners."""
        masks = {
            uid: torch.tensor(
                [
                    bool(sample.owner_active_mask.get(uid, False))
                    for sample in structured_batch
                ],
                dtype=torch.bool,
                device=device,
            )
            for uid in possible_agents
        }

        active_owners = tuple(
            uid
            for uid in possible_agents
            if bool(masks[uid].any().item())
        )

        return masks, active_owners

    @staticmethod
    def sample_owner_tensor_map_minibatches(
        *,
        tensors_by_owner,
        indexes,
        mini_batches: int,
    ):
        indexes = (
            indexes.tolist()
            if hasattr(
                indexes,
                "tolist",
            )
            else list(
                indexes
            )
        )

        quotient, remainder = divmod(
            len(
                indexes
            ),
            mini_batches,
        )

        flat = {
            uid: tensor.reshape(
                -1,
                1,
            )
            for uid, tensor
            in tensors_by_owner.items()
        }

        batches = []

        start = 0

        for batch_index in range(
            mini_batches
        ):
            size = (
                quotient
                + (
                    1
                    if batch_index
                    < remainder
                    else 0
                )
            )

            stop = (
                start
                + size
            )

            batch_indexes = (
                indexes[
                    start:stop
                ]
            )

            batches.append(
                {
                    uid: tensor[
                        batch_indexes
                    ]
                    for uid, tensor
                    in flat.items()
                }
            )

            start = stop

        return batches

    @staticmethod
    def evaluate_altruistic_critic_minibatch(
        *,
        structured_batch,
        altruistic_batch,
        possible_agents,
        owner_critic,
        mean_field_critic,
        skip_inactive=False,
        padding_reference=None,
        diagnostics=None,
    ):
        """Evaluate critics, optionally omitting rows masked out of both losses.

        Padding retains its physical row position and owner mapping. Its zero
        placeholders are never values for GAE/bootstrap, only unused loss rows.
        The public default retains full evaluation for other callers.
        """

        agents = tuple(
            possible_agents
        )

        if len(
            structured_batch
        ) != len(
            altruistic_batch
        ):
            raise ValueError(
                "structured/altruistic minibatch alignment mismatch"
            )

        owner_values = {
            uid: []
            for uid in agents
        }

        mean_field_values = {
            uid: []
            for uid in agents
        }

        for (
            structured_sample,
            altruistic_sample,
        ) in zip(
            structured_batch,
            altruistic_batch,
            strict=True,
        ):
            graph = getattr(
                structured_sample,
                "critic_graph_snapshot",
                None,
            )

            if graph is None:
                raise ValueError(
                    "critic graph is required for APPO critic training"
                )

            row_map = getattr(
                altruistic_sample,
                "owner_row_index",
                None,
            )

            if (
                not isinstance(
                    row_map,
                    dict,
                )
                or set(
                    row_map
                )
                != set(
                    agents
                )
            ):
                raise ValueError(
                    "owner-row mapping is invalid"
                )

            active = bool(getattr(structured_sample, "rollout_active", True))
            if diagnostics is not None:
                key = "valid_rows" if active else "padding_rows"
                diagnostics[key] = diagnostics.get(key, 0) + 1
            if skip_inactive and not active:
                if diagnostics is not None:
                    diagnostics["padding_rows_skipped"] = diagnostics.get("padding_rows_skipped", 0) + 1
                for uid in agents:
                    owner_values[uid].append(None)
                    mean_field_values[uid].append(None)
                continue

            if diagnostics is not None:
                diagnostics["owner_graph_forwards"] = diagnostics.get("owner_graph_forwards", 0) + 1
                diagnostics["mean_field_graph_forwards"] = diagnostics.get("mean_field_graph_forwards", 0) + 1

            owner_forward_all = getattr(
                owner_critic,
                "forward_all_owners",
                None,
            )
            mean_field_forward_all = getattr(
                mean_field_critic,
                "forward_all_owners",
                None,
            )

            owner_all_values = (
                owner_forward_all(graph)
                if callable(owner_forward_all)
                else None
            )
            mean_field_all_values = (
                mean_field_forward_all(graph)
                if callable(mean_field_forward_all)
                else None
            )

            for uid in agents:
                row = row_map[
                    uid
                ]

                owner_value = (
                    owner_all_values[row]
                    if owner_all_values is not None
                    else owner_critic.forward_owner(
                        graph,
                        owner_index=row,
                    )
                )

                mf_value = (
                    mean_field_all_values[row]
                    if mean_field_all_values is not None
                    else mean_field_critic.forward_owner(
                        graph,
                        owner_index=row,
                    )
                )

                if (
                    owner_value.numel()
                    != 1
                    or mf_value.numel()
                    != 1
                ):
                    raise ValueError(
                        "owner-conditioned critics must return scalar values"
                    )

                owner_values[
                    uid
                ].append(
                    owner_value.reshape(
                        1
                    )
                )

                mean_field_values[
                    uid
                ].append(
                    mf_value.reshape(
                        1
                    )
                )

        for value_map in (owner_values, mean_field_values):
            for uid, values in value_map.items():
                if any(value is None for value in values):
                    reference = next((value for value in values if value is not None), padding_reference)
                    if reference is None:
                        raise ValueError("all-padding critic minibatch requires padding_reference")
                    value_map[uid] = [reference.new_zeros(1) if value is None else value for value in values]

        return (
            {
                uid: torch.stack(
                    values,
                    dim=0,
                )
                for uid, values
                in owner_values.items()
            },
            {
                uid: torch.stack(
                    values,
                    dim=0,
                )
                for uid, values
                in mean_field_values.items()
            },
        )

    @staticmethod
    def evaluate_altruistic_bootstrap_values(
        *,
        next_critic_graphs,
        altruistic_sidecar,
        possible_agents,
        owner_critic,
        mean_field_critic,
        structured_samples=None,
    ):
        """Evaluate V_i(s_T+1) and V_i^MF(s_T+1).

        The final bootstrap uses the explicit next-state owner-row
        mapping recorded with the last rollout transition. The current
        state's row mapping is deliberately not reused.
        """
        agents = tuple(
            possible_agents
        )

        if (
            not agents
            or len(
                set(
                    agents
                )
            )
            != len(
                agents
            )
        ):
            raise ValueError(
                "possible_agents must contain unique owners"
            )

        num_envs = getattr(
            altruistic_sidecar,
            "num_envs",
            None,
        )

        if (
            num_envs is None
            or num_envs <= 0
            or len(
                altruistic_sidecar
            )
            <= 0
        ):
            raise ValueError(
                "altruistic rollout sidecar is empty"
            )

        graphs = tuple(
            next_critic_graphs
        )

        if len(
            graphs
        ) != num_envs:
            raise ValueError(
                "next critic graph environment count mismatch"
            )

        flat = (
            altruistic_sidecar
            .flatten()
        )

        if len(
            flat
        ) < num_envs:
            raise ValueError(
                "altruistic rollout alignment is invalid"
            )

        final_samples = list(flat[
            -num_envs:
        ])

        if structured_samples is not None:
            if len(structured_samples) != len(flat) or len(flat) % num_envs:
                raise ValueError("structured/bootstrap rollout alignment is invalid")
            graphs = list(graphs)
            for env_index in range(num_envs):
                final_samples[env_index] = None
                for index in range(len(flat) - num_envs + env_index, -1, -num_envs):
                    transition = structured_samples[index]
                    if transition.rollout_active and transition.duration_steps > 0:
                        graphs[env_index] = transition.next_critic_graph_snapshot
                        final_samples[env_index] = flat[index]
                        break

        owner_values = {
            uid: []
            for uid in agents
        }

        mean_field_values = {
            uid: []
            for uid in agents
        }

        with torch.no_grad():
            for (
                graph,
                sample,
            ) in zip(
                graphs,
                final_samples,
                strict=True,
            ):
                if sample is None:
                    # This env contributes only padding, hence no bootstrap
                    # target. Do not evaluate its still-open physical option.
                    for uid in agents:
                        zero = torch.zeros_like(flat[0].owner_rewards[uid]).reshape(1)
                        owner_values[uid].append(zero)
                        mean_field_values[uid].append(zero.clone())
                    continue
                if graph is None:
                    raise ValueError(
                        "next critic graph must not be None"
                    )

                row_map = getattr(
                    sample,
                    "next_owner_row_index",
                    None,
                )

                if (
                    not isinstance(
                        row_map,
                        dict,
                    )
                    or set(
                        row_map
                    )
                    != set(
                        agents
                    )
                ):
                    raise ValueError(
                        "next-state owner-row mapping is missing or invalid"
                    )

                owner_forward_all = getattr(
                    owner_critic,
                    "forward_all_owners",
                    None,
                )
                mean_field_forward_all = getattr(
                    mean_field_critic,
                    "forward_all_owners",
                    None,
                )
                owner_all_values = (
                    owner_forward_all(graph)
                    if callable(owner_forward_all)
                    else None
                )
                mean_field_all_values = (
                    mean_field_forward_all(graph)
                    if callable(mean_field_forward_all)
                    else None
                )

                for uid in agents:
                    owner_index = (
                        row_map[
                            uid
                        ]
                    )

                    owner_value = (
                        owner_all_values[owner_index]
                        if owner_all_values is not None
                        else owner_critic.forward_owner(
                            graph,
                            owner_index=(
                                owner_index
                            ),
                        )
                    )

                    mean_field_value = (
                        mean_field_all_values[owner_index]
                        if mean_field_all_values is not None
                        else mean_field_critic.forward_owner(
                            graph,
                            owner_index=(
                                owner_index
                            ),
                        )
                    )

                    for (
                        label,
                        value,
                    ) in (
                        (
                            "owner bootstrap critic",
                            owner_value,
                        ),
                        (
                            "mean-field bootstrap critic",
                            mean_field_value,
                        ),
                    ):
                        if (
                            not torch.is_tensor(
                                value
                            )
                            or value.numel()
                            != 1
                        ):
                            raise ValueError(
                                f"{label} must return exactly one scalar"
                            )

                    owner_values[
                        uid
                    ].append(
                        owner_value
                        .detach()
                        .reshape(
                            1
                        )
                    )

                    mean_field_values[
                        uid
                    ].append(
                        mean_field_value
                        .detach()
                        .reshape(
                            1
                        )
                    )

        return (
            {
                uid: torch.stack(
                    owner_values[
                        uid
                    ],
                    dim=0,
                ).reshape(
                    num_envs,
                    1,
                )
                for uid in agents
            },
            {
                uid: torch.stack(
                    mean_field_values[
                        uid
                    ],
                    dim=0,
                ).reshape(
                    num_envs,
                    1,
                )
                for uid in agents
            },
        )

    @staticmethod
    def evaluate_altruistic_rollout_values(
        *,
        structured_samples,
        altruistic_sidecar,
        possible_agents,
        owner_critic,
        mean_field_critic,
        next_state=False,
    ):
        """Evaluate V_i(s_t) and V_i^MF(s_t) over a recorded rollout.

        Owner identity is resolved exclusively through the explicit
        owner->graph-row mapping stored by the altruistic rollout
        sidecar. Owner names are never interpreted as tensor indexes.

        Returned values are detached because they are GAE target
        inputs. Critic gradients are computed later from critic losses,
        not through GAE target construction.
        """
        agents = tuple(
            possible_agents
        )

        if (
            not agents
            or len(
                set(
                    agents
                )
            )
            != len(
                agents
            )
        ):
            raise ValueError(
                "possible_agents must contain unique owners"
            )

        structured = tuple(
            structured_samples
        )

        num_steps = len(
            altruistic_sidecar
        )

        num_envs = getattr(
            altruistic_sidecar,
            "num_envs",
            None,
        )

        if (
            num_steps <= 0
            or num_envs is None
            or num_envs <= 0
        ):
            raise ValueError(
                "altruistic rollout sidecar is empty"
            )

        altruistic = (
            altruistic_sidecar
            .flatten()
        )

        expected_samples = (
            num_steps
            * num_envs
        )

        if (
            len(
                structured
            )
            != expected_samples
            or len(
                altruistic
            )
            != expected_samples
        ):
            raise ValueError(
                "structured/altruistic rollout alignment mismatch"
            )

        owner_samples = {
            uid: []
            for uid in agents
        }

        mean_field_samples = {
            uid: []
            for uid in agents
        }

        with torch.no_grad():
            for (
                structured_sample,
                altruistic_sample,
            ) in zip(
                structured,
                altruistic,
                strict=True,
            ):
                graph = getattr(
                    structured_sample,
                    "next_critic_graph_snapshot" if next_state else "critic_graph_snapshot",
                    None,
                )

                if graph is None:
                    raise ValueError(
                        "structured rollout sample is missing critic graph"
                    )

                row_map = getattr(
                    altruistic_sample,
                    "next_owner_row_index" if next_state else "owner_row_index",
                    None,
                )

                if (
                    not isinstance(
                        row_map,
                        dict,
                    )
                    or set(
                        row_map
                    )
                    != set(
                        agents
                    )
                ):
                    raise ValueError(
                        "altruistic owner-row mapping is invalid"
                    )

                owner_forward_all = getattr(
                    owner_critic,
                    "forward_all_owners",
                    None,
                )
                mean_field_forward_all = getattr(
                    mean_field_critic,
                    "forward_all_owners",
                    None,
                )
                owner_all_values = (
                    owner_forward_all(graph)
                    if callable(owner_forward_all)
                    else None
                )
                mean_field_all_values = (
                    mean_field_forward_all(graph)
                    if callable(mean_field_forward_all)
                    else None
                )

                for uid in agents:
                    owner_index = row_map[
                        uid
                    ]

                    owner_value = (
                        owner_all_values[owner_index]
                        if owner_all_values is not None
                        else owner_critic.forward_owner(
                            graph,
                            owner_index=(
                                owner_index
                            ),
                        )
                    )

                    mean_field_value = (
                        mean_field_all_values[owner_index]
                        if mean_field_all_values is not None
                        else mean_field_critic.forward_owner(
                            graph,
                            owner_index=(
                                owner_index
                            ),
                        )
                    )

                    for (
                        label,
                        value,
                    ) in (
                        (
                            "owner critic",
                            owner_value,
                        ),
                        (
                            "mean-field critic",
                            mean_field_value,
                        ),
                    ):
                        if (
                            not torch.is_tensor(
                                value
                            )
                            or value.numel()
                            != 1
                        ):
                            raise ValueError(
                                f"{label} must return one scalar "
                                "for one owner in one graph"
                            )

                    owner_samples[
                        uid
                    ].append(
                        owner_value
                        .detach()
                        .reshape(
                            1
                        )
                    )

                    mean_field_samples[
                        uid
                    ].append(
                        mean_field_value
                        .detach()
                        .reshape(
                            1
                        )
                    )

        shape = (
            num_steps,
            num_envs,
            1,
        )

        owner_values = {
            uid: torch.stack(
                owner_samples[
                    uid
                ],
                dim=0,
            ).reshape(
                shape
            )
            for uid in agents
        }

        mean_field_values = {
            uid: torch.stack(
                mean_field_samples[
                    uid
                ],
                dim=0,
            ).reshape(
                shape
            )
            for uid in agents
        }

        return (
            owner_values,
            mean_field_values,
        )

    @staticmethod
    def compute_altruistic_rollout_targets(
        *,
        sidecar,
        possible_agents,
        owner_values,
        mean_field_values,
        owner_bootstrap_values,
        mean_field_bootstrap_values,
        dones,
        discount_factor: float,
        lambda_coefficient: float,
        alpha,
        duration_steps=None,
        terminated=None,
        owner_next_values=None,
        mean_field_next_values=None,
    ):
        """Build APPO GAE targets from the recorded local-reward rollout.

        This is the common bridge between rollout collection and PPO:

            R_i       + V_i       -> A_i
            R_i^MF    + V_i^MF    -> A_i^MF
            A_i + alpha A_i^MF    -> actor advantage

        Owner and mean-field critic return targets remain independent.
        """
        from drl.MARL.altruism import (
            AltruisticGAEResult,
            combine_altruistic_advantages,
            compute_gae_channel,
        )

        agents = tuple(
            possible_agents
        )

        if (
            not agents
            or len(
                set(
                    agents
                )
            )
            != len(
                agents
            )
        ):
            raise ValueError(
                "possible_agents must contain unique owners"
            )

        num_steps = len(
            sidecar
        )

        num_envs = getattr(
            sidecar,
            "num_envs",
            None,
        )

        if (
            num_steps <= 0
            or num_envs is None
            or num_envs <= 0
        ):
            raise ValueError(
                "altruistic rollout sidecar is empty"
            )

        flat = sidecar.flatten()

        if len(
            flat
        ) != num_steps * num_envs:
            raise ValueError(
                "altruistic rollout sidecar alignment is invalid"
            )

        expected_shape = (
            num_steps,
            num_envs,
            1,
        )

        bootstrap_shape = (
            num_envs,
            1,
        )

        if duration_steps is None:
            duration_steps = torch.ones(expected_shape, dtype=torch.long, device=dones.device)
        elif (
            not torch.is_tensor(duration_steps)
            or tuple(duration_steps.shape) != expected_shape
            or duration_steps.device != dones.device
        ):
            raise ValueError(f"duration_steps must have shape {expected_shape} on dones.device")

        if (
            not torch.is_tensor(
                dones
            )
            or tuple(
                dones.shape
            )
            != expected_shape
        ):
            raise ValueError(
                "dones must have shape "
                f"{expected_shape}"
            )

        owner_rewards = {}
        mean_field_rewards = {}

        for uid in agents:
            owner_rewards[
                uid
            ] = torch.stack(
                [
                    sample.owner_rewards[
                        uid
                    ]
                    for sample in flat
                ]
            ).reshape(
                expected_shape
            )

            mean_field_rewards[
                uid
            ] = torch.stack(
                [
                    sample.mean_field_rewards[
                        uid
                    ]
                    for sample in flat
                ]
            ).reshape(
                expected_shape
            )

        for label, mapping in (
            (
                "owner_values",
                owner_values,
            ),
            (
                "mean_field_values",
                mean_field_values,
            ),
        ):
            if set(
                mapping
            ) != set(
                agents
            ):
                raise ValueError(
                    f"{label} must cover exactly possible_agents"
                )

            for uid in agents:
                if (
                    not torch.is_tensor(
                        mapping[
                            uid
                        ]
                    )
                    or tuple(
                        mapping[
                            uid
                        ].shape
                    )
                    != expected_shape
                ):
                    raise ValueError(
                        f"{label}[{uid}] must have shape "
                        f"{expected_shape}"
                    )

        for label, mapping in (
            (
                "owner_bootstrap_values",
                owner_bootstrap_values,
            ),
            (
                "mean_field_bootstrap_values",
                mean_field_bootstrap_values,
            ),
        ):
            if set(
                mapping
            ) != set(
                agents
            ):
                raise ValueError(
                    f"{label} must cover exactly possible_agents"
                )

            for uid in agents:
                if (
                    not torch.is_tensor(
                        mapping[
                            uid
                        ]
                    )
                    or tuple(
                        mapping[
                            uid
                        ].shape
                    )
                    != bootstrap_shape
                ):
                    raise ValueError(
                        f"{label}[{uid}] must have shape "
                        f"{bootstrap_shape}"
                    )

        owner_advantages = {}
        owner_returns = {}

        mean_field_advantages = {}
        mean_field_returns = {}

        for uid in agents:
            owner_result = compute_gae_channel(
                terminated=terminated,
                next_values=None if owner_next_values is None else owner_next_values[uid],
                rewards=(
                    owner_rewards[
                        uid
                    ]
                ),
                values=(
                    owner_values[
                        uid
                    ]
                ),
                bootstrap_value=(
                    owner_bootstrap_values[
                        uid
                    ]
                ),
                dones=dones,
                discount=(
                    discount_factor
                ),
                gae_lambda=(
                    lambda_coefficient
                ),
                duration_steps=duration_steps,
            )

            mean_field_result = (
                compute_gae_channel(
                    terminated=terminated,
                    next_values=None if mean_field_next_values is None else mean_field_next_values[uid],
                    rewards=(
                        mean_field_rewards[
                            uid
                        ]
                    ),
                    values=(
                        mean_field_values[
                            uid
                        ]
                    ),
                    bootstrap_value=(
                        mean_field_bootstrap_values[
                            uid
                        ]
                    ),
                    dones=dones,
                    discount=(
                        discount_factor
                    ),
                    gae_lambda=(
                        lambda_coefficient
                    ),
                    duration_steps=duration_steps,
                )
            )

            owner_advantages[
                uid
            ] = (
                owner_result
                .advantages
            )

            owner_returns[
                uid
            ] = (
                owner_result
                .returns
            )

            mean_field_advantages[
                uid
            ] = (
                mean_field_result
                .advantages
            )

            mean_field_returns[
                uid
            ] = (
                mean_field_result
                .returns
            )

        altruistic_advantages = (
            combine_altruistic_advantages(
                owner_advantages,
                mean_field_advantages,
                alpha=alpha,
            )
        )

        return AltruisticGAEResult(
            owner_advantages=(
                owner_advantages
            ),
            owner_returns=(
                owner_returns
            ),
            mean_field_advantages=(
                mean_field_advantages
            ),
            mean_field_returns=(
                mean_field_returns
            ),
            altruistic_advantages=(
                altruistic_advantages
            ),
        )

    @staticmethod
    def compute_owner_clipped_policy_losses(
        *,
        new_log_probs: dict[str, torch.Tensor],
        old_log_probs: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        ratio_clip: float,
        owner_active_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute independently clipped PPO losses for each active owner.

        ``advantages`` may be either:

        - one shared tensor: existing team-MAPPO behaviour;
        - one tensor per owner: selfish/altruistic APPO behaviour.

        PPO clipping itself is identical in both modes.
        """
        losses = {}

        if isinstance(
            advantages,
            dict,
        ):
            if set(
                advantages
            ) != set(
                new_log_probs
            ):
                raise ValueError(
                    "per-owner advantages must cover exactly "
                    "the policy owners"
                )

        for uid, next_log_prob in new_log_probs.items():
            owner_advantage = (
                advantages[
                    uid
                ]
                if isinstance(
                    advantages,
                    dict,
                )
                else advantages
            )

            if not torch.is_tensor(
                owner_advantage
            ):
                raise TypeError(
                    f"advantage for {uid} must be a tensor"
                )

            ratio = torch.exp(
                next_log_prob
                - old_log_probs[
                    uid
                ]
            )

            surrogate = (
                ratio
                * owner_advantage
            )

            surrogate_clipped = (
                torch.clamp(
                    ratio,
                    1.0 - ratio_clip,
                    1.0 + ratio_clip,
                )
                * owner_advantage
            )

            loss_terms = -torch.minimum(
                surrogate,
                surrogate_clipped,
            )

            if owner_active_masks is not None:
                mask = owner_active_masks[uid].to(
                    device=loss_terms.device,
                    dtype=torch.bool,
                )

                if not bool(mask.any().item()):
                    continue

                loss_terms = loss_terms[mask]

            losses[uid] = loss_terms.mean()

        return losses

    @staticmethod
    def compute_owner_actor_losses(
        *,
        new_log_probs: dict[str, torch.Tensor],
        old_log_probs: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        owner_entropies: dict[str, torch.Tensor],
        ratio_clip: float,
        entropy_loss_scale: float,
        owner_active_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Combine masked owner PPO loss and hierarchical entropy bonus."""
        policy_losses = MSSRMAPPO.compute_owner_clipped_policy_losses(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            ratio_clip=ratio_clip,
            owner_active_masks=owner_active_masks,
        )

        actor_losses = {}

        for uid, policy_loss in policy_losses.items():
            entropy = owner_entropies[uid]

            if entropy.ndim == 0:
                entropy_mean = entropy
            elif owner_active_masks is not None:
                mask = owner_active_masks[uid].to(
                    device=entropy.device,
                    dtype=torch.bool,
                )
                entropy_mean = entropy[mask].mean()
            else:
                entropy_mean = entropy.mean()

            actor_losses[uid] = (
                policy_loss
                - entropy_loss_scale * entropy_mean
            )

        return actor_losses

    @staticmethod
    def compute_and_broadcast_team_gae(
        *,
        memories: dict[str, Any],
        last_values: torch.Tensor,
        discount_factor: float,
        lambda_coefficient: float,
        time_limit_bootstrap: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute team GAE once and broadcast it to all owner memories."""
        if not memories:
            raise ValueError("memories must contain at least one owner")

        first_memory = next(iter(memories.values()))

        returns, advantages = compute_gae(
            rewards=first_memory.get_tensor_by_name("rewards"),
            terminated=first_memory.get_tensor_by_name("terminated"),
            truncated=first_memory.get_tensor_by_name("truncated"),
            values=first_memory.get_tensor_by_name("values"),
            last_values=last_values,
            discount_factor=discount_factor,
            lambda_coefficient=lambda_coefficient,
            time_limit_bootstrap=time_limit_bootstrap,
        )

        for memory in memories.values():
            memory.set_tensor_by_name("returns", returns)
            memory.set_tensor_by_name("advantages", advantages)

        return returns, advantages

    def reevaluate_structured_minibatch(
        self,
        *,
        structured_batch,
        owner_tensor_batch,
        evaluate_global_critic: bool = True,
    ):
        """Reevaluate stored semantic actions with the current actor and critic.

        Shared-actor contract
        ---------------------
        ``forward_all_owners`` performs one global H1/H2 forward for one graph
        snapshot and returns a normalized distribution description:

        - records
        - resource_payloads
        - structural_payloads
        - resource_log_std
        - structural_log_std

        The sampled semantic action itself remains in StructuredRolloutSample.
        """
        possible_agents = tuple(self.possible_agents)

        owner_log_prob_samples = {
            uid: []
            for uid in possible_agents
        }
        owner_entropy_samples = {
            uid: []
            for uid in possible_agents
        }

        policy_indexes = [
            index
            for index, sample in enumerate(structured_batch)
            if not (
                getattr(self, "skip_expert_policy_inference", False)
                and not sample.rollout_active
            )
        ]
        policy_outputs = self.forward_shared_actor_batch(
            actor=self.shared_actor,
            graphs=[structured_batch[index].graph_snapshot for index in policy_indexes],
            routings=[structured_batch[index].routing_snapshot for index in policy_indexes],
            possible_agents=list(possible_agents),
        )
        actor_output_by_index = dict(zip(policy_indexes, policy_outputs, strict=True))

        for index, sample in enumerate(structured_batch):
            if getattr(self, "skip_expert_policy_inference", False) and not sample.rollout_active:
                for uid in possible_agents:
                    owner_log_prob_samples[uid].append(torch.zeros(1, device=self.device))
                    owner_entropy_samples[uid].append(torch.zeros(1, device=self.device))
                continue
            actor_output = actor_output_by_index[index]

            required_keys = (
                "records",
                "resource_payloads",
                "structural_payloads",
                "resource_log_std",
                "structural_log_std",
            )

            missing = [
                key
                for key in required_keys
                if key not in actor_output
            ]

            if missing:
                raise KeyError(
                    "Shared actor output is missing required keys: "
                    + ", ".join(missing)
                )

            records = actor_output["records"]

            if torch.is_tensor(sample.hard_mask):
                hard_mask = sample.hard_mask
            else:
                if records:
                    device = records[0].presence_logit.device
                else:
                    device = self.device

                hard_mask = torch.as_tensor(
                    sample.hard_mask,
                    dtype=torch.bool,
                    device=device,
                )

            from drl.MARL.c1_feasibility import apply_availability
            actor_output = apply_availability(actor_output, sample.availability_snapshot)

            owner_log_probs = compute_owner_action_log_probs(
                records=records,
                resource_payloads=actor_output[
                    "resource_payloads"
                ],
                structural_payloads=actor_output[
                    "structural_payloads"
                ],
                action=sample.action,
                hard_mask=hard_mask,
                possible_agents=possible_agents,
                resource_log_std=actor_output[
                    "resource_log_std"
                ],
                structural_log_std=actor_output[
                    "structural_log_std"
                ],
            )

            owner_entropies = compute_owner_action_entropies(
                records=records,
                resource_payloads=actor_output[
                    "resource_payloads"
                ],
                structural_payloads=actor_output[
                    "structural_payloads"
                ],
                hard_mask=hard_mask,
                possible_agents=possible_agents,
                resource_log_std=actor_output[
                    "resource_log_std"
                ],
                structural_log_std=actor_output[
                    "structural_log_std"
                ],
            )

            for uid in possible_agents:
                owner_log_prob_samples[uid].append(
                    owner_log_probs[uid].reshape(1)
                )
                owner_entropy_samples[uid].append(
                    owner_entropies[uid].reshape(1)
                )

        new_log_probs = {
            uid: torch.stack(
                owner_log_prob_samples[uid],
                dim=0,
            )
            for uid in possible_agents
        }

        owner_entropies = {
            uid: torch.stack(
                owner_entropy_samples[uid],
                dim=0,
            )
            for uid in possible_agents
        }

        predicted_values = None

        if evaluate_global_critic:
            predicted_values = (
                self.evaluate_structured_critic_batch(
                    structured_batch=structured_batch,
                    owner_tensor_batch=owner_tensor_batch,
                )
            )

        return (
            new_log_probs,
            owner_entropies,
            predicted_values,
        )

    def coordinated_altruistic_update(
        self,
        *,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Run one fixed-alpha APPO update.

        The actor still uses the existing PPO loss. Only the
        advantage/critic branch differs from team-MAPPO.
        """
        del timestep
        del timesteps
        self._reset_update_metrics()

        if not self.memories:
            return

        required = (
            "altruistic_owner_critic",
            "altruistic_mean_field_critic",
            "owner_critic_optimizer",
            "mean_field_critic_optimizer",
            "altruistic_alpha",
            "_altruistic_rollout_sidecar",
        )

        missing = [
            name
            for name in required
            if not hasattr(
                self,
                name,
            )
        ]

        if missing:
            raise RuntimeError(
                "altruistic training is not fully configured: "
                + ", ".join(
                    missing
                )
            )

        first_uid = next(
            iter(
                self.possible_agents
            )
        )

        first_memory = (
            self.memories[
                first_uid
            ]
        )

        rollout_size = len(
            first_memory
        )

        structured_all = (
            self
            ._structured_rollout_sidecar
            .sample_by_index(
                range(
                    rollout_size
                )
            )
        )

        if len(
            self._altruistic_rollout_sidecar.flatten()
        ) != rollout_size:
            raise ValueError(
                "altruistic sidecar and skrl Memory are not aligned"
            )

        # Expert assembly contributes no learner rows. Preserve alignment checks
        # and post_interaction's normal clearing, but avoid evaluating critics
        # and GAE for a rollout that cannot produce an optimizer step.
        if not any(sample.rollout_active for sample in structured_all):
            return

        (
            owner_values,
            mean_field_values,
        ) = (
            self
            .evaluate_altruistic_rollout_values(
                structured_samples=(
                    structured_all
                ),
                altruistic_sidecar=(
                    self
                    ._altruistic_rollout_sidecar
                ),
                possible_agents=(
                    self.possible_agents
                ),
                owner_critic=(
                    self
                    .altruistic_owner_critic
                ),
                mean_field_critic=(
                    self
                    .altruistic_mean_field_critic
                ),
            )
        )

        (
            owner_bootstrap,
            mean_field_bootstrap,
        ) = (
            self
            .evaluate_altruistic_bootstrap_values(
                next_critic_graphs=(
                    self
                    ._current_next_critic_graph
                ),
                altruistic_sidecar=(
                    self
                    ._altruistic_rollout_sidecar
                ),
                possible_agents=(
                    self.possible_agents
                ),
                owner_critic=(
                    self
                    .altruistic_owner_critic
                ),
                mean_field_critic=(
                    self
                    .altruistic_mean_field_critic
                ),
                structured_samples=(structured_all if getattr(self, "_smdp_options_enabled", False) else None),
            )
        )

        terminated = (
            first_memory
            .get_tensor_by_name(
                "terminated"
            )
            .to(
                dtype=torch.bool
            )
        )

        truncated = (
            first_memory
            .get_tensor_by_name(
                "truncated"
            )
            .to(
                dtype=torch.bool
            )
        )

        dones = (
            terminated
            | truncated
        )

        owner_next_values = mean_field_next_values = None
        if bool((truncated & ~terminated).any()):
            owner_next_values, mean_field_next_values = self.evaluate_altruistic_rollout_values(
                structured_samples=structured_all,
                altruistic_sidecar=self._altruistic_rollout_sidecar,
                possible_agents=self.possible_agents,
                owner_critic=self.altruistic_owner_critic,
                mean_field_critic=self.altruistic_mean_field_critic,
                next_state=True,
            )

        targets = (
            self
            .compute_altruistic_rollout_targets(
                sidecar=(
                    self
                    ._altruistic_rollout_sidecar
                ),
                possible_agents=(
                    self.possible_agents
                ),
                owner_values=(
                    owner_values
                ),
                mean_field_values=(
                    mean_field_values
                ),
                owner_bootstrap_values=(
                    owner_bootstrap
                ),
                mean_field_bootstrap_values=(
                    mean_field_bootstrap
                ),
                dones=dones,
                terminated=terminated,
                owner_next_values=owner_next_values,
                mean_field_next_values=mean_field_next_values,
                discount_factor=(
                    self
                    .cfg
                    .discount_factor[
                        first_uid
                    ]
                ),
                lambda_coefficient=(
                    self
                    .cfg
                    .gae_lambda[
                        first_uid
                    ]
                ),
                alpha=(
                    self
                    .altruistic_alpha
                ),
                duration_steps=torch.tensor(
                    [int(getattr(sample, "duration_steps", 1)) for sample in structured_all],
                    dtype=torch.long,
                    device=dones.device,
                ).reshape(dones.shape),
            )
        )

        self._last_update_metrics.update(self.altruistic_target_diagnostics(
            targets=targets, structured_samples=structured_all,
            possible_agents=self.possible_agents, alpha=self.altruistic_alpha))

        # Passive pre-update value-fit diagnostics on active SMDP rows.
        # These metrics do not change PPO, GAE or the actor.
        from drl.MARL.c1_critic_quality import critic_fit_diagnostics

        self._last_update_metrics.update(
            critic_fit_diagnostics(
                owner_values=owner_values,
                mean_field_values=mean_field_values,
                owner_returns=targets.owner_returns,
                mean_field_returns=targets.mean_field_returns,
                structured_samples=structured_all,
                possible_agents=self.possible_agents,
            )
        )

        names = [
            "states",
            "log_prob",
        ]

        for _ in range(
            self
            .cfg
            .learning_epochs[
                first_uid
            ]
        ):
            (
                owner_batches,
                structured_batches,
            ) = (
                self
                .sample_coordinated_batches(
                    memories=(
                        self.memories
                    ),
                    sidecar=(
                        self
                        ._structured_rollout_sidecar
                    ),
                    names=names,
                    batch_size=(
                        rollout_size
                    ),
                    mini_batches=(
                        self
                        .cfg
                        .mini_batches[
                            first_uid
                        ]
                    ),
                )
            )

            indexes = (
                first_memory
                .sampling_indexes
            )

            mini_batches = (
                self
                .cfg
                .mini_batches[
                    first_uid
                ]
            )

            altruistic_batches = (
                self
                ._altruistic_rollout_sidecar
                .sample_minibatches(
                    indexes=indexes,
                    mini_batches=(
                        mini_batches
                    ),
                )
            )

            advantage_batches = (
                self
                .sample_owner_tensor_map_minibatches(
                    tensors_by_owner=(
                        targets
                        .altruistic_advantages
                    ),
                    indexes=indexes,
                    mini_batches=(
                        mini_batches
                    ),
                )
            )

            owner_old_value_batches = (
                self
                .sample_owner_tensor_map_minibatches(
                    tensors_by_owner=(
                        owner_values
                    ),
                    indexes=indexes,
                    mini_batches=(
                        mini_batches
                    ),
                )
            )

            owner_return_batches = (
                self
                .sample_owner_tensor_map_minibatches(
                    tensors_by_owner=(
                        targets
                        .owner_returns
                    ),
                    indexes=indexes,
                    mini_batches=(
                        mini_batches
                    ),
                )
            )

            mf_old_value_batches = (
                self
                .sample_owner_tensor_map_minibatches(
                    tensors_by_owner=(
                        mean_field_values
                    ),
                    indexes=indexes,
                    mini_batches=(
                        mini_batches
                    ),
                )
            )

            mf_return_batches = (
                self
                .sample_owner_tensor_map_minibatches(
                    tensors_by_owner=(
                        targets
                        .mean_field_returns
                    ),
                    indexes=indexes,
                    mini_batches=(
                        mini_batches
                    ),
                )
            )

            for (
                minibatch_index,
                structured_batch,
            ) in enumerate(
                structured_batches
            ):
                owner_tensor_batch = {
                    uid:
                        owner_batches[
                            uid
                        ][
                            minibatch_index
                        ]
                    for uid
                    in self.possible_agents
                }

                old_log_probs = {
                    uid:
                        owner_tensor_batch[
                            uid
                        ][
                            1
                        ]
                    for uid
                    in self.possible_agents
                }

                advantages = (
                    advantage_batches[
                        minibatch_index
                    ]
                )

                advantage_device = (
                    next(
                        iter(
                            advantages
                            .values()
                        )
                    )
                    .device
                )

                (
                    owner_active_masks,
                    active_owners,
                ) = (
                    self
                    .build_owner_activity_masks(
                        structured_batch=(
                            structured_batch
                        ),
                        possible_agents=(
                            self
                            .possible_agents
                        ),
                        device=(
                            advantage_device
                        ),
                    )
                )

                if not active_owners:
                    continue

                (
                    new_log_probs,
                    owner_entropies,
                    _,
                ) = (
                    self
                    .reevaluate_structured_minibatch(
                        structured_batch=(
                            structured_batch
                        ),
                        owner_tensor_batch=(
                            owner_tensor_batch
                        ),
                        evaluate_global_critic=False,
                    )
                )

                owner_policy_losses = (
                    self
                    .compute_owner_actor_losses(
                        new_log_probs=(
                            new_log_probs
                        ),
                        old_log_probs=(
                            old_log_probs
                        ),
                        advantages=(
                            advantages
                        ),
                        owner_entropies=(
                            owner_entropies
                        ),
                        ratio_clip=(
                            self
                            .cfg
                            .ratio_clip[
                                first_uid
                            ]
                        ),
                        entropy_loss_scale=(
                            self
                            .cfg
                            .entropy_loss_scale[
                                first_uid
                            ]
                        ),
                        owner_active_masks=(
                            owner_active_masks
                        ),
                    )
                )

                (
                    predicted_owner_values,
                    predicted_mf_values,
                ) = (
                    self
                    .evaluate_altruistic_critic_minibatch(
                        structured_batch=(
                            structured_batch
                        ),
                        altruistic_batch=(
                            altruistic_batches[
                                minibatch_index
                            ]
                        ),
                        possible_agents=(
                            self
                            .possible_agents
                        ),
                        owner_critic=(
                            self
                            .altruistic_owner_critic
                        ),
                        mean_field_critic=(
                            self
                            .altruistic_mean_field_critic
                        ),
                        # Generic train-mode encoders may consume dropout RNG
                        # even on padding. Native C1 keeps its models in eval.
                        skip_inactive=bool(getattr(self, "deterministic_update_mode", False)),
                        padding_reference=next(iter(advantages.values())),
                        diagnostics=self._critic_evaluation_metrics,
                    )
                )

                owner_value_loss = (
                    self
                    .compute_owner_conditioned_value_loss(
                        predicted_values=(
                            predicted_owner_values
                        ),
                        sampled_values=(
                            owner_old_value_batches[
                                minibatch_index
                            ]
                        ),
                        sampled_returns=(
                            owner_return_batches[
                                minibatch_index
                            ]
                        ),
                        value_clip=(
                            self
                            .cfg
                            .value_clip[
                                first_uid
                            ]
                        ),
                        value_loss_scale=(
                            self
                            .cfg
                            .value_loss_scale[
                                first_uid
                            ]
                        ),
                        sample_mask=torch.tensor([
                            sample.rollout_active for sample in structured_batch
                        ], device=advantage_device, dtype=torch.bool),
                    )
                )

                mean_field_value_loss = (
                    self
                    .compute_owner_conditioned_value_loss(
                        predicted_values=(
                            predicted_mf_values
                        ),
                        sampled_values=(
                            mf_old_value_batches[
                                minibatch_index
                            ]
                        ),
                        sampled_returns=(
                            mf_return_batches[
                                minibatch_index
                            ]
                        ),
                        value_clip=(
                            self
                            .cfg
                            .value_clip[
                                first_uid
                            ]
                        ),
                        value_loss_scale=(
                            self
                            .cfg
                            .value_loss_scale[
                                first_uid
                            ]
                        ),
                        sample_mask=torch.tensor([
                            sample.rollout_active for sample in structured_batch
                        ], device=advantage_device, dtype=torch.bool),
                    )
                )

                policy_metrics = self.policy_update_diagnostics(
                    new_log_probs=new_log_probs, old_log_probs=old_log_probs,
                    owner_entropies=owner_entropies, owner_active_masks=owner_active_masks,
                    ratio_clip=self.cfg.ratio_clip[first_uid])
                grad_norm_clip = getattr(self.cfg, "grad_norm_clip", 0.0)
                if isinstance(grad_norm_clip, Mapping):
                    grad_norm_clip = grad_norm_clip[first_uid]
                step_metrics = self.coordinated_altruistic_optimizer_step(
                    owner_policy_losses=(
                        owner_policy_losses
                    ),
                    owner_value_loss=(
                        owner_value_loss
                    ),
                    mean_field_value_loss=(
                        mean_field_value_loss
                    ),
                    actor_optimizer=(
                        self
                        .actor_optimizer
                    ),
                    owner_critic_optimizer=(
                        self
                        .owner_critic_optimizer
                    ),
                    mean_field_critic_optimizer=(
                        self
                        .mean_field_critic_optimizer
                    ),
                    active_owners=(
                        active_owners
                    ),
                    actor_updates_enabled=bool(getattr(self, "c1_actor_updates_enabled", True)),
                    grad_norm_clip=float(grad_norm_clip),
                )
                self._record_update_metrics(step_metrics, policy_metrics)

    def coordinated_update(
        self,
        *,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Run the configured team-MAPPO or fixed-alpha APPO update."""

        if (
            getattr(
                self,
                "_training_advantage_mode",
                "team",
            )
            == "altruistic"
        ):
            return self.coordinated_altruistic_update(
                timestep=timestep,
                timesteps=timesteps,
            )

        self._reset_update_metrics()
        # Existing team-MAPPO objective below is unchanged.
        if not self.memories:
            return

        first_uid = next(iter(self.possible_agents))
        first_memory = self.memories[first_uid]

        last_values = self.compute_global_bootstrap_value()

        self.compute_and_broadcast_team_gae(
            memories=self.memories,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor[first_uid],
            lambda_coefficient=self.cfg.gae_lambda[first_uid],
            time_limit_bootstrap=self.cfg.time_limit_bootstrap[first_uid],
        )

        names = [
            "states",
            "log_prob",
            "values",
            "returns",
            "advantages",
        ]

        for _ in range(self.cfg.learning_epochs[first_uid]):
            owner_batches, structured_batches = (
                self.sample_coordinated_batches(
                    memories=self.memories,
                    sidecar=self._structured_rollout_sidecar,
                    names=names,
                    batch_size=len(first_memory),
                    mini_batches=self.cfg.mini_batches[first_uid],
                )
            )

            for minibatch_index, structured_batch in enumerate(
                structured_batches
            ):
                owner_tensor_batch = {
                    uid: owner_batches[uid][minibatch_index]
                    for uid in self.possible_agents
                }

                old_log_probs = {
                    uid: owner_tensor_batch[uid][1]
                    for uid in self.possible_agents
                }

                sampled_values = owner_tensor_batch[first_uid][2]
                sampled_returns = owner_tensor_batch[first_uid][3]
                advantages = owner_tensor_batch[first_uid][4]

                owner_active_masks, active_owners = (
                    self.build_owner_activity_masks(
                        structured_batch=structured_batch,
                        possible_agents=self.possible_agents,
                        device=advantages.device,
                    )
                )

                (
                    new_log_probs,
                    owner_entropies,
                    predicted_values,
                ) = self.reevaluate_structured_minibatch(
                    structured_batch=structured_batch,
                    owner_tensor_batch=owner_tensor_batch,
                )

                owner_policy_losses = self.compute_owner_actor_losses(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    owner_entropies=owner_entropies,
                    ratio_clip=self.cfg.ratio_clip[first_uid],
                    entropy_loss_scale=(
                        self.cfg.entropy_loss_scale[first_uid]
                    ),
                    owner_active_masks=owner_active_masks,
                )

                value_loss = self.compute_global_value_loss(
                    predicted_values=predicted_values,
                    sampled_values=sampled_values,
                    sampled_returns=sampled_returns,
                    value_clip=self.cfg.value_clip[first_uid],
                    value_loss_scale=self.cfg.value_loss_scale[first_uid],
                )

                policy_metrics = self.policy_update_diagnostics(
                    new_log_probs=new_log_probs, old_log_probs=old_log_probs,
                    owner_entropies=owner_entropies, owner_active_masks=owner_active_masks,
                    ratio_clip=self.cfg.ratio_clip[first_uid])
                step_metrics = self.coordinated_optimizer_step(
                    owner_policy_losses=owner_policy_losses,
                    value_loss=value_loss,
                    actor_optimizer=self.actor_optimizer,
                    critic_optimizer=self.critic_optimizer,
                    active_owners=active_owners,
                )
                self._record_update_metrics(step_metrics, policy_metrics)

    @property
    def last_update_metrics(self):
        """JSON-safe means across the most recent update's actual minibatches."""
        return {**getattr(self, "_last_update_metrics", {}),
                **{"critic_" + key: value for key, value in
                   getattr(self, "_critic_evaluation_metrics", {}).items()}}

    @staticmethod
    def altruistic_target_diagnostics(*, targets, structured_samples, possible_agents, alpha):
        """Summarize actual pre-normalization rollout targets on actor-active rows.

        Flattening follows the sidecar/Memory time-major order. Each active owner
        contributes one sample; inactive Expert rows never enter these statistics.
        This detached observer neither changes targets nor consumes random state.
        """
        channels = {
            "owner_advantage": targets.owner_advantages,
            "mean_field_advantage": targets.mean_field_advantages,
            "scaled_mean_field_advantage": {
                uid: (alpha[uid] if isinstance(alpha, Mapping) else alpha) * tensor
                for uid, tensor in targets.mean_field_advantages.items()},
            "combined_advantage": targets.altruistic_advantages,
        }
        metrics = {}
        for name, tensors in channels.items():
            selected = []
            for uid in possible_agents:
                values = tensors[uid].detach().reshape(-1)
                mask = torch.tensor([bool(sample.owner_active_mask.get(uid, False))
                    for sample in structured_samples], device=values.device, dtype=torch.bool)
                if mask.numel() != values.numel():
                    raise ValueError("target diagnostics and structured rollout are not aligned")
                selected.append(values[mask])
            values = torch.cat(selected).to(dtype=torch.float64)
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"nonfinite active {name}")
            metrics[name + "_count"] = values.numel()
            for stat in ("mean", "std", "min", "max"):
                value = 0.
                if values.numel():
                    value = (values.std(unbiased=False) if stat == "std"
                             else getattr(values, stat)()).item()
                metrics[name + "_" + stat] = value
        return metrics

    def _reset_update_metrics(self):
        self._critic_evaluation_metrics = {}
        self._last_update_metrics = {key: 0 for key in (
            "actor_optimizer_steps", "global_critic_optimizer_steps",
            "owner_critic_optimizer_steps", "mean_field_critic_optimizer_steps", "minibatches")}
        self._update_metric_sums = {}

    def _record_update_metrics(self, step_metrics, policy_metrics):
        aliases = {"policy_loss": "actor_loss", "value_loss": "global_critic_loss",
                   "owner_value_loss": "owner_critic_loss", "mean_field_value_loss": "mean_field_critic_loss"}
        self._last_update_metrics["minibatches"] += 1
        count = self._last_update_metrics["minibatches"]
        for key, value in {**step_metrics, **policy_metrics}.items():
            key = aliases.get(key, key)
            value = float(value.detach().item()) if torch.is_tensor(value) else float(value)
            if not torch.isfinite(torch.tensor(value, dtype=torch.float64)).item():
                raise ValueError(f"nonfinite update metric {key}")
            if key.endswith("optimizer_steps"):
                self._last_update_metrics[key] = self._last_update_metrics.get(key, 0) + int(value)
            else:
                self._update_metric_sums[key] = self._update_metric_sums.get(key, 0.) + value
                self._last_update_metrics[key] = self._update_metric_sums[key]/count

    @staticmethod
    def policy_update_diagnostics(*, new_log_probs, old_log_probs, owner_entropies,
                                  owner_active_masks, ratio_clip):
        """Evaluate PPO diagnostics on exactly the active owner/sample mask."""
        differences, entropies = [], []
        for uid, new in new_log_probs.items():
            mask = owner_active_masks[uid].to(device=new.device, dtype=torch.bool)
            if not bool(mask.any().item()):
                continue
            differences.append((new-old_log_probs[uid])[mask].detach().reshape(-1))
            entropy = owner_entropies[uid]
            entropies.append((entropy if entropy.ndim == 0 else entropy[mask]).detach().reshape(-1))
        if not differences:
            return {}
        log_ratio = torch.cat(differences)
        ratio = log_ratio.exp()
        result = dict(entropy=torch.cat(entropies).mean(), ratio_mean=ratio.mean(),
                      approx_kl=((ratio-1)-log_ratio).mean(),
                      clip_fraction=((ratio-1).abs() > ratio_clip).float().mean())
        if any(not bool(torch.isfinite(value).all().item()) for value in result.values()):
            raise ValueError("nonfinite policy diagnostics")
        return result

    @staticmethod
    def _optimizer_gradient_norm(optimizer):
        gradients = [parameter.grad.detach() for group in optimizer.param_groups
                     for parameter in group["params"] if parameter.grad is not None]
        if any(not bool(torch.isfinite(grad).all().item()) for grad in gradients):
            raise ValueError("nonfinite optimizer gradient")
        return sum(float(grad.double().square().sum().item()) for grad in gradients)**.5

    @staticmethod
    def sample_aligned_owner_batches(
        *,
        memories: dict[str, Any],
        names: list[str],
        batch_size: int,
        mini_batches: int,
    ) -> dict[str, list[list[torch.Tensor]]]:
        """Sample identical rollout indexes from every owner memory."""
        if not memories:
            return {}

        first_uid = next(iter(memories))
        first_memory = memories[first_uid]

        batches = {
            first_uid: first_memory.sample(
                names=names,
                batch_size=batch_size,
                mini_batches=mini_batches,
            )
        }

        indexes = first_memory.sampling_indexes

        for uid, memory in memories.items():
            if uid == first_uid:
                continue

            batches[uid] = memory.sample_by_index(
                names=names,
                indexes=indexes,
                mini_batches=mini_batches,
            )

        return batches

    @staticmethod
    def sample_coordinated_batches(
        *,
        memories: dict[str, Any],
        sidecar: StructuredRolloutSidecar,
        names: list[str],
        batch_size: int,
        mini_batches: int,
    ):
        """Sample aligned tensor and structured minibatches."""
        owner_batches = MSSRMAPPO.sample_aligned_owner_batches(
            memories=memories,
            names=names,
            batch_size=batch_size,
            mini_batches=mini_batches,
        )

        if not memories:
            return owner_batches, []

        first_uid = next(iter(memories))
        indexes = memories[first_uid].sampling_indexes

        structured_batches = sidecar.sample_minibatches(
            indexes=indexes,
            mini_batches=mini_batches,
        )

        return owner_batches, structured_batches

    @staticmethod
    def compute_owner_conditioned_value_loss(
        *,
        predicted_values,
        sampled_values,
        sampled_returns,
        value_clip: float,
        value_loss_scale: float,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared value loss for one owner-conditioned critic."""

        owners = tuple(
            predicted_values
        )

        if (
            not owners
            or set(
                sampled_values
            )
            != set(
                owners
            )
            or set(
                sampled_returns
            )
            != set(
                owners
            )
        ):
            raise ValueError(
                "owner-conditioned value maps must "
                "cover identical owners"
            )

        losses = []

        for uid in owners:
            predicted = (
                predicted_values[
                    uid
                ]
            )

            old_value = (
                sampled_values[
                    uid
                ]
            )

            target = (
                sampled_returns[
                    uid
                ]
            )

            if sample_mask is not None:
                mask = sample_mask.to(device=predicted.device, dtype=torch.bool).reshape(-1)
                if predicted.shape[0] != mask.numel():
                    raise ValueError('critic sample mask must match minibatch rows')
                predicted = predicted[mask]
                old_value = old_value[mask]
                target = target[mask]
                if not mask.any():
                    losses.append(predicted.sum())
                    continue

            if value_clip > 0:
                predicted = (
                    old_value
                    + torch.clamp(
                        predicted
                        - old_value,
                        min=-value_clip,
                        max=value_clip,
                    )
                )

            losses.append(
                torch.nn.functional.mse_loss(
                    target,
                    predicted,
                )
            )

        return (
            value_loss_scale
            * torch.stack(
                losses
            ).mean()
        )

    @staticmethod
    def coordinated_altruistic_optimizer_step(
        *,
        owner_policy_losses,
        owner_value_loss,
        mean_field_value_loss,
        actor_optimizer,
        owner_critic_optimizer,
        mean_field_critic_optimizer,
        active_owners=None,
        profile_sink=None,
        actor_updates_enabled=True,
        grad_norm_clip: float = 0.0,
    ):
        """One actor step plus two critic steps (optional actor freeze)."""

        owners = (
            tuple(
                owner_policy_losses
            )
            if active_owners
            is None
            else tuple(
                active_owners
            )
        )

        if not owners:
            raise ValueError(
                "at least one active owner is required"
            )

        policy_loss = (
            torch.stack(
                [
                    owner_policy_losses[
                        uid
                    ]
                    for uid in owners
                ]
            )
            .mean()
        )

        if any(not bool(torch.isfinite(loss).all().item()) for loss in (policy_loss, owner_value_loss, mean_field_value_loss)):
            raise ValueError("nonfinite APPO optimizer loss")
        profile_start = perf_counter() if profile_sink is not None else None
        actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_grad_norm = 0.0
        actor_grad_norm_post_clip = 0.0
        if actor_updates_enabled:
            policy_loss.backward()
            actor_grad_norm = MSSRMAPPO._optimizer_gradient_norm(actor_optimizer)
            actor_grad_norm_post_clip = MSSRMAPPO._clip_optimizer_gradients(actor_optimizer, grad_norm_clip)
            actor_optimizer.step()

        if profile_sink is not None:
            profile_sink('update.actor_backward_optimizer', perf_counter() - profile_start)
        profile_start = perf_counter() if profile_sink is not None else None
        owner_critic_optimizer.zero_grad(
            set_to_none=True
        )

        owner_value_loss.backward()
        owner_grad_norm = MSSRMAPPO._optimizer_gradient_norm(owner_critic_optimizer)
        owner_grad_norm_post_clip = MSSRMAPPO._clip_optimizer_gradients(owner_critic_optimizer, grad_norm_clip)
        owner_critic_optimizer.step()

        if profile_sink is not None:
            profile_sink('update.owner_backward_optimizer', perf_counter() - profile_start)
        profile_start = perf_counter() if profile_sink is not None else None
        mean_field_critic_optimizer.zero_grad(
            set_to_none=True
        )

        mean_field_value_loss.backward()
        mf_grad_norm = MSSRMAPPO._optimizer_gradient_norm(mean_field_critic_optimizer)
        mf_grad_norm_post_clip = MSSRMAPPO._clip_optimizer_gradients(mean_field_critic_optimizer, grad_norm_clip)
        mean_field_critic_optimizer.step()

        if profile_sink is not None:
            profile_sink('update.mean_field_backward_optimizer', perf_counter() - profile_start)
        return {
            "actor_grad_norm": actor_grad_norm,
            "actor_grad_norm_post_clip": actor_grad_norm_post_clip,
            "owner_critic_grad_norm": owner_grad_norm,
            "owner_critic_grad_norm_post_clip": owner_grad_norm_post_clip,
            "mean_field_critic_grad_norm": mf_grad_norm,
            "mean_field_critic_grad_norm_post_clip": mf_grad_norm_post_clip,
            "actor_optimizer_steps": int(bool(actor_updates_enabled)),
            "owner_critic_optimizer_steps": 1,
            "mean_field_critic_optimizer_steps": 1,
            "policy_loss":
                policy_loss,
            "owner_value_loss":
                owner_value_loss,
            "mean_field_value_loss":
                mean_field_value_loss,
        }

    @staticmethod
    def _clip_optimizer_gradients(optimizer, limit: float) -> float:
        """Clip the whole optimizer's gradient norm and measure what step uses."""
        if limit > 0:
            parameters = [parameter for group in optimizer.param_groups
                          for parameter in group["params"] if parameter.grad is not None]
            torch.nn.utils.clip_grad_norm_(parameters, limit, error_if_nonfinite=True)
        return MSSRMAPPO._optimizer_gradient_norm(optimizer)

    @staticmethod
    def coordinated_optimizer_step(
        *,
        owner_policy_losses: dict[str, torch.Tensor],
        value_loss: torch.Tensor,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        active_owners: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aggregate active-owner policy losses and step actor/critic once."""
        owners = (
            tuple(owner_policy_losses)
            if active_owners is None
            else active_owners
        )

        policy_loss = torch.stack(
            [owner_policy_losses[uid] for uid in owners]
        ).mean()

        if any(not bool(torch.isfinite(loss).all().item()) for loss in (policy_loss, value_loss)):
            raise ValueError("nonfinite MAPPO optimizer loss")
        actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        actor_grad_norm = MSSRMAPPO._optimizer_gradient_norm(actor_optimizer)
        actor_optimizer.step()

        critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        critic_grad_norm = MSSRMAPPO._optimizer_gradient_norm(critic_optimizer)
        critic_optimizer.step()

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "actor_grad_norm": actor_grad_norm,
            "global_critic_grad_norm": critic_grad_norm,
            "actor_optimizer_steps": 1,
            "global_critic_optimizer_steps": 1,
        }
