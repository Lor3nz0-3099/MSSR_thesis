"""BC-v4 shared actor adapter for MSSR MAPPO."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from skrl.models.torch import Model

from drl.IL.bc_v1.action_set_head import (
    ActionSetPolicy,
    load_action_set_head_config,
)
from drl.IL.bc_v1.bc_v1_contract import (
    robot_spec_fingerprint,
)
from drl.IL.bc_v1.feature_schema import (
    feature_schema_fingerprint,
)
from drl.IL.bc_v1.model import (
    load_model_config,
)
from drl.IL.bc_v1.batching import (
    collate_policy_inputs,
)

from drl.MARL.policy import (
    bind_marl_candidates,
    compose_structural_payload_views,
    map_structural_payloads_to_candidates,
    route_resource_payloads,
    route_structural_numeric_arguments,
    route_structural_categorical_arguments,
    route_structural_boolean_arguments,
    route_structural_reference_arguments,
    route_structural_reference_set_arguments,
    route_structural_scopes,
)


BC_V4_CHECKPOINT_SCHEMA = (
    "mssr.bc_v1.checkpoint.action_set.v1"
)

BC_V4_MODEL_CONTRACT = "action_set_v1"

BC_V4_REFERENCE_SHA256 = (
    "d90a73e9b5ba2ecb579e4fc8733066b1"
    "c55d9e60dafeb1d71c029c49a7c21c6d"
)


class BCV4SharedActorWrapper(Model):
    """Expose BC-v4 H1/H2 as the shared MAPPO actor.

    The wrapped ``policy`` remains the canonical ActionSetPolicy:
    H1 is its ``backbone`` and H2 is its ``head``.

    MARL adds only stochastic-distribution parameters that BC did not
    need: Gaussian log-standard-deviations for numeric resource and
    structural payloads.
    """

    def __init__(
        self,
        *,
        policy: nn.Module,
        observation_space: Any = None,
        state_space: Any = None,
        action_space: Any = None,
        device: str | torch.device | None = None,
        resource_log_std_init: float = 0.0,
        structural_log_std_init: float = 0.0,
    ) -> None:
        # Keep the skrl model device consistent with the already-built
        # BC policy unless the caller explicitly chooses another device.
        if device is None:
            try:
                device = next(
                    policy.parameters()
                ).device
            except StopIteration:
                device = torch.device("cpu")

        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        self.policy = policy.to(self.device)
        # Explicit capability: MAPPO may request lazy payload views.
        self.prefer_lazy_payloads = True

        self.resource_log_std = nn.Parameter(
            torch.tensor(
                float(resource_log_std_init),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.structural_log_std = nn.Parameter(
            torch.tensor(
                float(structural_log_std_init),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.checkpoint_path: str | None = None
        self.checkpoint_sha256: str | None = None
        self.checkpoint_step: int | None = None
        self.checkpoint_identity: dict[str, Any] | None = None
        self.checkpoint_config: dict[str, Any] | None = None
        self.presence_threshold_logit = 0.0
        self.presence_calibration_sha256 = None
        self.presence_calibration_path = None
        # Toggle for controlled A/B parity tests; no checkpoint state change.
        self.prefer_batched_h2 = isinstance(self.policy, ActionSetPolicy)

    def compute(
        self,
        inputs,
        role: str = "",
    ):
        """Minimal skrl Model contract.

        Structured MAPPO uses ``forward_all_owners`` directly. This method
        exists so the wrapper remains a valid skrl Model and delegates the
        raw BC-v4 forward without changing its semantics.
        """
        del role
        return self.policy(inputs), {}

    def act(
        self,
        inputs,
        *,
        role: str = "",
    ):
        """Satisfy the skrl Model contract by delegating to compute()."""
        return self.compute(
            inputs,
            role=role,
        )

    @staticmethod
    def compute_sha256(
        checkpoint_path: str | Path,
    ) -> str:
        path = Path(checkpoint_path)

        digest = hashlib.sha256()

        with path.open("rb") as stream:
            for chunk in iter(
                lambda: stream.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @classmethod
    def from_checkpoint(
        cls,
        *,
        policy: nn.Module,
        checkpoint_path: str | Path,
        expected_sha256: str | None = None,
        map_location: str | torch.device = "cpu",
        observation_space: Any = None,
        state_space: Any = None,
        action_space: Any = None,
        device: str | torch.device | None = None,
        resource_log_std_init: float = 0.0,
        structural_log_std_init: float = 0.0,
    ) -> "BCV4SharedActorWrapper":
        """Strictly restore the complete BC-v4 H1/H2 policy."""
        path = Path(checkpoint_path)

        actual_sha256 = cls.compute_sha256(path)

        if (
            expected_sha256 is not None
            and actual_sha256 != expected_sha256
        ):
            raise ValueError(
                "BC-v4 checkpoint SHA256 mismatch: "
                f"expected {expected_sha256}, "
                f"got {actual_sha256}"
            )

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "BC-v4 checkpoint must be a dictionary"
            )

        schema_version = checkpoint.get(
            "schema_version"
        )
        model_contract = checkpoint.get(
            "model_contract"
        )

        if schema_version != BC_V4_CHECKPOINT_SCHEMA:
            raise ValueError(
                "Unsupported BC-v4 checkpoint schema: "
                f"{schema_version!r}"
            )

        if model_contract != BC_V4_MODEL_CONTRACT:
            raise ValueError(
                "Unsupported BC-v4 model contract: "
                f"{model_contract!r}"
            )

        model_state = checkpoint.get("model")

        if not isinstance(model_state, dict):
            raise ValueError(
                "BC-v4 checkpoint does not contain "
                "a model state_dict"
            )

        # Strict is intentional:
        # H1/H2 architecture must exactly match the checkpoint.
        policy.load_state_dict(
            model_state,
            strict=True,
        )

        wrapper = cls(
            policy=policy,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            resource_log_std_init=resource_log_std_init,
            structural_log_std_init=(
                structural_log_std_init
            ),
        )

        wrapper.checkpoint_path = str(path)
        wrapper.checkpoint_sha256 = actual_sha256
        wrapper.checkpoint_step = checkpoint.get(
            "step"
        )
        wrapper.checkpoint_identity = checkpoint.get(
            "identity"
        )
        wrapper.checkpoint_config = checkpoint.get(
            "config"
        )

        return wrapper

    @classmethod
    def from_repository_checkpoint(
        cls,
        *,
        spec,
        feature_schema,
        repo_root: str | Path,
        checkpoint_path: str | Path,
        expected_sha256: str | None = None,
        map_location: str | torch.device = "cpu",
        observation_space: Any = None,
        state_space: Any = None,
        action_space: Any = None,
        device: str | torch.device | None = None,
        resource_log_std_init: float = 0.0,
        structural_log_std_init: float = 0.0,
    ) -> "BCV4SharedActorWrapper":
        """Build the canonical BC-v4 ActionSetPolicy and restore H1/H2.

        The robot specification and feature schema are injected by the
        runtime/adapter. Their fingerprints must match the BC-v4
        checkpoint identity.

        This is deliberate: H1/H2 do not own SMORES-specific runtime
        knowledge. The adapter supplies the embodiment contract.
        """
        root = Path(repo_root)
        path = Path(checkpoint_path)

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "BC-v4 checkpoint must be a dictionary"
            )

        identity = checkpoint.get("identity")

        if not isinstance(identity, dict):
            raise ValueError(
                "BC-v4 checkpoint is missing identity metadata"
            )

        actual_robot_fingerprint = (
            robot_spec_fingerprint(spec)
        )
        expected_robot_fingerprint = identity.get(
            "robot_spec_fingerprint"
        )

        if (
            actual_robot_fingerprint
            != expected_robot_fingerprint
        ):
            raise ValueError(
                "BC-v4 robot spec fingerprint mismatch: "
                f"expected {expected_robot_fingerprint}, "
                f"got {actual_robot_fingerprint}"
            )

        actual_schema_fingerprint = (
            feature_schema_fingerprint(
                feature_schema
            )
        )
        expected_schema_fingerprint = identity.get(
            "feature_schema_fingerprint"
        )

        if (
            actual_schema_fingerprint
            != expected_schema_fingerprint
        ):
            raise ValueError(
                "BC-v4 feature schema fingerprint mismatch: "
                f"expected {expected_schema_fingerprint}, "
                f"got {actual_schema_fingerprint}"
            )

        h1_config_path = (
            root
            / "configs/bc_v1/gnn_v1.json"
        )
        head_config_path = (
            root
            / "configs/bc_v1/action_set_head_v1.json"
        )

        h1_config = load_model_config(
            h1_config_path
        )

        head_config = load_action_set_head_config(
            head_config_path
        )

        expected_head_config = identity.get(
            "head_config"
        )

        if (
            expected_head_config is not None
            and head_config != expected_head_config
        ):
            raise ValueError(
                "BC-v4 ActionSetHead configuration "
                "does not match checkpoint identity"
            )

        policy = ActionSetPolicy(
            spec,
            feature_schema,
            h1_config,
            head_config,
        )

        wrapper = cls.from_checkpoint(
            policy=policy,
            checkpoint_path=path,
            expected_sha256=expected_sha256,
            map_location=map_location,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            resource_log_std_init=(
                resource_log_std_init
            ),
            structural_log_std_init=(
                structural_log_std_init
            ),
        )

        wrapper.robot_spec_fingerprint = (
            actual_robot_fingerprint
        )
        wrapper.feature_schema_fingerprint = (
            actual_schema_fingerprint
        )
        wrapper.h1_config_path = str(
            h1_config_path
        )
        wrapper.head_config_path = str(
            head_config_path
        )

        from drl.MARL.bc_v4_runtime_inference import load_bc_v4_runtime_inference_config
        calibration = load_bc_v4_runtime_inference_config(repo_root)
        calibration_path = Path(repo_root).resolve() / 'configs/bc_v1/runtime_inference_v1.json'
        wrapper.presence_threshold_logit = float(calibration['deterministic_presence_threshold_logit'])
        wrapper.presence_calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        wrapper.presence_calibration_path = str(calibration_path)

        return wrapper

    def _materialize_forward_output(
        self,
        *,
        output: dict[str, Any],
        graph: Any,
        routing: dict,
        lazy_payloads: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(output, dict):
            raise TypeError(
                "ActionSetPolicy forward must return a dictionary"
            )

        presence_logits = output["presence_logits"]
        parameters = output["parameters"]

        records = bind_marl_candidates(
            routing=routing,
            presence_logits=presence_logits,
            presence_threshold_logit=self.presence_threshold_logit,
        )

        if lazy_payloads:
            from drl.MARL.runtime_payloads import RuntimePayloadCache
            cache = getattr(self, '_runtime_payload_cache', None)
            if cache is None:
                cache = self._runtime_payload_cache = RuntimePayloadCache()
            resources, structural = cache.views(
                parameters=parameters, graph=graph, routing=routing
            )
            return dict(
                prediction_template=output,
                records=records,
                resource_payloads=resources,
                structural_payloads=structural,
                resource_log_std=self.resource_log_std,
                structural_log_std=self.structural_log_std,
            )

        resource_payloads = route_resource_payloads(
            parameters["resource"]
        )

        structural = parameters["structural"]
        arguments = structural["arguments"]

        def routed(head, family, router):
            if "argument" in head:
                return router(
                    **{family: head},
                    structural_inputs=graph["structural"],
                )
            return head

        local_structural_payloads = compose_structural_payload_views(
            binding_count=len(routing["structural_bindings"]),
            numeric=routed(
                arguments.get("numeric", {}),
                "numeric",
                route_structural_numeric_arguments,
            ),
            categorical=routed(
                arguments.get("category", arguments.get("categorical", {})),
                "categorical",
                route_structural_categorical_arguments,
            ),
            boolean=routed(
                arguments.get("boolean", {}),
                "boolean",
                route_structural_boolean_arguments,
            ),
            reference=routed(
                arguments.get("reference", {}),
                "reference",
                route_structural_reference_arguments,
            ),
            reference_set=routed(
                arguments.get("reference_set", {}),
                "reference_set",
                route_structural_reference_set_arguments,
            ),
            scope=(
                route_structural_scopes(structural["scope"])
                if "binding" in structural["scope"]
                else structural["scope"]
            ),
        )

        structural_payloads = map_structural_payloads_to_candidates(
            records=records,
            structural_payloads=local_structural_payloads,
        )

        return {
            "prediction_template": output,
            "records": records,
            "resource_payloads": resource_payloads,
            "structural_payloads": structural_payloads,
            "resource_log_std": self.resource_log_std,
            "structural_log_std": self.structural_log_std,
        }

    @staticmethod
    def _split_h1_latents(
        *,
        batched_latents: dict[str, torch.Tensor],
        graphs: list[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor]]:
        graph_keys = ("graph_h", "context_h", "observation_h")
        family_keys = (
            ("module_h", "module_x"),
            ("connector_h", "connector_x"),
            ("resource_h", "resource_x"),
        )
        offsets = {latent_key: 0 for latent_key, _ in family_keys}
        result = []

        for graph_index, graph in enumerate(graphs):
            local = {
                key: batched_latents[key][graph_index:graph_index + 1]
                for key in graph_keys
            }
            for latent_key, input_key in family_keys:
                count = len(graph[input_key])
                start = offsets[latent_key]
                stop = start + count
                local[latent_key] = batched_latents[latent_key][start:stop]
                offsets[latent_key] = stop
            result.append(local)

        return result

    def forward_all_owners_batch(
        self,
        *,
        graphs: list[Any],
        routings: list[dict],
        possible_agents: list[str],
        lazy_payloads: bool = False,
    ) -> list[dict[str, Any]]:
        """Batch H1 and H2 across graphs, preserving per-graph MARL output ABI."""
        del possible_agents
        graphs = list(graphs)
        routings = list(routings)

        if not graphs:
            return []
        if len(graphs) != len(routings):
            raise ValueError("graph and routing batches must have the same size")
        if any(routing is None for routing in routings):
            raise ValueError("BCV4SharedActorWrapper requires routing snapshots")

        batched_inputs = collate_policy_inputs(graphs)
        batched_latents = self.policy.backbone.encode(batched_inputs)
        if self.prefer_batched_h2:
            from drl.MARL.h2_batching import split_action_set_h2_batch

            packed_output = self.policy.head(batched_latents, batched_inputs)
            outputs = split_action_set_h2_batch(
                packed=packed_output, graphs=graphs,
            )
        else:
            # Reference path for A/B regressions; no numerical shortcuts.
            local_latents = self._split_h1_latents(
                batched_latents=batched_latents, graphs=graphs,
            )
            outputs = [
                self.policy.head(latents, graph)
                for latents, graph in zip(local_latents, graphs, strict=True)
            ]

        return [
            self._materialize_forward_output(
                output=output,
                graph=graph,
                routing=routing,
                lazy_payloads=lazy_payloads,
            )
            for output, graph, routing in zip(
                outputs, graphs, routings, strict=True
            )
        ]

    def forward_all_owners(
        self,
        *,
        graph: Any,
        routing: dict,
        possible_agents: list[str],
        lazy_payloads: bool = False,
    ) -> dict[str, Any]:
        """Run H1/H2 once and expose MARL distribution views."""
        del possible_agents

        if routing is None:
            raise ValueError(
                "BCV4SharedActorWrapper requires a routing snapshot"
            )

        return self._materialize_forward_output(
            output=self.policy(graph),
            graph=graph,
            routing=routing,
            lazy_payloads=lazy_payloads,
        )
