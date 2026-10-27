"""Independent centralized GNN critic for MSSR MAPPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from skrl.models.torch import Model

from drl.IL.bc_v1.model import (
    BCGNN,
    load_model_config,
)


class GlobalCritic(Model):
    """Centralized variable-N critic.

    The critic owns an encoder completely independent from the actor.
    The encoder returns per-module ``module_h`` embeddings. They are
    pooled with permutation-invariant mean || max and mapped to one
    scalar global value per graph/environment.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        module_latent_dim: int,
        value_hidden_dim: int = 128,
        observation_space: Any = None,
        state_space: Any = None,
        action_space: Any = None,
        device: str | torch.device | None = None,
    ) -> None:
        if module_latent_dim <= 0:
            raise ValueError(
                "module_latent_dim must be positive"
            )

        if value_hidden_dim <= 0:
            raise ValueError(
                "value_hidden_dim must be positive"
            )

        if device is None:
            try:
                device = next(
                    encoder.parameters()
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

        self.encoder = encoder.to(
            self.device
        )

        self.module_latent_dim = int(
            module_latent_dim
        )

        self.value_head = nn.Sequential(
            nn.Linear(
                2 * self.module_latent_dim,
                value_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                value_hidden_dim,
                1,
            ),
        ).to(self.device)

    @staticmethod
    def _pool_mean_max(
        module_latents: torch.Tensor,
        graph_ptr: torch.Tensor,
    ) -> torch.Tensor:
        if module_latents.ndim != 2:
            raise ValueError(
                "module latents must be rank-2"
            )

        if graph_ptr.ndim != 1:
            raise ValueError(
                "graph_ptr must be rank-1"
            )

        if graph_ptr.numel() < 2:
            raise ValueError(
                "graph_ptr must describe at least one graph"
            )

        ptr = [
            int(value)
            for value in graph_ptr.detach().cpu().tolist()
        ]

        if ptr[0] != 0:
            raise ValueError(
                "graph_ptr must start at zero"
            )

        if ptr[-1] != module_latents.shape[0]:
            raise ValueError(
                "graph_ptr does not cover all module latents"
            )

        if any(
            right < left
            for left, right in zip(
                ptr,
                ptr[1:],
            )
        ):
            raise ValueError(
                "graph_ptr must be monotonic"
            )

        pooled = []

        for start, stop in zip(
            ptr,
            ptr[1:],
        ):
            if stop <= start:
                raise ValueError(
                    "global critic received an empty module set"
                )

            chunk = module_latents[
                start:stop
            ]

            pooled.append(
                torch.cat(
                    (
                        chunk.mean(
                            dim=0
                        ),
                        chunk.max(
                            dim=0
                        ).values,
                    ),
                    dim=-1,
                )
            )

        return torch.stack(
            pooled,
            dim=0,
        )

    def forward_global(
        self,
        graph,
    ) -> torch.Tensor:
        # Structured rollout minibatches contain one independently
        # tensorized privileged graph per environment/timestep.
        # Evaluate them through one critic API call without inventing
        # a fragile second graph-collation contract.
        if isinstance(graph, (list, tuple)):
            if not graph:
                raise ValueError(
                    "global critic received an empty graph batch"
                )

            return torch.cat(
                [
                    self.forward_global(item)
                    for item in graph
                ],
                dim=0,
            )

        if not isinstance(graph, dict):
            raise TypeError(
                "global critic graph must be a dictionary"
            )

        if "graph_ptr" not in graph:
            raise KeyError(
                "global critic graph is missing graph_ptr"
            )

        latents = self.encoder.encode(
            graph
        )

        if (
            not isinstance(latents, dict)
            or "module_h" not in latents
        ):
            raise ValueError(
                "critic encoder must return module_h latents"
            )

        module_latents = latents[
            "module_h"
        ]

        if (
            module_latents.ndim != 2
            or module_latents.shape[-1]
            != self.module_latent_dim
        ):
            raise ValueError(
                "critic module latent width does not match "
                "module_latent_dim"
            )

        pooled = self._pool_mean_max(
            module_latents,
            graph["graph_ptr"],
        )

        return self.value_head(
            pooled
        )

    def compute(
        self,
        inputs,
        role: str = "",
    ):
        del role

        if not isinstance(inputs, dict):
            raise TypeError(
                "skrl critic inputs must be a dictionary"
            )

        graph = inputs.get(
            "states",
            inputs,
        )

        if not isinstance(
            graph,
            (dict, list, tuple),
        ):
            raise TypeError(
                "GlobalCritic expects structured graph states"
            )

        return (
            self.forward_global(
                graph
            ),
            {},
        )

    def act(
        self,
        inputs,
        *,
        role: str = "",
    ):
        return self.compute(
            inputs,
            role=role,
        )

    @classmethod
    def from_repository_config(
        cls,
        *,
        spec,
        feature_schema,
        repo_root: str | Path,
        device: str | torch.device | None = None,
        value_hidden_dim: int = 128,
        observation_space: Any = None,
        state_space: Any = None,
        action_space: Any = None,
    ) -> "GlobalCritic":
        """Build a fresh critic using the canonical GNN architecture.

        Only architecture/configuration is reused from H1.
        No BC-v4 actor checkpoint weights are loaded.
        """
        root = Path(
            repo_root
        )

        h1_config = load_model_config(
            root
            / "configs/bc_v1/gnn_v1.json"
        )

        hidden_dim = h1_config.get(
            "hidden_dim"
        )

        if (
            not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise ValueError(
                "gnn_v1 hidden_dim must be a positive integer"
            )

        encoder = BCGNN(
            spec,
            feature_schema,
            h1_config,
        )

        return cls(
            encoder=encoder,
            module_latent_dim=hidden_dim,
            value_hidden_dim=value_hidden_dim,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
