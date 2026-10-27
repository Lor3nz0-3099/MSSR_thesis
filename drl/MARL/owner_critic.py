from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from skrl.models.torch import Model


def _positive_int(
    value: Any,
    *,
    label: str,
) -> int:
    if (
        type(value) is not int
        or value <= 0
    ):
        raise ValueError(
            f"{label} must be a positive integer"
        )

    return value


class OwnerConditionedCritic(Model):
    """Shared owner-conditioned value function.

    A graph is encoded once into per-module embeddings. For every module i
    in graph g, the value head receives

        [h_i || mean_j(h_j) || max_j(h_j)]

    where j ranges only over modules belonging to graph g.

    The same value-head parameters are used for every owner and every N.
    String/module identity is intentionally outside this class: owner rows
    are addressed only by their local tensor index inside each graph
    segment.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        module_latent_dim: int,
        value_hidden_dim: int = 128,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            observation_space=None,
            action_space=None,
            device=device,
        )

        if not isinstance(
            encoder,
            nn.Module,
        ):
            raise TypeError(
                "encoder must be a torch.nn.Module"
            )

        if not callable(
            getattr(
                encoder,
                "encode",
                None,
            )
        ):
            raise TypeError(
                "encoder must expose encode(graph)"
            )

        self.module_latent_dim = (
            _positive_int(
                module_latent_dim,
                label="module_latent_dim",
            )
        )

        value_hidden_dim = (
            _positive_int(
                value_hidden_dim,
                label="value_hidden_dim",
            )
        )

        self.encoder = encoder.to(
            self.device
        )

        self.value_head = nn.Sequential(
            nn.Linear(
                3
                * self.module_latent_dim,
                value_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                value_hidden_dim,
                1,
            ),
        ).to(
            self.device
        )

    def _encode(
        self,
        graph,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if not isinstance(
            graph,
            dict,
        ):
            raise TypeError(
                "owner critic graph must be a dictionary"
            )

        latents = self.encoder.encode(
            graph
        )

        if (
            not isinstance(
                latents,
                dict,
            )
            or "module_h"
            not in latents
        ):
            raise ValueError(
                "critic encoder must return module_h latents"
            )

        module_h = latents[
            "module_h"
        ]

        if (
            not torch.is_tensor(
                module_h
            )
            or module_h.ndim != 2
            or module_h.shape[
                1
            ]
            != self.module_latent_dim
        ):
            raise ValueError(
                "invalid module_h latent shape"
            )

        graph_ptr = graph.get(
            "graph_ptr"
        )

        if graph_ptr is None:
            graph_ptr = latents.get(
                "graph_ptr"
            )

        if (
            not torch.is_tensor(
                graph_ptr
            )
            or graph_ptr.ndim != 1
            or graph_ptr.numel() < 2
        ):
            raise ValueError(
                "graph_ptr must be a 1-D tensor with at least two entries"
            )

        graph_ptr = graph_ptr.to(
            device=module_h.device,
            dtype=torch.long,
        )

        ptr = graph_ptr.detach().cpu().tolist()

        if (
            ptr[0] != 0
            or ptr[-1]
            != module_h.shape[0]
            or any(
                left > right
                for left, right
                in zip(
                    ptr,
                    ptr[1:],
                )
            )
        ):
            raise ValueError(
                "malformed graph_ptr"
            )

        if any(
            left == right
            for left, right
            in zip(
                ptr,
                ptr[1:],
            )
        ):
            raise ValueError(
                "empty module set in graph segment"
            )

        return (
            module_h,
            graph_ptr,
        )

    @staticmethod
    def _ptr_list(
        graph_ptr: torch.Tensor,
    ) -> list[int]:
        return [
            int(
                value
            )
            for value in (
                graph_ptr
                .detach()
                .cpu()
                .tolist()
            )
        ]

    def _forward_dict_all(
        self,
        graph: dict,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        module_h, graph_ptr = (
            self._encode(
                graph
            )
        )

        ptr = self._ptr_list(
            graph_ptr
        )

        features = []

        for start, end in zip(
            ptr,
            ptr[
                1:
            ],
        ):
            segment = module_h[
                start:end
            ]

            mean = segment.mean(
                dim=0,
                keepdim=True,
            )

            maximum = segment.max(
                dim=0,
                keepdim=True,
            ).values

            count = (
                end
                - start
            )

            context = torch.cat(
                (
                    segment,
                    mean.expand(
                        count,
                        -1,
                    ),
                    maximum.expand(
                        count,
                        -1,
                    ),
                ),
                dim=1,
            )

            features.append(
                context
            )

        values = self.value_head(
            torch.cat(
                features,
                dim=0,
            )
        )

        return (
            values,
            graph_ptr,
        )

    def forward_all_owners(
        self,
        graph,
    ) -> torch.Tensor:
        """Return one scalar per module, preserving graph/module row order."""
        if isinstance(
            graph,
            (
                list,
                tuple,
            ),
        ):
            if not graph:
                raise ValueError(
                    "graph snapshot list must not be empty"
                )

            return torch.cat(
                [
                    self.forward_all_owners(
                        item
                    )
                    for item in graph
                ],
                dim=0,
            )

        values, _ = (
            self._forward_dict_all(
                graph
            )
        )

        return values

    def _normalize_owner_index(
        self,
        owner_index,
        *,
        graph_count: int,
        device,
    ) -> torch.Tensor:
        if torch.is_tensor(
            owner_index
        ):
            indexes = owner_index.to(
                device=device,
                dtype=torch.long,
            )

            if indexes.ndim == 0:
                indexes = indexes.reshape(
                    1
                )

        elif type(
            owner_index
        ) is int:
            indexes = torch.tensor(
                [
                    owner_index
                ],
                dtype=torch.long,
                device=device,
            )

        elif isinstance(
            owner_index,
            Sequence,
        ):
            indexes = torch.tensor(
                list(
                    owner_index
                ),
                dtype=torch.long,
                device=device,
            )

        else:
            raise TypeError(
                "owner_index must be an int or 1-D index sequence"
            )

        if (
            indexes.ndim != 1
            or indexes.numel()
            != graph_count
        ):
            raise ValueError(
                "owner_index must provide one local index per graph"
            )

        return indexes

    def forward_owner(
        self,
        graph,
        *,
        owner_index,
    ) -> torch.Tensor:
        """Select one local owner from each graph segment."""
        if isinstance(
            graph,
            (
                list,
                tuple,
            ),
        ):
            if not graph:
                raise ValueError(
                    "graph snapshot list must not be empty"
                )

            if torch.is_tensor(
                owner_index
            ):
                indexes = (
                    owner_index
                    .detach()
                    .cpu()
                    .reshape(
                        -1
                    )
                    .tolist()
                )
            elif type(
                owner_index
            ) is int:
                indexes = [
                    owner_index
                ]
            else:
                indexes = list(
                    owner_index
                )

            if len(
                indexes
            ) != len(
                graph
            ):
                raise ValueError(
                    "owner_index must provide one local index per graph"
                )

            return torch.cat(
                [
                    self.forward_owner(
                        item,
                        owner_index=int(
                            index
                        ),
                    )
                    for item, index
                    in zip(
                        graph,
                        indexes,
                    )
                ],
                dim=0,
            )

        all_values, graph_ptr = (
            self._forward_dict_all(
                graph
            )
        )

        ptr = self._ptr_list(
            graph_ptr
        )

        graph_count = (
            len(
                ptr
            )
            - 1
        )

        indexes = (
            self._normalize_owner_index(
                owner_index,
                graph_count=graph_count,
                device=all_values.device,
            )
        )

        absolute = []

        for graph_index, (
            start,
            end,
        ) in enumerate(
            zip(
                ptr,
                ptr[
                    1:
                ],
            )
        ):
            local = int(
                indexes[
                    graph_index
                ].item()
            )

            size = (
                end
                - start
            )

            if not (
                0
                <= local
                < size
            ):
                raise ValueError(
                    "owner_index is outside its graph segment"
                )

            absolute.append(
                start
                + local
            )

        absolute_index = torch.tensor(
            absolute,
            dtype=torch.long,
            device=all_values.device,
        )

        return all_values[
            absolute_index
        ]

    @classmethod
    def from_repository_config(
        cls,
        *,
        spec,
        feature_schema,
        repo_root,
        device=None,
        value_hidden_dim: int = 128,
    ):
        """Build a fresh owner-conditioned critic from canonical H1 config.

        Only the BCGNN architecture/configuration is reused.
        No BC-v4 checkpoint weights are loaded.
        """
        from pathlib import Path

        from drl.IL.bc_v1.model import (
            BCGNN,
            load_model_config,
        )

        root = Path(
            repo_root
        )

        h1_config = load_model_config(
            root
            / "configs"
            / "bc_v1"
            / "gnn_v1.json"
        )

        hidden_dim = h1_config.get(
            "hidden_dim"
        )

        if (
            not isinstance(
                hidden_dim,
                int,
            )
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
            device=(
                "cpu"
                if device is None
                else device
            ),
        )

    def compute(
        self,
        inputs,
        role: str = "",
    ):
        del role

        if not isinstance(
            inputs,
            dict,
        ):
            raise TypeError(
                "skrl critic inputs must be a dictionary"
            )

        graph = inputs.get(
            "states"
        )

        if graph is None:
            raise ValueError(
                "owner critic inputs require states"
            )

        if (
            "owner_index"
            not in inputs
        ):
            raise ValueError(
                "owner critic inputs require owner_index"
            )

        return (
            self.forward_owner(
                graph,
                owner_index=inputs[
                    "owner_index"
                ],
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


@dataclass(frozen=True)
class AltruisticCriticPair:
    owner: OwnerConditionedCritic
    mean_field: OwnerConditionedCritic


def build_altruistic_critic_pair(
    *,
    encoder_factory: Callable[
        [],
        nn.Module,
    ],
    module_latent_dim: int,
    value_hidden_dim: int = 128,
    device: str | torch.device = "cpu",
) -> AltruisticCriticPair:
    """Build two fully independent value networks for APPO."""

    if not callable(
        encoder_factory
    ):
        raise TypeError(
            "encoder_factory must be callable"
        )

    owner_encoder = (
        encoder_factory()
    )

    mean_field_encoder = (
        encoder_factory()
    )

    if (
        owner_encoder
        is mean_field_encoder
    ):
        raise ValueError(
            "encoder_factory must return independent encoder instances"
        )

    owner = OwnerConditionedCritic(
        encoder=owner_encoder,
        module_latent_dim=(
            module_latent_dim
        ),
        value_hidden_dim=(
            value_hidden_dim
        ),
        device=device,
    )

    mean_field = OwnerConditionedCritic(
        encoder=mean_field_encoder,
        module_latent_dim=(
            module_latent_dim
        ),
        value_hidden_dim=(
            value_hidden_dim
        ),
        device=device,
    )

    owner_ids = {
        id(
            parameter
        )
        for parameter
        in owner.parameters()
    }

    mean_field_ids = {
        id(
            parameter
        )
        for parameter
        in mean_field.parameters()
    }

    if not owner_ids.isdisjoint(
        mean_field_ids
    ):
        raise ValueError(
            "owner and mean-field critics must not share parameters"
        )

    return AltruisticCriticPair(
        owner=owner,
        mean_field=mean_field,
    )



def build_repository_altruistic_critic_pair(
    *,
    spec,
    feature_schema,
    repo_root,
    device=None,
    value_hidden_dim: int = 128,
) -> AltruisticCriticPair:
    """Build fresh independent V_i and V_i^MF critics.

    Both reuse the canonical BCGNN architecture only.
    Neither loads BC-v4 actor weights and the two critics share no
    parameters with each other.
    """
    from pathlib import Path

    from drl.IL.bc_v1.model import (
        BCGNN,
        load_model_config,
    )

    root = Path(
        repo_root
    )

    h1_config = load_model_config(
        root
        / "configs"
        / "bc_v1"
        / "gnn_v1.json"
    )

    hidden_dim = h1_config.get(
        "hidden_dim"
    )

    if (
        not isinstance(
            hidden_dim,
            int,
        )
        or hidden_dim <= 0
    ):
        raise ValueError(
            "gnn_v1 hidden_dim must be a positive integer"
        )

    def encoder_factory():
        # A NEW BCGNN instance on every call.
        return BCGNN(
            spec,
            feature_schema,
            h1_config,
        )

    return build_altruistic_critic_pair(
        encoder_factory=(
            encoder_factory
        ),
        module_latent_dim=(
            hidden_dim
        ),
        value_hidden_dim=(
            value_hidden_dim
        ),
        device=(
            "cpu"
            if device is None
            else device
        ),
    )
