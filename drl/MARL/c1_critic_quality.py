
"""Read-only pre-update critic diagnostics for C1."""

import math
import torch


def _channel(values_map, returns_map, samples, owners, prefix):
    residuals = []
    targets = []

    for owner in owners:
        if owner not in values_map or owner not in returns_map:
            raise ValueError(f"Missing {prefix} channel: {owner}")

        values = values_map[owner].detach().reshape(-1)
        returns = returns_map[owner].detach().reshape(-1)

        if values.numel() != len(samples) or returns.numel() != len(samples):
            raise ValueError(f"Misaligned {prefix} tensors")

        mask = torch.tensor(
            [
                bool(sample.rollout_active)
                and bool(sample.owner_active_mask.get(owner, False))
                for sample in samples
            ],
            dtype=torch.bool,
            device=values.device,
        )

        if not bool(mask.any()):
            continue

        predicted = values[mask].double()
        target = returns[mask].to(
            device=values.device,
            dtype=torch.float64,
        )

        if not (
            bool(torch.isfinite(predicted).all())
            and bool(torch.isfinite(target).all())
        ):
            raise ValueError(f"Nonfinite active {prefix} critic data")

        residuals.append(target - predicted)
        targets.append(target)

    if not targets:
        return {
            f"{prefix}_fit_count": 0,
            f"{prefix}_fit_evaluable": 0,
            f"{prefix}_fit_rmse": 0.0,
            f"{prefix}_fit_return_std": 0.0,
            f"{prefix}_fit_explained_variance": 0.0,
        }

    error = torch.cat(residuals)
    target = torch.cat(targets)

    count = int(target.numel())
    variance = float(target.var(unbiased=False))
    rmse = math.sqrt(float(error.square().mean()))
    target_std = math.sqrt(variance)

    # Explained variance is undefined for nearly constant returns.
    evaluable = count >= 2 and variance > 1e-10

    explained_variance = (
        float(1.0 - error.var(unbiased=False) / variance)
        if evaluable
        else 0.0
    )

    if not all(
        math.isfinite(x)
        for x in (rmse, target_std, explained_variance)
    ):
        raise ValueError(f"Nonfinite {prefix} diagnostic")

    return {
        f"{prefix}_fit_count": count,
        f"{prefix}_fit_evaluable": int(evaluable),
        f"{prefix}_fit_rmse": rmse,
        f"{prefix}_fit_return_std": target_std,
        f"{prefix}_fit_explained_variance": explained_variance,
    }


def critic_fit_diagnostics(
    *,
    owner_values,
    mean_field_values,
    owner_returns,
    mean_field_returns,
    structured_samples,
    possible_agents,
):
    """Measure pre-update fit without changing tensors or gradients."""

    return {
        **_channel(
            owner_values,
            owner_returns,
            structured_samples,
            possible_agents,
            "owner",
        ),
        **_channel(
            mean_field_values,
            mean_field_returns,
            structured_samples,
            possible_agents,
            "mean_field",
        ),
    }
