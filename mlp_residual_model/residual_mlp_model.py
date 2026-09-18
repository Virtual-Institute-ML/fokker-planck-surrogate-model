#!/usr/bin/env python3
"""Residual MLP for the Fokker-Planck PCA surrogate.

Formulation
-----------
Instead of directly predicting z(t), predict a time-normalized residual g:

    z(t) = z0(sigma0) + tau * g(theta, t)

where

    theta = [D0, nu_collision, sigma0]
    tau   = t / t_max

and z0(sigma0) is computed exactly by projecting the initial Maxwellian
onto the fixed PCA basis.

This guarantees

    z(t=0) = z0(sigma0)

by construction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FokkerPlanckResidualMLP(nn.Module):
    """Predict standardized time-normalized PCA residual coefficients."""

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 16,
        hidden_dims=(128, 256, 256, 128),
        activation: str = "gelu",
    ):
        super().__init__()

        if activation == "gelu":
            act_cls = nn.GELU
        elif activation == "relu":
            act_cls = nn.ReLU
        elif activation == "silu":
            act_cls = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_cls())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


def discrete_maxwellian(v: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
    """Return unit-normalized 1D Maxwellians on the supplied velocity grid.

    Parameters
    ----------
    v
        Velocity-cell centers, shape [Nv].
    sigma
        Scalar or array of widths, shape [N].

    Returns
    -------
    f0
        Shape [N, Nv] for array sigma, or [Nv] for scalar sigma.
    """
    v = np.asarray(v, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    if np.any(sigma_arr <= 0.0):
        raise ValueError("sigma must be positive")

    scalar_input = sigma_arr.ndim == 0
    sigma_1d = np.atleast_1d(sigma_arr)

    f = np.exp(
        -0.5 * (v[None, :] / sigma_1d[:, None]) ** 2
    ) / (sigma_1d[:, None] * np.sqrt(2.0 * np.pi))

    dv = float(v[1] - v[0])
    mass = np.sum(f, axis=1, keepdims=True) * dv
    f = f / mass

    return f[0] if scalar_input else f


def solution_forward_transform(
    f: np.ndarray,
    mode: str = "raw",
    epsilon: float = 1e-12,
) -> np.ndarray:
    if mode == "raw":
        return f
    if mode == "log10":
        return np.log10(np.maximum(f, 0.0) + epsilon)
    raise ValueError(f"Unsupported PCA transform: {mode}")


def project_initial_pca(
    sigma0: np.ndarray | float,
    velocity: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    pca_transform: str = "raw",
    pca_epsilon: float = 1e-12,
) -> np.ndarray:
    """Project the exact initial Maxwellian onto the retained PCA basis."""
    f0 = discrete_maxwellian(velocity, sigma0)

    scalar_input = f0.ndim == 1
    if scalar_input:
        f0 = f0[None, :]

    x0 = solution_forward_transform(
        f0,
        mode=pca_transform,
        epsilon=pca_epsilon,
    )

    z0 = (x0 - pca_mean[None, :]) @ pca_components.T

    return z0[0] if scalar_input else z0
