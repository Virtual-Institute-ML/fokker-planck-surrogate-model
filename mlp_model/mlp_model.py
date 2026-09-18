#!/usr/bin/env python3
"""Baseline MLP for the Fokker-Planck PCA surrogate."""

from __future__ import annotations

import torch
import torch.nn as nn


class FokkerPlanckMLP(nn.Module):
    """
    Map standardized physical inputs

        [log10(D0), log10(nu_collision), sigma0, t]

    to standardized PCA coefficients

        [PC1, ..., PC16].
    """

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
