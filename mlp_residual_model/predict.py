#!/usr/bin/env python3
"""Run the pretrained residual Fokker–Planck surrogate for one parameter set.

The checkpoint contains the input scaler, PCA basis, residual-target scaler,
and velocity grid, so no training dataset is required for inference.

Example
-------
python -m mlp_residual_model.predict \
    --d0 0.1 --nu 0.1 --sigma0 1.0 --time 2.0 \
    --plot artifacts/prediction.png \
    --csv artifacts/prediction.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mlp_residual_model.residual_mlp_model import (
    FokkerPlanckResidualMLP,
    project_initial_pca,
)

TRAINING_RANGES = {
    "d0": (1e-2, 1.0),
    "nu": (1e-2, 1.0),
    "sigma0": (0.5, 2.0),
    "time": (0.0, 5.0),
}


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["model_config"]
    model = FokkerPlanckResidualMLP(
        input_dim=int(config["input_dim"]),
        output_dim=int(config["output_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        activation=str(config["activation"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def warn_if_extrapolating(d0: float, nu: float, sigma0: float, time: float):
    values = {"d0": d0, "nu": nu, "sigma0": sigma0, "time": time}
    outside = []
    for name, value in values.items():
        lo, hi = TRAINING_RANGES[name]
        if not (lo <= value <= hi):
            outside.append(f"{name}={value:g} (trained on [{lo:g}, {hi:g}])")
    if outside:
        warnings.warn(
            "Extrapolation requested outside the v1.1 training domain: "
            + "; ".join(outside),
            RuntimeWarning,
        )


def inverse_pca(scores, checkpoint):
    mean = np.asarray(checkpoint["pca_mean"], dtype=np.float64)
    components = np.asarray(checkpoint["pca_components"], dtype=np.float64)
    transformed = scores @ components + mean
    mode = str(checkpoint.get("pca_transform", "raw"))
    epsilon = float(checkpoint.get("pca_epsilon", 1e-12))
    if mode == "raw":
        return transformed
    if mode == "log10":
        return np.maximum(10.0**transformed - epsilon, 0.0)
    raise ValueError(f"Unsupported PCA transform: {mode}")


def predict(model, checkpoint, d0, nu, sigma0, time, device):
    if d0 <= 0 or nu <= 0 or sigma0 <= 0 or time < 0:
        raise ValueError("d0, nu, and sigma0 must be positive; time must be >= 0.")

    velocity = np.asarray(checkpoint["velocity"], dtype=np.float64)
    pca_mean = np.asarray(checkpoint["pca_mean"], dtype=np.float64)
    pca_components = np.asarray(checkpoint["pca_components"], dtype=np.float64)
    pca_transform = str(checkpoint.get("pca_transform", "raw"))
    pca_epsilon = float(checkpoint.get("pca_epsilon", 1e-12))

    z0 = project_initial_pca(
        sigma0,
        velocity,
        pca_mean,
        pca_components,
        pca_transform,
        pca_epsilon,
    )

    if time == 0.0:
        z = z0
    else:
        raw_input = np.array(
            [np.log10(d0), np.log10(nu), sigma0, time],
            dtype=np.float64,
        )
        input_mean = np.asarray(checkpoint["input_mean"], dtype=np.float64)
        input_std = np.asarray(checkpoint["input_std"], dtype=np.float64)
        x = ((raw_input - input_mean) / input_std).astype(np.float32)

        with torch.no_grad():
            g_scaled = model(
                torch.from_numpy(x[None, :]).to(device)
            ).cpu().numpy()[0].astype(np.float64)

        residual_mean = np.asarray(checkpoint["residual_mean"], dtype=np.float64)
        residual_std = np.asarray(checkpoint["residual_std"], dtype=np.float64)
        g = g_scaled * residual_std + residual_mean
        tau = time / float(checkpoint["t_max"])
        z = z0 + tau * g

    f = inverse_pca(z, checkpoint)
    return velocity, np.asarray(f, dtype=np.float64)


def main(args):
    warn_if_extrapolating(args.d0, args.nu, args.sigma0, args.time)
    device = choose_device(args.device)
    model, checkpoint = load_model(Path(args.checkpoint), device)
    velocity, f = predict(
        model,
        checkpoint,
        args.d0,
        args.nu,
        args.sigma0,
        args.time,
        device,
    )

    dv = float(velocity[1] - velocity[0])
    mass = float(np.sum(f) * dv)

    print(f"device       : {device}")
    print(f"D0           : {args.d0:g}")
    print(f"nu_collision : {args.nu:g}")
    print(f"sigma0       : {args.sigma0:g}")
    print(f"t            : {args.time:g}")
    print(f"integral f dv: {mass:.12f}")
    print(f"minimum f    : {np.min(f):.6e}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            path,
            np.column_stack((velocity, f)),
            delimiter=",",
            header="v,f",
            comments="",
        )
        print(f"saved CSV    : {path}")

    if args.plot:
        path = Path(args.plot)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(velocity, f, linewidth=2.0)
        ax.set_xlabel(r"$v$")
        ax.set_ylabel(r"$f(v,t)$")
        ax.set_title(
            rf"Residual surrogate: $D_0={args.d0:g}$, "
            rf"$\nu={args.nu:g}$, $\sigma_0={args.sigma0:g}$, "
            rf"$t={args.time:g}$"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved plot   : {path}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="checkpoints/residual_v1.1/best_model.pt",
    )
    p.add_argument("--d0", type=float, required=True)
    p.add_argument("--nu", type=float, required=True)
    p.add_argument("--sigma0", type=float, required=True)
    p.add_argument("--time", type=float, required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--plot", default=None)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
