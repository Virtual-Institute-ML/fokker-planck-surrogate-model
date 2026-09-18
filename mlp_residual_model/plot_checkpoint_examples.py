#!/usr/bin/env python3
"""Plot four held-out FP solver vs residual-MLP predictions from a checkpoint.

Blue/reference curve:
    raw Fokker-Planck solution from the original trajectory HDF5

Prediction curve:
    residual MLP checkpoint -> PCA coefficients -> inverse PCA

The script evaluates the full held-out test set, then chooses one
representative (median-error) example near each requested time. This
avoids cherry-picking unusually good or bad examples.

Example
-------
python3 mlp_residual_model/plot_checkpoint_examples.py \
    --checkpoint checkpoints/residual_v1.1/best_model.pt \
    --ml-dataset artifacts/fp_pca16_ml_dataset_v1.1.h5 \
    --fp-dataset artifacts/fp_dataset_v1.1.h5 \
    --output artifacts/prediction_examples_4_reproduced.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mlp_residual_model.residual_mlp_model import (
    FokkerPlanckResidualMLP,
    project_initial_pca,
)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(
        path,
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


def load_test_data(path: Path):
    with h5py.File(path, "r") as h5:
        data = {
            "x": np.asarray(
                h5["test/inputs_scaled"][:],
                dtype=np.float32,
            ),
            "physical_inputs": np.asarray(
                h5["test/physical_inputs"][:],
                dtype=np.float64,
            ),
            "trajectory_id": np.asarray(
                h5["test/trajectory_id"][:],
                dtype=np.int64,
            ),
            "time_index": np.asarray(
                h5["test/time_index"][:],
                dtype=np.int64,
            ),
        }

    return data


def predict_scaled_residuals(
    model,
    x,
    batch_size,
    device,
):
    outputs = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))

            xb = torch.from_numpy(
                x[start:stop]
            ).to(device)

            pred = model(xb)
            outputs.append(pred.cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float64)


def inverse_pca(
    scores,
    pca_mean,
    pca_components,
    transform,
    epsilon,
):
    transformed = (
        scores @ pca_components
        + pca_mean[None, :]
    )

    if transform == "raw":
        return transformed

    if transform == "log10":
        return np.maximum(
            10.0**transformed - epsilon,
            0.0,
        )

    raise ValueError(f"Unsupported PCA transform: {transform}")


def reconstruct_residual_predictions(
    model,
    checkpoint,
    test,
    batch_size,
    device,
):
    physical = test["physical_inputs"]

    sigma0 = physical[:, 2]
    time = physical[:, 3]

    velocity = np.asarray(
        checkpoint["velocity"],
        dtype=np.float64,
    )
    pca_mean = np.asarray(
        checkpoint["pca_mean"],
        dtype=np.float64,
    )
    pca_components = np.asarray(
        checkpoint["pca_components"],
        dtype=np.float64,
    )

    pca_transform = str(
        checkpoint.get("pca_transform", "raw")
    )
    pca_epsilon = float(
        checkpoint.get("pca_epsilon", 1e-12)
    )

    residual_mean = np.asarray(
        checkpoint["residual_mean"],
        dtype=np.float64,
    )
    residual_std = np.asarray(
        checkpoint["residual_std"],
        dtype=np.float64,
    )

    t_max = float(checkpoint["t_max"])

    # Exact initial PCA coefficients.
    z0 = project_initial_pca(
        sigma0,
        velocity,
        pca_mean,
        pca_components,
        pca_transform,
        pca_epsilon,
    )

    tau = time / t_max

    g_scaled_pred = predict_scaled_residuals(
        model,
        test["x"],
        batch_size,
        device,
    )

    g_pred = (
        g_scaled_pred * residual_std[None, :]
        + residual_mean[None, :]
    )

    z_pred = (
        z0
        + tau[:, None] * g_pred
    )

    # Enforce t=0 explicitly, avoiding any round-off from multiplication.
    zero_mask = time == 0.0
    z_pred[zero_mask] = z0[zero_mask]

    f_pred = inverse_pca(
        z_pred,
        pca_mean,
        pca_components,
        pca_transform,
        pca_epsilon,
    )

    return velocity, f_pred


def load_raw_solver_solutions(
    fp_dataset_path,
    trajectory_id,
    time_index,
):
    """Read exact raw solver snapshots corresponding to ML test samples."""
    n_samples = len(trajectory_id)

    with h5py.File(fp_dataset_path, "r") as h5:
        solutions = h5["solutions"]
        n_velocity = solutions.shape[-1]

        f_solver = np.empty(
            (n_samples, n_velocity),
            dtype=np.float64,
        )

        # Group by trajectory to avoid many individual HDF5 reads.
        unique_trajectories = np.unique(trajectory_id)

        for traj in unique_trajectories:
            mask = trajectory_id == traj
            positions = np.where(mask)[0]
            local_time_indices = time_index[mask]

            trajectory_block = np.asarray(
                solutions[int(traj)],
                dtype=np.float64,
            )

            f_solver[positions] = trajectory_block[
                local_time_indices
            ]

    return f_solver


def relative_l1(true, pred):
    numerator = np.sum(
        np.abs(pred - true),
        axis=1,
    )
    denominator = np.sum(
        np.abs(true),
        axis=1,
    )

    return numerator / np.maximum(
        denominator,
        1e-300,
    )


def choose_representative_indices(
    times,
    errors,
    requested_times,
):
    """Choose median-error example at each requested output time."""
    selected = []

    unique_times = np.unique(times)

    for requested in requested_times:
        actual_time = unique_times[
            np.argmin(np.abs(unique_times - requested))
        ]

        candidates = np.where(
            np.isclose(times, actual_time)
        )[0]

        local_errors = errors[candidates]
        median_error = np.median(local_errors)

        local_index = np.argmin(
            np.abs(local_errors - median_error)
        )

        selected.append(
            int(candidates[local_index])
        )

    return selected


def plot_four_examples(
    velocity,
    f_solver,
    f_pred,
    test,
    errors,
    indices,
    output,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8.5),
        sharex=True,
    )

    for ax, idx in zip(axes.flat, indices):
        D0, nu, sigma0, t = (
            test["physical_inputs"][idx]
        )

        ax.plot(
            velocity,
            f_solver[idx],
            linewidth=2.0,
            label="FP solver",
        )
        ax.plot(
            velocity,
            f_pred[idx],
            linestyle="--",
            linewidth=2.0,
            label="Residual MLP + PCA",
        )

        ax.set_title(
            rf"$D_0={D0:.3g}$, "
            rf"$\nu={nu:.3g}$, "
            rf"$\sigma_0={sigma0:.3g}$, "
            rf"$t={t:.2f}$"
            "\n"
            rf"relative $L_1={errors[idx]:.2e}$",
            fontsize=10,
        )

        ax.set_xlabel(r"$v$")
        ax.set_ylabel(r"$f(v,t)$")
        ax.grid(alpha=0.20)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
    )

    #fig.suptitle(
    #    "Held-out test: Fokker–Planck solver vs residual ML surrogate",
    #    y=0.98,
    #    fontsize=14,
    #)

    fig.tight_layout(
        rect=(0, 0, 1, 0.94)
    )

    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main(args):
    checkpoint_path = Path(args.checkpoint)
    ml_dataset_path = Path(args.ml_dataset)
    fp_dataset_path = Path(args.fp_dataset)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device(args.device)

    print(f"Device      : {device}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"ML dataset  : {ml_dataset_path}")
    print(f"FP dataset  : {fp_dataset_path}")

    model, checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    test = load_test_data(
        ml_dataset_path
    )

    velocity, f_pred = reconstruct_residual_predictions(
        model,
        checkpoint,
        test,
        args.batch_size,
        device,
    )

    print("Loading raw held-out FP solver snapshots...")

    f_solver = load_raw_solver_solutions(
        fp_dataset_path,
        test["trajectory_id"],
        test["time_index"],
    )

    errors = relative_l1(
        f_solver,
        f_pred,
    )

    selected = choose_representative_indices(
        test["physical_inputs"][:, 3],
        errors,
        args.times,
    )

    print("\nSelected representative cases:")
    for rank, idx in enumerate(selected, start=1):
        D0, nu, sigma0, t = (
            test["physical_inputs"][idx]
        )

        print(
            f"{rank}: index={idx:6d}, "
            f"traj={test['trajectory_id'][idx]:4d}, "
            f"D0={D0:.5g}, "
            f"nu={nu:.5g}, "
            f"sigma0={sigma0:.5g}, "
            f"t={t:.3f}, "
            f"L1={errors[idx]:.3e}"
        )

    plot_four_examples(
        velocity,
        f_solver,
        f_pred,
        test,
        errors,
        selected,
        output_path,
    )

    print(f"\nSaved: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/residual_v1.1/best_model.pt",
    )
    parser.add_argument(
        "--ml-dataset",
        default="artifacts/fp_pca16_ml_dataset_v1.1.h5",
    )
    parser.add_argument(
        "--fp-dataset",
        default="artifacts/fp_dataset_v1.1.h5",
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/"
            "prediction_examples_4_reproduced.png"
        ),
    )

    parser.add_argument(
        "--times",
        type=float,
        nargs=4,
        default=[0.0, 0.5, 2.5, 4.5],
        help=(
            "Four target times. "
            "The nearest saved FP times are used."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
