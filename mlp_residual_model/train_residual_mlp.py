#!/usr/bin/env python3
"""Train the initial-condition-aware residual MLP surrogate.

Residual formulation
--------------------
The direct baseline predicts PCA coefficients z(t). This model instead uses

    z(t) = z0(sigma0) + tau * g(D0, nu_collision, sigma0, t),

where tau = t / t_max and z0 is obtained exactly from the initial
Maxwellian and the stored PCA basis.

Training target for t > 0:

    g_true = [z_true(t) - z0(sigma0)] / tau

The 16 components of g_true are standardized using TRAIN samples only.
Samples at t=0 are not needed in the loss because the initial condition
is satisfied exactly by construction.

The MLP architecture is intentionally kept identical to the baseline
MLP so that baseline-vs-residual comparisons isolate the formulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlp_residual_model.residual_mlp_model import (
    FokkerPlanckResidualMLP,
    project_initial_pca,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(path):
    with h5py.File(path, "r") as h5:
        data = {}

        for split in ("train", "validation", "test"):
            data[split] = {
                "x": np.asarray(h5[f"{split}/inputs_scaled"][:], dtype=np.float32),
                "physical_inputs": np.asarray(
                    h5[f"{split}/physical_inputs"][:], dtype=np.float64
                ),
                "targets_pca": np.asarray(
                    h5[f"{split}/targets_pca"][:], dtype=np.float64
                ),
                "trajectory_id": np.asarray(
                    h5[f"{split}/trajectory_id"][:], dtype=np.int32
                ),
                "time_index": np.asarray(
                    h5[f"{split}/time_index"][:], dtype=np.int32
                ),
            }

        meta = {
            "velocity": np.asarray(h5["velocity"][:], dtype=np.float64),
            "times": np.asarray(h5["times"][:], dtype=np.float64),
            "input_mean": np.asarray(h5["scalers/input_mean"][:], dtype=np.float64),
            "input_std": np.asarray(h5["scalers/input_std"][:], dtype=np.float64),
            "pca_mean": np.asarray(h5["pca/mean"][:], dtype=np.float64),
            "pca_components": np.asarray(
                h5["pca/components"][:], dtype=np.float64
            ),
            "pca_ev": np.asarray(
                h5["pca/explained_variance_ratio"][:], dtype=np.float64
            ),
            "pca_transform": str(h5.attrs.get("pca_transform", "raw")),
            "pca_epsilon": float(h5.attrs.get("pca_epsilon", 1e-12)),
            "n_pca": int(h5.attrs["n_pca"]),
        }

    return data, meta


def prepare_residual_targets(data, meta, residual_mean=None, residual_std=None):
    """Construct g=(z-z0)/tau for t>0 and exact z0 for every sample."""
    physical = data["physical_inputs"]
    sigma0 = physical[:, 2]
    t = physical[:, 3]

    t_max = float(np.max(meta["times"]))
    if t_max <= 0:
        raise ValueError("t_max must be positive")

    tau = t / t_max

    z0 = project_initial_pca(
        sigma0,
        meta["velocity"],
        meta["pca_mean"],
        meta["pca_components"],
        meta["pca_transform"],
        meta["pca_epsilon"],
    )

    nonzero = tau > 0.0

    g = np.empty_like(data["targets_pca"], dtype=np.float64)
    g[:] = np.nan
    g[nonzero] = (
        data["targets_pca"][nonzero] - z0[nonzero]
    ) / tau[nonzero, None]

    if residual_mean is None or residual_std is None:
        residual_mean = np.mean(g[nonzero], axis=0)
        residual_std = np.std(g[nonzero], axis=0)

        if np.any(residual_std <= 0.0):
            bad = np.where(residual_std <= 0.0)[0]
            raise ValueError(
                f"Zero residual-target variance at components {bad.tolist()}"
            )

    g_scaled = np.empty_like(g, dtype=np.float32)
    g_scaled[:] = np.nan
    g_scaled[nonzero] = (
        (g[nonzero] - residual_mean[None, :])
        / residual_std[None, :]
    ).astype(np.float32)

    return {
        "x_nonzero": data["x"][nonzero],
        "g_scaled_nonzero": g_scaled[nonzero],
        "nonzero_mask": nonzero,
        "tau": tau,
        "z0": z0,
        "g": g,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "t_max": t_max,
    }


def make_loader(x, y, batch_size, shuffle, num_workers=0):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = criterion(pred, y)

            n = x.shape[0]
            total_loss += loss.item() * n
            n_samples += n

    return total_loss / n_samples


def predict_array(model, x, batch_size, device):
    outputs = []

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            outputs.append(model(xb).cpu().numpy())

    return np.concatenate(outputs, axis=0)


def reconstruct_scores(model, data, residual, batch_size, device):
    """Predict full z(t), enforcing z(t=0)=z0 exactly."""
    z_pred = residual["z0"].copy()
    nonzero = residual["nonzero_mask"]

    if np.any(nonzero):
        g_scaled_pred = predict_array(
            model,
            data["x"][nonzero],
            batch_size,
            device,
        ).astype(np.float64)

        g_pred = (
            g_scaled_pred * residual["residual_std"][None, :]
            + residual["residual_mean"][None, :]
        )

        z_pred[nonzero] = (
            residual["z0"][nonzero]
            + residual["tau"][nonzero, None] * g_pred
        )

    return z_pred


def inverse_pca(scores, meta):
    transformed = scores @ meta["pca_components"] + meta["pca_mean"][None, :]

    if meta["pca_transform"] == "raw":
        return transformed

    if meta["pca_transform"] == "log10":
        eps = meta["pca_epsilon"]
        return np.maximum(10.0**transformed - eps, 0.0)

    raise ValueError(f"Unknown PCA transform: {meta['pca_transform']}")


def relative_l1(true, pred):
    numerator = np.sum(np.abs(pred - true), axis=1)
    denominator = np.sum(np.abs(true), axis=1)
    return numerator / np.maximum(denominator, 1e-300)


def relative_l2(true, pred):
    numerator = np.sqrt(np.sum((pred - true) ** 2, axis=1))
    denominator = np.sqrt(np.sum(true**2, axis=1))
    return numerator / np.maximum(denominator, 1e-300)


def plot_history(history, path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Residual MLP training history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram(errors, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(errors, bins=60)
    ax.set_xscale("log")
    ax.set_xlabel("Relative L1 error in reconstructed f(v,t)")
    ax.set_ylabel("Number of test snapshots")
    ax.set_title("Residual MLP test reconstruction error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_time(data, errors, path):
    t = data["physical_inputs"][:, 3]
    unique_t = np.unique(t)

    med = []
    p95 = []

    for ti in unique_t:
        e = errors[t == ti]
        med.append(np.median(e))
        p95.append(np.percentile(e, 95))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(unique_t, med, marker="o", label="median")
    ax.plot(unique_t, p95, marker="o", label="95th percentile")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Relative L1 error")
    ax.set_title("Residual MLP error versus time")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_examples(
    meta,
    data,
    f_true,
    f_pred,
    errors,
    path,
    n_examples,
    seed,
):
    rng = np.random.default_rng(seed)
    n_examples = min(n_examples, len(errors))
    chosen = rng.choice(len(errors), size=n_examples, replace=False)

    ncols = 2
    nrows = int(np.ceil(n_examples / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 3.8 * nrows),
        squeeze=False,
        sharex=True,
    )

    for panel, idx in enumerate(chosen):
        ax = axes.flat[panel]
        D0, nu, sigma0, t = data["physical_inputs"][idx]

        ax.plot(meta["velocity"], f_true[idx], label="PCA reference")
        ax.plot(
            meta["velocity"],
            f_pred[idx],
            linestyle="--",
            label="Residual MLP + PCA",
        )

        ax.set_title(
            f"D0={D0:.3g}, nu={nu:.3g}, sigma0={sigma0:.3g}, t={t:.3g}\n"
            f"relative L1={errors[idx]:.2e}",
            fontsize=9,
        )
        ax.set_ylabel("f(v,t)")
        ax.grid(alpha=0.2)

        if panel >= (nrows - 1) * ncols:
            ax.set_xlabel("v")

    for panel in range(n_examples, nrows * ncols):
        axes.flat[panel].axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(
        "Held-out test examples: PCA reference vs residual surrogate",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pc_metrics(z_true, z_pred, path):
    rows = []

    for i in range(z_true.shape[1]):
        diff = z_pred[:, i] - z_true[:, i]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))

        if np.std(z_true[:, i]) > 0 and np.std(z_pred[:, i]) > 0:
            corr = float(np.corrcoef(z_true[:, i], z_pred[:, i])[0, 1])
        else:
            corr = float("nan")

        rows.append(
            {
                "pc": i + 1,
                "rmse_raw_pca": rmse,
                "mae_raw_pca": mae,
                "correlation": corr,
            }
        )

    with Path(path).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)

    print(f"Device: {device}")
    print(f"Loading: {dataset_path}")

    data, meta = load_dataset(dataset_path)

    # Build residual targets. Only TRAIN is allowed to determine target scaling.
    residual = {}
    residual["train"] = prepare_residual_targets(data["train"], meta)

    residual_mean = residual["train"]["residual_mean"]
    residual_std = residual["train"]["residual_std"]

    residual["validation"] = prepare_residual_targets(
        data["validation"],
        meta,
        residual_mean,
        residual_std,
    )
    residual["test"] = prepare_residual_targets(
        data["test"],
        meta,
        residual_mean,
        residual_std,
    )

    print("\n=== Residual dataset ===")
    for split in ("train", "validation", "test"):
        n_total = len(data[split]["x"])
        n_nonzero = int(np.sum(residual[split]["nonzero_mask"]))
        n_zero = n_total - n_nonzero
        print(
            f"{split:10s}: total={n_total:7d}, "
            f"trained/evaluated residual={n_nonzero:7d}, "
            f"exact t=0={n_zero:5d}"
        )

    train_loader = make_loader(
        residual["train"]["x_nonzero"],
        residual["train"]["g_scaled_nonzero"],
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = make_loader(
        residual["validation"]["x_nonzero"],
        residual["validation"]["g_scaled_nonzero"],
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FokkerPlanckResidualMLP(
        input_dim=data["train"]["x"].shape[1],
        output_dim=meta["n_pca"],
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
    ).to(device)

    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n=== Model ===")
    print(model)
    print(f"Trainable parameters: {n_parameters:,}")

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    checkpoint_path = output_dir / "best_model.pt"

    print("\n=== Training ===")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            n = x.shape[0]
            total_loss += loss.item() * n
            n_samples += n

        train_loss = total_loss / n_samples
        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        scheduler.step(val_loss)

        improved = val_loss < best_val - args.min_delta

        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_dim": data["train"]["x"].shape[1],
                        "output_dim": meta["n_pca"],
                        "hidden_dims": list(args.hidden_dims),
                        "activation": args.activation,
                    },
                    "residual_mean": residual_mean,
                    "residual_std": residual_std,
                    "t_max": residual["train"]["t_max"],
                    "input_mean": meta["input_mean"],
                    "input_std": meta["input_std"],
                    "pca_mean": meta["pca_mean"],
                    "pca_components": meta["pca_components"],
                    "pca_transform": meta["pca_transform"],
                    "pca_epsilon": meta["pca_epsilon"],
                    "velocity": meta["velocity"],
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val,
                    "dataset": str(dataset_path),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % args.print_every == 0 or improved:
            print(
                f"Epoch {epoch:4d} | "
                f"train={train_loss:.6e} | "
                f"val={val_loss:.6e} | "
                f"lr={current_lr:.3e}"
                + ("  *" if improved else "")
            )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}"
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"\nLoaded best model from epoch {best_epoch} "
        f"(val={best_val:.6e})"
    )

    # ------------------------------------------------------------------
    # Test reconstruction: all samples, including exact t=0.
    # ------------------------------------------------------------------
    z_true = data["test"]["targets_pca"]
    z_pred = reconstruct_scores(
        model,
        data["test"],
        residual["test"],
        args.batch_size,
        device,
    )

    f_true = inverse_pca(z_true, meta)
    f_pred = inverse_pca(z_pred, meta)

    err_l1 = relative_l1(f_true, f_pred)
    err_l2 = relative_l2(f_true, f_pred)

    t_test = data["test"]["physical_inputs"][:, 3]
    t0_mask = t_test == 0.0
    tpos_mask = ~t0_mask

    summary = {
        "device": str(device),
        "trainable_parameters": int(n_parameters),
        "best_epoch": int(best_epoch),
        "best_validation_mse_scaled_residual": float(best_val),
        "test_relative_l1_all": {
            "median": float(np.median(err_l1)),
            "p95": float(np.percentile(err_l1, 95)),
            "max": float(np.max(err_l1)),
        },
        "test_relative_l1_t_positive": {
            "median": float(np.median(err_l1[tpos_mask])),
            "p95": float(np.percentile(err_l1[tpos_mask], 95)),
            "max": float(np.max(err_l1[tpos_mask])),
        },
        "test_relative_l1_t0": {
            "median": float(np.median(err_l1[t0_mask])),
            "p95": float(np.percentile(err_l1[t0_mask], 95)),
            "max": float(np.max(err_l1[t0_mask])),
        },
        "test_relative_l2_all": {
            "median": float(np.median(err_l2)),
            "p95": float(np.percentile(err_l2, 95)),
            "max": float(np.max(err_l2)),
        },
        "minimum_predicted_f": float(np.min(f_pred)),
        "fraction_negative_f_values": float(np.mean(f_pred < 0.0)),
        "t_max": float(residual["train"]["t_max"]),
        "formulation": "z(t)=z0(sigma0)+(t/t_max)*g(theta,t)",
        "architecture": {
            "input_dim": int(data["train"]["x"].shape[1]),
            "hidden_dims": list(args.hidden_dims),
            "output_dim": int(meta["n_pca"]),
            "activation": args.activation,
        },
    }

    # Save history.
    with (output_dir / "training_history.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "learning_rate"]
        )
        for i in range(len(history["train_loss"])):
            writer.writerow(
                [
                    i + 1,
                    history["train_loss"][i],
                    history["val_loss"][i],
                    history["learning_rate"][i],
                ]
            )

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        y_true_pca=z_true,
        y_pred_pca=z_pred,
        f_pca_true=f_true,
        f_mlp=f_pred,
        relative_l1=err_l1,
        relative_l2=err_l2,
        physical_inputs=data["test"]["physical_inputs"],
        trajectory_id=data["test"]["trajectory_id"],
        time_index=data["test"]["time_index"],
        z0=residual["test"]["z0"],
        tau=residual["test"]["tau"],
    )

    save_pc_metrics(
        z_true,
        z_pred,
        output_dir / "pc_metrics.csv",
    )

    plot_history(
        history,
        output_dir / "training_history.png",
    )
    plot_error_histogram(
        err_l1,
        output_dir / "test_error_histogram.png",
    )
    plot_error_vs_time(
        data["test"],
        err_l1,
        output_dir / "error_vs_time.png",
    )
    plot_prediction_examples(
        meta,
        data["test"],
        f_true,
        f_pred,
        err_l1,
        output_dir / "prediction_examples.png",
        args.n_examples,
        args.seed + 1,
    )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== Test results ===")
    print(
        "all t relative L1 : "
        f"median={np.median(err_l1):.3e}, "
        f"p95={np.percentile(err_l1, 95):.3e}, "
        f"max={np.max(err_l1):.3e}"
    )
    print(
        "t > 0 relative L1 : "
        f"median={np.median(err_l1[tpos_mask]):.3e}, "
        f"p95={np.percentile(err_l1[tpos_mask], 95):.3e}"
    )
    print(
        "t = 0 relative L1 : "
        f"median={np.median(err_l1[t0_mask]):.3e}, "
        f"p95={np.percentile(err_l1[t0_mask], 95):.3e}, "
        f"max={np.max(err_l1[t0_mask]):.3e}"
    )
    print(
        f"Minimum predicted f : {np.min(f_pred):.3e}"
    )
    print(
        f"Negative value frac.: {np.mean(f_pred < 0.0):.3e}"
    )

    print("\nSaved:")
    for name in (
        "best_model.pt",
        "training_history.csv",
        "training_history.png",
        "test_predictions.npz",
        "pc_metrics.csv",
        "test_error_histogram.png",
        "error_vs_time.png",
        "prediction_examples.png",
        "summary.json",
    ):
        print(f"  {output_dir / name}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--dataset",
        default="artifacts/fp_pca16_ml_dataset_v1.1.h5",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/residual",
    )

    p.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 256, 256, 128],
    )
    p.add_argument(
        "--activation",
        choices=("gelu", "relu", "silu"),
        default="gelu",
    )

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--min-lr", type=float, default=1e-6)

    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=40,
    )
    p.add_argument("--min-delta", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--n-examples", type=int, default=8)

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
