#!/usr/bin/env python3
"""Train a baseline MLP surrogate for the Fokker-Planck PCA dataset.

Expected dataset:
    fp_pca16_ml_dataset.h5

Inputs:
    standardized [log10(D0), log10(nu_collision), sigma0, t]

Targets:
    standardized first 16 PCA coefficients

The script:
    1. trains the MLP,
    2. monitors validation loss,
    3. saves the best checkpoint,
    4. evaluates the held-out test split,
    5. reconstructs f(v,t) from predicted PCA coefficients,
    6. saves diagnostic plots and CSV/JSON summaries.
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

from mlp_model.mlp_model import FokkerPlanckMLP


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
                "x": np.asarray(
                    h5[f"{split}/inputs_scaled"][:],
                    dtype=np.float32,
                ),
                "y": np.asarray(
                    h5[f"{split}/targets_scaled"][:],
                    dtype=np.float32,
                ),
                "physical_inputs": np.asarray(
                    h5[f"{split}/physical_inputs"][:],
                    dtype=np.float32,
                ),
                "targets_pca": np.asarray(
                    h5[f"{split}/targets_pca"][:],
                    dtype=np.float32,
                ),
                "trajectory_id": np.asarray(
                    h5[f"{split}/trajectory_id"][:],
                    dtype=np.int32,
                ),
                "time_index": np.asarray(
                    h5[f"{split}/time_index"][:],
                    dtype=np.int32,
                ),
            }

        meta = {
            "velocity": np.asarray(h5["velocity"][:], dtype=np.float64),
            "times": np.asarray(h5["times"][:], dtype=np.float64),
            "input_mean": np.asarray(
                h5["scalers/input_mean"][:],
                dtype=np.float64,
            ),
            "input_std": np.asarray(
                h5["scalers/input_std"][:],
                dtype=np.float64,
            ),
            "target_mean": np.asarray(
                h5["scalers/target_mean"][:],
                dtype=np.float64,
            ),
            "target_std": np.asarray(
                h5["scalers/target_std"][:],
                dtype=np.float64,
            ),
            "pca_mean": np.asarray(
                h5["pca/mean"][:],
                dtype=np.float64,
            ),
            "pca_components": np.asarray(
                h5["pca/components"][:],
                dtype=np.float64,
            ),
            "pca_ev": np.asarray(
                h5["pca/explained_variance_ratio"][:],
                dtype=np.float64,
            ),
            "pca_transform": str(h5.attrs.get("pca_transform", "raw")),
            "pca_epsilon": float(h5.attrs.get("pca_epsilon", 1e-12)),
            "n_pca": int(h5.attrs["n_pca"]),
        }

    return data, meta


def make_loader(x, y, batch_size, shuffle, num_workers=0):
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
    )

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

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples


def predict_array(model, x, batch_size, device):
    model.eval()

    outputs = []

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            outputs.append(pred.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def inverse_target_scaling(y_scaled, target_mean, target_std):
    return y_scaled * target_std[None, :] + target_mean[None, :]


def inverse_pca(scores, meta):
    transformed = (
        scores @ meta["pca_components"]
        + meta["pca_mean"][None, :]
    )

    if meta["pca_transform"] == "raw":
        return transformed

    if meta["pca_transform"] == "log10":
        eps = meta["pca_epsilon"]
        return np.maximum(10.0**transformed - eps, 0.0)

    raise ValueError(
        f"Unknown PCA transform: {meta['pca_transform']}"
    )


def relative_l1(true, pred):
    numerator = np.sum(np.abs(pred - true), axis=1)
    denominator = np.sum(np.abs(true), axis=1)
    denominator = np.maximum(denominator, 1e-300)
    return numerator / denominator


def relative_l2(true, pred):
    numerator = np.sqrt(np.sum((pred - true)**2, axis=1))
    denominator = np.sqrt(np.sum(true**2, axis=1))
    denominator = np.maximum(denominator, 1e-300)
    return numerator / denominator


def plot_history(history, output_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("MLP training history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_test_error_histogram(errors, output_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(errors, bins=50)
    ax.set_xscale("log")
    ax.set_xlabel("Relative L1 error in reconstructed f(v,t)")
    ax.set_ylabel("Number of test snapshots")
    ax.set_title("Test reconstruction error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_examples(
    meta,
    test,
    f_true,
    f_pred,
    errors,
    output_path,
    n_examples=8,
    seed=123,
):
    rng = np.random.default_rng(seed)

    n_examples = min(n_examples, len(errors))
    selected = rng.choice(
        len(errors),
        size=n_examples,
        replace=False,
    )

    ncols = 2
    nrows = int(np.ceil(n_examples / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 3.8 * nrows),
        squeeze=False,
        sharex=True,
    )

    v = meta["velocity"]

    for panel, idx in enumerate(selected):
        ax = axes.flat[panel]

        physical = test["physical_inputs"][idx]
        D0, nu, sigma0, time = physical

        ax.plot(v, f_true[idx], label="FP")
        ax.plot(
            v,
            f_pred[idx],
            linestyle="--",
            label="MLP + PCA",
        )

        ax.set_title(
            f"D0={D0:.3g}, nu={nu:.3g}, "
            f"sigma0={sigma0:.3g}, t={time:.3g}\n"
            f"relative L1={errors[idx]:.2e}",
            fontsize=9,
        )
        ax.set_ylabel("f(v,t)")
        ax.grid(alpha=0.20)

        if panel >= (nrows - 1) * ncols:
            ax.set_xlabel("v")

    for panel in range(n_examples, nrows * ncols):
        axes.flat[panel].axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
    )
    fig.suptitle(
        "Held-out test examples: direct FP vs MLP surrogate",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pc_metrics(y_true, y_pred, output_path):
    rows = []

    for i in range(y_true.shape[1]):
        diff = y_pred[:, i] - y_true[:, i]

        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        corr = float(
            np.corrcoef(
                y_true[:, i],
                y_pred[:, i],
            )[0, 1]
        )

        rows.append(
            {
                "pc": i + 1,
                "rmse_scaled": rmse,
                "mae_scaled": mae,
                "correlation": corr,
            }
        )

    with open(output_path, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=list(rows[0].keys()),
        )
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

    print("\n=== Dataset ===")
    for split in ("train", "validation", "test"):
        print(
            f"{split:10s}: "
            f"x={data[split]['x'].shape}, "
            f"y={data[split]['y'].shape}"
        )

    train_loader = make_loader(
        data["train"]["x"],
        data["train"]["y"],
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = make_loader(
        data["validation"]["x"],
        data["validation"]["y"],
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FokkerPlanckMLP(
        input_dim=data["train"]["x"].shape[1],
        output_dim=data["train"]["y"].shape[1],
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
    ).to(device)

    n_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
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

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    best_val = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

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

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            n_samples += batch_size

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
                        "output_dim": data["train"]["y"].shape[1],
                        "hidden_dims": list(args.hidden_dims),
                        "activation": args.activation,
                    },
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val,
                    "dataset": str(dataset_path),
                    "input_mean": meta["input_mean"],
                    "input_std": meta["input_std"],
                    "target_mean": meta["target_mean"],
                    "target_std": meta["target_std"],
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % args.print_every == 0
            or improved
        ):
            print(
                f"Epoch {epoch:4d} | "
                f"train={train_loss:.6e} | "
                f"val={val_loss:.6e} | "
                f"lr={current_lr:.3e}"
                + ("  *" if improved else "")
            )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}"
            )
            break

    # ------------------------------------------------------------------
    # Restore the best validation model
    # ------------------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"\nLoaded best model from epoch {best_epoch} "
        f"(val={best_val:.6e})"
    )

    # ------------------------------------------------------------------
    # Test prediction
    # ------------------------------------------------------------------
    y_pred_scaled = predict_array(
        model,
        data["test"]["x"],
        args.batch_size,
        device,
    )

    y_true_scaled = data["test"]["y"].astype(np.float64)
    y_pred_scaled = y_pred_scaled.astype(np.float64)

    scaled_mse = float(
        np.mean((y_pred_scaled - y_true_scaled)**2)
    )
    scaled_mae = float(
        np.mean(np.abs(y_pred_scaled - y_true_scaled))
    )

    # Back to physical PCA coefficients
    y_pred_pca = inverse_target_scaling(
        y_pred_scaled,
        meta["target_mean"],
        meta["target_std"],
    )

    y_true_pca = data["test"]["targets_pca"].astype(
        np.float64
    )

    # Direct PCA reconstruction using the true PCA scores:
    # this is the irreducible PCA truncation reference.
    f_pca_true = inverse_pca(
        y_true_pca,
        meta,
    )

    # MLP-predicted PCA reconstruction.
    f_mlp = inverse_pca(
        y_pred_pca,
        meta,
    )

    # Reference solution is the 16-PC reconstruction of the true scores.
    # For raw PCA in this project that PCA truncation error is already
    # ~1e-8, so comparison with f_pca_true isolates the MLP error.
    err_mlp_vs_pca = relative_l1(
        f_pca_true,
        f_mlp,
    )

    err_l2_mlp_vs_pca = relative_l2(
        f_pca_true,
        f_mlp,
    )

    # Basic positivity diagnostic.
    min_predicted_f = float(np.min(f_mlp))
    negative_fraction = float(np.mean(f_mlp < 0.0))

    summary = {
        "device": str(device),
        "trainable_parameters": int(n_parameters),
        "best_epoch": int(best_epoch),
        "best_validation_mse_scaled": float(best_val),
        "test_mse_scaled_pca": scaled_mse,
        "test_mae_scaled_pca": scaled_mae,
        "test_relative_l1_mlp_vs_pca": {
            "median": float(np.median(err_mlp_vs_pca)),
            "p95": float(np.percentile(err_mlp_vs_pca, 95)),
            "max": float(np.max(err_mlp_vs_pca)),
        },
        "test_relative_l2_mlp_vs_pca": {
            "median": float(np.median(err_l2_mlp_vs_pca)),
            "p95": float(np.percentile(err_l2_mlp_vs_pca, 95)),
            "max": float(np.max(err_l2_mlp_vs_pca)),
        },
        "minimum_predicted_f": min_predicted_f,
        "fraction_negative_f_values": negative_fraction,
        "architecture": {
            "input_dim": int(data["train"]["x"].shape[1]),
            "hidden_dims": list(args.hidden_dims),
            "output_dim": int(data["train"]["y"].shape[1]),
            "activation": args.activation,
        },
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    history_path = output_dir / "training_history.csv"

    with history_path.open("w", newline="") as fp:
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
        y_true_scaled=y_true_scaled,
        y_pred_scaled=y_pred_scaled,
        y_true_pca=y_true_pca,
        y_pred_pca=y_pred_pca,
        f_pca_true=f_pca_true,
        f_mlp=f_mlp,
        relative_l1=err_mlp_vs_pca,
        relative_l2=err_l2_mlp_vs_pca,
        physical_inputs=data["test"]["physical_inputs"],
        trajectory_id=data["test"]["trajectory_id"],
        time_index=data["test"]["time_index"],
    )

    save_pc_metrics(
        y_true_scaled,
        y_pred_scaled,
        output_dir / "pc_metrics.csv",
    )

    plot_history(
        history,
        output_dir / "training_history.png",
    )

    plot_test_error_histogram(
        err_mlp_vs_pca,
        output_dir / "test_error_histogram.png",
    )

    plot_prediction_examples(
        meta,
        data["test"],
        f_pca_true,
        f_mlp,
        err_mlp_vs_pca,
        output_dir / "prediction_examples.png",
        n_examples=args.n_examples,
        seed=args.seed + 1,
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== Test results ===")
    print(f"Scaled PCA MSE       : {scaled_mse:.6e}")
    print(f"Scaled PCA MAE       : {scaled_mae:.6e}")
    print(
        "f(v,t) relative L1  : "
        f"median={np.median(err_mlp_vs_pca):.3e}, "
        f"p95={np.percentile(err_mlp_vs_pca, 95):.3e}, "
        f"max={np.max(err_mlp_vs_pca):.3e}"
    )
    print(
        "f(v,t) relative L2  : "
        f"median={np.median(err_l2_mlp_vs_pca):.3e}, "
        f"p95={np.percentile(err_l2_mlp_vs_pca, 95):.3e}"
    )
    print(f"Minimum predicted f  : {min_predicted_f:.3e}")
    print(f"Negative value frac. : {negative_fraction:.3e}")

    print("\nSaved:")
    for name in (
        "best_model.pt",
        "training_history.csv",
        "training_history.png",
        "test_predictions.npz",
        "pc_metrics.csv",
        "test_error_histogram.png",
        "prediction_examples.png",
        "summary.json",
    ):
        print(f"  {output_dir / name}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        default="artifacts/fp_pca16_ml_dataset_v1.1.h5",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/baseline",
    )

    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 256, 256, 128],
    )
    parser.add_argument(
        "--activation",
        choices=("gelu", "relu", "silu"),
        default="gelu",
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=40,
    )
    parser.add_argument("--min-delta", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--n-examples", type=int, default=8)

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
