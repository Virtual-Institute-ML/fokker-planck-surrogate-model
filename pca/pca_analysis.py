#!/usr/bin/env python3
"""PCA analysis for the Fokker-Planck trajectory dataset.

The HDF5 dataset is expected to contain

    velocity                     [N_v]
    times                        [N_t]
    parameters                   [N_traj, N_param]
    parameter_names              [N_param]
    solutions                    [N_traj, N_t, N_v]

The trajectory split is performed BEFORE PCA:
    train trajectories -> fit PCA basis
    validation          -> reserved for later ML
    test                -> PCA reconstruction evaluation

By default PCA is applied directly to f(v,t) without per-bin
standardization. This is appropriate for the current unit-normalized
smooth distributions and keeps the PCA modes easy to interpret.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def decode_names(values):
    return [
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in values
    ]


def make_trajectory_split(
    n_trajectories: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
):
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_trajectories)

    n_train = int(round(train_fraction * n_trajectories))
    n_val = int(round(val_fraction * n_trajectories))
    n_train = min(max(n_train, 1), n_trajectories - 2)
    n_val = min(max(n_val, 1), n_trajectories - n_train - 1)

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def read_trajectory_block(h5, indices):
    """Read selected trajectories and flatten snapshots to [N_sample, N_v]."""
    solutions = h5["solutions"]
    # h5py fancy indexing requires increasing indices.
    indices = np.asarray(np.sort(indices), dtype=int)
    block = solutions[indices]
    return np.asarray(block, dtype=np.float64).reshape(-1, block.shape[-1])


def forward_transform(x, mode, epsilon):
    if mode == "raw":
        return x
    if mode == "log10":
        return np.log10(np.maximum(x, 0.0) + epsilon)
    raise ValueError(f"Unknown transform: {mode}")


def inverse_transform(x, mode, epsilon):
    if mode == "raw":
        return x
    if mode == "log10":
        return np.maximum(10.0**x - epsilon, 0.0)
    raise ValueError(f"Unknown transform: {mode}")


def relative_l1(true, pred):
    denom = np.sum(np.abs(true), axis=1)
    denom = np.maximum(denom, 1e-300)
    return np.sum(np.abs(pred - true), axis=1) / denom


def relative_l2(true, pred):
    denom = np.sqrt(np.sum(true**2, axis=1))
    denom = np.maximum(denom, 1e-300)
    return np.sqrt(np.sum((pred - true)**2, axis=1)) / denom


def reconstruction_from_scores(pca, scores, n_components):
    return (
        scores[:, :n_components] @ pca.components_[:n_components]
        + pca.mean_
    )


def component_grid(max_components):
    candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    values = [x for x in candidates if x <= max_components]
    if max_components not in values:
        values.append(max_components)
    return sorted(set(values))


def first_component_reaching(cumulative, target):
    hits = np.where(cumulative >= target)[0]
    if len(hits) == 0:
        return None
    return int(hits[0] + 1)


def plot_explained_variance(pca, output):
    n = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(n, cumulative, marker="o", markersize=3)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("PCA cumulative explained variance")
    ax.set_ylim(0.0, 1.005)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_components(pca, velocity, output, n_show=6):
    n_show = min(n_show, pca.n_components_)

    fig, axes = plt.subplots(
        n_show + 1,
        1,
        figsize=(8.0, 2.0 * (n_show + 1)),
        sharex=True,
    )

    axes[0].plot(velocity, pca.mean_)
    axes[0].set_ylabel("mean")
    axes[0].set_title("PCA mean and leading components")
    axes[0].grid(alpha=0.20)

    for i in range(n_show):
        axes[i + 1].plot(velocity, pca.components_[i])
        axes[i + 1].set_ylabel(f"PC{i + 1}")
        axes[i + 1].grid(alpha=0.20)

    axes[-1].set_xlabel("v")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_curve(rows, output):
    k = np.array([r["n_components"] for r in rows])
    l1_med = np.array([r["l1_median"] for r in rows])
    l1_p95 = np.array([r["l1_p95"] for r in rows])
    l2_med = np.array([r["l2_median"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(k, l1_med, marker="o", label="relative L1 median")
    ax.plot(k, l1_p95, marker="o", label="relative L1 95th percentile")
    ax.plot(k, l2_med, marker="o", label="relative L2 median")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("PCA reconstruction error on held-out trajectories")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_examples(
    h5_path,
    test_idx,
    pca,
    n_components,
    transform,
    epsilon,
    output,
    n_examples,
    seed,
):
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as h5:
        velocity = h5["velocity"][:]
        times = h5["times"][:]
        parameters = h5["parameters"][:]
        names = decode_names(h5["parameter_names"][:])

        n_examples = min(n_examples, len(test_idx))
        chosen_traj = rng.choice(test_idx, size=n_examples, replace=False)
        chosen_times = rng.integers(0, len(times), size=n_examples)

        ncols = 2
        nrows = int(np.ceil(n_examples / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7.0 * ncols, 3.7 * nrows),
            squeeze=False,
            sharex=True,
        )

        for panel, (traj, it) in enumerate(zip(chosen_traj, chosen_times)):
            ax = axes.flat[panel]
            true = np.asarray(h5["solutions"][int(traj), int(it)], dtype=np.float64)

            x = forward_transform(true[None, :], transform, epsilon)
            scores = pca.transform(x)
            recon_t = reconstruction_from_scores(pca, scores, n_components)
            pred = inverse_transform(recon_t, transform, epsilon)[0]

            err = relative_l1(true[None, :], pred[None, :])[0]

            ax.plot(velocity, true, label="FP")
            ax.plot(velocity, pred, linestyle="--", label=f"PCA ({n_components} PCs)")
            ax.set_title(
                f"traj={traj}, t={times[it]:.2f}, rel.L1={err:.2e}\n"
                + ", ".join(
                    f"{name}={value:.2e}" if abs(value) < 0.1
                    else f"{name}={value:.3g}"
                    for name, value in zip(names, parameters[traj])
                ),
                fontsize=9,
            )
            ax.set_ylabel("f(v,t)")
            ax.grid(alpha=0.20)

            if panel >= (nrows - 1) * ncols:
                ax.set_xlabel("v")

        for panel in range(n_examples, nrows * ncols):
            axes.flat[panel].axis("off")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle("PCA reconstruction examples: held-out trajectories", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        velocity = h5["velocity"][:]
        times = h5["times"][:]
        parameters = h5["parameters"][:]
        parameter_names = decode_names(h5["parameter_names"][:])
        n_traj, n_time, n_velocity = h5["solutions"].shape

        train_idx, val_idx, test_idx = make_trajectory_split(
            n_traj,
            args.train_fraction,
            args.val_fraction,
            args.seed,
        )

        print("=== Dataset ===")
        print(f"Input              : {input_path}")
        print(f"Trajectories       : {n_traj}")
        print(f"Snapshots/traj     : {n_time}")
        print(f"Velocity bins      : {n_velocity}")
        print(f"Train/val/test     : {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

        print("\nLoading training snapshots...")
        train_raw = read_trajectory_block(h5, train_idx)
        print(f"Training matrix    : {train_raw.shape}")

        print("Loading test snapshots...")
        test_raw = read_trajectory_block(h5, test_idx)
        print(f"Test matrix        : {test_raw.shape}")

    train_x = forward_transform(train_raw, args.transform, args.epsilon)
    test_x = forward_transform(test_raw, args.transform, args.epsilon)

    max_components = min(args.max_components, n_velocity, train_x.shape[0])
    if max_components < 1:
        raise ValueError("No PCA components available.")

    print(f"\nFitting PCA with up to {max_components} components...")
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized" if max_components < min(train_x.shape) else "full",
        random_state=args.seed,
    )
    pca.fit(train_x)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    scores_test = pca.transform(test_x)

    # Save basis and exact data split for reproducibility / later MLP work.
    joblib.dump(
        {
            "pca": pca,
            "transform": args.transform,
            "epsilon": args.epsilon,
            "velocity": velocity,
            "times": times,
            "parameter_names": parameter_names,
        },
        output_dir / "pca_model.joblib",
    )
    np.savez(
        output_dir / "trajectory_split.npz",
        train=train_idx,
        validation=val_idx,
        test=test_idx,
    )

    # Evaluate reconstruction using increasing numbers of PCs.
    rows = []
    print("\n=== Reconstruction on held-out test trajectories ===")
    for k in component_grid(max_components):
        recon_x = reconstruction_from_scores(pca, scores_test, k)
        recon_raw = inverse_transform(recon_x, args.transform, args.epsilon)

        l1 = relative_l1(test_raw, recon_raw)
        l2 = relative_l2(test_raw, recon_raw)

        row = {
            "n_components": int(k),
            "explained_variance": float(cumulative[k - 1]),
            "l1_median": float(np.median(l1)),
            "l1_p95": float(np.percentile(l1, 95)),
            "l1_max": float(np.max(l1)),
            "l2_median": float(np.median(l2)),
            "l2_p95": float(np.percentile(l2, 95)),
        }
        rows.append(row)

        print(
            f"{k:4d} PCs | EV={row['explained_variance']:.8f} "
            f"| L1 median={row['l1_median']:.3e} "
            f"| L1 p95={row['l1_p95']:.3e}"
        )

    with (output_dir / "reconstruction_metrics.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    thresholds = [float(x) for x in args.variance_targets.split(",")]
    target_rows = []
    print("\n=== Explained-variance targets ===")
    for target in thresholds:
        k = first_component_reaching(cumulative, target)
        target_rows.append((target, k))
        if k is None:
            print(f"{target:.6f}: not reached within {max_components} PCs")
        else:
            print(f"{target:.6f}: {k} PCs")

    # Use 99.9% EV for the example reconstruction when available.
    preferred_target = 0.999
    k_example = first_component_reaching(cumulative, preferred_target)
    if k_example is None:
        k_example = max_components

    plot_explained_variance(
        pca,
        output_dir / "explained_variance.png",
    )
    plot_components(
        pca,
        velocity,
        output_dir / "pca_components.png",
        n_show=args.n_show_components,
    )
    plot_error_curve(
        rows,
        output_dir / "reconstruction_error.png",
    )
    plot_reconstruction_examples(
        input_path,
        test_idx,
        pca,
        k_example,
        args.transform,
        args.epsilon,
        output_dir / "reconstruction_examples.png",
        args.n_examples,
        args.seed + 1,
    )

    with (output_dir / "pca_summary.txt").open("w") as fp:
        fp.write("Fokker-Planck PCA summary\n")
        fp.write("=========================\n\n")
        fp.write(f"input: {input_path}\n")
        fp.write(f"transform: {args.transform}\n")
        fp.write(f"trajectories: {n_traj}\n")
        fp.write(f"snapshots_per_trajectory: {n_time}\n")
        fp.write(f"velocity_bins: {n_velocity}\n")
        fp.write(
            f"train/validation/test trajectories: "
            f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}\n"
        )
        fp.write(f"max_components_fit: {max_components}\n\n")
        fp.write("Explained variance targets\n")
        for target, k in target_rows:
            if k is None:
                fp.write(f"  {target:.6f}: not reached <= {max_components}\n")
            else:
                fp.write(f"  {target:.6f}: {k} components\n")
        fp.write(f"\nexample_reconstruction_components: {k_example}\n")

    print("\nSaved:")
    print(f"  {output_dir / 'pca_model.joblib'}")
    print(f"  {output_dir / 'trajectory_split.npz'}")
    print(f"  {output_dir / 'reconstruction_metrics.csv'}")
    print(f"  {output_dir / 'explained_variance.png'}")
    print(f"  {output_dir / 'pca_components.png'}")
    print(f"  {output_dir / 'reconstruction_error.png'}")
    print(f"  {output_dir / 'reconstruction_examples.png'}")
    print(f"  {output_dir / 'pca_summary.txt'}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="artifacts/fp_dataset_v1.1.h5")
    p.add_argument("--output-dir", default="artifacts/pca")

    p.add_argument("--train-fraction", type=float, default=0.80)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-components", type=int, default=64)
    p.add_argument(
        "--variance-targets",
        default="0.99,0.999,0.9999",
        help="Comma-separated cumulative explained-variance targets.",
    )

    p.add_argument(
        "--transform",
        choices=("raw", "log10"),
        default="raw",
        help="PCA representation. Start with raw; log10 is useful for tail-focused tests.",
    )
    p.add_argument("--epsilon", type=float, default=1e-12)

    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--n-show-components", type=int, default=6)

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
