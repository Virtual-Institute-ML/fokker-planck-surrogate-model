#!/usr/bin/env python3
"""Analyze where the Fokker-Planck MLP surrogate makes its largest errors.

Input
-----
test_predictions.npz produced by train_mlp.py. Expected arrays:

    physical_inputs : [N, 4]
        columns = [D0, nu_collision, sigma0, t]

    trajectory_id   : [N]
    time_index      : [N]
    relative_l1     : [N]
    relative_l2     : [N]

    f_pca_true      : [N, Nv]
    f_mlp           : [N, Nv]

The script performs both snapshot-level and trajectory-level analysis.

Main questions
--------------
1. Does the error increase near the edges of parameter space?
2. Is the error associated with large/small D0, nu_collision, or sigma0?
3. Is the error concentrated at early or late times?
4. Are a few trajectories responsible for most of the large errors?
5. What do the worst reconstructed distributions actually look like?

Outputs
-------
- scatter plots of relative L1 error versus each input parameter
- binned median / 95th-percentile error versus each input parameter
- 2D median-error heatmaps in parameter space
- error-versus-time statistics
- parameter-boundary-distance analysis
- worst snapshot table and plots
- trajectory-level summary and worst-trajectory plots
- Spearman-rank correlation table
- JSON summary
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


PARAMETER_NAMES = ("D0", "nu_collision", "sigma0", "t")


def safe_positive(values, floor=1e-16):
    return np.maximum(np.asarray(values, dtype=np.float64), floor)


def percentile_summary(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "median": float(np.median(x)),
        "p68": float(np.percentile(x, 68.0)),
        "p90": float(np.percentile(x, 90.0)),
        "p95": float(np.percentile(x, 95.0)),
        "p99": float(np.percentile(x, 99.0)),
        "max": float(np.max(x)),
    }


def load_predictions(path):
    with np.load(path) as d:
        required = [
            "physical_inputs",
            "trajectory_id",
            "time_index",
            "relative_l1",
            "relative_l2",
            "f_pca_true",
            "f_mlp",
        ]
        missing = [name for name in required if name not in d]
        if missing:
            raise KeyError(
                "Missing arrays in test_predictions.npz: "
                + ", ".join(missing)
            )

        return {
            name: np.asarray(d[name])
            for name in required
        }


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scatter_error_vs_parameter(x, error, name, output):
    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)

    ax.scatter(x, error, s=10, alpha=0.35)

    if name in ("D0", "nu_collision"):
        ax.set_xscale("log")

    ax.set_yscale("log")
    ax.set_xlabel(name)
    ax.set_ylabel("Relative L1 error")
    ax.set_title(f"MLP error versus {name}")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_1d_bins(x, n_bins, log_x=False):
    x = np.asarray(x, dtype=np.float64)

    if log_x:
        positive = x[x > 0]
        if len(positive) == 0:
            raise ValueError("Logarithmic binning requires positive values.")
        xmin = np.min(positive)
        xmax = np.max(positive)
        if np.isclose(xmin, xmax):
            # Keep bins strictly monotonic even for a tiny/degenerate test split.
            xmin = xmin / 1.001
            xmax = xmax * 1.001
        edges = np.geomspace(xmin, xmax, n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        xmin = np.min(x)
        xmax = np.max(x)
        if np.isclose(xmin, xmax):
            span = max(abs(float(xmin)) * 1e-6, 1e-6)
            xmin = xmin - span
            xmax = xmax + span
        edges = np.linspace(xmin, xmax, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

    return edges, centers


def binned_error_stats(x, error, n_bins, log_x=False):
    edges, centers = make_1d_bins(x, n_bins, log_x)

    rows = []
    median = np.full(n_bins, np.nan)
    p95 = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)

    bin_id = np.digitize(x, edges) - 1
    bin_id[x == edges[-1]] = n_bins - 1

    for i in range(n_bins):
        mask = bin_id == i
        count[i] = int(np.sum(mask))

        if count[i] == 0:
            continue

        values = error[mask]
        median[i] = np.median(values)
        p95[i] = np.percentile(values, 95.0)

        rows.append(
            {
                "bin": i,
                "left": float(edges[i]),
                "right": float(edges[i + 1]),
                "center": float(centers[i]),
                "count": int(count[i]),
                "median_relative_l1": float(median[i]),
                "p95_relative_l1": float(p95[i]),
            }
        )

    return centers, median, p95, count, rows


def plot_binned_error(
    centers,
    median,
    p95,
    name,
    output,
    log_x=False,
):
    mask = np.isfinite(median)

    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)

    ax.plot(
        centers[mask],
        median[mask],
        marker="o",
        label="median",
    )
    ax.plot(
        centers[mask],
        p95[mask],
        marker="o",
        label="95th percentile",
    )

    if log_x:
        ax.set_xscale("log")

    ax.set_yscale("log")
    ax.set_xlabel(name)
    ax.set_ylabel("Relative L1 error")
    ax.set_title(f"Binned MLP error versus {name}")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def heatmap_median_error(
    x,
    y,
    error,
    x_name,
    y_name,
    output,
    n_bins=12,
    x_log=False,
    y_log=False,
):
    x_edges, _ = make_1d_bins(x, n_bins, x_log)
    y_edges, _ = make_1d_bins(y, n_bins, y_log)

    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1

    x_bin[x == x_edges[-1]] = n_bins - 1
    y_bin[y == y_edges[-1]] = n_bins - 1

    grid = np.full((n_bins, n_bins), np.nan)
    counts = np.zeros((n_bins, n_bins), dtype=int)

    for iy in range(n_bins):
        for ix in range(n_bins):
            mask = (x_bin == ix) & (y_bin == iy)
            counts[iy, ix] = int(np.sum(mask))

            if counts[iy, ix] > 0:
                grid[iy, ix] = np.median(error[mask])

    finite = grid[np.isfinite(grid) & (grid > 0)]
    if len(finite) == 0:
        return

    fig = plt.figure(figsize=(7.5, 5.8))
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        grid,
        shading="auto",
        norm=LogNorm(vmin=np.min(finite), vmax=np.max(finite)),
    )

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"Median relative L1 error: {x_name} vs {y_name}")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Median relative L1 error")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_boundary_distance(D0, nu, sigma0):
    """Normalized distance to the nearest boundary of 3D parameter space.

    D0 and nu are treated in log10 space because they were sampled there.
    sigma0 is treated linearly.

    distance = 0   -> exactly at a sampled boundary
    distance = 0.5 -> center in all normalized dimensions
    """

    coordinates = []

    for values, use_log in (
        (D0, True),
        (nu, True),
        (sigma0, False),
    ):
        values = np.asarray(values, dtype=np.float64)

        if use_log:
            values = np.log10(values)

        low = np.min(values)
        high = np.max(values)

        if high <= low:
            normalized = np.full_like(values, 0.5)
        else:
            normalized = (values - low) / (high - low)

        coordinates.append(normalized)

    coords = np.column_stack(coordinates)

    return np.min(
        np.minimum(coords, 1.0 - coords),
        axis=1,
    )


def plot_boundary_effect(distance, error, output):
    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)

    ax.scatter(distance, error, s=10, alpha=0.35)
    ax.set_yscale("log")
    ax.set_xlabel("Normalized distance to nearest parameter-space boundary")
    ax.set_ylabel("Relative L1 error")
    ax.set_title("Does the surrogate degrade near parameter-space boundaries?")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def aggregate_by_time(time_index, physical_time, error):
    unique_time_indices = np.unique(time_index)

    rows = []

    for idx in unique_time_indices:
        mask = time_index == idx
        values = error[mask]

        rows.append(
            {
                "time_index": int(idx),
                "t": float(np.median(physical_time[mask])),
                "count": int(np.sum(mask)),
                "median_relative_l1": float(np.median(values)),
                "p95_relative_l1": float(np.percentile(values, 95.0)),
                "max_relative_l1": float(np.max(values)),
            }
        )

    return rows


def plot_error_vs_time_aggregated(rows, output):
    t = np.array([r["t"] for r in rows])
    median = np.array([r["median_relative_l1"] for r in rows])
    p95 = np.array([r["p95_relative_l1"] for r in rows])

    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)

    ax.plot(t, median, marker="o", label="median")
    ax.plot(t, p95, marker="o", label="95th percentile")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Relative L1 error")
    ax.set_title("Test error versus time")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def trajectory_level_summary(
    trajectory_id,
    physical_inputs,
    error,
):
    rows = []

    for traj in np.unique(trajectory_id):
        mask = trajectory_id == traj
        indices = np.where(mask)[0]

        local_error = error[mask]
        physical = physical_inputs[indices]

        worst_local = int(np.argmax(local_error))
        worst_global_idx = int(indices[worst_local])

        rows.append(
            {
                "trajectory_id": int(traj),
                "D0": float(np.median(physical[:, 0])),
                "nu_collision": float(np.median(physical[:, 1])),
                "sigma0": float(np.median(physical[:, 2])),
                "median_relative_l1": float(np.median(local_error)),
                "p95_relative_l1": float(
                    np.percentile(local_error, 95.0)
                ),
                "max_relative_l1": float(np.max(local_error)),
                "time_of_max_error": float(
                    physical_inputs[worst_global_idx, 3]
                ),
                "snapshot_index_of_max_error": worst_global_idx,
            }
        )

    return rows


def plot_worst_trajectory_errors_over_time(
    trajectory_rows,
    trajectory_id,
    physical_inputs,
    error,
    output,
    n_worst,
):
    ordered = sorted(
        trajectory_rows,
        key=lambda r: r["p95_relative_l1"],
        reverse=True,
    )[:n_worst]

    fig = plt.figure(figsize=(8.0, 5.5))
    ax = fig.add_subplot(111)

    for row in ordered:
        traj = row["trajectory_id"]
        mask = trajectory_id == traj

        order = np.argsort(physical_inputs[mask, 3])

        ax.plot(
            physical_inputs[mask, 3][order],
            error[mask][order],
            marker="o",
            markersize=3,
            label=f"traj {traj}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Relative L1 error")
    ax.set_title(
        f"Error evolution for {len(ordered)} worst test trajectories"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_reconstruction(
    f_true,
    f_pred,
    physical,
    error,
    output,
    rank,
    snapshot_idx,
    trajectory_id,
):
    v_index = np.arange(len(f_true))

    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)

    ax.plot(v_index, f_true, label="PCA reference")
    ax.plot(v_index, f_pred, linestyle="--", label="MLP + PCA")

    D0, nu, sigma0, t = physical

    ax.set_xlabel("Velocity-bin index")
    ax.set_ylabel("f(v,t)")
    ax.set_title(
        f"Worst snapshot rank {rank}: traj={trajectory_id}, "
        f"snapshot={snapshot_idx}\n"
        f"D0={D0:.4g}, nu={nu:.4g}, sigma0={sigma0:.4g}, "
        f"t={t:.4g}, relative L1={error:.3e}"
    )
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def spearman_table(physical_inputs, error):
    if spearmanr is None:
        return []

    predictors = {
        "log10_D0": np.log10(physical_inputs[:, 0]),
        "log10_nu_collision": np.log10(physical_inputs[:, 1]),
        "sigma0": physical_inputs[:, 2],
        "t": physical_inputs[:, 3],
    }

    y = np.log10(safe_positive(error))

    rows = []

    for name, x in predictors.items():
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(x, y)

        rows.append(
            {
                "predictor": name,
                "spearman_rho_with_log10_error": float(rho),
                "p_value": float(p_value),
            }
        )

    return rows


def main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    worst_dir = output_dir / "worst_snapshots"
    worst_dir.mkdir(parents=True, exist_ok=True)

    data = load_predictions(input_path)

    physical_inputs = np.asarray(
        data["physical_inputs"],
        dtype=np.float64,
    )
    trajectory_id = np.asarray(
        data["trajectory_id"],
        dtype=np.int64,
    )
    time_index = np.asarray(
        data["time_index"],
        dtype=np.int64,
    )

    error_l1 = safe_positive(data["relative_l1"])
    error_l2 = safe_positive(data["relative_l2"])

    f_true = np.asarray(data["f_pca_true"], dtype=np.float64)
    f_pred = np.asarray(data["f_mlp"], dtype=np.float64)

    D0 = physical_inputs[:, 0]
    nu = physical_inputs[:, 1]
    sigma0 = physical_inputs[:, 2]
    t = physical_inputs[:, 3]

    n_samples = len(error_l1)

    print("=== Test prediction analysis ===")
    print(f"Input              : {input_path}")
    print(f"Snapshots          : {n_samples}")
    print(f"Trajectories       : {len(np.unique(trajectory_id))}")
    print()
    print("Relative L1:")
    for key, value in percentile_summary(error_l1).items():
        print(f"  {key:6s}: {value:.6e}")

    # --------------------------------------------------------------
    # Snapshot-level scatter + binned statistics
    # --------------------------------------------------------------
    parameter_data = {
        "D0": (D0, True),
        "nu_collision": (nu, True),
        "sigma0": (sigma0, False),
        "t": (t, False),
    }

    for name, (values, log_x) in parameter_data.items():
        scatter_error_vs_parameter(
            values,
            error_l1,
            name,
            output_dir / f"error_vs_{name}.png",
        )

        centers, median, p95, count, rows = binned_error_stats(
            values,
            error_l1,
            args.n_bins_1d,
            log_x=log_x,
        )

        write_csv(
            output_dir / f"binned_error_vs_{name}.csv",
            [
                "bin",
                "left",
                "right",
                "center",
                "count",
                "median_relative_l1",
                "p95_relative_l1",
            ],
            rows,
        )

        plot_binned_error(
            centers,
            median,
            p95,
            name,
            output_dir / f"binned_error_vs_{name}.png",
            log_x=log_x,
        )

    # --------------------------------------------------------------
    # Time aggregation
    # --------------------------------------------------------------
    time_rows = aggregate_by_time(
        time_index,
        t,
        error_l1,
    )

    write_csv(
        output_dir / "error_by_time.csv",
        [
            "time_index",
            "t",
            "count",
            "median_relative_l1",
            "p95_relative_l1",
            "max_relative_l1",
        ],
        time_rows,
    )

    plot_error_vs_time_aggregated(
        time_rows,
        output_dir / "error_by_time.png",
    )

    # --------------------------------------------------------------
    # 2D parameter-space maps
    # --------------------------------------------------------------
    heatmap_median_error(
        D0,
        nu,
        error_l1,
        "D0",
        "nu_collision",
        output_dir / "heatmap_D0_nu.png",
        n_bins=args.n_bins_2d,
        x_log=True,
        y_log=True,
    )

    heatmap_median_error(
        D0,
        sigma0,
        error_l1,
        "D0",
        "sigma0",
        output_dir / "heatmap_D0_sigma0.png",
        n_bins=args.n_bins_2d,
        x_log=True,
        y_log=False,
    )

    heatmap_median_error(
        nu,
        sigma0,
        error_l1,
        "nu_collision",
        "sigma0",
        output_dir / "heatmap_nu_sigma0.png",
        n_bins=args.n_bins_2d,
        x_log=True,
        y_log=False,
    )

    heatmap_median_error(
        t,
        D0,
        error_l1,
        "t",
        "D0",
        output_dir / "heatmap_time_D0.png",
        n_bins=args.n_bins_2d,
        x_log=False,
        y_log=True,
    )

    # --------------------------------------------------------------
    # Boundary-distance diagnostic
    # --------------------------------------------------------------
    boundary_distance = compute_boundary_distance(
        D0,
        nu,
        sigma0,
    )

    plot_boundary_effect(
        boundary_distance,
        error_l1,
        output_dir / "error_vs_boundary_distance.png",
    )

    _, _, _, _, boundary_rows = binned_error_stats(
        boundary_distance,
        error_l1,
        args.n_bins_1d,
        log_x=False,
    )

    write_csv(
        output_dir / "binned_error_vs_boundary_distance.csv",
        [
            "bin",
            "left",
            "right",
            "center",
            "count",
            "median_relative_l1",
            "p95_relative_l1",
        ],
        boundary_rows,
    )

    # --------------------------------------------------------------
    # Correlation table
    # --------------------------------------------------------------
    corr_rows = spearman_table(
        physical_inputs,
        error_l1,
    )

    if corr_rows:
        write_csv(
            output_dir / "spearman_correlations.csv",
            [
                "predictor",
                "spearman_rho_with_log10_error",
                "p_value",
            ],
            corr_rows,
        )

    # --------------------------------------------------------------
    # Trajectory-level aggregation
    # --------------------------------------------------------------
    trajectory_rows = trajectory_level_summary(
        trajectory_id,
        physical_inputs,
        error_l1,
    )

    write_csv(
        output_dir / "trajectory_summary.csv",
        [
            "trajectory_id",
            "D0",
            "nu_collision",
            "sigma0",
            "median_relative_l1",
            "p95_relative_l1",
            "max_relative_l1",
            "time_of_max_error",
            "snapshot_index_of_max_error",
        ],
        trajectory_rows,
    )

    plot_worst_trajectory_errors_over_time(
        trajectory_rows,
        trajectory_id,
        physical_inputs,
        error_l1,
        output_dir / "worst_trajectories_error_vs_time.png",
        args.n_worst_trajectories,
    )

    # --------------------------------------------------------------
    # Worst individual snapshots
    # --------------------------------------------------------------
    worst_order = np.argsort(error_l1)[::-1]
    n_worst = min(args.n_worst_snapshots, n_samples)
    worst_rows = []

    for rank, idx in enumerate(worst_order[:n_worst], start=1):
        D0_i, nu_i, sigma_i, t_i = physical_inputs[idx]

        worst_rows.append(
            {
                "rank": rank,
                "global_snapshot_index": int(idx),
                "trajectory_id": int(trajectory_id[idx]),
                "time_index": int(time_index[idx]),
                "D0": float(D0_i),
                "nu_collision": float(nu_i),
                "sigma0": float(sigma_i),
                "t": float(t_i),
                "relative_l1": float(error_l1[idx]),
                "relative_l2": float(error_l2[idx]),
                "min_predicted_f": float(np.min(f_pred[idx])),
            }
        )

        if rank <= args.n_worst_plots:
            plot_single_reconstruction(
                f_true[idx],
                f_pred[idx],
                physical_inputs[idx],
                error_l1[idx],
                worst_dir / f"worst_snapshot_{rank:02d}.png",
                rank,
                int(time_index[idx]),
                int(trajectory_id[idx]),
            )

    write_csv(
        output_dir / "worst_snapshots.csv",
        [
            "rank",
            "global_snapshot_index",
            "trajectory_id",
            "time_index",
            "D0",
            "nu_collision",
            "sigma0",
            "t",
            "relative_l1",
            "relative_l2",
            "min_predicted_f",
        ],
        worst_rows,
    )

    # --------------------------------------------------------------
    # Compact machine-readable summary
    # --------------------------------------------------------------
    worst_trajectory_rows = sorted(
        trajectory_rows,
        key=lambda r: r["p95_relative_l1"],
        reverse=True,
    )

    summary = {
        "input": str(input_path),
        "n_test_snapshots": int(n_samples),
        "n_test_trajectories": int(len(np.unique(trajectory_id))),
        "relative_l1": percentile_summary(error_l1),
        "relative_l2": percentile_summary(error_l2),
        "parameter_ranges": {
            "D0": [float(np.min(D0)), float(np.max(D0))],
            "nu_collision": [float(np.min(nu)), float(np.max(nu))],
            "sigma0": [float(np.min(sigma0)), float(np.max(sigma0))],
            "t": [float(np.min(t)), float(np.max(t))],
        },
        "worst_snapshot": worst_rows[0] if worst_rows else None,
        "worst_trajectory_by_p95": (
            worst_trajectory_rows[0]
            if worst_trajectory_rows
            else None
        ),
        "spearman_correlations": corr_rows,
    }

    (output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nSaved analysis to:")
    print(f"  {output_dir}")
    print("\nMost useful files to inspect first:")
    print(f"  {output_dir / 'error_by_time.png'}")
    print(f"  {output_dir / 'heatmap_D0_nu.png'}")
    print(f"  {output_dir / 'error_vs_boundary_distance.png'}")
    print(f"  {output_dir / 'trajectory_summary.csv'}")
    print(f"  {output_dir / 'worst_snapshots.csv'}")
    print(f"  {output_dir / 'worst_trajectories_error_vs_time.png'}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--input",
        default="artifacts/residual/test_predictions.npz",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/residual/error_analysis",
    )

    p.add_argument(
        "--n-bins-1d",
        type=int,
        default=12,
    )
    p.add_argument(
        "--n-bins-2d",
        type=int,
        default=10,
    )

    p.add_argument(
        "--n-worst-snapshots",
        type=int,
        default=20,
    )
    p.add_argument(
        "--n-worst-plots",
        type=int,
        default=10,
    )
    p.add_argument(
        "--n-worst-trajectories",
        type=int,
        default=8,
    )

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
