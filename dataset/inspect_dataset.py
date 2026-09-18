#!/usr/bin/env python3
"""Randomly inspect and evaluate trajectories in an FP HDF5 dataset.

Outputs
-------
1. A multi-panel PNG containing randomly selected trajectories.
2. A CSV table containing the sampled parameters and diagnostics.
3. A concise terminal summary that flags nearly unchanged cases and cases
   with substantial probability mass near the velocity-domain boundaries.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_dataset(path: Path):
    with h5py.File(path, "r") as h5:
        v = h5["velocity"][:]
        times = h5["times"][:]
        parameters = h5["parameters"][:]
        names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in h5["parameter_names"][:]
        ]
        diagnostics = {
            name: h5[f"diagnostics/{name}"][:]
            for name in h5["diagnostics"].keys()
        }
    return v, times, parameters, names, diagnostics


def choose_time_indices(nt: int, n_curves: int) -> np.ndarray:
    n_curves = max(2, min(n_curves, nt))
    return np.unique(np.linspace(0, nt - 1, n_curves, dtype=int))


def inspect(args: argparse.Namespace):
    dataset_path = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    v, times, parameters, names, diagnostics = read_dataset(dataset_path)
    n_total = len(parameters)
    n_samples = min(args.n_samples, n_total)

    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(n_total, size=n_samples, replace=False))
    time_idx = choose_time_indices(len(times), args.n_time_curves)

    # Read only the selected solution trajectories.
    with h5py.File(dataset_path, "r") as h5:
        selected = np.stack([h5["solutions"][int(i)] for i in indices], axis=0)

    ncols = args.ncols
    nrows = math.ceil(n_samples / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.1 * nrows),
        squeeze=False,
        sharex=True,
    )

    for panel, (idx, ftraj) in enumerate(zip(indices, selected)):
        ax = axes.flat[panel]
        for j in time_idx:
            ax.plot(v, ftraj[j], label=f"t={times[j]:.2f}")

        d0, nu, sigma0 = parameters[idx]
        edge = diagnostics["edge_mass_final"][idx]
        l1 = diagnostics["l1_change_final"][idx]
        mass_err = diagnostics["mass_error_max"][idx]

        ax.set_title(
            f"#{idx}  D0={d0:.2e}, nu={nu:.2e}, sigma={sigma0:.2f}\n"
            f"L1={l1:.3f}, edge={edge:.1e}, dM={mass_err:.1e}",
            fontsize=9,
        )
        ax.grid(alpha=0.20)

        if panel % ncols == 0:
            ax.set_ylabel("f(v,t)")
        if panel >= (nrows - 1) * ncols:
            ax.set_xlabel("v")

    for panel in range(n_samples, nrows * ncols):
        axes.flat[panel].axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        f"Random FP trajectories from {dataset_path.name}",
        y=0.995,
        fontsize=14,
    )
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.972),
        ncol=len(labels), fontsize=8
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    csv_path = output.with_suffix(".csv")
    diag_names = sorted(diagnostics.keys())
    with csv_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["trajectory_index", *names, *diag_names])
        for idx in indices:
            writer.writerow(
                [int(idx), *parameters[idx].tolist(), *[diagnostics[k][idx] for k in diag_names]]
            )

    # Dataset-wide evaluation summary.
    edge = diagnostics["edge_mass_final"]
    change = diagnostics["l1_change_final"]
    mass_err = diagnostics["mass_error_max"]
    min_f = diagnostics["min_f"]

    weak = change < args.weak_change_threshold
    boundary = edge > args.edge_mass_threshold

    print("=== Dataset diagnostics ===")
    print(f"Dataset                 : {dataset_path}")
    print(f"Total trajectories      : {n_total}")
    print(f"Randomly plotted        : {n_samples}")
    print(f"D0 range                : {parameters[:,0].min():.3e} -- {parameters[:,0].max():.3e}")
    print(f"nu_collision range      : {parameters[:,1].min():.3e} -- {parameters[:,1].max():.3e}")
    print(f"sigma0 range            : {parameters[:,2].min():.3f} -- {parameters[:,2].max():.3f}")
    print(f"Max mass error          : {mass_err.max():.3e}")
    print(f"Global minimum f        : {min_f.min():.3e}")
    print(f"Median final L1 change  : {np.median(change):.3f}")
    print(f"Max final edge mass     : {edge.max():.3e}")
    print(
        f"Nearly unchanged cases  : {weak.sum()} / {n_total} "
        f"(L1 < {args.weak_change_threshold:g})"
    )
    print(
        f"Boundary-warning cases  : {boundary.sum()} / {n_total} "
        f"(edge mass > {args.edge_mass_threshold:g})"
    )
    print(f"\nSaved figure            : {output}")
    print(f"Saved selected table    : {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="artifacts/fp_dataset_v1.1.h5")
    p.add_argument("--output", default="artifacts/random_candidate_review.png")
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--n-time-curves", type=int, default=5)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dpi", type=int, default=180)

    # These are review heuristics, not physical cuts.
    p.add_argument("--weak-change-threshold", type=float, default=0.02)
    p.add_argument("--edge-mass-threshold", type=float, default=1e-3)
    return p


if __name__ == "__main__":
    inspect(build_parser().parse_args())
