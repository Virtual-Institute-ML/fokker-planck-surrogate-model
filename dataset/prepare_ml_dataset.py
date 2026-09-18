#!/usr/bin/env python3
"""Build the ML dataset for the Fokker-Planck PCA surrogate.

Input:
    1) Fokker-Planck trajectory HDF5 produced by generate_dataset.py
    2) PCA model produced by pca_analysis.py
    3) trajectory_split.npz produced by pca_analysis.py

For every saved FP snapshot, construct

    physical input:
        [D0, nu_collision, sigma0, t]

    model input:
        [log10(D0), log10(nu_collision), sigma0, t]

    target:
        first N_PCA PCA coefficients

The input features and PCA targets are standardized using TRAINING
samples only. Validation and test samples never contribute to the
scaling statistics.

Default:
    N_PCA = 16

Output HDF5 structure:
    velocity
    times

    pca/
        mean
        components
        explained_variance_ratio

    scalers/
        input_mean
        input_std
        target_mean
        target_std

    train/
        physical_inputs
        inputs
        inputs_scaled
        targets_pca
        targets_scaled
        trajectory_id
        time_index

    validation/
        ...

    test/
        ...

The split remains trajectory-based, preventing snapshots from one
trajectory from appearing in multiple data splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np


PHYSICAL_INPUT_NAMES = (
    "D0",
    "nu_collision",
    "sigma0",
    "t",
)

MODEL_INPUT_NAMES = (
    "log10_D0",
    "log10_nu_collision",
    "sigma0",
    "t",
)


def decode_names(values):
    return [
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in values
    ]


def forward_solution_transform(x, mode, epsilon):
    if mode == "raw":
        return x
    if mode == "log10":
        return np.log10(np.maximum(x, 0.0) + epsilon)
    raise ValueError(f"Unsupported PCA transform: {mode}")


def inverse_solution_transform(x, mode, epsilon):
    if mode == "raw":
        return x
    if mode == "log10":
        return np.maximum(10.0**x - epsilon, 0.0)
    raise ValueError(f"Unsupported PCA transform: {mode}")


def relative_l1(true, pred):
    numerator = np.sum(np.abs(pred - true), axis=1)
    denominator = np.sum(np.abs(true), axis=1)
    denominator = np.maximum(denominator, 1e-300)
    return numerator / denominator


def validate_source_files(h5, pca_bundle, split, n_pca):
    velocity = np.asarray(h5["velocity"][:], dtype=np.float64)
    times = np.asarray(h5["times"][:], dtype=np.float64)
    parameter_names = decode_names(h5["parameter_names"][:])

    required = ["D0", "nu_collision", "sigma0"]
    if parameter_names != required:
        raise ValueError(
            "Expected parameter_names "
            f"{required}, but found {parameter_names}"
        )

    pca = pca_bundle["pca"]

    if n_pca > pca.n_components_:
        raise ValueError(
            f"Requested n_pca={n_pca}, but the fitted PCA model has only "
            f"{pca.n_components_} components."
        )

    if pca.mean_.shape[0] != velocity.size:
        raise ValueError(
            "PCA feature dimension does not match the FP velocity grid."
        )

    if "velocity" in pca_bundle:
        pca_velocity = np.asarray(pca_bundle["velocity"])
        if pca_velocity.shape != velocity.shape or not np.allclose(
            pca_velocity, velocity, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Velocity grid differs from the PCA training grid.")

    if "times" in pca_bundle:
        pca_times = np.asarray(pca_bundle["times"])
        if pca_times.shape != times.shape or not np.allclose(
            pca_times, times, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Time grid differs from the PCA training grid.")

    n_traj = h5["solutions"].shape[0]
    all_split_indices = np.concatenate(
        [split["train"], split["validation"], split["test"]]
    )

    if len(np.unique(all_split_indices)) != len(all_split_indices):
        raise ValueError("trajectory_split.npz contains duplicated indices.")

    if np.any(all_split_indices < 0) or np.any(all_split_indices >= n_traj):
        raise ValueError("trajectory_split.npz contains an invalid trajectory index.")


def build_split_arrays(
    h5,
    trajectory_indices,
    pca_bundle,
    n_pca,
):
    """Convert selected trajectories into snapshot-level ML samples."""

    trajectory_indices = np.asarray(trajectory_indices, dtype=np.int64)
    trajectory_indices = np.sort(trajectory_indices)

    times = np.asarray(h5["times"][:], dtype=np.float64)
    parameters = np.asarray(h5["parameters"][trajectory_indices], dtype=np.float64)
    solutions = np.asarray(h5["solutions"][trajectory_indices], dtype=np.float64)

    n_traj = len(trajectory_indices)
    n_time = len(times)
    n_velocity = solutions.shape[-1]

    if np.any(parameters[:, 0] <= 0.0):
        raise ValueError("D0 must be positive before log10 transform.")
    if np.any(parameters[:, 1] <= 0.0):
        raise ValueError("nu_collision must be positive before log10 transform.")

    # [N_traj, N_time, 4] -> [N_samples, 4]
    D0 = np.repeat(parameters[:, 0], n_time)
    nu = np.repeat(parameters[:, 1], n_time)
    sigma0 = np.repeat(parameters[:, 2], n_time)
    t = np.tile(times, n_traj)

    physical_inputs = np.column_stack((D0, nu, sigma0, t))
    inputs = np.column_stack(
        (
            np.log10(D0),
            np.log10(nu),
            sigma0,
            t,
        )
    )

    # Flatten all snapshots and transform them with the PCA representation
    # used during PCA fitting.
    flat_solutions = solutions.reshape(-1, n_velocity)
    mode = pca_bundle.get("transform", "raw")
    epsilon = float(pca_bundle.get("epsilon", 1e-12))

    pca_input = forward_solution_transform(
        flat_solutions,
        mode,
        epsilon,
    )

    pca = pca_bundle["pca"]
    full_scores = pca.transform(pca_input)
    targets_pca = full_scores[:, :n_pca]

    trajectory_id = np.repeat(trajectory_indices, n_time)
    time_index = np.tile(np.arange(n_time, dtype=np.int32), n_traj)

    return {
        "physical_inputs": physical_inputs,
        "inputs": inputs,
        "targets_pca": targets_pca,
        "trajectory_id": trajectory_id,
        "time_index": time_index,
        # Kept only in memory for reconstruction QA, not saved in output.
        "_solutions": flat_solutions,
    }


def fit_standardizer(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    if np.any(std <= 0.0):
        bad = np.where(std <= 0.0)[0]
        raise ValueError(f"Zero-variance dimensions found at indices {bad.tolist()}")

    return mean, std


def standardize(x, mean, std):
    return (x - mean) / std


def reconstruct_from_truncated_pca(targets_pca, pca_bundle, n_pca):
    pca = pca_bundle["pca"]

    transformed = (
        targets_pca @ pca.components_[:n_pca]
        + pca.mean_
    )

    return inverse_solution_transform(
        transformed,
        pca_bundle.get("transform", "raw"),
        float(pca_bundle.get("epsilon", 1e-12)),
    )


def write_string_dataset(group, name, strings):
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(
        name,
        data=np.asarray(strings, dtype=object),
        dtype=dtype,
    )


def save_split(group, data, input_mean, input_std, target_mean, target_std):
    """Save one split, using float32 for compact ML-ready arrays."""

    inputs_scaled = standardize(
        data["inputs"],
        input_mean,
        input_std,
    )
    targets_scaled = standardize(
        data["targets_pca"],
        target_mean,
        target_std,
    )

    n_samples = data["inputs"].shape[0]
    group.attrs["n_samples"] = n_samples
    group.attrs["n_trajectories"] = len(np.unique(data["trajectory_id"]))

    group.create_dataset(
        "physical_inputs",
        data=data["physical_inputs"].astype(np.float32),
        compression="gzip",
        shuffle=True,
    )
    group.create_dataset(
        "inputs",
        data=data["inputs"].astype(np.float32),
        compression="gzip",
        shuffle=True,
    )
    group.create_dataset(
        "inputs_scaled",
        data=inputs_scaled.astype(np.float32),
        compression="gzip",
        shuffle=True,
    )
    group.create_dataset(
        "targets_pca",
        data=data["targets_pca"].astype(np.float32),
        compression="gzip",
        shuffle=True,
    )
    group.create_dataset(
        "targets_scaled",
        data=targets_scaled.astype(np.float32),
        compression="gzip",
        shuffle=True,
    )
    group.create_dataset(
        "trajectory_id",
        data=data["trajectory_id"].astype(np.int32),
        compression="gzip",
        shuffle=True,
    )
    group.create_dataset(
        "time_index",
        data=data["time_index"].astype(np.int16),
        compression="gzip",
        shuffle=True,
    )


def split_reconstruction_stats(data, pca_bundle, n_pca):
    reconstructed = reconstruct_from_truncated_pca(
        data["targets_pca"],
        pca_bundle,
        n_pca,
    )
    errors = relative_l1(
        data["_solutions"],
        reconstructed,
    )

    return {
        "median_relative_l1": float(np.median(errors)),
        "p95_relative_l1": float(np.percentile(errors, 95.0)),
        "max_relative_l1": float(np.max(errors)),
    }


def main(args):
    fp_dataset = Path(args.fp_dataset)
    pca_model_path = Path(args.pca_model)
    split_path = Path(args.split)
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading PCA model...")
    pca_bundle = joblib.load(pca_model_path)

    with np.load(split_path) as split_file:
        split = {
            "train": np.asarray(split_file["train"], dtype=np.int64),
            "validation": np.asarray(split_file["validation"], dtype=np.int64),
            "test": np.asarray(split_file["test"], dtype=np.int64),
        }

    print("Loading and converting Fokker-Planck trajectories...")
    with h5py.File(fp_dataset, "r") as h5:
        validate_source_files(h5, pca_bundle, split, args.n_pca)

        velocity = np.asarray(h5["velocity"][:], dtype=np.float64)
        times = np.asarray(h5["times"][:], dtype=np.float64)

        data = {}
        for split_name in ("train", "validation", "test"):
            print(
                f"  {split_name:10s}: "
                f"{len(split[split_name])} trajectories"
            )
            data[split_name] = build_split_arrays(
                h5,
                split[split_name],
                pca_bundle,
                args.n_pca,
            )

    # ------------------------------------------------------------------
    # Standardization statistics: TRAIN SET ONLY
    # ------------------------------------------------------------------
    input_mean, input_std = fit_standardizer(
        data["train"]["inputs"]
    )
    target_mean, target_std = fit_standardizer(
        data["train"]["targets_pca"]
    )

    # ------------------------------------------------------------------
    # PCA truncation QA
    # ------------------------------------------------------------------
    reconstruction_stats = {}
    for split_name in ("train", "validation", "test"):
        reconstruction_stats[split_name] = split_reconstruction_stats(
            data[split_name],
            pca_bundle,
            args.n_pca,
        )

    # ------------------------------------------------------------------
    # Save ML-ready HDF5
    # ------------------------------------------------------------------
    pca = pca_bundle["pca"]

    with h5py.File(output_path, "w") as out:
        out.attrs["description"] = (
            "ML-ready Fokker-Planck dataset using PCA targets"
        )
        out.attrs["n_pca"] = args.n_pca
        out.attrs["pca_transform"] = pca_bundle.get("transform", "raw")
        out.attrs["pca_epsilon"] = float(pca_bundle.get("epsilon", 1e-12))
        out.attrs["source_fp_dataset"] = str(fp_dataset)
        out.attrs["source_pca_model"] = str(pca_model_path)
        out.attrs["source_split"] = str(split_path)

        out.create_dataset("velocity", data=velocity)
        out.create_dataset("times", data=times)

        write_string_dataset(
            out,
            "physical_input_names",
            PHYSICAL_INPUT_NAMES,
        )
        write_string_dataset(
            out,
            "model_input_names",
            MODEL_INPUT_NAMES,
        )
        write_string_dataset(
            out,
            "target_names",
            [f"PC{i + 1}" for i in range(args.n_pca)],
        )

        pca_group = out.create_group("pca")
        pca_group.create_dataset(
            "mean",
            data=pca.mean_.astype(np.float64),
        )
        pca_group.create_dataset(
            "components",
            data=pca.components_[:args.n_pca].astype(np.float64),
        )
        pca_group.create_dataset(
            "explained_variance_ratio",
            data=pca.explained_variance_ratio_[:args.n_pca].astype(np.float64),
        )

        scaler_group = out.create_group("scalers")
        scaler_group.create_dataset("input_mean", data=input_mean)
        scaler_group.create_dataset("input_std", data=input_std)
        scaler_group.create_dataset("target_mean", data=target_mean)
        scaler_group.create_dataset("target_std", data=target_std)

        for split_name in ("train", "validation", "test"):
            group = out.create_group(split_name)
            save_split(
                group,
                data[split_name],
                input_mean,
                input_std,
                target_mean,
                target_std,
            )

            stats = reconstruction_stats[split_name]
            for key, value in stats.items():
                group.attrs[f"pca16_{key}"] = value

    # ------------------------------------------------------------------
    # Save a compact human-readable summary
    # ------------------------------------------------------------------
    summary = {
        "n_pca": args.n_pca,
        "physical_input_names": list(PHYSICAL_INPUT_NAMES),
        "model_input_names": list(MODEL_INPUT_NAMES),
        "n_samples": {
            name: int(data[name]["inputs"].shape[0])
            for name in ("train", "validation", "test")
        },
        "n_trajectories": {
            name: int(len(split[name]))
            for name in ("train", "validation", "test")
        },
        "input_mean_train": input_mean.tolist(),
        "input_std_train": input_std.tolist(),
        "target_mean_train": target_mean.tolist(),
        "target_std_train": target_std.tolist(),
        "pca_reconstruction_relative_l1": reconstruction_stats,
        "cumulative_explained_variance_16": float(
            np.sum(pca.explained_variance_ratio_[:args.n_pca])
        ),
    }

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== ML dataset ===")
    print(f"Output               : {output_path}")
    print(f"PCA target dimension : {args.n_pca}")
    print(
        "Cumulative PCA EV    : "
        f"{summary['cumulative_explained_variance_16']:.12f}"
    )

    for name in ("train", "validation", "test"):
        print(
            f"{name:10s}: "
            f"{summary['n_trajectories'][name]:4d} trajectories, "
            f"{summary['n_samples'][name]:6d} samples"
        )

    print("\n=== Train input scaling ===")
    for name, mean, std in zip(
        MODEL_INPUT_NAMES,
        input_mean,
        input_std,
    ):
        print(
            f"{name:20s} mean={mean: .6e}  std={std: .6e}"
        )

    print("\n=== PCA truncation QA ===")
    for name in ("train", "validation", "test"):
        s = reconstruction_stats[name]
        print(
            f"{name:10s} "
            f"median L1={s['median_relative_l1']:.3e}, "
            f"p95={s['p95_relative_l1']:.3e}, "
            f"max={s['max_relative_l1']:.3e}"
        )

    print(f"\nSummary              : {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--fp-dataset",
        default="artifacts/fp_dataset_v1.1.h5",
        help="Trajectory HDF5 from generate_dataset.py",
    )
    parser.add_argument(
        "--pca-model",
        default="artifacts/pca/pca_model.joblib",
        help="PCA model from pca_analysis.py",
    )
    parser.add_argument(
        "--split",
        default="artifacts/pca/trajectory_split.npz",
        help="Trajectory split from pca_analysis.py",
    )
    parser.add_argument(
        "--output",
        default="artifacts/fp_pca16_ml_dataset_v1.1.h5",
    )
    parser.add_argument(
        "--n-pca",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
