#!/usr/bin/env python3
"""Generate a parameterized 1D Fokker-Planck solution library.

The physical setup follows the current solver.fp_solver implementation:

    df/dt = d/dv [ D(v) df/dv ] + nu_coll (f_eq - f)

with

    A(v) = 0,
    D(v) = D0 * (1 + 0.1 v^2),
    f(v,0) = Maxwellian(sigma0),
    f_eq(v) = Maxwellian(sigma0).

The sampled trajectory parameters are

    D0          : log-uniform via Sobol sampling
    nu_collision: log-uniform via Sobol sampling
    sigma0      : linear-uniform via Sobol sampling

Solutions are stored in one HDF5 file with shape
    solutions[n_trajectory, n_time, n_velocity].
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.stats import qmc

from solver.fp_solver import maxwellian, solve_fp, velocity_grid


PARAMETER_NAMES = ("D0", "nu_collision", "sigma0")


def sample_parameters(
    n: int,
    *,
    seed: int,
    d0_min: float,
    d0_max: float,
    nu_min: float,
    nu_max: float,
    sigma_min: float,
    sigma_max: float,
) -> np.ndarray:
    """Sobol sample of [D0, nu_collision, sigma0]."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 < d0_min < d0_max):
        raise ValueError("Require 0 < d0_min < d0_max")
    if not (0 < nu_min < nu_max):
        raise ValueError("Require 0 < nu_min < nu_max")
    if not (0 < sigma_min < sigma_max):
        raise ValueError("Require 0 < sigma_min < sigma_max")

    engine = qmc.Sobol(d=3, scramble=True, seed=seed)
    unit = engine.random(n)

    log_d0 = np.log10(d0_min) + unit[:, 0] * (
        np.log10(d0_max) - np.log10(d0_min)
    )
    log_nu = np.log10(nu_min) + unit[:, 1] * (
        np.log10(nu_max) - np.log10(nu_min)
    )
    sigma0 = sigma_min + unit[:, 2] * (sigma_max - sigma_min)

    return np.column_stack((10.0**log_d0, 10.0**log_nu, sigma0))


def trajectory_diagnostics(sol, edge_fraction: float = 0.10, vcut: float = 4.0):
    """Return useful QA diagnostics for one trajectory."""
    mass = sol.mass()
    mass0 = mass[0]
    mass_error_max = np.max(np.abs(mass - mass0)) / mass0

    dv = sol.dv
    vmax_abs = np.max(np.abs(sol.v))
    edge_mask = np.abs(sol.v) >= (1.0 - edge_fraction) * vmax_abs
    edge_mass_final = np.sum(sol.f[-1, edge_mask]) * dv / mass[-1]

    l1_change_final = np.sum(np.abs(sol.f[-1] - sol.f[0])) * dv / mass0

    return {
        "mass_error_max": float(mass_error_max),
        "min_f": float(np.min(sol.f)),
        "final_variance": float(sol.variance()[-1]),
        "final_tail_fraction": float(sol.tail_fraction(vcut)[-1]),
        "edge_mass_final": float(edge_mass_final),
        "l1_change_final": float(l1_change_final),
    }


def generate_dataset(args: argparse.Namespace) -> Path:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    params = sample_parameters(
        args.n_trajectories,
        seed=args.seed,
        d0_min=args.d0_min,
        d0_max=args.d0_max,
        nu_min=args.nu_min,
        nu_max=args.nu_max,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )

    v, _ = velocity_grid(args.vmin, args.vmax, args.nv)
    times = np.linspace(0.0, args.tmax, args.nt)

    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output} already exists. Use --overwrite to replace it."
        )

    t0 = time.time()

    with h5py.File(output, "w") as h5:
        # Coordinates and sampled parameters
        h5.create_dataset("velocity", data=v.astype(np.float64))
        h5.create_dataset("times", data=times.astype(np.float64))
        h5.create_dataset("parameters", data=params.astype(np.float64))

        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset(
            "parameter_names",
            data=np.asarray(PARAMETER_NAMES, dtype=object),
            dtype=str_dtype,
        )

        # Main solution cube. Chunking by trajectory makes random trajectory
        # inspection and mini-batch reads efficient.
        solutions = h5.create_dataset(
            "solutions",
            shape=(args.n_trajectories, args.nt, args.nv),
            dtype=np.float32,
            chunks=(1, args.nt, args.nv),
            compression="gzip",
            compression_opts=args.compression,
            shuffle=True,
        )

        # Per-trajectory quality-control quantities.
        diag_group = h5.create_group("diagnostics")
        diagnostics = {
            name: diag_group.create_dataset(name, shape=(args.n_trajectories,), dtype=np.float64)
            for name in (
                "mass_error_max",
                "min_f",
                "final_variance",
                "final_tail_fraction",
                "edge_mass_final",
                "l1_change_final",
            )
        }

        # Metadata needed to reproduce the library.
        h5.attrs["solver"] = "solver.fp_solver"
        h5.attrs["sampling"] = "scrambled Sobol"
        h5.attrs["seed"] = args.seed
        h5.attrs["equation"] = (
            "df/dt = -d(Af)/dv + d/dv[D(v) df/dv] + nu_collision*(f_eq-f)"
        )
        h5.attrs["drift"] = "A(v)=0"
        h5.attrs["diffusion"] = "D(v)=D0*(1+0.1*v**2)"
        h5.attrs["initial_distribution"] = "Maxwellian(sigma=sigma0)"
        h5.attrs["equilibrium_distribution"] = "Maxwellian(sigma=sigma0)"
        h5.attrs["dt"] = args.dt
        h5.attrs["vmin"] = args.vmin
        h5.attrs["vmax"] = args.vmax
        h5.attrs["nv"] = args.nv
        h5.attrs["tmax"] = args.tmax
        h5.attrs["nt"] = args.nt
        h5.attrs["D0_range"] = np.asarray([args.d0_min, args.d0_max])
        h5.attrs["nu_collision_range"] = np.asarray([args.nu_min, args.nu_max])
        h5.attrs["sigma0_range"] = np.asarray([args.sigma_min, args.sigma_max])
        h5.attrs["tail_vcut"] = args.vcut
        h5.attrs["edge_fraction"] = args.edge_fraction

        for i, (d0, nu_collision, sigma0) in enumerate(params):
            f0 = maxwellian(v, sigma=sigma0)
            f_eq = maxwellian(v, sigma=sigma0)

            def diffusion(x, d0=d0):
                return d0 * (1.0 + 0.1 * x**2)

            sol = solve_fp(
                v,
                f0,
                times,
                drift=0.0,
                diffusion=diffusion,
                nu_collision=float(nu_collision),
                f_equilibrium=f_eq,
                dt=args.dt,
                theta=1.0,
            )

            solutions[i] = sol.f.astype(np.float32)
            diag = trajectory_diagnostics(
                sol,
                edge_fraction=args.edge_fraction,
                vcut=args.vcut,
            )
            for name, value in diag.items():
                diagnostics[name][i] = value

            if (
                i == 0
                or (i + 1) % args.progress_every == 0
                or (i + 1) == args.n_trajectories
            ):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else np.nan
                print(
                    f"[{i + 1:5d}/{args.n_trajectories}] "
                    f"D0={d0:.3e}, nu={nu_collision:.3e}, sigma0={sigma0:.3f} "
                    f"| {rate:.2f} trajectories/s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\nSaved dataset: {output}")
    print(f"Elapsed time : {elapsed:.2f} s")
    print(f"Shape        : ({args.n_trajectories}, {args.nt}, {args.nv})")
    return output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--output", default="artifacts/fp_dataset_v1.1.h5")
    p.add_argument("--n-trajectories", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    # Current fiducial model and the first proposed spread.
    p.add_argument("--d0-min", type=float, default=1e-2)
    p.add_argument("--d0-max", type=float, default=1.0)
    p.add_argument("--nu-min", type=float, default=1e-2)
    p.add_argument("--nu-max", type=float, default=1.0)
    p.add_argument("--sigma-min", type=float, default=0.5)
    p.add_argument("--sigma-max", type=float, default=2.0)

    # Numerical grid: fixed across the entire dataset.
    p.add_argument("--vmin", type=float, default=-8.0)
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--nv", type=int, default=400)
    p.add_argument("--tmax", type=float, default=5.0)
    p.add_argument("--nt", type=int, default=51)
    p.add_argument("--dt", type=float, default=0.01)

    p.add_argument("--vcut", type=float, default=4.0)
    p.add_argument("--edge-fraction", type=float, default=0.10)
    p.add_argument("--compression", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--overwrite", action="store_true")

    return p


if __name__ == "__main__":
    generate_dataset(build_parser().parse_args())
