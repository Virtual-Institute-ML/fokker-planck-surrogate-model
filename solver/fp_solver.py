from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import splu

ArrayLikeFn = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


@dataclass
class FPSolution:
    v: np.ndarray
    times: np.ndarray
    f: np.ndarray  # shape: (ntimes, nv)

    @property
    def dv(self) -> float:
        return float(self.v[1] - self.v[0])

    def mass(self) -> np.ndarray:
        return np.sum(self.f, axis=1) * self.dv

    def mean(self) -> np.ndarray:
        m = self.mass()
        return np.sum(self.f * self.v[None, :], axis=1) * self.dv / m

    def second_moment(self) -> np.ndarray:
        m = self.mass()
        return np.sum(self.f * self.v[None, :] ** 2, axis=1) * self.dv / m

    def variance(self) -> np.ndarray:
        return self.second_moment() - self.mean() ** 2

    def tail_fraction(self, vcut: float) -> np.ndarray:
        mask = np.abs(self.v) >= vcut
        num = np.sum(self.f[:, mask], axis=1) * self.dv
        return num / self.mass()


def velocity_grid(vmin: float, vmax: float, nv: int) -> tuple[np.ndarray, float]:
    """Uniform finite-volume cell centers on [vmin, vmax]."""
    if nv < 3:
        raise ValueError("nv must be >= 3")
    if vmax <= vmin:
        raise ValueError("vmax must be > vmin")
    edges = np.linspace(vmin, vmax, nv + 1)
    dv = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, float(dv)


def normalize_distribution(f: np.ndarray, dv: float, mass: float = 1.0) -> np.ndarray:
    f = np.asarray(f, dtype=float).copy()
    current = np.sum(f) * dv
    if not np.isfinite(current) or current <= 0:
        raise ValueError("distribution must have positive finite integral")
    return f * (mass / current)


def maxwellian(
    v: np.ndarray,
    sigma: float = 1.0,
    mean: float = 0.0,
    mass: float = 1.0,
    discrete_normalize: bool = True,
) -> np.ndarray:
    """1D Gaussian/Maxwellian on a supplied velocity grid."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    f = np.exp(-0.5 * ((v - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    if discrete_normalize:
        dv = float(v[1] - v[0])
        f = normalize_distribution(f, dv, mass=mass)
    else:
        f = mass * f
    return f


def _evaluate_coefficient(coeff: ArrayLikeFn, x: np.ndarray, name: str) -> np.ndarray:
    if callable(coeff):
        out = np.asarray(coeff(x), dtype=float)
    else:
        arr = np.asarray(coeff, dtype=float)
        if arr.ndim == 0:
            out = np.full_like(x, float(arr))
        elif arr.shape == x.shape:
            out = arr.copy()
        else:
            raise ValueError(f"{name} must be scalar, callable, or have shape {x.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def chang_cooper_delta(w: np.ndarray) -> np.ndarray:
    """
    Chang-Cooper weighting
        delta(w) = 1/w - 1/(exp(w)-1)
    evaluated stably for small/large |w|.
    """
    w = np.asarray(w, dtype=float)
    delta = np.empty_like(w)

    small = np.abs(w) < 1e-5
    ws = w[small]
    delta[small] = 0.5 - ws / 12.0 + ws**3 / 720.0 - ws**5 / 30240.0

    mid = (~small) & (w > -50.0) & (w < 50.0)
    wm = w[mid]
    delta[mid] = 1.0 / wm - 1.0 / np.expm1(wm)

    pos = w >= 50.0
    delta[pos] = 1.0 / w[pos]

    neg = w <= -50.0
    delta[neg] = 1.0 + 1.0 / w[neg]

    return delta


def build_chang_cooper_operator(
    v: np.ndarray,
    drift: ArrayLikeFn = 0.0,
    diffusion: ArrayLikeFn = 1.0,
    diffusion_floor: float = 1e-14,
) -> csc_matrix:
    r"""
    Build the conservative finite-volume operator L for

        df/dt = - dJ/dv,
        J = A(v) f - D(v) df/dv,

    with zero-flux boundary conditions at both ends.

    A and D are evaluated at cell interfaces. D must be non-negative.
    When D -> 0 the interface flux reduces to first-order upwind advection.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or len(v) < 3:
        raise ValueError("v must be a 1D grid with at least 3 cells")
    dv_arr = np.diff(v)
    if not np.allclose(dv_arr, dv_arr[0], rtol=1e-12, atol=1e-14):
        raise ValueError("v grid must be uniform")
    dv = float(dv_arr[0])

    vi = 0.5 * (v[:-1] + v[1:])
    A = _evaluate_coefficient(drift, vi, "drift")
    D = _evaluate_coefficient(diffusion, vi, "diffusion")
    if np.any(D < 0):
        raise ValueError("diffusion coefficient must be non-negative")

    cL = np.zeros_like(vi)
    cR = np.zeros_like(vi)

    diffusive = D > diffusion_floor
    if np.any(diffusive):
        w = A[diffusive] * dv / D[diffusive]
        delta = chang_cooper_delta(w)
        cL[diffusive] = A[diffusive] * (1.0 - delta) + D[diffusive] / dv
        cR[diffusive] = A[diffusive] * delta - D[diffusive] / dv

    nondiff = ~diffusive
    if np.any(nondiff):
        Ap = A[nondiff]
        # Pure-advection limit: upwind flux.
        cL[nondiff] = np.where(Ap >= 0.0, Ap, 0.0)
        cR[nondiff] = np.where(Ap < 0.0, Ap, 0.0)

    n = len(v)
    lower = np.zeros(n - 1)
    diag = np.zeros(n)
    upper = np.zeros(n - 1)

    # Cell 0: df_0/dt = -J_{1/2}/dv because J_left_boundary = 0.
    diag[0] = -cL[0] / dv
    upper[0] = -cR[0] / dv

    # Interior cells.
    if n > 2:
        lower[:-1] = cL[:-1] / dv
        diag[1:-1] = (cR[:-1] - cL[1:]) / dv
        upper[1:] = -cR[1:] / dv

    # Last cell: df_N/dt = +J_{N-1/2}/dv because J_right_boundary = 0.
    lower[-1] = cL[-1] / dv
    diag[-1] = cR[-1] / dv

    return diags([lower, diag, upper], offsets=[-1, 0, 1], format="csc")


def _bgk_exact_step(f: np.ndarray, f_eq: np.ndarray, nu: float, dt: float) -> np.ndarray:
    if nu <= 0.0 or dt <= 0.0:
        return f
    e = np.exp(-nu * dt)
    return f_eq + (f - f_eq) * e


def solve_fp(
    v: np.ndarray,
    f0: np.ndarray,
    t_eval: np.ndarray,
    *,
    drift: ArrayLikeFn = 0.0,
    diffusion: ArrayLikeFn = 1.0,
    nu_collision: float = 0.0,
    f_equilibrium: Optional[np.ndarray] = None,
    dt: float = 1e-2,
    theta: float = 1.0,
    positivity_tol: float = 1e-12,
) -> FPSolution:
    r"""
    Solve the 1D conservative Fokker-Planck equation

        df/dt = -d/dv [A(v) f] + d/dv [D(v) df/dv]
                + nu_collision (f_eq - f)

    on a uniform velocity grid with zero-flux boundaries.

    Spatial discretization: Chang-Cooper finite volume.
    Time integration of FP transport: theta-method (theta=1 backward Euler by default).
    Collision integration: exact BGK relaxation with Strang splitting.

    Notes
    -----
    * theta=1 is the safest choice for positivity/robustness.
    * theta=0.5 gives Crank-Nicolson time centering but can lose monotonicity
      if dt is too large.
    """
    v = np.asarray(v, dtype=float)
    f = np.asarray(f0, dtype=float).copy()
    t_eval = np.asarray(t_eval, dtype=float)

    if f.shape != v.shape:
        raise ValueError("f0 and v must have the same shape")
    if np.any(~np.isfinite(f)) or np.min(f) < -positivity_tol:
        raise ValueError("f0 must be finite and non-negative")
    if len(v) < 3:
        raise ValueError("need at least 3 velocity cells")
    if not np.all(np.diff(t_eval) >= 0) or len(t_eval) == 0 or t_eval[0] < 0:
        raise ValueError("t_eval must be sorted, non-negative, and non-empty")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if not (0.5 <= theta <= 1.0):
        raise ValueError("theta must lie in [0.5, 1]")
    if nu_collision < 0:
        raise ValueError("nu_collision must be non-negative")

    dv = float(v[1] - v[0])
    if not np.allclose(np.diff(v), dv, rtol=1e-12, atol=1e-14):
        raise ValueError("v grid must be uniform")

    # Remove roundoff-scale negative values from input only.
    f[f < 0] = 0.0
    initial_mass = np.sum(f) * dv

    if nu_collision > 0:
        if f_equilibrium is None:
            raise ValueError("f_equilibrium is required when nu_collision > 0")
        f_eq = np.asarray(f_equilibrium, dtype=float).copy()
        if f_eq.shape != v.shape or np.any(f_eq < 0) or np.any(~np.isfinite(f_eq)):
            raise ValueError("f_equilibrium must be finite, non-negative, and match v")
        # BGK collisions should not inject/remove particles: match the initial mass.
        f_eq = normalize_distribution(f_eq, dv, mass=initial_mass)
    else:
        f_eq = np.zeros_like(f)

    L = build_chang_cooper_operator(v, drift=drift, diffusion=diffusion)
    I = eye(len(v), format="csc")

    out = np.empty((len(t_eval), len(v)), dtype=float)
    current_t = 0.0
    cache = {}  # cache LU and RHS operator by rounded step size

    def transport_step(state: np.ndarray, h: float) -> np.ndarray:
        if h <= 0:
            return state
        key = round(float(h), 15)
        if key not in cache:
            lhs = (I - theta * h * L).tocsc()
            lu = splu(lhs)
            rhs_op = (I + (1.0 - theta) * h * L).tocsc()
            cache[key] = (lu, rhs_op)
        lu, rhs_op = cache[key]
        return lu.solve(rhs_op @ state)

    for k, target_t in enumerate(t_eval):
        while current_t < target_t - 1e-14:
            h = min(dt, target_t - current_t)

            # Strang split for exact BGK relaxation around the FP transport step.
            if nu_collision > 0:
                f = _bgk_exact_step(f, f_eq, nu_collision, 0.5 * h)

            f = transport_step(f, h)

            if nu_collision > 0:
                f = _bgk_exact_step(f, f_eq, nu_collision, 0.5 * h)

            minf = float(np.min(f))
            if minf < -positivity_tol:
                raise FloatingPointError(
                    f"solution lost positivity: min(f)={minf:.3e} at t={current_t + h:.6g}. "
                    "Use theta=1 and/or reduce dt."
                )
            # Only clean roundoff-scale negatives.
            if minf < 0:
                f[f < 0] = 0.0
                # restore mass at roundoff level
                f *= initial_mass / (np.sum(f) * dv)

            current_t += h

        out[k] = f

    return FPSolution(v=v.copy(), times=t_eval.copy(), f=out)


if __name__ == "__main__":
    # Minimal example: velocity diffusion plus weak BGK relaxation.
    v, dv = velocity_grid(-8.0, 8.0, 400)
    f0 = maxwellian(v, sigma=1.0)
    feq = maxwellian(v, sigma=1.0)
    D0 = 0.1

    sol = solve_fp(
        v,
        f0,
        np.linspace(0.0, 5.0, 11),
        drift=0.0,
        diffusion=lambda x: D0 * (1.0 + 0.1 * x**2),
        nu_collision=0.1,
        f_equilibrium=feq,
        dt=0.01,
    )
    print("mass range:", sol.mass().min(), sol.mass().max())
    print("min f:", sol.f.min())
