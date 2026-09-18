import numpy as np
from solver.fp_solver import (
    velocity_grid,
    maxwellian,
    normalize_distribution,
    build_chang_cooper_operator,
    solve_fp,
)


def rel_l1(a, b, dv):
    return np.sum(np.abs(a - b)) * dv / (np.sum(np.abs(b)) * dv)


def test_operator_mass_conservation():
    v, dv = velocity_grid(-5, 5, 201)
    L = build_chang_cooper_operator(
        v,
        drift=lambda x: -0.3 * x,
        diffusion=lambda x: 0.2 * (1 + 0.1 * x**2),
    )
    rng = np.random.default_rng(0)
    f = rng.random(len(v))
    rate = np.sum(L @ f) * dv
    assert abs(rate) < 1e-12


def test_constant_diffusion_gaussian():
    v, dv = velocity_grid(-8, 8, 500)
    sigma0 = 1.0
    D = 0.2
    t = 0.75
    f0 = maxwellian(v, sigma=sigma0)
    sol = solve_fp(v, f0, np.array([0.0, t]), diffusion=D, drift=0.0, dt=0.002)
    sigma_t = np.sqrt(sigma0**2 + 2 * D * t)
    exact = maxwellian(v, sigma=sigma_t)
    err = rel_l1(sol.f[-1], exact, dv)
    assert err < 3e-3, err
    assert np.max(np.abs(sol.mass() - sol.mass()[0])) < 1e-11
    assert sol.f.min() >= -1e-13


def test_ornstein_uhlenbeck_stationary_maxwellian():
    v, dv = velocity_grid(-7, 7, 400)
    sigma = 1.2
    gamma = 0.4
    D = gamma * sigma**2
    f0 = maxwellian(v, sigma=sigma)
    sol = solve_fp(
        v,
        f0,
        np.array([0.0, 1.0, 5.0]),
        drift=lambda x: -gamma * x,
        diffusion=D,
        dt=0.02,
    )
    err = np.max(np.abs(sol.f[-1] - f0))
    assert err < 5e-12, err
    assert np.max(np.abs(sol.mass() - 1.0)) < 1e-11


def test_bgk_collision_exact():
    v, dv = velocity_grid(-8, 8, 400)
    f0 = maxwellian(v, sigma=1.8)
    feq = maxwellian(v, sigma=0.8)
    nu = 0.7
    t = 1.3
    # Turn off transport completely.
    sol = solve_fp(
        v,
        f0,
        np.array([0.0, t]),
        drift=0.0,
        diffusion=0.0,
        nu_collision=nu,
        f_equilibrium=feq,
        dt=0.07,
    )
    exact = feq + (f0 - feq) * np.exp(-nu * t)
    err = np.max(np.abs(sol.f[-1] - exact))
    assert err < 2e-13, err
    assert np.max(np.abs(sol.mass() - 1.0)) < 1e-11


def test_variable_diffusion_positive_and_conservative():
    v, dv = velocity_grid(-5, 5, 300)
    f0 = maxwellian(v, sigma=1.0)
    D0 = 0.1
    sol = solve_fp(
        v,
        f0,
        np.linspace(0, 5, 11),
        drift=0.0,
        diffusion=lambda x: D0 * (x**2 / 25.0),
        dt=0.01,
    )
    assert sol.f.min() >= -1e-13
    assert np.max(np.abs(sol.mass() - 1.0)) < 1e-10


if __name__ == "__main__":
    tests = [
        test_operator_mass_conservation,
        test_constant_diffusion_gaussian,
        test_ornstein_uhlenbeck_stationary_maxwellian,
        test_bgk_collision_exact,
        test_variable_diffusion_positive_and_conservative,
    ]
    for fn in tests:
        fn()
        print(f"PASS: {fn.__name__}")
