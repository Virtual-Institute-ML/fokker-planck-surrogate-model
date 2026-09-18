import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver.fp_solver import velocity_grid, maxwellian, solve_fp


def run_example():
    """
    Run a simple 1D Fokker-Planck example and return the solution.

    Equation:
        df/dt = -d(A f)/dv + d/dv[D(v) df/dv]
                + nu_collision (f_eq - f)

    Example setup:
        A(v) = 0
        D(v) = D0 * (1 + 0.1 v^2)
        f(v,0) = Maxwellian(sigma=1)
        f_eq(v) = Maxwellian(sigma=1)
    """

    # ------------------------------------------------------------------
    # 1. Velocity grid
    # ------------------------------------------------------------------
    vmin = -8.0
    vmax = 8.0
    Nv = 400
    v, dv = velocity_grid(vmin, vmax, Nv)

    # ------------------------------------------------------------------
    # 2. Initial and equilibrium distributions
    # ------------------------------------------------------------------
    sigma0 = 1.0
    f0 = maxwellian(v, sigma=sigma0)
    f_eq = maxwellian(v, sigma=sigma0)

    # ------------------------------------------------------------------
    # 3. Physical parameters
    # ------------------------------------------------------------------
    D0 = 0.1
    nu_collision = 0.1

    def drift(v):
        return np.zeros_like(v)

    def diffusion(v):
        return D0 * (1.0 + 0.1 * v**2)

    # ------------------------------------------------------------------
    # 4. Output times and internal timestep
    # ------------------------------------------------------------------
    times = np.linspace(0.0, 5.0, 11)
    dt = 0.01

    # ------------------------------------------------------------------
    # 5. Solve the Fokker-Planck equation
    # ------------------------------------------------------------------
    sol = solve_fp(
        v,
        f0,
        times,
        drift=drift,
        diffusion=diffusion,
        nu_collision=nu_collision,
        f_equilibrium=f_eq,
        dt=dt,
    )

    return sol


def plot_time_evolution(sol, savepath=None):
    """
    Plot f(v,t) for all saved output times.
    """

    plt.figure(figsize=(8, 5.5))

    for i, t in enumerate(sol.times):
        plt.plot(
            sol.v,
            sol.f[i],
            label=fr"$t={t:.1f}$",
        )

    plt.xlabel(r"$v$")
    plt.ylabel(r"$f(v,t)$")
    plt.yscale("log")
    plt.xlim(-8,8)
    plt.ylim(1e-6,1)
    plt.title("Time evolution of the Fokker-Planck solution")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"Figure saved to: {savepath}")

    plt.close()


def print_diagnostics(sol):
    """
    Print simple numerical diagnostics.
    """

    mass = sol.mass()

    print("=== Diagnostics ===")
    print(f"Initial mass : {mass[0]:.15f}")
    print(f"Final mass   : {mass[-1]:.15f}")
    print(f"Mass range   : {mass.min():.15f} -- {mass.max():.15f}")
    print(f"Minimum f    : {sol.f.min():.6e}")


if __name__ == "__main__":
    solution = run_example()

    print_diagnostics(solution)

    plot_time_evolution(
        solution,
        savepath="artifacts/fp_time_evolution.png",
    )
