"""
E73 S23 Lab 1 - 1D Time-Independent Schrödinger Equation (Finite Differences)

This script computes eigenvalues and eigenfunctions for:
- Infinite square well (infinite potential walls)
- Finite square well
- Rectangular barrier
- Simple harmonic oscillator

Units: SI internally; printed energies converted to eV; positions printed in nm for plots.
"""

from __future__ import annotations

import numpy as np
import scipy.constants as const
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt


# ================================
# Constants and unit conversions
# ================================
HBAR = const.hbar
ELECTRON_MASS = const.m_e
JOULE_TO_EV = 1.0 / const.e
NM = 1e-9


# ================================
# Core numerical building blocks
# ================================
def build_second_derivative_operator(num_points: int, dx: float) -> sparse.csr_matrix:
    """
    Build the second-derivative operator with Dirichlet boundary conditions
    using a 3-point central finite-difference stencil on a uniform grid.

    d2/dx2 ≈ (f[i+1] - 2 f[i] + f[i-1]) / dx^2
    """
    main_diag = -2.0 * np.ones(num_points)
    off_diag = 1.0 * np.ones(num_points - 1)
    diags = [off_diag, main_diag, off_diag]
    D2 = sparse.diags(diagonals=diags, offsets=[-1, 0, 1], shape=(num_points, num_points), format="csr")
    return D2 / (dx * dx)


def build_kinetic_operator(num_points: int, dx: float, mass: float) -> sparse.csr_matrix:
    """
    Kinetic energy operator T = -(ħ^2 / 2m) d2/dx2
    with Dirichlet boundary conditions on the grid ends.
    """
    d2 = build_second_derivative_operator(num_points, dx)
    prefactor = -(HBAR * HBAR) / (2.0 * mass)
    return prefactor * d2


def solve_schrodinger_1d(x: np.ndarray, potential: np.ndarray, mass: float, num_states: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the 1D time-independent Schrödinger equation for given grid x and potential V(x).

    Returns (energies [J], wavefunctions [normalized]). Energies are sorted ascending.
    """
    num_points = x.size
    dx = float(x[1] - x[0])

    # Build Hamiltonian H = T + V
    T = build_kinetic_operator(num_points, dx, mass)
    V = sparse.diags(potential, offsets=0, shape=(num_points, num_points), format="csr")
    H = (T + V).asformat("csr")

    # Compute k lowest eigenpairs. 'SA' -> smallest algebraic (handles negative energies too)
    k = min(num_states, num_points - 2)
    eigvals, eigvecs = eigsh(H, k=k, which="SA")

    # Sort ascending
    sort_idx = np.argsort(eigvals)
    energies = eigvals[sort_idx]
    wavefuncs = eigvecs[:, sort_idx]

    # Normalize wavefunctions (L2 on x)
    for j in range(wavefuncs.shape[1]):
        norm = np.sqrt(np.trapz(np.abs(wavefuncs[:, j]) ** 2, x))
        if norm > 0:
            wavefuncs[:, j] /= norm

    return energies, wavefuncs


# ================================
# Plotting utilities
# ================================
def plot_potential_and_states(
    x: np.ndarray,
    potential: np.ndarray,
    energies: np.ndarray,
    wavefuncs: np.ndarray,
    title: str,
    filename: str,
    max_states_to_plot: int = 5,
) -> None:
    """
    Plot V(x) (in eV) and overlay the first few eigenfunctions scaled and offset by their energies.
    """
    x_nm = x / NM
    V_eV = potential * JOULE_TO_EV

    plt.figure(figsize=(9, 6))
    plt.plot(x_nm, V_eV, "k-", lw=2, label="V(x)")

    num_plot = min(max_states_to_plot, energies.size)
    if num_plot > 0:
        # Scale wavefunctions for visualization
        psi = wavefuncs[:, :num_plot]
        E = energies[:num_plot] * JOULE_TO_EV

        # Compute a vertical scale so that wavefunctions are visible
        v_range = np.max(V_eV) - np.min(V_eV)
        if not np.isfinite(v_range) or v_range <= 0:
            v_range = 1.0
        scale = 0.1 * v_range

        for i in range(num_plot):
            psi_i = psi[:, i]
            # Normalize to max amplitude 1 for plotting
            max_amp = np.max(np.abs(psi_i))
            if max_amp == 0:
                continue
            psi_scaled = (psi_i / max_amp) * scale + E[i]
            plt.plot(x_nm, psi_scaled, lw=1.5, label=f"ψ{i}(x) + E{i}")
            # Draw a horizontal line for the energy level
            plt.hlines(E[i], x_nm[0], x_nm[-1], colors="C1", linestyles=":", alpha=0.6)

    plt.xlabel("x (nm)")
    plt.ylabel("Energy / Potential (eV)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_probability_density(
    x: np.ndarray,
    wavefuncs: np.ndarray,
    energies: np.ndarray,
    title: str,
    filename: str,
    max_states_to_plot: int = 5,
) -> None:
    """
    Plot the probability densities |ψ(x)|^2 for the first few eigenfunctions.
    """
    x_nm = x / NM
    plt.figure(figsize=(9, 6))

    num_plot = min(max_states_to_plot, wavefuncs.shape[1])
    if num_plot > 0:
        for i in range(num_plot):
            prob_density = np.abs(wavefuncs[:, i]) ** 2
            E_eV = energies[i] * JOULE_TO_EV
            plt.plot(x_nm, prob_density, lw=1.5, label=f"|ψ{i}(x)|², E{i}={E_eV:.3f} eV")

    plt.xlabel("x (nm)")
    plt.ylabel("Probability Density |ψ(x)|²")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def print_energies(label: str, energies_joule: np.ndarray, num: int | None = None) -> None:
    """
    Print energies in eV with a clear section header.
    """
    print("\n" + "=" * 72)
    print(f"{label}")
    print("-" * 72)
    arr = energies_joule * JOULE_TO_EV
    if num is not None:
        arr = arr[:num]
    for i, E in enumerate(arr):
        print(f"State {i:2d}: {E: .6f} eV")
    print("=" * 72)


# ================================
# Potential definitions
# ================================
def potential_infinite_well(x: np.ndarray) -> np.ndarray:
    """
    Infinite well: V=0 inside; Dirichlet BC at domain edges effectively act as infinite walls.
    """
    return np.zeros_like(x)


def potential_finite_square_well(x: np.ndarray, well_width: float, well_depth_eV: float) -> np.ndarray:
    """
    Finite square well centered at x=0:
    V(x) = -V0 inside |x| <= well_width/2, 0 outside.
    well_depth_eV is positive and interpreted as the magnitude of the negative potential inside the well.
    """
    V0 = well_depth_eV / JOULE_TO_EV  # convert eV -> J
    V = np.zeros_like(x)
    inside = np.abs(x) <= (well_width / 2.0)
    V[inside] = -V0
    return V


def potential_barrier(x: np.ndarray, barrier_width: float, barrier_height_eV: float) -> np.ndarray:
    """
    Rectangular barrier centered at x=0:
    V(x) = +V0 inside |x| <= barrier_width/2, 0 outside.
    """
    V0 = barrier_height_eV / JOULE_TO_EV  # convert eV -> J
    V = np.zeros_like(x)
    inside = np.abs(x) <= (barrier_width / 2.0)
    V[inside] = V0
    return V


def potential_harmonic_oscillator(x: np.ndarray, mass: float, omega: float) -> np.ndarray:
    """
    Simple harmonic oscillator: V(x) = 1/2 m ω^2 x^2
    """
    return 0.5 * mass * (omega ** 2) * (x ** 2)


# ================================
# Scenarios
# ================================
def run_infinite_square_well() -> None:
    # Domain and grid
    L = 1.0 * NM  # 1 nm wide infinite well
    num_points = 1200
    x = np.linspace(-L / 2.0, L / 2.0, num_points)
    V = potential_infinite_well(x)

    # Solve
    E, psi = solve_schrodinger_1d(x, V, ELECTRON_MASS, num_states=6)

    # Analytical comparison: E_n = (n^2 π^2 ħ^2) / (2 m L^2)
    n_vals = np.arange(1, 7)
    E_analytical = ((n_vals ** 2) * (np.pi ** 2) * (HBAR ** 2)) / (2.0 * ELECTRON_MASS * (L ** 2))

    print_energies("Infinite Square Well: Numerical energies (eV)", E)
    print("Analytical energies (eV) (n=1..6):")
    for n, Ea in zip(n_vals, E_analytical * JOULE_TO_EV):
        print(f"  n={n}: {Ea: .6f} eV")

    plot_potential_and_states(
        x,
        V,
        E,
        psi,
        title="Infinite Square Well (L = 1 nm)",
        filename="plot_infinite_well.png",
        max_states_to_plot=5,
    )
    print("Saved figure: plot_infinite_well.png")

    plot_probability_density(
        x,
        psi,
        E,
        title="Infinite Square Well: Probability Densities",
        filename="plot_infinite_well_prob_density.png",
        max_states_to_plot=5,
    )
    print("Saved figure: plot_infinite_well_prob_density.png")


def run_finite_square_well() -> None:
    # Domain larger than well, with Dirichlet at edges acting as high walls
    L_domain = 4.0 * NM  # total domain width 4 nm
    num_points = 1600
    x = np.linspace(-L_domain / 2.0, L_domain / 2.0, num_points)

    # Well parameters
    well_width = 0.6 * NM
    well_depth_eV = 0.50  # depth magnitude inside well (negative potential)
    V = potential_finite_square_well(x, well_width=well_width, well_depth_eV=well_depth_eV)

    # Solve
    E, psi = solve_schrodinger_1d(x, V, ELECTRON_MASS, num_states=8)

    print_energies("Finite Square Well: Numerical energies (eV)", E)
    # Highlight bound states (E < 0 in this convention)
    bound = E * JOULE_TO_EV < 0.0
    num_bound = int(np.count_nonzero(bound))
    if num_bound > 0:
        print(f"Bound states (E < 0 eV): {num_bound}")
        for i in range(num_bound):
            print(f"  Bound state {i}: {E[i] * JOULE_TO_EV: .6f} eV")
    else:
        print("No bound states found with these parameters.")

    plot_potential_and_states(
        x,
        V,
        E,
        psi,
        title=f"Finite Square Well (width = {well_width/NM:.2f} nm, depth = {well_depth_eV} eV)",
        filename="plot_finite_well.png",
        max_states_to_plot=5,
    )
    print("Saved figure: plot_finite_well.png")

    plot_probability_density(
        x,
        psi,
        E,
        title="Finite Square Well: Probability Densities",
        filename="plot_finite_well_prob_density.png",
        max_states_to_plot=5,
    )
    print("Saved figure: plot_finite_well_prob_density.png")


def run_barrier() -> None:
    # Domain with interior rectangular barrier
    L_domain = 4.0 * NM
    num_points = 1600
    x = np.linspace(-L_domain / 2.0, L_domain / 2.0, num_points)

    barrier_width = 0.5 * NM
    barrier_height_eV = 0.40
    V = potential_barrier(x, barrier_width=barrier_width, barrier_height_eV=barrier_height_eV)

    # Solve
    E, psi = solve_schrodinger_1d(x, V, ELECTRON_MASS, num_states=8)

    print_energies("Rectangular Barrier in Box: Numerical energies (eV)", E)

    plot_potential_and_states(
        x,
        V,
        E,
        psi,
        title=f"Barrier (width = {barrier_width/NM:.2f} nm, height = {barrier_height_eV} eV)",
        filename="plot_barrier.png",
        max_states_to_plot=5,
    )
    print("Saved figure: plot_barrier.png")

    plot_probability_density(
        x,
        psi,
        E,
        title="Rectangular Barrier: Probability Densities",
        filename="plot_barrier_prob_density.png",
        max_states_to_plot=5,
    )
    print("Saved figure: plot_barrier_prob_density.png")


def run_harmonic_oscillator() -> None:
    # Choose ω and domain scaled to the oscillator length x0 = sqrt(ħ/(mω))
    omega = 2.0e15  # rad/s (tunable)
    x0 = np.sqrt(HBAR / (ELECTRON_MASS * omega))
    span = 10.0 * x0  # cover ±10 x0
    num_points = 1800
    x = np.linspace(-span, span, num_points)

    V = potential_harmonic_oscillator(x, ELECTRON_MASS, omega)

    # Solve
    E, psi = solve_schrodinger_1d(x, V, ELECTRON_MASS, num_states=8)

    print_energies("Harmonic Oscillator: Numerical energies (eV)", E)

    # Analytical energies: E_n = ħω (n + 1/2)
    n_vals = np.arange(0, min(8, E.size))
    E_analytical = HBAR * omega * (n_vals + 0.5)
    print("Analytical energies (eV):")
    for n, Ea in zip(n_vals, E_analytical * JOULE_TO_EV):
        print(f"  n={n}: {Ea: .6f} eV")

    plot_potential_and_states(
        x,
        V,
        E,
        psi,
        title=f"Harmonic Oscillator (ω = {omega:.2e} rad/s)",
        filename="plot_harmonic_oscillator.png",
        max_states_to_plot=5,
    )
    print("Saved figure: plot_harmonic_oscillator.png")

    plot_probability_density(
        x,
        psi,
        E,
        title="Harmonic Oscillator: Probability Densities",
        filename="plot_harmonic_oscillator_prob_density.png",
        max_states_to_plot=5,
    )
    print("Saved figure: plot_harmonic_oscillator_prob_density.png")


def main() -> None:
    print("Running E73 S23 Lab 1 Schrödinger simulations...")
    run_infinite_square_well()
    run_finite_square_well()
    run_barrier()
    run_harmonic_oscillator()
    print("\nAll simulations complete. Figures saved in current directory.")


if __name__ == "__main__":
    main()


