E73 S23 Lab 1 – 1D Schrödinger Simulations

This project solves the time-independent Schrödinger equation in 1D with finite differences for four scenarios:
- Infinite square well
- Finite square well
- Rectangular barrier in a box
- Simple harmonic oscillator

Outputs:
- Prints labeled energy eigenvalues (in eV) for each scenario.
- Saves labeled plots overlaying potential and the first few eigenfunctions:
  - `plot_infinite_well.png`
  - `plot_finite_well.png`
  - `plot_barrier.png`
  - `plot_harmonic_oscillator.png`

Requirements
------------
- Python 3.9+
- numpy, scipy, matplotlib

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run
---

```bash
python3 schrodinger_1d_lab1.py
```

Notes
-----
- Energies are computed in Joules and printed in eV. Plots show x in nm and potentials/energies in eV.
- Infinite well uses Dirichlet boundary conditions at the domain edges to model infinite walls.
- Finite well depth is specified as a positive magnitude and applied as a negative potential inside the well.
- You can adjust grid size, domain width, and potential parameters in the scenario functions.

# Web interface

Open `index.html` directly in a browser to use the interactive, in-browser simulations (no build needed). You can adjust parameters and toggle the display of the potential, probability density, and wavefunction overlays. A download button lets you save the canvas as a PNG.

## GitHub Pages deployment

This repository can be deployed to GitHub Pages using the included workflow at `.github/workflows/deploy.yml`. After pushing to `main`, enable GitHub Pages in the repository settings with Source: GitHub Actions. The site will then be available at your repository's Pages URL.

