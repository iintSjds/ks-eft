# EFT Pseudopotential: Quasiparticle Band Narrowing from Frozen Core Dynamics

Companion code for:

> **Kohn–Sham Hamiltonian from Effective Field Theory: Quasiparticle Band Narrowing from Frozen Core Dynamics**
> Xiansheng Cai, Han Wang, Kun Chen
> [arXiv:2604.25199](https://arxiv.org/abs/2604.25199)

## Overview

This repository contains the computational pipeline for computing the
frozen-core quasiparticle renormalization factor $z^{\mathrm{core}}_{\nu\mathbf{k}}$
and the corrected quasiparticle band structure for Li, Na, K, Ca, Mg, Al, Si,
and LiH.

The pipeline has two stages:
1. **Stage 1** (`run_ks.jl`): Kohn–Sham DFT calculation (SCF + bands) using [DFTK.jl](https://dftk.org)
2. **Stage 2** (`freq_correction.jl`): Post-SCF quasiparticle correction using the EFT form factors and excitation energies

## Requirements

- Julia ≥ 1.10
- Packages: DFTK, Interpolations, SpecialFunctions, Unitful, UnitfulAtomic, JLD2, Plots, PseudoPotentialData, PyPlot
- Python with `matplotlib` installed (used by PyPlot for figure generation)

Install dependencies:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()  # resolves from Project.toml
```

If CUDA artifact download fails (no GPU needed), create `LocalPreferences.toml`:
```toml
[CUDA_Runtime_jll]
local = "true"
version = "none"
```

## Usage

From the repository root:
```bash
# Run KS-DFT for sodium
julia --project=. sodium/run_ks.jl

# Compute QP correction for sodium
julia --project=. sodium/freq_correction.jl

# Generate all figures
bash generate_figures.sh
```

## Directory Structure

```
├── Project.toml                    Julia project (run Pkg.instantiate())
├── generate_figures.sh             Reproduce all figures
├── lithium/                        Li (Z=3, 1s² core, analytic)
│   ├── run_ks.jl                   Stage 1: GTH LDA
│   ├── run_ks_eft.jl               Stage 1: EFT nonlocal PSP
│   ├── freq_correction_new.jl      Stage 2: coherent QP
│   ├── psp_eft_li.jl               EFT potential functions
│   └── psp_eft_nonlocal.jl         Nonlocal KB PSP
├── sodium/                         Na (Z=11, [Ne] core)
│   ├── atomic_hf.jl                Radial atomic DFT-LDA solver (shared)
│   ├── run_ks.jl                   Stage 1
│   └── freq_correction.jl          Stage 2
├── potassium/                      K (Z=19, [Ar] core)
├── calcium/                        Ca (Z=20, [Ar] core)
├── magnesium/                      Mg (Z=12, [Ne] core)
├── aluminum/                       Al (Z=13, [Ne] core)
├── silicon/                        Si (Z=14, [Ne] core)
├── lih/                            LiH (multi-site test)
└── plotting/                       Figure generation scripts
```

## Key Formula

The quasiparticle energy is:

$$\epsilon^{\mathrm{QP}}_{\nu\mathbf{k}} = \epsilon_F + z^{\mathrm{core}}_{\nu\mathbf{k}} (\epsilon^{\mathrm{KS}}_{\nu\mathbf{k}} - \epsilon_F)$$

where the frozen-core factor is:

$$z^{\mathrm{core}}_{\nu\mathbf{k}} = \frac{1}{1 + \sum_c |F^c_{\nu\mathbf{k}}|^2 / \Delta E_c^2}$$

with $F^c_{\nu\mathbf{k}} = \sum_{\mathbf{G}} c_{\nu\mathbf{k}}(\mathbf{G}) f^c_{|\mathbf{k}+\mathbf{G}|}$ the coherent form factor.

## License

MIT
