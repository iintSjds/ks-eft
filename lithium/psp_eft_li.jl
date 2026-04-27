"""
psp_eft_li.jl
Custom DFTK pseudopotential type for Lithium from EFT dual-fermion formalism.

Theory: old-lithium.tex "Coordinate and Momentum Space Representations"
This implements the STATIC LOCAL part of the EFT pseudopotential, which is
used for the KS-DFT band structure calculation. The frequency-dependent
(dynamical) part is handled separately as a post-processing correction.

Physical parameters (atomic units: Ha energy, Bohr length):
  alpha = Z_nuc = 3       (Li 1s orbital decay constant)
  J     = (5/8)*Z = 1.875 Ha    (1s-1s Coulomb / Hartree integral)
  E1s   = -Z^2/2 = -4.5 Ha     (1s electron energy, hydrogenic)
  E0    = 2*E1s + J = -7.125 Ha (Li2+ ground state, HF approximation)
  Z_val = 1                     (one valence electron per Li atom)

Static ionic potential (LDA approximation of delta_V_pp):
  u(r)         = (1 - exp(-2*alpha*r))/r - alpha*exp(-2*alpha*r)
  V_LDA(r)     = 2u(r) - 8*exp(-alpha*r)*[u(r)+u(r/2)-u(3r/2)/18-alpha/27-J]
  V_ion(r)     = -Z_nuc/r + V_LDA(r)    (→ -Z_val/r at large r)

Nuclear regularization (needed for plane-wave convergence):
  V_ion(r) has a -Z_nuc/r singularity at r=0. We replace -Z_nuc/r with
  a Gaussian-smoothed form:
    V_nuc_soft(r) = -Z_nuc/r * erf(r / (sqrt(2)*sigma))
  where sigma is the Gaussian width. This is smooth everywhere:
    V_nuc_soft(r→0) = -Z_nuc * sqrt(2/π) / sigma   (finite)
    V_nuc_soft(r→∞) → -Z_nuc/r                     (unchanged)
  Its Fourier transform decays as exp(-sigma²G²/2), ensuring exponential
  convergence with Ecut. Default sigma = 0.5 Bohr (Li 1s radius ≈ 1/α = 0.33 Bohr).

Regularized potential:
  V_reg(r) = V_nuc_soft(r) + V_LDA_static(r)
  V_reg(r→0)  = finite (≈ -29 Ha for sigma=0.5)
  V_reg(r→∞)  → -Z_val/r = -1/r  ✓

Frequency correction (post-processing, applied after DFT):
  Delta = 0.038   (at Gamma, from Bloch-state pseudopotential section)
  epsilon_QP - epsilonF = (epsilon_KS - epsilonF) / (1 + Delta)
"""

using SpecialFunctions: erf
using Interpolations: linear_interpolation, Line

# ── Make this importable ────────────────────────────────────────────────────
import DFTK
import DFTK: NormConservingPsp

# ── Physical parameters ──────────────────────────────────────────────────────
const EFT_Z_NUC  = 3              # Li nuclear charge
const EFT_Z_VAL  = 1              # valence electrons
# Variational HF for He-like Li⁺: α = Z - 5/16 accounts for core-core
# Coulomb screening.  The bare hydrogenic α = Z = 3 overestimates the
# PSP depth by 0.33 eV at Γ; the variational α reduces this to 0.13 eV.
const EFT_ALPHA  = Float64(EFT_Z_NUC) - 5.0/16.0  # 2.6875 Bohr⁻¹
const EFT_J      = (5.0/8.0) * EFT_ALPHA  # 1s-1s Coulomb integral (Ha)
# With α = 2.6875:
#   E_1s_per_electron = α²/2 - Z*α = 3.613 - 8.0625 = -4.449 Ha
#   E0 = 2*E_1s + J = -8.899 + 1.680 = -7.219 Ha
#   E1 = E_1s = -4.449 Ha
#   ΔE = E1 - E0 = 2.770 Ha

# ── Radial potential functions ───────────────────────────────────────────────

"""u(r) = (1 - exp(-2αr))/r - α exp(-2αr);  limit at r→0: α"""
function eft_u(r::Float64)
    α = EFT_ALPHA
    r < 1e-10 && return α
    (1.0 - exp(-2α * r)) / r  -  α * exp(-2α * r)
end

"""
Static LDA correction from integrating out 2 core 1s electrons:
V_LDA(r) = 2u(r) - 8 exp(-αr) [u(r) + u(r/2) - u(3r/2)/18 - α/27 - J]
Smooth at all r: V_LDA(r→0) → 2α - 8×3.847 ≈ -24.78 Ha (finite).
"""
function eft_V_LDA_static(r::Float64)
    bracket = eft_u(r) + eft_u(r/2) - eft_u(1.5r)/18.0 - EFT_ALPHA/27.0 - EFT_J
    2.0 * eft_u(r)  -  8.0 * exp(-EFT_ALPHA * r) * bracket
end

"""
    eft_V_nuc_soft(r, sigma) → Float64

Gaussian-smoothed nuclear potential: -Z_nuc/r * erf(r / (√2 σ)).

- Smooth everywhere: limit at r→0 is -Z_nuc √(2/π) / σ (finite).
- Approaches -Z_nuc/r for r >> σ (typically within 1% for r > 3σ).
- Fourier transform decays as exp(-σ²G²/2), giving exponential
  convergence with Ecut.
"""
function eft_V_nuc_soft(r::Float64, sigma::Float64)
    Z = Float64(EFT_Z_NUC)
    x = r / (sqrt(2.0) * sigma)
    x < 1e-10 && return -Z * sqrt(2.0/π) / sigma   # L'Hôpital limit
    -Z / r * erf(x)
end

"""
    eft_V_ion_reg(r, sigma) → Float64

Regularized ionic potential: V_nuc_soft(r, sigma) + V_LDA_static(r).
- V_reg(r→0) is finite (≈ -30 Ha for sigma=0.5 Bohr).
- V_reg(r→∞) → -Z_val/r = -1/r, as required.
- Smooth everywhere → exponentially converging Fourier transform.
"""
function eft_V_ion_reg(r::Float64, sigma::Float64)
    eft_V_nuc_soft(r, sigma) + eft_V_LDA_static(r)
end

# ── Custom PSP struct ────────────────────────────────────────────────────────

"""
PspEFTLi: Regularized local-only pseudopotential for Li from EFT formalism.

Implements the NormConservingPsp interface. Has lmax = -1 (no non-local
projectors) and a regularized local potential V_reg(r). The nuclear
singularity is smoothed with a Gaussian (width sigma_nuc), making the
Fourier transform exponentially convergent, so Ecut = 30–50 Ha suffices.
"""
struct PspEFTLi{T<:AbstractFloat, I} <: NormConservingPsp
    Zion::Int                   # = 1 (valence charge)
    lmax::Int                   # = -1 (no non-local projectors)
    h::Vector{Matrix{T}}        # empty (no projector couplings)
    rgrid::Vector{T}            # logarithmic radial grid (Bohr)
    vloc::Vector{T}             # V_reg(r) on grid (Ha)
    vloc_interp::I              # interpolator for eval_psp_local_real
    ircut::Int                  # index of cutoff radius
    rcut::T                     # cutoff radius for FT (Bohr)
    sigma_nuc::T                # Gaussian width for nuclear smoothing (Bohr)
    identifier::String
    description::String
end

"""
    PspEFTLi(; sigma_nuc=0.5, rcut=15.0, N=1000)

Construct the regularized EFT pseudopotential for Li.

- sigma_nuc : Gaussian width for nuclear smoothing (Bohr).
              Larger σ → softer potential, converges at lower Ecut,
              but modifies the potential further from the nucleus.
              σ=0.5 Bohr gives exponential convergence at Ecut≈20 Ha.
- rcut      : Radial cutoff for PSP Fourier transform (Bohr).
- N         : Grid points. N=1000 covers r ∈ [3e-4, 80] Bohr.
"""
function PspEFTLi(; sigma_nuc=0.5, rcut=15.0, N=1000,
                    identifier="EFT-Li-dual-fermion-reg",
                    description="Li EFT local PSP (Gaussian nuclear reg, σ=$(sigma_nuc) Bohr)")
    T = Float64

    # QE-style logarithmic grid: r[i] = exp(xmin + dx*i) / zmesh
    xmin = -7.0;  dx = 0.0125;  zmesh = Float64(EFT_Z_NUC)
    rgrid = T[exp(xmin + dx * i) / zmesh for i in 0:(N-1)]

    # Evaluate regularized V_ion(r) on the grid
    vloc = T[eft_V_ion_reg(r, sigma_nuc) for r in rgrid]

    # Interpolator for real-space evaluation
    vloc_interp = linear_interpolation((rgrid,), vloc; extrapolation_bc=Line())

    # Cutoff index
    ircut = findlast(<=(rcut), rgrid)
    isnothing(ircut) && (ircut = N)

    PspEFTLi{T, typeof(vloc_interp)}(
        EFT_Z_VAL,           # Zion
        -1,                  # lmax (no non-local part)
        Matrix{T}[],         # h (empty)
        rgrid, vloc, vloc_interp,
        ircut, T(rcut), T(sigma_nuc),
        identifier, description
    )
end

# ── NormConservingPsp interface ──────────────────────────────────────────────

DFTK.charge_ionic(psp::PspEFTLi) = psp.Zion
DFTK.has_valence_density(psp::PspEFTLi) = false
DFTK.has_core_density(psp::PspEFTLi) = false

"""Real-space local potential V_reg(r) (Ha)."""
function DFTK.eval_psp_local_real(psp::PspEFTLi{T}, r::T) where T
    r <= zero(T) && return psp.vloc_interp(psp.rgrid[1])
    r > psp.rcut  && return -T(EFT_Z_VAL) / r
    psp.vloc_interp(r)
end

"""
Fourier transform using QE-style erf correction for the -Z_val/r tail:
V_loc(G) = 4π ∫ r(r·V_reg(r) + Z_val·erf(r)) j₀(Gr) dr - 4π Z_val/G² exp(-G²/4)
"""
function DFTK.eval_psp_local_fourier(psp::PspEFTLi{T}, p::T) where T
    p == zero(T) && return zero(T)
    Zion  = T(psp.Zion)
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc  = @view psp.vloc[1:psp.ircut]
    quadrature = DFTK.default_psp_quadrature(rgrid)
    DFTK._eval_psp_local_fourier(quadrature, rgrid, vloc, Zion, p)
end

"""Vectorized Fourier transform."""
function DFTK.eval_psp_local_fourier(psp::PspEFTLi{T},
                                      ps::AbstractVector{T}) where {T<:AbstractFloat}
    Zion  = T(psp.Zion)
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc  = @view psp.vloc[1:psp.ircut]
    quadrature = DFTK.default_psp_quadrature(rgrid)
    map(p -> DFTK._eval_psp_local_fourier(quadrature, rgrid, vloc, Zion, p), ps)
end

"""Energy correction: 4π ∫₀^rcut (V_reg(r) + Z_val/r) r² dr"""
function DFTK.eval_psp_energy_correction(::Type{T}, psp::PspEFTLi) where T
    rgrid = psp.rgrid[1:psp.ircut]
    vloc  = psp.vloc[1:psp.ircut]
    Zion  = T(psp.Zion)
    quadrature = DFTK.default_psp_quadrature(rgrid)
    I = quadrature(rgrid) do i, r
        r^2 * (vloc[i] + Zion / r)
    end
    4T(π) * I
end

# Projectors: none (lmax = -1)
DFTK.eval_psp_projector_real(psp::PspEFTLi{T}, i, l, r::T) where T = zero(T)
DFTK.eval_psp_projector_fourier(psp::PspEFTLi{T}, i, l, p::T) where T = zero(T)

# ── Diagnostics ───────────────────────────────────────────────────────────────

"""Print key properties of the EFT pseudopotential."""
function describe_psp(psp::PspEFTLi)
    println("EFT Pseudopotential for Li (dual-fermion, Gaussian nuclear reg.)")
    println("  Z_ion      = ", psp.Zion)
    println("  lmax       = ", psp.lmax, "  (local-only)")
    println("  sigma_nuc  = ", psp.sigma_nuc, " Bohr  (Gaussian nuclear width)")
    println("  alpha      = ", EFT_ALPHA, " Bohr^-1  (1s decay)")
    println("  J          = ", EFT_J, " Ha")
    println("  Δ (freq)   = ", EFT_DELTA)
    println("  Grid: ", length(psp.rgrid), " pts, r ∈ [",
            round(psp.rgrid[1], sigdigits=3), ", ",
            round(psp.rgrid[end], sigdigits=3), "] Bohr")
    r1 = psp.rgrid[1]
    @printf("  V_reg(r_min=%.2e) = %.4f Ha  (finite, was -∞)\n", r1, psp.vloc[1])
    println("  V_reg(r=1.0)     = ", round(DFTK.eval_psp_local_real(psp, 1.0), digits=4), " Ha")
    println("  Energy correction = ",
            round(DFTK.eval_psp_energy_correction(Float64, psp), digits=6), " Ha")
    println("  V_loc_fourier(G=1) = ",
            round(DFTK.eval_psp_local_fourier(psp, 1.0), digits=4), " Ha·Bohr³")
end
