"""
freq_correction.jl
Apply the EFT frequency (dynamical) correction to the KS band structure.

Theory (old-lithium.tex, Sec. "Coordinate and Momentum Space Representations"):

The dynamical part of the pseudopotential in momentum space is:
    V_dyn(K, K') = (64π/α) · A_K · A_{K'} / (iω_n + (E₁ − E₀))

where
    A_K = (Q²+33)/((Q²+1)(Q²+9)²) − (J/α)/(Q²+1)²,   Q = K/α

Expanding to linear order in iω_n:
    V_dyn ≈ (64π/α) A_K A_{K'} [1/(E₁−E₀) − iω_n/(E₁−E₀)²]

The static part (1/(E₁−E₀) term) is already included in V_LDA.
The iω_n term renormalizes the quasiparticle weight.

Under LDA (K'→0), the correction for a Bloch state |nk⟩ is:
    ε_QP(nk) = εF + (ε_KS(nk) − εF) / (1 + Δ(nk))

where
    Δ(nk) = Σ_G |c_{nk}(G)|² · Δ(|k+G|)
    Δ(K) = (64π/α) · (33/81 − J/α) · A_K / (E₁ − E₀)²

Note: (33/81 − J/α) and A_K are both negative for small K, so Δ(K) > 0
and 1/(1+Δ) < 1, compressing the bandwidth.

Usage: julia --project=run/psp run/psp/freq_correction.jl
"""

using Printf
using LinearAlgebra
using Unitful
using UnitfulAtomic
using DFTK
using Plots

# Load EFT NC PSP
include(joinpath(@__DIR__, "psp_eft_nc.jl"))

outdir = @__DIR__

# ══════════════════════════════════════════════════════════════════════════════
# EFT frequency correction functions
# ══════════════════════════════════════════════════════════════════════════════

const FC_ALPHA = Float64(EFT_Z_NUC)   # α = 3 Bohr⁻¹
const FC_J     = (5.0/8.0) * EFT_Z_NUC  # J = 1.875 Ha
const FC_E1    = -Float64(EFT_Z_NUC)^2 / 2.0  # E_1s = -4.5 Ha
const FC_E0    = 2.0 * FC_E1 + FC_J   # E_0 = -7.125 Ha

"""
    A_K(K) → Float64

Momentum-space form factor from the ionization integral ⟨11|1,K⟩.
    A_K = (Q²+33)/((Q²+1)(Q²+9)²) − (J/α)/(Q²+1)²
where Q = K/α.
"""
function eft_A_K(K::Float64)
    Q = K / FC_ALPHA
    Q2 = Q^2
    (Q2 + 33.0) / ((Q2 + 1.0) * (Q2 + 9.0)^2) -
        (FC_J / FC_ALPHA) / (Q2 + 1.0)^2
end

"""
    delta_K(K) → Float64

Quasiparticle weight renormalization at total momentum K.

From old-lithium.tex Eq. for ε_K renormalization (no extra minus sign):
    Δ(K) = (64π/α) · (33/81 − J/α) · A(K) / (E₁ − E₀)²

Note: (33/81 − J/α) < 0 and A(K) < 0 for small K, so Δ(K) > 0.
This gives 1/(1+Δ) < 1, compressing the bandwidth as expected.
"""
function eft_delta_K(K::Float64)
    A0_factor = 33.0/81.0 - FC_J/FC_ALPHA   # = -0.2176
    AK = eft_A_K(K)
    denom = (FC_E1 - FC_E0)^2   # (E₁ − E₀)² = 2.625² = 6.890625
    64.0 * π / FC_ALPHA * A0_factor * AK / denom
end

"""
    delta_nk(ψnk, kpt, basis) → Float64

Bloch-state-averaged quasiparticle renormalization factor:
    Δ(nk) = Σ_G |c_{nk}(G)|² · Δ(|k+G|)

where c_{nk}(G) are the plane-wave coefficients of the eigenstate.
"""
function eft_delta_nk(ψnk::AbstractVector, kpt, basis)
    recip_lat = basis.model.recip_lattice
    k_frac = kpt.coordinate
    Gvecs = DFTK.G_vectors(basis, kpt)

    Δ = 0.0
    for (ig, G) in enumerate(Gvecs)
        kpG_frac = k_frac .+ G   # fractional
        kpG_cart = recip_lat * kpG_frac   # Cartesian (Bohr⁻¹)
        K = norm(kpG_cart)
        weight = abs2(ψnk[ig])
        Δ += weight * eft_delta_K(K)
    end
    Δ
end

# ══════════════════════════════════════════════════════════════════════════════
# Print diagnostic info about the correction
# ══════════════════════════════════════════════════════════════════════════════

println("=== EFT frequency correction parameters ===\n")
@printf("  α     = %.1f Bohr⁻¹\n", FC_ALPHA)
@printf("  J     = %.4f Ha\n", FC_J)
@printf("  E_1s  = %.4f Ha\n", FC_E1)
@printf("  E_0   = %.4f Ha\n", FC_E0)
@printf("  E₁−E₀ = %.4f Ha = %.2f eV\n", FC_E1 - FC_E0, (FC_E1 - FC_E0)*27.2114)
@printf("  A(0)  = %.6f\n", 33.0/81.0 - FC_J/FC_ALPHA)
@printf("  Δ(K=0)= %.6f\n", eft_delta_K(0.0))
@printf("  Δ(K=1)= %.6f\n", eft_delta_K(1.0))
@printf("  Δ(K=3)= %.6f\n", eft_delta_K(3.0))
@printf("  Δ(K=∞)→ 0\n")

# ══════════════════════════════════════════════════════════════════════════════
# SCF + Band structure with EFT NC PSP
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== SCF with EFT NC PSP ===\n")

a = 6.632
lattice = (a / 2) * [[-1  1  1]; [ 1 -1  1]; [ 1  1 -1]]
positions = [[0.0, 0.0, 0.0]]

psp_nc = PspEFTNC(sigma_nuc=0.5, r_c=2.5)
Li = ElementPsp(:Li, psp_nc)
model = model_LDA(lattice, [Li], positions;
                  temperature=0.001, smearing=DFTK.Smearing.FermiDirac())

Ecut = 25.0
kgrid = [8, 8, 8]
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis; tol=1e-8, mixing=KerkerMixing(),
                               is_converged=DFTK.ScfConvergenceEnergy(1e-8))

@printf("  εF = %+.6f Ha = %+.4f eV\n", scfres.εF, scfres.εF * 27.2114)

println("\n=== Computing band structure ===\n")

bands = compute_bands(scfres; n_bands=8, kline_density=40u"bohr")

# ══════════════════════════════════════════════════════════════════════════════
# Apply frequency correction to all eigenvalues
# ══════════════════════════════════════════════════════════════════════════════

println("=== Applying frequency correction ===\n")

n_kpts = length(bands.basis.kpoints)
n_bands = length(bands.eigenvalues[1])
εF = scfres.εF

eigenvalues_qp = similar(bands.eigenvalues)
deltas = Vector{Vector{Float64}}(undef, n_kpts)

for ik in 1:n_kpts
    kpt = bands.basis.kpoints[ik]
    eigenvalues_qp[ik] = similar(bands.eigenvalues[ik])
    deltas[ik] = zeros(n_bands)

    for n in 1:n_bands
        ψnk = bands.ψ[ik][:, n]
        Δ = eft_delta_nk(ψnk, kpt, bands.basis)
        deltas[ik][n] = Δ

        ε_ks = bands.eigenvalues[ik][n]
        eigenvalues_qp[ik][n] = εF + (ε_ks - εF) / (1.0 + Δ)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Print comparison at high-symmetry points
# ══════════════════════════════════════════════════════════════════════════════

println("=== Eigenvalues at high-symmetry points ===\n")

function find_kpoint(band_data, target_coord; tol=0.02)
    for (ik, kpt) in enumerate(band_data.basis.kpoints)
        if norm(kpt.coordinate .- target_coord) < tol
            return ik
        end
    end
    return nothing
end

hs_points = [
    ("Γ",  [0.0, 0.0, 0.0]),
    ("H",  [0.5, -0.5, 0.5]),
    ("N",  [0.0, 0.0, 0.5]),
    ("P",  [0.25, 0.25, 0.25]),
]

for (label, coord) in hs_points
    ik = find_kpoint(bands, coord)
    isnothing(ik) && continue

    @printf("  %-3s  %6s  %12s  %12s  %8s  %12s\n",
            label, "Band", "KS (eV)", "QP (eV)", "Δ(nk)", "Shift (eV)")
    for n in 1:min(6, n_bands)
        e_ks = (bands.eigenvalues[ik][n] - εF) * 27.2114
        e_qp = (eigenvalues_qp[ik][n] - εF) * 27.2114
        Δ = deltas[ik][n]
        shift = e_qp - e_ks
        @printf("       %3d     %+8.3f    %+8.3f    %6.4f    %+8.3f\n",
                n, e_ks, e_qp, Δ, shift)
    end
    println()
end

# ══════════════════════════════════════════════════════════════════════════════
# Print Δ statistics
# ══════════════════════════════════════════════════════════════════════════════

println("=== Δ(nk) statistics for band 1 (valence 2s) ===\n")

Δ_band1 = [deltas[ik][1] for ik in 1:n_kpts]
@printf("  min Δ = %.6f,  max Δ = %.6f,  mean Δ = %.6f\n",
        minimum(Δ_band1), maximum(Δ_band1), sum(Δ_band1)/length(Δ_band1))

# Check: the old constant Δ=0.038 was computed at Γ with G=G'=0 only.
# Our state-averaged Δ should differ because it includes all G components.
iΓ = find_kpoint(bands, [0.0, 0.0, 0.0])
if !isnothing(iΓ)
    @printf("  Δ at Γ (band 1, state-averaged) = %.6f\n", deltas[iΓ][1])
    @printf("  Δ(K=0) (single G=0 component)   = %.6f\n", eft_delta_K(0.0))
    @printf("  Old constant Δ (obsolete)        = 0.038\n")
end

# ══════════════════════════════════════════════════════════════════════════════
# Plot: KS vs QP band structure
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Plotting ===\n")

# Use DFTK's data_for_plotting to get kdistances and ticks
dat = DFTK.data_for_plotting(bands)

Ha_to_eV = 27.2114

# Extract KS eigenvalues array (n_kcoord × n_bands)
evals_ks = dat.eigenvalues   # already n_kcoord × n_bands × n_spin

# Build QP eigenvalues in the same format
n_kcoord = dat.n_kcoord
n_spin = dat.n_spin
evals_qp_arr = similar(evals_ks)
for σ in 1:n_spin
    for (ito, ik) in enumerate(DFTK.krange_spin(bands.basis, σ))
        evals_qp_arr[ito, :, σ] = eigenvalues_qp[ik]
    end
end

# Plot
p = plot(size=(800, 600), legend=:topright, ylabel="Energy (eV)", xlabel="")

n_plot = min(6, n_bands)
colors_ks = :blue
colors_qp = :red

for n in 1:n_plot
    ks_eV = (evals_ks[:, n, 1] .- εF) .* Ha_to_eV
    qp_eV = (evals_qp_arr[:, n, 1] .- εF) .* Ha_to_eV

    lb_ks = n == 1 ? "KS" : ""
    lb_qp = n == 1 ? "QP (freq. corr.)" : ""

    plot!(p, dat.kdistances, ks_eV; color=colors_ks, lw=2, label=lb_ks)
    plot!(p, dat.kdistances, qp_eV; color=colors_qp, lw=2, ls=:dash, label=lb_qp)
end

# Fermi level
hline!(p, [0.0]; color=:gray, ls=:dot, label="εF", lw=1)

# Ticks
vline!(p, dat.ticks.distances; color=:gray, lw=0.5, label="")
plot!(p; xticks=(dat.ticks.distances, dat.ticks.labels))
xlims!(p, dat.kdistances[1], dat.kdistances[end])
title!(p, "BCC Li — KS vs QP (EFT freq. correction)")

savefig(p, joinpath(outdir, "bands_qp.pdf"))
println("  → bands_qp.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Plot: Δ(K) function
# ══════════════════════════════════════════════════════════════════════════════

Ks = range(0, 15, length=500)
Δs = [eft_delta_K(Float64(K)) for K in Ks]

p_delta = plot(Ks, Δs; xlabel="K (Bohr⁻¹)", ylabel="Δ(K)",
               title="Quasiparticle renormalization Δ(K)", lw=2,
               legend=false, size=(600, 400))
vline!(p_delta, [FC_ALPHA]; ls=:dash, color=:gray, label="α")

savefig(p_delta, joinpath(outdir, "delta_K.pdf"))
println("  → delta_K.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Summary ===\n")

if !isnothing(iΓ)
    e_ks_Γ = (bands.eigenvalues[iΓ][1] - εF) * Ha_to_eV
    e_qp_Γ = (eigenvalues_qp[iΓ][1] - εF) * Ha_to_eV

    @printf("  Band 1 at Γ:  KS = %+.3f eV,  QP = %+.3f eV  (shift = %+.3f eV)\n",
            e_ks_Γ, e_qp_Γ, e_qp_Γ - e_ks_Γ)

    iH = find_kpoint(bands, [0.5, -0.5, 0.5])
    if !isnothing(iH)
        e_ks_H = (bands.eigenvalues[iH][1] - εF) * Ha_to_eV
        e_qp_H = (eigenvalues_qp[iH][1] - εF) * Ha_to_eV
        bw_ks = e_ks_H - e_ks_Γ
        bw_qp = e_qp_H - e_qp_Γ
        @printf("  Band 1 at H:  KS = %+.3f eV,  QP = %+.3f eV  (shift = %+.3f eV)\n",
                e_ks_H, e_qp_H, e_qp_H - e_ks_H)
        @printf("  Bandwidth Γ→H: KS = %.3f eV,  QP = %.3f eV  (ratio = %.4f)\n",
                bw_ks, bw_qp, bw_qp / bw_ks)
    end
end

println("\n=== Done ===")
