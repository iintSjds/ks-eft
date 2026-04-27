"""
freq_correction.jl (LiH)
Stage 2: EFT frequency correction for lithium hydride.

Loads KS data from lih_ks_data.jld2, applies multi-site coherent QP
correction using Li 1s channel (analytic form factor). H has no core
electrons → zero EFT contribution.

Multi-site coherent formula:
  F(nk) = Σ_G c_{nk}(G) e^{iG·τ_Li} f_{1s}(|k+G|)
  Δ(nk) = |F(nk)|² / ΔE²
  ε_QP = ε_F + (ε_KS - ε_F) / (1 + Δ(nk))

Since τ_Li = (0,0,0), the phase factor e^{iG·τ_Li} = 1.

Usage: julia --project=. lih/freq_correction.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] lih/freq_correction.jl")
    println("  Loads: lih/lih_ks_data.jld2")
    println("  Saves: lih/lih_qp_data.jld2, lih/lih_summary.txt")
    exit(0)
end

using Printf
using LinearAlgebra
using JLD2

outdir = @__DIR__
Ha_to_eV = 27.211386245988

# ══════════════════════════════════════════════════════════════════════════════
# Li EFT parameters (analytic, from lithium/psp_eft_li.jl)
# ══════════════════════════════════════════════════════════════════════════════

const LI_ALPHA = 3.0            # 1s orbital decay constant (Bohr⁻¹)
const LI_J     = (5.0/8.0) * 3  # 1s-1s Coulomb integral = 1.875 Ha
const LI_E1    = -4.5            # hydrogenic 1s energy (Ha)
const LI_E0    = 2.0 * LI_E1 + LI_J  # Li²⁺ ground state = -7.125 Ha
const LI_ΔE    = LI_E1 - LI_E0       # = 2.625 Ha (core excitation energy)

"""EFT form factor A_K for Li 1s channel."""
function eft_A_K(K::Float64)
    Q = K / LI_ALPHA
    Q2 = Q^2
    (Q2 + 33.0) / ((Q2 + 1.0) * (Q2 + 9.0)^2) -
        (LI_J / LI_ALPHA) / (Q2 + 1.0)^2
end

"""
Form factor f_{1s}(K) = 8√(π/α) A_K.
Has units of energy (Ha).
"""
function eft_f_1s(K::Float64)
    sqrt(64.0 * π / LI_ALPHA) * eft_A_K(K)
end

println("=== Li EFT parameters ===")
@printf("  α = %.1f Bohr⁻¹, J = %.4f Ha, ΔE = %.4f Ha\n", LI_ALPHA, LI_J, LI_ΔE)
@printf("  f(0) = %.6f Ha, f(0)/ΔE = %.6f\n", eft_f_1s(0.0), eft_f_1s(0.0)/LI_ΔE)

# ══════════════════════════════════════════════════════════════════════════════
# Load KS data
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Loading KS data ===\n")

ks_data = load(joinpath(outdir, "lih_ks_data.jld2"))
eigenvalues_arr = ks_data["eigenvalues"]
kdistances = ks_data["kdistances"]
tick_distances = ks_data["tick_distances"]
tick_labels = ks_data["tick_labels"]
εF = ks_data["εF"]
n_kpts = ks_data["n_kpts"]
a_conv = ks_data["a_conv"]
recip_lat = ks_data["recip_lattice"]
ψ_list = ks_data["ψ"]
kcoords = ks_data["kcoords"]
Gvecs_list = ks_data["Gvecs"]

n_bands_comp = size(eigenvalues_arr, 2)
@printf("  Loaded %d k-points, %d bands, εF = %.6f Ha\n",
        n_kpts, n_bands_comp, εF)

# ══════════════════════════════════════════════════════════════════════════════
# Multi-site coherent QP correction
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Coherent QP correction ===\n")

# Coherent: F(nk) = Σ_G c(G) f(|k+G|), Δ = |F|²/ΔE²
# Diagonal (for comparison): Δ_diag = Σ_G |c(G)|² f(|k+G|)f(0)/ΔE²
eqp_arr = similar(eigenvalues_arr)
eqp_diag_arr = similar(eigenvalues_arr)
delta_exact_arr = similar(eigenvalues_arr)
delta_diag_arr = similar(eigenvalues_arr)

f0 = eft_f_1s(0.0)

for ik in 1:n_kpts
    k_frac = kcoords[ik]
    Gvecs = Gvecs_list[ik]
    ψ = ψ_list[ik]

    for n in 1:n_bands_comp
        ψnk = ψ[:, n]

        # Coherent sum F = Σ_G c(G) f(|k+G|)
        F = complex(0.0)
        Δ_diag = 0.0
        for (ig, G) in enumerate(Gvecs)
            kpG = recip_lat * (k_frac .+ Float64.(G))
            K = norm(kpG)
            fK = eft_f_1s(K)
            F += ψnk[ig] * fK
            Δ_diag += abs2(ψnk[ig]) * fK * f0 / LI_ΔE^2
        end
        Δ_exact = abs2(F) / LI_ΔE^2

        delta_exact_arr[ik, n, 1] = Δ_exact
        delta_diag_arr[ik, n, 1] = Δ_diag

        ε_ks = eigenvalues_arr[ik, n, 1]
        eqp_arr[ik, n, 1] = εF + (ε_ks - εF) / (1.0 + Δ_exact)
        eqp_diag_arr[ik, n, 1] = εF + (ε_ks - εF) / (1.0 + Δ_diag)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════

println("=== Results ===\n")

# Find Γ-point
iΓ = nothing
for ik in 1:n_kpts
    if norm(kcoords[ik]) < 0.02
        global iΓ = ik
        break
    end
end

if !isnothing(iΓ)
    println("  Γ-point eigenvalues (eV relative to εF):")
    @printf("  %5s  %10s  %10s  %10s  %10s  %10s\n",
            "Band", "KS", "QP(coh)", "QP(diag)", "Δ(coh)", "Δ(diag)")
    println("  " * "-"^62)
    for n in 1:min(n_bands_comp, 8)
        ε_ks = (eigenvalues_arr[iΓ, n, 1] - εF) * Ha_to_eV
        ε_qp_ex = (eqp_arr[iΓ, n, 1] - εF) * Ha_to_eV
        ε_qp_di = (eqp_diag_arr[iΓ, n, 1] - εF) * Ha_to_eV
        @printf("  %5d  %+10.3f  %+10.3f  %+10.3f  %10.4f  %10.4f\n",
                n, ε_ks, ε_qp_ex, ε_qp_di,
                delta_exact_arr[iΓ, n, 1], delta_diag_arr[iΓ, n, 1])
    end
end

# Band depths
println("\n  Band structure summary (coherent formula):")
@printf("  %5s  %12s  %12s  %10s\n", "Band", "KS depth", "QP depth", "Narrow%")
println("  " * "-"^45)
for n in 1:min(n_bands_comp, 8)
    ks_depths = [(eigenvalues_arr[ik, n, 1] - εF) * Ha_to_eV for ik in 1:n_kpts]
    qp_depths = [(eqp_arr[ik, n, 1] - εF) * Ha_to_eV for ik in 1:n_kpts]
    ks_d = minimum(ks_depths)
    qp_d = minimum(qp_depths)
    narrow = ks_d != 0 ? (1 - qp_d / ks_d) * 100 : 0.0
    @printf("  %5d  %+12.3f  %+12.3f  %10.1f\n", n, ks_d, qp_d, narrow)
end

# ══════════════════════════════════════════════════════════════════════════════
# Save data
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Saving data ===\n")

jldopen(joinpath(outdir, "lih_qp_data.jld2"), "w") do f
    f["kdistances"] = kdistances
    f["eigenvalues"] = eigenvalues_arr
    f["eqp_arr"] = eqp_arr
    f["eqp_diag_arr"] = eqp_diag_arr
    f["delta_exact_arr"] = delta_exact_arr
    f["delta_diag_arr"] = delta_diag_arr
    f["εF"] = εF
    f["tick_distances"] = tick_distances
    f["tick_labels"] = tick_labels
    f["n_bands"] = n_bands_comp
    f["a_conv"] = a_conv
    f["ΔE_1s"] = LI_ΔE
    f["method"] = "coherent"
    f["f_1s_0"] = eft_f_1s(0.0)
end
println("  → lih_qp_data.jld2")

# Human-readable summary
open(joinpath(outdir, "lih_summary.txt"), "w") do io
    println(io, "# LiH EFT QP correction summary (coherent formula)")
    println(io, "# F(nk) = Σ_G c(G) f_{1s}(|k+G|), Δ = |F|²/ΔE²")
    println(io, "# H has no core → zero EFT correction from H sites")
    println(io)
    @printf(io, "Structure: rock salt (Fm3̄m), a = %.3f Bohr\n", a_conv)
    @printf(io, "εF = %.6f Ha = %.4f eV\n", εF, εF * Ha_to_eV)
    println(io)
    @printf(io, "Li EFT: α = %.1f Bohr⁻¹, J = %.4f Ha, ΔE = %.4f Ha\n",
            LI_ALPHA, LI_J, LI_ΔE)
    @printf(io, "f(0) = %.6f Ha, f(0)/ΔE = %.6f\n", eft_f_1s(0.0), eft_f_1s(0.0)/LI_ΔE)
    println(io)
    if !isnothing(iΓ)
        println(io, "Γ-point eigenvalues (eV relative to εF):")
        @printf(io, "%5s  %10s  %10s  %10s  %10s\n",
                "Band", "KS", "QP(coh)", "Δ(coh)", "Δ(diag)")
        for n in 1:min(n_bands_comp, 8)
            ε_ks = (eigenvalues_arr[iΓ, n, 1] - εF) * Ha_to_eV
            ε_qp = (eqp_arr[iΓ, n, 1] - εF) * Ha_to_eV
            @printf(io, "%5d  %+10.3f  %+10.3f  %10.4f  %10.4f\n",
                    n, ε_ks, ε_qp,
                    delta_exact_arr[iΓ, n, 1], delta_diag_arr[iΓ, n, 1])
        end
    end
end
println("  → lih_summary.txt")

println("\n=== Done ===")
