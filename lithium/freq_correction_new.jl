"""
freq_correction.jl (lithium) — NEW 3-PSP pipeline version
Stage 2: Load KS data from all 3 PSPs and apply the EFT coherent frequency correction.

Li uses analytic form factor f(K) — single 1s channel with closed-form expression.
No atomic solver needed. Uses the coherent formula: Δ(nk) = |Σ_G c(G) f(|k+G|) / ΔE|².

3 PSP sources:
  GTH LDA (Zion=1): li_ks_data.jld2      — all bands get QP
  EFT nonlocal (Zion=1): li_eft_ks_data.jld2 — all bands get QP
  PBE (Zion=3): li_pbe_ks_data.jld2      — band 1=1s core (no QP), band 2+=conduction (QP)

Usage: julia --project=. lithium/freq_correction_new.jl
Prerequisites:
  julia --project=. lithium/run_ks.jl
  julia --project=. lithium/run_ks_eft.jl
  julia --project=. lithium/run_ks_pbe.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] lithium/freq_correction_new.jl")
    println("  Loads: lithium/li_ks_data.jld2, li_eft_ks_data.jld2, li_pbe_ks_data.jld2")
    println("  Computes: analytic EFT frequency correction for all 3 PSPs")
    println("  Saves: lithium/li_qp_data.jld2")
    exit(0)
end

using Printf
using LinearAlgebra
using JLD2

outdir = @__DIR__
Ha_to_eV = 27.211386245988

# ══════════════════════════════════════════════════════════════════════════════
# Analytic EFT form factor for Li (single 1s channel)
#
# V_dyn(K,K') = f(K) f(K') / (ω - ΔE)  (separable)
# f(K) = 8√π / √α · A_K   where A_K is the dimensionless form factor
# ΔE = E_1 - E_0 = 2.625 Ha
# ══════════════════════════════════════════════════════════════════════════════

const LI_ALPHA = 3.0 - 5.0/16.0  # variational HF: α = Z - 5/16 = 2.6875 Bohr⁻¹
const LI_J     = (5.0/8.0) * LI_ALPHA  # 1s-1s Coulomb integral = 1.680 Ha
const LI_E1    = LI_ALPHA^2/2 - 3.0*LI_ALPHA  # per-electron kinetic+nuclear = -4.449 Ha
const LI_E0    = 2.0 * LI_E1 + LI_J  # Li⁺ ground state = -7.219 Ha
const LI_ΔE    = LI_E1 - LI_E0       # 2.770 Ha (excitation energy)

"""Dimensionless form factor A_K = f(K) / (8√π α^{-1/2})"""
function eft_A_K(K::Float64)
    Q = K / LI_ALPHA
    Q2 = Q^2
    (Q2 + 33.0) / ((Q2 + 1.0) * (Q2 + 9.0)^2) -
        (LI_J / LI_ALPHA) / (Q2 + 1.0)^2
end

"""Form factor f(K) = 8√π / √α · A_K"""
function eft_f_K(K::Float64)
    8.0 * sqrt(π / LI_ALPHA) * eft_A_K(K)
end

# Coherent Δ(K=0) for free electron: |f(0)/ΔE|²
@printf("  f(0) = %.6f\n", eft_f_K(0.0))
@printf("  Δ_FE(K=0) = |f(0)/ΔE|² = %.6f\n", eft_f_K(0.0)^2 / LI_ΔE^2)

# ══════════════════════════════════════════════════════════════════════════════
# Coherent QP correction
#
# Exact for separable V_dyn:
#   F = Σ_G c_{nk}(G) · f(|k+G|) / ΔE
#   Δ(nk) = |F|²
# ══════════════════════════════════════════════════════════════════════════════

function delta_nk(ψnk::AbstractVector, k_frac, Gvecs_int, recip_lat)
    F = complex(0.0)
    for ig in axes(Gvecs_int, 1)
        G = Gvecs_int[ig, :]
        K = norm(recip_lat * (k_frac .+ G))
        F += ψnk[ig] * eft_f_K(K) / abs(LI_ΔE)
    end
    return abs2(F)
end

function apply_qp!(eigenvalues_qp, deltas, psi, eigenvalues, k_coordinates,
                    G_vectors, recip_lattice, εF;
                    band_indices=nothing)
    n_kpts = length(psi)
    n_bands = length(eigenvalues[1])
    if isnothing(band_indices)
        band_indices = 1:n_bands
    end
    for ik in 1:n_kpts
        k_frac = k_coordinates[ik]
        Gvecs_int = G_vectors[ik]
        for n in 1:n_bands
            if n in band_indices
                ψnk = psi[ik][:, n]
                Δ = delta_nk(ψnk, k_frac, Gvecs_int, recip_lattice)
                deltas[ik][n] = Δ
                eigenvalues_qp[ik][n] = εF + (eigenvalues[ik][n] - εF) / (1.0 + Δ)
            else
                eigenvalues_qp[ik][n] = eigenvalues[ik][n]
                deltas[ik][n] = 0.0
            end
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Load and process each PSP dataset
# ══════════════════════════════════════════════════════════════════════════════

function find_kpoint(k_coords, target; tol=0.02)
    for (ik, kc) in enumerate(k_coords)
        norm(kc .- target) < tol && return ik
    end
    return nothing
end

struct PSPResult
    name::String
    Zion::Int
    εF::Float64
    eigenvalues::Vector{Vector{Float64}}
    eigenvalues_qp::Vector{Vector{Float64}}
    deltas::Vector{Vector{Float64}}
    k_coordinates::Vector{Vector{Float64}}
    # Plotting arrays
    kdistances::Vector{Float64}
    eigenvalues_array::Array{Float64,3}
    eqp_array::Array{Float64,3}
    tick_distances::Vector{Float64}
    tick_labels::Vector{String}
    n_spin::Int
    # Band statistics (for the valence/conduction 2s band)
    valence_band::Int  # which band index is the 2s valence
    depth_ks::Float64
    depth_qp::Float64
    bw_ks::Float64
    bw_qp::Float64
end

function process_psp(name::String, jld2_file::String; conduction_band::Int=1)
    println("\n=== Processing $name ===")
    ks = load(jld2_file)

    psi           = ks["psi"]
    eigenvalues   = ks["eigenvalues"]
    k_coordinates = ks["k_coordinates"]
    G_vectors     = ks["G_vectors"]
    recip_lattice = ks["recip_lattice"]
    εF            = ks["εF"]
    Zion          = ks["Zion"]

    n_kpts = length(psi)
    n_bands = length(eigenvalues[1])
    @printf("  %d k-points, %d bands, Zion=%d, εF=%.6f Ha\n", n_kpts, n_bands, Zion, εF)

    # Band indices to apply QP correction
    band_indices = conduction_band:n_bands

    eigenvalues_qp = [similar(eigenvalues[ik]) for ik in 1:n_kpts]
    deltas = [zeros(n_bands) for _ in 1:n_kpts]

    t0 = time()
    apply_qp!(eigenvalues_qp, deltas, psi, eigenvalues, k_coordinates,
              G_vectors, recip_lattice, εF; band_indices)
    @printf("  QP correction done in %.1f s\n", time() - t0)

    # Band statistics for the 2s valence/conduction band
    e_ks = [(eigenvalues[ik][conduction_band] - εF) * Ha_to_eV for ik in 1:n_kpts]
    e_qp = [(eigenvalues_qp[ik][conduction_band] - εF) * Ha_to_eV for ik in 1:n_kpts]
    depth_ks = minimum(e_ks)
    depth_qp = minimum(e_qp)
    bw_ks = maximum(e_ks) - minimum(e_ks)
    bw_qp = maximum(e_qp) - minimum(e_qp)

    # Build plotting arrays
    kdistances      = ks["kdistances"]
    eigenvalues_arr = ks["eigenvalues_array"]
    tick_distances  = ks["tick_distances"]
    tick_labels     = ks["tick_labels"]
    n_spin          = ks["n_spin"]
    krange_spin_map = ks["krange_spin_map"]

    eqp_arr = similar(eigenvalues_arr)
    for σ in 1:n_spin
        for (ito, ik) in enumerate(krange_spin_map[σ])
            eqp_arr[ito, :, σ] = eigenvalues_qp[ik]
        end
    end

    iΓ = find_kpoint(k_coordinates, [0.0, 0.0, 0.0])
    if !isnothing(iΓ)
        @printf("  Δ at Γ (band %d) = %.4f\n", conduction_band, deltas[iΓ][conduction_band])
    end

    @printf("  KS Γ depth (2s): %+.3f eV\n", depth_ks)
    @printf("  QP Γ depth (2s): %+.3f eV\n", depth_qp)
    @printf("  QP/KS ratio:     %.4f\n", depth_qp / depth_ks)
    @printf("  Narrowing:        %.1f%%\n", (1 - depth_qp/depth_ks) * 100)

    return PSPResult(name, Zion, εF,
                     eigenvalues, eigenvalues_qp, deltas, k_coordinates,
                     kdistances, eigenvalues_arr, eqp_arr,
                     tick_distances, tick_labels, n_spin,
                     conduction_band, depth_ks, depth_qp, bw_ks, bw_qp)
end

# Process all 3 PSPs
gth_file = joinpath(outdir, "li_ks_data.jld2")
eft_file = joinpath(outdir, "li_eft_ks_data.jld2")
pbe_file = joinpath(outdir, "li_pbe_ks_data.jld2")

results = Dict{String, PSPResult}()
results["gth"] = process_psp("GTH LDA (Zion=1)", gth_file; conduction_band=1)
results["eft"] = process_psp("EFT nonlocal (Zion=1)", eft_file; conduction_band=1)
results["pbe"] = process_psp("PBE (Zion=3)", pbe_file; conduction_band=2)

# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("  LITHIUM 3-PSP COMPARISON (2s band, eV relative to εF)")
println("="^70)

@printf("  %-22s  %10s  %10s  %10s  %10s  %8s\n",
        "", "KS Γ(eV)", "QP Γ(eV)", "BW_KS(eV)", "BW_QP(eV)", "QP/KS")
println("  " * "-"^76)
for key in ["gth", "eft", "pbe"]
    r = results[key]
    @printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
            r.name, r.depth_ks, r.depth_qp, r.bw_ks, r.bw_qp,
            r.depth_qp / r.depth_ks)
end

# High-symmetry point comparison
hs_points = [
    ("Γ",  [0.0, 0.0, 0.0]),
    ("H",  [0.5, -0.5, 0.5]),
    ("N",  [0.0, 0.0, 0.5]),
    ("P",  [0.25, 0.25, 0.25]),
]

println("\n  Band-by-band at high-symmetry points (2s band, eV):")
@printf("  %-3s  %12s  %12s  %12s  %12s  %12s  %12s\n",
        "", "GTH KS", "GTH QP", "EFT KS", "EFT QP", "PBE KS", "PBE QP")
println("  " * "-"^80)
for (label, coord) in hs_points
    @printf("  %-3s", label)
    for key in ["gth", "eft", "pbe"]
        r = results[key]
        ik = find_kpoint(r.k_coordinates, coord)
        if !isnothing(ik)
            vb = r.valence_band
            e_ks = (r.eigenvalues[ik][vb] - r.εF) * Ha_to_eV
            e_qp = (r.eigenvalues_qp[ik][vb] - r.εF) * Ha_to_eV
            @printf("  %+10.3f    %+10.3f  ", e_ks, e_qp)
        end
    end
    println()
end

# Free electron reference
lattice_matrix = load(gth_file, "lattice_matrix")
V_prim = abs(det(lattice_matrix))
n_e = 1  # Z_val for GTH
k_F = (3π^2 * n_e / V_prim)^(1/3)
E_F_free = k_F^2 / 2
@printf("\n  Free electron Γ depth: %+.3f eV\n", -E_F_free * Ha_to_eV)

# ══════════════════════════════════════════════════════════════════════════════
# Save form factor and Δ(K) data
# Δ_FE(K) = |f(K)/ΔE|² = free-electron coherent correction at momentum K
# ══════════════════════════════════════════════════════════════════════════════

K_grid = collect(range(0.0, 20.0, length=2000))
fc_values = [eft_f_K(K) for K in K_grid]
delta_total = [eft_f_K(K)^2 / LI_ΔE^2 for K in K_grid]

# Also compute f_c(K) from DFT-LDA atomic solver (same method as Na/K/... elements)
# for consistency in cross-element plots (delta_K_all.pdf)
println("\n=== Computing DFT-LDA form factor (for cross-element plot) ===")
include(joinpath(@__DIR__, "..", "sodium", "atomic_hf.jl"))

atom_core = solve_atom(3, [(1,0,2)]; dr=0.002, r_max=40.0)
atom_hole = solve_atom(3, [(1,0,1)]; dr=0.002, r_max=40.0)
ΔE_dft = atom_hole.E_total - atom_core.E_total

atom_full = solve_atom(3, [(1,0,2), (2,0,1)]; dr=0.002, r_max=40.0)
orb_1s = atom_full.orbitals[1]
rgrid_a = atom_full.rgrid
dr_a = atom_full.dr
N_a = length(rgrid_a)
V_H_a = orbital_coulomb_potential(orb_1s.u, 1.0, rgrid_a, dr_a)
J_a = dr_a * sum(orb_1s.u[i]^2 * V_H_a[i] for i in 1:N_a)

function dft_f_K_li(K::Float64)
    if K < 1e-10
        return sqrt(4π) * dr_a * sum(orb_1s.u[i] * (V_H_a[i] - J_a) * rgrid_a[i] for i in 1:N_a)
    end
    sqrt(4π) / K * dr_a * sum(orb_1s.u[i] * (V_H_a[i] - J_a) * sin(K * rgrid_a[i]) for i in 1:N_a)
end

fc_dft_values = [dft_f_K_li(K) for K in K_grid]
@printf("  ΔE (ΔSCF/DFT-LDA) = %.4f Ha\n", ΔE_dft)
@printf("  f(0)/ΔE: analytic(var)=%.4f, DFT-LDA=%.4f\n",
        eft_f_K(0.0)/LI_ΔE, dft_f_K_li(0.0)/ΔE_dft)

# ══════════════════════════════════════════════════════════════════════════════
# Save combined QP data
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Saving QP data ===")

save_dict = Dict{String, Any}()
for (key, r) in results
    save_dict["$(key)_eigenvalues_qp"] = r.eigenvalues_qp
    save_dict["$(key)_deltas"] = r.deltas
    save_dict["$(key)_eqp_array"] = r.eqp_array
    save_dict["$(key)_eigenvalues_array"] = r.eigenvalues_array
    save_dict["$(key)_kdistances"] = r.kdistances
    save_dict["$(key)_tick_distances"] = r.tick_distances
    save_dict["$(key)_tick_labels"] = r.tick_labels
    save_dict["$(key)_εF"] = r.εF
    save_dict["$(key)_Zion"] = r.Zion
    save_dict["$(key)_n_spin"] = r.n_spin
    save_dict["$(key)_valence_band"] = r.valence_band
    save_dict["$(key)_depth_ks"] = r.depth_ks
    save_dict["$(key)_depth_qp"] = r.depth_qp
    save_dict["$(key)_bw_ks"] = r.bw_ks
    save_dict["$(key)_bw_qp"] = r.bw_qp
end

save_dict["K_grid"] = K_grid
save_dict["fc_1s"] = fc_values
save_dict["delta_total"] = delta_total
save_dict["delta_K0"] = eft_f_K(0.0)^2 / LI_ΔE^2
save_dict["f0_1s"] = eft_f_K(0.0)
save_dict["ΔE_1s"] = LI_ΔE
save_dict["method"] = "coherent_analytic"
save_dict["psp_names"] = ["gth", "eft", "pbe"]
# DFT-LDA form factor (same method as Na/K/.., for cross-element plots)
save_dict["channel_names"] = ["1s"]
save_dict["channel_ΔE"] = [ΔE_dft]
save_dict["fc_per_channel"] = Dict("1s" => fc_dft_values)

outfile = joinpath(outdir, "li_qp_data.jld2")
jldopen(outfile, "w") do f
    for (k, v) in save_dict
        f[k] = v
    end
end
println("→ $outfile")

# Summary file
open(joinpath(outdir, "li_summary.txt"), "w") do io
    println(io, "# Li EFT QP correction summary (3-PSP comparison)")
    println(io, "# Generated by lithium/freq_correction_new.jl")
    println(io)
    @printf(io, "Z = 3, Structure: BCC, a = 6.632 Bohr\n")
    @printf(io, "Coherent formula: Δ(nk) = |Σ_G c(G) f(|k+G|) / ΔE|²\n")
    @printf(io, "Single 1s channel: f(K) = 8√π/√α · A_K (analytic)\n")
    @printf(io, "f(0) = %.6f, ΔE = %.4f Ha\n", eft_f_K(0.0), LI_ΔE)
    @printf(io, "Δ_FE(K=0) = |f(0)/ΔE|² = %.6f\n", eft_f_K(0.0)^2 / LI_ΔE^2)
    println(io)
    @printf(io, "%-22s  %10s  %10s  %10s  %10s  %8s\n",
            "", "KS Γ(eV)", "QP Γ(eV)", "BW_KS(eV)", "BW_QP(eV)", "QP/KS")
    for key in ["gth", "eft", "pbe"]
        r = results[key]
        @printf(io, "%-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
                r.name, r.depth_ks, r.depth_qp, r.bw_ks, r.bw_qp,
                r.depth_qp / r.depth_ks)
    end
end
println("→ li_summary.txt")

println("\n=== Done ===")
