"""
freq_correction_pbe.jl (sodium)
Load LDA (Zion=1) and PBE (Zion=9) KS data, apply EFT coherent QP correction, compare.

Na LDA uses Zion=1 (largecore): all core frozen, QP applied to all bands.
Na PBE uses Zion=9 (smallcore): 2s²2p⁶ are valence, QP applied ONLY to
  conduction band (band 5 = 3s) and above. Semicore bands 1-4 left unchanged.

This is a robustness test showing the EFT QP works even with semicore states.

Usage: julia --project=. sodium/freq_correction_pbe.jl
Prerequisites:
  julia --project=. sodium/run_ks.jl       (saves na_ks_data.jld2)
  julia --project=. sodium/run_ks_pbe.jl   (saves na_pbe_ks_data.jld2)
"""

if "--dry-run" in ARGS
    println("[dry-run] sodium/freq_correction_pbe.jl")
    println("  Loads: sodium/na_ks_data.jld2, sodium/na_pbe_ks_data.jld2")
    println("  Computes: atomic ΔSCF + form factors + coherent QP correction (LDA + PBE)")
    println("  Saves: sodium/na_pbe_qp_data.jld2")
    exit(0)
end

using Printf
using LinearAlgebra
using JLD2
using Interpolations: linear_interpolation

include(joinpath(@__DIR__, "atomic_hf.jl"))

outdir = @__DIR__

# ══════════════════════════════════════════════════════════════════════════════
# Load KS data (both LDA and PBE)
# ══════════════════════════════════════════════════════════════════════════════

println("=== Loading KS data ===")

lda_file = joinpath(outdir, "na_ks_data.jld2")
pbe_file = joinpath(outdir, "na_pbe_ks_data.jld2")

println("  LDA: $lda_file")
lda = load(lda_file)
@printf("  %d k-points, %d bands, Zion=%d, εF=%.6f Ha\n",
        length(lda["psi"]), length(lda["eigenvalues"][1]),
        lda["Zion"], lda["εF"])

println("  PBE: $pbe_file")
pbe = load(pbe_file)
@printf("  %d k-points, %d bands, Zion=%d, εF=%.6f Ha\n",
        length(pbe["psi"]), length(pbe["eigenvalues"][1]),
        pbe["Zion"], pbe["εF"])

conduction_band = pbe["conduction_band"]
@printf("  PBE conduction band index: %d\n", conduction_band)

# ══════════════════════════════════════════════════════════════════════════════
# Atomic solver: core excitation energies
# ══════════════════════════════════════════════════════════════════════════════

const NA_Z_NUC = 11
const NA_CORE_CONFIG = [(1,0,2), (2,0,2), (2,1,6)]
const NA_FULL_CONFIG = [(1,0,2), (2,0,2), (2,1,6), (3,0,1)]

println("\n=== Atomic solver: core excitation energies ===")

dr_atom = 0.002; r_max_atom = 40.0

atom_core = solve_atom(NA_Z_NUC, NA_CORE_CONFIG; dr=dr_atom, r_max=r_max_atom)
E_core = atom_core.E_total

atom_hole_2s = solve_atom(NA_Z_NUC, [(1,0,2),(2,0,1),(2,1,6)]; dr=dr_atom, r_max=r_max_atom)
ΔE_2s = atom_hole_2s.E_total - E_core

atom_hole_1s = solve_atom(NA_Z_NUC, [(1,0,1),(2,0,2),(2,1,6)]; dr=dr_atom, r_max=r_max_atom)
ΔE_1s = atom_hole_1s.E_total - E_core

@printf("  ΔE_1s = %.4f Ha (1/ΔE² = %.2e)\n", ΔE_1s, 1/ΔE_1s^2)
@printf("  ΔE_2s = %.4f Ha (1/ΔE² = %.2e)\n", ΔE_2s, 1/ΔE_2s^2)

# ══════════════════════════════════════════════════════════════════════════════
# Form factors
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Form factors ===")

atom_full = solve_atom(NA_Z_NUC, [NA_FULL_CONFIG...]; dr=dr_atom, r_max=r_max_atom)
rgrid = atom_full.rgrid
dr = atom_full.dr
N = length(rgrid)

struct FormFactor
    name::String
    u_c::Vector{Float64}
    V_H_c::Vector{Float64}
    J_c::Float64
    f0::Float64
    ΔE::Float64
end

form_factors = FormFactor[]
ΔE_map = Dict(1 => ΔE_1s, 2 => ΔE_2s)

for orb in atom_full.orbitals[1:end-1]  # skip valence 3s
    orb.l != 0 && continue
    V_H_c = orbital_coulomb_potential(orb.u, 1.0, rgrid, dr)
    J_c = dr * sum(orb.u[i]^2 * V_H_c[i] for i in 1:N)
    S_c = dr * sum(rgrid[i] * orb.u[i] for i in 1:N)
    W_c = dr * sum(rgrid[i] * orb.u[i] * V_H_c[i] for i in 1:N)
    f0 = sqrt(4π) * (W_c - J_c * S_c)
    ΔE = ΔE_map[orb.n]
    push!(form_factors, FormFactor("$(orb.n)s", copy(orb.u), V_H_c, J_c, f0, ΔE))
    @printf("  %s: f(0)=%.6f, ΔE=%.4f Ha, Δ_c(0)=%.6f\n",
            "$(orb.n)s", f0, ΔE, f0^2/ΔE^2)
end

function eval_fc(ff::FormFactor, K::Float64)
    K < 1e-10 && return ff.f0
    s = 0.0
    for i in 1:N
        s += ff.u_c[i] * (ff.V_H_c[i] - ff.J_c) * sin(K * rgrid[i]) * dr
    end
    return sqrt(4π) / K * s
end

const K_MAX = 20.0
const N_INTERP = 2000
const K_grid_interp = collect(range(0.0, K_MAX, length=N_INTERP))
fc_interp = [linear_interpolation(K_grid_interp,
    [eval_fc(ff, Float64(K)) for K in K_grid_interp]) for ff in form_factors]

function delta_K(K::Float64)
    Δ = 0.0
    for ff in form_factors
        Δ += eval_fc(ff, K) * ff.f0 / ff.ΔE^2
    end
    return Δ
end

@printf("\n  Δ(K=0) = %.6f\n", delta_K(0.0))

# ══════════════════════════════════════════════════════════════════════════════
# Coherent QP correction
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Applying coherent QP correction ===")

function delta_nk(ψnk::AbstractVector, k_frac, Gvecs_int, recip_lat)
    Δ_total = 0.0
    for (ic, ff) in enumerate(form_factors)
        F_c = complex(0.0)
        for ig in axes(Gvecs_int, 1)
            G = Gvecs_int[ig, :]
            K = norm(recip_lat * (k_frac .+ G))
            F_c += ψnk[ig] * fc_interp[ic](min(K, K_MAX - 0.01)) / abs(ff.ΔE)
        end
        Δ_total += abs2(F_c)
    end
    return Δ_total
end

Ha_to_eV = 27.211386245988

# --- LDA: apply QP to ALL bands ---
println("\n  LDA Zion=1: applying QP to all bands...")
n_kpts_lda = length(lda["psi"])
n_bands_lda = length(lda["eigenvalues"][1])
εF_lda = lda["εF"]

eigenvalues_qp_lda = [similar(lda["eigenvalues"][ik]) for ik in 1:n_kpts_lda]
deltas_lda = [zeros(n_bands_lda) for _ in 1:n_kpts_lda]

t0 = time()
for ik in 1:n_kpts_lda
    k_frac = lda["k_coordinates"][ik]
    Gvecs_int = lda["G_vectors"][ik]
    for n in 1:n_bands_lda
        ψnk = lda["psi"][ik][:, n]
        Δ = delta_nk(ψnk, k_frac, Gvecs_int, lda["recip_lattice"])
        deltas_lda[ik][n] = Δ
        eigenvalues_qp_lda[ik][n] = εF_lda + (lda["eigenvalues"][ik][n] - εF_lda) / (1.0 + Δ)
    end
end
@printf("  LDA QP done in %.1f s\n", time() - t0)

# --- PBE: apply QP ONLY to conduction band and above ---
@printf("  PBE Zion=9: applying QP to bands %d+ only...\n", conduction_band)
n_kpts_pbe = length(pbe["psi"])
n_bands_pbe = length(pbe["eigenvalues"][1])
εF_pbe = pbe["εF"]

eigenvalues_qp_pbe = [copy(pbe["eigenvalues"][ik]) for ik in 1:n_kpts_pbe]
deltas_pbe = [zeros(n_bands_pbe) for _ in 1:n_kpts_pbe]

t1 = time()
for ik in 1:n_kpts_pbe
    k_frac = pbe["k_coordinates"][ik]
    Gvecs_int = pbe["G_vectors"][ik]
    for n in conduction_band:n_bands_pbe
        ψnk = pbe["psi"][ik][:, n]
        Δ = delta_nk(ψnk, k_frac, Gvecs_int, pbe["recip_lattice"])
        deltas_pbe[ik][n] = Δ
        eigenvalues_qp_pbe[ik][n] = εF_pbe + (pbe["eigenvalues"][ik][n] - εF_pbe) / (1.0 + Δ)
    end
    # Bands 1 to conduction_band-1 are semicore: leave eigenvalues unchanged, Δ=0
end
@printf("  PBE QP done in %.1f s\n", time() - t1)

# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Results ===\n")

function find_kpoint(k_coords, target; tol=0.02)
    for (ik, kc) in enumerate(k_coords)
        norm(kc .- target) < tol && return ik
    end
    return nothing
end

hs_points = [
    ("Γ",  [0.0, 0.0, 0.0]),
    ("H",  [0.5, -0.5, 0.5]),
    ("N",  [0.0, 0.0, 0.5]),
    ("P",  [0.25, 0.25, 0.25]),
]

println("--- LDA Zion=1 (band 1 = 3s conduction) ---")
for (label, coord) in hs_points
    ik = find_kpoint(lda["k_coordinates"], coord)
    isnothing(ik) && continue
    e_ks = (lda["eigenvalues"][ik][1] - εF_lda) * Ha_to_eV
    e_qp = (eigenvalues_qp_lda[ik][1] - εF_lda) * Ha_to_eV
    Δ = deltas_lda[ik][1]
    @printf("  %s: KS=%+.3f eV, QP=%+.3f eV, Δ=%.4f\n", label, e_ks, e_qp, Δ)
end

println("\n--- PBE Zion=9 (band $conduction_band = 3s conduction) ---")
for (label, coord) in hs_points
    ik = find_kpoint(pbe["k_coordinates"], coord)
    isnothing(ik) && continue
    e_ks = (pbe["eigenvalues"][ik][conduction_band] - εF_pbe) * Ha_to_eV
    e_qp = (eigenvalues_qp_pbe[ik][conduction_band] - εF_pbe) * Ha_to_eV
    Δ = deltas_pbe[ik][conduction_band]
    @printf("  %s: KS=%+.3f eV, QP=%+.3f eV, Δ=%.4f\n", label, e_ks, e_qp, Δ)
end

# Conduction band statistics
println("\n=== Conduction band summary ===\n")

function conduction_stats(evals, eqp, εF, band_idx)
    e_ks = [(evals[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(evals)]
    e_qp = [(eqp[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(eqp)]
    depth_ks = minimum(e_ks)
    depth_qp = minimum(e_qp)
    bw_ks = maximum(e_ks) - minimum(e_ks)
    bw_qp = maximum(e_qp) - minimum(e_qp)
    return (; depth_ks, depth_qp, bw_ks, bw_qp)
end

s_lda = conduction_stats(lda["eigenvalues"], eigenvalues_qp_lda, εF_lda, 1)
s_pbe = conduction_stats(pbe["eigenvalues"], eigenvalues_qp_pbe, εF_pbe, conduction_band)

lattice_matrix = lda["lattice_matrix"]
V_prim = abs(det(lattice_matrix))
n_e_free = 1  # free-electron with 1 valence electron
k_F = (3π^2 * n_e_free / V_prim)^(1/3)
E_F_free = k_F^2 / 2

@printf("  %-22s  %10s  %10s  %10s  %10s  %8s\n",
        "", "KS Γ(eV)", "QP Γ(eV)", "BW_KS(eV)", "BW_QP(eV)", "QP/KS")
println("  " * "-"^76)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "LDA Zion=1 (band 1)", s_lda.depth_ks, s_lda.depth_qp,
        s_lda.bw_ks, s_lda.bw_qp, s_lda.depth_qp / s_lda.depth_ks)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "PBE Zion=9 (band $conduction_band)", s_pbe.depth_ks, s_pbe.depth_qp,
        s_pbe.bw_ks, s_pbe.bw_qp, s_pbe.depth_qp / s_pbe.depth_ks)
@printf("  %-22s  %+10.3f\n", "Free electron", -E_F_free * Ha_to_eV)

narrowing_lda = (1 - s_lda.depth_qp / s_lda.depth_ks) * 100
narrowing_pbe = (1 - s_pbe.depth_qp / s_pbe.depth_ks) * 100
@printf("\n  Narrowing: LDA=%.1f%%, PBE=%.1f%%\n", narrowing_lda, narrowing_pbe)

iΓ_lda = find_kpoint(lda["k_coordinates"], [0.0, 0.0, 0.0])
iΓ_pbe = find_kpoint(pbe["k_coordinates"], [0.0, 0.0, 0.0])
if !isnothing(iΓ_lda) && !isnothing(iΓ_pbe)
    @printf("  Δ(Γ): LDA=%.4f, PBE=%.4f\n",
            deltas_lda[iΓ_lda][1], deltas_pbe[iΓ_pbe][conduction_band])
end

println("\n  Channel breakdown at K=0:")
for ff in form_factors
    Δc = ff.f0^2 / ff.ΔE^2
    @printf("    %s: Δ_c(0) = %.6f  (%.1f%%)\n", ff.name, Δc, 100Δc/delta_K(0.0))
end

# ══════════════════════════════════════════════════════════════════════════════
# Save QP data
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Saving QP data ===")

# Build plotting arrays for both
function build_qp_arrays(ks_data, eqp, deltas_vec)
    n_spin = ks_data["n_spin"]
    krange_spin_map = ks_data["krange_spin_map"]
    eigenvalues_arr = ks_data["eigenvalues_array"]

    eqp_arr = similar(eigenvalues_arr)
    deltas_arr = similar(eigenvalues_arr)
    for σ in 1:n_spin
        for (ito, ik) in enumerate(krange_spin_map[σ])
            eqp_arr[ito, :, σ] = eqp[ik]
            deltas_arr[ito, :, σ] = deltas_vec[ik]
        end
    end
    return eqp_arr, deltas_arr
end

eqp_arr_lda, deltas_arr_lda = build_qp_arrays(lda, eigenvalues_qp_lda, deltas_lda)
eqp_arr_pbe, deltas_arr_pbe = build_qp_arrays(pbe, eigenvalues_qp_pbe, deltas_pbe)

# Precomputed Δ(K) grid
K_grid_save = collect(range(0.0, 20.0, length=2000))
Δ_total = [delta_K(K) for K in K_grid_save]
Δ_per_channel = Dict{String,Vector{Float64}}()
fc_per_channel = Dict{String,Vector{Float64}}()
for (ic, ff) in enumerate(form_factors)
    Δ_per_channel[ff.name] = [eval_fc(ff, K) * ff.f0 / ff.ΔE^2 for K in K_grid_save]
    fc_per_channel[ff.name] = [eval_fc(ff, K) for K in K_grid_save]
end

outfile = joinpath(outdir, "na_pbe_qp_data.jld2")
jldsave(outfile;
    # LDA results
    eigenvalues_qp_lda, deltas_lda,
    eqp_arr_lda, deltas_arr_lda,
    εF_lda,
    depth_ks_lda = s_lda.depth_ks, depth_qp_lda = s_lda.depth_qp,
    bw_ks_lda = s_lda.bw_ks, bw_qp_lda = s_lda.bw_qp,
    # PBE results
    eigenvalues_qp_pbe, deltas_pbe,
    eqp_arr_pbe, deltas_arr_pbe,
    εF_pbe, conduction_band,
    depth_ks_pbe = s_pbe.depth_ks, depth_qp_pbe = s_pbe.depth_qp,
    bw_ks_pbe = s_pbe.bw_ks, bw_qp_pbe = s_pbe.bw_qp,
    # Shared atomic data
    K_grid = K_grid_save,
    delta_total = Δ_total,
    delta_per_channel = Δ_per_channel,
    fc_per_channel = fc_per_channel,
    channel_names = [ff.name for ff in form_factors],
    channel_f0 = [ff.f0 for ff in form_factors],
    channel_J_c = [ff.J_c for ff in form_factors],
    channel_ΔE = [ff.ΔE for ff in form_factors],
    delta_K0 = delta_K(0.0),
    method = "coherent",
)
println("→ $outfile")

println("\n=== Done ===")
