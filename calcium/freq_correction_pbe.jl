"""
freq_correction_pbe.jl (calcium)
Stage 2: Load BOTH LDA and PBE KS data and apply coherent EFT frequency correction.

Demonstrates robustness: LDA uses Zion=2 ([Ar] core), PBE uses Zion=10 ([Ne] core,
semicore 3s²3p⁶ treated as valence). The QP correction uses the same [Ar]-core
EFT theory for both — but for PBE Zion=10, it is applied ONLY to conduction bands
(band 5+ = 4s and above), leaving semicore bands 1-4 (3s, 3p) unchanged.

Ca core (EFT theory): 1s² 2s² 2p⁶ 3s² 3p⁶ (18 electrons, [Ar])
Ca valence (LDA):      4s² (Zion=2)
Ca valence (PBE):      3s² 3p⁶ 4s² (Zion=10)

Usage: julia --project=. calcium/freq_correction_pbe.jl
Prerequisites:
  julia --project=. calcium/run_ks.jl       (LDA, saves ca_ks_data.jld2)
  julia --project=. calcium/run_ks_pbe.jl   (PBE, saves ca_pbe_ks_data.jld2)
"""

if "--dry-run" in ARGS
    println("[dry-run] calcium/freq_correction_pbe.jl")
    println("  Loads: calcium/ca_ks_data.jld2, calcium/ca_pbe_ks_data.jld2")
    println("  Computes: atomic ΔSCF + form factors + coherent QP correction (both PSPs)")
    println("  Saves: calcium/ca_pbe_qp_data.jld2")
    exit(0)
end

using Printf
using LinearAlgebra
using JLD2
using Interpolations: linear_interpolation

# Reuse the atomic solver from sodium
include(joinpath(@__DIR__, "..", "sodium", "atomic_hf.jl"))

outdir = @__DIR__

# ══════════════════════════════════════════════════════════════════════════════
# Load KS data (both LDA and PBE)
# ══════════════════════════════════════════════════════════════════════════════

println("=== Loading KS data ===\n")

lda_file = joinpath(outdir, "ca_ks_data.jld2")
pbe_file = joinpath(outdir, "ca_pbe_ks_data.jld2")

println("  LDA: $lda_file")
lda = load(lda_file)
@printf("    %d k-points, %d bands, Zion=%d, εF=%.6f Ha\n",
        length(lda["psi"]), length(lda["eigenvalues"][1]), lda["Zion"], lda["εF"])

println("  PBE: $pbe_file")
pbe = load(pbe_file)
@printf("    %d k-points, %d bands, Zion=%d, εF=%.6f Ha\n",
        length(pbe["psi"]), length(pbe["eigenvalues"][1]), pbe["Zion"], pbe["εF"])

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Core excitation energies via ΔSCF
# ══════════════════════════════════════════════════════════════════════════════

const CA_Z_NUC = 20
const CA_CORE_CONFIG = [(1,0,2), (2,0,2), (2,1,6), (3,0,2), (3,1,6)]
const CA_FULL_CONFIG = [(1,0,2), (2,0,2), (2,1,6), (3,0,2), (3,1,6), (4,0,2)]

println("\n=== Step 1: Core excitation energies (ΔSCF) ===\n")

dr_atom = 0.002; r_max_atom = 40.0

atom_core = solve_atom(CA_Z_NUC, CA_CORE_CONFIG; dr=dr_atom, r_max=r_max_atom)
E_core = atom_core.E_total

atom_hole_3s = solve_atom(CA_Z_NUC, [(1,0,2),(2,0,2),(2,1,6),(3,0,1),(3,1,6)]; dr=dr_atom, r_max=r_max_atom)
ΔE_3s = atom_hole_3s.E_total - E_core

atom_hole_2s = solve_atom(CA_Z_NUC, [(1,0,2),(2,0,1),(2,1,6),(3,0,2),(3,1,6)]; dr=dr_atom, r_max=r_max_atom)
ΔE_2s = atom_hole_2s.E_total - E_core

atom_hole_1s = solve_atom(CA_Z_NUC, [(1,0,1),(2,0,2),(2,1,6),(3,0,2),(3,1,6)]; dr=dr_atom, r_max=r_max_atom)
ΔE_1s = atom_hole_1s.E_total - E_core

@printf("  ΔE_1s = %.4f Ha (1/ΔE² = %.2e)\n", ΔE_1s, 1/ΔE_1s^2)
@printf("  ΔE_2s = %.4f Ha (1/ΔE² = %.2e)\n", ΔE_2s, 1/ΔE_2s^2)
@printf("  ΔE_3s = %.4f Ha (1/ΔE² = %.2e)\n", ΔE_3s, 1/ΔE_3s^2)

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Form factors f_c(K)
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Step 2: Form factors ===\n")

atom_full = solve_atom(CA_Z_NUC, [CA_FULL_CONFIG...]; dr=dr_atom, r_max=r_max_atom)
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
ΔE_map = Dict(1 => ΔE_1s, 2 => ΔE_2s, 3 => ΔE_3s)

for orb in atom_full.orbitals[1:end-1]  # skip valence 4s
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

# Precompute per-channel interpolation
const K_MAX = 20.0
const N_INTERP = 2000
const K_grid_interp = collect(range(0.0, K_MAX, length=N_INTERP))
fc_interp = [linear_interpolation(K_grid_interp,
    [eval_fc(ff, Float64(K)) for K in K_grid_interp]) for ff in form_factors]

# Diagonal Δ(K) for reference
function delta_K(K::Float64)
    Δ = 0.0
    for ff in form_factors
        Δ += eval_fc(ff, K) * ff.f0 / ff.ΔE^2
    end
    return Δ
end

@printf("\n  Δ(K=0) = %.6f\n", delta_K(0.0))

println("\n  Channel breakdown at K=0:")
for ff in form_factors
    Δc = ff.f0^2 / ff.ΔE^2
    @printf("    %s: Δ_c(0) = %.6f  (%.1f%%)\n", ff.name, Δc, 100Δc/delta_K(0.0))
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Coherent QP correction
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Step 3: Applying coherent QP correction ===\n")

# Coherent (exact) Δ(nk) = Σ_c |Σ_G c(G) f_c(|k+G|) / ΔE_c|²
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

function apply_qp_correction(ks_data; band_range=nothing)
    psi           = ks_data["psi"]
    eigenvalues   = ks_data["eigenvalues"]
    k_coordinates = ks_data["k_coordinates"]
    G_vectors     = ks_data["G_vectors"]
    recip_lattice = ks_data["recip_lattice"]
    εF            = ks_data["εF"]

    n_kpts = length(psi)
    n_bands = length(eigenvalues[1])
    if isnothing(band_range)
        band_range = 1:n_bands
    end

    eigenvalues_qp = deepcopy(eigenvalues)
    deltas = [zeros(n_bands) for _ in 1:n_kpts]

    for ik in 1:n_kpts
        k_frac = k_coordinates[ik]
        Gvecs_int = G_vectors[ik]
        for n in band_range
            ψnk = psi[ik][:, n]
            Δ = delta_nk(ψnk, k_frac, Gvecs_int, recip_lattice)
            deltas[ik][n] = Δ
            eigenvalues_qp[ik][n] = εF + (eigenvalues[ik][n] - εF) / (1.0 + Δ)
        end
    end
    return eigenvalues_qp, deltas
end

# LDA Zion=2: apply QP to all bands
println("  LDA Zion=2: applying QP to all bands...")
t0 = time()
eqp_lda, deltas_lda = apply_qp_correction(lda)
@printf("  Done in %.1f s\n", time() - t0)

# PBE Zion=10: apply QP ONLY to conduction bands (band 5+ = 4s)
# Bands 1-4: 3s + 3p semicore → no QP correction
conduction_band = pbe["conduction_band"]
n_bands_pbe = length(pbe["eigenvalues"][1])
println("\n  PBE Zion=10: applying QP to bands $conduction_band:$n_bands_pbe only...")
t1 = time()
eqp_pbe, deltas_pbe = apply_qp_correction(pbe; band_range=conduction_band:n_bands_pbe)
@printf("  Done in %.1f s\n", time() - t1)

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

# High-symmetry points (FCC)
hs_points = [
    ("Γ",  [0.0, 0.0, 0.0]),
    ("X",  [0.5, 0.0, 0.5]),
    ("L",  [0.5, 0.5, 0.5]),
    ("W",  [0.5, 0.25, 0.75]),
    ("K",  [0.375, 0.375, 0.75]),
]

println("--- LDA Zion=2 (band 1 = 4s conduction) ---")
for (label, coord) in hs_points
    ik = find_kpoint(lda["k_coordinates"], coord)
    isnothing(ik) && continue
    e_ks = (lda["eigenvalues"][ik][1] - lda["εF"]) * Ha_to_eV
    e_qp = (eqp_lda[ik][1] - lda["εF"]) * Ha_to_eV
    Δ = deltas_lda[ik][1]
    @printf("  %s: KS=%+.3f eV, QP=%+.3f eV, Δ=%.4f\n", label, e_ks, e_qp, Δ)
end

println("\n--- PBE Zion=10 (band $conduction_band = 4s conduction) ---")
for (label, coord) in hs_points
    ik = find_kpoint(pbe["k_coordinates"], coord)
    isnothing(ik) && continue
    e_ks = (pbe["eigenvalues"][ik][conduction_band] - pbe["εF"]) * Ha_to_eV
    e_qp = (eqp_pbe[ik][conduction_band] - pbe["εF"]) * Ha_to_eV
    Δ = deltas_pbe[ik][conduction_band]
    @printf("  %s: KS=%+.3f eV, QP=%+.3f eV, Δ=%.4f\n", label, e_ks, e_qp, Δ)
end

# Conduction band statistics
function conduction_stats(eigenvalues, eqp, εF, band_idx)
    e_ks = [(eigenvalues[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(eigenvalues)]
    e_qp = [(eqp[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(eqp)]
    depth_ks = minimum(e_ks)
    depth_qp = minimum(e_qp)
    bw_ks = maximum(e_ks) - minimum(e_ks)
    bw_qp = maximum(e_qp) - minimum(e_qp)
    return (; depth_ks, depth_qp, bw_ks, bw_qp)
end

s_lda = conduction_stats(lda["eigenvalues"], eqp_lda, lda["εF"], 1)
s_pbe = conduction_stats(pbe["eigenvalues"], eqp_pbe, pbe["εF"], conduction_band)

a = lda["a_bohr"]
V_prim = a^3 / 4  # FCC primitive cell volume
n_e = 2  # Z_val=2 for free electron (physical valence)
k_F = (3π^2 * n_e / V_prim)^(1/3)
E_F_free = k_F^2 / 2

println("\n=== Conduction band summary ===\n")
@printf("  %-22s  %10s  %10s  %10s  %10s  %8s\n",
        "", "KS Γ(eV)", "QP Γ(eV)", "BW_KS(eV)", "BW_QP(eV)", "QP/KS")
println("  " * "-"^76)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "LDA Zion=2", s_lda.depth_ks, s_lda.depth_qp,
        s_lda.bw_ks, s_lda.bw_qp, s_lda.depth_qp / s_lda.depth_ks)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "PBE Zion=10", s_pbe.depth_ks, s_pbe.depth_qp,
        s_pbe.bw_ks, s_pbe.bw_qp, s_pbe.depth_qp / s_pbe.depth_ks)
@printf("  %-22s  %+10.3f\n", "Free electron", -E_F_free * Ha_to_eV)
@printf("  %-22s  %10s\n", "Expt (ARPES)", "3.30")
@printf("  %-22s  %10s\n", "eDMFT (Mandal)", "3.24")

narrowing_lda = (1 - s_lda.depth_qp / s_lda.depth_ks) * 100
narrowing_pbe = (1 - s_pbe.depth_qp / s_pbe.depth_ks) * 100
@printf("\n  Narrowing: LDA=%.1f%%, PBE=%.1f%%\n", narrowing_lda, narrowing_pbe)

iΓ_lda = find_kpoint(lda["k_coordinates"], [0.0, 0.0, 0.0])
iΓ_pbe = find_kpoint(pbe["k_coordinates"], [0.0, 0.0, 0.0])
if !isnothing(iΓ_lda) && !isnothing(iΓ_pbe)
    @printf("  Δ(Γ): LDA=%.4f, PBE=%.4f\n",
            deltas_lda[iΓ_lda][1], deltas_pbe[iΓ_pbe][conduction_band])
end

# ══════════════════════════════════════════════════════════════════════════════
# Save QP data
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Saving QP data ===")

# Build plotting arrays for both
function build_plot_arrays(ks_data, eqp, deltas)
    kdistances      = ks_data["kdistances"]
    eigenvalues_arr = ks_data["eigenvalues_array"]
    n_spin          = ks_data["n_spin"]
    krange_spin_map = ks_data["krange_spin_map"]

    eqp_arr = similar(eigenvalues_arr)
    deltas_arr = similar(eigenvalues_arr)
    for σ in 1:n_spin
        for (ito, ik) in enumerate(krange_spin_map[σ])
            eqp_arr[ito, :, σ] = eqp[ik]
            deltas_arr[ito, :, σ] = deltas[ik]
        end
    end
    return eqp_arr, deltas_arr
end

eqp_arr_lda, deltas_arr_lda = build_plot_arrays(lda, eqp_lda, deltas_lda)
eqp_arr_pbe, deltas_arr_pbe = build_plot_arrays(pbe, eqp_pbe, deltas_pbe)

# Δ(K) on a fine grid
K_grid_save = collect(range(0.0, 20.0, length=2000))
Δ_total = [delta_K(K) for K in K_grid_save]
Δ_per_channel = Dict{String,Vector{Float64}}()
fc_per_channel = Dict{String,Vector{Float64}}()
for (ic, ff) in enumerate(form_factors)
    Δ_per_channel[ff.name] = [eval_fc(ff, K) * ff.f0 / ff.ΔE^2 for K in K_grid_save]
    fc_per_channel[ff.name] = [eval_fc(ff, K) for K in K_grid_save]
end

outfile = joinpath(outdir, "ca_pbe_qp_data.jld2")
jldsave(outfile;
    # LDA QP
    eqp_lda, deltas_lda, eqp_arr_lda, deltas_arr_lda,
    lda_depth_ks = s_lda.depth_ks, lda_depth_qp = s_lda.depth_qp,
    lda_bw_ks = s_lda.bw_ks, lda_bw_qp = s_lda.bw_qp,
    # PBE QP
    eqp_pbe, deltas_pbe, eqp_arr_pbe, deltas_arr_pbe,
    pbe_depth_ks = s_pbe.depth_ks, pbe_depth_qp = s_pbe.depth_qp,
    pbe_bw_ks = s_pbe.bw_ks, pbe_bw_qp = s_pbe.bw_qp,
    pbe_conduction_band = conduction_band,
    # Δ(K) grid
    K_grid = K_grid_save,
    delta_total = Δ_total,
    delta_per_channel = Δ_per_channel,
    fc_per_channel = fc_per_channel,
    # Channel info
    channel_names = [ff.name for ff in form_factors],
    channel_f0 = [ff.f0 for ff in form_factors],
    channel_J_c = [ff.J_c for ff in form_factors],
    channel_ΔE = [ff.ΔE for ff in form_factors],
    # Summary
    delta_K0 = delta_K(0.0),
    method = "coherent",
)
println("→ $outfile")

println("\n=== Done ===")
