"""
freq_correction_pbe.jl (aluminum)
Stage 2: Load LDA and PBE KS data, apply the EFT coherent frequency correction,
and compare results.

Demonstrates that the QP correction is robust to the choice of XC functional
in the static PSP (both Zion=3, [Ne] core).

Uses the coherent (exact) Δ(nk) = Σ_c |Σ_G c(G) f_c(|k+G|) / ΔE_c|²

Al core: 1s² 2s² 2p⁶ ([Ne], 10 electrons)
Valence: 3s² 3p¹ (Z_val = 3)

Usage: julia --project=. aluminum/freq_correction_pbe.jl
Prerequisites:
  julia --project=. aluminum/run_ks.jl       (produces al_ks_data.jld2)
  julia --project=. aluminum/run_ks_pbe.jl   (produces al_pbe_ks_data.jld2)
"""

if "--dry-run" in ARGS
    println("[dry-run] aluminum/freq_correction_pbe.jl")
    println("  Loads: aluminum/al_ks_data.jld2, aluminum/al_pbe_ks_data.jld2")
    println("  Computes: atomic ΔSCF + form factors + coherent QP correction (LDA & PBE)")
    println("  Saves: aluminum/al_pbe_qp_data.jld2")
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
# Load KS data (LDA + PBE)
# ══════════════════════════════════════════════════════════════════════════════

lda_file = joinpath(outdir, "al_ks_data.jld2")
pbe_file = joinpath(outdir, "al_pbe_ks_data.jld2")

println("=== Loading KS data ===")
println("  LDA: $lda_file")
println("  PBE: $pbe_file")

ks_lda = load(lda_file)
ks_pbe = load(pbe_file)

@printf("  LDA: %d k-points, %d bands, εF = %.6f Ha\n",
        length(ks_lda["psi"]), length(ks_lda["eigenvalues"][1]), ks_lda["εF"])
@printf("  PBE: %d k-points, %d bands, εF = %.6f Ha\n",
        length(ks_pbe["psi"]), length(ks_pbe["eigenvalues"][1]), ks_pbe["εF"])

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Core excitation energies via ΔSCF
# ══════════════════════════════════════════════════════════════════════════════

const AL_Z_NUC = 13
const AL_CORE_CONFIG = [(1,0,2), (2,0,2), (2,1,6)]
const AL_FULL_CONFIG = [(1,0,2), (2,0,2), (2,1,6), (3,0,2), (3,1,1)]

println("\n=== Atomic solver: core excitation energies ===")

dr_atom = 0.002; r_max_atom = 40.0

atom_core = solve_atom(AL_Z_NUC, AL_CORE_CONFIG; dr=dr_atom, r_max=r_max_atom)
E_core = atom_core.E_total

atom_hole_2s = solve_atom(AL_Z_NUC, [(1,0,2),(2,0,1),(2,1,6)]; dr=dr_atom, r_max=r_max_atom)
ΔE_2s = atom_hole_2s.E_total - E_core

atom_hole_1s = solve_atom(AL_Z_NUC, [(1,0,1),(2,0,2),(2,1,6)]; dr=dr_atom, r_max=r_max_atom)
ΔE_1s = atom_hole_1s.E_total - E_core

@printf("  ΔE_1s = %.4f Ha (1/ΔE² = %.2e)\n", ΔE_1s, 1/ΔE_1s^2)
@printf("  ΔE_2s = %.4f Ha (1/ΔE² = %.2e)\n", ΔE_2s, 1/ΔE_2s^2)

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Form factors f_c(K)
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Form factors ===")

atom_full = solve_atom(AL_Z_NUC, [AL_FULL_CONFIG...]; dr=dr_atom, r_max=r_max_atom)
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

for orb in atom_full.orbitals[1:end-2]  # skip valence 3s, 3p
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

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Coherent QP correction (applied to both LDA and PBE)
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Applying coherent QP correction ===")

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

function apply_qp(ks_data)
    psi           = ks_data["psi"]
    eigenvalues   = ks_data["eigenvalues"]
    k_coordinates = ks_data["k_coordinates"]
    G_vectors     = ks_data["G_vectors"]
    recip_lattice = ks_data["recip_lattice"]
    εF            = ks_data["εF"]

    n_kpts  = length(psi)
    n_bands = length(eigenvalues[1])

    eigenvalues_qp = [similar(eigenvalues[ik]) for ik in 1:n_kpts]
    deltas = [zeros(n_bands) for _ in 1:n_kpts]

    for ik in 1:n_kpts
        k_frac = k_coordinates[ik]
        Gvecs_int = G_vectors[ik]
        for n in 1:n_bands
            ψnk = psi[ik][:, n]
            Δ = delta_nk(ψnk, k_frac, Gvecs_int, recip_lattice)
            deltas[ik][n] = Δ
            eigenvalues_qp[ik][n] = εF + (eigenvalues[ik][n] - εF) / (1.0 + Δ)
        end
    end
    return eigenvalues_qp, deltas
end

println("  LDA...")
t0 = time()
eqp_lda, deltas_lda = apply_qp(ks_lda)
@printf("  LDA QP done in %.1f s\n", time() - t0)

println("  PBE...")
t1 = time()
eqp_pbe, deltas_pbe = apply_qp(ks_pbe)
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

# High-symmetry points (FCC)
hs_points = [
    ("Γ",  [0.0, 0.0, 0.0]),
    ("X",  [0.5, 0.0, 0.5]),
    ("L",  [0.5, 0.5, 0.5]),
    ("W",  [0.5, 0.25, 0.75]),
    ("K",  [0.375, 0.375, 0.75]),
]

println("  --- LDA: Band-by-band at high-symmetry points (eV relative to εF) ---")
εF_lda = ks_lda["εF"]
for (label, coord) in hs_points
    ik = find_kpoint(ks_lda["k_coordinates"], coord)
    isnothing(ik) && continue
    @printf("  %s:", label)
    for n in 1:min(4, length(ks_lda["eigenvalues"][1]))
        e_ks = (ks_lda["eigenvalues"][ik][n] - εF_lda) * Ha_to_eV
        e_qp = (eqp_lda[ik][n] - εF_lda) * Ha_to_eV
        @printf("  n%d: KS=%+.2f QP=%+.2f", n, e_ks, e_qp)
    end
    println()
end

println("\n  --- PBE: Band-by-band at high-symmetry points (eV relative to εF) ---")
εF_pbe = ks_pbe["εF"]
for (label, coord) in hs_points
    ik = find_kpoint(ks_pbe["k_coordinates"], coord)
    isnothing(ik) && continue
    @printf("  %s:", label)
    for n in 1:min(4, length(ks_pbe["eigenvalues"][1]))
        e_ks = (ks_pbe["eigenvalues"][ik][n] - εF_pbe) * Ha_to_eV
        e_qp = (eqp_pbe[ik][n] - εF_pbe) * Ha_to_eV
        @printf("  n%d: KS=%+.2f QP=%+.2f", n, e_ks, e_qp)
    end
    println()
end

# Lowest band statistics
function band_stats(eigenvalues, eigenvalues_qp, εF, band_idx)
    e_ks = [(eigenvalues[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(eigenvalues)]
    e_qp = [(eigenvalues_qp[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(eigenvalues_qp)]
    return (depth_ks=minimum(e_ks), depth_qp=minimum(e_qp),
            bw_ks=maximum(e_ks)-minimum(e_ks), bw_qp=maximum(e_qp)-minimum(e_qp))
end

s_lda = band_stats(ks_lda["eigenvalues"], eqp_lda, εF_lda, 1)
s_pbe = band_stats(ks_pbe["eigenvalues"], eqp_pbe, εF_pbe, 1)

# Free-electron reference
lattice_matrix = ks_lda["lattice_matrix"]
V_prim = abs(det(lattice_matrix))
n_e = ks_lda["Z_val"]
k_F = (3π^2 * n_e / V_prim)^(1/3)
E_F_free = k_F^2 / 2

println("\n=== Lowest band (n=1) comparison ===\n")
@printf("  %-22s  %10s  %10s  %10s  %10s  %8s\n",
        "", "KS Γ(eV)", "QP Γ(eV)", "BW_KS(eV)", "BW_QP(eV)", "QP/KS")
println("  " * "-"^76)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "LDA Zion=3", s_lda.depth_ks, s_lda.depth_qp, s_lda.bw_ks, s_lda.bw_qp,
        s_lda.depth_qp / s_lda.depth_ks)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "PBE Zion=3", s_pbe.depth_ks, s_pbe.depth_qp, s_pbe.bw_ks, s_pbe.bw_qp,
        s_pbe.depth_qp / s_pbe.depth_ks)
@printf("  %-22s  %+10.3f\n", "Free electron", -E_F_free * Ha_to_eV)

narrowing_lda = (1 - s_lda.depth_qp / s_lda.depth_ks) * 100
narrowing_pbe = (1 - s_pbe.depth_qp / s_pbe.depth_ks) * 100
@printf("\n  Narrowing (n=1): LDA=%.1f%%, PBE=%.1f%%\n", narrowing_lda, narrowing_pbe)

iΓ_lda = find_kpoint(ks_lda["k_coordinates"], [0.0, 0.0, 0.0])
iΓ_pbe = find_kpoint(ks_pbe["k_coordinates"], [0.0, 0.0, 0.0])
if !isnothing(iΓ_lda) && !isnothing(iΓ_pbe)
    @printf("\n  Δ at Γ (n=1): LDA=%.6f, PBE=%.6f\n",
            deltas_lda[iΓ_lda][1], deltas_pbe[iΓ_pbe][1])
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

# Build plotting arrays for both LDA and PBE
function build_plot_arrays(ks_data, eigenvalues_qp, deltas)
    kdistances      = ks_data["kdistances"]
    eigenvalues_arr = ks_data["eigenvalues_array"]
    n_spin          = ks_data["n_spin"]
    krange_spin_map = ks_data["krange_spin_map"]

    eqp_arr = similar(eigenvalues_arr)
    deltas_arr = similar(eigenvalues_arr)
    for σ in 1:n_spin
        for (ito, ik) in enumerate(krange_spin_map[σ])
            eqp_arr[ito, :, σ] = eigenvalues_qp[ik]
            deltas_arr[ito, :, σ] = deltas[ik]
        end
    end
    return eqp_arr, deltas_arr
end

eqp_arr_lda, deltas_arr_lda = build_plot_arrays(ks_lda, eqp_lda, deltas_lda)
eqp_arr_pbe, deltas_arr_pbe = build_plot_arrays(ks_pbe, eqp_pbe, deltas_pbe)

# Δ(K) on a fine grid
K_grid_save = collect(range(0.0, 20.0, length=2000))
Δ_total = [delta_K(K) for K in K_grid_save]
Δ_per_channel = Dict{String,Vector{Float64}}()
fc_per_channel = Dict{String,Vector{Float64}}()
for (ic, ff) in enumerate(form_factors)
    Δ_per_channel[ff.name] = [eval_fc(ff, K) * ff.f0 / ff.ΔE^2 for K in K_grid_save]
    fc_per_channel[ff.name] = [eval_fc(ff, K) for K in K_grid_save]
end

outfile = joinpath(outdir, "al_pbe_qp_data.jld2")
jldsave(outfile;
    # LDA QP results
    lda_eigenvalues_qp = eqp_lda,
    lda_deltas = deltas_lda,
    lda_eqp_arr = eqp_arr_lda,
    lda_deltas_arr = deltas_arr_lda,
    lda_depth_ks = s_lda.depth_ks,
    lda_depth_qp = s_lda.depth_qp,
    lda_bw_ks = s_lda.bw_ks,
    lda_bw_qp = s_lda.bw_qp,
    # PBE QP results
    pbe_eigenvalues_qp = eqp_pbe,
    pbe_deltas = deltas_pbe,
    pbe_eqp_arr = eqp_arr_pbe,
    pbe_deltas_arr = deltas_arr_pbe,
    pbe_depth_ks = s_pbe.depth_ks,
    pbe_depth_qp = s_pbe.depth_qp,
    pbe_bw_ks = s_pbe.bw_ks,
    pbe_bw_qp = s_pbe.bw_qp,
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
    narrowing_lda = narrowing_lda,
    narrowing_pbe = narrowing_pbe,
    method = "coherent",
)
println("→ $outfile")

println("\n=== Done ===")
