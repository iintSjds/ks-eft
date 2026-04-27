"""
freq_correction_pbe.jl (potassium)
Load LDA (Zion=1) and PBE (Zion=9) KS data, apply the EFT coherent frequency
correction, and compare.

K core (EFT theory): [Ar] = 1s² 2s² 2p⁶ 3s² 3p⁶ (18 electrons)
K valence (LDA Zion=1):  4s¹
K valence (PBE Zion=9):  3s² 3p⁶ 4s¹ (semicore included)

For PBE Zion=9: QP correction applied ONLY to conduction band (band 5 = 4s)
and above. Semicore bands 1-4 (3s, 3p) are left unchanged.

Usage: julia --project=. potassium/freq_correction_pbe.jl
Prerequisites:
  julia --project=. potassium/run_ks.jl
  julia --project=. potassium/run_ks_pbe.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] potassium/freq_correction_pbe.jl")
    println("  Loads: potassium/k_ks_data.jld2, potassium/k_pbe_ks_data.jld2")
    println("  Computes: atomic ΔSCF + form factors + coherent QP correction")
    println("  Saves: potassium/k_pbe_qp_data.jld2")
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
# Constants
# ══════════════════════════════════════════════════════════════════════════════

const K_Z_NUC = 19
const K_CORE_CONFIG = [(1,0,2), (2,0,2), (2,1,6), (3,0,2), (3,1,6)]  # [Ar] = 18 electrons
const K_FULL_CONFIG = [(1,0,2), (2,0,2), (2,1,6), (3,0,2), (3,1,6), (4,0,1)]  # 19 electrons
const Ha_to_eV = 27.211386245988

# ══════════════════════════════════════════════════════════════════════════════
# Load KS data (both LDA and PBE)
# ══════════════════════════════════════════════════════════════════════════════

println("=== Loading KS data ===\n")

lda_file = joinpath(outdir, "k_ks_data.jld2")
pbe_file = joinpath(outdir, "k_pbe_ks_data.jld2")

println("  LDA: $lda_file")
lda = load(lda_file)
@printf("    %d k-points, %d bands, Zion=%d, εF=%.6f Ha\n",
        length(lda["psi"]), length(lda["eigenvalues"][1]), lda["Zion"], lda["εF"])

println("  PBE: $pbe_file")
pbe = load(pbe_file)
@printf("    %d k-points, %d bands, Zion=%d, εF=%.6f Ha\n",
        length(pbe["psi"]), length(pbe["eigenvalues"][1]), pbe["Zion"], pbe["εF"])

conduction_band = pbe["conduction_band"]
@printf("    Conduction band index: %d (4s)\n", conduction_band)

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Compute excitation energies via ΔSCF
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Step 1: Core excitation energies (ΔSCF) ===\n")

dr_atom = 0.002
r_max_atom = 40.0

println("--- Core ground state ([Ar] = 1s² 2s² 2p⁶ 3s² 3p⁶, 18 electrons) ---")
atom_core = solve_atom(K_Z_NUC, K_CORE_CONFIG; dr=dr_atom, r_max=r_max_atom)
E_core = atom_core.E_total
@printf("  E_core = %.6f Ha\n", E_core)

println("\n--- 3s hole (1s² 2s² 2p⁶ 3s¹ 3p⁶, 17 electrons) ---")
hole_3s_config = [(1,0,2), (2,0,2), (2,1,6), (3,0,1), (3,1,6)]
atom_hole_3s = solve_atom(K_Z_NUC, hole_3s_config; dr=dr_atom, r_max=r_max_atom)
ΔE_3s = atom_hole_3s.E_total - E_core
@printf("  ΔE_3s = %.6f Ha = %.2f eV\n", ΔE_3s, ΔE_3s * Ha_to_eV)

println("\n--- 2s hole (1s² 2s¹ 2p⁶ 3s² 3p⁶, 17 electrons) ---")
hole_2s_config = [(1,0,2), (2,0,1), (2,1,6), (3,0,2), (3,1,6)]
atom_hole_2s = solve_atom(K_Z_NUC, hole_2s_config; dr=dr_atom, r_max=r_max_atom)
ΔE_2s = atom_hole_2s.E_total - E_core
@printf("  ΔE_2s = %.6f Ha = %.2f eV\n", ΔE_2s, ΔE_2s * Ha_to_eV)

println("\n--- 1s hole (1s¹ 2s² 2p⁶ 3s² 3p⁶, 17 electrons) ---")
hole_1s_config = [(1,0,1), (2,0,2), (2,1,6), (3,0,2), (3,1,6)]
atom_hole_1s = solve_atom(K_Z_NUC, hole_1s_config; dr=dr_atom, r_max=r_max_atom)
ΔE_1s = atom_hole_1s.E_total - E_core
@printf("  ΔE_1s = %.6f Ha = %.2f eV\n", ΔE_1s, ΔE_1s * Ha_to_eV)

println("\n--- Excitation energy summary ---")
@printf("  ΔE_1s = %.4f Ha  (1/ΔE² = %.2e)\n", ΔE_1s, 1/ΔE_1s^2)
@printf("  ΔE_2s = %.4f Ha  (1/ΔE² = %.2e)\n", ΔE_2s, 1/ΔE_2s^2)
@printf("  ΔE_3s = %.4f Ha  (1/ΔE² = %.2e)\n", ΔE_3s, 1/ΔE_3s^2)

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Compute form factors f_c(K)
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Step 2: Form factors f_c(K) ===\n")

atom_full = solve_atom(K_Z_NUC, [K_FULL_CONFIG...]; dr=dr_atom, r_max=r_max_atom)
rgrid = atom_full.rgrid
dr = atom_full.dr
N = length(rgrid)

struct FormFactor
    name::String
    u_c::Vector{Float64}
    V_H_c::Vector{Float64}
    J_c::Float64
    S_c::Float64
    W_c::Float64
    f0::Float64
    ΔE::Float64
end

form_factors = FormFactor[]
ΔE_map = Dict(1 => ΔE_1s, 2 => ΔE_2s, 3 => ΔE_3s)

for (idx, orb) in enumerate(atom_full.orbitals[1:end-1])  # skip valence (4s)
    orb.l != 0 && continue  # only s-orbitals

    V_H_c = orbital_coulomb_potential(orb.u, 1.0, rgrid, dr)
    J_c = dr * sum(orb.u[i]^2 * V_H_c[i] for i in 1:N)
    S_c = dr * sum(rgrid[i] * orb.u[i] for i in 1:N)
    W_c = dr * sum(rgrid[i] * orb.u[i] * V_H_c[i] for i in 1:N)
    f0 = sqrt(4π) * (W_c - J_c * S_c)

    ΔE = ΔE_map[orb.n]
    name = "$(orb.n)s"

    push!(form_factors, FormFactor(name, copy(orb.u), V_H_c, J_c, S_c, W_c, f0, ΔE))

    @printf("  %s: J_c=%.6f, S_c=%.6f, W_c=%.6f, f(0)=%.6f, ΔE=%.4f Ha\n",
            name, J_c, S_c, W_c, f0, ΔE)
    @printf("       Δ_c(0) = f(0)²/ΔE² = %.6f\n", f0^2 / ΔE^2)
end

function eval_fc(ff::FormFactor, K::Float64)
    K < 1e-10 && return ff.f0
    integral = 0.0
    for i in 1:N
        integral += ff.u_c[i] * (ff.V_H_c[i] - ff.J_c) * sin(K * rgrid[i]) * dr
    end
    return sqrt(4π) / K * integral
end

function delta_K(K::Float64)
    Δ = 0.0
    for ff in form_factors
        fK = eval_fc(ff, K)
        Δ += fK * ff.f0 / ff.ΔE^2
    end
    return Δ
end

@printf("\n  Δ(K=0) = %.6f\n", delta_K(0.0))

# Precompute per-channel interpolation
const K_MAX = 20.0
const N_INTERP = 2000
const K_grid_interp = collect(range(0.0, K_MAX, length=N_INTERP))
fc_interp = [linear_interpolation(K_grid_interp,
    [eval_fc(ff, Float64(K)) for K in K_grid_interp]) for ff in form_factors]

println("  Precomputed f_c(K) on $(N_INTERP)-point grid")

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

function apply_qp(ks_data; band_indices=nothing)
    psi_d       = ks_data["psi"]
    evals_d     = ks_data["eigenvalues"]
    kcoords_d   = ks_data["k_coordinates"]
    Gvecs_d     = ks_data["G_vectors"]
    recip_lat_d = ks_data["recip_lattice"]
    εF_d        = ks_data["εF"]

    n_kpts = length(psi_d)
    n_bands = length(evals_d[1])
    if isnothing(band_indices)
        band_indices = 1:n_bands
    end

    eigenvalues_qp = deepcopy(evals_d)
    deltas = [zeros(n_bands) for _ in 1:n_kpts]

    for ik in 1:n_kpts
        k_frac = kcoords_d[ik]
        Gvecs_int = Gvecs_d[ik]
        for n in band_indices
            ψnk = psi_d[ik][:, n]
            Δ = delta_nk(ψnk, k_frac, Gvecs_int, recip_lat_d)
            deltas[ik][n] = Δ
            eigenvalues_qp[ik][n] = εF_d + (evals_d[ik][n] - εF_d) / (1.0 + Δ)
        end
    end
    return eigenvalues_qp, deltas
end

# LDA Zion=1: apply to all bands
println("  LDA Zion=1: applying QP to all bands...")
t0 = time()
eqp_lda, deltas_lda = apply_qp(lda)
@printf("  Done in %.1f s\n", time() - t0)

# PBE Zion=9: apply only to conduction band (band 5 = 4s) and above
n_bands_pbe = length(pbe["eigenvalues"][1])
@printf("  PBE Zion=9: applying QP to bands %d-%d only (4s and above)...\n",
        conduction_band, n_bands_pbe)
t1 = time()
eqp_pbe, deltas_pbe = apply_qp(pbe; band_indices=conduction_band:n_bands_pbe)
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

# BCC high-symmetry points
hs_points = [
    ("Γ",  [0.0, 0.0, 0.0]),
    ("H",  [0.5, -0.5, 0.5]),
    ("N",  [0.0, 0.0, 0.5]),
    ("P",  [0.25, 0.25, 0.25]),
]

println("--- LDA Zion=1 (band 1 = 4s conduction) ---")
lda_kcoords = lda["k_coordinates"]
lda_evals = lda["eigenvalues"]
εF_lda = lda["εF"]
for (label, coord) in hs_points
    ik = find_kpoint(lda_kcoords, coord)
    isnothing(ik) && continue
    e_ks = (lda_evals[ik][1] - εF_lda) * Ha_to_eV
    e_qp = (eqp_lda[ik][1] - εF_lda) * Ha_to_eV
    Δ = deltas_lda[ik][1]
    @printf("  %s: KS=%+.3f eV, QP=%+.3f eV, Δ=%.4f\n", label, e_ks, e_qp, Δ)
end

println("\n--- PBE Zion=9 (band $conduction_band = 4s conduction) ---")
pbe_kcoords = pbe["k_coordinates"]
pbe_evals = pbe["eigenvalues"]
εF_pbe = pbe["εF"]
for (label, coord) in hs_points
    ik = find_kpoint(pbe_kcoords, coord)
    isnothing(ik) && continue
    e_ks = (pbe_evals[ik][conduction_band] - εF_pbe) * Ha_to_eV
    e_qp = (eqp_pbe[ik][conduction_band] - εF_pbe) * Ha_to_eV
    Δ = deltas_pbe[ik][conduction_band]
    @printf("  %s: KS=%+.3f eV, QP=%+.3f eV, Δ=%.4f\n", label, e_ks, e_qp, Δ)
end

# ══════════════════════════════════════════════════════════════════════════════
# Conduction band summary
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Conduction band summary ===\n")

function conduction_band_stats(evals, eqp, εF, band_idx)
    e_ks = [(evals[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(evals)]
    e_qp = [(eqp[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(eqp)]
    depth_ks = minimum(e_ks)
    depth_qp = minimum(e_qp)
    bw_ks = maximum(e_ks) - minimum(e_ks)
    bw_qp = maximum(e_qp) - minimum(e_qp)
    return (; depth_ks, depth_qp, bw_ks, bw_qp)
end

s_lda = conduction_band_stats(lda_evals, eqp_lda, εF_lda, 1)
s_pbe = conduction_band_stats(pbe_evals, eqp_pbe, εF_pbe, conduction_band)

a = lda["a_bohr"]
V_prim = a^3 / 2  # BCC primitive cell volume
k_F = (3π^2 / V_prim)^(1/3)  # Zion=1 free electron
E_F_free = k_F^2 / 2

@printf("  %-22s  %10s  %10s  %10s  %10s  %8s\n",
        "", "KS Γ(eV)", "QP Γ(eV)", "BW_KS(eV)", "BW_QP(eV)", "QP/KS")
println("  " * "-"^76)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "LDA Zion=1", s_lda.depth_ks, s_lda.depth_qp,
        s_lda.bw_ks, s_lda.bw_qp, s_lda.depth_qp / s_lda.depth_ks)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "PBE Zion=9", s_pbe.depth_ks, s_pbe.depth_qp,
        s_pbe.bw_ks, s_pbe.bw_qp, s_pbe.depth_qp / s_pbe.depth_ks)
@printf("  %-22s  %+10.3f\n", "Free electron", -E_F_free * Ha_to_eV)

narrowing_lda = (1 - s_lda.depth_qp / s_lda.depth_ks) * 100
narrowing_pbe = (1 - s_pbe.depth_qp / s_pbe.depth_ks) * 100
@printf("\n  Narrowing: LDA=%.1f%%, PBE=%.1f%%\n", narrowing_lda, narrowing_pbe)

iΓ_lda = find_kpoint(lda_kcoords, [0.0, 0.0, 0.0])
iΓ_pbe = find_kpoint(pbe_kcoords, [0.0, 0.0, 0.0])
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

# Build QP arrays for plotting (both LDA and PBE)
function build_qp_arrays(ks_data, eqp, deltas_d)
    n_spin          = ks_data["n_spin"]
    krange_spin_map = ks_data["krange_spin_map"]
    eigenvalues_arr = ks_data["eigenvalues_array"]

    eqp_arr = similar(eigenvalues_arr)
    deltas_arr = similar(eigenvalues_arr)
    for σ in 1:n_spin
        for (ito, ik) in enumerate(krange_spin_map[σ])
            eqp_arr[ito, :, σ] = eqp[ik]
            deltas_arr[ito, :, σ] = deltas_d[ik]
        end
    end
    return eqp_arr, deltas_arr
end

eqp_arr_lda, deltas_arr_lda = build_qp_arrays(lda, eqp_lda, deltas_lda)
eqp_arr_pbe, deltas_arr_pbe = build_qp_arrays(pbe, eqp_pbe, deltas_pbe)

# Δ(K) grid for saving
K_grid_save = collect(range(0.0, 20.0, length=2000))
Δ_total = [delta_K(K) for K in K_grid_save]
Δ_per_channel = Dict{String,Vector{Float64}}()
fc_per_channel = Dict{String,Vector{Float64}}()
for (ic, ff) in enumerate(form_factors)
    Δ_per_channel[ff.name] = [eval_fc(ff, K) * ff.f0 / ff.ΔE^2 for K in K_grid_save]
    fc_per_channel[ff.name] = [eval_fc(ff, K) for K in K_grid_save]
end

outfile = joinpath(outdir, "k_pbe_qp_data.jld2")
jldsave(outfile;
    # LDA results
    lda_eigenvalues_qp = eqp_lda,
    lda_deltas = deltas_lda,
    lda_eqp_arr = eqp_arr_lda,
    lda_deltas_arr = deltas_arr_lda,
    lda_depth_ks = s_lda.depth_ks,
    lda_depth_qp = s_lda.depth_qp,
    lda_bw_ks = s_lda.bw_ks,
    lda_bw_qp = s_lda.bw_qp,
    # PBE results
    pbe_eigenvalues_qp = eqp_pbe,
    pbe_deltas = deltas_pbe,
    pbe_eqp_arr = eqp_arr_pbe,
    pbe_deltas_arr = deltas_arr_pbe,
    pbe_depth_ks = s_pbe.depth_ks,
    pbe_depth_qp = s_pbe.depth_qp,
    pbe_bw_ks = s_pbe.bw_ks,
    pbe_bw_qp = s_pbe.bw_qp,
    pbe_conduction_band = conduction_band,
    # Shared EFT data
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
