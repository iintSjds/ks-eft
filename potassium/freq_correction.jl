"""
freq_correction.jl (potassium)
Stage 2: Load KS data and apply the EFT coherent frequency correction.

Uses the coherent (exact) Δ(nk) = Σ_c |Σ_G c(G) f_c(|k+G|) / ΔE_c|²

K core: 1s² 2s² 2p⁶ 3s² 3p⁶ ([Ar], 18 electrons)
Valence: 4s¹ (Z_val = 1)

Usage: julia --project=. potassium/freq_correction.jl
Prerequisite: julia --project=. potassium/run_ks.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] potassium/freq_correction.jl")
    println("  Loads: potassium/k_ks_data.jld2")
    println("  Computes: atomic ΔSCF + form factors + coherent QP correction")
    println("  Saves: potassium/k_qp_data.jld2")
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

# ══════════════════════════════════════════════════════════════════════════════
# Load KS data
# ══════════════════════════════════════════════════════════════════════════════

ks_file = joinpath(outdir, "k_ks_data.jld2")
println("=== Loading KS data from $ks_file ===")

ks = load(ks_file)
psi           = ks["psi"]
eigenvalues   = ks["eigenvalues"]
k_coordinates = ks["k_coordinates"]
G_vectors     = ks["G_vectors"]
recip_lattice = ks["recip_lattice"]
εF            = ks["εF"]

@printf("  %d k-points, %d bands, εF = %.6f Ha\n",
        length(psi), length(eigenvalues[1]), εF)

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

# ΔSCF for each s-orbital hole
println("\n--- 3s hole (1s² 2s² 2p⁶ 3s¹ 3p⁶, 17 electrons) ---")
hole_3s_config = [(1,0,2), (2,0,2), (2,1,6), (3,0,1), (3,1,6)]
atom_hole_3s = solve_atom(K_Z_NUC, hole_3s_config; dr=dr_atom, r_max=r_max_atom)
ΔE_3s = atom_hole_3s.E_total - E_core
@printf("  E_hole_3s = %.6f Ha\n", atom_hole_3s.E_total)
@printf("  ΔE_3s = %.6f Ha = %.2f eV\n", ΔE_3s, ΔE_3s * 27.2114)
@printf("  Koopmans: -ε_3s = %.6f Ha\n", -atom_core.orbitals[4].eps)

println("\n--- 2s hole (1s² 2s¹ 2p⁶ 3s² 3p⁶, 17 electrons) ---")
hole_2s_config = [(1,0,2), (2,0,1), (2,1,6), (3,0,2), (3,1,6)]
atom_hole_2s = solve_atom(K_Z_NUC, hole_2s_config; dr=dr_atom, r_max=r_max_atom)
ΔE_2s = atom_hole_2s.E_total - E_core
@printf("  E_hole_2s = %.6f Ha\n", atom_hole_2s.E_total)
@printf("  ΔE_2s = %.6f Ha = %.2f eV\n", ΔE_2s, ΔE_2s * 27.2114)
@printf("  Koopmans: -ε_2s = %.6f Ha\n", -atom_core.orbitals[2].eps)

println("\n--- 1s hole (1s¹ 2s² 2p⁶ 3s² 3p⁶, 17 electrons) ---")
hole_1s_config = [(1,0,1), (2,0,2), (2,1,6), (3,0,2), (3,1,6)]
atom_hole_1s = solve_atom(K_Z_NUC, hole_1s_config; dr=dr_atom, r_max=r_max_atom)
ΔE_1s = atom_hole_1s.E_total - E_core
@printf("  E_hole_1s = %.6f Ha\n", atom_hole_1s.E_total)
@printf("  ΔE_1s = %.6f Ha = %.2f eV\n", ΔE_1s, ΔE_1s * 27.2114)
@printf("  Koopmans: -ε_1s = %.6f Ha\n", -atom_core.orbitals[1].eps)

# Summary
println("\n--- Excitation energy summary ---")
@printf("  ΔE_1s = %.4f Ha  (1/ΔE² = %.2e)\n", ΔE_1s, 1/ΔE_1s^2)
@printf("  ΔE_2s = %.4f Ha  (1/ΔE² = %.2e)\n", ΔE_2s, 1/ΔE_2s^2)
@printf("  ΔE_3s = %.4f Ha  (1/ΔE² = %.2e)\n", ΔE_3s, 1/ΔE_3s^2)

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Compute form factors f_c(K)
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Step 2: Form factors f_c(K) ===\n")

# Use the full atom (with valence) for core orbitals
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

# Map n quantum number to excitation energy
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

println("\n  Δ(K) at selected momenta:")
for K in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    @printf("    K = %5.1f Bohr⁻¹:  Δ = %+.6f", K, delta_K(K))
    for ff in form_factors
        fK = eval_fc(ff, K)
        Δc = fK * ff.f0 / ff.ΔE^2
        @printf("  (%s: %+.6f)", ff.name, Δc)
    end
    println()
end

# Precompute per-channel interpolation
const K_MAX = 20.0
const N_INTERP = 2000
const K_grid_interp = collect(range(0.0, K_MAX, length=N_INTERP))
fc_interp = [linear_interpolation(K_grid_interp,
    [eval_fc(ff, Float64(K)) for K in K_grid_interp]) for ff in form_factors]

@printf("\n  Δ(K=0) = %.6f\n", delta_K(0.0))

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Coherent QP correction
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
n_kpts = length(psi)
n_bands = length(eigenvalues[1])

eigenvalues_qp = [similar(eigenvalues[ik]) for ik in 1:n_kpts]
deltas = [zeros(n_bands) for _ in 1:n_kpts]

t0 = time()
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
@printf("QP correction done in %.1f s\n", time() - t0)

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

println("  Band-by-band at high-symmetry points (eV relative to εF):")
for (label, coord) in hs_points
    ik = find_kpoint(k_coordinates, coord)
    isnothing(ik) && continue
    @printf("  %s:", label)
    for n in 1:min(4, n_bands)
        e_ks = (eigenvalues[ik][n] - εF) * Ha_to_eV
        e_qp = (eigenvalues_qp[ik][n] - εF) * Ha_to_eV
        @printf("  n%d: KS=%+.2f QP=%+.2f", n, e_ks, e_qp)
    end
    println()
end

# Lowest band statistics
e1_ks = [eigenvalues[ik][1] for ik in 1:n_kpts]
e1_qp = [eigenvalues_qp[ik][1] for ik in 1:n_kpts]
depth_ks = (minimum(e1_ks) - εF) * Ha_to_eV
depth_qp = (minimum(e1_qp) - εF) * Ha_to_eV
bw_ks = (maximum(e1_ks) - minimum(e1_ks)) * Ha_to_eV
bw_qp = (maximum(e1_qp) - minimum(e1_qp)) * Ha_to_eV

lattice_matrix = ks["lattice_matrix"]
V_prim = abs(det(lattice_matrix))
n_e = ks["Z_val"]
k_F = (3π^2 * n_e / V_prim)^(1/3)
E_F_free = k_F^2 / 2

println("\n=== Lowest band (n=1) statistics ===\n")
@printf("  KS bandwidth:  %.3f eV\n", bw_ks)
@printf("  QP bandwidth:  %.3f eV\n", bw_qp)
@printf("  Narrowing:     %.1f%%\n", (1 - bw_qp/bw_ks) * 100)
@printf("  KS Γ depth:    %+.3f eV\n", depth_ks)
@printf("  QP Γ depth:    %+.3f eV\n", depth_qp)
@printf("  QP/KS ratio:   %.4f\n", depth_qp / depth_ks)
@printf("  Free electron: Γ depth = %+.3f eV\n", -E_F_free * Ha_to_eV)

iΓ = find_kpoint(k_coordinates, [0.0, 0.0, 0.0])
if !isnothing(iΓ)
    @printf("\n  Δ at Γ:\n")
    for n in 1:min(4, n_bands)
        @printf("    n=%d: Δ=%.6f\n", n, deltas[iΓ][n])
    end
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

kdistances      = ks["kdistances"]
eigenvalues_arr = ks["eigenvalues_array"]
tick_distances  = ks["tick_distances"]
tick_labels     = ks["tick_labels"]
n_spin          = ks["n_spin"]
krange_spin_map = ks["krange_spin_map"]

eqp_arr = similar(eigenvalues_arr)
deltas_arr = similar(eigenvalues_arr)
for σ in 1:n_spin
    for (ito, ik) in enumerate(krange_spin_map[σ])
        eqp_arr[ito, :, σ] = eigenvalues_qp[ik]
        deltas_arr[ito, :, σ] = deltas[ik]
    end
end

K_grid_save = collect(range(0.0, 20.0, length=2000))
Δ_total = [delta_K(K) for K in K_grid_save]
Δ_per_channel = Dict{String,Vector{Float64}}()
fc_per_channel = Dict{String,Vector{Float64}}()
for (ic, ff) in enumerate(form_factors)
    Δ_per_channel[ff.name] = [eval_fc(ff, K) * ff.f0 / ff.ΔE^2 for K in K_grid_save]
    fc_per_channel[ff.name] = [eval_fc(ff, K) for K in K_grid_save]
end

outfile = joinpath(outdir, "k_qp_data.jld2")
jldsave(outfile;
    eigenvalues_qp, deltas,
    eqp_arr, deltas_arr,
    K_grid = K_grid_save,
    delta_total = Δ_total,
    delta_per_channel = Δ_per_channel,
    fc_per_channel = fc_per_channel,
    channel_names = [ff.name for ff in form_factors],
    channel_f0 = [ff.f0 for ff in form_factors],
    channel_J_c = [ff.J_c for ff in form_factors],
    channel_ΔE = [ff.ΔE for ff in form_factors],
    depth_ks, depth_qp, bw_ks, bw_qp,
    delta_K0 = delta_K(0.0),
    method = "coherent",
)
println("→ $outfile")

# Human-readable summary
open(joinpath(outdir, "k_summary.txt"), "w") do io
    println(io, "# K EFT QP correction summary (coherent formula)")
    println(io, "# Generated by potassium/freq_correction.jl")
    println(io)
    @printf(io, "Z = %d, Z_val = %d\n", K_Z_NUC, ks["Z_val"])
    @printf(io, "Core: [Ar] = 1s² 2s² 2p⁶ 3s² 3p⁶ (18 electrons)\n")
    @printf(io, "Valence: 4s¹\n")
    @printf(io, "Structure: BCC, a = %.3f Bohr\n", ks["a_bohr"])
    println(io)
    @printf(io, "ΔE_1s = %.6f Ha\n", ΔE_1s)
    @printf(io, "ΔE_2s = %.6f Ha\n", ΔE_2s)
    @printf(io, "ΔE_3s = %.6f Ha\n", ΔE_3s)
    println(io)
    for ff in form_factors
        @printf(io, "%s: f(0) = %.6f, J_c = %.6f, ΔE = %.4f Ha, Δ_c(0) = %.6f\n",
                ff.name, ff.f0, ff.J_c, ff.ΔE, ff.f0^2/ff.ΔE^2)
    end
    println(io)
    @printf(io, "Δ(K=0) total = %.6f\n", delta_K(0.0))
    @printf(io, "Method: coherent (exact)\n")
    println(io)
    @printf(io, "εF = %.6f Ha = %.4f eV\n", εF, εF * Ha_to_eV)
    @printf(io, "KS Γ depth (n=1) = %.4f eV\n", depth_ks)
    @printf(io, "QP Γ depth (n=1) = %.4f eV\n", depth_qp)
    @printf(io, "QP/KS ratio = %.4f\n", depth_qp / depth_ks)
    @printf(io, "Narrowing = %.1f%%\n", (1 - depth_qp/depth_ks) * 100)
    @printf(io, "KS bandwidth (n=1) = %.4f eV\n", bw_ks)
    @printf(io, "QP bandwidth (n=1) = %.4f eV\n", bw_qp)
    println(io)
    @printf(io, "PSP: %s (Zion=%d)\n", ks["psp_identifier"], ks["Zion"])
    @printf(io, "Ecut = %.1f Ha, kgrid = %s\n", ks["Ecut"], join(ks["kgrid"], "×"))
end
println("→ k_summary.txt")

println("\n=== Done ===")
