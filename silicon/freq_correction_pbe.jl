"""
freq_correction_pbe.jl (silicon)
Stage 2 (PBE): Load both LDA and PBE KS data, apply EFT coherent QP correction,
and compare results.

Si core: 1s² 2s² 2p⁶ ([Ne], 10 electrons)
Valence: 3s² 3p² (Z_val = 4)

Both LDA and PBE use Zion=4 (largecore). The EFT frequency correction is
identical for both — only the static KS eigenvalues/wavefunctions differ.

Usage: julia --project=. silicon/freq_correction_pbe.jl
Prerequisites:
  julia --project=. silicon/run_ks.jl       (produces si_ks_data.jld2)
  julia --project=. silicon/run_ks_pbe.jl   (produces si_pbe_ks_data.jld2)
"""

if "--dry-run" in ARGS
    println("[dry-run] silicon/freq_correction_pbe.jl")
    println("  Loads: silicon/si_ks_data.jld2 (LDA), silicon/si_pbe_ks_data.jld2 (PBE)")
    println("  Computes: atomic ΔSCF + form factors + coherent QP correction for both")
    println("  Saves: silicon/si_pbe_qp_data.jld2")
    exit(0)
end

using Printf
using LinearAlgebra
using JLD2
using Interpolations: linear_interpolation

# Reuse the atomic solver from sodium
include(joinpath(@__DIR__, "..", "sodium", "atomic_hf.jl"))

outdir = @__DIR__
Ha_to_eV = 27.211386245988

# ══════════════════════════════════════════════════════════════════════════════
# Load KS data (LDA + PBE)
# ══════════════════════════════════════════════════════════════════════════════

lda_file = joinpath(outdir, "si_ks_data.jld2")
pbe_file = joinpath(outdir, "si_pbe_ks_data.jld2")
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

const SI_Z_NUC = 14
const SI_CORE_CONFIG = [(1,0,2), (2,0,2), (2,1,6)]
const SI_FULL_CONFIG = [(1,0,2), (2,0,2), (2,1,6), (3,0,2), (3,1,2)]

println("\n=== Atomic solver: core excitation energies ===")

dr_atom = 0.002; r_max_atom = 40.0

atom_core = solve_atom(SI_Z_NUC, SI_CORE_CONFIG; dr=dr_atom, r_max=r_max_atom)
E_core = atom_core.E_total

atom_hole_1s = solve_atom(SI_Z_NUC, [(1,0,1),(2,0,2),(2,1,6)]; dr=dr_atom, r_max=r_max_atom)
ΔE_1s = atom_hole_1s.E_total - E_core

atom_hole_2s = solve_atom(SI_Z_NUC, [(1,0,2),(2,0,1),(2,1,6)]; dr=dr_atom, r_max=r_max_atom)
ΔE_2s = atom_hole_2s.E_total - E_core

@printf("  ΔE_1s = %.4f Ha (1/ΔE² = %.2e)\n", ΔE_1s, 1/ΔE_1s^2)
@printf("  ΔE_2s = %.4f Ha (1/ΔE² = %.2e)\n", ΔE_2s, 1/ΔE_2s^2)

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Form factors f_c(K)
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Form factors ===")

atom_full = solve_atom(SI_Z_NUC, [SI_FULL_CONFIG...]; dr=dr_atom, r_max=r_max_atom)
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
# Step 3: Coherent QP correction for both LDA and PBE
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

function apply_qp(ks_data)
    psi           = ks_data["psi"]
    eigenvalues   = ks_data["eigenvalues"]
    k_coordinates = ks_data["k_coordinates"]
    G_vectors     = ks_data["G_vectors"]
    recip_lattice = ks_data["recip_lattice"]
    εF            = ks_data["εF"]

    n_kpts = length(psi)
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
# Results: lowest band (n=1)
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Lowest band (n=1) statistics ===\n")

function band_stats(eigenvalues, eigenvalues_qp, εF, band_idx)
    e_ks = [(eigenvalues[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(eigenvalues)]
    e_qp = [(eigenvalues_qp[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(eigenvalues_qp)]
    return (depth_ks=minimum(e_ks), depth_qp=minimum(e_qp),
            bw_ks=maximum(e_ks)-minimum(e_ks), bw_qp=maximum(e_qp)-minimum(e_qp))
end

s_lda = band_stats(ks_lda["eigenvalues"], eqp_lda, ks_lda["εF"], 1)
s_pbe = band_stats(ks_pbe["eigenvalues"], eqp_pbe, ks_pbe["εF"], 1)

# Free-electron reference
lattice_matrix = ks_lda["lattice_matrix"]
V_prim = abs(det(lattice_matrix))
n_e = ks_lda["Z_val"]
k_F = (3π^2 * n_e / V_prim)^(1/3)
E_F_free = k_F^2 / 2

@printf("  %-22s  %10s  %10s  %10s  %10s  %8s\n",
        "Band 1", "KS Γ(eV)", "QP Γ(eV)", "BW_KS(eV)", "BW_QP(eV)", "QP/KS")
println("  " * "-"^76)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "LDA Zion=4", s_lda.depth_ks, s_lda.depth_qp, s_lda.bw_ks, s_lda.bw_qp,
        s_lda.depth_qp / s_lda.depth_ks)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "PBE Zion=4", s_pbe.depth_ks, s_pbe.depth_qp, s_pbe.bw_ks, s_pbe.bw_qp,
        s_pbe.depth_qp / s_pbe.depth_ks)
@printf("  %-22s  %+10.3f\n", "Free electron", -E_F_free * Ha_to_eV)

narrowing_lda = (1 - s_lda.depth_qp / s_lda.depth_ks) * 100
narrowing_pbe = (1 - s_pbe.depth_qp / s_pbe.depth_ks) * 100
@printf("\n  Narrowing (n=1): LDA=%.1f%%, PBE=%.1f%%\n", narrowing_lda, narrowing_pbe)

# ══════════════════════════════════════════════════════════════════════════════
# Band gap analysis
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Band gap analysis ===\n")

# Si has 4 valence electrons per atom, 2 atoms = 8 electrons = 4 occupied bands
n_occ = 4

function find_kpoint(k_coords, target; tol=0.02)
    for (ik, kc) in enumerate(k_coords)
        norm(kc .- target) < tol && return ik
    end
    return nothing
end

function gap_analysis(eigenvalues, eigenvalues_qp, εF, k_coordinates, label)
    n_kpts = length(eigenvalues)

    global vbm_ks = -Inf; global cbm_ks = Inf
    global vbm_qp = -Inf; global cbm_qp = Inf
    global vbm_ik = 0; global cbm_ik = 0

    for ik in 1:n_kpts
        e_v_ks = eigenvalues[ik][n_occ]
        e_c_ks = eigenvalues[ik][n_occ + 1]
        e_v_qp = eigenvalues_qp[ik][n_occ]
        e_c_qp = eigenvalues_qp[ik][n_occ + 1]
        if e_v_ks > vbm_ks
            global vbm_ks = e_v_ks
            global vbm_ik = ik
        end
        if e_c_ks < cbm_ks
            global cbm_ks = e_c_ks
            global cbm_ik = ik
        end
        global vbm_qp = max(vbm_qp, e_v_qp)
        global cbm_qp = min(cbm_qp, e_c_qp)
    end

    gap_ks = (cbm_ks - vbm_ks) * Ha_to_eV
    gap_qp = (cbm_qp - vbm_qp) * Ha_to_eV

    @printf("  [%s] KS  indirect gap = %.4f eV\n", label, gap_ks)
    @printf("  [%s] QP  indirect gap = %.4f eV\n", label, gap_qp)
    @printf("  [%s] Gap change = %+.4f eV (%.1f%%)\n", label,
            gap_qp - gap_ks, (gap_qp - gap_ks) / gap_ks * 100)

    # Direct gap at Γ
    iΓ = find_kpoint(k_coordinates, [0.0, 0.0, 0.0])
    global dgap_ks = NaN; global dgap_qp = NaN
    if !isnothing(iΓ)
        dgap_ks = (eigenvalues[iΓ][n_occ+1] - eigenvalues[iΓ][n_occ]) * Ha_to_eV
        dgap_qp = (eigenvalues_qp[iΓ][n_occ+1] - eigenvalues_qp[iΓ][n_occ]) * Ha_to_eV
        @printf("  [%s] KS  direct gap at Γ = %.4f eV\n", label, dgap_ks)
        @printf("  [%s] QP  direct gap at Γ = %.4f eV\n", label, dgap_qp)
    end

    return (; gap_ks, gap_qp, dgap_ks, dgap_qp)
end

println("  --- LDA ---")
gaps_lda = gap_analysis(ks_lda["eigenvalues"], eqp_lda, ks_lda["εF"],
                        ks_lda["k_coordinates"], "LDA")
println()
println("  --- PBE ---")
gaps_pbe = gap_analysis(ks_pbe["eigenvalues"], eqp_pbe, ks_pbe["εF"],
                        ks_pbe["k_coordinates"], "PBE")

println("\n  Experiment: ~1.17 eV (indirect), ~3.4 eV (direct at Γ)")

# ══════════════════════════════════════════════════════════════════════════════
# Δ at Γ
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Δ at high-symmetry points ===\n")

n_bands_lda = length(ks_lda["eigenvalues"][1])
n_bands_pbe = length(ks_pbe["eigenvalues"][1])

iΓ_lda = find_kpoint(ks_lda["k_coordinates"], [0.0, 0.0, 0.0])
iΓ_pbe = find_kpoint(ks_pbe["k_coordinates"], [0.0, 0.0, 0.0])

if !isnothing(iΓ_lda)
    @printf("  LDA Δ at Γ:\n")
    for n in 1:min(6, n_bands_lda)
        @printf("    n=%d: Δ=%.6f\n", n, deltas_lda[iΓ_lda][n])
    end
end
if !isnothing(iΓ_pbe)
    @printf("  PBE Δ at Γ:\n")
    for n in 1:min(6, n_bands_pbe)
        @printf("    n=%d: Δ=%.6f\n", n, deltas_pbe[iΓ_pbe][n])
    end
end

if !isnothing(iΓ_lda) && !isnothing(iΓ_pbe)
    @printf("\n  Δ(Γ,n=1): LDA=%.6f, PBE=%.6f\n",
            deltas_lda[iΓ_lda][1], deltas_pbe[iΓ_pbe][1])
end

println("\n  Channel breakdown at K=0:")
for ff in form_factors
    Δc = ff.f0^2 / ff.ΔE^2
    @printf("    %s: Δ_c(0) = %.6f  (%.1f%%)\n", ff.name, Δc, 100Δc/delta_K(0.0))
end

# ══════════════════════════════════════════════════════════════════════════════
# Summary comparison table
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Summary comparison ===\n")

@printf("  %-16s  %10s  %10s  %10s  %10s\n",
        "", "KS gap", "QP gap", "KS Γ gap", "QP Γ gap")
println("  " * "-"^60)
@printf("  %-16s  %10.4f  %10.4f  %10.4f  %10.4f\n",
        "LDA Zion=4", gaps_lda.gap_ks, gaps_lda.gap_qp,
        gaps_lda.dgap_ks, gaps_lda.dgap_qp)
@printf("  %-16s  %10.4f  %10.4f  %10.4f  %10.4f\n",
        "PBE Zion=4", gaps_pbe.gap_ks, gaps_pbe.gap_qp,
        gaps_pbe.dgap_ks, gaps_pbe.dgap_qp)
@printf("  %-16s  %10s  %10s  %10s  %10s\n",
        "Experiment", "1.17", "", "3.4", "")

# ══════════════════════════════════════════════════════════════════════════════
# Save QP data
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Saving QP data ===")

# Build plotting arrays from KS data + QP eigenvalues
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

outfile = joinpath(outdir, "si_pbe_qp_data.jld2")
jldsave(outfile;
    # LDA QP results
    eqp_lda, deltas_lda, eqp_arr_lda, deltas_arr_lda,
    depth_ks_lda = s_lda.depth_ks, depth_qp_lda = s_lda.depth_qp,
    bw_ks_lda = s_lda.bw_ks, bw_qp_lda = s_lda.bw_qp,
    gap_ks_lda = gaps_lda.gap_ks, gap_qp_lda = gaps_lda.gap_qp,
    # PBE QP results
    eqp_pbe, deltas_pbe, eqp_arr_pbe, deltas_arr_pbe,
    depth_ks_pbe = s_pbe.depth_ks, depth_qp_pbe = s_pbe.depth_qp,
    bw_ks_pbe = s_pbe.bw_ks, bw_qp_pbe = s_pbe.bw_qp,
    gap_ks_pbe = gaps_pbe.gap_ks, gap_qp_pbe = gaps_pbe.gap_qp,
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
