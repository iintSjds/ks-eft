"""
freq_correction_pbe.jl (lithium)
Compare LDA (Zion=1) vs PBE (Zion=3) PSP with the same EFT frequency correction.
Li uses analytic Δ(K) (no atomic solver needed).

LDA largecore: freezes 1s² core → Zion=1, band 1 = 2s valence
PBE largecore: all electrons   → Zion=3, band 1 = 1s, band 2 = 2s conduction

The QP correction is applied to all LDA bands but only to the conduction band
(band 2 = 2s) and above for PBE.

Usage: julia --project=. lithium/freq_correction_pbe.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] lithium/freq_correction_pbe.jl")
    println("  Computes: Li SCF + bands (LDA Zion=1 + PBE Zion=3) + EFT QP comparison")
    println("  Saves: lithium/li_pbe_band_data.jls")
    exit(0)
end

using Printf
using LinearAlgebra
using Unitful
using UnitfulAtomic
using DFTK
using PseudoPotentialData
using Serialization

outdir = @__DIR__
Ha_to_eV = 27.2114

# ══════════════════════════════════════════════════════════════════════════════
# Analytic EFT frequency correction for Li
# ══════════════════════════════════════════════════════════════════════════════

const LI_ALPHA = 3.0            # 1s orbital decay constant (Bohr⁻¹)
const LI_J     = (5.0/8.0) * 3  # 1s-1s Coulomb integral = 1.875 Ha
const LI_E1    = -4.5            # hydrogenic 1s energy (Ha)
const LI_E0    = 2.0 * LI_E1 + LI_J  # Li²⁺ ground state = -7.125 Ha

function eft_A_K(K::Float64)
    Q = K / LI_ALPHA
    Q2 = Q^2
    (Q2 + 33.0) / ((Q2 + 1.0) * (Q2 + 9.0)^2) -
        (LI_J / LI_ALPHA) / (Q2 + 1.0)^2
end

function eft_delta_K(K::Float64)
    A0_factor = 33.0/81.0 - LI_J/LI_ALPHA
    AK = eft_A_K(K)
    denom = (LI_E1 - LI_E0)^2
    64.0 * π / LI_ALPHA * A0_factor * AK / denom
end

@printf("  Δ(K=0) = %.6f\n", eft_delta_K(0.0))

# ══════════════════════════════════════════════════════════════════════════════
# SCF + bands
# ══════════════════════════════════════════════════════════════════════════════

a = 6.632  # BCC Li lattice constant in Bohr
lattice = (a / 2) * [[-1  1  1]; [ 1 -1  1]; [ 1  1 -1]]
positions = [[0.0, 0.0, 0.0]]
kgrid = [8, 8, 8]
kld = 20u"bohr"

println("\n=== SCF + bands ===\n")

# --- LDA largecore Zion=1 ---
pf_lda = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
psp_lda = load_psp(pf_lda, :Li)
@printf("  LDA PSP: Zion=%d\n", psp_lda.Zion)

Ecut_lda = 15.0
n_bands_lda = 4
model_lda = model_LDA(lattice, [ElementPsp(:Li, psp_lda)], positions;
                      temperature=0.001, smearing=DFTK.Smearing.FermiDirac())
basis_lda = PlaneWaveBasis(model_lda; Ecut=Ecut_lda, kgrid)
println("  [LDA] SCF starting...")
t0 = time()
scfres_lda = self_consistent_field(basis_lda; tol=1e-8, mixing=KerkerMixing(),
                                   is_converged=DFTK.ScfConvergenceEnergy(1e-8))
println("  [LDA] SCF done in $(round(time()-t0, digits=1)) s, εF = $(round(scfres_lda.εF, digits=6)) Ha")
bands_lda = compute_bands(scfres_lda; n_bands=n_bands_lda, kline_density=kld)

# --- PBE largecore Zion=3 (all electrons) ---
pf_pbe = PseudoFamily("cp2k.nc.sr.pbe.v0_1.largecore.gth")
psp_pbe = load_psp(pf_pbe, :Li)
@printf("\n  PBE PSP: Zion=%d\n", psp_pbe.Zion)

# Higher Ecut for core 1s state
Ecut_pbe = 30.0
n_bands_pbe = 6
model_pbe = model_PBE(lattice, [ElementPsp(:Li, psp_pbe)], positions;
                      temperature=0.001, smearing=DFTK.Smearing.FermiDirac())
basis_pbe = PlaneWaveBasis(model_pbe; Ecut=Ecut_pbe, kgrid)
println("  [PBE] SCF starting...")
t1 = time()
scfres_pbe = self_consistent_field(basis_pbe; tol=1e-8, mixing=KerkerMixing(),
                                   is_converged=DFTK.ScfConvergenceEnergy(1e-8))
println("  [PBE] SCF done in $(round(time()-t1, digits=1)) s, εF = $(round(scfres_pbe.εF, digits=6)) Ha")
bands_pbe = compute_bands(scfres_pbe; n_bands=n_bands_pbe, kline_density=kld)

# ══════════════════════════════════════════════════════════════════════════════
# Apply QP correction
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== QP correction ===\n")

function delta_nk(ψnk, kpt, basis)
    recip_lat = basis.model.recip_lattice
    k_frac = kpt.coordinate
    Gvecs = DFTK.G_vectors(basis, kpt)
    Δ = 0.0
    for (ig, G) in enumerate(Gvecs)
        K = norm(recip_lat * (k_frac .+ G))
        Δ += abs2(ψnk[ig]) * eft_delta_K(K)
    end
    return Δ
end

function apply_qp(bands, εF; band_indices=nothing)
    n_kpts = length(bands.basis.kpoints)
    n_bands_computed = length(bands.eigenvalues[1])
    if isnothing(band_indices)
        band_indices = 1:n_bands_computed
    end
    eqp = deepcopy(bands.eigenvalues)
    deltas = [zeros(n_bands_computed) for _ in 1:n_kpts]
    for ik in 1:n_kpts
        kpt = bands.basis.kpoints[ik]
        for n in band_indices
            Δ = delta_nk(bands.ψ[ik][:, n], kpt, bands.basis)
            deltas[ik][n] = Δ
            eqp[ik][n] = εF + (bands.eigenvalues[ik][n] - εF) / (1.0 + Δ)
        end
    end
    return eqp, deltas
end

# LDA Zion=1: apply to all bands
println("  LDA Zion=1: applying QP to all bands...")
eqp_lda, deltas_lda = apply_qp(bands_lda, scfres_lda.εF)

# PBE Zion=3: band 1 = 1s (core), band 2 = 2s (conduction)
# Apply correction only to conduction band and above
conduction_band = 2
println("  PBE Zion=3: applying QP to conduction band $conduction_band only...")
eqp_pbe, deltas_pbe = apply_qp(bands_pbe, scfres_pbe.εF;
                                band_indices=conduction_band:n_bands_pbe)

# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════

println("=== Results ===\n")

function find_kpoint(band_data, target_coord; tol=0.02)
    for (ik, kpt) in enumerate(band_data.basis.kpoints)
        norm(kpt.coordinate .- target_coord) < tol && return ik
    end
    return nothing
end

function band_stats(evals, eqp, εF, band_idx)
    e_ks = [(evals[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(evals)]
    e_qp = [(eqp[ik][band_idx] - εF) * Ha_to_eV for ik in eachindex(eqp)]
    return (depth_ks=minimum(e_ks), depth_qp=minimum(e_qp),
            bw_ks=maximum(e_ks)-minimum(e_ks), bw_qp=maximum(e_qp)-minimum(e_qp))
end

# LDA band 1 = 2s valence; PBE band 2 = 2s conduction
s_lda = band_stats(bands_lda.eigenvalues, eqp_lda, scfres_lda.εF, 1)
s_pbe = band_stats(bands_pbe.eigenvalues, eqp_pbe, scfres_pbe.εF, conduction_band)

@printf("  %-22s  %10s  %10s  %10s  %10s  %8s\n",
        "", "KS Γ(eV)", "QP Γ(eV)", "BW_KS(eV)", "BW_QP(eV)", "QP/KS")
println("  " * "-"^76)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "LDA Zion=1", s_lda.depth_ks, s_lda.depth_qp, s_lda.bw_ks, s_lda.bw_qp,
        s_lda.depth_qp / s_lda.depth_ks)
@printf("  %-22s  %+10.3f  %+10.3f  %10.3f  %10.3f  %8.4f\n",
        "PBE Zion=3", s_pbe.depth_ks, s_pbe.depth_qp, s_pbe.bw_ks, s_pbe.bw_qp,
        s_pbe.depth_qp / s_pbe.depth_ks)

narrowing_lda = (1 - s_lda.depth_qp / s_lda.depth_ks) * 100
narrowing_pbe = (1 - s_pbe.depth_qp / s_pbe.depth_ks) * 100
@printf("\n  Narrowing (2s band): LDA=%.1f%%, PBE=%.1f%%\n", narrowing_lda, narrowing_pbe)

iΓ_lda = find_kpoint(bands_lda, [0.0, 0.0, 0.0])
iΓ_pbe = find_kpoint(bands_pbe, [0.0, 0.0, 0.0])
if !isnothing(iΓ_lda) && !isnothing(iΓ_pbe)
    @printf("  Δ(Γ,2s): LDA=%.4f, PBE=%.4f\n",
            deltas_lda[iΓ_lda][1], deltas_pbe[iΓ_pbe][conduction_band])
end

# ══════════════════════════════════════════════════════════════════════════════
# Save full band data
# ══════════════════════════════════════════════════════════════════════════════

function save_band_arrays(bands, eqp, scfres)
    dat = DFTK.data_for_plotting(bands)
    eqp_arr = similar(dat.eigenvalues)
    for σ in 1:dat.n_spin
        for (ito, ik) in enumerate(DFTK.krange_spin(bands.basis, σ))
            eqp_arr[ito, :, σ] = eqp[ik]
        end
    end
    return Dict(
        "kdistances" => dat.kdistances,
        "eigenvalues" => dat.eigenvalues,
        "eqp" => eqp_arr,
        "εF" => scfres.εF,
        "tick_distances" => dat.ticks.distances,
        "tick_labels" => dat.ticks.labels,
        "n_bands" => size(dat.eigenvalues, 2),
    )
end

lda_data = save_band_arrays(bands_lda, eqp_lda, scfres_lda)
pbe_data = save_band_arrays(bands_pbe, eqp_pbe, scfres_pbe)

lda_data["Zion"] = 1
lda_data["depth_ks"] = s_lda.depth_ks
lda_data["depth_qp"] = s_lda.depth_qp
lda_data["bw_ks"] = s_lda.bw_ks
lda_data["bw_qp"] = s_lda.bw_qp

pbe_data["Zion"] = 3
pbe_data["conduction_band"] = conduction_band
pbe_data["depth_ks"] = s_pbe.depth_ks
pbe_data["depth_qp"] = s_pbe.depth_qp
pbe_data["bw_ks"] = s_pbe.bw_ks
pbe_data["bw_qp"] = s_pbe.bw_qp

band_data = Dict("lda" => lda_data, "pbe" => pbe_data)
serialize(joinpath(outdir, "li_pbe_band_data.jls"), band_data)
println("  → li_pbe_band_data.jls")

println("\n=== Done ===")
