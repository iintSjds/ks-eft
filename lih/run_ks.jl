"""
run_ks.jl (LiH)
Stage 1: KS-DFT (SCF + bands) for lithium hydride.

LiH rock salt structure (Fm3̄m): Li at (0,0,0), H at (½,½,½).
Uses cp2k LDA largecore PSPs (Li Zion=1, H Zion=1).

Saves plain arrays to lih_ks_data.jld2 for Stage 2 (freq_correction.jl).

Usage: julia --project=. lih/run_ks.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] lih/run_ks.jl")
    println("  Computes: LiH SCF + bands (LDA, cp2k largecore)")
    println("  Saves: lih/lih_ks_data.jld2")
    exit(0)
end

using Printf
using LinearAlgebra
using Unitful
using UnitfulAtomic
using DFTK
using PseudoPotentialData
using JLD2

outdir = @__DIR__
Ha_to_eV = 27.211386245988

# ══════════════════════════════════════════════════════════════════════════════
# Crystal structure: LiH rock salt
# ══════════════════════════════════════════════════════════════════════════════

a_conv = 7.714  # conventional lattice constant (Bohr), expt ≈ 4.083 Å

# FCC primitive lattice vectors
lattice = (a_conv / 2) * [[0  1  1]; [1  0  1]; [1  1  0]]

# Fractional coordinates in primitive cell
pos_Li = [0.0, 0.0, 0.0]
pos_H  = [0.5, 0.5, 0.5]

println("=== LiH rock salt ===")
@printf("  a = %.3f Bohr = %.3f Å\n", a_conv, a_conv * 0.529177)

# ══════════════════════════════════════════════════════════════════════════════
# SCF + bands
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== SCF + bands ===\n")

pf = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
psp_li = load_psp(pf, :Li)
psp_h  = load_psp(pf, :H)

@printf("  Li PSP: Zion=%d\n", psp_li.Zion)
@printf("  H  PSP: Zion=%d\n", psp_h.Zion)

atoms = [ElementPsp(:Li, psp_li), ElementPsp(:H, psp_h)]
positions = [pos_Li, pos_H]

Ecut = 30.0
kgrid = [6, 6, 6]
n_bands = 8

model = model_LDA(lattice, atoms, positions)
basis = PlaneWaveBasis(model; Ecut, kgrid)

println("  SCF starting...")
t0 = time()
scfres = self_consistent_field(basis; tol=1e-8,
                               is_converged=DFTK.ScfConvergenceEnergy(1e-8))
@printf("  SCF done in %.1f s, εF = %.6f Ha\n", time() - t0, scfres.εF)

bands = compute_bands(scfres; n_bands, kline_density=20u"bohr")

# ══════════════════════════════════════════════════════════════════════════════
# Extract plain arrays and save
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Saving KS data ===\n")

dat = DFTK.data_for_plotting(bands)

# Bloch coefficients for QP correction
ψ_list = []
kcoords = []
Gvecs_list = []
recip_lat = basis.model.recip_lattice

for (ik, kpt) in enumerate(bands.basis.kpoints)
    push!(ψ_list, bands.ψ[ik])
    push!(kcoords, Vector{Float64}(kpt.coordinate))
    # Convert to plain Vector{Vector{Int}} for JLD2 portability
    push!(Gvecs_list, [Vector{Int}(g) for g in DFTK.G_vectors(bands.basis, kpt)])
end

jldopen(joinpath(outdir, "lih_ks_data.jld2"), "w") do f
    f["eigenvalues"] = dat.eigenvalues
    f["kdistances"] = dat.kdistances
    f["tick_distances"] = collect(Float64, dat.ticks.distances)
    f["tick_labels"] = collect(String, dat.ticks.labels)
    f["εF"] = scfres.εF
    f["n_bands"] = n_bands
    f["n_kpts"] = length(bands.basis.kpoints)
    f["a_conv"] = a_conv
    f["Ecut"] = Ecut
    f["recip_lattice"] = Matrix{Float64}(recip_lat)
    f["ψ"] = ψ_list
    f["kcoords"] = kcoords
    f["Gvecs"] = Gvecs_list
end

println("  → lih_ks_data.jld2")

# Print Γ-point eigenvalues
for (ik, kpt) in enumerate(bands.basis.kpoints)
    if norm(kpt.coordinate) < 0.02
        println("\n  Γ-point KS eigenvalues (eV below εF):")
        for n in 1:min(length(bands.eigenvalues[ik]), 8)
            @printf("    n=%d: %+.3f eV\n", n,
                    (bands.eigenvalues[ik][n] - scfres.εF) * Ha_to_eV)
        end
        break
    end
end

println("\n=== Done ===")
