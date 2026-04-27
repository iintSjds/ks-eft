"""
run_ks_pbe.jl (lithium)
Stage 1: Run KS-DFT (SCF + bands) with PBE largecore Zion=3 PSP.

BCC Li, Zion=3 (all-electron), PBE largecore PSP.
Band 1 = 1s core state, band 2 = 2s conduction.
Higher Ecut=30 needed for the 1s core state.
Saves: lithium/li_pbe_ks_data.jld2

Usage: julia --project=. lithium/run_ks_pbe.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] lithium/run_ks_pbe.jl")
    println("  Computes: Li SCF + bands (PBE Zion=3, BCC)")
    println("  Saves: lithium/li_pbe_ks_data.jld2")
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

# ══════════════════════════════════════════════════════════════════════════════
# Crystal setup
# ══════════════════════════════════════════════════════════════════════════════

a = 6.632  # Bohr (BCC Li)
lattice = (a / 2) * [[-1  1  1]; [ 1 -1  1]; [ 1  1 -1]]  # primitive BCC
positions = [[0.0, 0.0, 0.0]]

pf = PseudoFamily("cp2k.nc.sr.pbe.v0_1.largecore.gth")
psp = load_psp(pf, :Li)
@printf("PSP: %s (Zion=%d, lmax=%d)\n", psp.identifier, psp.Zion, psp.lmax)

model = model_PBE(lattice, [ElementPsp(:Li, psp)], positions;
                  temperature=0.001, smearing=DFTK.Smearing.FermiDirac())

Ecut = 30.0
kgrid = [8, 8, 8]
basis = PlaneWaveBasis(model; Ecut, kgrid)
println("Basis: Ecut=$Ecut, kgrid=$kgrid")

# ══════════════════════════════════════════════════════════════════════════════
# SCF
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== SCF ===")
t0 = time()
scfres = self_consistent_field(basis; tol=1e-8, mixing=KerkerMixing(),
                               is_converged=DFTK.ScfConvergenceEnergy(1e-8))
@printf("SCF done in %.1f s\n", time() - t0)
@printf("E_tot = %+.8f Ha\n", scfres.energies.total)
@printf("εF = %+.6f Ha = %+.4f eV\n", scfres.εF, scfres.εF * 27.2114)

# ══════════════════════════════════════════════════════════════════════════════
# Band structure
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Bands ===")
n_bands_calc = 6
kld = 20u"bohr"
t1 = time()
bands = compute_bands(scfres; n_bands=n_bands_calc, kline_density=kld)
@printf("Band computation done in %.1f s\n", time() - t1)

# ══════════════════════════════════════════════════════════════════════════════
# Extract and save all data
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Saving KS data ===")

n_kpts = length(bands.basis.kpoints)

psi = [Matrix(bands.ψ[ik]) for ik in 1:n_kpts]
eigenvalues = [collect(bands.eigenvalues[ik]) for ik in 1:n_kpts]
k_coordinates = [Vector{Float64}(kpt.coordinate) for kpt in bands.basis.kpoints]

G_vectors = Vector{Matrix{Int}}(undef, n_kpts)
for ik in 1:n_kpts
    Gvecs = collect(DFTK.G_vectors(bands.basis, bands.basis.kpoints[ik]))
    G_vectors[ik] = reduce(hcat, [[G[1], G[2], G[3]] for G in Gvecs])' |> Matrix
end

recip_lattice = Matrix{Float64}(bands.basis.model.recip_lattice)

dat = DFTK.data_for_plotting(bands)
kdistances = dat.kdistances
eigenvalues_array = dat.eigenvalues
tick_distances = dat.ticks.distances
tick_labels = dat.ticks.labels
n_spin = dat.n_spin

krange_spin_map = [collect(DFTK.krange_spin(bands.basis, σ)) for σ in 1:n_spin]
lattice_matrix = Matrix{Float64}(model.lattice)

outfile = joinpath(outdir, "li_pbe_ks_data.jld2")
jldsave(outfile;
    psi, eigenvalues, k_coordinates, G_vectors, recip_lattice,
    εF = scfres.εF,
    kdistances, eigenvalues_array, tick_distances, tick_labels,
    n_spin, krange_spin_map, lattice_matrix,
    Ecut, kgrid, n_bands = n_bands_calc,
    psp_identifier = psp.identifier,
    Zion = Int(psp.Zion),
    element = "Li", Z_nuc = 3, Z_val = 3,
    structure = "BCC", a_bohr = a,
    conduction_band = 2,
)
println("→ $outfile")
@printf("  %d k-points, %d bands, %d G-vectors (max)\n",
        n_kpts, n_bands_calc, maximum(size.(psi, 1)))

println("\n=== Done ===")
