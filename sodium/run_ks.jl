"""
run_ks.jl (sodium)
Stage 1: Run KS-DFT (SCF + bands) and save all data needed for QP correction.

BCC Na, Zion=1, [Ne] core, GTH LDA largecore PSP.
Saves: sodium/na_ks_data.jld2

Usage: julia --project=. sodium/run_ks.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] sodium/run_ks.jl")
    println("  Computes: Na SCF + bands (GTH Zion=1, BCC)")
    println("  Saves: sodium/na_ks_data.jld2")
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

# Crystal setup
a = 8.107
lattice = (a / 2) * [[-1  1  1]; [ 1 -1  1]; [ 1  1 -1]]
positions = [[0.0, 0.0, 0.0]]

pf = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
psp = load_psp(pf, :Na)
@printf("PSP: %s (Zion=%d, lmax=%d)\n", psp.identifier, psp.Zion, psp.lmax)

model = model_LDA(lattice, [ElementPsp(:Na, psp)], positions;
                  temperature=0.001, smearing=DFTK.Smearing.FermiDirac())

Ecut = 15.0
kgrid = [8, 8, 8]
basis = PlaneWaveBasis(model; Ecut, kgrid)
println("Basis: Ecut=$Ecut, kgrid=$kgrid")

# SCF
println("\n=== SCF ===")
t0 = time()
scfres = self_consistent_field(basis; tol=1e-8, mixing=KerkerMixing(),
                               is_converged=DFTK.ScfConvergenceEnergy(1e-8))
@printf("SCF done in %.1f s\n", time() - t0)
@printf("E_tot = %+.8f Ha\n", scfres.energies.total)
@printf("εF = %+.6f Ha = %+.4f eV\n", scfres.εF, scfres.εF * 27.2114)

# Bands
println("\n=== Bands ===")
n_bands_calc = 4
kld = 20u"bohr"
t1 = time()
bands = compute_bands(scfres; n_bands=n_bands_calc, kline_density=kld)
@printf("Band computation done in %.1f s\n", time() - t1)

# Extract and save
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
n_spin = dat.n_spin
krange_spin_map = [collect(DFTK.krange_spin(bands.basis, σ)) for σ in 1:n_spin]
lattice_matrix = Matrix{Float64}(model.lattice)

outfile = joinpath(outdir, "na_ks_data.jld2")
jldsave(outfile;
    psi, eigenvalues, k_coordinates, G_vectors, recip_lattice,
    εF = scfres.εF,
    kdistances = dat.kdistances,
    eigenvalues_array = dat.eigenvalues,
    tick_distances = dat.ticks.distances,
    tick_labels = dat.ticks.labels,
    n_spin, krange_spin_map, lattice_matrix,
    Ecut, kgrid, n_bands = n_bands_calc,
    psp_identifier = psp.identifier,
    Zion = Int(psp.Zion),
    element = "Na", Z_nuc = 11, Z_val = 1,
    structure = "BCC", a_bohr = a,
)
println("→ $outfile")
@printf("  %d k-points, %d bands, %d G-vectors (max)\n",
        n_kpts, n_bands_calc, maximum(size.(psi, 1)))

println("\n=== Done ===")
