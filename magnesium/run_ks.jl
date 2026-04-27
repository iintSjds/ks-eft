"""
run_ks.jl (magnesium)
Stage 1: Run KS-DFT (SCF + bands) and save all data needed for QP correction.

HCP Mg, Zion=2, [Ne] core, GTH LDA largecore PSP.
Saves: magnesium/mg_ks_data.jld2

Usage: julia --project=. magnesium/run_ks.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] magnesium/run_ks.jl")
    println("  Computes: Mg SCF + bands (GTH Zion=2, HCP)")
    println("  Saves: magnesium/mg_ks_data.jld2")
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

a = 6.066       # Bohr (3.21 Å)
ca_ratio = 1.624
c = a * ca_ratio

lattice = [a    a/2        0.0;
           0.0  a*sqrt(3)/2  0.0;
           0.0  0.0          c  ]

positions = [[0.0, 0.0, 0.0], [1/3, 2/3, 1/2]]

pf = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
psp = load_psp(pf, :Mg)
@printf("PSP: %s (Zion=%d, lmax=%d)\n", psp.identifier, psp.Zion, psp.lmax)

atoms = [ElementPsp(:Mg, psp), ElementPsp(:Mg, psp)]
model = model_LDA(lattice, atoms, positions;
                  temperature=0.001, smearing=DFTK.Smearing.FermiDirac())

Ecut = 20.0
kgrid = [8, 8, 6]
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
n_bands_calc = 8
kld = 20u"bohr"
t1 = time()
bands = compute_bands(scfres; n_bands=n_bands_calc, kline_density=kld)
@printf("Band computation done in %.1f s\n", time() - t1)

# ══════════════════════════════════════════════════════════════════════════════
# Extract and save all data
# ══════════════════════════════════════════════════════════════════════════════

println("\n=== Saving KS data ===")

n_kpts = length(bands.basis.kpoints)

# Extract wavefunctions as plain arrays
psi = [Matrix(bands.ψ[ik]) for ik in 1:n_kpts]

# Extract eigenvalues
eigenvalues = [collect(bands.eigenvalues[ik]) for ik in 1:n_kpts]

# Extract k-point coordinates
k_coordinates = [Vector{Float64}(kpt.coordinate) for kpt in bands.basis.kpoints]

# Extract G-vectors as matrices of integers (N_G × 3)
G_vectors = Vector{Matrix{Int}}(undef, n_kpts)
for ik in 1:n_kpts
    Gvecs = collect(DFTK.G_vectors(bands.basis, bands.basis.kpoints[ik]))
    G_vectors[ik] = reduce(hcat, [[G[1], G[2], G[3]] for G in Gvecs])'  |> Matrix
end

# Reciprocal lattice
recip_lattice = Matrix{Float64}(bands.basis.model.recip_lattice)

# Plotting data
dat = DFTK.data_for_plotting(bands)
kdistances = dat.kdistances
eigenvalues_array = dat.eigenvalues  # (n_kpts_plot, n_bands, n_spin)
tick_distances = dat.ticks.distances
tick_labels = dat.ticks.labels
n_spin = dat.n_spin

# k-point index mapping for plotting (krange_spin)
krange_spin_map = [collect(DFTK.krange_spin(bands.basis, σ)) for σ in 1:n_spin]

# Lattice matrix for free-electron reference
lattice_matrix = Matrix{Float64}(model.lattice)

# Save
outfile = joinpath(outdir, "mg_ks_data.jld2")
jldsave(outfile;
    psi, eigenvalues, k_coordinates, G_vectors, recip_lattice,
    εF = scfres.εF,
    kdistances, eigenvalues_array, tick_distances, tick_labels,
    n_spin, krange_spin_map, lattice_matrix,
    Ecut, kgrid, n_bands = n_bands_calc,
    psp_identifier = psp.identifier,
    Zion = Int(psp.Zion),
    element = "Mg", Z_nuc = 12, Z_val = 2,
    structure = "HCP", a_bohr = a, c_bohr = c, ca_ratio = ca_ratio,
)
println("→ $outfile")
@printf("  %d k-points, %d bands, %d G-vectors (max)\n",
        n_kpts, n_bands_calc, maximum(size.(psi, 1)))

println("\n=== Done ===")
