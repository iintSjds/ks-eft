"""
run_ks_pbe.jl (calcium)
Stage 1: Run KS-DFT (SCF + bands) with PBE functional and save all data needed for QP correction.

FCC Ca, Zion=10, [Ne] core (smallcore), GTH PBE PSP.
This is a robustness test: LDA uses Zion=2 ([Ar] core), PBE uses Zion=10 ([Ne] core).
10 valence electrons: 3s² 3p⁶ 4s² → 6 occupied bands.
Saves: calcium/ca_pbe_ks_data.jld2

Usage: julia --project=. calcium/run_ks_pbe.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] calcium/run_ks_pbe.jl")
    println("  Computes: Ca SCF + bands (PBE GTH Zion=10, smallcore, FCC)")
    println("  Saves: calcium/ca_pbe_ks_data.jld2")
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

a = 10.545  # Bohr (5.58 Å)
lattice = (a / 2) * [[0  1  1]; [1  0  1]; [1  1  0]]  # primitive FCC
positions = [[0.0, 0.0, 0.0]]

pf = PseudoFamily("cp2k.nc.sr.pbe.v0_1.smallcore.gth")
psp = load_psp(pf, :Ca)
@printf("PSP: %s (Zion=%d, lmax=%d)\n", psp.identifier, psp.Zion, psp.lmax)

model = model_PBE(lattice, [ElementPsp(:Ca, psp)], positions;
                  temperature=0.001, smearing=DFTK.Smearing.FermiDirac())

Ecut = 40.0
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
# 10 valence electrons: 3s(2) + 3p(6) + 4s(2) = 6 occupied bands
# Need extra unoccupied bands
n_bands_calc = 12
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
outfile = joinpath(outdir, "ca_pbe_ks_data.jld2")
jldsave(outfile;
    psi, eigenvalues, k_coordinates, G_vectors, recip_lattice,
    εF = scfres.εF,
    kdistances, eigenvalues_array, tick_distances, tick_labels,
    n_spin, krange_spin_map, lattice_matrix,
    Ecut, kgrid, n_bands = n_bands_calc,
    psp_identifier = psp.identifier,
    Zion = Int(psp.Zion),
    element = "Ca", Z_nuc = 20, Z_val = 10,
    structure = "FCC", a_bohr = a,
    conduction_band = 5,
)
println("→ $outfile")
@printf("  %d k-points, %d bands, %d G-vectors (max)\n",
        n_kpts, n_bands_calc, maximum(size.(psi, 1)))

println("\n=== Done ===")
