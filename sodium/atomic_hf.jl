"""
atomic_hf.jl
Radial atomic DFT-LDA solver for closed-shell atoms.

Solves the Kohn-Sham equations self-consistently on a uniform radial grid:
  [-1/2 d²/dr² + l(l+1)/(2r²) + V_eff(r)] u_nl(r) = ε_nl u_nl(r)
  V_eff(r) = -Z/r + V_H(ρ) + V_x(ρ)

Uses LDA exchange (Dirac formula, no correlation) for the self-consistent
potential. The resulting orbitals are used to construct the EFT pseudopotential.

Usage: include("atomic_hf.jl"); orbs = solve_atom(11, [(1,0,2),(2,0,2),(2,1,6),(3,0,1)])
"""

using Printf
using LinearAlgebra
using LinearAlgebra: LAPACK

# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

struct AtomicOrbital
    n::Int          # principal quantum number
    l::Int          # angular momentum
    occ::Float64    # occupation (electrons)
    eps::Float64    # eigenvalue (Ha)
    u::Vector{Float64}  # radial wavefunction u(r) = r R(r), ∫u²dr = 1
end

struct AtomicResult
    Z::Int
    orbitals::Vector{AtomicOrbital}
    rgrid::Vector{Float64}
    dr::Float64
    V_H::Vector{Float64}      # Hartree potential
    V_x::Vector{Float64}      # exchange potential
    rho::Vector{Float64}      # total electron density (radial: ρ(r), so ∫4πr²ρ dr = N_e)
    E_total::Float64
end

# ══════════════════════════════════════════════════════════════════════════════
# Hartree potential from radial density
# ══════════════════════════════════════════════════════════════════════════════

"""
    compute_V_H(rho_4pi_r2, rgrid, dr) → V_H

Compute Hartree potential from charge density on a radial grid.
Input: rho_4pi_r2[i] = 4π r² ρ(r_i) × dr  (charge in each shell)
Actually, we take ρ(r) as input (density per volume) and compute:

V_H(r) = (1/r) ∫₀ʳ 4π r'² ρ(r') dr' + ∫ᵣ^∞ 4π r' ρ(r') dr'

Using u_nl(r) = r R_nl(r), the density is:
ρ(r) = Σ_nl (n_nl / 4π) |R_nl(r)|² = Σ_nl (n_nl / 4π) u_nl(r)² / r²
"""
function compute_V_H(rho::Vector{Float64}, rgrid::Vector{Float64}, dr::Float64)
    N = length(rgrid)
    V_H = zeros(N)

    # Cumulative integral from 0 to r: Q(r) = ∫₀ʳ 4π r'² ρ(r') dr'
    Q = zeros(N)
    for i in 1:N
        r = rgrid[i]
        Q[i] = (i > 1 ? Q[i-1] : 0.0) + 4π * r^2 * rho[i] * dr
    end

    # Integral from r to ∞: I(r) = ∫ᵣ^∞ 4π r' ρ(r') dr'
    I_tail = zeros(N)
    for i in N:-1:1
        r = rgrid[i]
        I_tail[i] = (i < N ? I_tail[i+1] : 0.0) + 4π * r * rho[i] * dr
    end

    for i in 1:N
        r = rgrid[i]
        V_H[i] = Q[i] / r + I_tail[i]
    end

    return V_H
end

# ══════════════════════════════════════════════════════════════════════════════
# LDA exchange potential (Dirac formula)
# ══════════════════════════════════════════════════════════════════════════════

"""
LDA exchange potential: V_x(r) = -(3ρ(r)/π)^{1/3}
"""
function compute_V_x(rho::Vector{Float64})
    V_x = similar(rho)
    for i in eachindex(rho)
        ρ = max(rho[i], 0.0)
        V_x[i] = ρ > 1e-30 ? -(3ρ / π)^(1/3) : 0.0
    end
    return V_x
end

# ══════════════════════════════════════════════════════════════════════════════
# Radial eigenvalue solver for a given l
# ══════════════════════════════════════════════════════════════════════════════

"""
    solve_radial(V_eff, l, rgrid, dr; n_states=3) → (evals, evecs)

Solve [-1/2 d²/dr² + l(l+1)/(2r²) + V_eff(r)] u(r) = ε u(r)
on a uniform grid with u(0) = u(r_max) = 0.

Returns the lowest n_states eigenvalues and normalized eigenvectors.
"""
function solve_radial(V_eff::Vector{Float64}, l::Int,
                      rgrid::Vector{Float64}, dr::Float64;
                      n_states::Int=5)
    N = length(rgrid)

    # Build tridiagonal Hamiltonian
    diag_main = zeros(N)
    for i in 1:N
        r = rgrid[i]
        centrifugal = l * (l + 1) / (2.0 * r^2)
        diag_main[i] = 1.0 / dr^2 + centrifugal + V_eff[i]
    end
    diag_off = fill(-1.0 / (2.0 * dr^2), N - 1)

    # Use LAPACK stegr! to compute only the lowest n_states eigenvalues/vectors
    # This is O(N·k) instead of O(N³) for full diagonalization
    dv = copy(diag_main)
    ev = copy(diag_off)
    evals, evecs = LAPACK.stegr!('V', 'I', dv, ev, 0.0, 0.0, 1, n_states)

    return evals, evecs
end

# ══════════════════════════════════════════════════════════════════════════════
# Main atomic solver
# ══════════════════════════════════════════════════════════════════════════════

"""
    solve_atom(Z, config; dr=0.002, r_max=40.0, tol=1e-8, maxiter=200, mixing=0.3)

Solve the atomic Kohn-Sham equations self-consistently.

Arguments:
  Z: nuclear charge
  config: vector of (n, l, occ) tuples, e.g. [(1,0,2), (2,0,2), (2,1,6), (3,0,1)]
  dr: grid spacing (Bohr)
  r_max: grid extent (Bohr)
  tol: convergence threshold on eigenvalues (Ha)
  maxiter: maximum SCF iterations
  mixing: linear mixing parameter for density

Returns: AtomicResult
"""
function solve_atom(Z::Int, config::Vector{Tuple{Int,Int,Int}};
                    dr::Float64=0.002, r_max::Float64=40.0,
                    tol::Float64=1e-8, maxiter::Int=300, mixing::Float64=0.3)
    N = round(Int, r_max / dr) - 1
    rgrid = [(i + 1) * dr for i in 0:(N-1)]  # r from dr to (r_max - dr)

    # Determine unique l values and which states belong to each
    l_values = unique([c[2] for c in config])

    # Initialize density with Thomas-Fermi or hydrogenic guess
    rho = zeros(N)
    N_electrons = sum(c[3] for c in config)
    for i in 1:N
        r = rgrid[i]
        # Simple exponential guess: ρ ∝ exp(-2Z r / n_max)
        rho[i] = N_electrons * (Z / π) * exp(-2.0 * Z * r) / (4π)
    end
    # Normalize to N_electrons
    norm_rho = 4π * dr * sum(rgrid[i]^2 * rho[i] for i in 1:N)
    rho .*= N_electrons / norm_rho

    # Storage for orbitals
    orbitals = Vector{AtomicOrbital}(undef, length(config))
    eps_old = zeros(length(config))

    V_H = zeros(N)
    V_x = zeros(N)

    @printf("=== Atomic solver: Z=%d, N_e=%d ===\n", Z, N_electrons)
    @printf("  Grid: %d pts, dr=%.4f, r_max=%.1f Bohr\n", N, dr, r_max)
    lchars = "spdf"
    @printf("  Config: %s\n", join(["$(n)$(lchars[l+1])^$(occ)" for (n,l,occ) in config], " "))

    converged = false
    for iter in 1:maxiter
        # Build potentials
        V_H = compute_V_H(rho, rgrid, dr)
        V_x = compute_V_x(rho)

        V_eff = [-Float64(Z) / rgrid[i] + V_H[i] + V_x[i] for i in 1:N]

        # Solve for each l value
        rho_new = zeros(N)
        for l in l_values
            # States with this l
            states_l = [(idx, n, occ) for (idx, (n, ll, occ)) in enumerate(config) if ll == l]
            n_needed = length(states_l)

            evals_l, evecs_l = solve_radial(V_eff, l, rgrid, dr; n_states=max(n_needed + 2, 5))

            # Find bound states
            bound = findall(e -> e < 0, evals_l)
            if length(bound) < n_needed
                @printf("  WARNING: only %d bound states for l=%d (need %d)\n",
                        length(bound), l, n_needed)
            end

            # Assign states: for l=0, the n-th s orbital is the n-th eigenstate
            # For l=1, the first p orbital (n=2) is the 1st eigenstate, etc.
            for (state_idx, (idx, n, occ)) in enumerate(states_l)
                if state_idx <= length(bound)
                    i_ev = bound[state_idx]
                else
                    i_ev = state_idx
                end

                eps_nl = evals_l[i_ev]
                u_nl = evecs_l[:, i_ev]

                # Normalize
                norm = sqrt(dr * sum(u_nl .^ 2))
                u_nl ./= norm

                # Convention: first significant lobe positive
                i_first = findfirst(x -> abs(x) > 1e-10, u_nl)
                if !isnothing(i_first) && u_nl[i_first] < 0
                    u_nl .*= -1
                end

                orbitals[idx] = AtomicOrbital(n, l, Float64(occ), eps_nl, copy(u_nl))

                # Add to new density: ρ(r) = Σ (occ / 4π) |R_nl|² = Σ (occ / 4π) u²/r²
                for i in 1:N
                    rho_new[i] += occ / (4π) * u_nl[i]^2 / rgrid[i]^2
                end
            end
        end

        # Check convergence
        eps_new = [orb.eps for orb in orbitals]
        delta_eps = maximum(abs.(eps_new .- eps_old))

        if iter % 20 == 1 || delta_eps < tol * 10
            @printf("  iter %3d: Δε = %.2e", iter, delta_eps)
            for (idx, (n, l, occ)) in enumerate(config)
                @printf("  ε_%d%s=%.6f", n, "spdf"[l+1], orbitals[idx].eps)
            end
            println()
        end

        eps_old .= eps_new

        if delta_eps < tol && iter > 3
            @printf("  Converged in %d iterations (Δε = %.2e)\n", iter, delta_eps)
            converged = true
            break
        end

        # Mix density
        rho .= (1 - mixing) .* rho .+ mixing .* rho_new
    end

    if !converged
        @printf("  WARNING: not converged after %d iterations\n", maxiter)
    end

    # Compute total energy (approximate: sum of eigenvalues - double counting)
    E_eig = sum(orb.occ * orb.eps for orb in orbitals)
    E_H = 0.5 * 4π * dr * sum(rgrid[i]^2 * rho[i] * V_H[i] for i in 1:N)
    # Exchange energy: E_x = -(3/4)(3/π)^{1/3} ∫ ρ^{4/3} d³r
    E_x = -(3/4) * (3/π)^(1/3) * 4π * dr * sum(rgrid[i]^2 * max(rho[i], 0.0)^(4/3) for i in 1:N)
    # V_x contribution to eigenvalue sum
    E_vx = 4π * dr * sum(rgrid[i]^2 * rho[i] * V_x[i] for i in 1:N)
    E_total = E_eig - E_H + E_x - E_vx

    @printf("  E_eig = %.6f Ha, E_H = %.6f Ha, E_x = %.6f Ha\n", E_eig, E_H, E_x)
    @printf("  E_total ≈ %.6f Ha\n", E_total)

    return AtomicResult(Z, orbitals, rgrid, dr, V_H, V_x, rho, E_total)
end

# ══════════════════════════════════════════════════════════════════════════════
# Utility: compute Hartree potential from a SINGLE orbital
# ══════════════════════════════════════════════════════════════════════════════

"""
    orbital_coulomb_potential(u_nl, occ, rgrid, dr) → V

Compute the Coulomb (Hartree) potential from a single orbital with occupation occ:
V(r) = occ × [1/r ∫₀ʳ u²(r') dr' + ∫ᵣ^∞ u²(r')/r' dr']
"""
function orbital_coulomb_potential(u_nl::Vector{Float64}, occ::Float64,
                                   rgrid::Vector{Float64}, dr::Float64)
    N = length(rgrid)
    V = zeros(N)

    # Q(r) = ∫₀ʳ u²(r') dr'  (cumulative)
    Q = zeros(N)
    Q[1] = u_nl[1]^2 * dr
    for i in 2:N
        Q[i] = Q[i-1] + u_nl[i]^2 * dr
    end

    # I(r) = ∫ᵣ^∞ u²(r')/r' dr'  (reverse cumulative)
    I_tail = zeros(N)
    I_tail[N] = u_nl[N]^2 / rgrid[N] * dr
    for i in (N-1):-1:1
        I_tail[i] = I_tail[i+1] + u_nl[i]^2 / rgrid[i] * dr
    end

    for i in 1:N
        V[i] = occ * (Q[i] / rgrid[i] + I_tail[i])
    end

    return V
end

# ══════════════════════════════════════════════════════════════════════════════
# Utility: compute w_i(r) = ∫ φ_i(r')/|r-r'| d³r' for LDA exchange
# ══════════════════════════════════════════════════════════════════════════════

"""
    orbital_w_function(u_nl, l, rgrid, dr) → w(r)

Compute w(r) = ∫ φ_nlm(r') / |r-r'| d³r' for the EFT exchange LDA formula.

For l=0 (s-orbital):
  w(r) = ∫₀^∞ r'² R(r') / max(r,r') dr' = ∫₀^∞ u(r') r' / max(r,r') dr' ...

Wait, need to be more careful. φ = R(r) Y_lm, and R = u/r.

For l=0:
  w(r) = ∫₀^∞ r'² [u(r')/r'] / max(r,r') dr'
       = ∫₀^∞ u(r') r' / max(r,r') dr'
       = (1/r) ∫₀ʳ u(r') r' dr' + ∫ᵣ^∞ u(r') dr'

Wait, that's not right either. Let me redo.

∫ φ(r')/|r-r'| d³r' for φ = R(r) Y₀₀ = u(r)/(r√(4π)):

= ∫₀^∞ r'² dr' [u(r')/(r'√(4π))] ∫ dΩ' Y₀₀(Ω') / |r-r'|

Using the multipole expansion and orthogonality (picks l=0):
= (4π/(√(4π))) ∫₀^∞ r' u(r') / max(r,r') dr' × Y₀₀(Ω)
= √(4π) × Y₀₀(Ω) × ∫₀^∞ r' u(r') / max(r,r') dr'
= ∫₀^∞ r' u(r') / max(r,r') dr'

For l=1 (p-orbital, summed over m):
The w function after summing over all m gives a spherically symmetric result:
  Σ_m φ_{1m}(r) × w_{1m}(r) = R_21(r) × ∫₀^∞ r'² R_21(r') × (r</r>²) dr'
  = (u(r)/r) × ∫₀^∞ u(r') r' × (min(r,r')/max(r,r')²) dr'
"""
function orbital_w_s(u_nl::Vector{Float64}, rgrid::Vector{Float64}, dr::Float64)
    N = length(rgrid)
    w = zeros(N)

    # For l=0: w(r) = ∫₀^∞ r' u(r') / max(r,r') dr'
    # = (1/r) ∫₀ʳ r' u(r') dr' + ∫ᵣ^∞ u(r') dr'

    # Q(r) = ∫₀ʳ r' u(r') dr'
    Q = zeros(N)
    Q[1] = rgrid[1] * u_nl[1] * dr
    for i in 2:N
        Q[i] = Q[i-1] + rgrid[i] * u_nl[i] * dr
    end

    # I(r) = ∫ᵣ^∞ u(r') dr'
    I_tail = zeros(N)
    I_tail[N] = u_nl[N] * dr
    for i in (N-1):-1:1
        I_tail[i] = I_tail[i+1] + u_nl[i] * dr
    end

    for i in 1:N
        w[i] = Q[i] / rgrid[i] + I_tail[i]
    end

    return w
end

"""
    orbital_exchange_p(u_2p, rgrid, dr) → V_x_p(r)

Compute the l=1 exchange LDA contribution (summed over m).
The contribution is: Σ_m φ_{1m}(r) w_{1m}(r) = (u/r) × integral

For l=1: uses the l=1 multipole of 1/|r-r'|:
  (4π/3) r</r>² = (4π/3) min(r,r')/max(r,r')²

After summing over m (gives 3/(4π) factor from Σ|Y₁ₘ|²=3/4π):

  Σ_m φ_{1m}(r) × [∫ φ_{1m}(r')/|r-r'| d³r']
  = (u(r)/r) × ∫₀^∞ u(r') r' min(r,r')/max(r,r')² dr'

Returns the spherically averaged exchange contribution (radial function).
"""
function orbital_w_p_summed(u_nl::Vector{Float64}, rgrid::Vector{Float64}, dr::Float64)
    N = length(rgrid)
    result = zeros(N)

    # We need I(r) = ∫₀^∞ u(r') r' × min(r,r')/max(r,r')² dr'
    # = ∫₀ʳ u(r') r' × r'/r² dr' + ∫ᵣ^∞ u(r') r' × r/r'² dr'
    # = (1/r²) ∫₀ʳ u(r') r'² dr' + r ∫ᵣ^∞ u(r')/r' dr'

    # Q(r) = ∫₀ʳ u(r') r'² dr'
    Q = zeros(N)
    Q[1] = u_nl[1] * rgrid[1]^2 * dr
    for i in 2:N
        Q[i] = Q[i-1] + u_nl[i] * rgrid[i]^2 * dr
    end

    # I_tail(r) = ∫ᵣ^∞ u(r')/r' dr'
    I_tail = zeros(N)
    I_tail[N] = u_nl[N] / rgrid[N] * dr
    for i in (N-1):-1:1
        I_tail[i] = I_tail[i+1] + u_nl[i] / rgrid[i] * dr
    end

    for i in 1:N
        r = rgrid[i]
        integral = Q[i] / r^2 + r * I_tail[i]
        # The full contribution: (u(r)/r) × integral
        result[i] = u_nl[i] / r * integral
    end

    return result
end
