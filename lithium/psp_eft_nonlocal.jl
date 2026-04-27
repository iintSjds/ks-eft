"""
psp_eft_nonlocal.jl
Norm-conserving pseudopotential for Li from the FULL nonlocal EFT potential.

The EFT pseudopotential (before the LDA approximation) is:
  δV_pp(r,r') = 2u(r)δ(r-r')                              [Hartree with core]
              - φ₁ₛ(r)φ₁ₛ(r')/|r-r'|                      [exchange with 1s]
              - φ₁ₛ(r)φ₁ₛ(r')(u(r)+u(r')-J)               [ΣG₂Σ static part]
              + φ₁ₛ(r)φ₁ₛ(r')(u(r)-J)(u(r')-J)/(iω+E₁-E₀) [dynamical]

The l-decomposition:
  l=0: all three static nonlocal terms contribute
  l≥1: only exchange contributes (projection term vanishes by angular selection)

Strategy:
  1. Solve the l-dependent radial equation with full nonlocal kernel
  2. TM pseudize each l-channel (both l=0 and l=1)
  3. Schrödinger-invert to get semilocal V_sl^l(r)
  4. Kleinman-Bylander transform → V_local + projectors

Usage: julia --project=run/psp run/psp/lithium/psp_eft_nonlocal.jl
"""

using Printf
using LinearAlgebra
using SpecialFunctions: erf, besselj
using Interpolations: linear_interpolation, Line

import DFTK
import DFTK: NormConservingPsp

include(joinpath(@__DIR__, "psp_eft_li.jl"))

# ══════════════════════════════════════════════════════════════════════════════
# Physical constants
# ══════════════════════════════════════════════════════════════════════════════

const NL_Z = Float64(EFT_Z_NUC)     # 3
const NL_ALPHA = Float64(EFT_ALPHA)  # 2.6875 Bohr⁻¹ (variational HF)
const NL_J = (5.0/8.0) * NL_ALPHA   # 1.680 Ha

# Core 1s radial wavefunction: u_1s(r) = r R_1s(r), ∫|u_1s|²dr = 1
# R_1s(r) = 2α^{3/2} exp(-αr), so u_1s(r) = 2α^{3/2} r exp(-αr)
u_1s(r) = 2.0 * NL_ALPHA^1.5 * r * exp(-NL_ALPHA * r)

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Build and solve the nonlocal radial Hamiltonian for each l
# ══════════════════════════════════════════════════════════════════════════════

"""
    exchange_kernel_l0(r_i, r_j)

Exchange kernel for l=0:
  K_x(r,r') = -u_1s(r) u_1s(r') / max(r,r')
"""
function exchange_kernel_l0(r_i::Float64, r_j::Float64)
    -u_1s(r_i) * u_1s(r_j) / max(r_i, r_j)
end

"""
    exchange_kernel_l1(r_i, r_j)

Exchange kernel for l=1:
  K_x(r,r') = -(u_1s(r) u_1s(r') / 3) × r_< / r_>²
"""
function exchange_kernel_l1(r_i::Float64, r_j::Float64)
    r_min = min(r_i, r_j)
    r_max = max(r_i, r_j)
    -(u_1s(r_i) * u_1s(r_j) / 3.0) * r_min / r_max^2
end

"""
    projection_kernel_l0(r_i, r_j)

Projection (ΣG₂Σ static) kernel for l=0 only:
  K_p(r,r') = -u_1s(r) u_1s(r') × [u_eft(r) + u_eft(r') - J]
where u_eft(r) is the EFT u(r) function from psp_eft_li.jl.
"""
function projection_kernel_l0(r_i::Float64, r_j::Float64)
    -u_1s(r_i) * u_1s(r_j) * (eft_u(r_i) + eft_u(r_j) - NL_J)
end

"""
    solve_nonlocal_radial(l; dr=0.01, r_max=30.0)

Solve the radial Schrödinger equation with the full nonlocal EFT potential
for angular momentum l.

H = -½d²/dr² + V_local(r) + l(l+1)/(2r²) + K_l(r,r')

where V_local(r) = -Z/r + 2u(r) and K_l is the nonlocal kernel.

Returns: (eigenvalues, eigenvectors, rgrid, V_bare) where V_bare = -Z/r + 2u(r)
"""
function solve_nonlocal_radial(l::Int; dr::Float64=0.01, r_max::Float64=30.0)
    N = round(Int, r_max / dr) - 1
    rgrid = [(i + 1) * dr for i in 0:(N-1)]

    # Bare local potential (nuclear + Hartree from core) WITHOUT centrifugal
    V_bare = [-NL_Z / r + 2.0 * eft_u(r) for r in rgrid]

    # Full diagonal: kinetic + local + centrifugal
    V_diag = [V_bare[i] + l * (l + 1) / (2.0 * rgrid[i]^2) for i in 1:N]

    # Build tridiagonal kinetic + local potential part
    diag_main = [1.0 / dr^2 + V_diag[i] for i in 1:N]
    diag_off = fill(-1.0 / (2.0 * dr^2), N - 1)
    H = zeros(N, N)
    for i in 1:N
        H[i, i] = diag_main[i]
    end
    for i in 1:(N-1)
        H[i, i+1] = diag_off[i]
        H[i+1, i] = diag_off[i]
    end

    # Add nonlocal kernel
    r_nl_max = 10.0 / NL_ALPHA  # ~3.3 Bohr; beyond this u_1s < 1e-8
    i_nl_max = min(N, round(Int, r_nl_max / dr))

    @printf("  Building nonlocal kernel for l=%d (N=%d, nl_range=%d)...\n", l, N, i_nl_max)

    for i in 1:i_nl_max
        ri = rgrid[i]
        for j in 1:i_nl_max
            rj = rgrid[j]
            if l == 0
                K = exchange_kernel_l0(ri, rj) + projection_kernel_l0(ri, rj)
            elseif l == 1
                K = exchange_kernel_l1(ri, rj)
            else
                K = 0.0
            end
            H[i, j] += K * dr
        end
    end

    # Symmetrize
    H .= 0.5 .* (H .+ H')

    @printf("  Diagonalizing %d×%d matrix...\n", N, N)
    evals, evecs = eigen(Symmetric(H))

    # Normalize eigenvectors: ∫|u|²dr = 1
    for k in 1:min(10, N)
        evecs[:, k] ./= sqrt(dr * sum(evecs[:, k] .^ 2))
        # Convention: first significant lobe positive
        i_peak = argmax(abs.(evecs[:, k]))
        if evecs[i_peak, k] < 0
            evecs[:, k] .*= -1
        end
    end

    return evals, evecs, rgrid, V_bare
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Troullier-Martins pseudization (general l)
# ══════════════════════════════════════════════════════════════════════════════

function count_nodes(u::AbstractVector)
    nodes = 0
    for i in 2:length(u)
        if u[i] * u[i-1] < 0.0
            nodes += 1
        end
    end
    nodes
end

"""
TM pseudization of u_ae for angular momentum l inside r_c.
u_ps(r) = r^{l+1} exp(p(r)) for r < r_c, where p(r) = c0 + c2 r² + c4 r⁴ + ...

7 conditions: value, 4 derivatives at r_c, norm conservation, zero curvature of V_ps at r=0.
The zero-curvature condition is: c4 = -c2²/(2l+5).

Returns u_ps on the full grid.
"""
function tm_pseudize(u_ae::Vector{Float64}, ε::Float64,
                     rgrid::Vector{Float64}, r_c::Float64, l::Int)
    dr = rgrid[2] - rgrid[1]
    i_rc = argmin(abs.(rgrid .- r_c))
    rc = rgrid[i_rc]

    # Define F(r) = u(r) / r^{l+1}, so u_ps = r^{l+1} exp(p(r))
    # φ = log(F) at r_c and derivatives via finite differences
    lp1 = l + 1
    F_at(i) = u_ae[i] / rgrid[i]^lp1
    F_rc = F_at(i_rc)
    φ_rc = log(abs(F_rc))

    h = dr
    # 5-point stencils for derivatives of F
    F1 = (-F_at(i_rc+2) + 8F_at(i_rc+1) - 8F_at(i_rc-1) + F_at(i_rc-2)) / (12h)
    F2 = (-F_at(i_rc+2) + 16F_at(i_rc+1) - 30F_at(i_rc) + 16F_at(i_rc-1) - F_at(i_rc-2)) / (12h^2)
    F3 = (-F_at(i_rc+3) + 8F_at(i_rc+2) - 13F_at(i_rc+1) + 13F_at(i_rc-1) - 8F_at(i_rc-2) + F_at(i_rc-3)) / (8h^3)
    F4 = (-F_at(i_rc+3) + 12F_at(i_rc+2) - 39F_at(i_rc+1) + 56F_at(i_rc) -
          39F_at(i_rc-1) + 12F_at(i_rc-2) - F_at(i_rc-3)) / (6h^4)

    # Convert to derivatives of φ = log(F)
    φ1 = F1 / F_rc
    φ2 = F2 / F_rc - φ1^2
    φ3 = F3 / F_rc - 3φ1 * F2 / F_rc + 2φ1^3
    # More carefully: φ3 = F3/F - 3(F1/F)(F2/F) + 2(F1/F)^3
    φ3 = F3 / F_rc - 3φ1 * (F2 / F_rc) + 2φ1^3
    # φ4 = F4/F - 4(F3/F)(F1/F) - 3(F2/F)^2 + 12(F2/F)(F1/F)^2 - 6(F1/F)^4
    φ4 = F4/F_rc - 4(F3/F_rc)*φ1 - 3(F2/F_rc)^2 + 12(F2/F_rc)*φ1^2 - 6φ1^4

    norm_ae = dr * sum(u_ae[1:i_rc] .^ 2)

    rc2 = rc^2; rc4 = rc^4; rc6 = rc^6; rc8 = rc^8; rc10 = rc^10; rc12 = rc^12

    # Zero-curvature condition: c4 = -c2²/(2l+5)
    zc_denom = Float64(2l + 5)

    function residual(c)
        c0, c2, c4, c6, c8, c10, c12 = c
        p = c0 + c2*rc2 + c4*rc4 + c6*rc6 + c8*rc8 + c10*rc10 + c12*rc12
        p1 = 2c2*rc + 4c4*rc^3 + 6c6*rc^5 + 8c8*rc^7 + 10c10*rc^9 + 12c12*rc^11
        p2 = 2c2 + 12c4*rc2 + 30c6*rc4 + 56c8*rc6 + 90c10*rc8 + 132c12*rc10
        p3 = 24c4*rc + 120c6*rc^3 + 336c8*rc^5 + 720c10*rc^7 + 1320c12*rc^9
        p4 = 24c4 + 360c6*rc2 + 1680c8*rc4 + 5040c10*rc6 + 11880c12*rc8
        F1 = p - φ_rc; F2 = p1 - φ1; F3 = p2 - φ2; F4 = p3 - φ3; F5 = p4 - φ4
        F7 = c4 + c2^2 / zc_denom
        # Norm conservation
        norm_ps = 0.0
        for i in 1:i_rc
            r = rgrid[i]
            pval = c0 + c2*r^2 + c4*r^4 + c6*r^6 + c8*r^8 + c10*r^10 + c12*r^12
            norm_ps += r^(2lp1) * exp(2pval) * dr
        end
        F6 = norm_ps - norm_ae
        [F1, F2, F3, F4, F5, F6, F7]
    end

    function jacobian(c)
        c0, c2, c4, c6, c8, c10, c12 = c
        J = zeros(7, 7)
        J[1,:] = [1, rc2, rc4, rc6, rc8, rc10, rc12]
        J[2,:] = [0, 2rc, 4rc^3, 6rc^5, 8rc^7, 10rc^9, 12rc^11]
        J[3,:] = [0, 2, 12rc2, 30rc4, 56rc6, 90rc8, 132rc10]
        J[4,:] = [0, 0, 24rc, 120rc^3, 336rc^5, 720rc^7, 1320rc^9]
        J[5,:] = [0, 0, 24, 360rc2, 1680rc4, 5040rc6, 11880rc8]
        for j in 1:7
            powers = [0, 2, 4, 6, 8, 10, 12]
            deriv = 0.0
            for i in 1:i_rc
                r = rgrid[i]
                pval = c0 + c2*r^2 + c4*r^4 + c6*r^6 + c8*r^8 + c10*r^10 + c12*r^12
                deriv += r^(2lp1) * exp(2pval) * 2.0 * r^(powers[j]) * dr
            end
            J[6, j] = deriv
        end
        J[7,:] = [0, 2c2/zc_denom, 1, 0, 0, 0, 0]
        J
    end

    c = zeros(7)
    c[1] = φ_rc
    for iter in 1:100
        F = residual(c)
        res_norm = maximum(abs.(F))
        if res_norm < 1e-12
            @printf("  TM(l=%d) converged in %d iterations (res=%.2e)\n", l, iter, res_norm)
            break
        end
        Jac = jacobian(c)
        dc = Jac \ (-F)
        α_step = 1.0
        for _ in 1:20
            c_new = c + α_step * dc
            F_new = residual(c_new)
            if maximum(abs.(F_new)) < res_norm; break; end
            α_step *= 0.5
        end
        c .+= α_step * dc
    end

    u_ps = copy(u_ae)
    for i in 1:i_rc
        r = rgrid[i]
        p = c[1] + c[2]*r^2 + c[3]*r^4 + c[4]*r^6 + c[5]*r^8 + c[6]*r^10 + c[7]*r^12
        u_ps[i] = r^lp1 * exp(p)
    end
    if sign(u_ps[i_rc]) != sign(u_ae[i_rc]) && abs(u_ae[i_rc]) > 1e-15
        u_ps[1:i_rc] .*= -1
    end

    u_ps, c, i_rc
end

"""
Schrödinger inversion: given u_ps and ε, recover V_sl(r).

V_ps(r) = ε + (l+1)p'(r)/r + ½(p'(r))² + ½p''(r)

where p(r) = log(u(r)/r^{l+1}).

Equivalently: V(r) = ε + ½u''(r)/u(r) - l(l+1)/(2r²)
"""
function invert_potential(u_ps::Vector{Float64}, ε::Float64,
                          rgrid::Vector{Float64}; l::Int=0)
    N = length(rgrid)
    dr = rgrid[2] - rgrid[1]
    V = zeros(N)
    for i in 3:(N-2)
        u_pp = (-u_ps[i+2] + 16u_ps[i+1] - 30u_ps[i] + 16u_ps[i-1] - u_ps[i-2]) / (12dr^2)
        if abs(u_ps[i]) > 1e-30
            V[i] = ε + 0.5 * u_pp / u_ps[i] - l * (l + 1) / (2.0 * rgrid[i]^2)
        else
            V[i] = ε
        end
    end
    V[1] = V[3]; V[2] = V[3]; V[N-1] = V[N-2]; V[N] = V[N-2]
    V
end

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: KB PSP construction
# ══════════════════════════════════════════════════════════════════════════════

"""
    PspEFTNonlocal

Norm-conserving pseudopotential from the full nonlocal EFT.
Has V_local from l=1 channel + KB projector for l=0.
"""
struct PspEFTNonlocal{T<:AbstractFloat, I} <: NormConservingPsp
    Zion::Int
    lmax::Int
    h::Vector{Matrix{T}}

    # Local potential (from l=1 semilocal channel, pseudized)
    rgrid::Vector{T}
    vloc::Vector{T}
    vloc_interp::I
    ircut::Int
    rcut::T

    # Projector (l=0 KB)
    proj_rgrid::Vector{T}      # fine grid for projector
    proj_l0::Vector{T}         # p_0(r) = δV(r) × u_ps^0(r)
    proj_l0_interp::I          # interpolator

    # Diagnostics
    r_c::T
    epsilon_1s::T
    epsilon_2s::T
    epsilon_2p::T
    identifier::String
    description::String
end

"""
    PspEFTNonlocal(; r_c=2.5, rcut=15.0, ...)

Build the KB pseudopotential from the full nonlocal EFT.
Both l=0 and l=1 channels are TM-pseudized to get smooth semilocal potentials.
"""
function PspEFTNonlocal(; r_c::Float64=2.5, rcut::Float64=15.0,
                          dr_solve::Float64=0.01, r_max_solve::Float64=30.0,
                          N_dftk::Int=2000)
    T = Float64
    println("=== Constructing nonlocal EFT PSP ===")

    # Step 1: Solve l=0 channel (with exchange + projection)
    println("\n--- l=0 channel (exchange + projection) ---")
    evals0, evecs0, rgrid, V_bare = solve_nonlocal_radial(0; dr=dr_solve, r_max=r_max_solve)

    # Find bound states
    bound0 = findall(e -> e < 0, evals0)
    @printf("  Found %d bound states for l=0\n", length(bound0))
    ε_1s = evals0[bound0[1]]
    ε_2s = evals0[bound0[2]]
    u_1s_ae = evecs0[:, bound0[1]]
    u_2s_ae = evecs0[:, bound0[2]]
    @printf("  ε_1s = %.6f Ha (%.4f eV), nodes=%d\n", ε_1s, ε_1s*27.2114, count_nodes(u_1s_ae))
    @printf("  ε_2s = %.6f Ha (%.4f eV), nodes=%d\n", ε_2s, ε_2s*27.2114, count_nodes(u_2s_ae))

    # Step 2: Solve l=1 channel (exchange only, weak)
    println("\n--- l=1 channel (exchange only) ---")
    evals1, evecs1, _, _ = solve_nonlocal_radial(1; dr=dr_solve, r_max=r_max_solve)

    bound1 = findall(e -> e < 0, evals1)
    ε_2p = NaN
    u_2p_ae = nothing
    if length(bound1) >= 1
        ε_2p = evals1[bound1[1]]
        u_2p_ae = evecs1[:, bound1[1]]
        @printf("  ε_2p = %.6f Ha (%.4f eV), nodes=%d\n", ε_2p, ε_2p*27.2114, count_nodes(u_2p_ae))
    else
        @printf("  No bound l=1 state found.\n")
    end

    # Step 3: TM pseudize l=0 (2s state)
    println("\n--- TM pseudization (l=0, 2s) ---")
    u_ps0, tm_c0, i_rc0 = tm_pseudize(u_2s_ae, ε_2s, rgrid, r_c, 0)
    @printf("  u_ps0 nodes: %d, norm error: %.2e\n",
            count_nodes(u_ps0),
            abs(dr_solve * sum(u_ps0[1:i_rc0].^2) - dr_solve * sum(u_2s_ae[1:i_rc0].^2)))

    # Step 4: TM pseudize l=1 (2p state)
    println("\n--- TM pseudization (l=1, 2p) ---")
    if !isnothing(u_2p_ae)
        u_ps1, tm_c1, i_rc1 = tm_pseudize(u_2p_ae, ε_2p, rgrid, r_c, 1)
        @printf("  u_ps1 nodes: %d, norm error: %.2e\n",
                count_nodes(u_ps1),
                abs(dr_solve * sum(u_ps1[1:i_rc1].^2) - dr_solve * sum(u_2p_ae[1:i_rc1].^2)))
    else
        error("Need bound l=1 state for KB construction")
    end

    # Step 5: Schrödinger-invert to get semilocal potentials
    println("\n--- Schrödinger inversion ---")
    V_sl0 = invert_potential(u_ps0, ε_2s, rgrid; l=0)
    V_sl1 = invert_potential(u_ps1, ε_2p, rgrid; l=1)

    # Fix V_sl outside r_c to match V_bare (they should already match)
    for i in (i_rc0+1):length(V_sl0)
        V_sl0[i] = V_bare[i]
    end
    for i in (i_rc0+1):length(V_sl1)
        V_sl1[i] = V_bare[i]
    end

    @printf("  V_sl^0(0) = %.4f Ha, V_sl^0(rc) = %.4f Ha\n", V_sl0[3], V_sl0[i_rc0])
    @printf("  V_sl^1(0) = %.4f Ha, V_sl^1(rc) = %.4f Ha\n", V_sl1[3], V_sl1[i_rc0])

    # Step 6: KB construction
    # V_local_ps = V_sl^1 (smooth l=1 semilocal potential)
    # δV^0(r) = V_sl^0(r) - V_local_ps(r)
    # Projector: p_0(r) = δV^0(r) × u_ps^0(r)
    # h_0 = 1 / ⟨u_ps^0|δV^0|u_ps^0⟩
    println("\n--- Kleinman-Bylander construction ---")

    δV0 = V_sl0 .- V_sl1

    # KB projector: p(r) = ΔV(r) × R_ps(r) = ΔV(r) × u_ps(r)/r
    # This is the 3D radial projector (DFTK convention, same as HGH)
    R_ps0 = u_ps0 ./ rgrid  # R(r) = u(r)/r
    proj0 = δV0 .* R_ps0

    # KB denominator: E_0 = ∫ R_ps × δV × R_ps × r² dr = ∫ u² × δV dr
    E_KB = dr_solve * sum(u_ps0 .* δV0 .* u_ps0)
    h_00 = 1.0 / E_KB
    @printf("  δV^0(0) = %.4f Ha, δV^0(rc) = %.6f Ha\n", δV0[3], δV0[i_rc0])
    @printf("  max|δV^0| = %.4f Ha\n", maximum(abs.(δV0)))
    @printf("  E_KB = %.6f Ha, h_00 = %.4f Ha\n", E_KB, h_00)
    @printf("  (GTH reference: h^{l=0} = 1.859 Ha)\n")

    # Step 7: Tabulate on DFTK-compatible log grid
    println("\n--- Building DFTK PSP struct ---")

    xmin = -7.0; dx_log = 0.00625; zmesh = NL_Z
    rgrid_dftk = T[exp(xmin + dx_log * i) / zmesh for i in 0:(N_dftk-1)]

    # Interpolators from the solve grid
    V_sl1_interp = linear_interpolation((rgrid,), V_sl1; extrapolation_bc=Line())
    proj0_interp = linear_interpolation((rgrid,), proj0; extrapolation_bc=Line())

    # V_local on DFTK grid
    vloc = T[r <= rgrid[end] ? V_sl1_interp(r) : -T(EFT_Z_VAL)/r for r in rgrid_dftk]
    vloc_dftk_interp = linear_interpolation((rgrid_dftk,), vloc; extrapolation_bc=Line())
    ircut = findlast(<=(rcut), rgrid_dftk)
    isnothing(ircut) && (ircut = N_dftk)

    # Projector on DFTK grid
    proj_dftk = T[r <= rgrid[end] ? proj0_interp(r) : zero(T) for r in rgrid_dftk]
    proj_dftk_interp = linear_interpolation((rgrid_dftk,), proj_dftk; extrapolation_bc=Line())

    id = @sprintf("EFT-Li-NL-rc%.1f", r_c)
    desc = @sprintf("Li EFT nonlocal NC PSP (KB, r_c=%.2f)", r_c)

    h_mat = [T[h_00;;]]  # lmax=0, one projector: h[1] is 1×1 matrix

    psp = PspEFTNonlocal{T, typeof(vloc_dftk_interp)}(
        EFT_Z_VAL, 0, h_mat,
        rgrid_dftk, vloc, vloc_dftk_interp, ircut, T(rcut),
        rgrid_dftk, proj_dftk, proj_dftk_interp,
        T(r_c), T(ε_1s), T(ε_2s), T(isnan(ε_2p) ? 0.0 : ε_2p),
        id, desc
    )

    @printf("  Zion=%d, lmax=%d, h^{l=0}=%.4f Ha\n", psp.Zion, psp.lmax, h_00)
    println("\n=== Nonlocal EFT PSP construction complete ===")
    return psp
end

# ══════════════════════════════════════════════════════════════════════════════
# NormConservingPsp interface
# ══════════════════════════════════════════════════════════════════════════════

DFTK.charge_ionic(psp::PspEFTNonlocal) = psp.Zion
DFTK.has_valence_density(psp::PspEFTNonlocal) = false
DFTK.has_core_density(psp::PspEFTNonlocal) = false

function DFTK.eval_psp_local_real(psp::PspEFTNonlocal{T}, r::T) where T
    r <= zero(T) && return psp.vloc_interp(psp.rgrid[1])
    r > psp.rcut && return -T(EFT_Z_VAL) / r
    psp.vloc_interp(r)
end

function DFTK.eval_psp_local_fourier(psp::PspEFTNonlocal{T}, p::T) where T
    p == zero(T) && return zero(T)
    Zion = T(psp.Zion)
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc = @view psp.vloc[1:psp.ircut]
    quadrature = DFTK.default_psp_quadrature(rgrid)
    DFTK._eval_psp_local_fourier(quadrature, rgrid, vloc, Zion, p)
end

function DFTK.eval_psp_local_fourier(psp::PspEFTNonlocal{T},
                                      ps::AbstractVector{T}) where {T<:AbstractFloat}
    Zion = T(psp.Zion)
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc = @view psp.vloc[1:psp.ircut]
    quadrature = DFTK.default_psp_quadrature(rgrid)
    map(p -> DFTK._eval_psp_local_fourier(quadrature, rgrid, vloc, Zion, p), ps)
end

function DFTK.eval_psp_energy_correction(::Type{T}, psp::PspEFTNonlocal) where T
    rgrid = psp.rgrid[1:psp.ircut]
    vloc = psp.vloc[1:psp.ircut]
    Zion = T(psp.Zion)
    quadrature = DFTK.default_psp_quadrature(rgrid)
    I = quadrature(rgrid) do i, r
        r^2 * (vloc[i] + Zion / r)
    end
    4T(π) * I
end

# Projector: l=0 only, one projector (i=1)
# Use ::Real to avoid ambiguity with DFTK's fallback method
function DFTK.eval_psp_projector_real(psp::PspEFTNonlocal, i, l, r::Real)
    T = typeof(float(r))
    l != 0 && return zero(T)
    i != 1 && return zero(T)
    r <= zero(T) && return zero(T)
    r > psp.rcut && return zero(T)
    T(psp.proj_l0_interp(Float64(r)))
end

function DFTK.eval_psp_projector_fourier(psp::PspEFTNonlocal, i, l, p::Real)
    T = typeof(float(p))
    l != 0 && return zero(T)
    i != 1 && return zero(T)
    p_f = Float64(p)
    rgrid = psp.proj_rgrid
    proj = psp.proj_l0
    ircut = psp.ircut
    if p_f < 1e-10
        # q→0 limit: p̃(0) = 4π ∫ r² p(r) dr
        quadrature = DFTK.default_psp_quadrature(rgrid[1:ircut])
        return T(4π * quadrature(rgrid[1:ircut]) do i_r, r
            r^2 * proj[i_r]
        end)
    end
    quadrature = DFTK.default_psp_quadrature(rgrid[1:ircut])
    T(4π * quadrature(rgrid[1:ircut]) do i_r, r
        r^2 * proj[i_r] * DFTK.sphericalbesselj_fast(0, p_f * r)
    end)
end
