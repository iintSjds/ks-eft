"""
plot_expt_comparison.jl
Compare computed KS and QP bands with ARPES data.
Overlays both LDA and PBE results to demonstrate robustness.

Outputs:
  fig/nakmgal_bands_expt.pdf — wide 4-panel: Na, K, Mg, Al vs ARPES (for PRL main text)
  fig/delta_K_all.pdf      — f_c(K)/ΔE_c for dominant channel, all seven elements

Requires:
  sodium/{na_ks_data,na_pbe_ks_data,na_qp_data,na_pbe_qp_data}.jld2
  potassium/{k_ks_data,k_pbe_ks_data,k_qp_data,k_pbe_qp_data}.jld2
  magnesium/{mg_ks_data,mg_pbe_ks_data,mg_qp_data,mg_pbe_qp_data}.jld2
  lithium/li_qp_data.jld2
  calcium/ca_qp_data.jld2
  aluminum/al_qp_data.jld2
  silicon/si_qp_data.jld2
  notes/expts/naband1.csv, naband2.csv, kband.csv, mgband.csv

Usage: julia --project=. notes/plot_expt_comparison.jl
"""

if "--dry-run" in ARGS
    println("[dry-run] notes/plot_expt_comparison.jl")
    println("  Plots: 3-panel Na/K/Mg vs ARPES, f_c(K)/ΔE_c for all 7 elements")
    println("  Reads: {na,k,mg}_ks_data.jld2, {na,k,mg}_pbe_ks_data.jld2")
    println("         {na,k,mg}_qp_data.jld2, {na,k,mg}_pbe_qp_data.jld2")
    println("         {li,ca,al,si}_qp_data.jld2")
    println("         notes/expts/{naband1,naband2,kband,mgband}.csv")
    println("  Saves: notes/fig/nakmg_bands_expt.pdf, notes/fig/delta_K_all.pdf")
    exit(0)
end

using JLD2
using PyPlot

Ha_to_eV = 27.211386245988
figdir = joinpath(@__DIR__, "fig")
mkpath(figdir)
exptdir = joinpath(@__DIR__, "expts")
basedir = joinpath(@__DIR__, "..")

# ── Load computed band data (LDA KS + QP) ───────────────────────────────────

na_ks  = load(joinpath(basedir, "sodium",    "na_ks_data.jld2"))
na_qp  = load(joinpath(basedir, "sodium",    "na_qp_data.jld2"))
k_ks   = load(joinpath(basedir, "potassium", "k_ks_data.jld2"))
k_qp   = load(joinpath(basedir, "potassium", "k_qp_data.jld2"))
mg_ks  = load(joinpath(basedir, "magnesium", "mg_ks_data.jld2"))
mg_qp  = load(joinpath(basedir, "magnesium", "mg_qp_data.jld2"))

# ── Load PBE KS + QP data ───────────────────────────────────────────────────

na_pbe_ks = load(joinpath(basedir, "sodium",    "na_pbe_ks_data.jld2"))
na_pbe_qp = load(joinpath(basedir, "sodium",    "na_pbe_qp_data.jld2"))
k_pbe_ks  = load(joinpath(basedir, "potassium", "k_pbe_ks_data.jld2"))
k_pbe_qp  = load(joinpath(basedir, "potassium", "k_pbe_qp_data.jld2"))
mg_pbe_ks = load(joinpath(basedir, "magnesium", "mg_pbe_ks_data.jld2"))
mg_pbe_qp = load(joinpath(basedir, "magnesium", "mg_pbe_qp_data.jld2"))

# ── Load experimental ARPES data ─────────────────────────────────────────────

function load_csv(path)
    lines = readlines(path)
    x = Float64[]; y = Float64[]
    for line in lines[2:end]  # skip header
        parts = split(line, ',')
        push!(x, parse(Float64, parts[1]))
        push!(y, parse(Float64, parts[2]))
    end
    return x, y
end

na_exp1_x, na_exp1_y = load_csv(joinpath(exptdir, "naband1.csv"))
na_exp2_x, na_exp2_y = load_csv(joinpath(exptdir, "naband2.csv"))
k_exp_x_raw, k_exp_y = load_csv(joinpath(exptdir, "kband.csv"))
mg_exp_x, mg_exp_y = load_csv(joinpath(exptdir, "mgband.csv"))
al_exp_x, al_exp_y = load_csv(joinpath(exptdir, "alband.csv"))

# ── Load Al KS + QP data ────────────────────────────────────────────────────
al_ks  = load(joinpath(basedir, "aluminum", "al_ks_data.jld2"))
al_qp_full  = load(joinpath(basedir, "aluminum", "al_qp_data.jld2"))
al_pbe_ks = load(joinpath(basedir, "aluminum", "al_pbe_ks_data.jld2"))
al_pbe_qp = load(joinpath(basedir, "aluminum", "al_pbe_qp_data.jld2"))

# ── Helper: build a band-data dict from KS + QP JLD2 files ──────────────────

function make_band_data(ks, qp_eqp_arr)
    Dict(
        "eigenvalues" => ks["eigenvalues_array"],
        "eqp"         => qp_eqp_arr,
        "εF"          => ks["εF"],
        "tick_distances" => ks["tick_distances"],
        "kdistances"  => ks["kdistances"],
        "n_bands"     => ks["n_bands"],
    )
end

na_data  = make_band_data(na_ks, na_qp["eqp_arr"])
k_data   = make_band_data(k_ks,  k_qp["eqp_arr"])
mg_data  = make_band_data(mg_ks, mg_qp["eqp_arr"])

# ── Extract N→Γ segment, normalized to [0,1] ────────────────────────────────
# BCC path: Γ→H→N→Γ→P→H|P→N  ⟹  N→Γ = ticks[3]→ticks[4]

function extract_NGN_norm(data)
    td = data["tick_distances"]
    kd = data["kdistances"]
    εF = data["εF"]

    d_N = td[3]
    d_Γ = td[4]

    idx = findall(x -> d_N - 1e-8 <= x <= d_Γ + 1e-8, kd)

    # Normalize: 0 at N, 1 at Γ
    x_norm = [(kd[i] - d_N) / (d_Γ - d_N) for i in idx]

    n_bands = data["n_bands"]
    ks_bands = [(data["eigenvalues"][idx, n, 1] .- εF) .* Ha_to_eV for n in 1:n_bands]
    qp_bands = [(data["eqp"][idx, n, 1] .- εF) .* Ha_to_eV for n in 1:n_bands]

    # Return also the physical N→Γ distance in Å⁻¹ for normalizing expt data
    bohr_to_ang_inv = 1 / 0.529177
    NG_ang = (d_Γ - d_N) * bohr_to_ang_inv

    return x_norm, ks_bands, qp_bands, NG_ang
end

na_xn, na_ks_bands, na_qp_bands, na_NG = extract_NGN_norm(na_data)
k_xn, k_ks_bands, k_qp_bands, k_NG = extract_NGN_norm(k_data)

# ── Extract PBE conduction band for Na and K ─────────────────────────────────
# Na PBE QP uses keys: eqp_arr_pbe, εF_pbe, conduction_band
# K  PBE QP uses keys: pbe_eqp_arr, pbe_conduction_band

function extract_NGN_norm_pbe_cond(pbe_ks, pbe_qp, eqp_key, cond_key)
    td = pbe_ks["tick_distances"]
    kd = pbe_ks["kdistances"]
    εF = pbe_ks["εF"]
    eqp_arr = pbe_qp[eqp_key]
    cond = haskey(pbe_qp, cond_key) ? pbe_qp[cond_key] : pbe_ks["conduction_band"]

    d_N = td[3]
    d_Γ = td[4]
    idx = findall(x -> d_N - 1e-8 <= x <= d_Γ + 1e-8, kd)
    x_norm = [(kd[i] - d_N) / (d_Γ - d_N) for i in idx]

    ks_band = (pbe_ks["eigenvalues_array"][idx, cond, 1] .- εF) .* Ha_to_eV
    qp_band = (eqp_arr[idx, cond, 1] .- εF) .* Ha_to_eV
    return x_norm, ks_band, qp_band
end

na_pbe_xn, na_pbe_ks_band, na_pbe_qp_band = extract_NGN_norm_pbe_cond(
    na_pbe_ks, na_pbe_qp, "eqp_arr_pbe", "conduction_band")
k_pbe_xn, k_pbe_ks_band, k_pbe_qp_band = extract_NGN_norm_pbe_cond(
    k_pbe_ks, k_pbe_qp, "pbe_eqp_arr", "pbe_conduction_band")

# ── Extract Γ→Z(=A) segment for Mg, normalized to [0,1] ─────────────────────

function extract_GAG_norm(data)
    td = data["tick_distances"]
    kd = data["kdistances"]
    εF = data["εF"]

    d_Γ = td[4]
    d_Z = td[5]

    idx = findall(x -> d_Γ - 1e-8 <= x <= d_Z + 1e-8, kd)

    # Normalize: 0 at Γ, 1 at Z(=A)
    x_norm = [(kd[i] - d_Γ) / (d_Z - d_Γ) for i in idx]

    n_bands = data["n_bands"]
    ks_bands = [(data["eigenvalues"][idx, n, 1] .- εF) .* Ha_to_eV for n in 1:n_bands]
    qp_bands = [(data["eqp"][idx, n, 1] .- εF) .* Ha_to_eV for n in 1:n_bands]

    return x_norm, ks_bands, qp_bands
end

mg_xn, mg_ks_bands, mg_qp_bands = extract_GAG_norm(mg_data)

# Extract PBE bands for Mg (same Zion, all bands)
function extract_GAG_norm_pbe(pbe_ks, pbe_qp, eqp_key)
    td = pbe_ks["tick_distances"]
    kd = pbe_ks["kdistances"]
    εF = pbe_ks["εF"]
    eqp_arr = pbe_qp[eqp_key]

    d_Γ = td[4]
    d_Z = td[5]
    idx = findall(x -> d_Γ - 1e-8 <= x <= d_Z + 1e-8, kd)
    x_norm = [(kd[i] - d_Γ) / (d_Z - d_Γ) for i in idx]

    n_bands = pbe_ks["n_bands"]
    ks_bands = [(pbe_ks["eigenvalues_array"][idx, n, 1] .- εF) .* Ha_to_eV for n in 1:n_bands]
    qp_bands = [(eqp_arr[idx, n, 1] .- εF) .* Ha_to_eV for n in 1:n_bands]
    return x_norm, ks_bands, qp_bands
end

mg_pbe_xn, mg_pbe_ks_bands, mg_pbe_qp_bands = extract_GAG_norm_pbe(
    mg_pbe_ks, mg_pbe_qp, "pbe_eqp_arr")

# ── Extract Γ→X segment for Al (FCC: tick 1→2), normalized to [0,1] ─────────
function extract_GX_norm(ks, eqp_arr)
    td = ks["tick_distances"]
    kd = ks["kdistances"]
    εF = ks["εF"]
    d_Γ = td[1]
    d_X = td[2]
    idx = findall(x -> d_Γ - 1e-8 <= x <= d_X + 1e-8, kd)
    x_norm = [(kd[i] - d_Γ) / (d_X - d_Γ) for i in idx]
    n_bands = ks["n_bands"]
    ks_bands = [(ks["eigenvalues_array"][idx, n, 1] .- εF) .* Ha_to_eV for n in 1:n_bands]
    qp_bands = [(eqp_arr[idx, n, 1] .- εF) .* Ha_to_eV for n in 1:n_bands]
    return x_norm, ks_bands, qp_bands
end

al_xn, al_ks_bands, al_qp_bands = extract_GX_norm(al_ks, al_qp_full["eqp_arr"])
al_pbe_xn, al_pbe_ks_bands, al_pbe_qp_bands = extract_GX_norm(
    al_pbe_ks, al_pbe_qp["pbe_eqp_arr"])

# Normalize Na experimental data (originally in Å⁻¹)
# x>0 is toward N, x<0 is toward -N; Γ is at 0
# Normalize: divide by N→Γ distance
na_exp1_xn = na_exp1_x ./ na_NG
na_exp2_xn = na_exp2_x ./ na_NG

# K experimental data is already normalized ratio (N=-1, Γ=0, N=+1)
k_exp_xn = k_exp_x_raw

# ══════════════════════════════════════════════════════════════════════════════
# Combined 4-panel wide figure: Na | K | Mg | Al
# ══════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2, ax3, ax4) = subplots(1, 4; figsize=(10.5, 3.2), sharey=false)

# ── Na panel ─────────────────────────────────────────────────────────────────
for n in 1:1  # lowest band only
    # x_norm goes 0 (N) → 1 (Γ); map to: left N at -1, Γ at 0, right N at +1
    x_left = na_xn .- 1.0        # N(-1) → Γ(0)
    x_right = 1.0 .- na_xn       # Γ(0) → N(+1)
    ax1.plot(x_left, na_ks_bands[n]; color="blue", lw=1.5, label="LDA KS")
    ax1.plot(x_left, na_qp_bands[n]; color="red", lw=1.5, ls="--", label="LDA QP")
    ax1.plot(x_right, na_ks_bands[n]; color="blue", lw=1.5)
    ax1.plot(x_right, na_qp_bands[n]; color="red", lw=1.5, ls="--")
end
# PBE conduction band overlay
begin
    x_left = na_pbe_xn .- 1.0
    x_right = 1.0 .- na_pbe_xn
    ax1.plot(x_left, na_pbe_ks_band; color="blue", lw=1.5, ls=":", alpha=0.6, label="PBE KS")
    ax1.plot(x_left, na_pbe_qp_band; color="red", lw=1.5, ls=":", alpha=0.6, label="PBE QP")
    ax1.plot(x_right, na_pbe_ks_band; color="blue", lw=1.5, ls=":", alpha=0.6)
    ax1.plot(x_right, na_pbe_qp_band; color="red", lw=1.5, ls=":", alpha=0.6)
end
ax1.scatter(na_exp1_xn, na_exp1_y; s=12, color="black", alpha=0.7, zorder=5,
            marker="o", label="Expt. 1")
ax1.scatter(na_exp2_xn, na_exp2_y; s=12, color="green", alpha=0.7, zorder=5,
            marker="s", label="Expt. 2")
ax1.axhline(0.0; color="gray", ls=":", lw=0.8)
ax1.set_ylim(-4.0, 0.0)
ax1.set_xlim(-1.0, 1.0)
ax1.set_xticks([-1, 0, 1])
ax1.set_xticklabels(["N", "Γ", "N"])
ax1.set_ylabel("Energy (eV)")
ax1.set_title("Na"; loc="left", fontsize=10)
ax1.legend(loc="lower right", framealpha=0.9, fontsize=6, ncol=2)

# ── K panel ──────────────────────────────────────────────────────────────────
for n in 1:1
    local x_left = k_xn .- 1.0
    local x_right = 1.0 .- k_xn
    ax2.plot(x_left, k_ks_bands[n]; color="blue", lw=1.5, label="LDA KS")
    ax2.plot(x_left, k_qp_bands[n]; color="red", lw=1.5, ls="--", label="LDA QP")
    ax2.plot(x_right, k_ks_bands[n]; color="blue", lw=1.5)
    ax2.plot(x_right, k_qp_bands[n]; color="red", lw=1.5, ls="--")
end
# PBE conduction band overlay
begin
    local x_left = k_pbe_xn .- 1.0
    local x_right = 1.0 .- k_pbe_xn
    ax2.plot(x_left, k_pbe_ks_band; color="blue", lw=1.5, ls=":", alpha=0.6, label="PBE KS")
    ax2.plot(x_left, k_pbe_qp_band; color="red", lw=1.5, ls=":", alpha=0.6, label="PBE QP")
    ax2.plot(x_right, k_pbe_ks_band; color="blue", lw=1.5, ls=":", alpha=0.6)
    ax2.plot(x_right, k_pbe_qp_band; color="red", lw=1.5, ls=":", alpha=0.6)
end
ax2.scatter(k_exp_xn, k_exp_y; s=12, color="black", alpha=0.7, zorder=5,
            marker="o", label="Expt.")
ax2.axhline(0.0; color="gray", ls=":", lw=0.8)
ax2.set_ylim(-2.8, 0.0)
ax2.set_xlim(-1.0, 1.0)
ax2.set_xticks([-1, 0, 1])
ax2.set_xticklabels(["N", "Γ", "N"])
ax2.set_title("K"; loc="left", fontsize=10)
ax2.legend(loc="lower right", framealpha=0.9, fontsize=6, ncol=2)

# ── Mg panel ─────────────────────────────────────────────────────────────────
# Plot all 3 occupied bands; let ylim clip above εF (no NaN masking)
n_plot_mg = 3
for n in 1:n_plot_mg
    lb_ks = n == 1 ? "LDA KS" : ""
    lb_qp = n == 1 ? "LDA QP" : ""
    # Left half: Γ(-1) → A(0)
    ax3.plot(mg_xn .- 1.0, mg_ks_bands[n]; color="blue", lw=1.5, label=lb_ks)
    ax3.plot(mg_xn .- 1.0, mg_qp_bands[n]; color="red", lw=1.5, ls="--", label=lb_qp)
    # Right half: A(0) → Γ(+1)
    ax3.plot(1.0 .- mg_xn, mg_ks_bands[n]; color="blue", lw=1.5)
    ax3.plot(1.0 .- mg_xn, mg_qp_bands[n]; color="red", lw=1.5, ls="--")
end
# PBE overlay (same Zion, all bands)
for n in 1:n_plot_mg
    lb_ks = n == 1 ? "PBE KS" : ""
    lb_qp = n == 1 ? "PBE QP" : ""
    ax3.plot(mg_pbe_xn .- 1.0, mg_pbe_ks_bands[n]; color="blue", lw=1.5, ls=":", alpha=0.6, label=lb_ks)
    ax3.plot(mg_pbe_xn .- 1.0, mg_pbe_qp_bands[n]; color="red", lw=1.5, ls=":", alpha=0.6, label=lb_qp)
    ax3.plot(1.0 .- mg_pbe_xn, mg_pbe_ks_bands[n]; color="blue", lw=1.5, ls=":", alpha=0.6)
    ax3.plot(1.0 .- mg_pbe_xn, mg_pbe_qp_bands[n]; color="red", lw=1.5, ls=":", alpha=0.6)
end
ax3.scatter(mg_exp_x, mg_exp_y; s=12, color="black", alpha=0.7, zorder=5,
            marker="o", label="Expt.")
ax3.axhline(0.0; color="gray", ls=":", lw=0.8)
ax3.set_ylim(-7.5, 0.0)
ax3.set_xlim(-1.0, 1.0)
ax3.set_xticks([-1, 0, 1])
ax3.set_xticklabels(["Γ", "A", "Γ"])
ax3.set_title("Mg"; loc="left", fontsize=10)
ax3.legend(loc="lower right", framealpha=0.9, fontsize=6, ncol=2)

# ── Al panel (Γ→X) ──────────────────────────────────────────────────────────
n_plot_al = 3
for n in 1:n_plot_al
    lb_ks = n == 1 ? "LDA KS" : ""
    lb_qp = n == 1 ? "LDA QP" : ""
    ax4.plot(al_xn, al_ks_bands[n]; color="blue", lw=1.5, label=lb_ks)
    ax4.plot(al_xn, al_qp_bands[n]; color="red", lw=1.5, ls="--", label=lb_qp)
end
for n in 1:n_plot_al
    lb_ks = n == 1 ? "PBE KS" : ""
    lb_qp = n == 1 ? "PBE QP" : ""
    ax4.plot(al_pbe_xn, al_pbe_ks_bands[n]; color="blue", lw=1.5, ls=":", alpha=0.6, label=lb_ks)
    ax4.plot(al_pbe_xn, al_pbe_qp_bands[n]; color="red", lw=1.5, ls=":", alpha=0.6, label=lb_qp)
end
ax4.scatter(al_exp_x, al_exp_y; s=12, color="black", alpha=0.7, zorder=5,
            marker="o", label="Expt.")
ax4.axhline(0.0; color="gray", ls=":", lw=0.8)
ax4.set_ylim(-12.5, 0.5)
ax4.set_xlim(0.0, 1.0)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(["Γ", "X"])
ax4.set_title("Al"; loc="left", fontsize=10)
ax4.legend(loc="lower right", framealpha=0.9, fontsize=6, ncol=2)

tight_layout()
subplots_adjust(wspace=0.30)
savefig(joinpath(figdir, "nakmgal_bands_expt.pdf"))
println("Saved → fig/nakmgal_bands_expt.pdf")
close(fig)

# ── f_c(K)/ΔE_c comparison (all elements, dominant channel) ─────────────────

# Load QP data for all 7 elements (for f_c/ΔE_c plot)
li_qp = load(joinpath(basedir, "lithium",   "li_qp_data.jld2"))
na_qp_d = na_qp   # already loaded
k_qp_d  = k_qp    # already loaded
ca_qp = load(joinpath(basedir, "calcium",   "ca_qp_data.jld2"))
mg_qp_d = mg_qp   # already loaded
al_qp = load(joinpath(basedir, "aluminum",  "al_qp_data.jld2"))
si_qp = load(joinpath(basedir, "silicon",   "si_qp_data.jld2"))

# Extract dominant-channel f_c(K)/ΔE_c from each QP file
function get_fc_over_dE(qp)
    Kg = qp["K_grid"]
    channel_names = qp["channel_names"]
    channel_ΔE    = qp["channel_ΔE"]
    fc_dict       = qp["fc_per_channel"]
    # Dominant channel = smallest |ΔE|
    idx = argmin(abs.(channel_ΔE))
    name = channel_names[idx]
    fc = fc_dict[name]
    dE = abs(channel_ΔE[idx])
    return Kg, fc ./ dE
end

# Li now has channel_names/channel_ΔE/fc_per_channel (DFT-LDA, same as other elements)
fig, ax = subplots(figsize=(4.5, 3.0))
for (qp, color, ls, label) in [
    (k_qp_d,  "red",      "-",  "K"),
    (li_qp,   "green",    "-",  "Li"),
    (ca_qp,   "cyan",     "-",  "Ca"),
    (na_qp_d, "blue",     "--", "Na"),
    (mg_qp_d, "#8B4513",  "--", "Mg"),
    (al_qp,   "purple",   ":",  "Al"),
    (si_qp,   "orange",   ":",  "Si"),
]
    Kg, fc_dE = get_fc_over_dE(qp)
    idx = Kg .<= 10.0
    ax.plot(Kg[idx], fc_dE[idx]; color=color, lw=2, ls=ls, label=label)
end
ax.axhline(0.0; color="gray", ls=":", lw=0.5)
ax.set_xlim(0.0, 10.0)
ax.set_xlabel(raw"$K$ (Bohr$^{-1}$)")
ax.set_ylabel(raw"$f_c(K) / \Delta E_c$")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9, ncol=2)
tight_layout()
savefig(joinpath(figdir, "delta_K_all.pdf"))
println("Saved → fig/delta_K_all.pdf")
close(fig)

println("\n=== Done ===")
