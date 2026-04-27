#!/usr/bin/env bash
# generate_figures.sh — Reproduce all figures and compile the PRL paper.
#
# Usage: bash generate_figures.sh [--data-only | --plots-only | --tex-only | --dry-run]
#
# Steps:
#   1. Generate data (.jls files) — runs DFT, can take minutes
#   2. Generate figures (.pdf)    — fast, uses saved data
#   3. Compile LaTeX              — pdflatex
#
# --dry-run: Each Julia script prints what it would compute and save, then exits.
#            Copy and LaTeX steps print what they would do. No heavy computation.
#
# Tested with: Julia 1.11.5
# Requirements: julia (with packages resolved via Pkg.instantiate()), pdflatex, matplotlib (via PyPlot)
# Set JULIA_CMD to override the julia binary, e.g.: JULIA_CMD="julia +1.11.5" bash generate_figures.sh
#
# Figures produced (all in notes/fig/):
#   li_bands_3way.pdf         — Fig 1: Li 3-way band comparison
#   delta_K_all.pdf           — Fig 2: Δ(K) for all seven elements
#   nakmg_bands_expt.pdf      — Fig 3: Na/K/Mg bands vs ARPES (wide 3-panel)
#   pbe_bands_comparison.pdf  — Fig 4: LDA vs PBE band comparison (2×4 panel)
#   li_delta_K_validation.pdf — Appendix: Li Δ(K) validation

set -euo pipefail
cd "$(dirname "$0")"

JULIA="${JULIA_CMD:-julia} --project=."
DRY_RUN=""

# Warn if Julia version differs from tested version
JULIA_VER=$($JULIA -e 'print(VERSION)' 2>/dev/null || echo "unknown")
if [[ "$JULIA_VER" != "1.11.5" ]]; then
    echo "WARNING: Tested with Julia 1.11.5, but found $JULIA_VER"
    echo "         Set JULIA_CMD to override, e.g.: JULIA_CMD='julia +1.11.5' bash $0"
    echo ""
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    shift || true
    echo "=== DRY RUN: showing what each script would do ==="
    echo ""
fi

# Ensure output directory exists
mkdir -p notes/fig

step_data() {
    echo "=== Step 1: Generating data ==="

    echo "  [1/16] Li Δ(K) data (analytic, fast)..."
    $JULIA lithium/gen_delta_data.jl $DRY_RUN

    echo "  [2/16] Li bands + figure (DFT, may take a few minutes)..."
    $JULIA lithium/compare_qp_nonlocal.jl $DRY_RUN

    echo "  [3/16] Li Δ(K) validation plot (DFT)..."
    $JULIA lithium/validate_numerical_delta.jl $DRY_RUN

    echo "  [4/16] Na bands + Δ(K) data (DFT)..."
    $JULIA sodium/gen_band_data.jl $DRY_RUN

    echo "  [5/16] K bands + Δ(K) data (DFT)..."
    $JULIA potassium/freq_correction.jl $DRY_RUN

    echo "  [6/16] Mg bands + Δ(K) data (DFT)..."
    $JULIA magnesium/freq_correction.jl $DRY_RUN

    echo "  [7/16] Ca bands + Δ(K) data (DFT)..."
    $JULIA calcium/freq_correction.jl $DRY_RUN

    echo "  [8/16] Al bands + Δ(K) data (DFT)..."
    $JULIA aluminum/freq_correction.jl $DRY_RUN

    echo "  [9/16] Si bands + Δ(K) data (DFT)..."
    $JULIA silicon/freq_correction.jl $DRY_RUN

    echo "  [10/16] Li PBE comparison (DFT)..."
    $JULIA lithium/freq_correction_pbe.jl $DRY_RUN

    echo "  [11/16] Na PBE comparison (DFT)..."
    $JULIA sodium/freq_correction_pbe.jl $DRY_RUN

    echo "  [12/16] K PBE comparison (DFT)..."
    $JULIA potassium/freq_correction_pbe.jl $DRY_RUN

    echo "  [13/16] Ca PBE comparison (DFT)..."
    $JULIA calcium/freq_correction_pbe.jl $DRY_RUN

    echo "  [14/16] Mg PBE comparison (DFT)..."
    $JULIA magnesium/freq_correction_pbe.jl $DRY_RUN

    echo "  [15/16] Al PBE comparison (DFT)..."
    $JULIA aluminum/freq_correction_pbe.jl $DRY_RUN

    echo "  [16/16] Si PBE comparison (DFT)..."
    $JULIA silicon/freq_correction_pbe.jl $DRY_RUN

    echo "  Data generation complete."
}

step_plots() {
    echo "=== Step 2: Generating plots ==="

    echo "  [1/4] Li Fig 1 (GTH/EFT/PBE × KS/QP)..."
    $JULIA notes/plot_li_fig1.jl $DRY_RUN

    echo "  [2/4] Na+K full BZ bands..."
    $JULIA notes/plot_nak_combined.jl $DRY_RUN

    echo "  [3/4] ARPES comparison + Δ(K) (all 7 elements)..."
    $JULIA notes/plot_expt_comparison.jl $DRY_RUN

    echo "  [4/4] PBE band comparison (6-panel)..."
    $JULIA notes/plot_pbe_bands.jl $DRY_RUN

    echo "  Plot generation complete."
}

step_tex() {
    echo "=== Step 3: Compiling LaTeX ==="
    if [[ -n "$DRY_RUN" ]]; then
        echo "    [dry-run] would run: pdflatex notes/prl.tex (×2)"
        echo "    [dry-run] output: notes/prl.pdf"
    else
        cd notes
        pdflatex -interaction=nonstopmode prl.tex > /dev/null
        pdflatex -interaction=nonstopmode prl.tex > /dev/null
        echo "  Output: notes/prl.pdf"
        cd ..
    fi
}

case "${1:-all}" in
    --data-only)  step_data ;;
    --plots-only) step_plots ;;
    --tex-only)   step_tex ;;
    all)
        step_data
        step_plots
        step_tex
        echo "=== All done ==="
        ;;
    *)
        echo "Usage: $0 [--dry-run] [--data-only | --plots-only | --tex-only]"
        exit 1
        ;;
esac
