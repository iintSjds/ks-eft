#!/usr/bin/env bash
# generate_figures.sh — Reproduce all data and figures from the paper.
#
# Usage:
#   bash generate_figures.sh              # run everything
#   bash generate_figures.sh data         # Stage 1+2 only (DFT + QP)
#   bash generate_figures.sh plot         # plotting only (needs data)
#   bash generate_figures.sh data plot    # same as no args
#   bash generate_figures.sh --dry-run    # show what would run
#   bash generate_figures.sh --dry-run plot
#
# Environment:
#   JULIA_CMD      override julia binary (default: julia)
#   JULIA_PROJECT  override --project path (default: .)

set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=""
STEPS=()

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        data)      STEPS+=(data) ;;
        plot)      STEPS+=(plot) ;;
        *)         echo "Usage: $0 [--dry-run] [data] [plot]"; exit 1 ;;
    esac
done

# Default: run all steps
[[ ${#STEPS[@]} -eq 0 ]] && STEPS=(data plot)

[[ -n "$DRY_RUN" ]] && echo "=== DRY RUN ===" && echo ""

JULIA="${JULIA_CMD:-julia} --project=${JULIA_PROJECT:-.}"
mkdir -p fig

step_data() {
    echo "=== Step 1: Generating data (KS-DFT + QP correction) ==="

    for elem in lithium sodium potassium calcium aluminum silicon magnesium; do
        echo "  [$elem] Stage 1: KS-DFT (LDA)..."
        $JULIA "$elem/run_ks.jl" $DRY_RUN

        if [[ "$elem" == "lithium" ]]; then
            echo "  [$elem] Stage 1: EFT nonlocal PSP..."
            $JULIA "$elem/run_ks_eft.jl" $DRY_RUN
            echo "  [$elem] Stage 1: PBE..."
            $JULIA "$elem/run_ks_pbe.jl" $DRY_RUN
            echo "  [$elem] Stage 2: QP correction (all 3 PSPs)..."
            $JULIA "$elem/freq_correction_new.jl" $DRY_RUN
        else
            echo "  [$elem] Stage 2: QP correction..."
            $JULIA "$elem/freq_correction.jl" $DRY_RUN
            if [[ -f "$elem/run_ks_pbe.jl" ]]; then
                echo "  [$elem] Stage 1: KS-DFT (PBE)..."
                $JULIA "$elem/run_ks_pbe.jl" $DRY_RUN
                echo "  [$elem] Stage 2: QP correction (PBE)..."
                $JULIA "$elem/freq_correction_pbe.jl" $DRY_RUN
            fi
        fi
    done

    echo "  [lih] Stage 1 + Stage 2..."
    $JULIA lih/run_ks.jl $DRY_RUN
    $JULIA lih/freq_correction.jl $DRY_RUN

    echo "  Data generation complete."
}

step_plot() {
    echo "=== Step 2: Generating plots ==="

    echo "  [1/2] Na+K+Mg+Al bands vs ARPES + Δ(K)..."
    $JULIA plotting/plot_expt_comparison.jl $DRY_RUN

    echo "  [2/2] PBE band comparison..."
    $JULIA plotting/plot_pbe_bands.jl $DRY_RUN

    echo "  Plot generation complete."
}

for step in "${STEPS[@]}"; do
    "step_$step"
done

echo "=== Done ==="
