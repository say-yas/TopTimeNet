#!/bin/bash
# run_rebuild_best_config.sh
#
# Run directly on the login node, no SLURM needed.
# Rebuilds best_config_<modeltype>.json (and best_config.json) from
# all_trials.csv (or trial_summary.csv files) for every modeltype whose
# search directory exists under PATH_OUTPUT.
#
# Usage:
#     bash run_rebuild_best_config.sh
#     bash run_rebuild_best_config.sh --modeltype trans1   # one type only
#     bash run_rebuild_best_config.sh --alpha 0.8
set -euo pipefail

# PARAMETERS: edit these
ALPHA="0.7"
CONDA_ENV="TDA"
ALL_MODELTYPES=("trans1" "cnn1")

# PATHS: edit these
BASE="/home/sharareh.sayyad/Computtional_topology/Computational_topology"
PATH_CODE="${BASE}/ml_classification/transformer_cnn_time_series"
PATH_OUTPUT="${BASE}/ml_classification/output_transformer_cnn"

# Shared root; modeltype subdirs live inside this, matching submit_hparam_search.sh
PATH_SEARCH_ROOT="${PATH_OUTPUT}/hparam_search_results"

SCRIPT="${PATH_CODE}/rebuild_best_config.py"

# Parse CLI flags
FORCE_MODELTYPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --modeltype) FORCE_MODELTYPE="$2"; shift 2 ;;
        --alpha)     ALPHA="$2";           shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -n "${FORCE_MODELTYPE}" ]] && ALL_MODELTYPES=("${FORCE_MODELTYPE}")

# Validate script exists before activating conda
if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: rebuild script not found: ${SCRIPT}"
    exit 1
fi

# Activate conda
module load anaconda3
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Rebuild per modeltype
SUCCEEDED=()
SKIPPED=()
FAILED=()

for MODELTYPE in "${ALL_MODELTYPES[@]}"; do

    SEARCH_DIR="${PATH_SEARCH_ROOT}/${MODELTYPE}"

    echo ""
    echo "=============================================================="
    echo "  Modeltype  : ${MODELTYPE}"
    echo "  Search dir : ${SEARCH_DIR}"
    echo "=============================================================="

    if [[ ! -d "${SEARCH_DIR}" ]]; then
        echo "  SKIP: directory does not exist."
        SKIPPED+=("${MODELTYPE}")
        continue
    fi

    # Check there is something to work with
    HAS_TRIALS_CSV=0
    HAS_SUMMARIES=0
    [[ -f "${SEARCH_DIR}/all_trials.csv" ]] && HAS_TRIALS_CSV=1
    [[ -n "$(find "${SEARCH_DIR}" -name "trial_summary.csv" 2>/dev/null | head -1)" ]] \
        && HAS_SUMMARIES=1

    if [[ "${HAS_TRIALS_CSV}" -eq 0 && "${HAS_SUMMARIES}" -eq 0 ]]; then
        echo "  SKIP: no all_trials.csv or trial_summary.csv files found."
        SKIPPED+=("${MODELTYPE}")
        continue
    fi

    # Run the rebuild script for this modeltype
    if python "${SCRIPT}" \
            --search-dir "${SEARCH_DIR}" \
            --alpha      "${ALPHA}" \
            --modeltypes "${MODELTYPE}"; then

        BEST="${SEARCH_DIR}/best_config_${MODELTYPE}.json"
        BEST_GENERIC="${SEARCH_DIR}/best_config.json"

        if [[ -s "${BEST}" ]]; then
            echo "  OK  ${BEST}"
            SUCCEEDED+=("${MODELTYPE}")
        elif [[ -s "${BEST_GENERIC}" ]]; then
            echo "  OK  ${BEST_GENERIC}  (generic fallback)"
            SUCCEEDED+=("${MODELTYPE}")
        else
            echo "  ERROR: script exited 0 but no best config was written."
            FAILED+=("${MODELTYPE}")
        fi
    else
        echo "  ERROR: rebuild script failed for modeltype '${MODELTYPE}'."
        FAILED+=("${MODELTYPE}")
    fi

done

# Summary
echo ""
echo "=============================================================="
echo "  SUMMARY"
echo "=============================================================="
[[ ${#SUCCEEDED[@]} -gt 0 ]] && echo "  Succeeded : ${SUCCEEDED[*]}"
[[ ${#SKIPPED[@]}   -gt 0 ]] && echo "  Skipped   : ${SKIPPED[*]}"
[[ ${#FAILED[@]}    -gt 0 ]] && echo "  Failed    : ${FAILED[*]}"
echo "=============================================================="

# Exit non-zero if anything failed
[[ ${#FAILED[@]} -gt 0 ]] && exit 1
exit 0
