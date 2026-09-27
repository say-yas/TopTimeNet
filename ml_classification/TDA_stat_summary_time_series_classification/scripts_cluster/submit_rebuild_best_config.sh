#!/bin/bash
# submit_rebuild_best_config.sh
# Run directly on the login node, no SLURM needed.
#
# For every seg*/ subdirectory under BASE_SEARCH_DIR, calls
# rebuild_best_config.py, which writes two files into that directory:
#
#     best_config.json        highest-scoring trial
#     best_config_small.json  Pareto-optimal trial (score up / n_params down)
#
# Set FIND_SMALL="false" to skip best_config_small.json.
# Set MODE="single" to process only SEARCH_DIR instead of all seg*/ dirs.

BASE_SEARCH_DIR="/home/sharareh.sayyad/Computtional_topology/Computational_topology/ml_classification/output_tda_stat_summary/hparam_search_results"

MODE="all"                   # "single" or "all"
SEARCH_DIR="${BASE_SEARCH_DIR}/seg500"   # used only when MODE=single

ALPHA="0.6"
METRIC="f1_test"             # set to "" to use alpha*f1 + (1-alpha)*gmean
NO_INJECT_DEFAULTS="false"

FIND_SMALL="true"            # "true" to also write best_config_small.json
TOL="0.01"                   # Pareto score tolerance (e.g. 0.01 = accept 1% drop)

CONDA_ENV="TDA"
SCRIPT="/home/sharareh.sayyad/Computtional_topology/Computational_topology/ml_classification/TDA_stat_summary_time_series_classification/config_related/rebuild_best_config.py"
TRAIN_SCRIPT="main_train_time_series_tda_stat_summary.py"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"


# rebuild_one <seg_dir>
# Calls rebuild_best_config.py for one seg directory. Outputs land
# directly inside <seg_dir>:
#     <seg_dir>/best_config.json
#     <seg_dir>/best_config_small.json  (if FIND_SMALL=true)
rebuild_one() {
    local seg_dir="$1"

    echo ""
    echo "------------------------------------------------------------"
    echo "  Processing: ${seg_dir}"
    echo "------------------------------------------------------------"

    local args=(
        --search-dir "${seg_dir}"
        --alpha      "${ALPHA}"
    )
    [ -n "${METRIC}" ]                   && args+=( --metric "${METRIC}" )
    [ "${NO_INJECT_DEFAULTS}" = "true" ] && args+=( --no-inject-defaults )
    [ "${FIND_SMALL}"         = "true" ] && args+=( --small --tol "${TOL}" )

    python "${SCRIPT}" "${args[@]}"
}


# MODE=single
if [ "${MODE}" = "single" ]; then

    rebuild_one "${SEARCH_DIR}"

    echo ""
    echo "=============================================================="
    echo "  Output files:"
    for f in "${SEARCH_DIR}/best_config.json" \
              "${SEARCH_DIR}/best_config_small.json"; do
        [ -f "${f}" ] && echo "    ${f}"
    done
    echo "=============================================================="


# MODE=all: loop over every seg*/ subdirectory
elif [ "${MODE}" = "all" ]; then

    mapfile -t SEG_DIRS < <(find "${BASE_SEARCH_DIR}" -maxdepth 1 -type d -name 'seg*' | sort)

    if [ ${#SEG_DIRS[@]} -eq 0 ]; then
        echo "ERROR: no seg*/ subdirectories found under ${BASE_SEARCH_DIR}"
        exit 1
    fi

    echo "Found ${#SEG_DIRS[@]} seg*/ directories:"
    for d in "${SEG_DIRS[@]}"; do echo "  ${d}"; done

    for seg_dir in "${SEG_DIRS[@]}"; do
        rebuild_one "${seg_dir}"
    done

    # final summary
    echo ""
    echo "=============================================================="
    echo "  Results per segmentation duration:"
    echo ""
    printf "    %-10s  %-14s  %-20s\n" "seg_dur" "best_config" "best_config_small"
    printf "    %-10s  %-14s  %-20s\n" "-------" "-----------" "-----------------"

    for seg_dir in "${SEG_DIRS[@]}"; do
        seg_name=$(basename "${seg_dir}")
        seg_num="${seg_name#seg}"

        best_exists="no"
        small_exists="no"
        [ -s "${seg_dir}/best_config.json"       ] && best_exists="yes"
        [ -s "${seg_dir}/best_config_small.json"  ] && small_exists="yes"

        printf "    %-10s  %-14s  %-20s\n" "${seg_num}" "${best_exists}" "${small_exists}"
    done

    echo ""
    echo "  Files are inside each seg*/ directory."
    echo ""
    echo "  To train with best config:"
    for seg_dir in "${SEG_DIRS[@]}"; do
        f="${seg_dir}/best_config.json"
        [ -s "${f}" ] && echo "    python ${TRAIN_SCRIPT} --config ${f}"
    done

    if [ "${FIND_SMALL}" = "true" ]; then
        echo ""
        echo "  To train with small config:"
        for seg_dir in "${SEG_DIRS[@]}"; do
            f="${seg_dir}/best_config_small.json"
            [ -s "${f}" ] && echo "    python ${TRAIN_SCRIPT} --config ${f}"
        done
    fi
    echo "=============================================================="

else
    echo "ERROR: unknown MODE='${MODE}'. Set MODE=single or MODE=all."
    exit 1
fi
