#!/bin/bash
# submit_config_search.sh
# Generates search configs, runs the hyperparameter search trial array (or
# sequential job), and merges results, for each (modeltype, segmentation
# duration) combination.
set -euo pipefail

# PARAMETERS: edit these
N_TRIALS=250
STRATEGY="random"
OBJECTIVE_ALPHA=0.7
NUM_EPOCHS=500
PATIENCE=50
NUM_TRAINING=1
RANDOM_SEED=43
ARRAY=true
MAX_PARALLEL=10
BALANCE_STRATEGY="undersample"   # none | undersample | oversample | hybrid

# Resources per trial
BLAS_THREADS=4
CPUS_PER_TASK=$(( BLAS_THREADS + 2 ))
MEM_PER_TRIAL="30G"
GPU_TYPE="tesla"          # set "" for any GPU, or "a100", "v100", etc.
TIME_PER_TRIAL="2-08:00:00"
TIME_SEQ="7-00:00:00"

# PATHS: edit these
BASE="/home/sharareh.sayyad/Computtional_topology/Computational_topology"
PATH_CODE="${BASE}/ml_classification/transformer_cnn_time_series"
PATH_OUTPUT="${BASE}/ml_classification/output_transformer_cnn"
PATH_DATA="${BASE}/dataset/extended_teaspoon_dataset"
CONDA_ENV="TDA"

HPARAM_SCRIPT="${PATH_CODE}/config_related/search_best_config.py"
REBUILD_SCRIPT="${PATH_CODE}/config_related/rebuild_best_config.py"

# Single root for all search results
PATH_SEARCH_ROOT="${PATH_OUTPUT}/hparam_search_results"

# Model types and segmentation durations to search
ALL_MODELTYPES=("trans1" "cnn1")
ALL_SEG_DURATIONS=(1000)

# Parse CLI flags
DRY_RUN=0
FORCE_MODELTYPE=""
FORCE_SEG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --modeltype)   FORCE_MODELTYPE="$2"; shift 2 ;;
        --seg)         FORCE_SEG="$2";       shift 2 ;;
        --sequential)  ARRAY=false;           shift ;;
        --dry-run)     DRY_RUN=1;             shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -n "${FORCE_MODELTYPE}" ]] && ALL_MODELTYPES=("${FORCE_MODELTYPE}")
[[ -n "${FORCE_SEG}" ]]       && ALL_SEG_DURATIONS=("${FORCE_SEG}")

# Helpers

_sbatch() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY-RUN] sbatch $*" >&2
        echo "9999999"
    else
        sbatch --parsable "$@"
    fi
}

if ! command -v jq &>/dev/null; then
    echo "ERROR: 'jq' is required but not found."
    echo "  conda install -c conda-forge jq   or   module load jq"
    exit 1
fi

# Activate conda once (needed for --generate-configs step on the login node)
module load anaconda3
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ -n "${GPU_TYPE}" ]]; then
    GPU_GRES="gpu:${GPU_TYPE}:1"
else
    GPU_GRES="gpu:1"
fi

mkdir -p "${PATH_SEARCH_ROOT}"

# _write_search_config  $modeltype  $seg_duration  $out_path  $path_save
#
# segmentation_duration is fixed (not in search space).
# balance_strategy is fixed.
# optimizer and muon_lr are searchable (included in search_space).
_write_search_config() {
    local modeltype="$1"
    local seg_duration="$2"
    local out_path="$3"
    local path_save="$4"

    local fixed
    fixed=$(jq -n \
        --arg     path_data         "${PATH_DATA}/" \
        --arg     path_save         "${path_save}/" \
        --arg     modeltype         "${modeltype}" \
        --argjson seg_duration      "${seg_duration}" \
        --argjson num_epochs        "${NUM_EPOCHS}" \
        --argjson num_training      "${NUM_TRAINING}" \
        --argjson random_seed       "${RANDOM_SEED}" \
        --arg     balance_strategy  "${BALANCE_STRATEGY}" \
        '{
            "path_data"               : $path_data,
            "h5_filename"             : "all_extended_teaspoon_datasets.h5",
            "path_save"               : $path_save,
            "length_series"           : 100000,
            "segmentation_duration"   : $seg_duration,
            "exclude_states"          : ["default"],
            "force_cpu"               : false,
            "modeltype"               : $modeltype,
            "num_channels"            : 1,
            "conv1d_kernel_size"      : 3,
            "cnn_base_channels"       : 32,
            "cnn_channel_multipliers" : [1,2,4],
            "cnn_pooling"             : "mean",
            "test_size"               : 0.20,
            "val_size"                : 0.20,
            "num_epochs"              : $num_epochs,
            "verbose"                 : false,
            "num_training"            : $num_training,
            "random_seed"             : $random_seed,
            "reliability_threshold"   : 0.6,
            "balance_strategy"        : $balance_strategy
        }')

    # optimizer and muon_lr are searchable for both model types.
    # muon_lr is only used by the training code when optimizer=="muon";
    # the training script guards this: if optimizer != "muon", muon_lr is ignored.
    local search_space_trans1
    search_space_trans1=$(jq -n \
        --argjson patience "${PATIENCE}" \
        '{
            "optimizer"            : {"type": "choice",      "values": ["adamW", "adam", "muon"]},
            "lr"                   : {"type": "log_uniform", "low": 0.00001, "high": 0.001},
            "muon_lr"              : {"type": "log_uniform", "low": 0.0001,  "high": 0.01},
            "batch_size"           : {"type": "int_choice",  "values": [64, 128, 256]},
            "embed_size"           : {"type": "int_choice",  "values": [8, 16, 32, 64, 128, 256]},
            "nhead_encoder"        : {"type": "int_choice",  "values": [1, 2, 4, 8]},
            "dim_feedforward"      : {"type": "int_choice",  "values": [64, 128, 256, 512]},
            "num_encoderlayers"    : {"type": "int_choice",  "values": [2, 3, 4, 5]},
            "dropout"              : {"type": "choice",      "values": [0.0]},
            "size_linear_layers"   : {"type": "int_choice",  "values": [16, 32, 64, 128, 256]},
            "conv1d_emb"           : {"type": "choice",      "values": [true, false]},
            "norm_type"            : {"type": "choice",      "values": ["per-channel", "per-timestep", "global"]},
            "patience"             : {"type": "int_choice",  "values": [$patience]}
        }')

    local search_space_cnn1
    search_space_cnn1=$(jq -n \
        --argjson patience "${PATIENCE}" \
        '{
            "optimizer"               : {"type": "choice",     "values": ["adamW", "adam", "muon"]},
            "lr"                      : {"type": "log_uniform","low": 0.00001, "high": 0.001},
            "muon_lr"                 : {"type": "log_uniform","low": 0.0001,  "high": 0.01},
            "batch_size"              : {"type": "int_choice", "values": [64, 128, 256]},
            "cnn_base_channels"       : {"type": "int_choice", "values": [16, 32, 64]},
            "cnn_channel_multipliers" : {"type": "choice",     "values": [[1,2],[1,2,4],[1,2,4,8]]},
            "cnn_pooling"             : {"type": "choice",     "values": ["mean", "max", "last"]},
            "dropout"                 : {"type": "choice",     "values": [0.0, 0.1]},
            "size_linear_layers"      : {"type": "int_choice", "values": [16, 32, 64, 128]},
            "norm_type"               : {"type": "choice",     "values": ["per-channel", "per-timestep", "global"]},
            "patience"                : {"type": "int_choice", "values": [$patience]}
        }')

    local search_space
    if [[ "${modeltype}" == "trans1" ]]; then
        search_space="${search_space_trans1}"
    else
        search_space="${search_space_cnn1}"
    fi

    jq -n \
        --argjson fixed        "${fixed}" \
        --argjson search_space "${search_space}" \
        --arg     strategy     "${STRATEGY}" \
        --argjson n_trials     "${N_TRIALS}" \
        --argjson alpha        "${OBJECTIVE_ALPHA}" \
        '{
            "fixed"                    : $fixed,
            "search_space"             : $search_space,
            "strategy"                 : $strategy,
            "n_trials"                 : $n_trials,
            "objective_alpha"          : $alpha,
            "enforce_head_divisibility": true
        }' > "${out_path}"

    echo "  search config -> ${out_path}"
}

# Main loop: one submission block per (modeltype, seg_duration)
ALL_ARRAY_IDS=()
ALL_MERGE_IDS=()

for MODELTYPE in "${ALL_MODELTYPES[@]}"; do

    echo ""
    echo "=============================================================="
    echo "  Model type : ${MODELTYPE}"
    echo "=============================================================="

    MODEL_DIR="${PATH_SEARCH_ROOT}/${MODELTYPE}"
    mkdir -p "${MODEL_DIR}"

    for SEG in "${ALL_SEG_DURATIONS[@]}"; do

        echo ""
        echo "  -- seg${SEG} --"

        SEG_DIR="${MODEL_DIR}/seg${SEG}"
        LOG_DIR="${SEG_DIR}/logs"
        mkdir -p "${SEG_DIR}" "${LOG_DIR}"

        TAG="${MODELTYPE}_seg${SEG}"
        CONFIG_FILE="${SEG_DIR}/search_config_${TAG}.json"

        # Step 1: write search config
        _write_search_config "${MODELTYPE}" "${SEG}" "${CONFIG_FILE}" "${SEG_DIR}"

        # Step 2: generate trial configs (no GPU needed)
        echo "  Generating trial configs ..."
        python "${HPARAM_SCRIPT}" \
            --config "${CONFIG_FILE}" \
            --generate-configs

        N_GENERATED=$(find "${SEG_DIR}/trial_configs" \
            -name "trial_*.json" 2>/dev/null | wc -l)
        echo "  Generated ${N_GENERATED} trial configs"

        if [[ "${N_GENERATED}" -eq 0 ]]; then
            echo "  ERROR: no trial configs produced for ${TAG}, skipping."
            continue
        fi

        if [[ "${ARRAY}" == "true" ]]; then

            # Step 3: array script
            ARRAY_SCRIPT="${SEG_DIR}/slurm_array_${TAG}.sh"
            cat > "${ARRAY_SCRIPT}" << ARRAY_EOF
#!/bin/bash
#SBATCH --job-name=hpsearch_${TAG}
#SBATCH --output=${LOG_DIR}/trial_%a_%j.out
#SBATCH --error=${LOG_DIR}/trial_%a_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --time=${TIME_PER_TRIAL}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gres=${GPU_GRES}
#SBATCH --mem=${MEM_PER_TRIAL}
#SBATCH --array=1-${N_GENERATED}%${MAX_PARALLEL}

set -euo pipefail

export OMP_NUM_THREADS=${BLAS_THREADS}
export OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}

module load anaconda3
module load cuda
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "=============================================================="
echo "  Modeltype        : ${MODELTYPE}"
echo "  Seg dur          : ${SEG}"
echo "  Balance strategy : ${BALANCE_STRATEGY}"
echo "  Optimizer        : (searchable, see trial config)"
echo "  Trial            : \${SLURM_ARRAY_TASK_ID} / ${N_GENERATED}"
echo "  Node             : \${SLURMD_NODENAME}"
echo "  GPU              : \${CUDA_VISIBLE_DEVICES:-none}"
echo "  Output dir       : ${SEG_DIR}"
echo "=============================================================="

time python "${HPARAM_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --trial-idx "\${SLURM_ARRAY_TASK_ID}"

echo "Trial \${SLURM_ARRAY_TASK_ID} finished."
ARRAY_EOF

            # Step 4: merge script
            MERGE_SCRIPT="${SEG_DIR}/slurm_merge_${TAG}.sh"
            cat > "${MERGE_SCRIPT}" << MERGE_EOF
#!/bin/bash
#SBATCH --job-name=merge_${TAG}
#SBATCH --output=${LOG_DIR}/merge_%j.out
#SBATCH --error=${LOG_DIR}/merge_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -uo pipefail

module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "Merging results for ${TAG} ..."
echo "Output dir : ${SEG_DIR}"

# Primary merge via search_best_config.py
python "${HPARAM_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --merge-results

# Fallback: rebuild if primary produced nothing
BEST="${SEG_DIR}/best_config_${TAG}.json"
BEST_GENERIC="${SEG_DIR}/best_config.json"

if [[ ! -s "\${BEST}" && ! -s "\${BEST_GENERIC}" ]]; then
    echo "WARNING: no best config after merge, running rebuild fallback ..."
    python "${REBUILD_SCRIPT}" \
        --search-dir "${SEG_DIR}" \
        --alpha "${OBJECTIVE_ALPHA}" \
        --modeltypes "${MODELTYPE}"
fi

if [[ -s "\${BEST}" ]]; then
    echo "SUCCESS: \${BEST}"
    python3 -c "import json; cfg=json.load(open('\${BEST}')); print(json.dumps(cfg, indent=2))"
elif [[ -s "\${BEST_GENERIC}" ]]; then
    echo "SUCCESS (generic): \${BEST_GENERIC}"
else
    echo "ERROR: no best config produced for ${TAG}."
    echo "  Check trial logs in: ${LOG_DIR}/"
    exit 1
fi

echo "Results -> ${SEG_DIR}/all_trials.csv"
MERGE_EOF

            chmod +x "${ARRAY_SCRIPT}" "${MERGE_SCRIPT}"

            ARRAY_JOB_ID=$(_sbatch "${ARRAY_SCRIPT}")
            echo "  Array job : ${ARRAY_JOB_ID}  (${N_GENERATED} trials, max ${MAX_PARALLEL} parallel)"
            ALL_ARRAY_IDS+=("${ARRAY_JOB_ID}")

            MERGE_JOB_ID=$(_sbatch \
                --dependency="afterany:${ARRAY_JOB_ID}" \
                "${MERGE_SCRIPT}")
            echo "  Merge job : ${MERGE_JOB_ID}  (runs after array)"
            ALL_MERGE_IDS+=("${MERGE_JOB_ID}")

        else
            # Sequential mode
            SEQ_SCRIPT="${SEG_DIR}/slurm_seq_${TAG}.sh"
            cat > "${SEQ_SCRIPT}" << SEQ_EOF
#!/bin/bash
#SBATCH --job-name=hpsearch_seq_${TAG}
#SBATCH --output=${LOG_DIR}/seq_%j.out
#SBATCH --error=${LOG_DIR}/seq_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --time=${TIME_SEQ}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gres=${GPU_GRES}
#SBATCH --mem=60G

set -uo pipefail

export OMP_NUM_THREADS=${BLAS_THREADS}
export OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}

module load anaconda3
module load cuda
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "Sequential search - ${TAG}  job=\${SLURM_JOB_ID}  node=\${SLURMD_NODENAME}"
echo "Balance strategy  : ${BALANCE_STRATEGY}"
echo "Optimizer         : (searchable, see trial config)"
echo "Output dir        : ${SEG_DIR}"

time python "${HPARAM_SCRIPT}" --config "${CONFIG_FILE}"

# Fallback rebuild if best config is missing
BEST="${SEG_DIR}/best_config_${TAG}.json"
if [[ ! -s "\${BEST}" ]]; then
    echo "Running rebuild fallback ..."
    python "${REBUILD_SCRIPT}" \
        --search-dir "${SEG_DIR}" \
        --alpha "${OBJECTIVE_ALPHA}" \
        --modeltypes "${MODELTYPE}"
fi

echo "Done, results in ${SEG_DIR}/"
SEQ_EOF

            chmod +x "${SEQ_SCRIPT}"
            SEQ_JOB_ID=$(_sbatch "${SEQ_SCRIPT}")
            echo "  Sequential job : ${SEQ_JOB_ID}"
            ALL_ARRAY_IDS+=("${SEQ_JOB_ID}")
        fi

    done  # seg loop
done  # modeltype loop

# Summary
echo ""
echo "=================================================================="
echo "  Submitted for modeltypes : ${ALL_MODELTYPES[*]}"
echo "  Seg durations            : ${ALL_SEG_DURATIONS[*]}"
echo "  Balance strategy         : ${BALANCE_STRATEGY}"
echo "  Optimizer                : (searchable, adamW | adam | muon)"
echo "  muon_lr                  : (searchable, log_uniform 0.0001-0.01)"
if [[ "${ARRAY}" == "true" ]]; then
echo "  Array job IDs : ${ALL_ARRAY_IDS[*]:-none}"
echo "  Merge job IDs : ${ALL_MERGE_IDS[*]:-none}"
else
echo "  Sequential job IDs : ${ALL_ARRAY_IDS[*]:-none}"
fi
echo "  Trials per (modeltype x seg) : ${N_TRIALS}  (max parallel: ${MAX_PARALLEL})"
echo "  Resources per trial          : ${MEM_PER_TRIAL} RAM  |  ${CPUS_PER_TASK} CPUs  |  1 GPU (${GPU_TYPE:-any})"
echo ""
echo "  Directory structure:"
echo "  ${PATH_SEARCH_ROOT}/"
for MT in "${ALL_MODELTYPES[@]}"; do
echo "    ${MT}/"
    for SEG in "${ALL_SEG_DURATIONS[@]}"; do
echo "      seg${SEG}/"
echo "        - search_config_${MT}_seg${SEG}.json"
echo "        - trial_configs/          (per-trial hparam JSONs)"
echo "        - trial_*/                (per-trial outputs)"
echo "        - logs/                   (SLURM stdout/stderr)"
echo "        - all_trials.csv"
echo "        - best_config_${MT}_seg${SEG}.json"
    done
done
echo ""
echo "  Monitor    : squeue -u \$USER"
if [[ ${#ALL_ARRAY_IDS[@]} -gt 0 ]]; then
    ALL_IDS="${ALL_ARRAY_IDS[*]} ${ALL_MERGE_IDS[*]:-}"
    echo "  Cancel all : scancel ${ALL_IDS}"
fi
echo "=================================================================="
