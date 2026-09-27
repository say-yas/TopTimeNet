#!/bin/bash
# submit_config_search.sh
# Generates search_config.json, pre-builds the TDA cache, runs the trial
# array, and merges results: one SLURM pipeline per segmentation duration.

N_TRIALS=400
STRATEGY="random"
OBJECTIVE_ALPHA=0.6
NUM_EPOCHS=500
PATIENCE=20
NUM_TRAINING=1
RANDOM_SEED=42
MAX_PARALLEL=10

SEG_DUR_VALUES=(1000)

# Default constants, matching _DEFAULTS in search_best_config.py
LS_DEFAULT="0.1"
GCN_DEFAULT="1.0"
EMBED_DIM_DEFAULT="32"
FUSION_DEFAULT="low_rank"
RANK_DEFAULT="8"
N_ATTN_LAYERS_DEFAULT="1"
N_HEADS_DEFAULT="4"
FFN_DIM_DEFAULT="0"
NORM_TYPE_DEFAULT="none"
MUON_LR_DEFAULT="0.02"

# IQR clipping bounds: pipeline constants, not hyperparameters
CLIP_LO_PCT="1.0"
CLIP_HI_PCT="99.0"

# Noise augmentation: disabled during search
NOISE_AUG_SIGMA="0.0"

# Temperature scaling: disabled during search
USE_TEMPERATURE_SCALING="false"

# Robustness sweep: disabled during search (null), enabled in best-model runs
NOISE_LEVELS_SEARCH="null"
NOISE_BATCH_SIZE="32"

N_HOM_DIMS=2
TAKENS_DIM=2
TAKENS_DELAY=20
N_BETTI_BINS=50
N_PI_BINS=15
PI_SIGMA=0.05
FPS_N_PTS=250
PH_WORKERS=4
PRECOMPUTE_BATCH_SIZE=128
USE_CACHE=true
FORCE_RECOMPUTE=false

KEEP_STATES='["periodic", "chaotic"]'
BALANCE_STRATEGY="undersample"

BLAS_THREADS=4
CPUS_PREBUILD=$(( PH_WORKERS + 2 ))
CPUS_TRIAL=2
MEM_PREBUILD="30G"
MEM_TRIAL="8G"

PATH_DATA="/home/sharareh.sayyad/Computtional_topology/Computational_topology/dataset/extended_teaspoon_dataset"
PATH_OUTPUT="/home/sharareh.sayyad/Computtional_topology/Computational_topology/ml_classification/output_tda_stat_summary"
PATH_SEARCH="${PATH_OUTPUT}/hparam_search_results"
PATH_CODE="/home/sharareh.sayyad/Computtional_topology/Computational_topology/ml_classification/TDA_stat_summary_time_series_classification/"
CONDA_ENV="TDA"
REBUILD_SCRIPT="${PATH_CODE}/config_related/rebuild_best_config.py"
TRAIN_SCRIPT="main_train_time_series_tda_stat_summary.py"
SCRIPT="config_related/search_best_config.py"
PREBUILD_SCRIPT="${PATH_CODE}/precompute_tda.py"
TMP_DIR="tmp_out_err"

mkdir -p "${PATH_SEARCH}/${TMP_DIR}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

ALL_ARRAY_JOB_IDS=()
ALL_MERGE_JOB_IDS=()
ALL_PREBUILD_JOB_IDS=()

for SEG_DUR in "${SEG_DUR_VALUES[@]}"; do

    echo ""
    echo "=============================================================="
    echo "  seg_dur=${SEG_DUR}  fps_n_pts=${FPS_N_PTS}"
    echo "=============================================================="

    SEG_SEARCH_DIR="${PATH_SEARCH}/seg${SEG_DUR}"
    mkdir -p "${SEG_SEARCH_DIR}/${TMP_DIR}"

    CONFIG_FILE="${SEG_SEARCH_DIR}/search_config.json"
    TDA_CACHE_PATH="${SEG_SEARCH_DIR}/tda_cache.pt"

    cat > "${CONFIG_FILE}" << JSONEOF
{
    "fixed": {
        "path_data"             : "${PATH_DATA}/",
        "h5_filename"           : "all_extended_teaspoon_datasets.h5",
        "path_save"             : "${SEG_SEARCH_DIR}/",
        "length_series"         : 100000,
        "segmentation_duration" : ${SEG_DUR},
        "num_channels"          : 1,
        "exclude_states"        : [],
        "keep_states"           : ${KEEP_STATES},
        "balance_strategy"      : "${BALANCE_STRATEGY}",
        "force_cpu"             : false,
        "random_seed"           : ${RANDOM_SEED},
        "test_size"             : 0.10,
        "val_size"              : 0.10,
        "num_epochs"            : ${NUM_EPOCHS},
        "optimizer"             : "adam",
        "muon_lr"               : ${MUON_LR_DEFAULT},
        "verbose"               : false,
        "num_training"          : ${NUM_TRAINING},
        "reliability_threshold" : 0.6,
        "n_hom_dims"            : ${N_HOM_DIMS},
        "takens_dim"            : ${TAKENS_DIM},
        "takens_delay"          : ${TAKENS_DELAY},
        "n_betti_bins"          : ${N_BETTI_BINS},
        "n_pi_bins"             : ${N_PI_BINS},
        "pi_sigma"              : ${PI_SIGMA},
        "fps_n_pts"             : ${FPS_N_PTS},
        "ph_workers"            : ${PH_WORKERS},
        "precompute_batch_size" : ${PRECOMPUTE_BATCH_SIZE},
        "use_cache"             : ${USE_CACHE},
        "tda_cache_path"        : "${TDA_CACHE_PATH}",
        "force_recompute"       : ${FORCE_RECOMPUTE},
        "label_smoothing"       : ${LS_DEFAULT},
        "grad_clip_norm"        : ${GCN_DEFAULT},
        "embed_dim"             : ${EMBED_DIM_DEFAULT},
        "fusion"                : "${FUSION_DEFAULT}",
        "rank"                  : ${RANK_DEFAULT},
        "n_attn_layers"         : ${N_ATTN_LAYERS_DEFAULT},
        "n_heads"               : ${N_HEADS_DEFAULT},
        "ffn_dim"               : ${FFN_DIM_DEFAULT},
        "norm_type"             : "${NORM_TYPE_DEFAULT}",
        "head_hidden"           : [64, 32],
        "activation"            : "gelu",
        "dropout"               : 0.1,
        "clip_lo_pct"           : ${CLIP_LO_PCT},
        "clip_hi_pct"           : ${CLIP_HI_PCT},
        "noise_aug_sigma"       : ${NOISE_AUG_SIGMA},
        "use_temperature_scaling": ${USE_TEMPERATURE_SCALING},
        "noise_levels"          : ${NOISE_LEVELS_SEARCH},
        "noise_batch_size"      : ${NOISE_BATCH_SIZE}
    },
    "search_space": {
        "lr"             : {"type": "log_uniform",  "low": 1e-6,  "high": 5e-3},
        "batch_size"     : {"type": "int_choice",   "values": [128]},
        "patience"       : {"type": "int_choice",   "values": [50]},
        "optimizer"      : {"type": "choice",       "values": ["adam", "adamw"]},
        "muon_lr"        : {"type": "log_uniform",  "low": 1e-3,  "high": 0.1},
        "label_smoothing": {"type": "choice",       "values": [0.0, 0.05, 0.1]},
        "grad_clip_norm" : {"type": "choice",       "values": [0.5, 1.0, 2.0, 5.0]},
        "norm_type"      : {"type": "choice",       "values": ["none", "global", "per-channel"]},
        "embed_dim"      : {"type": "int_choice",   "values": [16, 32, 64, 128]},
        "fusion"         : {"type": "choice",       "values": ["bilinear", "low_rank", "linear_attn", "gated", "mgta"]},
        "rank"           : {"type": "int_choice",   "values": [4, 8, 16]},
        "n_attn_layers"  : {"type": "int_choice",   "values": [1, 2]},
        "n_heads"        : {"type": "int_choice",   "values": [2, 4, 8]},
        "head_hidden"    : {"type": "choice",       "values": [[], [32], [64], [32, 16], [64, 32], [128, 64], [64, 32, 16]]},
        "activation"     : {"type": "choice",       "values": ["relu", "gelu", "leaky_relu"]},
        "dropout"        : {"type": "choice",       "values": [0.0, 0.05, 0.1, 0.2]}
    },
    "strategy"                  : "${STRATEGY}",
    "n_trials"                  : ${N_TRIALS},
    "objective_alpha"           : ${OBJECTIVE_ALPHA},
    "enforce_head_divisibility" : true
}
JSONEOF
    echo "  search_config.json -> ${CONFIG_FILE}"

    cd "${PATH_CODE}"
    python "${SCRIPT}" --config "${CONFIG_FILE}" --generate-configs
    if [ $? -ne 0 ]; then
        echo "  ERROR: --generate-configs failed"
        cd "${PATH_SEARCH}"; continue
    fi
    N_GENERATED=$(ls "${SEG_SEARCH_DIR}/trial_configs/trial_"*.json 2>/dev/null | wc -l)
    echo "  Generated ${N_GENERATED} trial configs"
    if [ "${N_GENERATED}" -eq 0 ]; then
        echo "  ERROR: no configs produced"
        cd "${PATH_SEARCH}"; continue
    fi

    cd "${SEG_SEARCH_DIR}"

    # SLURM: pre-build TDA cache
    prebuild_job="${SEG_SEARCH_DIR}/slurm_prebuild_seg${SEG_DUR}.sh"
    cat > "${prebuild_job}" << EOF
#!/bin/bash
#SBATCH --job-name=tda_cache_seg${SEG_DUR}
#SBATCH --output=${SEG_SEARCH_DIR}/${TMP_DIR}/prebuild_%j.out
#SBATCH --error=${SEG_SEARCH_DIR}/${TMP_DIR}/prebuild_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --partition=cas
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PREBUILD}
#SBATCH --mem=${MEM_PREBUILD}

export OMP_NUM_THREADS=${BLAS_THREADS}
export OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}

module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "Pre-build seg_dur=${SEG_DUR} | fps_n_pts=${FPS_N_PTS} | \${SLURMD_NODENAME}"

cd ${PATH_CODE}
time python "${PREBUILD_SCRIPT}" --config "${CONFIG_FILE}"

EXIT_CODE=\$?
if [ \${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: precompute_tda.py exited \${EXIT_CODE}"
    exit \${EXIT_CODE}
fi
if [ ! -f "${TDA_CACHE_PATH}" ]; then
    echo "ERROR: cache file not found after build"
    exit 1
fi

# Validate cache keys and dimensions
python3 -c "
import torch, sys
p = torch.load('${TDA_CACHE_PATH}', weights_only=True)
keys = list(p.keys())
print(f'Cache keys: {keys}')
print(f'feat_dim={p[\"feat_dim\"]}  N={len(p[\"features\"])}')
if 'labels' not in p:
    print('ERROR: labels key missing from cache')
    sys.exit(1)
if 'layout' not in p:
    print('WARNING: layout key missing, old cache format')
if 'clip_q_lo' not in p:
    print('WARNING: clip_q_lo missing, cache built without IQR clipping')
else:
    print('IQR clip bounds: present')
print('Cache OK')
"
if [ \$? -ne 0 ]; then
    echo "ERROR: cache validation failed"
    exit 1
fi

echo "Cache verified: \$(du -sh "${TDA_CACHE_PATH}")"
echo "Pre-build complete."
EOF

    chmod +x "${prebuild_job}"
    PREBUILD_JOB_ID=$(sbatch --parsable "${prebuild_job}")
    echo "  Prebuild job  : ${PREBUILD_JOB_ID}"
    ALL_PREBUILD_JOB_IDS+=("${PREBUILD_JOB_ID}")

    # SLURM: trial array
    array_script="${SEG_SEARCH_DIR}/slurm_array_seg${SEG_DUR}.sh"
    cat > "${array_script}" << EOF
#!/bin/bash
#SBATCH --job-name=tda_seg${SEG_DUR}
#SBATCH --output=${SEG_SEARCH_DIR}/${TMP_DIR}/trial_%a_%j.out
#SBATCH --error=${SEG_SEARCH_DIR}/${TMP_DIR}/trial_%a_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_TRIAL}
#SBATCH --mem=${MEM_TRIAL}
#SBATCH --array=1-${N_GENERATED}%${MAX_PARALLEL}

export OMP_NUM_THREADS=${BLAS_THREADS}
export OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}

module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "Trial \${SLURM_ARRAY_TASK_ID} | seg_dur=${SEG_DUR} | \${SLURMD_NODENAME}"

if [ ! -f "${TDA_CACHE_PATH}" ]; then
    echo "ABORT: tda_cache.pt missing"
    exit 1
fi

python3 -c "
import torch, sys
p = torch.load('${TDA_CACHE_PATH}', weights_only=True)
if 'labels' not in p:
    print('ABORT: cache missing labels key, delete cache and rerun prebuild')
    sys.exit(1)
print(f'Cache OK: feat_dim={p[\"feat_dim\"]}  N={len(p[\"features\"])}')
"
if [ \$? -ne 0 ]; then
    exit 1
fi

cd ${PATH_CODE}
time python ${SCRIPT} --config ${CONFIG_FILE} --trial-idx \${SLURM_ARRAY_TASK_ID}
echo "Trial \${SLURM_ARRAY_TASK_ID} done."
EOF

    # SLURM: merge and best config
    merge_script="${SEG_SEARCH_DIR}/slurm_merge_seg${SEG_DUR}.sh"
    cat > "${merge_script}" << EOF
#!/bin/bash
#SBATCH --job-name=tda_merge_seg${SEG_DUR}
#SBATCH --output=${SEG_SEARCH_DIR}/${TMP_DIR}/merge_%j.out
#SBATCH --error=${SEG_SEARCH_DIR}/${TMP_DIR}/merge_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

cd ${PATH_CODE}
python ${SCRIPT} --config ${CONFIG_FILE} --merge-results

BEST="${SEG_SEARCH_DIR}/best_config_seg${SEG_DUR}.json"
if [ ! -s "\${BEST}" ]; then
    python ${REBUILD_SCRIPT} \
        --search-dir ${SEG_SEARCH_DIR} \
        --alpha      ${OBJECTIVE_ALPHA} \
        --metric     f1_test \
        --out        "\${BEST}"
fi
[ -s "\${BEST}" ] && echo "SUCCESS -> \${BEST}" || { echo "ERROR: best_config missing"; exit 1; }
EOF

    chmod +x "${array_script}" "${merge_script}"

    ARRAY_JOB_ID=$(sbatch --parsable \
        --dependency=afterany:"${PREBUILD_JOB_ID}" \
        "${array_script}")
    MERGE_JOB_ID=$(sbatch --parsable \
        --dependency=afterany:"${ARRAY_JOB_ID}" \
        "${merge_script}")

    ALL_ARRAY_JOB_IDS+=("${ARRAY_JOB_ID}")
    ALL_MERGE_JOB_IDS+=("${MERGE_JOB_ID}")

    echo "  Array job     : ${ARRAY_JOB_ID}  (${N_GENERATED} trials)"
    echo "  Merge job     : ${MERGE_JOB_ID}"

    cd "${PATH_SEARCH}"
done

echo ""
echo "=============================================================="
for i in "${!SEG_DUR_VALUES[@]}"; do
    seg="${SEG_DUR_VALUES[$i]}"
    echo "  seg_dur=${seg}"
    echo "    prebuild=${ALL_PREBUILD_JOB_IDS[$i]:-n/a} -> array=${ALL_ARRAY_JOB_IDS[$i]:-n/a} -> merge=${ALL_MERGE_JOB_IDS[$i]:-n/a}"
    echo "    cache   : ${PATH_SEARCH}/seg${seg}/tda_cache.pt"
    echo "    results : ${PATH_SEARCH}/seg${seg}/all_trials.csv"
done
echo "  Monitor: squeue -u \$USER"
echo "=============================================================="
