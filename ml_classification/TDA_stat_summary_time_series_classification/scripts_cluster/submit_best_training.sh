#!/bin/bash
# submit_best_training.sh
# Submits SLURM array jobs for best_config.json and best_config_small.json
# found inside each seg*/ directory.
#
# Output structure per seg dir:
#     <seg_dir>/best_training/        runs using best_config.json
#     <seg_dir>/best_training_small/  runs using best_config_small.json
#
# Each of those contains:
#     task_1/ .. task_N_RUNS/   one isolated output dir per array task
#     all_trials_combined.csv  all tasks' all_trials.csv rows stacked
#     final_summary.txt        overall mean +/- std across all N_RUNS runs
#
# Notes
# -----
# - NOISE_LEVELS defines the sigma values for the post-training robustness
#   sweep (e.g. "0.0,0.05,0.1,0.2,0.5,1.0"). Set NOISE_LEVELS="" to
#   disable the sweep entirely.
# - NOISE_AUG_SIGMA enables training-time noise augmentation, written into
#   every per-task config JSON under "noise_aug_sigma". Set to "0.0" to
#   disable augmentation. Verify this value against whatever actually
#   produced your reported results before relying on it; the two do not
#   always agree by default.
# - USE_TEMPERATURE_SCALING is written as a JSON boolean into every
#   per-task config. "true" enables post-hoc calibration (recommended),
#   "false" disables it.
# - Each array task gets its own task_<N>/ output subdirectory, so
#   concurrent tasks never write to the same results.csv / all_trials.csv
#   at the same time. A separate SLURM job (dependent on the array
#   finishing) aggregates all tasks' all_trials.csv rows into
#   all_trials_combined.csv and computes the overall mean +/- std into
#   final_summary.txt.

# PARAMETERS
N_RUNS=10
MAX_PARALLEL=10
PH_WORKERS=4
BLAS_THREADS=4
MEM="30G"
TIME="2-00:00:00"


# Defaults, used when a key is absent from the config JSON.
LS_DEFAULT="0.1"
GCN_DEFAULT="1.0"
EMBED_DIM_DEFAULT="32"
FUSION_DEFAULT="low_rank"
RANK_DEFAULT="8"
N_ATTN_LAYERS_DEFAULT="1"
N_HEADS_DEFAULT="4"
FFN_DIM_DEFAULT="0"
NORM_TYPE_DEFAULT="none"
OPTIMIZER_DEFAULT="adam"
MUON_LR_DEFAULT="0.02"

# Noise levels for robustness sweep: comma-separated sigma values.
# Set to "" to disable (no sweep, no extra files written).
NOISE_LEVELS="0.0,0.025,0.05,0.075,0.1,0.2,0.5,1.0"

# Noise augmentation sigma for training. Set to "0.0" to disable.
NOISE_AUG_SIGMA="0.05"

# Temperature scaling toggle.
# "true" fits T on the val set after each run (recommended).
# "false" skips calibration.
USE_TEMPERATURE_SCALING="true"

# MODE
MODE="all"   # "single" or "all"

# Paths
PATH_OUTPUT="/home/sharareh.sayyad/Computtional_topology/Computational_topology/ml_classification/output_tda_stat_summary"
BASE_SEARCH_DIR="${PATH_OUTPUT}/hparam_search_results"
PATH_CODE="/home/sharareh.sayyad/Computtional_topology/Computational_topology/ml_classification/TDA_stat_summary_time_series_classification"
CONDA_ENV="TDA"
TRAIN_SCRIPT="main_train_time_series_tda_stat_summary.py"

# Used only when MODE=single
SEARCH_DIR="${BASE_SEARCH_DIR}/seg500"

# Auto-detect: if invoked from inside a seg*/ directory, switch to single
CURRENT_DIR="$(pwd)"
if [[ "$(basename "${CURRENT_DIR}")" == seg* ]]; then
    echo "Auto-detected seg*/ directory, switching to MODE=single"
    MODE="single"
    SEARCH_DIR="${CURRENT_DIR}"
fi


# Build a JSON array string from NOISE_LEVELS for injection into the
# per-task config.
#   NOISE_LEVELS="0.0,0.05,0.5"  ->  NOISE_LEVELS_JSON="[0.0,0.05,0.5]"
#   NOISE_LEVELS=""              ->  NOISE_LEVELS_JSON="null"
if [ -n "${NOISE_LEVELS}" ]; then
    NOISE_LEVELS_JSON="[${NOISE_LEVELS}]"
else
    NOISE_LEVELS_JSON="null"
fi


# submit_one <best_config_path> <results_dir> <job_label>
submit_one() {
    local best_config="$1"
    local results_dir="$2"
    local job_label="$3"

    if [ ! -f "${best_config}" ]; then
        echo "    SKIP: config not found: ${best_config}"
        return 1
    fi

    mkdir -p "${results_dir}"

    # read ph_workers from config (fall back to global PH_WORKERS)
    local ph_workers="${PH_WORKERS}"
    if command -v jq &>/dev/null; then
        local jq_ph
        jq_ph=$(jq -r '.ph_workers // empty' "${best_config}" 2>/dev/null)
        [ -n "${jq_ph}" ] && [ "${jq_ph}" != "null" ] && ph_workers="${jq_ph}"
    fi
    local cpus_per_task=$(( ph_workers + 2 ))

    # read summary values for echo (fall back to shell defaults)
    local cfg_ls cfg_gcn cfg_embed cfg_fusion cfg_rank cfg_norm
    local cfg_optimizer cfg_muon_lr
    cfg_ls="${LS_DEFAULT}";           cfg_gcn="${GCN_DEFAULT}"
    cfg_embed="${EMBED_DIM_DEFAULT}"; cfg_fusion="${FUSION_DEFAULT}"
    cfg_rank="${RANK_DEFAULT}";       cfg_norm="${NORM_TYPE_DEFAULT}"
    cfg_optimizer="${OPTIMIZER_DEFAULT}"; cfg_muon_lr="${MUON_LR_DEFAULT}"
    if command -v jq &>/dev/null; then
        cfg_ls=$(        jq -r ".label_smoothing  // \"${LS_DEFAULT}\""           "${best_config}" 2>/dev/null)
        cfg_gcn=$(       jq -r ".grad_clip_norm   // \"${GCN_DEFAULT}\""          "${best_config}" 2>/dev/null)
        cfg_embed=$(     jq -r ".embed_dim        // \"${EMBED_DIM_DEFAULT}\""    "${best_config}" 2>/dev/null)
        cfg_fusion=$(    jq -r ".fusion           // \"${FUSION_DEFAULT}\""       "${best_config}" 2>/dev/null)
        cfg_rank=$(      jq -r ".rank             // \"${RANK_DEFAULT}\""         "${best_config}" 2>/dev/null)
        cfg_norm=$(      jq -r ".norm_type        // \"${NORM_TYPE_DEFAULT}\""    "${best_config}" 2>/dev/null)
        cfg_optimizer=$( jq -r ".optimizer        // \"${OPTIMIZER_DEFAULT}\""   "${best_config}" 2>/dev/null)
        cfg_muon_lr=$(   jq -r ".muon_lr          // \"${MUON_LR_DEFAULT}\""     "${best_config}" 2>/dev/null)
    fi

    local array_script="${results_dir}/slurm_${job_label}.sh"

    cat > "${array_script}" << SLURM
#!/bin/bash
#SBATCH --job-name=tda_${job_label}
#SBATCH --output=${results_dir}/run_%a_%j.out
#SBATCH --error=${results_dir}/run_%a_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --partition=cas
#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem=${MEM}
#SBATCH --array=1-${N_RUNS}%${MAX_PARALLEL}

export OMP_NUM_THREADS=${BLAS_THREADS}
export OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}

module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "job=${job_label} | Run \${SLURM_ARRAY_TASK_ID}/${N_RUNS} | Node \${SLURMD_NODENAME}"

TASK_SEED=\$(( SLURM_JOB_ID * 100 + SLURM_ARRAY_TASK_ID ))
echo "  random_seed      : \${TASK_SEED}"

# Each task gets its own output dir so parallel tasks never write to the
# same results.csv / all_trials.csv at the same time.
TASK_DIR="${results_dir}/task_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\${TASK_DIR}"
echo "  task_dir          : \${TASK_DIR}"

# Per-task config patch: overrides random_seed and path_save, and fills
# in every architecture/regularisation/robustness key from the source
# config (falling back to the shell defaults above if a key is absent).
TASK_CONFIG="\${TASK_DIR}/config_task_\${SLURM_ARRAY_TASK_ID}.json"
jq --argjson seed                  "\${TASK_SEED}" \
   --arg     path_save             "\${TASK_DIR}" \
   --argjson ls                    "\$(jq ".label_smoothing              // ${LS_DEFAULT}"            "${best_config}")" \
   --argjson gcn                   "\$(jq ".grad_clip_norm               // ${GCN_DEFAULT}"           "${best_config}")" \
   --argjson embed_dim             "\$(jq ".embed_dim                    // ${EMBED_DIM_DEFAULT}"     "${best_config}")" \
   --arg     fusion                "\$(jq -r ".fusion                    // \"${FUSION_DEFAULT}\""    "${best_config}")" \
   --argjson rank                  "\$(jq ".rank                         // ${RANK_DEFAULT}"          "${best_config}")" \
   --argjson n_attn                "\$(jq ".n_attn_layers                // ${N_ATTN_LAYERS_DEFAULT}" "${best_config}")" \
   --argjson n_heads               "\$(jq ".n_heads                      // ${N_HEADS_DEFAULT}"       "${best_config}")" \
   --argjson ffn_dim               "\$(jq ".ffn_dim                      // ${FFN_DIM_DEFAULT}"       "${best_config}")" \
   --arg     norm_type             "\$(jq -r ".norm_type                 // \"${NORM_TYPE_DEFAULT}\"" "${best_config}")" \
   --arg     optimizer             "\$(jq -r ".optimizer                 // \"${OPTIMIZER_DEFAULT}\"" "${best_config}")" \
   --argjson muon_lr               "\$(jq ".muon_lr                      // ${MUON_LR_DEFAULT}"       "${best_config}")" \
   --argjson noise_levels          '${NOISE_LEVELS_JSON}' \
   --argjson noise_aug_sigma       '${NOISE_AUG_SIGMA}' \
   --argjson use_temp_scaling      '${USE_TEMPERATURE_SCALING}' \
   '.random_seed              = \$seed             |
    .path_save                = \$path_save        |
    .label_smoothing          = \$ls               |
    .grad_clip_norm           = \$gcn              |
    .embed_dim                = \$embed_dim        |
    .fusion                   = \$fusion           |
    .rank                     = \$rank             |
    .n_attn_layers            = \$n_attn           |
    .n_heads                  = \$n_heads          |
    .ffn_dim                  = \$ffn_dim          |
    .norm_type                = \$norm_type        |
    .optimizer                = \$optimizer        |
    .muon_lr                  = \$muon_lr          |
    .noise_levels              = \$noise_levels     |
    .noise_aug_sigma          = \$noise_aug_sigma  |
    .use_temperature_scaling  = \$use_temp_scaling' \
    "${best_config}" > "\${TASK_CONFIG}"

# Echo all patched values
echo "  path_save               : \${TASK_DIR}"
echo "  label_smoothing         : \$(jq '.label_smoothing'                       "\${TASK_CONFIG}")"
echo "  grad_clip_norm          : \$(jq '.grad_clip_norm'                        "\${TASK_CONFIG}")"
echo "  embed_dim               : \$(jq '.embed_dim'                             "\${TASK_CONFIG}")"
echo "  fusion                  : \$(jq -r '.fusion'                             "\${TASK_CONFIG}")"
echo "  rank                    : \$(jq '.rank'                                  "\${TASK_CONFIG}")"
echo "  n_attn_layers           : \$(jq '.n_attn_layers'                         "\${TASK_CONFIG}")"
echo "  n_heads                 : \$(jq '.n_heads'                               "\${TASK_CONFIG}")"
echo "  ffn_dim                 : \$(jq '.ffn_dim'                               "\${TASK_CONFIG}")"
echo "  norm_type               : \$(jq -r '.norm_type'                          "\${TASK_CONFIG}")"
echo "  optimizer               : \$(jq -r '.optimizer'                          "\${TASK_CONFIG}")"
echo "  muon_lr                 : \$(jq '.muon_lr'                               "\${TASK_CONFIG}")"
echo "  noise_levels            : \$(jq -c '.noise_levels // "disabled"'         "\${TASK_CONFIG}")"
echo "  noise_aug_sigma         : \$(jq '.noise_aug_sigma'                       "\${TASK_CONFIG}")"
echo "  use_temperature_scaling : \$(jq '.use_temperature_scaling'               "\${TASK_CONFIG}")"
echo "  segmentation_dur        : \$(jq '.segmentation_duration // "n/a"'        "\${TASK_CONFIG}")"

cd ${PATH_CODE}
time python ${TRAIN_SCRIPT} --config "\${TASK_CONFIG}"

rm -f "\${TASK_CONFIG}"
echo "Run \${SLURM_ARRAY_TASK_ID} done."
SLURM

    chmod +x "${array_script}"
    local job_id
    job_id=$(sbatch --parsable "${array_script}")

    echo "    Submitted job ${job_id}  (${N_RUNS} runs)"
    echo "    Config    : ${best_config}"
    echo "    Logs      : ${results_dir}/"
    echo "    optimizer=${cfg_optimizer}  muon_lr=${cfg_muon_lr}"
    echo "    fusion=${cfg_fusion}  embed_dim=${cfg_embed}  rank=${cfg_rank}  norm_type=${cfg_norm}"
    echo "    label_smoothing=${cfg_ls}  grad_clip_norm=${cfg_gcn}"
    echo "    noise_levels=${NOISE_LEVELS:-disabled}"
    echo "    noise_aug_sigma=${NOISE_AUG_SIGMA}  use_temperature_scaling=${USE_TEMPERATURE_SCALING}"

    # Aggregation job: runs only after the whole array finishes
    local aggregate_script="${results_dir}/slurm_${job_label}_aggregate.sh"

    cat > "${aggregate_script}" << SLURM2
#!/bin/bash
#SBATCH --job-name=agg_${job_label}
#SBATCH --output=${results_dir}/aggregate_%j.out
#SBATCH --error=${results_dir}/aggregate_%j.err
#SBATCH --partition=cas
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --dependency=afterany:${job_id}

module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

python - << 'PY'
import glob
import pandas as pd

results_dir = "${results_dir}"
files = sorted(glob.glob(results_dir + "/task_*/all_trials.csv"))

if not files:
    print(f"No per-task all_trials.csv files found under {results_dir}")
else:
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as exc:
            print(f"  WARNING: could not read {f}: {exc}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(results_dir + "/all_trials_combined.csv", index=False)

    metrics = ["mean_accuracy", "mean_f1", "mean_gmean", "mean_precision", "mean_recall"]
    lines = [f"Aggregated over {len(df_all)} run(s) found in {len(files)} task dir(s)\\n"]
    for m in metrics:
        if m in df_all.columns:
            mu  = df_all[m].mean()
            sd  = df_all[m].std()
            lines.append(f"{m:16s}: {mu:.4f} +/- {sd:.4f}\\n")

    with open(results_dir + "/final_summary.txt", "w") as fh:
        fh.writelines(lines)

    print("".join(lines))
    print(f"Combined rows written to {results_dir}/all_trials_combined.csv")
    print(f"Summary written to       {results_dir}/final_summary.txt")
PY
SLURM2

    chmod +x "${aggregate_script}"
    local agg_job_id
    agg_job_id=$(sbatch --parsable "${aggregate_script}")
    echo "    Aggregation job ${agg_job_id}  (runs automatically after ${job_id} finishes)"
    echo "    Final mean will be in: ${results_dir}/final_summary.txt"
}


# process_seg_dir <seg_dir>
process_seg_dir() {
    local seg_dir="$1"
    local seg_name
    seg_name=$(basename "${seg_dir}")

    echo ""
    echo "------------------------------------------------------------"
    echo "  ${seg_name}"
    echo "------------------------------------------------------------"

    local cfg_best="${seg_dir}/best_config.json"
    local dir_best="${seg_dir}/best_training"
    echo "  [best]"
    submit_one "${cfg_best}" "${dir_best}" "${seg_name}"

    local cfg_small="${seg_dir}/best_config_small.json"
    local dir_small="${seg_dir}/best_training_small"
    echo "  [small]"
    submit_one "${cfg_small}" "${dir_small}" "${seg_name}_small"
}


# MODE=single
if [ "${MODE}" = "single" ]; then

    process_seg_dir "${SEARCH_DIR}"

    echo ""
    echo "=============================================================="
    echo "  Monitor : squeue -u \$USER"
    echo "=============================================================="


# MODE=all
elif [ "${MODE}" = "all" ]; then

    mapfile -t SEG_DIRS < <(find "${BASE_SEARCH_DIR}" -maxdepth 1 -type d -name 'seg*' | sort)

    if [ ${#SEG_DIRS[@]} -eq 0 ]; then
        echo "ERROR: no seg*/ subdirectories found under ${BASE_SEARCH_DIR}"
        exit 1
    fi

    echo "Found ${#SEG_DIRS[@]} seg*/ directories:"
    for d in "${SEG_DIRS[@]}"; do echo "  ${d}"; done

    echo ""
    echo "=============================================================="
    echo "  Submitting best + small jobs for each seg dir ..."
    echo "  N_RUNS                  : ${N_RUNS}"
    echo "  noise_levels            : ${NOISE_LEVELS:-disabled}"
    echo "  noise_aug_sigma         : ${NOISE_AUG_SIGMA}"
    echo "  use_temperature_scaling : ${USE_TEMPERATURE_SCALING}"
    echo "=============================================================="

    for seg_dir in "${SEG_DIRS[@]}"; do
        process_seg_dir "${seg_dir}"
    done

    echo ""
    echo "=============================================================="
    echo "  All jobs submitted.  Monitor: squeue -u \$USER"
    echo ""
    echo "  Output layout:"
    for seg_dir in "${SEG_DIRS[@]}"; do
        seg_name=$(basename "${seg_dir}")
        echo "    ${seg_dir}/"
        echo "      best_training/        best_config.json runs"
        echo "        task_1..task_${N_RUNS}/   one isolated run each"
        echo "        final_summary.txt         mean +/- std over all ${N_RUNS} runs"
        echo "      best_training_small/  best_config_small.json runs (same layout)"
    done
    echo "=============================================================="

else
    echo "ERROR: unknown MODE='${MODE}'.  Set MODE=single or MODE=all."
    exit 1
fi
