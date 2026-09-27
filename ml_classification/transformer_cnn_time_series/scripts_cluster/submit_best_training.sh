#!/bin/bash
# submit_best_training.sh
#
# Finds every best_config_<modeltype>.json at any depth under PATH_SEARCH,
# submits a SLURM job array for each, and saves results in the same
# directory as the config file.
#
# Usage:
#     bash submit_best_training.sh                        # all found configs
#     bash submit_best_training.sh --modeltype trans1     # only trans1 configs
#     bash submit_best_training.sh --dry-run              # print, don't submit
#
# Notes
# -----
# Every per-task config gets NOISE_LEVELS, USE_TEMPERATURE_SCALING,
# ECE_BINS, and NOISE_BATCH_SIZE patched in, mirroring the pattern used for
# TopTimeNet's own best-training submit script. This triggers the
# post-training robustness sweep implemented in
# run_training_transformer_multiset.py / main_train_time_series_transformer.py,
# producing robustness_sweep_run{N}.csv / robustness_all_runs.csv for both
# the CNN and Transformer baselines, in the same schema as TopTimeNet's
# sweep, so results from all three models can be plotted with the same
# plot_robustness_paper.py script.
#
# Set NOISE_LEVELS="" below to disable the sweep entirely and restore the
# no-robustness-sweep behaviour.
set -euo pipefail


# PARAMETERS: edit these
N_RUNS=10
MAX_PARALLEL=10
BLAS_THREADS=4
CPUS_PER_TASK=$(( BLAS_THREADS + 2 ))
MEM="30G"
TIME="6-00:00:00"
GPU_TYPE="tesla"          # set "" for any GPU, or "a100", "v100", etc.
CONDA_ENV="TDA"

# Noise levels for the post-training robustness sweep, comma-separated
# sigma values, matching those used for TopTimeNet so the three models'
# sweeps are directly comparable. Set to "" to disable (no sweep, no
# extra files).
NOISE_LEVELS="0.0,0.025,0.05,0.075,0.1,0.2,0.5,1.0"

# Temperature scaling toggle.
# "true"  = fit T on val set after each run (recommended, matches TopTimeNet).
# "false" = skip calibration (T=1.0 in the sweep).
USE_TEMPERATURE_SCALING="true"

# Number of equal-width bins for Expected Calibration Error.
ECE_BINS="10"

# Batch size used during the sweep's forward passes.
NOISE_BATCH_SIZE="32"

# PATHS: edit these
BASE="/home/sharareh.sayyad/Computtional_topology/Computational_topology"
PATH_CODE="${BASE}/ml_classification/transformer_cnn_time_series"
PATH_OUTPUT="${BASE}/ml_classification/output_transformer_cnn"

# Root results directory; script searches recursively inside this
PATH_SEARCH="${PATH_OUTPUT}/hparam_search_results"

TRAIN_SCRIPT="${PATH_CODE}/main_train_time_series_transformer.py"

# Parse CLI flags
DRY_RUN=0
FORCE_MODELTYPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;            shift ;;
        --modeltype) FORCE_MODELTYPE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Build a JSON array string from NOISE_LEVELS for injection into the
# per-task config.
#   NOISE_LEVELS="0.0,0.05,0.5"  ->  NOISE_LEVELS_JSON="[0.0,0.05,0.5]"
#   NOISE_LEVELS=""              ->  NOISE_LEVELS_JSON="null"
if [[ -n "${NOISE_LEVELS}" ]]; then
    NOISE_LEVELS_JSON="[${NOISE_LEVELS}]"
else
    NOISE_LEVELS_JSON="null"
fi

# Validate dependencies
if [[ -n "${GPU_TYPE}" ]]; then
    GPU_GRES="gpu:${GPU_TYPE}:1"
else
    GPU_GRES="gpu:1"
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
    echo "  Check TRAIN_SCRIPT / PATH_CODE / BASE near the top of this script."
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is required to patch noise-sweep settings into per-task configs."
    echo "  Install jq or set NOISE_LEVELS=\"\" to disable the sweep and remove this dependency."
    exit 1
fi

# Discover all best_config_*.json files at any depth under PATH_SEARCH

# Build the find pattern: either any modeltype or the forced one
if [[ -n "${FORCE_MODELTYPE}" ]]; then
    FIND_PATTERN="best_config_${FORCE_MODELTYPE}.json"
else
    FIND_PATTERN="best_config_*.json"
fi

echo ""
echo "Searching for ${FIND_PATTERN} under:"
echo "  ${PATH_SEARCH}"
echo ""

# Collect matching files into an array
mapfile -t CONFIG_FILES < <(find "${PATH_SEARCH}" -type f -name "${FIND_PATTERN}" | sort)

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No files matching '${FIND_PATTERN}' found under ${PATH_SEARCH}"
    echo ""
    echo "Your directory layout should have files like:"
    echo "  ${PATH_SEARCH}/cnn1/seg1000/best_config_cnn1.json"
    echo "  ${PATH_SEARCH}/trans1/seg800/best_config_trans1.json"
    echo "  (or any depth)"
    exit 1
fi

echo "Found ${#CONFIG_FILES[@]} config file(s):"
for f in "${CONFIG_FILES[@]}"; do
    echo "  ${f}"
done
echo ""
echo "noise_levels             : ${NOISE_LEVELS:-disabled}"
if [[ -n "${NOISE_LEVELS}" ]]; then
    echo "use_temperature_scaling  : ${USE_TEMPERATURE_SCALING}"
    echo "ece_bins                 : ${ECE_BINS}"
    echo "noise_batch_size         : ${NOISE_BATCH_SIZE}"
fi
echo ""

# Root-level logs dir (for the merge job)
mkdir -p "${PATH_SEARCH}/logs"

# Helpers

_sbatch() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY-RUN] would submit: $*" >&2
        echo "999999"
    else
        sbatch --parsable "$@"
    fi
}

# Submit one job array per discovered config file
ALL_ARRAY_IDS=()
SUBMITTED_CONFIGS=()

for BEST_CONFIG in "${CONFIG_FILES[@]}"; do

    # The directory containing this config is where results will be saved
    CONFIG_DIR="$(dirname "${BEST_CONFIG}")"

    # Extract modeltype from the filename: best_config_<modeltype>.json
    CONFIG_BASENAME="$(basename "${BEST_CONFIG}")"
    MODELTYPE="${CONFIG_BASENAME#best_config_}"
    MODELTYPE="${MODELTYPE%.json}"

    # A short label for SLURM job names / script names: use relative path
    # from PATH_SEARCH, e.g. "cnn1_seg1000"
    REL_PATH="${CONFIG_DIR#${PATH_SEARCH}/}"          # e.g. cnn1/seg1000
    JOB_LABEL="${MODELTYPE}__$(echo "${REL_PATH}" | tr '/' '_')"  # e.g. cnn1__cnn1_seg1000

    # Single directory for everything transient per run: SLURM .out/.err
    # and the per-task seed-injected config JSON.
    LOG_DIR="${CONFIG_DIR}/logs"
    mkdir -p "${LOG_DIR}"

    # Dedicated output subdirectory for these reruns, kept separate from
    # CONFIG_DIR itself: CONFIG_DIR also holds the hp-search's own
    # all_trials.csv / results.csv, and pointing path_save directly at
    # CONFIG_DIR would merge those files with these runs' output
    # (mismatched schemas concatenated into one CSV).
    RESULTS_DIR="${CONFIG_DIR}/best_training_runs"
    mkdir -p "${RESULTS_DIR}"

    echo "=============================================================="
    echo "  Modeltype  : ${MODELTYPE}"
    echo "  Config     : ${BEST_CONFIG}"
    echo "  Output dir : ${CONFIG_DIR}"
    echo "  Job label  : ${JOB_LABEL}"
    echo "=============================================================="

    # Array script written next to the config file
    ARRAY_SCRIPT="${CONFIG_DIR}/slurm_best_training_${JOB_LABEL}.sh"

    cat > "${ARRAY_SCRIPT}" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=best_${JOB_LABEL}
#SBATCH --output=${LOG_DIR}/run_%a_%j.out
#SBATCH --error=${LOG_DIR}/run_%a_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --time=${TIME}
#SBATCH --partition=cas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gres=${GPU_GRES}
#SBATCH --mem=${MEM}
#SBATCH --array=1-${N_RUNS}%${MAX_PARALLEL}

export OMP_NUM_THREADS=${BLAS_THREADS}
export OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}

module load anaconda3
module load cuda

# Resolve the TDA env's python by absolute path; do not trust PATH.
# "module load anaconda3" can point \$(conda info --base) at a shared/site
# Anaconda install rather than your personal ~/.conda. If that happens,
# "conda activate ${CONDA_ENV}" either fails or silently activates the
# wrong (module's) base env, which has no torch, and the job would
# continue using that wrong python instead of stopping, since a failed
# 'conda activate' alone doesn't halt the script.
#
# Fix: still activate conda for env vars and library paths, but resolve
# and hard-verify the actual python binary by absolute path under
# ~/.conda, and use that absolute path for every python invocation below,
# so PATH ambiguity from the module system can no longer silently
# substitute the wrong interpreter.
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV} || echo "WARNING: 'conda activate ${CONDA_ENV}' reported an error, continuing with explicit python path below." >&2

PYTHON_BIN="\${HOME}/.conda/envs/${CONDA_ENV}/bin/python"

if [[ ! -x "\${PYTHON_BIN}" ]]; then
    echo "ERROR: expected python not found at \${PYTHON_BIN}" >&2
    echo "  \$(conda info --base) resolved to: \$(conda info --base)" >&2
    echo "  Available conda envs:" >&2
    conda env list >&2
    exit 1
fi

echo "  Using python : \${PYTHON_BIN}"
"\${PYTHON_BIN}" -c "import torch; print('  torch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())" || {
    echo "ERROR: torch import failed using \${PYTHON_BIN}, aborting before wasting compute time." >&2
    exit 1
}

echo "=============================================================="
echo "  Modeltype  : ${MODELTYPE}"
echo "  Config dir : ${CONFIG_DIR}"
echo "  Run        : \${SLURM_ARRAY_TASK_ID} / ${N_RUNS}"
echo "  Node       : \${SLURMD_NODENAME}"
echo "  GPU        : \${CUDA_VISIBLE_DEVICES:-none}"
echo "  Job        : \${SLURM_JOB_ID}  Task: \${SLURM_ARRAY_TASK_ID}"
echo "=============================================================="

# Per-task unique seed (safe against 32-bit overflow)
BASE_SEED=50   # fixed, chosen once, document this value in the paper
               # so the 10 runs are exactly reproducible by anyone re-running this script
TASK_SEED=\$(( BASE_SEED + SLURM_ARRAY_TASK_ID ))
echo "  random_seed : \${TASK_SEED}"

# Inject seed and point path_save at this config's directory. Also inject
# noise_levels / use_temperature_scaling / ece_bins / noise_batch_size,
# mirroring the TDA pipeline's own best-training submit script. Task
# config lives in the same logs/ dir as the .out/.err files (a single
# directory for all per-run artifacts).
TASK_CONFIG="${LOG_DIR}/config_job\${SLURM_JOB_ID}_task\${SLURM_ARRAY_TASK_ID}.json"

jq --argjson seed             "\${TASK_SEED}" \
   --arg     path_save        "${RESULTS_DIR}/" \
   --argjson noise_levels     '${NOISE_LEVELS_JSON}' \
   --argjson use_temp_scaling '${USE_TEMPERATURE_SCALING}' \
   --argjson ece_bins         '${ECE_BINS}' \
   --argjson noise_batch_size '${NOISE_BATCH_SIZE}' \
   '.random_seed              = \$seed              |
    .path_save                = \$path_save         |
    .noise_levels              = \$noise_levels      |
    .use_temperature_scaling  = \$use_temp_scaling  |
    .ece_bins                 = \$ece_bins           |
    .noise_batch_size         = \$noise_batch_size' \
    "${BEST_CONFIG}" > "\${TASK_CONFIG}"

if [[ ! -s "\${TASK_CONFIG}" ]]; then
    echo "ERROR: failed to write \${TASK_CONFIG}, aborting this task." >&2
    exit 1
fi

echo "  noise_levels            : \$(jq -c '.noise_levels // "disabled"'      "\${TASK_CONFIG}")"
echo "  use_temperature_scaling : \$(jq '.use_temperature_scaling'             "\${TASK_CONFIG}")"
echo "  ece_bins                : \$(jq '.ece_bins'                            "\${TASK_CONFIG}")"
echo "  noise_batch_size        : \$(jq '.noise_batch_size'                    "\${TASK_CONFIG}")"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found at run time: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

# Clean up task config on exit (success or failure)
trap 'rm -f "\${TASK_CONFIG}"' EXIT

echo "Starting: \${PYTHON_BIN} ${TRAIN_SCRIPT} --config \${TASK_CONFIG}"
time "\${PYTHON_BIN}" "${TRAIN_SCRIPT}" --config "\${TASK_CONFIG}"

echo "Run \${SLURM_ARRAY_TASK_ID} / ${N_RUNS} completed."
SLURM_EOF

    chmod +x "${ARRAY_SCRIPT}"

    ARRAY_JOB_ID=$(_sbatch "${ARRAY_SCRIPT}")
    ALL_ARRAY_IDS+=("${ARRAY_JOB_ID}")
    SUBMITTED_CONFIGS+=("${BEST_CONFIG}")
    echo "  Array job submitted : ${ARRAY_JOB_ID}"
    echo ""

done

# Guard: nothing submitted
if [[ ${#ALL_ARRAY_IDS[@]} -eq 0 ]]; then
    echo "WARNING: no jobs submitted."
    exit 0
fi

# Merge job: runs after all arrays finish. Walks every directory that
# contained a best_config file, reads its all_trials.csv / results.csv,
# and writes all_trials_merged.csv to PATH_SEARCH (the root).
DEP_IDS=$(IFS=":"; echo "${ALL_ARRAY_IDS[*]}")
MERGE_SCRIPT="${PATH_SEARCH}/slurm_merge.sh"

# Build a newline-separated list of config dirs for the Python script
CONFIG_DIRS_LIST=""
for cfg in "${SUBMITTED_CONFIGS[@]}"; do
    CONFIG_DIRS_LIST+="$(dirname "${cfg}")"$'\n'
done

cat > "${MERGE_SCRIPT}" << MERGE_EOF
#!/bin/bash
#SBATCH --job-name=merge_best_training
#SBATCH --output=${PATH_SEARCH}/logs/merge_%j.out
#SBATCH --error=${PATH_SEARCH}/logs/merge_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "Merging results from all config directories ..."
echo "Root: ${PATH_SEARCH}"

python3 - << 'PYEOF'
import glob, os, pandas as pd

root = "${PATH_SEARCH}"

# Collect all directories that contain a best_config_*.json at any depth
config_dirs = []
for dirpath, dirnames, filenames in os.walk(root):
    if any(f.startswith("best_config_") and f.endswith(".json") for f in filenames):
        config_dirs.append(dirpath)

config_dirs.sort()
print(f"Found {len(config_dirs)} config directorie(s):")
for d in config_dirs:
    print(f"  {d}")

dfs = []
sweep_dfs = []
for cdir in config_dirs:
    # Infer modeltype from the json filename
    modeltype = "unknown"
    for f in os.listdir(cdir):
        if f.startswith("best_config_") and f.endswith(".json"):
            modeltype = f[len("best_config_"):-len(".json")]
            break

    # Relative path from root for labelling, e.g. cnn1/seg1000
    rel = os.path.relpath(cdir, root)

    # Results live in best_training_runs/ next to the config, kept
    # separate from the hp-search's own all_trials.csv / results.csv in
    # cdir itself.
    results_dir = os.path.join(cdir, "best_training_runs")

    for fname in ("all_trials.csv", "results.csv"):
        csv = os.path.join(results_dir, fname)
        if os.path.exists(csv):
            try:
                df = pd.read_csv(csv)
                # Drop non-numeric summary rows. results.csv has a
                # trailing "summary" row identified via run_idx;
                # all_trials.csv has no run_idx column at all, so only
                # filter when the column is actually present (this
                # avoids an "Unalignable boolean Series" error that
                # occurs if a mismatched-index default Series is used
                # as a fallback).
                if "run_idx" in df.columns:
                    mask = pd.to_numeric(df["run_idx"], errors="coerce").notna()
                    df = df[mask]
                df["modeltype"]  = modeltype
                df["config_dir"] = rel
                df["source_file"] = csv
                dfs.append(df)
                print(f"  {rel:30s}: {len(df):4d} rows  from  {fname}")
            except Exception as e:
                print(f"  WARNING: could not read {csv}: {e}")
            break

    # robustness_all_runs.csv is not written per-task by
    # run_training_transformer_multiset.py, since that write would
    # silently overwrite itself whenever multiple SLURM array tasks
    # shared the same results_dir, leaving only the last task's single
    # run behind. Each task instead writes a uniquely named
    # robustness_sweep_run{N}.csv (N = SLURM_ARRAY_TASK_ID), so we glob
    # and combine those directly here, then write the combined result
    # back to results_dir/robustness_all_runs.csv for this config
    # directory before folding it into the root-level merge below.
    sweep_files = sorted(glob.glob(os.path.join(results_dir, "robustness_sweep_run*.csv")))
    if sweep_files:
        per_config_dfs = []
        seen_run_ids = set()
        for sf in sweep_files:
            try:
                sdf = pd.read_csv(sf)
            except Exception as e:
                print(f"  WARNING: could not read {sf}: {e}")
                continue
            if "run_idx" in sdf.columns:
                ids = set(sdf["run_idx"].unique().tolist())
                dupes = ids & seen_run_ids
                if dupes:
                    print(f"  WARNING: {sf} contains run_idx value(s) {dupes} "
                          f"already seen in another file under {results_dir}, "
                          f"check for a stale file with a colliding run_idx.")
                seen_run_ids |= ids
            per_config_dfs.append(sdf)

        if per_config_dfs:
            rdf = pd.concat(per_config_dfs, ignore_index=True)
            # Write the combined file back for this config dir, so
            # results_dir/robustness_all_runs.csv exists as before (now
            # correctly reflecting all runs instead of just the last task).
            rdf.to_csv(os.path.join(results_dir, "robustness_all_runs.csv"), index=False)
            rdf["modeltype"]  = modeltype
            rdf["config_dir"] = rel
            sweep_dfs.append(rdf)
            n_runs_found = rdf["run_idx"].nunique() if "run_idx" in rdf.columns else len(sweep_files)
            print(f"  {rel:30s}: {len(rdf):4d} robustness rows from {len(sweep_files)} "
                  f"file(s), {n_runs_found} distinct run(s)  from  robustness_sweep_run*.csv")

if not dfs:
    print("No result files found, nothing to merge.")
    raise SystemExit(0)

merged = pd.concat(dfs, ignore_index=True)
out = os.path.join(root, "all_trials_merged.csv")
merged.to_csv(out, index=False)
print(f"\nMerged {len(merged)} rows -> {out}")

# combined robustness CSV across all config dirs
if sweep_dfs:
    merged_sweep = pd.concat(sweep_dfs, ignore_index=True)
    sweep_out = os.path.join(root, "robustness_all_runs_merged.csv")
    merged_sweep.to_csv(sweep_out, index=False)
    print(f"Merged {len(merged_sweep)} robustness rows -> {sweep_out}")

metric_cols = [c for c in (
    "accuracy_test", "f1_test", "gmean_test",
    "precision_test", "recall_test", "reliability_test",
    "total_wall_time_s", "total_params",
) if c in merged.columns]

if metric_cols:
    for col in metric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    print("\nSummary by modeltype and config_dir:")
    print(merged.groupby(["modeltype", "config_dir"])[metric_cols]
          .agg(["mean", "std"])
          .round(4)
          .to_string())
PYEOF

echo "Merge complete -> ${PATH_SEARCH}/all_trials_merged.csv"
MERGE_EOF

chmod +x "${MERGE_SCRIPT}"
MERGE_JOB_ID=$(_sbatch \
    --dependency="afterany:${DEP_IDS}" \
    "${MERGE_SCRIPT}")

# Summary
echo ""
echo "=============================================================="
echo "  Submitted ${#ALL_ARRAY_IDS[@]} array job(s):"
for i in "${!ALL_ARRAY_IDS[@]}"; do
    echo "    [${ALL_ARRAY_IDS[$i]}]  ${SUBMITTED_CONFIGS[$i]}"
done
echo ""
echo "  Merge job : ${MERGE_JOB_ID}  (depends on all arrays)"
echo ""
echo "  Results saved next to each config file:"
for cfg in "${SUBMITTED_CONFIGS[@]}"; do
    echo "    $(dirname "${cfg}")/"
    echo "      logs/                        (SLURM .out/.err and per-task seed-injected configs)"
    echo "      best_training_runs/          (rerun outputs, kept separate from hp-search results)"
    echo "        results.csv                    (per-run metrics)"
    echo "        all_trials.csv                 (aggregated plus summary row)"
    echo "        robustness_sweep_run{N}.csv     per-sigma metrics for run N"
    echo "          (N = SLURM array task ID; one file per task, no collisions)"
    echo "        robustness_all_runs.csv         written by the merge job below,"
    echo "                                         combining all robustness_sweep_run*.csv"
    echo "      results.csv                  (untouched, belongs to the hp-search)"
    echo "      all_trials.csv               (untouched, belongs to the hp-search)"
done
echo ""
echo "  Merged summary:"
echo "    ${PATH_SEARCH}/all_trials_merged.csv"
echo "    ${PATH_SEARCH}/robustness_all_runs_merged.csv"
echo ""
echo "  Monitor  : squeue -u \$USER"
echo "  Cancel   : scancel ${ALL_ARRAY_IDS[*]} ${MERGE_JOB_ID}"
echo "=============================================================="
