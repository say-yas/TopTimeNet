#!/bin/bash
# submit_raw_signal_robustness.sh
# Submits a 3-stage SLURM pipeline (train -> sweep -> merge) for both
# best_config.json and best_config_small.json found inside each seg*/
# directory, mirroring submit_best_training.sh's structure.
#
# Output structure per seg dir:
#     <seg_dir>/raw_robustness/        runs using best_config.json
#     <seg_dir>/raw_robustness_small/  runs using best_config_small.json
#
# Each of those contains:
#     logs/                                 SLURM .out/.err, all 3 stages
#     artifacts/run_<i>.pt                  stage 1: trained weights + test split
#     sweep_results/run<i>_sigma<s>.csv     stage 2: one row per (run, sigma)
#     raw_vs_feature_robustness_merged.csv  stage 3: all runs x all sigmas
#     raw_vs_feature_summary_merged.csv     stage 3: mean +/- std across runs
#     raw_vs_feature_robustness_merged.png  stage 3
#
# Why 3 stages, and why sigma is its own array dimension
# --------------------------------------------------------
# raw_signal_robustness_sweep.py recomputes persistent homology from
# scratch for every test sample; this happens once per (run, sigma) pair
# and is the real bottleneck (ripser), unlike the feature-level-only
# sweep in submit_best_training.sh. So instead of one array over runs
# (each run then looping over every noise level sequentially), this
# script uses:
#   Stage 1 (train) : array of N_RUNS tasks, trains one model per task,
#                      saves weights and test-split indices to a
#                      persistent artifact (--mode train). Nothing is
#                      swept yet.
#   Stage 2 (sweep) : array of N_RUNS * N_SIGMA tasks, flattened so every
#                      noise level is its own parallel unit of work, not
#                      just every run (--mode sweep). Depends on stage 1.
#   Stage 3 (merge) : combines every stage-2 CSV into the same combined
#                      summary a single-process run would have produced.
#                      Depends on stage 2.
#
# No jq or config-patching is needed anywhere in this pipeline (unlike
# submit_best_training.sh): raw_signal_robustness_sweep.py takes its run
# parameters (--mode, --run_idx, --sigma, --base_seed, --artifact,
# --sweep_out) as CLI arguments, so best_config*.json is passed through
# to every stage unmodified: every hyperparameter (optimizer, muon_lr,
# label_smoothing, norm_type, etc.) is read directly from the exact file
# you already validated.
set -euo pipefail

# PARAMETERS
N_RUNS=30                  # independent training runs per config
BLAS_THREADS=4
CPUS_PER_TASK=$(( BLAS_THREADS + 2 ))
MEM_TRAIN="24G"
MEM_SWEEP="16G"
TIME_TRAIN="04:00:00"     # training alone (cached features, no TDA recompute) is fast
TIME_SWEEP="06:00:00"     # one (run, sigma) pair: full ripser recompute over the test set
MAX_PARALLEL_TRAIN=5
MAX_PARALLEL_SWEEP=20     # sweep tasks are more numerous and independent, raise this
                          # as high as your cluster/fair-share allows
CONDA_ENV="TDA"

# Passed straight through as CLI args to raw_signal_robustness_sweep.py.
NOISE_LEVELS_CSV="0.0,0.025,0.05,0.075,0.1,0.2,0.5,1.0"
BATCH_SIZE="32"            # batch size for TDA feature recomputation (keep small, ripser)

# Verify this against whatever actually produced your reported main
# results before relying on it: best_config*.json's own noise_aug_sigma
# may not match the value your training run actually used.
NOISE_AUG_SIGMA="0.05"

# Same caution as NOISE_AUG_SIGMA above: best_config*.json's own
# use_temperature_scaling may not match what actually trained the
# reported main results. Without the correct value here, ECE/confidence
# values are computed with temperature=1.0 (no scaling) instead of the
# true fitted value; accuracy is unaffected either way, since argmax is
# temperature-invariant.
USE_TEMPERATURE_SCALING="true"

# MODE
MODE="all"   # "single" or "all"

# Paths
PATH_OUTPUT="/home/sharareh.sayyad/Computtional_topology/Computational_topology/ml_classification/output_tda_stat_summary"
BASE_SEARCH_DIR="${PATH_OUTPUT}/hparam_search_results"
PATH_CODE="/home/sharareh.sayyad/Computtional_topology/Computational_topology/ml_classification/TDA_stat_summary_time_series_classification"
CONDA_ENV_PATH_NOTE="see CONDA_ENV above"
SCRIPT="${PATH_CODE}/raw_signal_robustness_sweep.py"

# Used only when MODE=single
SEARCH_DIR="${BASE_SEARCH_DIR}/seg500"

# Auto-detect: if invoked from inside a seg*/ directory, switch to single
CURRENT_DIR="$(pwd)"
if [[ "$(basename "${CURRENT_DIR}")" == seg* ]]; then
    echo "Auto-detected seg*/ directory, switching to MODE=single"
    MODE="single"
    SEARCH_DIR="${CURRENT_DIR}"
fi

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: raw_signal_robustness_sweep.py not found: ${SCRIPT}"
    echo "  Check SCRIPT / PATH_CODE near the top of this script."
    exit 1
fi

# Turn NOISE_LEVELS_CSV into a bash array + count, for the flattened array size
IFS=',' read -r -a NOISE_LEVELS_ARR <<< "${NOISE_LEVELS_CSV}"
N_SIGMA=${#NOISE_LEVELS_ARR[@]}
N_SWEEP_TASKS=$(( N_RUNS * N_SIGMA ))
NOISE_LEVELS_BASH_ARRAY="(${NOISE_LEVELS_ARR[*]})"

# Shared python-resolution and import-check block, inlined into stage 1
# and stage 2's heredocs. A module-loaded conda can point
# $(conda info --base) at the wrong (shared) Anaconda install, letting
# `conda activate` silently activate an env without torch instead of
# failing loudly, so the actual python binary is resolved by absolute
# path and hard-verified before any real work starts.
PYTHON_RESOLVE_BLOCK='
source $(conda info --base)/etc/profile.d/conda.sh
conda activate CONDA_ENV_PLACEHOLDER || echo "WARNING: conda activate reported an error, continuing with explicit python path below." >&2
PYTHON_BIN="${HOME}/.conda/envs/CONDA_ENV_PLACEHOLDER/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: expected python not found at ${PYTHON_BIN}" >&2
    echo "  \$(conda info --base) resolved to: $(conda info --base)" >&2
    conda env list >&2
    exit 1
fi
echo "  Using python : ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch, ripser; print(\"  torch:\", torch.__version__, \"| CUDA:\", torch.cuda.is_available())" || {
    echo "ERROR: torch/ripser import failed using ${PYTHON_BIN}, aborting." >&2
    exit 1
}
'
PYTHON_RESOLVE_BLOCK="${PYTHON_RESOLVE_BLOCK//CONDA_ENV_PLACEHOLDER/${CONDA_ENV}}"


# submit_one <best_config_path> <results_dir> <job_label>
# Submits the full 3-stage pipeline for one config, writing everything
# under results_dir (which lives directly inside the seg*/ dir, exactly
# like submit_best_training.sh's best_training / best_training_small).
submit_one() {
    local best_config="$1"
    local results_dir="$2"
    local job_label="$3"

    if [ ! -f "${best_config}" ]; then
        echo "    SKIP: config not found: ${best_config}"
        return 1
    fi

    local log_dir="${results_dir}/logs"
    local artifact_dir="${results_dir}/artifacts"
    local sweep_dir="${results_dir}/sweep_results"
    mkdir -p "${log_dir}" "${artifact_dir}" "${sweep_dir}"

    echo "    Config     : ${best_config}"
    echo "    Output dir : ${results_dir}"

    # Stage 1: train, array of N_RUNS tasks
    local train_script="${results_dir}/slurm_${job_label}_train.sh"

    cat > "${train_script}" << SLURM
#!/bin/bash
#SBATCH --job-name=rawrob_train_${job_label}
#SBATCH --output=${log_dir}/train_%a_%j.out
#SBATCH --error=${log_dir}/train_%a_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --partition=cas
#SBATCH --time=${TIME_TRAIN}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_TRAIN}
#SBATCH --array=1-${N_RUNS}%${MAX_PARALLEL_TRAIN}

export OMP_NUM_THREADS=${BLAS_THREADS}
export OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}

module load anaconda3

${PYTHON_RESOLVE_BLOCK}

echo "job=${job_label} [train] | Run \${SLURM_ARRAY_TASK_ID}/${N_RUNS} | Node \${SLURMD_NODENAME}"

TASK_SEED=\$(( ( SLURM_JOB_ID % 100000 ) * 1000 ))
ARTIFACT="${artifact_dir}/run_\${SLURM_ARRAY_TASK_ID}.pt"

echo "  base_seed : \${TASK_SEED}"
echo "  artifact  : \${ARTIFACT}"

cd ${PATH_CODE}
time "\${PYTHON_BIN}" "${SCRIPT}" \
    --config "${best_config}" \
    --mode train \
    --run_idx "\${SLURM_ARRAY_TASK_ID}" \
    --base_seed "\${TASK_SEED}" \
    --noise_aug_sigma "${NOISE_AUG_SIGMA}" \
    --use_temperature_scaling "${USE_TEMPERATURE_SCALING}" \
    --artifact "\${ARTIFACT}"

# Explicit failure checking. Without this, a python crash here is
# silently swallowed: the script's last command would otherwise be the
# closing echo, which always exits 0, so SLURM would mark this array
# task COMPLETED even though no artifact was written, which is exactly
# what makes stage 2 fail downstream with a confusing "artifact not
# found" error instead of a clear failure here.
TRAIN_EXIT=\$?
if [[ \${TRAIN_EXIT} -ne 0 ]]; then
    echo "ERROR: --mode train exited \${TRAIN_EXIT} for run \${SLURM_ARRAY_TASK_ID}, see this task's .err log above." >&2
    exit \${TRAIN_EXIT}
fi
if [[ ! -f "\${ARTIFACT}" ]]; then
    echo "ERROR: --mode train reported success but \${ARTIFACT} was not created, treating as a failure." >&2
    exit 1
fi

echo "Train task \${SLURM_ARRAY_TASK_ID}/${N_RUNS} done."
SLURM

    chmod +x "${train_script}"
    local train_job_id
    train_job_id=$(sbatch --parsable "${train_script}")
    echo "    Stage 1 (train, ${N_RUNS} tasks) submitted : ${train_job_id}"

    # Stage 2: sweep, flattened array of N_RUNS * N_SIGMA tasks
    # Array index i (1-based) maps to:
    #   run_idx = ((i-1) / N_SIGMA) + 1
    #   sigma   = NOISE_LEVELS_ARR[(i-1) % N_SIGMA]
    local sweep_script="${results_dir}/slurm_${job_label}_sweep.sh"

    cat > "${sweep_script}" << SLURM
#!/bin/bash
#SBATCH --job-name=rawrob_sweep_${job_label}
#SBATCH --output=${log_dir}/sweep_%a_%j.out
#SBATCH --error=${log_dir}/sweep_%a_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --partition=cas
#SBATCH --time=${TIME_SWEEP}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_SWEEP}
#SBATCH --array=1-${N_SWEEP_TASKS}%${MAX_PARALLEL_SWEEP}
#SBATCH --dependency=afterany:${train_job_id}

export OMP_NUM_THREADS=${BLAS_THREADS}
export OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export MKL_NUM_THREADS=${BLAS_THREADS}
export NUMEXPR_NUM_THREADS=${BLAS_THREADS}

module load anaconda3

${PYTHON_RESOLVE_BLOCK}

echo "job=${job_label} [sweep] | Task \${SLURM_ARRAY_TASK_ID}/${N_SWEEP_TASKS} | Node \${SLURMD_NODENAME}"

NOISE_LEVELS=${NOISE_LEVELS_BASH_ARRAY}
N_SIGMA=${N_SIGMA}

IDX0=\$(( SLURM_ARRAY_TASK_ID - 1 ))
RUN_IDX=\$(( IDX0 / N_SIGMA + 1 ))
SIGMA_IDX=\$(( IDX0 % N_SIGMA ))
SIGMA="\${NOISE_LEVELS[\${SIGMA_IDX}]}"

ARTIFACT="${artifact_dir}/run_\${RUN_IDX}.pt"
SWEEP_OUT="${sweep_dir}/run\${RUN_IDX}_sigma\${SIGMA}.csv"

if [[ ! -f "\${ARTIFACT}" ]]; then
    # Stage 2 launches even if some stage-1 runs failed (dependency type
    # afterany), so that other runs' sweeps still proceed instead of the
    # whole pipeline blocking on one bad run. A missing artifact here
    # means run \${RUN_IDX}'s training genuinely failed; check
    # ${log_dir}/train_\${RUN_IDX}_*.err for the real cause. This task
    # exits 0 (a clean skip, not a failure) so it doesn't clutter
    # squeue/sacct with spurious FAILED entries for something stage 1
    # already reported.
    echo "SKIP: run artifact not found: \${ARTIFACT}" >&2
    echo "SKIP: this means training for run \${RUN_IDX} did not complete successfully." >&2
    echo "SKIP: check ${log_dir}/train_\${RUN_IDX}_*.err for the actual error." >&2
    exit 0
fi

echo "  run_idx : \${RUN_IDX}   sigma : \${SIGMA}"

cd ${PATH_CODE}
time "\${PYTHON_BIN}" "${SCRIPT}" \
    --config "${best_config}" \
    --mode sweep \
    --artifact "\${ARTIFACT}" \
    --run_idx "\${RUN_IDX}" \
    --sigma "\${SIGMA}" \
    --sweep_out "\${SWEEP_OUT}" \
    --batch_size ${BATCH_SIZE}

SWEEP_EXIT=\$?
if [[ \${SWEEP_EXIT} -ne 0 ]]; then
    echo "ERROR: --mode sweep exited \${SWEEP_EXIT} for run \${RUN_IDX} sigma \${SIGMA}, see this task's .err log above." >&2
    exit \${SWEEP_EXIT}
fi
if [[ ! -f "\${SWEEP_OUT}" ]]; then
    echo "ERROR: --mode sweep reported success but \${SWEEP_OUT} was not created, treating as a failure." >&2
    exit 1
fi

echo "Sweep task \${SLURM_ARRAY_TASK_ID}/${N_SWEEP_TASKS} done (run \${RUN_IDX}, sigma \${SIGMA})."
SLURM

    chmod +x "${sweep_script}"
    local sweep_job_id
    sweep_job_id=$(sbatch --parsable "${sweep_script}")
    echo "    Stage 2 (sweep, ${N_SWEEP_TASKS} tasks) submitted : ${sweep_job_id}  (after stage 1)"

    # Stage 3: merge, combine every stage-2 CSV for this config
    local merge_script="${results_dir}/slurm_${job_label}_merge.sh"

    cat > "${merge_script}" << MERGE
#!/bin/bash
#SBATCH --job-name=rawrob_merge_${job_label}
#SBATCH --output=${log_dir}/merge_%j.out
#SBATCH --error=${log_dir}/merge_%j.err
#SBATCH --partition=cas
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --dependency=afterany:${sweep_job_id}

module load anaconda3

${PYTHON_RESOLVE_BLOCK}

"\${PYTHON_BIN}" - << 'PY'
import glob, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sweep_dir = "${sweep_dir}"
results_dir = "${results_dir}"

files = sorted(glob.glob(os.path.join(sweep_dir, "run*_sigma*.csv")))
print(f"Found {len(files)} per-(run, sigma) CSV(s) in {sweep_dir}")

if not files:
    print("Nothing to merge.")
    raise SystemExit(0)

dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f))
    except Exception as exc:
        print(f"  WARNING: could not read {f}: {exc}")

combined = pd.concat(dfs, ignore_index=True)
combined_path = os.path.join(results_dir, "raw_vs_feature_robustness_merged.csv")
combined.to_csv(combined_path, index=False)
print(f"Merged {len(combined)} rows from {len(dfs)} file(s) -> {combined_path}")

summary = combined.groupby(["level", "sigma"])["accuracy"].agg(["mean", "std"]).reset_index()
summary_path = os.path.join(results_dir, "raw_vs_feature_summary_merged.csv")
summary.to_csv(summary_path, index=False)
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
for level, color in [("raw_signal", "#D85A30"), ("feature", "#378ADD")]:
    sub = summary[summary["level"] == level].sort_values("sigma")
    if sub.empty:
        continue
    ax.plot(sub["sigma"], sub["mean"], marker="o", lw=2, color=color, label=level)
    ax.fill_between(sub["sigma"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                     color=color, alpha=0.2)
ax.set_xlabel("Gaussian noise sigma")
ax.set_ylabel("Test accuracy")
n_runs_found = combined["run_idx"].nunique()
ax.set_title(f"Raw-signal vs. feature-level robustness ({n_runs_found} run(s))")
ax.legend()
fig.tight_layout()
fig_path = os.path.join(results_dir, "raw_vs_feature_robustness_merged.png")
fig.savefig(fig_path, dpi=150)
print(f"Comparison plot -> {fig_path}")
PY
MERGE_EXIT=\$?
if [[ \${MERGE_EXIT} -ne 0 ]]; then
    echo "ERROR: merge step exited \${MERGE_EXIT}." >&2
    exit \${MERGE_EXIT}
fi
MERGE

    chmod +x "${merge_script}"
    local merge_job_id
    merge_job_id=$(sbatch --parsable --dependency="afterany:${sweep_job_id}" "${merge_script}")
    echo "    Stage 3 (merge) submitted : ${merge_job_id}  (after stage 2)"
    echo "    Final summary will be in: ${results_dir}/raw_vs_feature_summary_merged.csv"
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
    local dir_best="${seg_dir}/raw_robustness"
    echo "  [large / best_config.json]"
    submit_one "${cfg_best}" "${dir_best}" "${seg_name}"

    local cfg_small="${seg_dir}/best_config_small.json"
    local dir_small="${seg_dir}/raw_robustness_small"
    echo "  [small / best_config_small.json]"
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
    echo "  Submitting raw-signal robustness pipelines for large + small ..."
    echo "  N_RUNS                        : ${N_RUNS}"
    echo "  noise_levels (N_SIGMA=${N_SIGMA})  : ${NOISE_LEVELS_CSV}"
    echo "  Stage-2 array size per config  : ${N_SWEEP_TASKS}  (N_RUNS x N_SIGMA)"
    echo "  batch_size (TDA recompute)    : ${BATCH_SIZE}"
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
        echo "      raw_robustness/              best_config.json"
        echo "        logs/                             (SLURM .out/.err, all 3 stages)"
        echo "        artifacts/run_<i>.pt              (stage 1: trained weights + test split)"
        echo "        sweep_results/run<i>_sigma<s>.csv (stage 2: one row per (run, sigma))"
        echo "        raw_vs_feature_robustness_merged.csv  (stage 3: all runs x all sigmas)"
        echo "        raw_vs_feature_summary_merged.csv     (stage 3: mean +/- std)"
        echo "      raw_robustness_small/        best_config_small.json (same layout)"
    done
    echo "=============================================================="

else
    echo "ERROR: unknown MODE='${MODE}'.  Set MODE=single or MODE=all."
    exit 1
fi
