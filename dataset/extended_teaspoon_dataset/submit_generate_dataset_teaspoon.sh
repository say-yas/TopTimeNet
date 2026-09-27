#!/bin/bash
# ── PARAMETERS ────────────────────────────────────────────────────────────────
CPUS_PER_TASK=10
PATH_DATA="/home/sharareh.sayyad/Computtional_topology/Computational_topology/dataset/extended_teaspoon_dataset"
PATH_CODE="/home/sharareh.sayyad/Computtional_topology/Computational_topology/src/dataset_generation"
SCRIPT_NAME="generate_dataset_extended_teaspoon.py"
CONDA_ENV="TDA"

# ── CREATE OUTPUT DIRECTORY ───────────────────────────────────────────────────
mkdir -p "${PATH_DATA}"

# ── WRITE AND SUBMIT SLURM SCRIPT ─────────────────────────────────────────────
cat > "${PATH_DATA}/slurm_generate_datasets.sh" << EOF
#!/bin/bash
#SBATCH --job-name=gen_teaspoon
#SBATCH --output=${PATH_DATA}/%x_%j.out
#SBATCH --error=${PATH_DATA}/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sharareh.sayyad@wsu.edu
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}

module load anaconda3
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "Job \$SLURM_JOB_ID on \$SLURMD_NODENAME | CPUs: \$SLURM_CPUS_PER_TASK"

cd ${PATH_CODE}
time python ${SCRIPT_NAME} ${PATH_DATA}/

H5_FILE="${PATH_DATA}/all_extended_teaspoon_datasets.h5"
[ -f "\${H5_FILE}" ] \
    && echo "SUCCESS: \${H5_FILE}  (\$(du -sh \${H5_FILE} | cut -f1))" \
    || echo "WARNING: HDF5 not found — check error log."
EOF

chmod +x "${PATH_DATA}/slurm_generate_datasets.sh"

echo "Created : ${PATH_DATA}/slurm_generate_datasets.sh"
echo "Run     : sbatch ${PATH_DATA}/slurm_generate_datasets.sh"