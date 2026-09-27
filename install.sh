#!/bin/bash
set -e

# remove this if you don't want to use conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate TDA

# Verify correct Python version
echo "Using Python: $(python --version)"
echo "Using pip: $(pip --version)"

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Clearing stale bytecode caches..."
find . -name "__pycache__" -exec rm -rf {} +

echo "Installing package..."
pip install -e .


echo "Done!"