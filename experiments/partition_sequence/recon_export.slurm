#!/bin/bash -l
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 08:00:00
#SBATCH -J mbirjax-export
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm-mbirjax-export-%j.log
#
# recon_export.slurm -- run recon_and_save.py (recon volumes + logs -> slice_viewer h5).
# Config lives in config.yaml (recon_exports block); select with PS_EXPORT, subset datasets
# with PS_DATASETS.  Runs on whatever branch the mbirjax editable install is on.
#
#     sbatch --export=ALL,PS_EXPORT=scale1k_snapshots,PS_DATASETS=<tag> recon_export.slurm
#     sbatch --export=ALL,PS_TRIAL=1,PS_OUT_DIR=<scratch> -t 00:20:00 recon_export.slurm
set -uo pipefail

PREAMBLE_FILE="${PREAMBLE_FILE:-$HOME/load_conda_cuda.sh}"
if [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
fi
conda activate mbirjax

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
python recon_and_save.py
