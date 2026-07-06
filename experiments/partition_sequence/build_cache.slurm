#!/bin/bash -l
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 02:00:00
#SBATCH -J mbirjax-pscache
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm-mbirjax-pscache-%j.log
#
# build_cache.slurm -- preprocess datasets into the shared depot cache (build_cache.py).
# Only needed when adding a NEW dataset / downsampling; existing caches are reused.  Pass
# specific tags via PS_TAGS (space-separated) or build every missing dataset by default:
#
#     sbatch build_cache.slurm                              # any missing dataset in the config
#     sbatch --export=ALL,PS_TAGS="z62_4x4 sic_4x4" build_cache.slurm
#
# The GPU is only used by SiC's view-alignment direct recon; other datasets are CPU preprocess.
set -uo pipefail

PREAMBLE_FILE="${PREAMBLE_FILE:-$HOME/load_conda_cuda.sh}"
if [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
fi
conda activate mbirjax

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
python build_cache.py ${PS_TAGS:-}
