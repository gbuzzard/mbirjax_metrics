#!/bin/bash -l
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 08:00:00
#SBATCH -J mbirjax-pseq
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm-mbirjax-pseq-%j.log
#
# run_study.slurm -- run one partition-sequence experiment (run_study.py) as a batch job
# on Gautschi.  The run configuration lives in config.yaml; select the experiment with
# PS_EXPERIMENT (default below).  Submit from this directory:
#
#     cd <...>/mbirjax_metrics/experiments/partition_sequence
#     python build_cache.py                 # once; reuses the shared depot cache if present
#     sbatch --export=ALL,PS_EXPERIMENT=tail_4x4 run_study.slurm
#
# ONE GPU on purpose: the subsampled sinograms fit a single device, and a fixed device count
# keeps per-iteration wall times comparable across every run in the study.  Bump -t for big
# experiments.  The node preamble is the same file the nightly regression uses
# ($HOME/load_conda_cuda.sh: module load conda/cuda + proxy exports).
set -uo pipefail

PREAMBLE_FILE="${PREAMBLE_FILE:-$HOME/load_conda_cuda.sh}"
if [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
fi
conda activate mbirjax

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
export PS_EXPERIMENT="${PS_EXPERIMENT:-tail_4x4}"
echo "=== run_study experiment: $PS_EXPERIMENT ==="
python run_study.py
