#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q standby
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 04:00:00
#SBATCH -J mbirjax-nightly
#SBATCH --mail-user=buzzard@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH -o slurm-mbirjax-nightly-%j.log
#
# nightly_regression.slurm — run ONE nightly regression pass as a batch job on Gautschi.
#
# This is for a MANUAL batch test of the cluster pipeline:
#     cd <your standing mbirjax_metrics checkout>
#     sbatch tooling/regression/nightly_regression.slurm
# (In an interactive GPU session you can instead just run run_regression.sh directly — it sources
# the same preamble.)  The SCHEDULED nightly does NOT use this file: enable_nightly.sh installs a
# scrontab entry that runs run_regression.sh with the SLURM_* options from run_configs.env, so keep
# the #SBATCH directives above in sync with those knobs (or just edit run_configs.env and re-run
# enable_nightly.sh for the real schedule).
#
# It forwards to run_regression.sh, which sources the node preamble (module load conda + proxy via
# PREAMBLE_FILE in regression.env), fresh-clones metrics + mbirjax, and measures every tracked
# branch whose tip moved.  The log lands next to where you submitted from (slurm-...-%j.log).
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec bash "$REPO/tooling/regression/run_regression.sh"
