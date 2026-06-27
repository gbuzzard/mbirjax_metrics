#!/bin/bash
# clean_outputs.sh
# ----------------
# Remove the GITIGNORED profiling run artifacts so the tree is clean (everything here regenerates by
# re-running the experiments/profiling scripts).  Conservative by default: it only touches the
# profiling output dirs, NOT the shared nightly results under tooling/scaling_tests/results/.
#
# Usage:   ./dev_scripts/clean_outputs.sh           # remove traces/ hlo/ ncu/
#          ./dev_scripts/clean_outputs.sh --results # also remove the MANUAL profiling result YAMLs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
P="$ROOT/experiments/profiling"

echo "Cleaning profiling run artifacts under $P"
rm -rf "$P/traces" "$P/hlo" "$P/ncu"
echo "  removed: traces/ hlo/ ncu/"

if [ "${1:-}" = "--results" ]; then
  # Only the profiling-specific manual YAMLs (named distinctly), never the nightly regression_*.yaml.
  R="$ROOT/tooling/scaling_tests/results"
  rm -f "$R"/compile_time_*.yaml "$R"/static_cone_back_*.yaml "$R"/gpu_inventory.yaml 2>/dev/null || true
  echo "  removed: results/{compile_time_*,static_cone_back_*,gpu_inventory}.yaml"
fi
echo "Done."
