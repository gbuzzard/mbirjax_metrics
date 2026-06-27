#!/bin/bash
# install_tensorboard.sh
# ----------------------
# Install ORTHOGONAL, analysis-only profiling extras into the CURRENTLY ACTIVE conda env
# (e.g. the shared `mbirjax` env you profile in).  These tools — TensorBoard's profile UI and
# friends — are imported by the `tensorboard` BINARY, never during a traced run, so they do NOT
# change what is measured.  Installing them here (in mbirjax_metrics) instead of in the mbirjax
# repo's installer keeps the library's env definition untouched while letting us use them here.
#
# This is the home for any other ORTHOGONAL extra (pprof, perfetto trace_processor, ...).  Guiding
# principle: profiling MEASURES THE PRODUCTION ENVIRONMENT, including whatever pins it uses.  So the
# hard rule here is that an extra must not move jax/jaxlib (or anything else load-bearing) — if it
# does, this env is no longer the production env we mean to measure.  The script snapshots jax/jaxlib
# and aborts loudly if an install moved them (so you can roll the extra back).  Idempotent.
#
# Usage:
#   conda activate mbirjax      # the env you profile in
#   ./dev_scripts/install_tensorboard.sh
set -euo pipefail

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
  echo "ERROR: activate the env you profile in first (e.g. 'conda activate mbirjax'); refusing to install into '${CONDA_DEFAULT_ENV:-<none>}'."
  exit 1
fi
echo "Installing orthogonal profiling extras into env: $CONDA_DEFAULT_ENV"

# Snapshot the load-bearing versions BEFORE, so we can prove they didn't move.
before=$(python -c 'import jax, jaxlib; print(jax.__version__, jaxlib.__version__)')

# --- orthogonal, analysis-only packages (must NOT depend on / upgrade jax) ---
pip install -U tensorboard tensorboard-plugin-profile
# add more orthogonal extras here as needed, e.g.:
#   pip install -U <analysis-only-pkg>

after=$(python -c 'import jax, jaxlib; print(jax.__version__, jaxlib.__version__)')
echo "jax/jaxlib before: $before"
echo "jax/jaxlib after : $after"
if [ "$before" != "$after" ]; then
  echo ""
  echo "  !!! WARNING: an extra moved jax/jaxlib ($before -> $after) — orthogonality VIOLATED."
  echo "  !!! Profiling now measures a different jax than production.  Pin it back, e.g.:"
  echo "  !!!     pip install 'jax==${before%% *}' 'jaxlib==${before##* }'"
  exit 2
fi
echo "OK — jax/jaxlib unchanged; measurements stay comparable to production."
