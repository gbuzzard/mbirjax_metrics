#!/bin/bash
# run_gpu_all.sh — reproduce the full GPU profiling phase end-to-end (the record of how the GPU
# results are produced).  Runnable AND a recipe: read it top-to-bottom to see every command.
#
# WHERE: on Gautschi, from the repo root, in the PRODUCTION mbirjax env (the one you ship):
#     conda activate mbirjax
#     module load cuda/12.9.0                 # puts nsys/ncu on PATH (cf. mbirjax/dev_scripts/clean_install_all.sh)
#     ./experiments/profiling/run_gpu_all.sh
#   Optional, once: ./dev_scripts/install_tensorboard.sh   # adds the TensorBoard trace UI (orthogonal; won't move jax)
#
# PRINCIPLE (load-bearing): profiling must measure the PRODUCTION environment, INCLUDING its jax pin.
# This script ABORTS on the EXCLUDED jax (0.10.2, a regression that 4x-slowed the cone band kernel) and
# WARNS if jax isn't the expected pin, so we never silently re-measure the wrong version.  Edit
# EXPECTED_JAX only when production's pin legitimately changes.
#
# OUTPUTS (all gitignored; re-running overwrites):
#   tooling/scaling_tests/results/{gpu_inventory,static_cone_back_gpu,compile_time_gpu}.yaml
#   experiments/profiling/traces/<ts>_cone_{back,forward}_256x256x256_n*/   (Perfetto + printed self-time)
#   experiments/profiling/hlo/cone_{pixel,band}_*.txt
#   experiments/profiling/ncu/{back_pixel_256,back_band_256}.csv
#
# Each step is independent: a failure (e.g. ncu permission) is reported but does NOT stop the rest.

set -uo pipefail
EXPECTED_JAX="0.10.1"     # production pin (warn if different)
EXCLUDED_JAX="0.10.2"     # known-bad regression (hard abort)

cd "$(dirname "$0")/../.." || { echo "cannot cd to repo root"; exit 1; }
P="experiments/profiling"

# ── env + jax-pin guard ───────────────────────────────────────────────────────
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" = "base" ]; then
  echo "ERROR: activate the production env first (e.g. 'conda activate mbirjax')."; exit 1
fi
ACTUAL=$(python -c 'import jax, jaxlib; print(jax.__version__, jaxlib.__version__)')
ACTUAL_JAX=${ACTUAL%% *}
echo "=================================================================="
echo "  env: $CONDA_DEFAULT_ENV   jax/jaxlib: $ACTUAL   (expected jax $EXPECTED_JAX)"
echo "=================================================================="
if [ "$ACTUAL_JAX" = "$EXCLUDED_JAX" ]; then
  echo "ERROR: jax $ACTUAL_JAX is EXCLUDED (regression — 4x-slowed the cone band kernel). Aborting."; exit 1
fi
if [ "$ACTUAL_JAX" != "$EXPECTED_JAX" ]; then
  echo "WARNING: jax $ACTUAL_JAX != expected $EXPECTED_JAX. Confirm production moved, then update EXPECTED_JAX."
fi
HAVE_NCU=1; command -v ncu >/dev/null 2>&1 || { HAVE_NCU=0; echo "WARNING: ncu not on PATH (try 'module load cuda/12.9.0') — ncu steps will be SKIPPED."; }

run() { echo; echo "############### $1 ###############"; shift; "$@" || echo "  >>> step FAILED (continuing): $*"; }

# ── JAX-only experiments (no NVIDIA tools needed) ─────────────────────────────
run "STEP 0: GPU inventory (devices, versions, nsys/ncu, topology, idle temps)"      python -u $P/gpu_inventory.py
run "STEP 1: static cone back kernels — pixel vs band (the PLATFORM INVERSION)"       python -u $P/static_cone_back_kernels.py
run "STEP 2: compile-time attribution (trace/lower/compile, GPU autotuning)"          python -u $P/compile_time_projectors.py
run "STEP 3: trace cone BACK (n=1 short-circuit pixel; n=2 banded reduce-scatter)"    python -u $P/trace_back_projection.py
run "STEP 4: trace cone FORWARD (GPU1) — gap from the first GPU pass, included here"  python -u $P/trace_forward_projection.py
run "STEP 5: region breakdown (profile_measure — trace+HLO -> self-time per named_scope region)" python -u $P/profile_measure.py

# ── Nsight Compute roofline (needs ncu on PATH + GPU perf-counter permission) ──
mkdir -p $P/ncu
if [ "$HAVE_NCU" = "1" ]; then
  echo; echo "############### STEP 6: ncu roofline — n=1 PIXEL kernel (loop_add / dynamic_update_slice) ###############"
  ncu --profile-from-start off --set basic --kernel-name "regex:add_fusion|dynamic_update_slice" \
      --launch-count 6 --target-processes all \
      --csv --log-file $P/ncu/back_pixel_256.csv \
      python $P/ncu_back_projection.py || echo "  >>> ncu PIXEL failed (ERR_NVGPUCTRPERM = counters locked; see README)"

  echo; echo "############### STEP 7: ncu roofline — BAND kernel transpose (input_transpose_fusion) ###############"
  ncu --profile-from-start off --set basic --kernel-name "regex:transpose_fusion|input_transpose" \
      --launch-count 6 --target-processes all \
      --csv --log-file $P/ncu/back_band_256.csv \
      python $P/ncu_band_kernel.py || echo "  >>> ncu BAND failed (ERR_NVGPUCTRPERM = counters locked; see README)"
else
  echo; echo "############### STEPS 6-7 (ncu) SKIPPED — ncu not on PATH ###############"
fi

echo; echo "############### DONE ###############"
echo "  results: tooling/scaling_tests/results/{gpu_inventory,static_cone_back_gpu,compile_time_gpu}.yaml"
echo "           $P/results/profile_gpu_*.yaml   (the per-named-region breakdown)"
echo "  traces : $P/traces/    hlo: $P/hlo/    ncu: $P/ncu/"
echo
echo "  Optional / open experiments (edit a constant at the top of the script, then re-run that one):"
echo "    * parallel beam on GPU : set GEOMETRY=\"parallel\" in trace_back_projection.py / trace_forward_projection.py"
echo "    * 512^3 scale-up       : set SIZE=(512,512,512) in the trace / static scripts"
echo "    * ncu --set full       : swap --set basic -> --set full on STEP 6 to name the exact saturated pipe"
