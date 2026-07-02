#!/usr/bin/env bash
# lib_env.sh — the shared "prepare the dedicated regression env + install a library worktree"
# mechanism, sourced by BOTH the nightly (run_regression.sh) and the manual backfill
# (action_scripts/add_run.sh).  Using ONE mechanism means a seeded/backfilled run is produced by the
# SAME pipeline as a nightly run — same dedicated env, same editable install, same deps — so the two
# land comparably on the dashboard timeline.
#
# Why the editable install (not just PYTHONPATH) is load-bearing: a modern `pip install -e` registers a
# MetaPathFinder on sys.meta_path, which Python consults BEFORE the PYTHONPATH-based finder.  So merely
# prepending the worktree to PYTHONPATH does NOT override an editable mbirjax already installed in the
# active env — `import mbirjax` would still resolve to that env's install.  Installing the worktree
# editable into the DEDICATED env re-points the finder at the worktree, which is what actually selects
# the code under measurement.  (This is also why we never run in the user's dev env: re-pointing its
# editable install at a throwaway worktree — then deleting the worktree — would break it.)
#
# Requires (set by sourcing regression.env first): CONDA_ENV, CONDA_PYTHON, HARNESS_DEPS,
#   INSTALL_EXTRAS_cpu / INSTALL_EXTRAS_gpu.
# Provides:
#   reg_activate_env       ensure conda reachable -> auto-create + activate $CONDA_ENV -> install $HARNESS_DEPS
#   reg_plat_extras        echo "<plat> <extras>"  (gpu if nvidia-smi sees a GPU, else cpu)
#   reg_install_lib WT EX  pip install -e "WT[EX]" into the active env (re-points the editable finder)
# Logs via the caller's log() if it defines one, else a plain prefix.

_reg_log() { if declare -F log >/dev/null 2>&1; then log "$@"; else echo "[lib_env] $*"; fi; }

reg_activate_env() {
  # Ensure conda is reachable: the cluster's PREAMBLE_FILE puts it on PATH; on a Mac/plain CLI it may
  # not be, so fall back to the usual install locations (a no-op when it's already on PATH).
  if ! command -v conda >/dev/null 2>&1; then
    for s in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
      [ -f "$s/etc/profile.d/conda.sh" ] && { . "$s/etc/profile.d/conda.sh"; break; }
    done
  fi
  command -v conda >/dev/null 2>&1 || {
    _reg_log "FATAL: 'conda' not found.  On the cluster, set PREAMBLE_FILE in regression.env to a"
    _reg_log "       script that loads conda (e.g. PREAMBLE_FILE=\"\$HOME/load_conda_cuda.sh\")."
    return 2
  }
  # (Re)source conda.sh so `conda activate` is a defined shell function in THIS (sub)shell — it does
  # not survive an exec, and a subshell needs it too.
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  # Auto-create the DEDICATED env if missing, so a fresh machine self-bootstraps (deps are installed
  # per-run below + by reg_install_lib, so a bare python env suffices).  Never the user's dev env.
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    _reg_log "conda env '$CONDA_ENV' not found — creating it (one-time on this machine)."
    conda create -y -q -n "$CONDA_ENV" "python=${CONDA_PYTHON:-3.11}" \
      || { _reg_log "FATAL: could not create conda env '$CONDA_ENV'."; return 2; }
  fi
  conda activate "$CONDA_ENV" || { _reg_log "FATAL: conda activate '$CONDA_ENV' failed."; return 2; }
  # Harness's own deps (scaling_common imports matplotlib/ruamel at module level) — idempotent.
  if [ -n "${HARNESS_DEPS:-}" ]; then
    # shellcheck disable=SC2086
    pip install --quiet $HARNESS_DEPS || _reg_log "WARN: harness deps install failed (engine may not import)."
  fi
}

reg_plat_extras() {   # platform signal only — no jax import here
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "gpu ${INSTALL_EXTRAS_gpu:-}"
  else
    echo "cpu ${INSTALL_EXTRAS_cpu:-}"
  fi
}

reg_install_lib() {   # $1=worktree  $2=extras ; caller logs/redirects/handles the exit code
  pip install -e "$1[$2]"
}

# Dependency canary: force jax/jaxlib to the latest on PyPI (NOT the other deps).  No exclusion is passed
# — the per-branch `reg_install_lib` that follows re-resolves against the branch's pyproject and pulls an
# excluded version (e.g. 0.10.2) back down, so the `jax!=…` list stays single-sourced there.  The cuda
# extra is derived from the run's extras so the matching CUDA plugin wheels come along on GPU.
reg_upgrade_jax() {   # $1=extras (e.g. "cuda12,test") ; caller logs/redirects/handles the exit code
  local jax_pkg="jax"
  case ",$1," in
    *,cuda12,*) jax_pkg="jax[cuda12]" ;;
    *,cuda13,*) jax_pkg="jax[cuda13]" ;;
  esac
  pip install -U "$jax_pkg" jaxlib
}

# Dependency canary, periodic FULL refresh: eager-upgrade ALL of the worktree's deps to the latest the
# pyproject allows (numpy/scipy/etc. too, not just jax).  Uses the editable install so the `jax!=…`
# exclusion is honored natively.  Caller logs/redirects/handles the exit code.
reg_upgrade_all() {   # $1=worktree  $2=extras
  pip install -e "$1[$2]" --upgrade --upgrade-strategy eager
}
