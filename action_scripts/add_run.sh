#!/usr/bin/env bash
# Add one performance run to the tracked time series in this metrics repo, measured from a SPECIFIC
# mbirjax commit (e.g. to seed older prerelease baselines onto the timeline).
#
# Usage:
#   action_scripts/add_run.sh --local      Measure the branch currently checked out in your CWD's
#                                           mbirjax repo (must have NO uncommitted changes).
#   action_scripts/add_run.sh <ref>        Measure <ref> (a branch, tag, or commit sha) resolved in
#                                           the mbirjax repo at MBIRJAX_REPO (default: ../mbirjax).
#   action_scripts/add_run.sh              Print this help and exit.
# Add --sbatch to any of the above (on a SLURM cluster) to SUBMIT the run as a batch job on a GPU node
# (resources from run_configs.env's SLURM_* knobs) instead of running it in this session.
#
# Either way it checks out the chosen commit into a throwaway git worktree (your working tree is
# untouched) and measures it through the SAME pipeline as the nightly — the dedicated `mbirjax_regression`
# conda env with the worktree pip-installed editable (NOT your dev env, which is left untouched) — then
# writes results/<plat>/<branch>/regression_<plat>_<commit-time>_<sha8>.yaml, so the run lands on the
# dashboard timeline at its COMMIT time, comparable to the nightly runs around it.  No gate is applied
# (a backfilled run is reference data, not a pass/fail checkpoint), and a nonzero exit keeps the terminal open.
#
# Installing the worktree editable is what SELECTS the code under measurement: a modern editable install
# registers a sys.meta_path finder that takes precedence over PYTHONPATH, so pointing the engine at the
# worktree via PYTHONPATH alone would NOT override a different mbirjax already installed in the env (it
# would silently measure that one).  Hence the dedicated env + pip install -e the worktree (see lib_env.sh).

if (return 0 2>/dev/null); then _sourced=1; else _sourced=0; fi

# --sbatch (cluster): resubmit this exact invocation (minus the flag) as a SLURM batch job and exit, so
# the measurement runs on a GPU compute node instead of here.  See tooling/regression/sbatch_submit.sh.
case " $* " in *" --sbatch "*)
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; _REPO="$(cd "$_HERE/.." && pwd)"
  # shellcheck disable=SC1091
  source "$_REPO/tooling/regression/regression.env"
  # shellcheck disable=SC1091
  source "$_REPO/tooling/regression/sbatch_submit.sh"
  _ARGS=(); for _a in "$@"; do [ "$_a" = "--sbatch" ] || _ARGS+=("$_a"); done
  submit_sbatch "mbirjax-addrun" bash "$_HERE/add_run.sh" "${_ARGS[@]}"
  _rc=$?
  if [ "$_sourced" -eq 1 ]; then return "$_rc"; else exit "$_rc"; fi
  ;;
esac

(
  WT=""; SRC=""
  trap 'rc=$?
        [ -n "$WT" ] && { git -C "$SRC" worktree remove --force "$WT" 2>/dev/null; rm -rf "$(dirname "$WT")" 2>/dev/null; }
        if [ "$rc" -ne 0 ]; then
          printf "\nadd_run.sh failed (exit %s).\n" "$rc" >&2
          [ -t 0 ] && read -r -p "Press Enter to close... " _ </dev/tty || true
        fi' EXIT
  set -euo pipefail
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO="$(cd "$HERE/.." && pwd)"

  # Config + the shared env/install mechanism (CONDA_ENV=mbirjax_regression, INSTALL_EXTRAS_*,
  # HARNESS_DEPS, CONDA_PYTHON) — the same files the nightly sources, so this stays in lockstep with it.
  # The nightly sources these under `set -uo` (no -e); relax -e here too so a benign nonzero in the
  # config (now or later) can't abort the run (unset vars still trip `set -u` at use, as intended).
  set +e
  # shellcheck disable=SC1091
  source "$REPO/tooling/regression/regression.env"
  # shellcheck disable=SC1091
  source "$REPO/tooling/regression/lib_env.sh"
  set -e

  usage() {
    cat <<'EOF'
Add one performance run to this metrics repo, measured from a SPECIFIC mbirjax commit
(e.g. to seed older prerelease baselines onto the dashboard timeline).

Usage:
  action_scripts/add_run.sh --local    Measure the branch checked out in your current mbirjax repo
                                        (must have no uncommitted changes to tracked files).
  action_scripts/add_run.sh <ref>      Measure <ref> (a branch, tag, or commit sha) from the mbirjax
                                        repo at MBIRJAX_REPO (default: ../mbirjax).  <ref> also names
                                        the dashboard branch group, so prefer a branch/tag name.
  action_scripts/add_run.sh            Print this help and exit.

Add --sbatch (on a SLURM cluster) to submit the measurement as a batch job on a GPU node — resources
from run_configs.env's SLURM_* knobs — instead of running it in this session.

It checks out the commit into a throwaway worktree (your working tree is untouched) and measures it
through the same pipeline as the nightly — the dedicated mbirjax_regression conda env with the worktree
pip-installed editable (your dev env is untouched) — writing
results/<plat>/<branch>/regression_<plat>_<commit-time>_<sha8>.yaml at its COMMIT time on the timeline.
No gate is applied (a backfilled run isn't a pass/fail checkpoint).
EOF
  }
  if [ "$#" -eq 0 ]; then usage; exit 0; fi

  # ---- resolve the mbirjax repo + the commit to measure -------------------------------------------
  if [ "$1" = "--local" ]; then
    SRC="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
    [ -n "$SRC" ] && [ -d "$SRC/mbirjax" ] || { echo "--local: run this from inside an mbirjax checkout." >&2; exit 2; }
    # Uncommitted CHANGES to tracked files (untracked files like .claude/ don't affect the commit
    # we check out, so they don't block).
    [ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "--local: working tree has uncommitted changes — commit or stash first." >&2; exit 2; }
    COMMITISH="$(git -C "$SRC" rev-parse HEAD)"
    BRANCH="$(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
  else
    SRC="${MBIRJAX_REPO:-"$(cd "$REPO/.." && pwd)/mbirjax"}"
    [ -d "$SRC/.git" ] && [ -d "$SRC/mbirjax" ] || { echo "mbirjax repo not found at $SRC (set MBIRJAX_REPO)." >&2; exit 2; }
    git -C "$SRC" rev-parse --verify --quiet "$1^{commit}" >/dev/null || { echo "ref '$1' not found in $SRC." >&2; exit 2; }
    COMMITISH="$1"
    # branch label: the ref name if it is a branch, else the ref string itself
    if git -C "$SRC" show-ref --verify --quiet "refs/heads/$1"; then BRANCH="$1"; else BRANCH="$1"; fi
  fi
  SLUG="${BRANCH//\//_}"

  # ---- dedicated env (create if missing) + activate + harness deps (shared with the nightly) -------
  # Runs in mbirjax_regression, NOT your dev env — your dev env's editable install is left untouched.
  # In a `source add_run.sh` invocation this activate happens in add_run's subshell, so your current
  # shell's active env is unaffected too.
  reg_activate_env || exit $?

  # ---- platform + pip extras + output dir (shared with run_regression.sh) --------------------------
  read -r PLAT EXTRAS <<<"$(reg_plat_extras)"
  OUT="$REPO/results/$PLAT/$SLUG"; mkdir -p "$OUT"

  # ---- isolated checkout of the chosen commit, then measure --------------------------------------
  WT="$(mktemp -d)/lib"
  git -C "$SRC" worktree add --quiet --detach "$WT" "$COMMITISH"
  SHA="$(git -C "$WT" rev-parse --short=8 HEAD)"
  echo "add_run: $PLAT · branch=$BRANCH · commit=$SHA · src=$SRC · env=$CONDA_ENV"
  echo "         -> $OUT"

  # Install the worktree editable into the dedicated env — THIS selects the code under measurement
  # (re-points the editable finder at $WT).  First time pulls jax, so it can be slow.
  echo "add_run: installing library [$EXTRAS] into $CONDA_ENV (first time pulls jax — can be slow)..."
  reg_install_lib "$WT" "$EXTRAS" || { echo "add_run: pip install -e '$WT[$EXTRAS]' into $CONDA_ENV failed." >&2; exit 2; }

  # REG_GATE=0: a backfilled run is reference data, not a pass/fail checkpoint (no nonzero exit, no
  # gate).  The engine still records a day-over-day note vs the prior commit's run, if any.  lib_root=$WT
  # gives the engine the worktree for provenance (and PYTHONPATH); the editable install above is what
  # actually fixes which code imports.
  REG_LIB_ROOT="$WT" REG_OUT_DIR="$OUT" REG_RUN_TAG="$BRANCH" REG_GATE=0 \
    python "$REPO/tooling/scaling_tests/run_nightly.py"
)
_rc=$?
if [ "$_sourced" -eq 1 ]; then return "$_rc"; else exit "$_rc"; fi
