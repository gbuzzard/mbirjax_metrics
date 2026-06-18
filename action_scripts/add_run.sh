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
#
# Either way it checks out the chosen commit into a throwaway git worktree (your working tree is
# untouched), runs the engine against it, and writes results/<plat>/<branch>/regression_<plat>_<commit
# time>_<sha8>.yaml — so the run lands on the dashboard timeline at its COMMIT time.  No golden gate
# is applied (a baseline isn't a pass/fail checkpoint), and a nonzero exit keeps the terminal open.

if (return 0 2>/dev/null); then _sourced=1; else _sourced=0; fi

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

It checks out the commit into a throwaway worktree (your working tree is untouched), measures it,
and writes results/<plat>/<branch>/regression_<plat>_<commit-time>_<sha8>.yaml — placing the run on
the timeline at its COMMIT time.  No golden gate is applied (a baseline isn't a pass/fail checkpoint).
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

  # ---- best-effort: activate the mbirjax conda env ------------------------------------------------
  if [ "${CONDA_DEFAULT_ENV:-}" != "mbirjax" ]; then
    set +e
    command -v conda >/dev/null 2>&1 || for s in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
      [ -f "$s/etc/profile.d/conda.sh" ] && . "$s/etc/profile.d/conda.sh" && break
    done
    command -v conda >/dev/null 2>&1 && eval "$(conda shell.bash hook)" && conda activate mbirjax
    set -e
  fi

  # ---- platform + output dir (mirror run_regression.sh) -------------------------------------------
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then PLAT="gpu"; else PLAT="cpu"; fi
  OUT="$REPO/results/$PLAT/$SLUG"; mkdir -p "$OUT"

  # ---- isolated checkout of the chosen commit, then measure --------------------------------------
  WT="$(mktemp -d)/lib"
  git -C "$SRC" worktree add --quiet --detach "$WT" "$COMMITISH"
  SHA="$(git -C "$WT" rev-parse --short=8 HEAD)"
  echo "add_run: $PLAT · branch=$BRANCH · commit=$SHA · src=$SRC"
  echo "         -> $OUT"

  # REG_GATE=0: a baseline is reference data, not a pass/fail checkpoint (no nonzero exit, no golden
  # gate).  The engine still records a day-over-day note vs the prior commit's run, if any.
  REG_LIB_ROOT="$WT" REG_OUT_DIR="$OUT" REG_RUN_TAG="$BRANCH" REG_GATE=0 \
    python "$REPO/tooling/scaling_tests/run_nightly.py"
)
_rc=$?
if [ "$_sourced" -eq 1 ]; then return "$_rc"; else exit "$_rc"; fi
