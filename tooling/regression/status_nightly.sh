#!/usr/bin/env bash
# status_nightly.sh — report whether the scheduled nightly regression will actually run.  TWO layers
# must both be true for a nightly to do anything:
#   1. the SCHEDULE is installed     (macOS: a loaded launchd agent; cluster: the scrontab block)
#   2. the ENABLED kill-switch is 1   (regression.env; run_regression.sh exits immediately when 0)
# Read-only: touches no config, schedule, or results.  Companion to enable_nightly.sh / disable_nightly.sh.
set -euo pipefail
# Keep an interactive terminal open on a nonzero exit so the error stays visible.
if [ -t 0 ]; then
  trap '_ec=$?; [ "$_ec" -ne 0 ] && { echo; echo ">>> $(basename "$0") exited with status $_ec — press Enter to close."; read -r _ || true; }' EXIT
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/regression.env"

# The recent-run summary reuses tooling/viewer/build_dashboard.py, so it needs the SAME interpreter
# the dashboard build uses — the `mbirjax` conda env, where its only dep (PyYAML) lives (see
# build_dashboard.sh).  Prefer that env (then the harness env, then bare python) and validate by
# importing yaml; MBIRJAX_STATUS_PYTHON overrides.
_find_python() {
  local cands=() py base d roots=("$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge")
  [ -n "${MBIRJAX_STATUS_PYTHON:-}" ] && cands+=("$MBIRJAX_STATUS_PYTHON")
  [ "${CONDA_DEFAULT_ENV:-}" = "mbirjax" ] && cands+=("python")        # dashboard's env already active
  command -v conda >/dev/null 2>&1 && base="$(conda info --base 2>/dev/null || true)" || base=""
  for r in ${base:+"$base"} "${roots[@]}"; do                          # mbirjax env first (= dashboard), then harness env
    cands+=("$r/envs/mbirjax/bin/python" "$r/envs/${CONDA_ENV:-mbirjax_regression}/bin/python")
  done
  cands+=("python3" "python")                                          # last resort (system python often lacks yaml)
  for py in "${cands[@]}"; do
    [ -n "$py" ] || continue
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import yaml" >/dev/null 2>&1; then
      printf '%s\n' "$py"; return 0
    fi
  done
  return 1
}

echo "mbirjax nightly status"
scheduled=0   # set to 1 by the platform block below if a schedule is installed

if [ "$(uname -s)" != "Darwin" ]; then
  # ── Linux / cluster (SLURM scrontab) ────────────────────────────────────────────────────────
  echo "  platform: cluster (SLURM scrontab)"
  if ! command -v scrontab >/dev/null 2>&1; then
    echo "  schedule: scrontab NOT FOUND — this cluster's Slurm lacks the cron feature."
  else
    B="# mbirjax-nightly-BEGIN"; E="# mbirjax-nightly-END"
    CUR="$(scrontab -l 2>/dev/null)" || CUR=""
    if printf '%s\n' "$CUR" | grep -qF "$B"; then
      scheduled=1
      SCHED_LINE="$(printf '%s\n' "$CUR" | sed -n "/$B/,/$E/p" | grep -vE '^#' | head -1)"
      echo "  schedule: INSTALLED — cron \"${SCHED_LINE%% bash *}\""
      echo "  wrapper:  $HERE/run_regression.sh"
    else
      echo "  schedule: not installed  (run ./enable_nightly.sh to install)"
    fi
    if command -v squeue >/dev/null 2>&1; then
      Q="$(squeue --me --name=mbirjax-nightly -h 2>/dev/null)" || Q=""
      [ -n "$Q" ] && { echo "  in queue now:"; printf '%s\n' "$Q" | sed 's/^/    /'; }
    fi
  fi
else
  # ── macOS / launchd ─────────────────────────────────────────────────────────────────────────
  echo "  platform: macOS (launchd)"
  LABEL="com.mbirjax.regression"; PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  LOGDIR="$HOME/.mbirjax/regression"
  if launchctl list 2>/dev/null | grep -qF "$LABEL"; then
    scheduled=1; echo "  schedule: LOADED ($LABEL)"
  elif [ -f "$PLIST" ]; then
    echo "  schedule: plist present but NOT loaded  (run ./enable_nightly.sh)"
  else
    echo "  schedule: not installed  (run ./enable_nightly.sh to install)"
  fi
  if [ -f "$PLIST" ]; then
    HR="$(grep -oE '<key>Hour</key><integer>[0-9]+' "$PLIST" | grep -oE '[0-9]+$' || true)"
    MN="$(grep -oE '<key>Minute</key><integer>[0-9]+' "$PLIST" | grep -oE '[0-9]+$' || true)"
    [ -n "${HR:-}" ] && printf '  runs at: daily %02d:%02d (local)\n' "$HR" "${MN:-0}"
  fi
  if [ -f "$LOGDIR/launchd.out.log" ]; then
    echo "  last out: $LOGDIR/launchd.out.log ($(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$LOGDIR/launchd.out.log"))"
  fi
fi

# ── kill-switch + overall verdict ───────────────────────────────────────────────────────────────
echo "  ENABLED kill-switch (regression.env): ${ENABLED:-0}"
echo
if [ "$scheduled" = "1" ] && [ "${ENABLED:-0}" = "1" ]; then
  echo "✅ Nightly WILL run — scheduled and ENABLED=1.  Wakes on POLL_SCHEDULE=\"$POLL_SCHEDULE\";"
  echo "   actual work happens only when a tracked branch has moved (fire-on-change)."
elif [ "$scheduled" = "1" ]; then
  echo "⏸  Scheduled, but ENABLED=0 — the wrapper exits immediately.  Set ENABLED=1 in"
  echo "   regression.env to resume (no reinstall needed)."
elif [ "${ENABLED:-0}" = "1" ]; then
  echo "❌ NOT scheduled — nothing fires.  Run ./enable_nightly.sh to install the schedule"
  echo "   (ENABLED=1 already, so it runs as soon as it's scheduled)."
else
  echo "❌ Fully off — not scheduled AND ENABLED=0.  Run ./enable_nightly.sh and set ENABLED=1."
fi

# ── recent activity: when it last fired + the tile info for the runs that produced results ──────
echo
# When the schedule last actually woke (log mtime), platform-appropriate.
if [ "$(uname -s)" = "Darwin" ]; then
  FIRED_LOG="$HOME/.mbirjax/regression/launchd.out.log"
else
  FIRED_LOG="$(ls -t "$WORK_DIR"/nightly-*.log 2>/dev/null | head -1 || true)"
fi
if [ -n "${FIRED_LOG:-}" ] && [ -f "$FIRED_LOG" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then WHEN="$(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$FIRED_LOG")"
  else WHEN="$(date -r "$FIRED_LOG" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '?')"; fi
  echo "last wake: $WHEN   (log: $FIRED_LOG)"
else
  echo "last wake: no nightly log found yet (it hasn't fired on this machine, or logs live elsewhere)"
fi

# Tile-style summary of recent runs, via the dashboard's own collect_data() (see recent_runs.py).
# Report on the persistent metrics clone the nightly writes to ($WORK_DIR/metrics) when it has
# results; otherwise this checkout (what's been pulled here).  recent_runs.py imports that repo's
# build_dashboard, so collect_data() reads the matching results/.
MC_ROOT="$WORK_DIR/metrics"
REPO_ROOT_DIR="$(cd "$HERE/../.." && pwd)"
if [ -d "$MC_ROOT/results" ] && ls "$MC_ROOT"/results/*/*/regression_*.yaml >/dev/null 2>&1; then
  TARGET_ROOT="$MC_ROOT"
else
  TARGET_ROOT="$REPO_ROOT_DIR"
fi
echo
PYBIN="$(_find_python || true)"
if [ -n "$PYBIN" ] && "$PYBIN" "$HERE/recent_runs.py" "$TARGET_ROOT" 6; then
  :
else
  echo "recent runs (from $TARGET_ROOT/results) — no PyYAML-capable Python found (the dashboard's"
  echo "mbirjax env or MBIRJAX_STATUS_PYTHON); showing filenames only:"
  ls -t "$TARGET_ROOT"/results/*/*/regression_*.yaml 2>/dev/null | head -6 | sed 's/^/  /' || true
fi
