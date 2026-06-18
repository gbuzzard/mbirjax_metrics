#!/usr/bin/env bash
# enable_nightly.sh — install + start the scheduled nightly regression.  Platform-aware:
#   macOS         -> launchd agent (filled from com.mbirjax.regression.plist).  Works now.
#   Linux/cluster -> scrontab + nightly_regression.slurm.  NOT YET — pending the slurm script.
#
# Run it from the regression/ dir (in the metrics clone for the real install, or here during dev).
# On macOS it fills the plist from regression.env (schedule + wrapper path + a conda PATH) and loads
# it; launchd runs at the scheduled time and at the next wake if the laptop slept.  Re-run after
# editing regression.env / run_configs.env to apply changes.
set -euo pipefail
# Keep an interactive terminal open on a nonzero exit so the error stays visible.
if [ -t 0 ]; then
  trap '_ec=$?; [ "$_ec" -ne 0 ] && { echo; echo ">>> $(basename "$0") exited with status $_ec — press Enter to close."; read -r _ || true; }' EXIT
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/regression.env"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "enable_nightly: scheduled runs on $(uname -s) are not implemented yet."
  echo "  macOS uses launchd (this path); the cluster will use scrontab + nightly_regression.slurm,"
  echo "  which is still to be written.  For now, trigger a pass by hand: action_scripts/run_one_night.sh"
  exit 1
fi

# ── macOS / launchd ───────────────────────────────────────────────────────────────────────────
LABEL="com.mbirjax.regression"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
WRAPPER="$HERE/run_regression.sh"
TEMPLATE="$HERE/com.mbirjax.regression.plist"
LOGDIR="$HOME/.mbirjax/regression"

[ -f "$WRAPPER" ] || { echo "ERROR: wrapper not found at $WRAPPER"; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH (run from a shell where conda works)."; exit 1; }

# Daily HH:MM from POLL_SCHEDULE ("M H * * *"); only daily is supported by this installer.
MIN="$(echo "$POLL_SCHEDULE" | awk '{print $1}')"; [ "$MIN" = "*" ] && MIN=0
HR="$(echo "$POLL_SCHEDULE"  | awk '{print $2}')"; [ "$HR"  = "*" ] && HR=2
CONDA_BIN="$(dirname "$(command -v conda)")"

mkdir -p "$LOGDIR" "$HOME/Library/LaunchAgents"
sed -e "s|@LABEL@|$LABEL|g" \
    -e "s|@WRAPPER@|$WRAPPER|g" \
    -e "s|@HOUR@|$HR|g" -e "s|@MINUTE@|$MIN|g" \
    -e "s|@PATH@|$CONDA_BIN:/usr/bin:/bin:/usr/sbin:/sbin|g" \
    -e "s|@LOGOUT@|$LOGDIR/launchd.out.log|g" \
    -e "s|@LOGERR@|$LOGDIR/launchd.err.log|g" \
    "$TEMPLATE" > "$PLIST"

launchctl unload -w "$PLIST" 2>/dev/null || true
launchctl load -w "$PLIST"
printf 'Loaded %s — runs daily at %02d:%02d\n' "$LABEL" "$HR" "$MIN"
echo "  wrapper: $WRAPPER"
echo "  logs:    $LOGDIR/launchd.{out,err}.log"
echo "  (ENABLED=$ENABLED in regression.env is the in-wrapper kill-switch; this controls the schedule.)"
