#!/usr/bin/env bash
# disable_nightly.sh — stop the scheduled nightly regression.  Platform-aware:
#   macOS         -> unload + remove the launchd agent (works now).
#   Linux/cluster -> remove the scrontab entry.  NOT YET — pending the slurm script.
# Stops the SCHEDULE only; it does not touch config or any results.  (For a quick pause without
# uninstalling, set ENABLED=0 in regression.env — the wrapper then exits immediately.)
set -euo pipefail
# Keep an interactive terminal open on a nonzero exit so the error stays visible.
if [ -t 0 ]; then
  trap '_ec=$?; [ "$_ec" -ne 0 ] && { echo; echo ">>> $(basename "$0") exited with status $_ec — press Enter to close."; read -r _ || true; }' EXIT
fi
if [ "$(uname -s)" != "Darwin" ]; then
  echo "disable_nightly: scheduled runs on $(uname -s) are not implemented yet (cluster will use"
  echo "  scrontab + nightly_regression.slurm).  Nothing to disable."
  exit 0
fi

# ── macOS / launchd ───────────────────────────────────────────────────────────────────────────
LABEL="com.mbirjax.regression"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl unload -w "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "Disabled $LABEL (unloaded + removed $PLIST)."
