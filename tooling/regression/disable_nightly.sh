#!/usr/bin/env bash
# disable_nightly.sh — stop the scheduled nightly regression.  Platform-aware:
#   macOS         -> unload + remove the launchd agent.
#   Linux/cluster -> remove the managed scrontab block.
# Stops the SCHEDULE only; it does not touch config or any results.  (For a quick pause without
# uninstalling, set ENABLED=0 in regression.env — the wrapper then exits immediately.)
set -euo pipefail
# Keep an interactive terminal open on a nonzero exit so the error stays visible.
if [ -t 0 ]; then
  trap '_ec=$?; [ "$_ec" -ne 0 ] && { echo; echo ">>> $(basename "$0") exited with status $_ec — press Enter to close."; read -r _ || true; }' EXIT
fi
if [ "$(uname -s)" != "Darwin" ]; then
  # ── Linux / cluster (SLURM scrontab) ────────────────────────────────────────────────────────
  command -v scrontab >/dev/null 2>&1 || { echo "disable_nightly: scrontab not found; nothing to disable."; exit 0; }
  B="# mbirjax-nightly-BEGIN"; E="# mbirjax-nightly-END"
  CUR="$(scrontab -l 2>/dev/null)" || CUR=""
  if printf '%s\n' "$CUR" | grep -qF "$B"; then
    printf '%s\n' "$CUR" | sed "/$B/,/$E/d" | scrontab -
    echo "Removed the mbirjax-nightly scrontab block (other entries left intact)."
  else
    echo "No mbirjax-nightly scrontab block found; nothing to disable."
  fi
  exit 0
fi

# ── macOS / launchd ───────────────────────────────────────────────────────────────────────────
LABEL="com.mbirjax.regression"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl unload -w "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "Disabled $LABEL (unloaded + removed $PLIST)."
