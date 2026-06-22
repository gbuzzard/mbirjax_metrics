#!/usr/bin/env bash
# enable_nightly.sh — install + start the scheduled nightly regression.  Platform-aware:
#   macOS         -> launchd agent (filled from com.mbirjax.regression.plist).
#   Linux/cluster -> a managed scrontab block: a daily batch job (SLURM_* opts from run_configs.env)
#                    running the wrapper.  Re-run after editing run_configs.env to apply changes.
#
# Run it from the regression/ dir (in the metrics clone for the real install, or here during dev).
# On macOS it fills the plist from regression.env (schedule + wrapper path + a conda PATH) and loads
# it; launchd runs at the scheduled time but only if the laptop is awake.  Re-run after
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
  # ── Linux / cluster (SLURM scrontab) ────────────────────────────────────────────────────────
  # Install a managed scrontab block: a daily batch job (on POLL_SCHEDULE, with the SLURM_* options
  # from run_configs.env) that runs the wrapper.  fire-on-change makes most nights a seconds-long
  # no-op; a tracked branch that moved triggers the real sweep.
  command -v scrontab >/dev/null 2>&1 || {
    echo "ERROR: scrontab not found — this cluster's Slurm lacks the cron feature."; exit 1; }
  WRAPPER="$HERE/run_regression.sh"
  [ -f "$WRAPPER" ] || { echo "ERROR: wrapper not found at $WRAPPER"; exit 1; }
  LOGDIR="$WORK_DIR"; mkdir -p "$LOGDIR"
  B="# mbirjax-nightly-BEGIN"; E="# mbirjax-nightly-END"   # markers for the managed block
  OPTS="-A ${SLURM_ACCOUNT} -p ${SLURM_PARTITION} -q ${SLURM_QOS} -N1"
  OPTS="$OPTS --gpus-per-node=${SLURM_GPUS_PER_NODE} -n ${SLURM_NTASKS} -t ${SLURM_WALLTIME}"
  OPTS="$OPTS -J mbirjax-nightly --mail-user=${NOTIFY} --mail-type=FAIL -o ${LOGDIR}/nightly-%j.log"
  BLOCK="$(printf '%s\n#SCRON %s\n%s bash %s\n%s' "$B" "$OPTS" "$POLL_SCHEDULE" "$WRAPPER" "$E")"
  # Drop any existing managed block, then append the fresh one (leaving the user's other entries).
  CUR="$(scrontab -l 2>/dev/null | sed "/$B/,/$E/d")" || CUR=""
  { [ -n "$CUR" ] && printf '%s\n' "$CUR"; printf '%s\n' "$BLOCK"; } | scrontab -
  echo "Installed scrontab nightly:"
  echo "  schedule: $POLL_SCHEDULE   account: $SLURM_ACCOUNT   $SLURM_PARTITION/$SLURM_QOS   ${SLURM_GPUS_PER_NODE} GPU   t=$SLURM_WALLTIME"
  echo "  wrapper:  $WRAPPER"
  echo "  logs:     $LOGDIR/nightly-<jobid>.log"
  echo "  inspect:  scrontab -l   |   squeue --me"
  echo "  (ENABLED=$ENABLED in regression.env is the in-wrapper kill-switch; this controls the schedule.)"
  exit 0
fi

# ── macOS / launchd ───────────────────────────────────────────────────────────────────────────
LABEL="com.mbirjax.regression"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
WRAPPER="$HERE/run_regression.sh"
TEMPLATE="$HERE/com.mbirjax.regression.plist"
LOGDIR="$HOME/.mbirjax/regression"

[ -f "$WRAPPER" ] || { echo "ERROR: wrapper not found at $WRAPPER"; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH (run from a shell where conda works)."; exit 1; }

# macOS runs at MACOS_NIGHTLY_TIME (24h "HH:MM", from run_configs.env) — a time the Mac is AWAKE (a
# scheduled wake from sleep is a dark wake that won't fire a LaunchAgent).  The cluster path uses POLL_SCHEDULE.
RUN_TIME="${MACOS_NIGHTLY_TIME:-09:00}"
case "$RUN_TIME" in
  [0-9]:[0-5][0-9]|[0-1][0-9]:[0-5][0-9]|2[0-3]:[0-5][0-9]) ;;
  *) echo "ERROR: MACOS_NIGHTLY_TIME='$RUN_TIME' must be 24-hour HH:MM (e.g. 09:00)." >&2; exit 2 ;;
esac
HR=$((10#${RUN_TIME%%:*})); MIN=$((10#${RUN_TIME##*:}))

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
