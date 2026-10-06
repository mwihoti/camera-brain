#!/usr/bin/env bash
# SERVER SIDE. Runs the whole Camera Brain headless on the remote server:
# tracker (+ VLM narration) and the Telegram bot, sharing CAMERA_DATA_DIR.
# Streams arrive via the laptop's stream_bridge.sh (ssh -R tunnels).
#
#   ./run_server.sh v380          # tracker on the tunnelled V380 + telegram bot
#   ./run_server.sh mevo
#   ./run_server.sh v380 --zoom 2 --at 60,30
#
# Logs: $CAMERA_DATA_DIR/logs/{track,telegram}.log. Each process auto-restarts.
set -u
cd "$(dirname "$0")"
# shellcheck disable=SC1090
[ -f "$HOME/.config/camera-agent.env" ] && set -a && . "$HOME/.config/camera-agent.env" && set +a
export CAMERA_DATA_DIR=${CAMERA_DATA_DIR:-/root/camera-data}
PY=${PYTHON:-python3}
CAM=${1:-v380}; shift || true

case "$CAM" in
  v380) SRC=${CAMERA_SRC:-rtsp://admin:@127.0.0.1:18554/live/ch00_0} ;;
  mevo) SRC=${CAMERA_SRC:-tcp://127.0.0.1:9001} ;;
  *) echo "usage: $0 v380|mevo [track_live.py args]"; exit 1 ;;
esac

export CAMERA_SRC=$SRC   # telegram_watch /clip records from this URL
mkdir -p "$CAMERA_DATA_DIR/logs"
log() { printf '%s server: %s\n' "$(date +%H:%M:%S)" "$*"; }

supervise() {  # name, cmd...
  local name=$1; shift
  while true; do
    log "start $name"
    "$@" >>"$CAMERA_DATA_DIR/logs/$name.log" 2>&1
    log "$name exited ($?); restart in 10s"; sleep 10
  done
}

trap 'log "stopping"; kill 0' INT TERM
supervise track    "$PY" track_live.py --camera "$CAM" --src "$SRC" --headless "$@" &
supervise telegram "$PY" telegram_watch.py &
log "camera brain running on $CAM <- $SRC ; data in $CAMERA_DATA_DIR"
wait
