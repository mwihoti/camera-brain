#!/usr/bin/env bash
# LAPTOP SIDE. The only thing the laptop still does: carry the camera streams
# to the server over the existing ssh alias (qm-personal). No inbound
# ports are needed on either side — everything rides ssh -R reverse tunnels.
#
#   ./stream_bridge.sh v380     # RTSP + ONVIF for the bulb cam
#   ./stream_bridge.sh mevo     # SRT is UDP (ssh can't forward it), so ffmpeg
#                               # pulls SRT locally and serves MPEG-TS over TCP
#   ./stream_bridge.sh both
#
# Server then reads:
#   v380  rtsp://admin:@127.0.0.1:18554/live/ch00_0  (ONVIF/PTZ: 127.0.0.1:8899)
#   mevo  tcp://127.0.0.1:9001
#
# Every leg auto-restarts; kill this script to stop all of it.
set -u
HOST=${BRIDGE_HOST:-qm-personal}
V380_RPORT=${V380_RPORT:-18554}   # server-side RTSP port (8554 is taken by a sandbox service)
V380_IP=${V380_IP:-192.168.1.111}
MEVO_SRT=${MEVO_SRT:-"srt://192.168.2.159:4201?mode=caller&latency=50000"}
WHAT=${1:-both}

log() { printf '%s bridge: %s\n' "$(date +%H:%M:%S)" "$*"; }

# ssh -R for TCP services on the LAN (RTSP is TCP-interleaved on the server
# side; ONVIF is plain HTTP). ServerAlive* makes a dead Wi-Fi hop exit fast so
# the loop reconnects.
tunnel_v380() {
  while true; do
    log "v380 tunnel up (rtsp $V380_RPORT, onvif 8899)"
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
        -R "$V380_RPORT":"$V380_IP":554 -R 8899:"$V380_IP":8899 "$HOST"
    log "v380 tunnel dropped; retry in 5s"; sleep 5
  done
}

# Mevo: local ffmpeg pulls SRT (one client at a time — this is that client),
# remuxes to MPEG-TS and listens on a local TCP port; ssh -R exposes that port
# on the server. -c copy: no re-encode on the laptop.
mevo_relay() {
  while true; do
    log "mevo relay: srt -> tcp://127.0.0.1:9001 (listen)"
    ffmpeg -nostdin -loglevel warning -fflags nobuffer \
           -i "$MEVO_SRT" -c copy -f mpegts "tcp://127.0.0.1:9001?listen=1"
    log "mevo relay ended; retry in 5s"; sleep 5
  done
}
tunnel_mevo() {
  while true; do
    log "mevo tunnel up (tcp 9001)"
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
        -R 9001:127.0.0.1:9001 "$HOST"
    log "mevo tunnel dropped; retry in 5s"; sleep 5
  done
}

trap 'log "stopping"; kill 0' INT TERM
case "$WHAT" in
  v380) tunnel_v380 & ;;
  mevo) mevo_relay & tunnel_mevo & ;;
  both) tunnel_v380 & mevo_relay & tunnel_mevo & ;;
  *) echo "usage: $0 v380|mevo|both"; exit 1 ;;
esac
wait
