#!/usr/bin/env bash
# Push Camera Brain data to the remote sandbox so the server-side Telegram
# bot stays fresh: tracker events + crops, and the live annotated frame.
# Run on the laptop whenever the tracker runs:  ./sync_to_server.sh &

HOST=qm-camera-brain
DEST=/root/camera-data

while true; do
  rsync -az --delete "$HOME/Videos/camera/tracks/" "$HOST:$DEST/tracks/" 2>/dev/null
  [ -f /tmp/camera_live.jpg ] && \
    rsync -az /tmp/camera_live.jpg "$HOST:$DEST/camera_live.jpg" 2>/dev/null
  sleep 5
done
