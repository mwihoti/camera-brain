#!/usr/bin/env bash
# Continuous recorder for the V380 camera.
# Writes 10-minute MP4 segments (no re-encoding, so CPU cost is near zero) to
# ~/Videos/camera/recordings/. Old-file cleanup is handled by agent_watch.py.
# Run under systemd (camera-record.service) so it reconnects if the stream drops.

URL="rtsp://admin:@192.168.1.111:554/live/ch00_0"
DIR="$HOME/Videos/camera/recordings"
mkdir -p "$DIR"

# -display_rotation 180: camera is mounted upside down; this tags the video so
# players show it right side up, without re-encoding.
exec ffmpeg -nostdin -loglevel warning -rtsp_transport tcp \
  -display_rotation 180 -i "$URL" \
  -map 0:v:0 -c copy \
  -f segment -segment_time 600 -segment_atclocktime 1 -reset_timestamps 1 \
  -strftime 1 "$DIR/%Y-%m-%d_%H-%M-%S.mp4"
