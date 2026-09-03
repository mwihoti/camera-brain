# Camera Brain — handoff for any agent (read this first)

Owner: Mwihoti (danielmwihoti@gmail.com), Nairobi. Building in public.
Repo: github.com/mwihoti/camera-brain.

DECISION (2026-09-03): ALL camera processing now runs on the remote server
(the qm sandbox), not the laptop. The laptop's only job is to get the camera
streams to the server (relay/tunnel — see "Server-side run" below). Cameras
sit on the laptop's LAN and are NOT directly reachable from the server, so
every pipeline must consume a stream URL that the laptop pushes/tunnels.

## Server-side run (how it works now)
- Server: 32-core CPU, no GPU, 251 GB RAM. YOLO11n on CPU is fine. sshd runs
  here and the laptop already has the ssh alias `qm-camera-brain`, so the
  bridge is ssh -R reverse tunnels — no inbound ports anywhere.
- Laptop (only job):  v380-kit/stream_bridge.sh v380|mevo|both
    v380: ssh -R 8554->cam:554 (RTSP, TCP-interleaved) + 8899->cam:8899 (ONVIF)
    mevo: SRT is UDP (ssh can't forward it) -> local ffmpeg pulls SRT, serves
          MPEG-TS on tcp://127.0.0.1:9001?listen, ssh -R 9001 exposes it.
- Server:  v380-kit/run_server.sh v380|mevo [track_live args]
    supervises track_live.py --headless --src <tunnel URL> + telegram_watch.py;
    both share CAMERA_DATA_DIR (/root/camera-data) so sync_to_server.sh and
    rsync are no longer needed. Logs in $CAMERA_DATA_DIR/logs/.
- track_live.py now: --src URL override (or $CAMERA_SRC), --headless (no
  ffplay), day folder rolls over at midnight, reads all keys from
  ~/.config/camera-agent.env. Server env has CAMERA_DATA_DIR and
  TELEGRAM_BOT_TOKEN; NVIDIA_API_KEY still needs adding on the server.
- ptz.py / ONVIF reboot: point at 127.0.0.1:8899 when run from the server.
- python deps on server: /opt/agent-venv (opencv-python-headless, ultralytics).

## Cameras
- V380 bulb cam: RTSP unlocked via ceshi.ini on SD card; ONVIF PTZ on 8899;
  mounted upside down (everything hflip,vflip); encoder locked 720p q4.
- Mevo Core: srt://192.168.2.159:4201?mode=caller (1080p), ONE SRT client
  at a time; wedged slot = power-cycle + re-toggle SRT in the app. Status
  via avahi-browse -rt _ls-cameraman._tcp. No open control API.

## Hard-won rules
- Small VLMs parrot prompt examples verbatim and invent objects on empty
  scenes → never put example sentences in prompts; ground with detections;
  let the VLM count people itself (YOLO misses small/distant ones).
- NIM inline images must stay < ~150KB (800x450 q75 ok; 720p → 400/500).
- NVIDIA free tier is flaky (500/502/timeouts): circuit breaker → fallback.
- Weak Wi-Fi: substreams, long timeouts, auto-reconnect, ONVIF reboot.

## Where things are going (PLAN.md)
Pilot at a gate → three numbers (accuracy, reliability, unit cost) →
case study → NVIDIA Inception, Lacuna Fund (Kenyan street dataset),
Antler. Next builds: vehicle speed, plate OCR on crops, queryable camera
memory, Kenyan-classes YOLO fine-tune, dashboard (build on the sandbox).
