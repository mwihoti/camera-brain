# Camera Brain — handoff for any agent (read this first)

Owner: Mwihoti (danielmwihoti@gmail.com), Nairobi. Building in public.
Repo: github.com/mwihoti/camera-brain. Laptop = camera edge; qm sandbox =
server side (Telegram bot, dashboard dev). Cameras are on the laptop's LAN
and are NOT reachable from the sandbox.

## What exists (all in v380attendancekit/v380-kit/)
- track_live.py   THE camera brain: YOLO11n+ByteTrack boxes every object
                  (IDs, moving/stagnant), saves full-res crops of new
                  people/vehicles, NVIDIA VLM narrates the environment in a
                  band — prompt is GROUNDED by YOLO's detections (stops
                  hallucination). --camera mevo|v380, --zoom N --at X,Y.
                  Writes /tmp/camera_live.jpg for the Telegram /snap.
- live_detect.py  VLM caption viewer (room = 2-frame activity analysis).
- traffic_watch.py / attendance_watch.py / agent_watch.py / record.sh / ptz.py
- telegram_watch.py  alerts + /snap /report /status. Server mode via
                  CAMERA_DATA_DIR; sync_to_server.sh rsyncs laptop→sandbox.
- Models: nemotron-nano-12b-v2-vl (works) → cosmos-reason2-8b (404 until
  the NVIDIA account activates it) → claude haiku CLI (last resort).
  NVIDIA key + Telegram token live in ~/.config/camera-agent.env (never
  in the repo).

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
