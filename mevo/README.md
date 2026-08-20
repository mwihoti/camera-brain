# Mevo Core — computer vision project

Logitech Mevo Core camera (unboxed 2026-08-20).

## Hardware
- 4K30 recording, streams up to 1080p30
- Micro Four Thirds (MFT) lens mount — **LENS NOT INCLUDED, camera needs one to work**
- WiFi 6E + Bluetooth, 2x USB-C, HDMI output, 3.5mm audio in, microSD slot
- 3-mic array, ~6h battery, Mevo app for control (iOS/Android)

## Ways to get the feed on the computer (best first)
1. **USB webcam mode (UVC)** — plug USB-C into the laptop; shows up as
   /dev/video* and ffmpeg/OpenCV read it directly. Simplest, zero network.
2. **SRT streaming** — ffmpeg ingests natively: `ffmpeg -i "srt://CAM_IP:PORT"`.
   Configured from the Mevo app. Works over the LAN like our V380 RTSP setup.
3. **NDI** — needs NDI runtime/tools on Linux; may require Mevo Pro sub.
4. **HDMI out** — via a cheap USB HDMI capture dongle -> /dev/video*.

## Plan
- Reuse the v380-kit pipeline (record.sh / traffic_watch.py / live_detect.py):
  only the input URL/device changes; everything downstream (motion gating,
  NVIDIA/Claude analysis, CSV logging, viewer) works as-is.
- 4K frames = much better small-object detection and possibly readable plates.
- First step when a lens is available: plug USB-C, check `ls /dev/video*`,
  then `ffplay /dev/video0`.

Related project: ~/Work/v380/v380attendancekit/v380-kit (V380 bulb camera).

## WORKING CONNECTION (2026-08-20)
- Camera: mevo-M77D8.local = 192.168.2.159 on "BCNBO 5GHz" WiFi
- Stream: `srt://192.168.2.159:4201?mode=caller` (1080p H.264; SRT mode
  toggled on in the Mevo app; camera is the SRT listener)
- View live: `ffplay -fflags nobuffer "srt://192.168.2.159:4201?mode=caller"`
- Status/discovery: `avahi-browse -rt _ls-cameraman._tcp` (TXT shows battery,
  srt_mode, uvc_mode etc). Camera HTTP 38000/38001 = control API (app only).
- UVC/USB webcam mode exists but stayed off; USB-C connection charges only
  until uvc_mode is enabled in the app.

## Commands (scripts live in ~/Work/v380/v380attendancekit/v380-kit)

```bash
cd ~/Work/v380/v380attendancekit/v380-kit

# Camera brain: YOLO boxes every object (IDs, moving/stagnant) + NVIDIA
# environment narration band. Saves full-res crops of new people/vehicles
# to ~/Videos/camera/tracks/YYYY-MM-DD/ + events.csv
.venv/bin/python track_live.py --camera mevo

# Same, with digital zoom (2x, centered at 50% across / 60% down; the crop
# happens BEFORE detection, so zoom = more pixels per car = better plates)
.venv/bin/python track_live.py --camera mevo --zoom 2 --at 50,60

# AI-labeled viewer only (VLM captions, no boxes). Scenes: room | traffic
.venv/bin/python live_detect.py --camera mevo --scene room

# Plain live view, no AI
ffplay -fflags nobuffer "srt://192.168.2.159:4201?mode=caller"

# Stop everything
pkill -f track_live.py; pkill -f live_detect.py; pkill ffplay

# Outputs
ls ~/Videos/camera/tracks/$(date +%F)/
cat ~/Videos/camera/tracks/$(date +%F)/events.csv
```

## Gotchas learned the hard way
- **One SRT client at a time.** Stop any viewer before starting another. If
  connections start failing with I/O errors and the camera's TXT record says
  `streaming=true` with nothing connected, the slot is wedged: power-cycle
  the camera and re-toggle SRT in the app.
- The AI stack is NVIDIA nemotron → cosmos-reason2 (activates once the
  account can invoke it) → claude CLI as last resort. NVIDIA key lives in
  ~/.config/camera-agent.env.
- Brain narration cadence is bound by cloud inference (~8s/look); the YOLO
  boxes are the instant layer (~100ms, local CPU, ~25% load).
