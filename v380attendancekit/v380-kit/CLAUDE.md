# Project: V380 camera → live feed on laptop + room attendance

This is a handoff note for Claude Code. Read this first to continue the project.

## Goal
The user (Mwihoti) has a **V380 Pro bulb camera, model Q16S-1** (uses the V380 Pro
app, WiFi 2.4G). They want to:
1. View the camera's live feed on their laptop.
2. Automatically track **room attendance** — headcount + entry/exit — logged
   leanly (CSV + one snapshot per event, **no continuous video** to save storage).
Named face-recognition attendance was considered but deprioritized: the bulb
camera's high, wide angle makes faces unreliable. Headcount/entry-exit is the
chosen approach.

## How the camera connects
The camera normally only talks to the V380 Pro phone app over a proprietary P2P
protocol. To reach it from a laptop we enable its hidden **RTSP** mode:
- `ceshi.ini` (in this folder) goes on the microSD card root; reboot the camera.
  It contains `rtsp_enable=1` / `onvif_enable=1` under `[CONST_PARAM]`.
- Then the stream is at: `rtsp://admin:PASSWORD@CAMERA_IP:554/live/ch00_0`
  (try `/live/ch00_1`, `/onvif1`, `/11` if that path fails). Login is on the
  camera label: user `admin`, plus a `pwd` value.
- Verify in VLC before running code.

## Files in this folder
- `attendance.py` — MAIN deliverable. YOLO person detection + tracking.
  - DOOR mode (`--line x1,y1,x2,y2`): counts IN/OUT across a virtual doorway line.
  - ROOM mode (no line): logs occupancy count changes + periodic heartbeat.
  - Output: `attendance_data/attendance.csv` + `attendance_data/snaps/*.jpg`.
  - Deps: `requirements-attendance.txt` (opencv-python, ultralytics).
- `watch.py` — simpler live-view + motion/optional-YOLO tool (earlier version).
- `README.md` — full setup walkthrough (RTSP enable, find IP, run).
- `QUICKSTART-attendance.md` — short run guide for attendance.py.
- `ceshi.ini` — SD-card file to enable RTSP.
- `requirements.txt`, `requirements-attendance.txt` — dependencies.

## Current status / next steps
- Code is written and syntax-checked; NOT yet run against a real camera.
- Immediate next step for the user: put `ceshi.ini` on the SD card, reboot camera,
  confirm the RTSP URL plays in VLC.
- Then: `pip install -r requirements-attendance.txt` and run `attendance.py`.
- Likely follow-ups the user may want: help picking the `--line` coordinates for
  their doorway, a live count dashboard, phone/email alerts when the room
  fills/empties, or a daily Excel report generated from attendance.csv.

## Notes for whoever continues
- The tracker must run on the USER'S laptop (needs their LAN + a display window).
- Keep it storage-lean: do not add continuous video recording unless asked.
- For better door counts, aim the camera so people cross the frame sideways and
  place the line where a whole body clearly passes.
