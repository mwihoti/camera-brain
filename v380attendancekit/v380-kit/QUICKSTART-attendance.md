# Attendance Tracker — Quick Start

Lean room attendance from your V380 camera. Saves a CSV register + one small
snapshot per event. **No continuous video** — stays tiny on disk.

## 0. Enable RTSP on the camera (one time)
Copy the file **`ceshi.ini`** to the root of your microSD card, put the card in
the camera, and power-cycle it. This switches on RTSP so your laptop can connect.
(If your V380 Pro app already has an ONVIF/RTSP toggle in settings, you can use
that instead.)

## 1. Get the camera on your network
Connect the camera in the V380 Pro app to the **same Wi-Fi** as your laptop.
Find its IP (router page, or `arp -a` on Windows / `nmap -sn 192.168.1.0/24`).

## 2. Confirm the stream works in VLC
Media → Open Network Stream → try:
```
rtsp://admin:YOURPASSWORD@CAMERA_IP:554/live/ch00_0
```
If it plays, that URL is your `--url`. (The camera label shows user `admin` and
a `pwd`.)

## 3. Install + run
```
pip install -r requirements-attendance.txt
```

**Doorway attendance (most accurate)** — count people crossing an in/out line:
```
python attendance.py --url "rtsp://admin:PWD@CAMERA_IP:554/live/ch00_0" --line 0,360,1280,360
```
`--line x1,y1,x2,y2` is the line across the doorway in pixels. Not sure of the
numbers? Run room mode first (below), press **s** to save a snapshot, open it,
and read off where the doorway sits. If IN/OUT are reversed, add `--flip`.

**Room occupancy (open room, no clear door)** — logs how many are present:
```
python attendance.py --url "rtsp://admin:PWD@CAMERA_IP:554/live/ch00_0"
```

Headless (no window, e.g. leave it running): add `--no-window`.

## 4. Read the results
Everything lands in `attendance_data/`:
- `attendance.csv` — the register (open in Excel). Columns: timestamp, event,
  track_id, direction, occupancy, snapshot.
- `snaps/` — one small JPG per event.

Door mode logs each `IN`/`OUT` with running `INSIDE` count and a `SESSION_END`
summary. Room mode logs each `count_change` plus a `heartbeat` every 5 min.

## Notes on accuracy with this camera
The bulb camera's wide, high angle is great for counting bodies, weaker for
faces. For the cleanest doorway counts, aim it so people pass roughly across the
frame (not straight toward/away from the lens), and place the line where their
whole body clearly crosses.

## Next steps I can add on request
- Live count dashboard, or a phone/email alert when the room fills/empties.
- Named roll-call (face recognition) if you mount a camera at door/face height.
- A daily Excel report auto-built from the CSV.
