#!/usr/bin/env python3
"""
attendance.py — Lean room attendance from a V380 (RTSP) camera.

Storage-light by design: it saves ONE small snapshot per event and appends a row
to a CSV register. No continuous video recording.

Two modes
---------
  door : counts people crossing a virtual line = IN / OUT  (best for a doorway)
  room : tracks how many people are present over time       (open room view)

The mode is chosen automatically:
  - give --line  -> door mode
  - no --line    -> room mode

Setup
-----
    pip install -r requirements-attendance.txt

Run (door mode — draw the line across the doorway once, see --line below):
    python attendance.py --url "rtsp://admin:PWD@192.168.1.50:554/live/ch00_0" \
        --line 0,360,1280,360

Run (room / occupancy mode):
    python attendance.py --url "rtsp://admin:PWD@192.168.1.50:554/live/ch00_0"

First run downloads a small YOLO model (~6 MB) automatically.

Keys in the window:  q = quit   s = snapshot   (or run --no-window for headless)

Tip to find your line coordinates: run once, press 's' to save a snapshot, open
it, and read the pixel X,Y of where the doorway is. --line is x1,y1,x2,y2.
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from datetime import datetime

import cv2


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fstamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def side_of_line(px, py, x1, y1, x2, y2):
    """Sign tells which side of the line the point is on."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def open_stream(url, retries=5):
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if cap.isOpened():
            print(f"[{now_str()}] Connected.")
            return cap
        print(f"[{now_str()}] Connect attempt {attempt}/{retries} failed...")
        cap.release()
        time.sleep(2)
    raise SystemExit("Could not open the stream. Check URL / password / network.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="RTSP URL of the camera")
    ap.add_argument("--line", default=None,
                    help="doorway line 'x1,y1,x2,y2' -> enables DOOR mode")
    ap.add_argument("--flip", action="store_true",
                    help="swap which crossing direction counts as IN")
    ap.add_argument("--outdir", default="attendance_data", help="output folder")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="detection confidence threshold")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model file")
    ap.add_argument("--room-heartbeat", type=int, default=300,
                    help="room mode: log the current count every N seconds")
    ap.add_argument("--no-window", action="store_true", help="headless mode")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    snaps_dir = os.path.join(args.outdir, "snaps")
    os.makedirs(snaps_dir, exist_ok=True)
    log_path = os.path.join(args.outdir, "attendance.csv")
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["timestamp", "event", "track_id",
                         "direction", "occupancy", "snapshot"])

    try:
        from ultralytics import YOLO
    except Exception:
        raise SystemExit(
            "ultralytics is not installed. Run:\n"
            "    pip install -r requirements-attendance.txt")
    model = YOLO(args.model)
    print(f"[{now_str()}] Model loaded.")

    door_mode = args.line is not None
    if door_mode:
        x1, y1, x2, y2 = (int(v) for v in args.line.split(","))
        print(f"[{now_str()}] DOOR mode. Line=({x1},{y1})->({x2},{y2})")
    else:
        print(f"[{now_str()}] ROOM (occupancy) mode.")

    def snapshot(frame, tag):
        path = os.path.join(snaps_dir, f"{tag}_{fstamp()}.jpg")
        cv2.imwrite(path, frame)
        return path

    def log_event(event, frame, track_id="", direction="", occupancy=""):
        snap = snapshot(frame, event)
        writer.writerow([now_str(), event, track_id, direction, occupancy, snap])
        log.flush()
        print(f"[{now_str()}] {event} id={track_id} {direction} "
              f"occupancy={occupancy}")

    cap = open_stream(args.url)

    prev_side = {}          # track_id -> last sign relative to line
    counted_in = 0
    counted_out = 0
    last_room_count = -1
    last_heartbeat = 0.0

    print(f"[{now_str()}] Running. Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"[{now_str()}] Stream hiccup, reconnecting...")
                cap.release()
                cap = open_stream(args.url)
                continue

            # Track people (class 0). persist=True keeps IDs across frames.
            results = model.track(frame, persist=True, classes=[0],
                                  conf=args.conf, verbose=False)
            r = results[0]
            present = 0
            centroids = {}
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
                xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                present = len(ids)
                for tid, b in zip(ids, xyxy):
                    cx = (b[0] + b[2]) // 2
                    cy = (b[1] + b[3]) // 2
                    centroids[tid] = (cx, cy)
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]),
                                  (0, 200, 0), 2)
                    cv2.putText(frame, f"#{tid}", (b[0], b[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

            if door_mode:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                for tid, (cx, cy) in centroids.items():
                    s = side_of_line(cx, cy, x1, y1, x2, y2)
                    sign = 1 if s > 0 else (-1 if s < 0 else 0)
                    if tid in prev_side and sign != 0 and prev_side[tid] != 0 \
                            and sign != prev_side[tid]:
                        going_in = (sign > 0)
                        if args.flip:
                            going_in = not going_in
                        if going_in:
                            counted_in += 1
                            occ = counted_in - counted_out
                            log_event("IN", frame, tid, "in", occ)
                        else:
                            counted_out += 1
                            occ = counted_in - counted_out
                            log_event("OUT", frame, tid, "out", occ)
                    if sign != 0:
                        prev_side[tid] = sign
                occ_now = counted_in - counted_out
                cv2.putText(frame, f"IN:{counted_in}  OUT:{counted_out}  "
                            f"INSIDE:{occ_now}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # ROOM mode: log whenever the count changes, plus heartbeats.
                if present != last_room_count:
                    log_event("count_change", frame, occupancy=present)
                    last_room_count = present
                now = time.time()
                if now - last_heartbeat >= args.room_heartbeat:
                    log_event("heartbeat", frame, occupancy=present)
                    last_heartbeat = now
                cv2.putText(frame, f"PEOPLE IN ROOM: {present}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, now_str(), (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if not args.no_window:
                cv2.imshow("Attendance — q quit, s snapshot", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("s"):
                    print(f"[{now_str()}] snapshot -> {snapshot(frame, 'manual')}")
    except KeyboardInterrupt:
        print(f"\n[{now_str()}] Stopping.")
    finally:
        # Final session summary row.
        if door_mode:
            writer.writerow([now_str(), "SESSION_END", "", "",
                             f"in={counted_in};out={counted_out};"
                             f"inside={counted_in - counted_out}", ""])
        else:
            writer.writerow([now_str(), "SESSION_END", "", "",
                             f"last_count={max(last_room_count,0)}", ""])
        log.flush()
        cap.release()
        log.close()
        cv2.destroyAllWindows()
        print(f"[{now_str()}] Register saved to {log_path}")


if __name__ == "__main__":
    main()
