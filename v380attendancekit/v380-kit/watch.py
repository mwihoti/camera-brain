#!/usr/bin/env python3
"""
watch.py — Live view + activity analysis for a V380 Pro (RTSP) camera.

What it does
------------
- Connects to the camera's RTSP stream and shows it live on your laptop.
- Detects MOTION (fast, always on).
- Optionally detects PEOPLE with YOLO (accurate; needs `ultralytics` installed).
- Saves a snapshot + a short video clip whenever an "event" starts.
- Writes an events.csv log you can open in Excel for analysis.

Run it
------
    python watch.py --url "rtsp://admin:PASSWORD@192.168.1.50:554/live/ch00_0"

Add people detection (after `pip install ultralytics`):
    python watch.py --url "rtsp://..." --detect people

Keys while the window is open:  q = quit   s = manual snapshot
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fstamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_stream(url, retries=5):
    """Open the RTSP stream, retrying because Wi-Fi cameras drop connections."""
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # Keep latency low: don't let OpenCV buffer a huge backlog.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if cap.isOpened():
            print(f"[{ts()}] Connected to stream.")
            return cap
        print(f"[{ts()}] Connect attempt {attempt}/{retries} failed, retrying...")
        cap.release()
        time.sleep(2)
    raise SystemExit(
        "Could not open the stream. Check the URL, password, and that the "
        "camera + laptop are on the same network."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="RTSP URL of the camera")
    ap.add_argument("--detect", choices=["motion", "people"], default="motion",
                    help="motion (default) or people (needs ultralytics)")
    ap.add_argument("--outdir", default="captures", help="where to save clips/snaps")
    ap.add_argument("--sensitivity", type=int, default=25,
                    help="motion pixel threshold, lower = more sensitive")
    ap.add_argument("--min-area", type=int, default=1500,
                    help="ignore motion blobs smaller than this many pixels")
    ap.add_argument("--cooldown", type=float, default=10.0,
                    help="seconds to keep recording after last activity")
    ap.add_argument("--no-window", action="store_true",
                    help="run headless (no live window), just log + save")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "events.csv")
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["timestamp", "event", "detail", "snapshot"])

    # Optional YOLO model
    model = None
    if args.detect == "people":
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")  # small, auto-downloads first run
            print(f"[{ts()}] YOLO loaded — detecting people.")
        except Exception as e:
            print(f"[{ts()}] Could not load YOLO ({e}). Falling back to motion.")
            args.detect = "motion"

    cap = open_stream(args.url)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500,
                                                  varThreshold=args.sensitivity,
                                                  detectShadows=False)

    recording = False
    writer_vid = None
    last_active = 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    fps = fps if 1 < fps < 60 else 15.0

    print(f"[{ts()}] Watching. Press Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"[{ts()}] Stream hiccup, reconnecting...")
                cap.release()
                cap = open_stream(args.url)
                continue

            active = False
            detail = ""

            if args.detect == "people" and model is not None:
                res = model(frame, verbose=False, classes=[0])  # class 0 = person
                boxes = res[0].boxes
                n = len(boxes)
                if n > 0:
                    active = True
                    detail = f"{n} person(s)"
                    for b in boxes.xyxy.cpu().numpy().astype(int):
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            else:
                mask = back_sub.apply(frame)
                mask = cv2.medianBlur(mask, 5)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                big = [c for c in cnts if cv2.contourArea(c) > args.min_area]
                if big:
                    active = True
                    detail = f"{len(big)} moving region(s)"
                    for c in big:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)

            cv2.putText(frame, ts(), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            now = time.time()
            if active:
                last_active = now
                if not recording:
                    recording = True
                    snap = os.path.join(args.outdir, f"snap_{fstamp()}.jpg")
                    cv2.imwrite(snap, frame)
                    h, w = frame.shape[:2]
                    clip = os.path.join(args.outdir, f"clip_{fstamp()}.mp4")
                    writer_vid = cv2.VideoWriter(
                        clip, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.writerow([ts(), "activity_start", detail, snap])
                    log.flush()
                    print(f"[{ts()}] ACTIVITY: {detail} -> {snap}")

            if recording:
                writer_vid.write(frame)
                if now - last_active > args.cooldown:
                    recording = False
                    writer_vid.release()
                    writer_vid = None
                    writer.writerow([ts(), "activity_end", "", ""])
                    log.flush()
                    print(f"[{ts()}] activity ended.")

            if not args.no_window:
                cv2.imshow("V380 live — q quit, s snapshot", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("s"):
                    snap = os.path.join(args.outdir, f"manual_{fstamp()}.jpg")
                    cv2.imwrite(snap, frame)
                    print(f"[{ts()}] manual snapshot -> {snap}")
    except KeyboardInterrupt:
        print(f"\n[{ts()}] Stopping.")
    finally:
        cap.release()
        if writer_vid is not None:
            writer_vid.release()
        log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
