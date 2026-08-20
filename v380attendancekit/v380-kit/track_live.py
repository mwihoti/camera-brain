#!/usr/bin/env python3
"""Camera with a brain: full object tracking + AI environment narration.

Two layers on one live window:
  - YOLO11n + ByteTrack (local, every frame): boxes EVERY object it can see
    (all 80 classes), each with a persistent ID and measured moving/stagnant
    state. New people/vehicles get a full-resolution crop saved to disk.
  - NVIDIA vision model (cloud, every ~10s): describes the whole environment
    and surroundings in a caption band under the video.

Usage:
    track_live.py --camera mevo [--zoom 2 --at 60,30]
    track_live.py --camera v380
Quit: q in the window / Ctrl-C.

Outputs:
    ~/Videos/camera/tracks/YYYY-MM-DD/events.csv        one row per new object
    ~/Videos/camera/tracks/YYYY-MM-DD/id<N>_<cls>.jpg   full-res first crop
"""

import argparse
import base64
import csv
import json
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ---- args / camera presets --------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--camera", choices=["v380", "mevo"], default="mevo")
ap.add_argument("--zoom", type=float, default=1.0, help="digital zoom factor")
ap.add_argument("--at", default="50,50", help="zoom center as X,Y percent")
args = ap.parse_args()
ZOOM = max(1.0, args.zoom)
CX_PCT, CY_PCT = (float(v) for v in args.at.split(","))

if args.camera == "mevo":
    SRC = "srt://192.168.2.159:4201?mode=caller&latency=50000"
    W, H = 1920, 1080
    VFILTER = "fps=4"
    DISP_W, DISP_H = 1280, 720
else:
    SRC = "rtsp://admin:@192.168.1.111:554/live/ch00_0"
    W, H = 1280, 720
    VFILTER = "hflip,vflip,fps=2"
    DISP_W, DISP_H = 1280, 720

OUT = Path.home() / "Videos" / "camera" / "tracks" / f"{datetime.now():%Y-%m-%d}"
CROP_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF = 0.35
MOVE_PX = 25
MOVE_WINDOW = 3.0
CROP_MIN_AGE = 3
BAND_H = 118
BRAIN_SEC = 1           # near-continuous: next analysis starts right after the last

_envfile = Path.home() / ".config" / "camera-agent.env"
if "NVIDIA_API_KEY" not in os.environ and _envfile.exists():
    for line in _envfile.read_text().splitlines():
        if line.startswith("NVIDIA_API_KEY=") and len(line.split("=", 1)[1]) > 5:
            os.environ["NVIDIA_API_KEY"] = line.split("=", 1)[1].strip()
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODELS = ["nvidia/nemotron-nano-12b-v2-vl", "nvidia/cosmos-reason2-8b"]
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

def brain_prompt(seen: str) -> str:
    # No example sentences (small models parrot them verbatim) and no scene
    # assumptions -- describe only what is actually in the image, grounded by
    # what the local YOLO detector genuinely found.
    return (
        "Describe ONLY what is actually visible in this camera image. "
        f"A local object detector found at least: {seen}. Use that as a "
        "starting point - the detector often MISSES small, distant, or "
        "partially hidden people and objects, so look carefully and count "
        "for yourself (especially people). You may report things you can "
        "clearly see that the detector missed, but never describe anything "
        "that is not in the image. Report with positions (left/right/center): "
        "1) what kind of place this is, 2) how many people you can actually "
        "count and what each is doing, 3) objects/furniture/vehicles with "
        "colors, 4) surroundings, 5) lighting. "
        'Reply ONLY JSON: {"labels": ["...", "..."]} - up to 8 short specific '
        "observations, most important first."
    )
# ----------------------------------------------------------------------------

brain = {"labels": ["environment brain warming up..."], "at": "", "src": ""}
lock = threading.Lock()
latest = {"jpg": None, "seen": "nothing detected yet"}


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def nvidia_describe(jpg, model, prompt):
    b64 = base64.b64encode(jpg).decode()
    r = requests.post(
        NVIDIA_URL,
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json"},
        json={"model": model,
              "messages": [{"role": "user",
                            "content": f'<img src="data:image/jpeg;base64,{b64}" /> {prompt}'}],
              "max_tokens": 400, "temperature": 0.1},
        timeout=25)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def claude_describe(jpg, prompt):
    tmp = Path("/tmp/brain_frame.jpg")
    tmp.write_bytes(jpg)
    r = subprocess.run(["claude", "-p", "--model", "haiku", "--allowedTools", "Read"],
                       input=f"Read the image file at {tmp}.\n{prompt}",
                       capture_output=True, text=True, timeout=90)
    return r.stdout


def brain_thread():
    fails = 0
    while True:
        jpg = latest["jpg"]
        if jpg is not None:
            prompt = brain_prompt(latest["seen"])
            raw = src = None
            if NVIDIA_API_KEY:
                for model in NVIDIA_MODELS:
                    try:
                        raw = nvidia_describe(jpg, model, prompt)
                        src = model.split("/")[-1].split("-")[0]
                        break
                    except Exception:
                        continue
            if raw is None:
                try:
                    raw, src = claude_describe(jpg, prompt), "claude"
                except Exception:
                    raw = None
            labels = None
            if raw:
                try:
                    s, e = raw.find("{"), raw.rfind("}")
                    labels = json.loads(raw[s:e + 1]).get("labels", [])[:8]
                except (json.JSONDecodeError, ValueError):
                    labels = None
            if labels:
                fails = 0
                with lock:
                    brain.update(labels=labels, at=f"{datetime.now():%H:%M:%S}", src=src)
            else:
                fails += 1
                with lock:
                    brain.update(src=f"retrying x{fails}")
        time.sleep(BRAIN_SEC if fails == 0 else min(BRAIN_SEC * (fails + 1), 40))


def start_source():
    return subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error",
         "-fflags", "nobuffer", "-flags", "low_delay", "-analyzeduration", "0",
         "-probesize", "500000", "-i", SRC,
         "-vf", VFILTER, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE)


def events_csv():
    p = OUT / "events.csv"
    if not p.exists():
        with p.open("w", newline="") as f:
            csv.writer(f).writerow(["time", "id", "class", "crop"])
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("loading YOLO11n...")
    model = YOLO("yolo11n.pt")
    log(f"camera brain on {args.camera}: all-object tracking + environment narration")
    threading.Thread(target=brain_thread, daemon=True).start()

    src = start_source()
    sink = subprocess.Popen(
        ["ffplay", "-loglevel", "error", "-fflags", "nobuffer",
         "-window_title", f"{args.camera} camera brain", "-f", "mjpeg", "-i", "-"],
        stdin=subprocess.PIPE)

    hist, ages, saved = {}, {}, set()
    nbytes = W * H * 3

    try:
        while sink.poll() is None:
            buf = src.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                log("stream ended; reconnecting in 5s")
                src.kill()
                time.sleep(5)
                src = start_source()
                continue
            frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
            if ZOOM > 1.0:
                cw, ch = int(W / ZOOM), int(H / ZOOM)
                cx = int(W * CX_PCT / 100); cy = int(H * CY_PCT / 100)
                x1z = min(max(0, cx - cw // 2), W - cw)
                y1z = min(max(0, cy - ch // 2), H - ch)
                frame = frame[y1z:y1z + ch, x1z:x1z + cw]
            now = time.time()

            # feed the brain a downscaled copy (NIM inline payload limit)
            small = cv2.resize(frame, (800, 450))
            ok, bjpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                latest["jpg"] = bjpg.tobytes()

            # YOLO on ALL classes
            # imgsz 960: better recall on small/distant people (costs ~2x CPU)
            r = model.track(frame, persist=True, verbose=False, conf=CONF,
                            imgsz=960, tracker="bytetrack.yaml")[0]

            if r.boxes is not None and r.boxes.id is not None:
                for box, tid, cls in zip(r.boxes.xyxy.numpy(),
                                         r.boxes.id.int().numpy(),
                                         r.boxes.cls.int().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    name = r.names[int(cls)]
                    cxp, cyp = (x1 + x2) / 2, (y1 + y2) / 2
                    h = hist.setdefault(tid, deque(maxlen=40))
                    h.append((now, cxp, cyp))
                    ages[tid] = ages.get(tid, 0) + 1

                    past = [(t, x, y) for t, x, y in h if now - t <= MOVE_WINDOW]
                    disp_px = (np.hypot(past[-1][1] - past[0][1],
                                        past[-1][2] - past[0][2]) if len(past) > 1 else 0)
                    moving = disp_px > MOVE_PX
                    state = "moving" if moving else "stagnant"
                    color = (60, 220, 60) if moving else (200, 200, 60)

                    if tid not in saved and ages[tid] >= CROP_MIN_AGE and int(cls) in CROP_CLASSES:
                        saved.add(tid)
                        pad = int((x2 - x1) * 0.15)
                        crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                        cpath = OUT / f"id{tid}_{name}.jpg"
                        cv2.imwrite(str(cpath), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        with events_csv().open("a", newline="") as f:
                            csv.writer(f).writerow(
                                [f"{datetime.now():%H:%M:%S}", int(tid), name, cpath.name])
                        log(f"NEW {name} #{tid} ({int(x2-x1)}x{int(y2-y1)}px) -> {cpath.name}")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name} #{tid} {state}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # ground the brain in what YOLO actually found this frame
            if r.boxes is not None and r.boxes.cls is not None and len(r.boxes.cls):
                counts = {}
                for c in r.boxes.cls.int().numpy():
                    counts[r.names[int(c)]] = counts.get(r.names[int(c)], 0) + 1
                latest["seen"] = ", ".join(f"{v}x {k}" for k, v in
                                           sorted(counts.items(), key=lambda kv: -kv[1]))
            else:
                latest["seen"] = "no objects detected"

            for tid in [t for t, hh in hist.items() if now - hh[-1][0] > 10]:
                hist.pop(tid, None)
                ages.pop(tid, None)

            disp = cv2.resize(frame, (DISP_W, DISP_H)) if frame.shape[1] != DISP_W else frame
            n_live = len([1 for hh in hist.values() if now - hh[-1][0] < 2])
            cv2.putText(disp, f"live objects: {n_live}  captured today: {len(saved)}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 220, 120), 2, cv2.LINE_AA)

            canvas = np.zeros((DISP_H + BAND_H, DISP_W, 3), dtype=np.uint8)
            canvas[:DISP_H] = disp
            with lock:
                labels, at, bsrc = brain["labels"], brain["at"], brain["src"]
            header = f"environment brain [{bsrc}] {at}" if at else "environment brain (warming up)"
            cv2.putText(canvas, header, (12, DISP_H + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 120), 1, cv2.LINE_AA)
            for i, lab in enumerate(labels):
                col = 12 if i < 4 else DISP_W // 2 + 12
                row = DISP_H + 42 + (i % 4) * 20
                cv2.putText(canvas, f"- {lab[:72]}", (col, row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            ok, out = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                try:
                    sink.stdin.write(out.tobytes())
                    sink.stdin.flush()
                except BrokenPipeError:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        src.kill()
        sink.kill()


if __name__ == "__main__":
    main()
