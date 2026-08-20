#!/usr/bin/env python3
"""Live AI-labeled view for either camera.

Plays the live feed in a window with a caption band listing what the AI
currently sees. NVIDIA NIM (nemotron VL) is the primary analyzer with a
circuit breaker; the `claude` CLI is the fallback.

Usage:
    live_detect.py                          # V380 bulb cam, traffic scene
    live_detect.py --camera mevo            # Mevo Core (SRT), room scene
    live_detect.py --camera mevo --scene traffic
Quit: q in the window / Ctrl-C.
"""

import argparse
import base64
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

# ---- camera presets ---------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--camera", choices=["v380", "mevo"], default="v380")
ap.add_argument("--scene", choices=["traffic", "room"], default=None)
args = ap.parse_args()

if args.camera == "mevo":
    SRC = "srt://192.168.2.159:4201?mode=caller&latency=50000"   # 50ms SRT buffer
    W, H = 1280, 720          # display size (scaled down from 1080p)
    VFILTER = "fps=2,scale=1280:720"   # right side up already
    SCENE = args.scene or "room"
else:
    SRC = "rtsp://admin:@192.168.1.111:554/live/ch00_1"   # V380 substream
    W, H = 640, 360
    VFILTER = "hflip,vflip,fps=2"      # V380 is mounted upside down
    SCENE = args.scene or "traffic"

BAND_H = 96
DETECT_W, DETECT_H = 800, 450   # detection frames stay under NIM's ~180KB cap

_envfile = Path.home() / ".config" / "camera-agent.env"
if "NVIDIA_API_KEY" not in os.environ and _envfile.exists():
    for line in _envfile.read_text().splitlines():
        if line.startswith("NVIDIA_API_KEY=") and len(line.split("=", 1)[1]) > 5:
            os.environ["NVIDIA_API_KEY"] = line.split("=", 1)[1].strip()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
# Tried in order per cycle; cosmos activates automatically once the NVIDIA
# account can invoke it (open build.nvidia.com/nvidia/cosmos-reason2-8b once).
NVIDIA_MODELS = ["nvidia/nemotron-nano-12b-v2-vl", "nvidia/cosmos-reason2-8b"]
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DETECT_SEC = 3 if NVIDIA_API_KEY else 10

PROMPTS = {
    "traffic": (
        "This is an aerial street camera view where objects appear small. Look "
        "carefully and report: 1) vehicles moving on the road (type+color), "
        "2) parked vehicles (count+colors), 3) people or motorcycles anywhere. "
        'Reply ONLY JSON: {"labels": ["...", "..."]} — one concrete observation '
        "per label with counts and colors, up to 5 labels. Report every vehicle "
        "and person you can find, even tiny or distant ones."
    ),
    "room": (
        "These are TWO frames from an indoor camera taken a few seconds apart: "
        "the FIRST image is earlier, the SECOND is now. Compare them and "
        "describe what is HAPPENING: 1) each person and what they are DOING "
        "(use action verbs — infer motion from what changed between frames), "
        "2) anything that moved or changed, 3) notable context. "
        'Reply ONLY JSON: {"labels": ["...", "..."]} — up to 5 labels, actions '
        'first. If frames are identical and nobody is present, say '
        '["room is still, no one present"].'
    ),
}
PROMPT = PROMPTS[SCENE]
PAIR = SCENE == "room"          # room activity needs two frames to see motion
if PAIR:
    DETECT_W, DETECT_H = 640, 360   # two images must share the payload budget
# ----------------------------------------------------------------------------

state = {"labels": ["starting..."], "at": "", "src": ""}
lock = threading.Lock()
latest = {"jpg": None, "prev": None}   # prev = frame from ~6s earlier (pair mode)


def detect_nvidia(jpg: bytes, prev: bytes = None, model: str = None):
    imgs = ""
    if prev is not None:
        imgs += f'<img src="data:image/jpeg;base64,{base64.b64encode(prev).decode()}" /> '
    imgs += f'<img src="data:image/jpeg;base64,{base64.b64encode(jpg).decode()}" />'
    r = requests.post(
        NVIDIA_URL,
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": f"{imgs} {PROMPT}"}],
              "max_tokens": 250, "temperature": 0.1},
        timeout=20)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], model.split("/")[-1].split("-")[0]


def detect_claude(jpg: bytes, prev: bytes = None):
    tmp = Path("/tmp/live_detect_frame.jpg")
    tmp.write_bytes(jpg)
    files = f"the image file {tmp}"
    if prev is not None:
        tmp_prev = Path("/tmp/live_detect_prev.jpg")
        tmp_prev.write_bytes(prev)
        files = f"BOTH image files: first {tmp_prev} (earlier), then {tmp} (now)"
    r = subprocess.run(
        ["claude", "-p", "--model", "haiku", "--allowedTools", "Read"],
        input=f"Read {files}.\n{PROMPT}",
        capture_output=True, text=True, timeout=90)
    return r.stdout, "claude"


def detector() -> None:
    fails = 0
    nvidia_fails = 0
    nvidia_skip_until = 0.0   # circuit breaker: rest NVIDIA after repeated failures
    while True:
        jpg = latest["jpg"]
        if jpg is not None:
            prev = latest["prev"] if PAIR else None
            raw = src = None
            if NVIDIA_API_KEY and time.time() >= nvidia_skip_until:
                for model in NVIDIA_MODELS:
                    try:
                        raw, src = detect_nvidia(jpg, prev, model)
                        nvidia_fails = 0
                        break
                    except Exception:
                        continue
                if raw is None:
                    nvidia_fails += 1
                    if nvidia_fails >= 2:   # NVIDIA is down: use claude for 5 min
                        nvidia_skip_until = time.time() + 300
            if raw is None:
                try:
                    raw, src = detect_claude(jpg, prev)
                except Exception:
                    raw = None
            labels = None
            if raw:
                try:
                    s, e = raw.find("{"), raw.rfind("}")
                    labels = json.loads(raw[s:e + 1]).get("labels", [])[:5]
                except (json.JSONDecodeError, ValueError):
                    labels = None
            if labels:
                fails = 0
                with lock:
                    state.update(labels=labels, at=f"{datetime.now():%H:%M:%S}", src=src)
            else:
                fails += 1
                with lock:
                    state.update(src=f"retrying x{fails}")   # keep last good labels
        time.sleep(DETECT_SEC if fails == 0 else min(DETECT_SEC * (fails + 1), 30))


def annotate(frame):
    canvas = np.zeros((H + BAND_H, W, 3), dtype=np.uint8)
    canvas[:H] = frame
    with lock:
        labels, at, src = state["labels"], state["at"], state["src"]
    header = f"AI view [{src}] {at}" if at else "AI view (warming up)"
    cv2.putText(canvas, header, (12, H + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 120), 1, cv2.LINE_AA)
    for i, lab in enumerate(labels):
        col = 12 if i < 3 else W // 2 + 12
        row = H + 44 + (i % 3) * 22
        cv2.putText(canvas, f"- {lab[:70]}", (col, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def start_source() -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error",
         "-fflags", "nobuffer", "-flags", "low_delay", "-analyzeduration", "0",
         "-probesize", "500000", "-i", SRC,
         "-vf", VFILTER, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE)


def main() -> None:
    mode = f"NVIDIA {' -> '.join(m.split('/')[-1] for m in NVIDIA_MODELS)}" if NVIDIA_API_KEY else "claude haiku"
    print(f"live view: {args.camera} / {SCENE} scene / {mode}. Quit: q / Ctrl-C")
    threading.Thread(target=detector, daemon=True).start()

    src = start_source()
    sink = subprocess.Popen(
        ["ffplay", "-loglevel", "error", "-fflags", "nobuffer",
         "-window_title", f"{args.camera} AI view", "-f", "mjpeg", "-i", "-"],
        stdin=subprocess.PIPE)

    nbytes = W * H * 3
    try:
        while sink.poll() is None:
            buf = src.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                print("stream ended; reconnecting in 5s")
                src.kill()
                time.sleep(5)
                src = start_source()
                continue
            frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            det = cv2.resize(frame, (DETECT_W, DETECT_H)) if W > DETECT_W else frame
            ok, jpg = cv2.imencode(".jpg", det, [cv2.IMWRITE_JPEG_QUALITY, 75 if PAIR else 78])
            if ok:
                now = time.time()
                # promote current to prev every ~6s so the pair spans real motion
                if latest["jpg"] is not None and now - getattr(main, "_pt", 0) > 6:
                    latest["prev"] = latest["jpg"]
                    main._pt = now
                latest["jpg"] = jpg.tobytes()
            ok, out = cv2.imencode(".jpg", annotate(frame), [cv2.IMWRITE_JPEG_QUALITY, 80])
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
