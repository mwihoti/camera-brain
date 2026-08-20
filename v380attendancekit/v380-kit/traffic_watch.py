#!/usr/bin/env python3
"""Real-time traffic analysis from the V380 camera.

Reads the camera continuously at 1 frame/second. When motion is detected it
sends the frame to a vision model (NVIDIA NIM if NVIDIA_API_KEY is set,
otherwise the `claude` CLI) which counts vehicles, identifies type and
color, and attempts to read number plates. Every event with at least one
vehicle is saved as:

  - a snapshot JPEG:  ~/Videos/camera/traffic/snapshots/HH-MM-SS.jpg
  - a CSV row:        ~/Videos/camera/traffic/traffic-YYYY-MM-DD.csv
                      (time, vehicle count, types, colors, plates,
                       pedestrians, snapshot path, notes)

Usage:
    traffic_watch.py           run the watcher (or via camera-traffic.service)
    traffic_watch.py --test    analyze one frame right now and print the result
    traffic_watch.py --report  per-hour vehicle counts from today's CSV

Plate reading works when vehicles pass close enough that the plate is
legible in a 720p frame -- a gate or driveway view, not a distant road.
"""

import base64
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

# ---- configuration ---------------------------------------------------------
RTSP = "rtsp://admin:@192.168.1.111:554/live/ch00_1"  # 360p substream (lighter for weak Wi-Fi; main is ch00_0)

BASE = Path.home() / "Videos" / "camera" / "traffic"
SPOOL = BASE / "spool"          # rolling 1fps frames (auto-pruned)
SNAPSHOTS = BASE / "snapshots"  # one JPEG per vehicle event

MOTION_THRESHOLD = 0.03   # mean per-pixel change (0..1) that counts as motion
COOLDOWN = 8              # min seconds between AI analyses
SPOOL_KEEP_SEC = 600      # spool frames older than this are deleted
RETENTION_DAYS = 30       # snapshot retention

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
CLAUDE_MODEL = "haiku"

GRAY_W, GRAY_H = 64, 36
GRAY_BYTES = GRAY_W * GRAY_H

PROMPT = (
    "This is a frame from a traffic-monitoring camera. Analyze it and reply "
    "with ONLY a JSON object, no other text, in this exact shape:\n"
    '{"vehicles": [{"type": "car|truck|bus|motorcycle|bicycle|other", '
    '"color": "...", "plate": "characters or unreadable"}], '
    '"pedestrians": 0, "notes": "one short sentence"}\n'
    "List every vehicle visible. For plates, transcribe the characters only "
    "if they are actually legible; otherwise use \"unreadable\" - never guess. "
    "If there are no vehicles, use an empty vehicles list."
)
# ----------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def csv_path() -> Path:
    p = BASE / f"traffic-{datetime.now():%Y-%m-%d}.csv"
    if not p.exists():
        with p.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["time", "vehicles", "types", "colors", "plates",
                 "pedestrians", "snapshot", "notes"])
    return p


def start_ffmpeg() -> subprocess.Popen:
    """One ffmpeg: 1fps JPEGs into the spool + 1fps tiny gray frames on stdout."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
        "-i", RTSP,
        "-vf", "hflip,vflip,fps=1", "-q:v", "2",
        "-strftime", "1", str(SPOOL / "%Y%m%d_%H%M%S.jpg"),
        "-vf", f"hflip,vflip,fps=1,scale={GRAY_W}:{GRAY_H},format=gray",
        "-f", "rawvideo", "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def latest_spool_frame():
    frames = sorted(SPOOL.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    return frames[-1] if frames else None


def nvidia_describe(image: Path):
    if not NVIDIA_API_KEY:
        return None
    b64 = base64.b64encode(image.read_bytes()).decode()
    try:
        r = requests.post(
            NVIDIA_URL,
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}",
                     "Accept": "application/json"},
            json={"model": NVIDIA_MODEL,
                  "messages": [{"role": "user",
                                "content": f'{PROMPT} <img src="data:image/jpeg;base64,{b64}" />'}],
                  "max_tokens": 300, "temperature": 0.1},
            timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip() or None
    except (requests.RequestException, KeyError, IndexError) as e:
        log(f"nvidia call failed, falling back to claude: {e}")
        return None


def claude_describe(image: Path):
    prompt = f"Read the image file at {image}.\n{PROMPT}"
    try:
        r = subprocess.run(
            ["claude", "-p", "--model", CLAUDE_MODEL, "--allowedTools", "Read"],
            input=prompt, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        log("claude call timed out")
        return None
    out = r.stdout.strip()
    if r.returncode != 0 or not out:
        log(f"claude failed rc={r.returncode}: {r.stderr.strip()[:200]}")
        return None
    return out


def parse_json(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def analyze(frame: Path) -> None:
    raw = nvidia_describe(frame) or claude_describe(frame)
    if raw is None:
        return
    data = parse_json(raw)
    if data is None:
        log(f"unparseable model reply: {raw[:120]}")
        return
    vehicles = data.get("vehicles") or []
    if not vehicles:
        log(f"motion but no vehicles ({data.get('notes', '')[:60]})")
        return

    snap = SNAPSHOTS / f"{datetime.now():%H-%M-%S}.jpg"
    snap.write_bytes(frame.read_bytes())
    with csv_path().open("a", newline="") as f:
        csv.writer(f).writerow([
            f"{datetime.now():%H:%M:%S}",
            len(vehicles),
            "|".join(v.get("type", "?") for v in vehicles),
            "|".join(v.get("color", "?") for v in vehicles),
            "|".join(v.get("plate", "?") for v in vehicles),
            data.get("pedestrians", 0),
            str(snap),
            data.get("notes", ""),
        ])
    plates = [v.get("plate") for v in vehicles if v.get("plate") not in (None, "unreadable")]
    log(f"logged {len(vehicles)} vehicle(s)"
        + (f", plates: {', '.join(plates)}" if plates else "")
        + f" -> {snap.name}")


def prune() -> None:
    now = time.time()
    for f in SPOOL.glob("*.jpg"):
        if now - f.stat().st_mtime > SPOOL_KEEP_SEC:
            f.unlink()
    cutoff = now - RETENTION_DAYS * 86400
    for f in SNAPSHOTS.glob("*.jpg"):
        if f.stat().st_mtime < cutoff:
            f.unlink()


def watch() -> None:
    vision = NVIDIA_MODEL if NVIDIA_API_KEY else f"claude {CLAUDE_MODEL}"
    log(f"traffic watcher started (1 fps, motion>{MOTION_THRESHOLD}, vision: {vision})")
    proc = start_ffmpeg()
    prev = None
    last_analysis = 0.0
    last_prune = 0.0

    while True:
        chunk = proc.stdout.read(GRAY_BYTES)
        if not chunk or len(chunk) < GRAY_BYTES:
            log("ffmpeg stream ended; restarting in 10s")
            proc.kill()
            time.sleep(10)
            proc = start_ffmpeg()
            prev = None
            continue
        gray = np.frombuffer(chunk, dtype=np.uint8).astype(np.int16)
        if prev is not None:
            score = float(np.abs(gray - prev).mean()) / 255.0
            if score > MOTION_THRESHOLD and time.time() - last_analysis > COOLDOWN:
                frame = latest_spool_frame()
                if frame is not None:
                    last_analysis = time.time()
                    analyze(frame)
        prev = gray
        if time.time() - last_prune > 300:
            last_prune = time.time()
            prune()


def test_once() -> None:
    out = Path("/tmp/traffic_test.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
         "-i", RTSP, "-frames:v", "1", "-q:v", "2", "-vf", "hflip,vflip", str(out)],
        timeout=120, check=True)
    raw = nvidia_describe(out) or claude_describe(out)
    print(f"frame: {out}\nmodel reply:\n{raw}")
    data = parse_json(raw or "")
    print(f"\nparsed: {json.dumps(data, indent=2) if data else 'FAILED TO PARSE'}")


def report() -> None:
    p = BASE / f"traffic-{datetime.now():%Y-%m-%d}.csv"
    if not p.exists():
        sys.exit("no traffic CSV for today yet")
    hours = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            hours.setdefault(row["time"][:2], 0)
            hours[row["time"][:2]] += int(row["vehicles"])
    print(f"Vehicles per hour — {datetime.now():%Y-%m-%d}")
    for h in sorted(hours):
        print(f"  {h}:00  {'#' * min(hours[h], 60)} {hours[h]}")
    print(f"  total: {sum(hours.values())}")


def main() -> None:
    for d in (SPOOL, SNAPSHOTS):
        d.mkdir(parents=True, exist_ok=True)
    if "--test" in sys.argv:
        test_once()
    elif "--report" in sys.argv:
        report()
    else:
        watch()


if __name__ == "__main__":
    main()
