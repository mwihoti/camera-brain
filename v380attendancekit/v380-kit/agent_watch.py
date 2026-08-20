#!/usr/bin/env python3
"""AI camera agent for the V380 camera.

Every POLL_SEC seconds it grabs a frame from the camera's low-res substream
and measures motion by diffing a tiny grayscale copy against the previous
one. When motion is detected it sends the frame to Claude (via the `claude`
CLI, which uses your Claude subscription -- no API key needed) and appends a
timestamped observation to a daily Markdown log. Every SUMMARY_EVERY seconds
it asks Claude to write a rolling summary of the day so far.

Video recording is NOT done here -- record.sh handles that. This script also
prunes recordings older than RETENTION_DAYS.

Outputs:
    ~/Videos/camera/log/YYYY-MM-DD.md   -- observations + summaries (read this)
    ~/Videos/camera/frames/latest.jpg   -- most recent captured frame
    ~/Videos/camera/recordings/*.mp4    -- written by record.sh
"""

import base64
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

# ---- configuration ---------------------------------------------------------
# Frames are grabbed from the 720p main stream (the camera's max) so the AI
# gets the clearest possible image; grabbing 1 frame / 15s costs nothing.
RTSP_SUB = "rtsp://admin:@192.168.1.111:554/live/ch00_0"

BASE = Path.home() / "Videos" / "camera"
FRAME_JPG = BASE / "frames" / "latest.jpg"
FRAME_GRAY = BASE / "frames" / "latest.gray"
LOG_DIR = BASE / "log"
REC_DIR = BASE / "recordings"

POLL_SEC = 15             # how often to grab a frame
MOTION_THRESHOLD = 0.04   # mean per-pixel change (0..1) that counts as motion
MIN_ANALYZE_GAP = 60      # min seconds between AI analyses (cost control)
SUMMARY_EVERY = 3600      # seconds between rolling summaries
RETENTION_DAYS = 7        # recordings older than this are deleted
MODEL = "haiku"           # claude CLI model (used for summaries + fallback)

# NVIDIA NIM vision model for per-frame analysis. Key comes from the
# NVIDIA_API_KEY env var (set in ~/.config/camera-agent.env for systemd).
# If unset or the call fails, frame analysis falls back to the claude CLI.
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

GRAY_W, GRAY_H = 64, 36
# ----------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def grab_frame():
    """Capture one JPEG (for Claude) + one tiny gray frame (for motion).

    Returns the grayscale frame as an int16 numpy array, or None on failure.
    """
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-i", RTSP_SUB,
        # rotate 180 (camera mounted upside down) + gentle sharpen; q:v 2 = high quality JPEG
        "-frames:v", "1", "-q:v", "2",
        "-vf", "hflip,vflip,unsharp=5:5:0.6:3:3:0.3", str(FRAME_JPG),
        "-frames:v", "1", "-vf", f"hflip,vflip,scale={GRAY_W}:{GRAY_H},format=gray",
        "-f", "rawvideo", str(FRAME_GRAY),
    ]
    try:
        subprocess.run(cmd, timeout=90, check=True, capture_output=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        log(f"frame grab failed: {e}")
        return None
    data = FRAME_GRAY.read_bytes()
    if len(data) < GRAY_W * GRAY_H:
        return None
    return np.frombuffer(data[: GRAY_W * GRAY_H], dtype=np.uint8).astype(np.int16)


def logfile() -> Path:
    p = LOG_DIR / f"{datetime.now():%Y-%m-%d}.md"
    if not p.exists():
        p.write_text(f"# Camera log — {datetime.now():%A %Y-%m-%d}\n\n")
    return p


def recent_observations(n: int = 6) -> str:
    lines = [l for l in logfile().read_text().splitlines() if l.startswith("- **")]
    return "\n".join(lines[-n:]) or "(none yet)"


def run_claude(prompt: str, timeout: int = 180):
    try:
        # prompt goes via stdin; --allowedTools Read lets it open the frame image
        r = subprocess.run(
            ["claude", "-p", "--model", MODEL, "--allowedTools", "Read"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log("claude call timed out")
        return None
    out = r.stdout.strip()
    if r.returncode != 0 or not out:
        log(f"claude failed rc={r.returncode}: {r.stderr.strip()[:200]}")
        return None
    return out


def nvidia_describe(instructions: str):
    """Describe the current frame with an NVIDIA NIM vision model.

    NIM vision models take the image inline as an <img> data URI in the
    message text (inline limit ~180KB; our substream JPEGs are ~30KB).
    """
    if not NVIDIA_API_KEY:
        return None
    b64 = base64.b64encode(FRAME_JPG.read_bytes()).decode()
    try:
        r = requests.post(
            NVIDIA_URL,
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}",
                     "Accept": "application/json"},
            json={
                "model": NVIDIA_MODEL,
                "messages": [{
                    "role": "user",
                    "content": f'{instructions} <img src="data:image/jpeg;base64,{b64}" />',
                }],
                "max_tokens": 120,
                "temperature": 0.2,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip() or None
    except (requests.RequestException, KeyError, IndexError) as e:
        log(f"nvidia call failed, falling back to claude: {e}")
        return None


def analyze_frame(motion_score: float) -> None:
    instructions = (
        "This is a frame from a home security camera; motion was just "
        f"detected (score {motion_score:.2f}). Recent observations for "
        f"context:\n{recent_observations()}\n\n"
        "Respond with ONLY 1-2 plain sentences describing what is happening "
        "in the frame: people (how many, what they are doing, clothing), "
        "animals, vehicles, or objects that changed. If the frame shows "
        "nothing of interest (empty scene, lighting change), start your "
        "reply with 'Nothing notable'."
    )
    desc = nvidia_describe(instructions)
    src = "nvidia"
    if desc is None:
        desc = run_claude(
            f"You are a home security camera analyst. Read the image file at "
            f"{FRAME_JPG}.\n{instructions}"
        )
        src = "claude"
    if desc is None:
        return
    with logfile().open("a") as f:
        f.write(f"- **{datetime.now():%H:%M:%S}** (motion {motion_score:.2f}): {desc}\n")
    log(f"observation [{src}]: {desc[:100]}")


def write_summary() -> None:
    content = logfile().read_text()
    if "- **" not in content:
        return  # nothing happened yet today
    prompt = (
        "Below is today's activity log from a home security camera. Write a "
        "short summary (3-5 sentences) of what has happened so far today: "
        "notable events, how many times people appeared, any patterns. "
        "Respond with ONLY the summary text.\n\n" + content
    )
    summary = run_claude(prompt, timeout=240)
    if summary is None:
        return
    with logfile().open("a") as f:
        f.write(f"\n## Summary as of {datetime.now():%H:%M}\n\n{summary}\n\n")
    log("hourly summary written")


def prune_recordings() -> None:
    cutoff = time.time() - RETENTION_DAYS * 86400
    removed = 0
    for f in REC_DIR.glob("*.mp4"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    if removed:
        log(f"pruned {removed} recordings older than {RETENTION_DAYS} days")


def main() -> None:
    for d in (FRAME_JPG.parent, LOG_DIR, REC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    vision = NVIDIA_MODEL if NVIDIA_API_KEY else f"claude {MODEL} (no NVIDIA_API_KEY)"
    log(f"camera agent started (poll {POLL_SEC}s, motion>{MOTION_THRESHOLD}, vision: {vision})")
    prev = None
    last_analysis = 0.0
    last_summary = time.time()
    last_prune = 0.0

    while True:
        gray = grab_frame()
        if gray is not None:
            if prev is not None:
                score = float(np.abs(gray - prev).mean()) / 255.0
                if score > MOTION_THRESHOLD and time.time() - last_analysis > MIN_ANALYZE_GAP:
                    last_analysis = time.time()
                    analyze_frame(score)
            prev = gray

        if time.time() - last_summary > SUMMARY_EVERY:
            last_summary = time.time()
            write_summary()

        if time.time() - last_prune > 86400:
            last_prune = time.time()
            prune_recordings()

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
