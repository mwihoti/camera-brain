#!/usr/bin/env python3
"""Telegram bridge for Camera Brain.

Sends an alert (photo + caption) to your Telegram whenever the tracker logs
a new person/vehicle, and answers commands:

    /snap     current live view (the tracker's annotated frame)
    /clip [s] record a short video (default 8s, 3-10s) and send it
    /report   today's numbers (unique objects, last events)
    /status   what's running
    /help     this list

Setup (once):
  1. In Telegram, talk to @BotFather -> /newbot -> copy the token.
  2. Add to ~/.config/camera-agent.env:   TELEGRAM_BOT_TOKEN=123456:ABC...
  3. Run this script, then send /start to your bot in Telegram --
     it captures your chat id automatically and saves it.

Run:  .venv/bin/python telegram_watch.py   (or add a systemd service)
"""

import csv
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

ENVFILE = Path.home() / ".config" / "camera-agent.env"


def load_env():
    if ENVFILE.exists():
        for line in ENVFILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
API = f"https://api.telegram.org/bot{TOKEN}"

# Server mode: CAMERA_DATA_DIR is the dir track_live.py writes to on the same
# server (run_server.sh sets it). Default = local laptop paths.
_DATA = os.environ.get("CAMERA_DATA_DIR", "")
if _DATA:
    TRACKS = Path(_DATA) / "tracks"
    LIVE_FRAME = Path(_DATA) / "camera_live.jpg"
else:
    TRACKS = Path.home() / "Videos" / "camera" / "tracks"
    LIVE_FRAME = Path("/tmp/camera_live.jpg")   # written by track_live.py

# /clip records straight from the camera stream (run_server.sh exports
# CAMERA_SRC), so the clip is clean video, not the annotated frames.
CLIP_SRC = os.environ.get("CAMERA_SRC", "")
CLIP_VF = os.environ.get("CLIP_VF", "hflip,vflip")   # camera is mounted upside down
CLIP_DEFAULT, CLIP_MIN, CLIP_MAX = 8, 3, 10
_clip_lock = threading.Lock()


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def send_text(chat, text):
    requests.post(f"{API}/sendMessage", json={"chat_id": chat, "text": text}, timeout=20)


def send_photo(chat, path, caption=""):
    with open(path, "rb") as f:
        requests.post(f"{API}/sendPhoto", data={"chat_id": chat, "caption": caption[:1000]},
                      files={"photo": f}, timeout=60)


def send_video(chat, path, caption=""):
    with open(path, "rb") as f:
        requests.post(f"{API}/sendVideo",
                      data={"chat_id": chat, "caption": caption[:1000], "supports_streaming": "true"},
                      files={"video": f}, timeout=120)


def record_clip(chat, secs):
    """Record `secs` from the camera stream with ffmpeg and send it. One at a time."""
    if not _clip_lock.acquire(blocking=False):
        send_text(chat, "Already recording a clip - try again in a few seconds.")
        return
    out = Path(tempfile.gettempdir()) / f"camera_clip_{datetime.now():%H%M%S}.mp4"
    try:
        send_text(chat, f"Recording {secs}s...")
        # -timeout (us): give up fast if the tunnel/stream stalls instead of hanging
        cmd = ["ffmpeg", "-loglevel", "error", "-rtsp_transport", "tcp", "-timeout", "10000000"] \
            if CLIP_SRC.startswith("rtsp") else ["ffmpeg", "-loglevel", "error", "-rw_timeout", "10000000"]
        cmd += ["-i", CLIP_SRC, "-t", str(secs), "-vf", CLIP_VF + ",scale='min(1280,iw)':-2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                "-an", "-movflags", "+faststart", "-y", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=secs + 40)
        if r.returncode != 0 or not out.exists() or out.stat().st_size < 1000:
            log(f"clip failed: {r.stderr.strip()[-200:]}")
            send_text(chat, "Couldn't record - is the camera stream up? (/status)")
            return
        send_video(chat, out, f"clip {secs}s - {datetime.now():%H:%M:%S}")
        log(f"clip sent: {secs}s, {out.stat().st_size // 1024} KB")
    except subprocess.TimeoutExpired:
        log("clip timed out (stream stalled)")
        send_text(chat, "Camera stream stalled - the laptop bridge may have dropped. Try again in a moment (/status).")
    except Exception as e:
        log(f"clip error: {str(e)[:120]}")
        send_text(chat, "Clip failed - check the server logs.")
    finally:
        out.unlink(missing_ok=True)
        _clip_lock.release()


def today_dir():
    return TRACKS / f"{datetime.now():%Y-%m-%d}"


def report_text():
    p = today_dir() / "events.csv"
    if not p.exists():
        return "No events logged today yet."
    with p.open() as f:
        rows = list(csv.DictReader(f))
    kinds = {}
    for r in rows:
        kinds[r["class"]] = kinds.get(r["class"], 0) + 1
    lines = [f"Camera Brain report {datetime.now():%Y-%m-%d %H:%M}",
             f"unique objects today: {len(rows)}"]
    lines += [f"  {k}: {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])]
    lines.append("last events:")
    lines += [f"  {r['time']} {r['class']} #{r['id']}" for r in rows[-5:]]
    return "\n".join(lines)


def handle_command(chat, text):
    cmd = text.split()[0].lower().split("@")[0]
    if cmd == "/start":
        send_text(chat, "Camera Brain connected. Commands: /snap /clip /report /status /help")
    elif cmd == "/snap":
        if LIVE_FRAME.exists() and time.time() - LIVE_FRAME.stat().st_mtime < 30:
            send_photo(chat, LIVE_FRAME, f"live view {datetime.now():%H:%M:%S}")
        else:
            send_text(chat, "No fresh live frame - is track_live.py running?")
    elif cmd == "/clip":
        if not CLIP_SRC:
            send_text(chat, "/clip needs CAMERA_SRC (start via run_server.sh).")
            return
        arg = text.split()[1:2]
        secs = int(arg[0]) if arg and arg[0].isdigit() else CLIP_DEFAULT
        secs = max(CLIP_MIN, min(CLIP_MAX, secs))
        threading.Thread(target=record_clip, args=(chat, secs), daemon=True).start()
    elif cmd == "/report":
        send_text(chat, report_text())
    elif cmd == "/status":
        running = subprocess.run(["pgrep", "-f", "track_live.py"], capture_output=True).returncode == 0
        fresh = LIVE_FRAME.exists() and time.time() - LIVE_FRAME.stat().st_mtime < 30
        send_text(chat, f"tracker: {'RUNNING' if running else 'stopped'}\n"
                        f"live frame: {'fresh' if fresh else 'stale/none'}\n"
                        f"events today: {(report_text().splitlines() + ['none yet'])[1]}")
    else:
        send_text(chat, "Commands: /snap /clip /report /status")


def poll_commands():
    """Long-poll Telegram for messages; auto-capture chat id on first contact."""
    global CHAT_ID
    offset = 0
    while True:
        try:
            r = requests.get(f"{API}/getUpdates",
                             params={"timeout": 50, "offset": offset}, timeout=60).json()
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                msg = u.get("message") or {}
                chat = str(msg.get("chat", {}).get("id", ""))
                text = msg.get("text", "")
                if not chat:
                    continue
                if not CHAT_ID:
                    CHAT_ID = chat
                    with ENVFILE.open("a") as f:
                        f.write(f"TELEGRAM_CHAT_ID={chat}\n")
                    log(f"chat id captured: {chat}")
                if text.startswith("/"):
                    handle_command(chat, text)
        except Exception as e:
            log(f"poll error: {str(e)[:80]}")
            time.sleep(10)


def watch_events():
    """Tail today's events.csv; send each new row as a photo alert."""
    seen = set()
    first_pass = True
    while True:
        p = today_dir() / "events.csv"
        if p.exists():
            with p.open() as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                key = (r["time"], r["id"])
                if key in seen:
                    continue
                seen.add(key)
                if first_pass or not CHAT_ID:
                    continue  # don't replay history on startup
                crop = today_dir() / r["crop"]
                caption = f"NEW {r['class']} #{r['id']} at {r['time']}"
                try:
                    if crop.exists():
                        send_photo(CHAT_ID, crop, caption)
                    else:
                        send_text(CHAT_ID, caption)
                    log(f"alert sent: {caption}")
                except Exception as e:
                    log(f"alert failed: {str(e)[:80]}")
            first_pass = False
        time.sleep(3)


def main():
    if not TOKEN:
        raise SystemExit("Add TELEGRAM_BOT_TOKEN=... to ~/.config/camera-agent.env first "
                         "(get one from @BotFather in Telegram).")
    log("telegram bridge started"
        + (f" (chat {CHAT_ID})" if CHAT_ID else " - send /start to your bot to link"))
    threading.Thread(target=watch_events, daemon=True).start()
    poll_commands()


if __name__ == "__main__":
    main()
