#!/usr/bin/env python3
"""Telegram bridge for Camera Brain.

Sends an alert (photo + caption) to your Telegram whenever the tracker logs
a new person/vehicle, and answers commands:

    /snap     current live view (the tracker's annotated frame)
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

# Server mode: set CAMERA_DATA_DIR to a dir kept fresh by sync_to_server.sh
# (laptop rsyncs tracker data there). Default = local laptop paths.
_DATA = os.environ.get("CAMERA_DATA_DIR", "")
if _DATA:
    TRACKS = Path(_DATA) / "tracks"
    LIVE_FRAME = Path(_DATA) / "camera_live.jpg"
else:
    TRACKS = Path.home() / "Videos" / "camera" / "tracks"
    LIVE_FRAME = Path("/tmp/camera_live.jpg")   # written by track_live.py


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def send_text(chat, text):
    requests.post(f"{API}/sendMessage", json={"chat_id": chat, "text": text}, timeout=20)


def send_photo(chat, path, caption=""):
    with open(path, "rb") as f:
        requests.post(f"{API}/sendPhoto", data={"chat_id": chat, "caption": caption[:1000]},
                      files={"photo": f}, timeout=60)


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
        send_text(chat, "Camera Brain connected. Commands: /snap /report /status /help")
    elif cmd == "/snap":
        if LIVE_FRAME.exists() and time.time() - LIVE_FRAME.stat().st_mtime < 30:
            send_photo(chat, LIVE_FRAME, f"live view {datetime.now():%H:%M:%S}")
        else:
            send_text(chat, "No fresh live frame - is track_live.py running?")
    elif cmd == "/report":
        send_text(chat, report_text())
    elif cmd == "/status":
        running = subprocess.run(["pgrep", "-f", "track_live.py"], capture_output=True).returncode == 0
        fresh = LIVE_FRAME.exists() and time.time() - LIVE_FRAME.stat().st_mtime < 30
        send_text(chat, f"tracker: {'RUNNING' if running else 'stopped'}\n"
                        f"live frame: {'fresh' if fresh else 'stale/none'}\n"
                        f"events today: {report_text().splitlines()[1]}")
    else:
        send_text(chat, "Commands: /snap /report /status")


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
