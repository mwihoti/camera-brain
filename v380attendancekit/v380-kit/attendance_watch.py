#!/usr/bin/env python3
"""Room attendance tracking with face recognition for the V380 camera.

Watches the camera at 1 frame/second. Every face seen is embedded with
OpenCV's SFace model and matched against a gallery of known faces:

  - A NEW face gets a persistent ID (person_001, ...) and its snapshot is
    saved once to faces/ -- the same person is never registered twice.
  - A KNOWN face just refreshes that person's "last seen" time.

A person is "inside" from the moment they are first seen until they have
not been seen for EXIT_TIMEOUT seconds. From that, the tracker derives:
total people who entered, how many are inside right now, peak attendance
and when it happened, when the room first filled and when it emptied, and
each person's stay duration.

All face processing is local (OpenCV on CPU) -- no cloud calls. Only the
optional --summary narrative uses the `claude` CLI.

Files (under ~/Videos/camera/attendance/):
    faces/person_NNN.jpg + .npy      face snapshot + embedding, one per person
    sessions-YYYY-MM-DD.csv          person_id, entry, exit, minutes
    occupancy-YYYY-MM-DD.csv         time, occupancy, event  (+id / -id)
    state.json                       live state (survives restarts)

Usage:
    attendance_watch.py              run the tracker (or camera-attendance.service)
    attendance_watch.py --test       one frame: detect + identify faces, print result
    attendance_watch.py --status     who is inside right now
    attendance_watch.py --report     today's stats (totals, peak, average stay)
    attendance_watch.py --summary    --report + a Claude-written event narrative
"""

import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---- configuration ---------------------------------------------------------
RTSP = "rtsp://admin:@192.168.1.111:554/live/ch00_0"
W, H = 1280, 720

KIT = Path(__file__).resolve().parent
DET_MODEL = KIT / "models" / "face_detection_yunet_2023mar.onnx"
REC_MODEL = KIT / "models" / "face_recognition_sface_2021dec.onnx"

BASE = Path.home() / "Videos" / "camera" / "attendance"
FACES = BASE / "faces"
STATE = BASE / "state.json"

FPS = 1                    # analysis frame rate
EXIT_TIMEOUT = 180         # unseen for this many seconds => counted as exited
MATCH_THRESHOLD = 0.363    # SFace cosine similarity for "same person"
MIN_SCORE = 0.85           # detector confidence to consider a face at all
REGISTER_SCORE = 0.9       # confidence needed to register a NEW person
REGISTER_MIN_W = 48        # min face width (px) to register a NEW person
# ----------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def hhmm(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# ---- models / gallery ------------------------------------------------------

def load_models():
    det = cv2.FaceDetectorYN.create(str(DET_MODEL), "", (W, H), MIN_SCORE, 0.3, 50)
    rec = cv2.FaceRecognizerSF.create(str(REC_MODEL), "")
    return det, rec


def load_gallery() -> dict:
    return {p.stem: np.load(p) for p in FACES.glob("person_*.npy")}


def next_person_id(gallery: dict) -> str:
    nums = [int(pid.split("_")[1]) for pid in gallery] or [0]
    return f"person_{max(nums) + 1:03d}"


def best_match(rec, gallery: dict, feat):
    best_pid, best_sim = None, MATCH_THRESHOLD
    for pid, known in gallery.items():
        sim = rec.match(feat, known, cv2.FaceRecognizerSF_FR_COSINE)
        if sim >= best_sim:
            best_pid, best_sim = pid, sim
    return best_pid


def register(gallery: dict, feat, img, box) -> str:
    pid = next_person_id(gallery)
    gallery[pid] = feat
    np.save(FACES / f"{pid}.npy", feat)
    x, y, w, h = (max(0, int(v)) for v in box[:4])
    pad = int(w * 0.3)
    crop = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    cv2.imwrite(str(FACES / f"{pid}.jpg"), crop)
    return pid


# ---- persistence -----------------------------------------------------------

def sessions_csv() -> Path:
    p = BASE / f"sessions-{datetime.now():%Y-%m-%d}.csv"
    if not p.exists():
        with p.open("w", newline="") as f:
            csv.writer(f).writerow(["person_id", "entry", "exit", "minutes"])
    return p


def occupancy_csv() -> Path:
    p = BASE / f"occupancy-{datetime.now():%Y-%m-%d}.csv"
    if not p.exists():
        with p.open("w", newline="") as f:
            csv.writer(f).writerow(["time", "occupancy", "event"])
    return p


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"inside": {}, "peak": 0, "peak_time": None}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state))


def occupancy_event(state: dict, event: str) -> None:
    n = len(state["inside"])
    with occupancy_csv().open("a", newline="") as f:
        csv.writer(f).writerow([f"{datetime.now():%H:%M:%S}", n, event])
    if n > state["peak"]:
        state["peak"] = n
        state["peak_time"] = f"{datetime.now():%H:%M:%S}"


def close_session(state: dict, pid: str) -> None:
    rec = state["inside"].pop(pid)
    minutes = (rec["last_seen"] - rec["first_seen"]) / 60
    with sessions_csv().open("a", newline="") as f:
        csv.writer(f).writerow(
            [pid, hhmm(rec["first_seen"]), hhmm(rec["last_seen"]), f"{minutes:.1f}"])
    occupancy_event(state, f"-{pid}")
    log(f"EXIT  {pid} (stayed {minutes:.0f} min, {len(state['inside'])} inside)")


# ---- main watcher ----------------------------------------------------------

def start_ffmpeg() -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
         "-i", RTSP, "-vf", f"hflip,vflip,fps={FPS}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE)


def process_frame(img, det, rec, gallery, state):
    now = time.time()
    _, faces = det.detect(img)
    for face in (faces if faces is not None else []):
        score, fw = float(face[-1]), float(face[2])
        feat = rec.feature(rec.alignCrop(img, face))
        pid = best_match(rec, gallery, feat)
        if pid is None:
            if score < REGISTER_SCORE or fw < REGISTER_MIN_W:
                continue  # too blurry/small to safely create a new identity
            pid = register(gallery, feat, img, face)
            log(f"NEW   {pid} registered (score {score:.2f})")
        if pid in state["inside"]:
            state["inside"][pid]["last_seen"] = now
        else:
            state["inside"][pid] = {"first_seen": now, "last_seen": now}
            occupancy_event(state, f"+{pid}")
            log(f"ENTRY {pid} ({len(state['inside'])} inside)")
    for pid in list(state["inside"]):
        if now - state["inside"][pid]["last_seen"] > EXIT_TIMEOUT:
            close_session(state, pid)


def watch() -> None:
    det, rec = load_models()
    gallery = load_gallery()
    state = load_state()
    # close sessions left over from a previous run that are now stale
    for pid in list(state["inside"]):
        if time.time() - state["inside"][pid]["last_seen"] > EXIT_TIMEOUT:
            close_session(state, pid)
    log(f"attendance tracker started ({len(gallery)} known faces, "
        f"{len(state['inside'])} inside, exit timeout {EXIT_TIMEOUT}s)")

    proc = start_ffmpeg()
    nbytes = W * H * 3
    last_save = 0.0
    while True:
        buf = proc.stdout.read(nbytes)
        if not buf or len(buf) < nbytes:
            log("ffmpeg stream ended; restarting in 10s")
            proc.kill()
            save_state(state)
            time.sleep(10)
            proc = start_ffmpeg()
            continue
        img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        process_frame(img, det, rec, gallery, state)
        if time.time() - last_save > 30:
            last_save = time.time()
            save_state(state)


# ---- reporting -------------------------------------------------------------

def gather_stats() -> str:
    state = load_state()
    sessions = []
    sp = BASE / f"sessions-{datetime.now():%Y-%m-%d}.csv"
    if sp.exists():
        with sp.open() as f:
            sessions = list(csv.DictReader(f))
    inside = state["inside"]
    entered_ids = {s["person_id"] for s in sessions} | set(inside)
    total_entries = len(sessions) + len(inside)
    stays = [float(s["minutes"]) for s in sessions]
    avg_stay = sum(stays) / len(stays) if stays else 0.0

    first_in = last_empty = "-"
    op = BASE / f"occupancy-{datetime.now():%Y-%m-%d}.csv"
    if op.exists():
        with op.open() as f:
            rows = list(csv.DictReader(f))
        entries = [r for r in rows if r["event"].startswith("+")]
        empties = [r for r in rows if r["occupancy"] == "0"]
        if entries:
            first_in = entries[0]["time"]
        if empties and not inside:
            last_empty = empties[-1]["time"]

    lines = [
        f"Attendance report — {datetime.now():%A %Y-%m-%d %H:%M}",
        f"  unique people entered today : {len(entered_ids)}",
        f"  total entries (with returns): {total_entries}",
        f"  inside right now            : {len(inside)}  "
        f"({', '.join(inside) if inside else 'room empty'})",
        f"  peak attendance             : {state['peak']}"
        + (f" at {state['peak_time']}" if state["peak_time"] else ""),
        f"  room first occupied         : {first_in}",
        f"  room emptied                : {last_empty}",
        f"  average stay (completed)    : {avg_stay:.1f} min",
    ]
    if sessions:
        lines.append("  completed visits:")
        for s in sessions:
            lines.append(f"    {s['person_id']}: {s['entry']} -> {s['exit']}"
                         f" ({s['minutes']} min)")
    return "\n".join(lines)


def status() -> None:
    state = load_state()
    if not state["inside"]:
        print("room is empty")
        return
    print(f"{len(state['inside'])} inside:")
    for pid, rec in state["inside"].items():
        mins = (time.time() - rec["first_seen"]) / 60
        print(f"  {pid}: since {hhmm(rec['first_seen'])} ({mins:.0f} min)")


def summary() -> None:
    stats = gather_stats()
    print(stats + "\n")
    prompt = (
        "Write a short narrative summary (4-6 sentences) of this event/room "
        "attendance data: the flow of the day, busiest period, notable "
        "patterns. Respond with ONLY the summary.\n\n" + stats
    )
    try:
        r = subprocess.run(["claude", "-p", "--model", "haiku"],
                           input=prompt, capture_output=True, text=True, timeout=240)
        print("Event summary:\n" + r.stdout.strip())
    except subprocess.TimeoutExpired:
        print("(claude summary timed out — stats above are complete)")


def test_once() -> None:
    det, rec = load_models()
    gallery = load_gallery()
    out = Path("/tmp/attendance_test.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
         "-i", RTSP, "-frames:v", "1", "-vf", "hflip,vflip", str(out)],
        timeout=30, check=True)
    img = cv2.imread(str(out))
    _, faces = det.detect(img)
    n = 0 if faces is None else len(faces)
    print(f"frame: {out} — {n} face(s) detected")
    for face in (faces if faces is not None else []):
        feat = rec.feature(rec.alignCrop(img, face))
        pid = best_match(rec, gallery, feat)
        print(f"  score={float(face[-1]):.2f} width={int(face[2])}px -> "
              + (f"known: {pid}" if pid else "new/unknown person"))


def main() -> None:
    for d in (FACES,):
        d.mkdir(parents=True, exist_ok=True)
    if "--test" in sys.argv:
        test_once()
    elif "--status" in sys.argv:
        status()
    elif "--report" in sys.argv:
        print(gather_stats())
    elif "--summary" in sys.argv:
        summary()
    else:
        watch()


if __name__ == "__main__":
    main()
