# Camera Brain 🧠📹 — cameras that understand

Camera Brain turns affordable cameras into intelligent observers.
Started as a weekend experiment with a KSh 2,500 V380 bulb camera; now a
layered vision system running entirely on a laptop in Nairobi.

## What it does

- **Watches** — live feeds from a V380 bulb cam (hidden RTSP unlocked via an
  SD-card config) and a Logitech Mevo Core (SRT at 1080p)
- **Tracks** — YOLO11 + ByteTrack box every object with persistent IDs and
  measured moving/stagnant state, locally on CPU (~100ms/frame)
- **Understands** — NVIDIA vision models (Nemotron, Cosmos Reason) narrate
  the environment, *grounded by the local detector's real findings* to stop
  hallucinations; Claude as final fallback
- **Remembers** — daily activity logs, traffic CSVs (counts, types, colors,
  plates), face-ID attendance with occupancy and stay statistics
- **Survives** — circuit breakers, model fallback chains, stream auto-
  reconnect, remote camera reboot over ONVIF; built for weak Wi-Fi and
  flaky free-tier APIs

## Architecture

```
camera (RTSP/SRT) ──► motion gate (pixel math, free)
                      └─► YOLO11 tracker (local CPU: boxes, IDs, moving/stagnant)
                           ├─► full-res crops of new people/vehicles
                           └─► detections ground the VLM prompt
                                └─► NVIDIA Nemotron → Cosmos → Claude
                                     └─► environment narration + logs + CSVs
```

The design principle: **local models do the constant cheap work; cloud
models are called rarely, on evidence, for meaning.** ~90% fewer API calls
than a VLM-only design, and the detector supervises the narrator.

## Components (`v380attendancekit/v380-kit/`)

| File | Purpose |
|---|---|
| `track_live.py` | The camera brain: all-object tracking + grounded AI narration |
| `live_detect.py` | Live feed with VLM caption band (room/traffic scenes) |
| `traffic_watch.py` | Road analysis: vehicle counts/types/colors/plates → CSV |
| `attendance_watch.py` | Face-ID attendance: unique IDs, occupancy, peak, stay time |
| `agent_watch.py` | Motion-gated scene descriptions + hourly summaries |
| `record.sh` | 24/7 stream-copy recorder, 10-min MP4 segments, auto-retention |
| `ptz.py` | Terminal pan/tilt control over ONVIF |
| `ceshi.ini` | The SD-card file that unlocks the V380's hidden RTSP mode |

`mevo/README.md` documents the Mevo Core SRT integration; `PLAN.md` holds
the roadmap from prototype to product.

## Hardware it runs on

One consumer laptop (i5, no GPU), one KSh 2,500 bulb camera, one Mevo Core.
That's the point.

## Status

Active development — building in public. See `PLAN.md` for the roadmap:
tracker ✅, pilot deployment, Kenyan street dataset, edge boxes.
