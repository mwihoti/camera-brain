# V380 Pro — Live Feed on Laptop + Activity Analysis

A starter kit to get your V380 bulb camera (model Q16S-1, V380 Pro app) streaming
to your laptop and automatically flagging activity.

## How it all connects (the mental model)

```
[Camera] --Wi-Fi--> [Your router] <--Wi-Fi/Ethernet-- [Your laptop]
   |                                                        |
   |  RTSP video stream (once enabled)                      |
   +----------------> rtsp://user:pass@CAMERA_IP:554/... --> watch.py (OpenCV)
                                                              |
                                              live view + motion/person detection
                                              + snapshots + clips + events.csv
```

The camera normally only talks to the **V380 Pro phone app** over a private
protocol. To reach it from a laptop you turn on its hidden **RTSP** mode, which
is the standard way cameras hand video to other software.

## Step 1 — Enable RTSP/ONVIF on the camera

Two methods; try the app one first.

**A. From the V380 Pro app (some firmware versions):**
Open the camera → **Settings (gear)** → look for **ONVIF** / **RTSP** /
"Third-party" / "LAN protocol" and switch it **ON**. Set/note a device username
and password (often `admin`). The label on your camera shows `User: admin` and a
`pwd:` value — that's your starting login.

**B. If there's no such menu (common on bulb cams) — SD-card trick:**
1. On a computer, create a plain text file named `ceshi.ini`.
2. Put exactly this inside:
   ```
   [CONST_PARAM]
   rtsp_enable=1
   onvif_enable=1
   ```
3. Copy it to the root of a microSD card, insert into the camera, power-cycle it.
4. The camera reads the flag on boot and enables RTSP.

## Step 2 — Find the camera's IP address

Both devices must be on the **same Wi-Fi**. Then either:
- Check your router's admin page → connected devices, or
- The V380 app sometimes shows it under device info, or
- Scan your network (replace with your subnet):
  ```
  # Windows:  arp -a
  # Mac/Linux: nmap -sn 192.168.1.0/24
  ```
Look for a device whose maker is the camera chipset. Note the IP, e.g. `192.168.1.50`.

## Step 3 — Build your RTSP URL

V380 cameras vary; try these paths in order until one shows video:
```
rtsp://admin:PASSWORD@192.168.1.50:554/live/ch00_0
rtsp://admin:PASSWORD@192.168.1.50:554/live/ch00_1     (sub-stream, lower res)
rtsp://admin:PASSWORD@192.168.1.50:554/onvif1
rtsp://admin:PASSWORD@192.168.1.50:554/11
```
Quick test with VLC: **Media → Open Network Stream** → paste a URL. If VLC plays
it, that exact URL works for the tool below.

## Step 4 — Run the tool on your laptop

```
pip install -r requirements.txt

# Live view + motion detection:
python watch.py --url "rtsp://admin:PASSWORD@192.168.1.50:554/live/ch00_0"

# People detection (first install the model library):
pip install ultralytics
python watch.py --url "rtsp://..." --detect people
```

While the window is open: **q** quits, **s** saves a snapshot.
Everything lands in the `captures/` folder, including `events.csv` (open in Excel).

Useful flags: `--sensitivity`, `--min-area`, `--cooldown`, `--no-window` (headless).

## Step 5 — "Rebuild" / extend it

The tool is one file (`watch.py`) so it's easy to grow:
- **Alerts:** in the `ACTIVITY` block, add an email/Telegram/webhook call.
- **24/7 recording server:** point [Frigate](https://frigate.video) or
  [Blue Iris] / [Agent DVR] at the same RTSP URL for NVR-grade recording + a web UI.
- **Smarter analysis:** swap `yolov8n.pt` for a bigger model, or add counting,
  zones ("only alert near the door"), or loitering timers.
- **Remote viewing:** keep recording local and expose only a dashboard, or use a
  VPN back to your network — avoid opening port 554 straight to the internet.

## Security note
These budget cameras are known for weak defaults. Change the device password,
keep it on a guest/IoT Wi-Fi network if you can, and don't port-forward it
directly to the public internet.
