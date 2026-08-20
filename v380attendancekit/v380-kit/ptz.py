#!/usr/bin/env python3
"""Pan/tilt control for the V380 camera over ONVIF (port 8899).

Usage:
    ptz.py left|right|up|down [seconds]   # move for N seconds (default 1.0)
    ptz.py stop                           # stop any movement
    ptz.py look                           # grab a frame and print its path

Steer while watching live:  run `mpv --profile=low-latency --rtsp-transport=tcp
"rtsp://admin:@192.168.1.111:554/live/ch00_0"` in one terminal and ptz.py
commands in another. The stream lags a second or two behind real movement.
"""

import subprocess
import sys
import time

import requests

CAMERA = "192.168.1.111"
PTZ_URL = f"http://{CAMERA}:8899/onvif/ptz_service"
PROFILE = "PROFILE_000"
SPEED = 0.5
# The camera is mounted upside down, so screen-relative directions are the
# reverse of the lens's. If left/right feel swapped, set INVERT = False.
INVERT = True

ENV = ('<?xml version="1.0"?>'
       '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
       "<s:Body>{body}</s:Body></s:Envelope>")


def soap(body: str) -> None:
    r = requests.post(PTZ_URL, data=ENV.format(body=body),
                      headers={"Content-Type": "application/soap+xml"}, timeout=10)
    r.raise_for_status()
    if "Fault" in r.text:
        sys.exit(f"camera returned a fault:\n{r.text[:500]}")


def move(x: float, y: float, seconds: float) -> None:
    if INVERT:
        x, y = -x, -y
    soap('<ContinuousMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">'
         f"<ProfileToken>{PROFILE}</ProfileToken>"
         f'<Velocity><PanTilt xmlns="http://www.onvif.org/ver10/schema" x="{x}" y="{y}"/></Velocity>'
         "</ContinuousMove>")
    time.sleep(seconds)
    stop()


def stop() -> None:
    soap('<Stop xmlns="http://www.onvif.org/ver20/ptz/wsdl">'
         f"<ProfileToken>{PROFILE}</ProfileToken>"
         "<PanTilt>true</PanTilt><Zoom>true</Zoom></Stop>")


def look() -> None:
    out = "/tmp/ptz_view.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
         "-i", f"rtsp://admin:@{CAMERA}:554/live/ch00_0",
         "-frames:v", "1", "-q:v", "2",
         "-vf", "hflip,vflip,unsharp=5:5:0.6:3:3:0.3", out],
        timeout=90, check=True)
    print(out)


DIRECTIONS = {
    "left": (-SPEED, 0.0),
    "right": (SPEED, 0.0),
    "up": (0.0, SPEED),
    "down": (0.0, -SPEED),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in (*DIRECTIONS, "stop", "look"):
        sys.exit(__doc__.strip())
    cmd = sys.argv[1]
    if cmd == "stop":
        stop()
    elif cmd == "look":
        look()
    else:
        seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        move(*DIRECTIONS[cmd], seconds)


if __name__ == "__main__":
    main()
